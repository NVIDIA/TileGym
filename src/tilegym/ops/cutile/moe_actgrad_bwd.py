# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from enum import IntEnum
from types import SimpleNamespace
from typing import Optional

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import get_cached_ct_kernel
from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl

# Module-level tune cache: problem signature -> (best config, tuned kernel).
_moe_actgrad_bwd_tune_cache: dict = {}

ConstInt = ct.Constant[int]

_PAD_ZERO = ct.PaddingMode.ZERO


# Express exp(-x^2) through exp2 so the GELU path can use flush-to-zero.
_LOG2_E = 1.4426950408889634
_INV_SQRT_2 = 0.7071067811865475
_INV_SQRT_2_PI = 0.3989422804014327


class _ActivationType(IntEnum):
    SWIGLU = 0
    GEGLU = 1
    REGLU = 2
    SILU = 3
    RELU = 4
    GELU = 5
    RELU_SQ = 6


# String-to-enum mapping for the public `activation_type` argument.
_ACT_STR_TO_INT = {
    "swiglu": _ActivationType.SWIGLU,
    "geglu": _ActivationType.GEGLU,
    "reglu": _ActivationType.REGLU,
    "silu": _ActivationType.SILU,
    "relu": _ActivationType.RELU,
    "gelu": _ActivationType.GELU,
    "relu_sq": _ActivationType.RELU_SQ,
}

# Activation helpers operate on FP32 tiles and return FP32 results.


def _sigmoid_f32(x):
    # Approximate tanh avoids an explicit exp2 and reciprocal.
    return 0.5 * (1.0 + ct.tanh(0.5 * x, rounding_mode=ct.RoundingMode.APPROX))


def _erf_approx_with_exp(x):
    """Return the A&S 7.1.26 erf approximation and its exp(-x^2) term."""
    P = 0.3275911
    A1 = 0.254829592
    A2 = -0.284496736
    A3 = 1.421413741
    A4 = -1.453152027
    A5 = 1.061405429
    ax = ct.abs(x)
    t = ct.truediv(
        1.0,
        1.0 + P * ax,
        flush_to_zero=True,
        rounding_mode=ct.RoundingMode.APPROX,
    )
    # Horner form keeps the intermediate FP32 live ranges short.
    poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))))
    exp_term = ct.exp2(-ax * ax * _LOG2_E, flush_to_zero=True)
    result = 1.0 - poly * exp_term
    return ct.where(x >= 0.0, result, -result), exp_term


def _silu_derivative(x, dy):
    sig_x = _sigmoid_f32(x)
    silu_x = x * sig_x
    dx = dy * (sig_x * (1.0 + x * (1.0 - sig_x)))
    return dx, silu_x


def _swiglu_derivative(g, u, dy):
    sig_g = _sigmoid_f32(g)
    silu_g = g * sig_g
    swiglu_output = silu_g * u
    dg = dy * u * (sig_g * (1.0 + g * (1.0 - sig_g)))
    du = dy * silu_g
    return dg, du, swiglu_output


def _relu_derivative(x, dy):
    relu_x = ct.maximum(x, 0.0)
    relu_prime = ct.where(x > 0, 1.0, 0.0)
    dx = dy * relu_prime
    return dx, relu_x


def _reglu_derivative(g, u, dy):
    relu_g = ct.maximum(g, 0.0)
    relu_prime_g = ct.where(g > 0, 1.0, 0.0)
    dg = dy * u * relu_prime_g
    du = dy * relu_g
    reglu_output = relu_g * u
    return dg, du, reglu_output


def _gelu_with_derivative(x):
    erf_val, exp_term = _erf_approx_with_exp(x * _INV_SQRT_2)
    pdf = _INV_SQRT_2_PI * exp_term
    gelu_x = 0.5 * x * (1.0 + erf_val)
    gelu_prime = 0.5 * (1.0 + erf_val) + x * pdf
    return gelu_x, gelu_prime


def _gelu_derivative(x, dy):
    gelu_x, gelu_prime = _gelu_with_derivative(x)
    dx = dy * gelu_prime
    return dx, gelu_x


def _geglu_derivative(g, u, dy):
    gelu_g, gelu_prime = _gelu_with_derivative(g)
    dg = dy * u * gelu_prime
    du = dy * gelu_g
    geglu_output = gelu_g * u
    return dg, du, geglu_output


def _relu_sq_derivative(x, dy):
    relu_x = ct.maximum(x, 0.0)
    relu_sq_out = relu_x * relu_x
    dx = dy * 2.0 * relu_x
    return dx, relu_sq_out


def _moe_actgrad_bwd_autotune_configs(intermediate_dim: int, hidden_dim: int, device: torch.device):
    """Yield architecture-specific autotune configs."""
    gpu_capability = torch.cuda.get_device_capability(device)
    use_b200_configs = gpu_capability in {(10, 0), (10, 3)}

    if use_b200_configs:
        candidates = [(64, 128, 64), (64, 256, 64), (128, 128, 64), (128, 256, 64)]
    elif gpu_capability in {(12, 0), (12, 1)}:
        candidates = [(64, 128, 64), (128, 128, 64)]
    else:
        candidates = [(128, 128, 64), (128, 128, 128)]

    use_high_occ_bm64 = use_b200_configs and 1024 <= hidden_dim <= 2048 and intermediate_dim >= 1024

    for block_m, block_n, block_k in candidates:
        occupancies = (1, 2, 3) if use_high_occ_bm64 and (block_m, block_n, block_k) == (64, 128, 64) else (1,)
        for occupancy in occupancies:
            yield SimpleNamespace(
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_ctas=1,
                occupancy=occupancy,
            )


@ct.kernel
def _moe_actgrad_bwd_kernel(
    # Inputs
    dout_ptr,
    x_gather_idx_ptr,
    w2_trans_ptr,
    z_ptr,
    topk_scores_ptr,
    s_scatter_idx_ptr,
    expert_offset_ptr,
    # Outputs
    dz_ptr,
    y1s_ptr,
    ds_ptr,
    # Dimensions
    hidden_dim: ConstInt,
    intermediate_dim: ConstInt,
    # Tile sizes
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
    # Compile-time flags
    ACTIVATION_TYPE: ConstInt,
    IS_GLU: ConstInt,
    EPI_NSUB: ConstInt,
):
    """Compute MoE down-projection activation gradients.

    Each program computes a per-expert grouped GEMM tile, applies activation
    backward and routing-score scaling, writes dz and y1s, and atomically
    accumulates ds across N tiles. GLU inputs store contiguous gate and up
    halves along the last dimension.
    """
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    pid_expert = ct.bid(2)

    expert_start = ct.load(expert_offset_ptr, index=(pid_expert,), shape=(1,)).item()
    expert_end = ct.load(expert_offset_ptr, index=(pid_expert + 1,), shape=(1,)).item()
    num_tokens = expert_end - expert_start

    if num_tokens <= 0:
        return

    m_start = pid_m * BLOCK_M
    # The grid uses an upper bound on tokens per expert; skip excess tiles.
    if m_start >= num_tokens:
        return

    n_start = pid_n * BLOCK_N
    if n_start >= intermediate_dim:
        return

    m_offs = m_start + ct.arange(BLOCK_M, dtype=ct.int32)
    m_mask = m_offs < num_tokens
    token_indices = expert_start + m_offs

    x_gather_indices = ct.gather(x_gather_idx_ptr, (token_indices,), check_bounds=True, padding_value=0)
    s_scatter_indices = ct.gather(s_scatter_idx_ptr, (token_indices,), check_bounds=True, padding_value=0)
    s_vals = ct.gather(topk_scores_ptr, (s_scatter_indices,), check_bounds=True, padding_value=0.0)

    n_offs = n_start + ct.arange(BLOCK_N, dtype=ct.int32)

    # Step 1: Grouped GEMM, dy1 = dout[x_gather_idx] @ W2[:I]^T.
    # A two-dimensional gather preserves contiguous hidden-dimension accesses
    # and avoids scalar pointer loads.
    rows_col = x_gather_indices[:, None]
    acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)

    for k in range(0, ct.cdiv(hidden_dim, BLOCK_K)):
        col_indices = k * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
        dout_block = ct.gather(
            dout_ptr,
            (rows_col, col_indices[None, :]),
            padding_value=0.0,
            latency=10,
        )

        w2_block = ct.load(
            w2_trans_ptr,
            index=(pid_expert, k, pid_n),
            shape=(1, BLOCK_K, BLOCK_N),
            order=(0, 2, 1),
            padding_mode=_PAD_ZERO,
            allow_tma=True,
            latency=1,
        )
        w2_block = ct.reshape(w2_block, (BLOCK_K, BLOCK_N))

        acc = ct.mma(dout_block, w2_block, acc)

    dy1_f32 = acc

    # Step 2: Activation backward and routing-score scaling.
    # Zero invalid N lanes before reduction. The per-expert M tail is masked
    # before atomic_add because check_bounds only protects global ds bounds.
    n_mask = (n_offs < intermediate_dim)[None, :]
    dy1_used = ct.where(n_mask, dy1_f32, 0.0)

    s_expanded = s_vals[:, None]

    z_expert = z_ptr.slice(axis=0, start=expert_start, stop=expert_end)
    dz_expert = dz_ptr.slice(axis=0, start=expert_start, stop=expert_end)
    y1s_expert = y1s_ptr.slice(axis=0, start=expert_start, stop=expert_end)

    if IS_GLU:
        # Slicing the gate/up halves lets load/store mask the I boundary without
        # constraining BLOCK_N.
        z_gate = z_expert.slice(axis=1, start=0, stop=intermediate_dim)
        z_up = z_expert.slice(axis=1, start=intermediate_dim, stop=2 * intermediate_dim)
        dz_gate = dz_expert.slice(axis=1, start=0, stop=intermediate_dim)
        dz_up = dz_expert.slice(axis=1, start=intermediate_dim, stop=2 * intermediate_dim)

        if EPI_NSUB == 1:
            # Preserve the full-tile epilogue for short hidden dimensions,
            # where sub-tiling costs more than it saves.
            g_block = ct.load(
                z_gate,
                index=(pid_m, pid_n),
                shape=(BLOCK_M, BLOCK_N),
                padding_mode=_PAD_ZERO,
            )
            g_f32 = g_block.astype(ct.float32)

            u_block = ct.load(
                z_up,
                index=(pid_m, pid_n),
                shape=(BLOCK_M, BLOCK_N),
                padding_mode=_PAD_ZERO,
            )
            u_f32 = u_block.astype(ct.float32)

            if ACTIVATION_TYPE == 0:  # SWIGLU
                dg, du, fwd_output = _swiglu_derivative(g_f32, u_f32, dy1_used)
            elif ACTIVATION_TYPE == 1:  # GEGLU
                dg, du, fwd_output = _geglu_derivative(g_f32, u_f32, dy1_used)
            else:  # REGLU
                dg, du, fwd_output = _reglu_derivative(g_f32, u_f32, dy1_used)

            ds_partial_val = ct.sum(dy1_used * fwd_output, axis=1)

            ct.store(
                dz_gate,
                index=(pid_m, pid_n),
                tile=(dg * s_expanded).astype(dz_ptr.dtype),
                allow_tma=False,
            )

            ct.store(
                dz_up,
                index=(pid_m, pid_n),
                tile=(du * s_expanded).astype(dz_ptr.dtype),
                allow_tma=False,
            )

            ct.store(
                y1s_expert,
                index=(pid_m, pid_n),
                tile=(fwd_output * s_expanded).astype(y1s_ptr.dtype),
                allow_tma=False,
            )
        else:
            # Keep the GEMM at full BLOCK_N, but process the activation
            # epilogue in 32-column slices to reduce live FP32 tiles and
            # shared-memory traffic.
            BN_SUB = BLOCK_N // EPI_NSUB
            ds_partial_val = ct.zeros((BLOCK_M,), dtype=ct.float32)
            for j in range(EPI_NSUB):
                col_blk = pid_n * EPI_NSUB + j
                n_offs_j = n_start + j * BN_SUB + ct.arange(BN_SUB, dtype=ct.int32)
                n_mask_j = (n_offs_j < intermediate_dim)[None, :]
                acc_j = ct.extract(dy1_f32, index=(0, j), shape=(BLOCK_M, BN_SUB))
                dy_j = ct.where(n_mask_j, acc_j, 0.0)

                g_j = ct.load(
                    z_gate,
                    index=(pid_m, col_blk),
                    shape=(BLOCK_M, BN_SUB),
                    padding_mode=_PAD_ZERO,
                ).astype(ct.float32)
                u_j = ct.load(
                    z_up,
                    index=(pid_m, col_blk),
                    shape=(BLOCK_M, BN_SUB),
                    padding_mode=_PAD_ZERO,
                ).astype(ct.float32)

                if ACTIVATION_TYPE == 0:  # SWIGLU
                    dg_j, du_j, fwd_j = _swiglu_derivative(g_j, u_j, dy_j)
                elif ACTIVATION_TYPE == 1:  # GEGLU
                    dg_j, du_j, fwd_j = _geglu_derivative(g_j, u_j, dy_j)
                else:  # REGLU
                    dg_j, du_j, fwd_j = _reglu_derivative(g_j, u_j, dy_j)

                ds_partial_val = ds_partial_val + ct.sum(dy_j * fwd_j, axis=1)

                ct.store(
                    dz_gate,
                    index=(pid_m, col_blk),
                    tile=(dg_j * s_expanded).astype(dz_ptr.dtype),
                    allow_tma=False,
                )
                ct.store(
                    dz_up,
                    index=(pid_m, col_blk),
                    tile=(du_j * s_expanded).astype(dz_ptr.dtype),
                    allow_tma=False,
                )
                ct.store(
                    y1s_expert,
                    index=(pid_m, col_blk),
                    tile=(fwd_j * s_expanded).astype(y1s_ptr.dtype),
                    allow_tma=False,
                )

    else:
        z_block = ct.load(
            z_expert,
            index=(pid_m, pid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=_PAD_ZERO,
        )
        z_f32 = z_block.astype(ct.float32)

        if ACTIVATION_TYPE == 3:  # SILU
            dz_raw, fwd_output = _silu_derivative(z_f32, dy1_used)
        elif ACTIVATION_TYPE == 4:  # RELU
            dz_raw, fwd_output = _relu_derivative(z_f32, dy1_used)
        elif ACTIVATION_TYPE == 5:  # GELU
            dz_raw, fwd_output = _gelu_derivative(z_f32, dy1_used)
        else:  # RELU_SQ
            dz_raw, fwd_output = _relu_sq_derivative(z_f32, dy1_used)

        ds_partial_val = ct.sum(dy1_used * fwd_output, axis=1)

        ct.store(
            dz_expert,
            index=(pid_m, pid_n),
            tile=(dz_raw * s_expanded).astype(dz_ptr.dtype),
            allow_tma=False,
        )

        ct.store(
            y1s_expert,
            index=(pid_m, pid_n),
            tile=(fwd_output * s_expanded).astype(y1s_ptr.dtype),
            allow_tma=False,
        )

    # Step 3: Accumulate the routing-score gradient atomically across N tiles.
    # Mask the per-expert M tail explicitly because check_bounds only protects
    # the global ds extent.
    ds_partial_val = ct.where(m_mask, ds_partial_val, 0.0)
    ct.atomic_add(ds_ptr, (token_indices,), ds_partial_val, check_bounds=True, memory_order=ct.MemoryOrder.RELAXED)


def _validate_epi_nsub(block_n: int, epi_nsub: int) -> int:
    """Validate and return the host-side epilogue subdivision count."""
    if not isinstance(block_n, int) or isinstance(block_n, bool) or block_n <= 0:
        raise ValueError(f"BLOCK_N must be a positive integer; got {block_n!r}")
    if (
        not isinstance(epi_nsub, int)
        or isinstance(epi_nsub, bool)
        or epi_nsub <= 0
        or epi_nsub > block_n
        or block_n % epi_nsub != 0
    ):
        raise ValueError(
            f"EPI_NSUB must be a positive integer that is <= BLOCK_N and divides BLOCK_N; "
            f"got EPI_NSUB={epi_nsub!r}, BLOCK_N={block_n!r}"
        )
    return epi_nsub


@register_impl("moe_actgrad_bwd", backend="cutile")
def moe_actgrad_bwd(
    dout: torch.Tensor,
    h: torch.Tensor,
    w2: torch.Tensor,
    dh: torch.Tensor,
    ds: torch.Tensor,
    b2: Optional[torch.Tensor],
    db2: Optional[torch.Tensor],
    a_prime: torch.Tensor,
    topk_scores: torch.Tensor,
    expert_frequency_offset: torch.Tensor,
    x_gather_idx: torch.Tensor,
    s_scatter_idx: torch.Tensor,
    activation_type: str,
    max_tokens_per_expert: Optional[int] = None,
    *,
    kernel_configs: Optional[dict] = None,
) -> None:
    """cuTile impl of moe_actgrad_bwd. See dispatch wrapper for full doc."""
    del b2, db2  # Retained for API compatibility.

    if activation_type not in _ACT_STR_TO_INT:
        raise ValueError(f"Unsupported activation_type: {activation_type!r}. Supported: {list(_ACT_STR_TO_INT.keys())}")
    act_int = int(_ACT_STR_TO_INT[activation_type])
    is_glu = act_int in {_ActivationType.SWIGLU, _ActivationType.GEGLU, _ActivationType.REGLU}
    is_glu_int = int(is_glu)

    num_experts = w2.shape[0]
    intermediate_dim = w2.shape[1]
    hidden_dim = w2.shape[2]
    input_tokens = dout.shape[0]
    total_tokens = h.shape[0]
    dtype = h.dtype

    if total_tokens == 0:
        return

    if max_tokens_per_expert is None:
        # This fallback synchronizes the GPU and CPU; callers should pass the
        # routing result when it is already available.
        max_tokens_per_expert = int((expert_frequency_offset[1:] - expert_frequency_offset[:-1]).max().item())

    if max_tokens_per_expert <= 0:
        return

    gpu_capability = torch.cuda.get_device_capability(dout.device)
    use_b200_configs = gpu_capability in {(10, 0), (10, 3)}

    def _epi_nsub(block_n: int) -> int:
        _validate_epi_nsub(block_n, 1)
        # Use 32-column activation slices for large B200 GLU workloads; retain
        # the full-tile epilogue elsewhere.
        if not use_b200_configs or not is_glu or hidden_dim < 4096 or block_n % 32 != 0:
            return 1
        return _validate_epi_nsub(block_n, max(1, block_n // 32))

    common_args = (
        dout,
        x_gather_idx,
        w2,
        h,
        topk_scores,
        s_scatter_idx,
        expert_frequency_offset,
        dh,
        a_prime,
        ds,
        hidden_dim,
        intermediate_dim,
    )

    def _args_fn(cfg, epi_nsub):
        return common_args + (
            cfg.BLOCK_M,
            cfg.BLOCK_N,
            cfg.BLOCK_K,
            act_int,
            is_glu_int,
            epi_nsub,
        )

    def _args_fn_for_tune(cfg):
        # Each autotune trial needs an empty atomic accumulation buffer.
        ds.zero_()
        return _args_fn(cfg, _epi_nsub(cfg.BLOCK_N))

    def _grid_fn(cfg):
        return (
            (max_tokens_per_expert + cfg.BLOCK_M - 1) // cfg.BLOCK_M,
            (intermediate_dim + cfg.BLOCK_N - 1) // cfg.BLOCK_N,
            num_experts,
        )

    stream = torch.cuda.current_stream(dout.device)

    if kernel_configs is not None:
        launch_cfg = SimpleNamespace(
            BLOCK_M=kernel_configs.get("BLOCK_M", 128),
            BLOCK_N=kernel_configs.get("BLOCK_N", 128),
            BLOCK_K=kernel_configs.get("BLOCK_K", 128),
            num_ctas=kernel_configs.get("num_ctas", None),
            occupancy=kernel_configs.get("occupancy", None),
        )
        launch_epi_nsub = (
            _validate_epi_nsub(launch_cfg.BLOCK_N, kernel_configs["EPI_NSUB"])
            if "EPI_NSUB" in kernel_configs
            else _epi_nsub(launch_cfg.BLOCK_N)
        )
        launch_kernel = _moe_actgrad_bwd_kernel
        if launch_cfg.num_ctas is not None or launch_cfg.occupancy is not None:
            launch_kernel = get_cached_ct_kernel(
                _moe_actgrad_bwd_kernel._pyfunc,
                num_ctas=launch_cfg.num_ctas,
                occupancy=launch_cfg.occupancy,
                device=dout.device,
            )
    elif is_autotune_disabled():
        launch_cfg = SimpleNamespace(
            BLOCK_M=64 if gpu_capability in {(12, 0), (12, 1)} else 128,
            BLOCK_N=128,
            BLOCK_K=64,
            num_ctas=1,
            occupancy=1,
        )
        launch_epi_nsub = _epi_nsub(launch_cfg.BLOCK_N)
        launch_kernel = get_cached_ct_kernel(
            _moe_actgrad_bwd_kernel._pyfunc,
            num_ctas=launch_cfg.num_ctas,
            occupancy=launch_cfg.occupancy,
            device=dout.device,
        )
    else:
        cache_key = (
            max_tokens_per_expert,
            input_tokens,
            total_tokens,
            intermediate_dim,
            hidden_dim,
            num_experts,
            act_int,
            dtype,
            str(dout.device),
        )
        if cache_key not in _moe_actgrad_bwd_tune_cache:
            result = exhaustive_search(
                list(_moe_actgrad_bwd_autotune_configs(intermediate_dim, hidden_dim, dout.device)),
                stream,
                _grid_fn,
                _moe_actgrad_bwd_kernel,
                _args_fn_for_tune,
                lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _moe_actgrad_bwd_tune_cache[cache_key] = (
                best_cfg,
                get_cached_ct_kernel(
                    _moe_actgrad_bwd_kernel._pyfunc,
                    num_ctas=best_cfg.num_ctas,
                    occupancy=best_cfg.occupancy,
                    device=dout.device,
                ),
            )
            # The final autotune trial leaves ds accumulated; clear it before
            # launching the selected configuration.
            ds.zero_()
        launch_cfg, launch_kernel = _moe_actgrad_bwd_tune_cache[cache_key]
        launch_epi_nsub = _epi_nsub(launch_cfg.BLOCK_N)

    # The in-place API requires ds to be zero before this accumulation launch.
    ct.launch(
        stream,
        _grid_fn(launch_cfg),
        launch_kernel,
        _args_fn(launch_cfg, launch_epi_nsub),
    )
