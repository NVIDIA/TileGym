# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Flash Attention implementation.
Uses CUDA C++ tile kernels for prefill FMHA forward and backward.

Optimized with:
- tensor_span + partition_view for structured memory access
- All dimensions as template parameters
- Native type MMA for tensor core utilization
"""

import math
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl

from .autotuner import Config
from .autotuner import SearchSpace
from .autotuner import TileCppAutotuner
from .autotuner import autotune
from .autotuner import is_autotuning_enabled
from .utils._cuda_utils import TileCppKernel
from .utils._dump_types import dump_kernel_types


def _next_power_of_2(n):
    """Return the next power of 2 >= n."""
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


# Define kernels
_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "attention.cuh",
    kernel_name="prefill_fmha_fwd_kernel",
)

_bwd_preprocess_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "attention.cuh",
    kernel_name="fmha_bwd_preprocess_kernel",
)

_bwd_main_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "attention.cuh",
    kernel_name="fmha_bwd_main_kernel",
)


def _early_config_prune(named_args: dict, cfg: Config) -> bool:
    """Filter out invalid configs.

    - When causal: require BLOCK_M % BLOCK_N == 0 for correct masking.
    - When num_ctas > 1 (CGA cluster):  the grid X dim (cdiv(S_qo, BLOCK_M))
      must be divisible by num_ctas, otherwise the launch fails.
    """
    is_causal = named_args.get("is_causal", False)
    if is_causal and cfg.BLOCK_M % cfg.BLOCK_N != 0:
        return False
    num_ctas = cfg.num_ctas if cfg.num_ctas is not None else 1
    if num_ctas > 1:
        S_qo = named_args.get("S_qo")
        if S_qo is not None:
            grid_x = (S_qo + cfg.BLOCK_M - 1) // cfg.BLOCK_M
            if grid_x % num_ctas != 0:
                return False
    return True


def _get_configs(is_backward=False):
    capability = torch.cuda.get_device_capability()

    if capability in [(12, 0), (12, 1)]:
        configs = [
            Config(BLOCK_M=64, BLOCK_N=64, occupancy=2),
        ]
    elif capability == (9, 0):
        configs = [
            Config(BLOCK_M=BM, BLOCK_N=BN, occupancy=2, num_stages=s)
            for BM in [64, 128]
            for BN in [64, 128]
            for s in [2, 3]
        ]
    elif capability == (8, 0):
        configs = [
            Config(BLOCK_M=BM, BLOCK_N=BN, occupancy=2, num_stages=s)
            for BM in [64, 128]
            for BN in [32, 64]
            for s in [3, 4]
        ]
    else:
        if is_backward:
            configs = [
                Config(BLOCK_M=128, BLOCK_N=128, num_stages=7),
            ]
        else:
            configs = [
                Config(BLOCK_M=256, BLOCK_N=128, num_ctas=1, occupancy=1),
                Config(BLOCK_M=128, BLOCK_N=128, num_ctas=1, occupancy=2),
                Config(BLOCK_M=256, BLOCK_N=128, num_ctas=1, occupancy=2),
                Config(BLOCK_M=256, BLOCK_N=128, num_ctas=2, occupancy=2),
            ]

    return SearchSpace(configs, predicate_fn=_early_config_prune)


def _launch_fwd_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
    has_backward: bool,
    query_group_size: int,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_D: int,
    occupancy: int = 1,
    num_ctas: int = 1,
):
    """Launch the forward kernel with optimized template parameters."""
    dump_kernel_types("prefill_fmha_fwd_kernel", q, k, v)
    dtype = q.dtype
    B, H, S_qo, D = q.shape
    H_kv = k.shape[1]
    S_kv = k.shape[2]

    template_params = [
        B,
        H,
        H_kv,
        S_qo,
        S_kv,
        BLOCK_M,
        BLOCK_N,
        BLOCK_D,
        is_causal,
        has_backward,
        occupancy,
        num_ctas,
    ]

    kernel, _, _ = _fwd_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        # Simplified signature: just pointers and scale
        signature="const {T}*, const {T}*, const {T}*, {T}*, float*, float",
    )

    grid = (_cdiv(S_qo, BLOCK_M), B * H, 1)

    _fwd_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(q.data_ptr()),
            np.uint64(k.data_ptr()),
            np.uint64(v.data_ptr()),
            np.uint64(out.data_ptr()),
            np.uint64(lse.data_ptr()),
            np.float32(sm_scale),
        ],
    )


def _launch_bwd_preprocess_kernel(
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    minus_delta: torch.Tensor,
    minus_l: torch.Tensor,
    sm_scale: float,
    BLOCK_M: int,
    BLOCK_D: int,
    occupancy: int = 1,
):
    """Launch the backward preprocess kernel."""
    dump_kernel_types("fmha_bwd_preprocess_kernel", out, dout, lse)
    dtype = out.dtype
    B, H, S_qo, _ = out.shape

    template_params = [B, H, S_qo, BLOCK_M, BLOCK_D, occupancy]

    kernel, _, _ = _bwd_preprocess_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature="const {T}*, const {T}*, const float*, float*, float*, float",
    )

    grid = (_cdiv(S_qo, BLOCK_M), B * H, 1)

    _bwd_preprocess_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(out.data_ptr()),
            np.uint64(dout.data_ptr()),
            np.uint64(lse.data_ptr()),
            np.uint64(minus_delta.data_ptr()),
            np.uint64(minus_l.data_ptr()),
            np.float32(sm_scale),
        ],
    )


def _launch_bwd_main_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    minus_l: torch.Tensor,
    minus_delta: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    sm_scale: float,
    is_causal: bool,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_D: int,
    occupancy: int = 1,
):
    """Launch the backward main kernel."""
    dump_kernel_types("fmha_bwd_main_kernel", q, k, v, dout)
    dtype = q.dtype
    B, H, S_qo, _ = q.shape
    S_kv = k.shape[2]

    template_params = [B, H, S_qo, S_kv, BLOCK_M, BLOCK_N, BLOCK_D, is_causal, occupancy]

    kernel, _, _ = _bwd_main_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature="const {T}*, const {T}*, const {T}*, const {T}*, const float*, const float*, float*, {T}*, {T}*, float",
    )

    grid = (_cdiv(S_kv, BLOCK_N), B * H, 1)

    _bwd_main_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(q.data_ptr()),
            np.uint64(k.data_ptr()),
            np.uint64(v.data_ptr()),
            np.uint64(dout.data_ptr()),
            np.uint64(minus_l.data_ptr()),
            np.uint64(minus_delta.data_ptr()),
            np.uint64(dq.data_ptr()),
            np.uint64(dk.data_ptr()),
            np.uint64(dv.data_ptr()),
            np.float32(sm_scale),
        ],
    )


# =============================================================================
# Autotuned Kernel Wrappers
# =============================================================================


@autotune(search_space=_get_configs(is_backward=False))
def _autotune_fwd(
    q,
    k,
    v,
    out,
    lse,
    sm_scale,
    is_causal,
    has_backward,
    query_group_size,
    BLOCK_D,
    autotuner: TileCppAutotuner | None = None,
):
    """Autotuned forward that searches over BLOCK_M x BLOCK_N."""
    B, H, S_qo, _ = q.shape
    S_kv = k.shape[2]
    key = ("tilecpp_fmha_fwd", str(q.dtype), S_qo, S_kv, BLOCK_D, is_causal, query_group_size)
    named_args = {"is_causal": is_causal, "S_qo": S_qo}

    def launch_fn(cfg: Config):
        occupancy = cfg.occupancy if cfg.occupancy is not None else 1
        num_ctas = cfg.num_ctas if cfg.num_ctas is not None else 1
        _launch_fwd_kernel(
            q,
            k,
            v,
            out,
            lse,
            sm_scale,
            is_causal,
            has_backward,
            query_group_size,
            cfg.BLOCK_M,
            cfg.BLOCK_N,
            BLOCK_D,
            occupancy=occupancy,
            num_ctas=num_ctas,
        )

    def grid_fn(args: dict, cfg: Config) -> tuple:
        return (_cdiv(S_qo, cfg.BLOCK_M), B * H, 1)

    autotuner(
        torch.cuda.current_stream(),
        key=key,
        launch_fn=launch_fn,
        grid_fn=grid_fn,
        named_args=named_args,
    )


@autotune(search_space=_get_configs(is_backward=True))
def _autotune_bwd(
    q,
    k,
    v,
    o,
    l,
    do,
    minus_delta,
    minus_l,
    dq,
    dk,
    dv,
    sm_scale,
    is_causal,
    BLOCK_D,
    autotuner: TileCppAutotuner | None = None,
):
    """Autotuned backward that searches over BLOCK_M x BLOCK_N for both bwd kernels."""
    B, H, S_qo, _ = q.shape
    S_kv = k.shape[2]
    key = ("tilecpp_fmha_bwd", str(q.dtype), S_qo, S_kv, BLOCK_D, is_causal)
    named_args = {"is_causal": is_causal}

    def launch_fn(cfg: Config):
        occupancy = cfg.occupancy if cfg.occupancy is not None else 1
        dq.zero_()
        _launch_bwd_preprocess_kernel(
            o,
            do,
            l,
            minus_delta,
            minus_l,
            sm_scale,
            cfg.BLOCK_M,
            BLOCK_D,
            occupancy=occupancy,
        )
        _launch_bwd_main_kernel(
            q,
            k,
            v,
            do,
            minus_l,
            minus_delta,
            dq,
            dk,
            dv,
            sm_scale,
            is_causal,
            cfg.BLOCK_M,
            cfg.BLOCK_N,
            BLOCK_D,
            occupancy=occupancy,
        )

    def grid_fn(args: dict, cfg: Config) -> tuple:
        return (_cdiv(S_kv, cfg.BLOCK_N), B * H, 1)

    autotuner(
        torch.cuda.current_stream(),
        key=key,
        launch_fn=launch_fn,
        grid_fn=grid_fn,
        named_args=named_args,
    )


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sm_scale,
        is_causal,
        has_backward=False,
    ):
        B, H, S_qo, BLOCK_D = q.shape
        assert k.shape == v.shape
        num_head_kv = k.shape[1]
        S_kv = k.shape[2]
        assert S_qo == S_kv, "S_qo != S_kv is not supported, please use attention variant kernel instead"

        # Kernel pointer arithmetic assumes packed row-major layout.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if H == num_head_kv:
            query_group_size = 0
        else:
            assert H % num_head_kv == 0
            query_group_size = int(H / num_head_kv)

        o = torch.empty_like(q)
        l = torch.empty((B, H, S_qo), device=q.device, dtype=torch.float32)

        if is_autotuning_enabled():
            _autotune_fwd(
                q,
                k,
                v,
                o,
                l,
                sm_scale,
                is_causal,
                has_backward,
                query_group_size,
                BLOCK_D,
            )
        else:
            gpu_capability = torch.cuda.get_device_capability()
            if gpu_capability in [(12, 0), (12, 1)]:
                BLOCK_M, BLOCK_N, num_ctas, occupancy = 64, 64, 1, 2
            elif gpu_capability == (9, 0):
                BLOCK_M, BLOCK_N, num_ctas, occupancy = 128, 128, 1, 2
            elif gpu_capability == (8, 0):
                BLOCK_M, BLOCK_N, num_ctas, occupancy = 128, 64, 1, 2
            else:
                BLOCK_M, BLOCK_N, num_ctas, occupancy = 256, 128, 1, 1

            _launch_fwd_kernel(
                q,
                k,
                v,
                o,
                l,
                sm_scale,
                is_causal,
                has_backward,
                query_group_size,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
                occupancy=occupancy,
                num_ctas=num_ctas,
            )

        ctx.save_for_backward(q, k, v, o, l)
        ctx.sm_scale = sm_scale
        ctx.shapes = (B, H, S_qo, S_kv)
        ctx.is_causal = is_causal
        ctx.BLOCK_D = BLOCK_D
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        B, H, S_qo, S_kv = ctx.shapes
        is_causal = ctx.is_causal
        BLOCK_D = ctx.BLOCK_D

        if k.shape[1] != H:
            raise NotImplementedError(
                "fmha backward currently supports MHA only "
                f"(got H_q={H}, H_kv={k.shape[1]}); GQA backward is not implemented."
            )

        # The bwd main kernel iterates Q tiles of BLOCK_M rows and atomic_adds
        # dQ via raw pointer arithmetic without per-lane bounds checks; if
        # S_qo is not a multiple of BLOCK_M the last block writes past the
        # current head's slice of dQ. The smallest BLOCK_M across all autotune
        # configs is 64, so require S_qo % 64 == 0.
        if S_qo % 64 != 0:
            raise NotImplementedError(
                f"fmha backward requires S_qo ({S_qo}) to be a multiple of 64; "
                f"the kernel writes dQ via atomic_add without per-lane bounds checks."
            )

        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        minus_delta = torch.empty_like(l)
        minus_l = torch.empty_like(l)

        assert dq.stride() == q.stride()
        assert dk.stride() == k.stride()
        assert dv.stride() == v.stride()
        assert do.stride() == o.stride()

        if is_autotuning_enabled():
            _autotune_bwd(
                q,
                k,
                v,
                o,
                l,
                do,
                minus_delta,
                minus_l,
                dq,
                dk,
                dv,
                ctx.sm_scale,
                is_causal,
                BLOCK_D,
            )
        else:
            gpu_capability = torch.cuda.get_device_capability()
            if gpu_capability in [(12, 0), (12, 1)]:
                BLOCK_M, BLOCK_N, occupancy = 64, 64, 2
            elif gpu_capability == (9, 0):
                BLOCK_M, BLOCK_N, occupancy = 128, 128, 2
            elif gpu_capability == (8, 0):
                BLOCK_M, BLOCK_N, occupancy = 128, 64, 2
            else:
                BLOCK_M, BLOCK_N, occupancy = 128, 128, 1

            _launch_bwd_preprocess_kernel(
                o,
                do,
                l,
                minus_delta,
                minus_l,
                ctx.sm_scale,
                BLOCK_M,
                BLOCK_D,
                occupancy=occupancy,
            )
            _launch_bwd_main_kernel(
                q,
                k,
                v,
                do,
                minus_l,
                minus_delta,
                dq,
                dk,
                dv,
                ctx.sm_scale,
                is_causal,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
                occupancy=occupancy,
            )

        return dq, dk, dv, None, None, None


@register_impl("fmha", backend="tilecpp")
def tilecpp_fmha(
    q,
    k,
    v,
    scaling=None,
    is_causal=True,
    **kwargs,
):
    """
    CUDA Tile C++ Flash Multi-Head Attention.

    Args:
        q: Query tensor [B, H, S, D]
        k: Key tensor [B, H, S, D]
        v: Value tensor [B, H, S, D]
        scaling: Softmax scaling factor (default: 1/sqrt(D))
        is_causal: Whether to apply causal masking

    Returns:
        Output tensor [B, H, S, D]
    """
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))
    # Enable backward if any input requires grad or explicitly requested
    has_backward = kwargs.get("has_backward", False)
    if q.requires_grad or k.requires_grad or v.requires_grad:
        has_backward = True
    o = _Attention.apply(q, k, v, scaling, is_causal, has_backward)
    return o
