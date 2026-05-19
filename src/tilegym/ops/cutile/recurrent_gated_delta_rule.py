# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import os
from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.ops.cutile.utils import next_power_of_2


# Naming conventions in cuda.tile DSL kernels:
# - UPPER_CASE_SNAKE_NAMING: Compile-time constants
# - CamelCaseNaming: Runtime vectors or tensors
# - lower_case_snake_naming: Runtime scalars
@ct.kernel(occupancy=2)
def _recurrent_gated_delta_rule_fwd_kernel(
    Query,  # (B, T, H, QK)
    Key,  # (B, T, H, QK)
    Value,  # (B, T, HV, V)
    Gate,  # (B, T, HV)
    Beta,  # (B, T, HV)
    Output,  # (B, T, HV, V)
    StateIn,  # (B, HV, QK, V)
    StateOut,  # (B, HV, QK, V)
    scale: float,
    HAS_INITIAL_STATE: ct.Constant[bool],
    OUTPUT_FINAL_STATE: ct.Constant[bool],
    USE_QK_L2NORM: ct.Constant[bool],
    T: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_V: ct.Constant[int],
    SMALL_TILE_USE_TMA: ct.Constant[bool],
    LARGE_TILE_USE_TMA: ct.Constant[bool],
):
    """Grid: (B * Hv, ceil(V / BLOCK_V), 1)."""
    idx_bhv = ct.bid(0)
    idx_v = ct.bid(1)

    H = Query.shape[2]
    HV = Value.shape[2]
    idx_b = idx_bhv // HV
    idx_hv = idx_bhv % HV
    idx_h = idx_hv // (HV // H)

    if HAS_INITIAL_STATE:
        State = ct.load(
            StateIn,
            index=(idx_b, idx_hv, 0, idx_v),
            shape=(1, 1, BLOCK_K, BLOCK_V),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=LARGE_TILE_USE_TMA,
        ).reshape((BLOCK_K, BLOCK_V))
        State = ct.astype(State, ct.float32)
    else:
        State = ct.zeros((BLOCK_K, BLOCK_V), dtype=ct.float32)

    for idx_t in range(T):
        QueryT = ct.load(
            Query,
            index=(idx_b, idx_t, idx_h, 0),
            shape=(1, 1, 1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=LARGE_TILE_USE_TMA,
        ).reshape((BLOCK_K,))
        QueryT = ct.astype(QueryT, ct.float32)

        KeyT = ct.load(
            Key,
            index=(idx_b, idx_t, idx_h, 0),
            shape=(1, 1, 1, BLOCK_K),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=LARGE_TILE_USE_TMA,
        ).reshape((BLOCK_K,))
        KeyT = ct.astype(KeyT, ct.float32)

        if USE_QK_L2NORM:
            QueryT = QueryT * ct.rsqrt(ct.sum(QueryT * QueryT, axis=0) + 1e-6)
            KeyT = KeyT * ct.rsqrt(ct.sum(KeyT * KeyT, axis=0) + 1e-6)
        QueryT = QueryT * scale

        ValueT = ct.load(
            Value,
            index=(idx_b, idx_t, idx_hv, idx_v),
            shape=(1, 1, 1, BLOCK_V),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=SMALL_TILE_USE_TMA,
        ).reshape((BLOCK_V,))
        ValueT = ct.astype(ValueT, ct.float32)

        gate_t = ct.astype(ct.gather(Gate, (idx_b, idx_t, idx_hv), check_bounds=False), ct.float32)
        beta_t = ct.astype(ct.gather(Beta, (idx_b, idx_t, idx_hv), check_bounds=False), ct.float32)

        State = State * ct.exp(gate_t)
        KeyT = ct.expand_dims(KeyT, axis=1)
        KvMemT = ct.sum(State * KeyT, axis=0)
        DeltaT = (ValueT - KvMemT) * beta_t
        State = State + KeyT * ct.expand_dims(DeltaT, axis=0)
        OutT = ct.sum(State * ct.expand_dims(QueryT, axis=1), axis=0)

        ct.store(
            Output,
            index=(idx_b, idx_t, idx_hv, idx_v),
            tile=ct.astype(ct.reshape(OutT, (1, 1, 1, BLOCK_V)), Output.dtype),
            allow_tma=SMALL_TILE_USE_TMA,
        )

    if OUTPUT_FINAL_STATE:
        ct.store(
            StateOut,
            index=(idx_b, idx_hv, 0, idx_v),
            tile=ct.reshape(State, (1, 1, BLOCK_K, BLOCK_V)),
            allow_tma=LARGE_TILE_USE_TMA,
        )


@ct.kernel(occupancy=2)
def _recurrent_gated_delta_rule_fwd_kernel_persistent(
    Query,  # (B, T, H, QK)
    Key,  # (B, T, H, QK)
    Value,  # (B, T, HV, V)
    Gate,  # (B, T, HV)
    Beta,  # (B, T, HV)
    Output,  # (B, T, HV, V)
    StateIn,  # (B, HV, QK, V)
    StateOut,  # (B, HV, QK, V)
    scale: float,
    HAS_INITIAL_STATE: ct.Constant[bool],
    OUTPUT_FINAL_STATE: ct.Constant[bool],
    USE_QK_L2NORM: ct.Constant[bool],
    T: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    BLOCK_V: ct.Constant[int],
    SMALL_TILE_USE_TMA: ct.Constant[bool],
    LARGE_TILE_USE_TMA: ct.Constant[bool],
):
    """Grid: (min(NUM_SMS, B * HV * cdiv(V, BLOCK_V)),); grid-strides over (b, hv, pid_v)."""
    idx_CGA = ct.bid(0)
    num_CGAs = ct.num_blocks(0)

    B = Query.shape[0]
    H = Query.shape[2]
    HV = Value.shape[2]
    NUM_V_BLOCKS = ct.cdiv(Value.shape[3], BLOCK_V)
    NUM_BLOCKS = B * HV * NUM_V_BLOCKS
    H_PER_GROUP = HV // H

    for idx_block in range(idx_CGA, NUM_BLOCKS, num_CGAs):
        idx_bhv = idx_block // NUM_V_BLOCKS
        idx_v = idx_block % NUM_V_BLOCKS
        idx_b = idx_bhv // HV
        idx_hv = idx_bhv % HV
        idx_h = idx_hv // H_PER_GROUP

        if HAS_INITIAL_STATE:
            State = ct.load(
                StateIn,
                index=(idx_b, idx_hv, 0, idx_v),
                shape=(1, 1, BLOCK_K, BLOCK_V),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=LARGE_TILE_USE_TMA,
            ).reshape((BLOCK_K, BLOCK_V))
            State = ct.astype(State, ct.float32)
        else:
            State = ct.zeros((BLOCK_K, BLOCK_V), dtype=ct.float32)

        for idx_t in range(T):
            QueryT = ct.load(
                Query,
                index=(idx_b, idx_t, idx_h, 0),
                shape=(1, 1, 1, BLOCK_K),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=LARGE_TILE_USE_TMA,
            ).reshape((BLOCK_K,))
            QueryT = ct.astype(QueryT, ct.float32)

            KeyT = ct.load(
                Key,
                index=(idx_b, idx_t, idx_h, 0),
                shape=(1, 1, 1, BLOCK_K),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=LARGE_TILE_USE_TMA,
            ).reshape((BLOCK_K,))
            KeyT = ct.astype(KeyT, ct.float32)

            if USE_QK_L2NORM:
                QueryT = QueryT * ct.rsqrt(ct.sum(QueryT * QueryT, axis=0) + 1e-6)
                KeyT = KeyT * ct.rsqrt(ct.sum(KeyT * KeyT, axis=0) + 1e-6)
            QueryT = QueryT * scale

            ValueT = ct.load(
                Value,
                index=(idx_b, idx_t, idx_hv, idx_v),
                shape=(1, 1, 1, BLOCK_V),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=SMALL_TILE_USE_TMA,
            ).reshape((BLOCK_V,))
            ValueT = ct.astype(ValueT, ct.float32)

            gate_t = ct.astype(ct.gather(Gate, (idx_b, idx_t, idx_hv), check_bounds=False), ct.float32)
            beta_t = ct.astype(ct.gather(Beta, (idx_b, idx_t, idx_hv), check_bounds=False), ct.float32)

            State = State * ct.exp(gate_t)
            KeyT = ct.expand_dims(KeyT, axis=1)
            KvMemT = ct.sum(State * KeyT, axis=0)
            Delta = (ValueT - KvMemT) * beta_t
            State = State + KeyT * ct.expand_dims(Delta, axis=0)
            OutputT = ct.sum(State * ct.expand_dims(QueryT, axis=1), axis=0)

            ct.store(
                Output,
                index=(idx_b, idx_t, idx_hv, idx_v),
                tile=ct.astype(ct.reshape(OutputT, (1, 1, 1, BLOCK_V)), Output.dtype),
                allow_tma=SMALL_TILE_USE_TMA,
            )

        if OUTPUT_FINAL_STATE:
            ct.store(
                StateOut,
                index=(idx_b, idx_hv, 0, idx_v),
                tile=ct.reshape(State, (1, 1, BLOCK_K, BLOCK_V)),
                allow_tma=LARGE_TILE_USE_TMA,
            )


def _autotune_configs(V: int, B: int, Hv: int, num_sms: int):
    # Work-aware BLOCK_V: aim for ~2x SM oversubscription on non-persistent grid
    # (B * Hv * ceil(V / BLOCK_V)). Smaller B -> smaller BLOCK_V -> more V-blocks.
    target_v_blocks = max(1, 2 * num_sms // max(1, B * Hv))
    target_bv = 1 << (max(8, V // target_v_blocks) - 1).bit_length()
    target_bv = min(V, target_bv)
    block_v_candidates = sorted({max(8, target_bv // 2), target_bv, min(V, target_bv * 2)})
    use_tma_small_large_resp = [(False, True), (True, True)]
    for block_v in block_v_candidates:
        for occupancy in (2, 3, 4, 6):
            for small_tile_use_tma, large_tile_use_tma in use_tma_small_large_resp:
                yield SimpleNamespace(
                    BLOCK_V=block_v,
                    occupancy=occupancy,
                    SMALL_TILE_USE_TMA=small_tile_use_tma,
                    LARGE_TILE_USE_TMA=large_tile_use_tma,
                )


def _grid(persistent, B, HV, V, BLOCK_V, device):
    num_v_blocks = ct.cdiv(V, BLOCK_V)
    if persistent:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        return (min(num_sms, B * HV * num_v_blocks),)
    return (B * HV, num_v_blocks, 1)


def _autotune(
    query,
    key,
    value,
    g,
    beta,
    output,
    initial_state,
    final_state,
    scale,
    has_initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel,
    B,
    T,
    Hv,
    V,
    BLOCK_K,
    persistent,
):
    dummy = torch.empty(1, 1, 1, 1, device=query.device, dtype=torch.float32)
    device = query.device

    def args_fn(cfg):
        return (
            query,
            key,
            value,
            g,
            beta,
            output,
            initial_state if has_initial_state else dummy,
            final_state if output_final_state else dummy,
            scale,
            has_initial_state,
            output_final_state,
            use_qk_l2norm_in_kernel,
            T,
            BLOCK_K,
            cfg.BLOCK_V,
            cfg.SMALL_TILE_USE_TMA,
            cfg.LARGE_TILE_USE_TMA,
        )

    def make_grid_fn(persistent):
        return lambda cfg: _grid(persistent, B, Hv, V, cfg.BLOCK_V, device)

    def hints_fn(cfg):
        return {"occupancy": cfg.occupancy}

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    kernel_candidates = {
        False: _recurrent_gated_delta_rule_fwd_kernel,
        True: _recurrent_gated_delta_rule_fwd_kernel_persistent,
    }
    if persistent is not None:
        kernel_candidates = {persistent: kernel_candidates[persistent]}
    best_kernel, best_config, best_is_persistent, best_time = None, None, None, float("inf")
    for persistent, kernel in kernel_candidates.items():
        result = ct.tune.exhaustive_search(
            list(_autotune_configs(V, B, Hv, num_sms)),
            torch.cuda.current_stream(),
            make_grid_fn(persistent),
            kernel,
            args_fn,
            hints_fn,
            quiet=True,
        )
        if result.best.mean_us < best_time:
            best_time = result.best.mean_us
            best_kernel, best_config, best_is_persistent = kernel, result.best.config, persistent
    assert best_kernel is not None
    best_kernel = best_kernel.replace_hints(occupancy=best_config.occupancy)
    return best_kernel, best_config, best_is_persistent


class _RecurrentGatedDeltaRuleCuTile(torch.autograd.Function):
    autotune_cache = {}

    @staticmethod
    def forward(
        ctx, query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel, persistent
    ):
        B, T, H, QK = query.shape
        HV, V = value.shape[-2:]
        assert H <= HV and HV % H == 0
        initial_dtype = query.dtype
        device = query.device

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        if has_initial_state := (initial_state is not None):
            initial_state = initial_state.contiguous()

        output = torch.empty(B, T, HV, V, device=device, dtype=initial_dtype)
        final_state = torch.empty(B, HV, QK, V, device=device, dtype=torch.float32) if output_final_state else None

        BLOCK_K = next_power_of_2(QK)
        scale = 1.0 / math.sqrt(QK)

        if os.environ.get("DISABLE_AUTOTUNE", "0") == "1":
            best_kernel = (
                _recurrent_gated_delta_rule_fwd_kernel_persistent
                if persistent
                else _recurrent_gated_delta_rule_fwd_kernel
            )
            best_config = SimpleNamespace(
                BLOCK_V=64,
                occupancy=2,
                SMALL_TILE_USE_TMA=True,
                LARGE_TILE_USE_TMA=True,
            )
            best_is_persistent = bool(persistent)
        else:
            cache_key = (
                B,
                T,
                H,
                HV,
                QK,
                V,
                initial_dtype,
                has_initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
                persistent,
                str(device),
            )
            if (cached := _RecurrentGatedDeltaRuleCuTile.autotune_cache.get(cache_key)) is None:
                cached = _autotune(
                    query,
                    key,
                    value,
                    g,
                    beta,
                    output,
                    initial_state,
                    final_state,
                    scale,
                    has_initial_state,
                    output_final_state,
                    use_qk_l2norm_in_kernel,
                    B,
                    T,
                    HV,
                    V,
                    BLOCK_K,
                    persistent,
                )
                _RecurrentGatedDeltaRuleCuTile.autotune_cache[cache_key] = cached
            best_kernel, best_config, best_is_persistent = cached

        grid = _grid(best_is_persistent, B, HV, V, best_config.BLOCK_V, device)

        if has_initial_state and output_final_state:
            init_arg, final_arg = initial_state, final_state
        else:
            dummy = torch.empty(1, 1, 1, 1, device=device, dtype=torch.float32)
            init_arg = initial_state if has_initial_state else dummy
            final_arg = final_state if output_final_state else dummy

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            best_kernel,
            (
                query,
                key,
                value,
                g,
                beta,
                output,
                init_arg,
                final_arg,
                scale,
                has_initial_state,
                output_final_state,
                use_qk_l2norm_in_kernel,
                T,
                BLOCK_K,
                best_config.BLOCK_V,
                best_config.SMALL_TILE_USE_TMA,
                best_config.LARGE_TILE_USE_TMA,
            ),
        )

        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        raise NotImplementedError("Backward pass not implemented for RecurrentGatedDeltaRuleCuTile")


@register_impl("recurrent_gated_delta_rule", backend="cutile")
def recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    """Drop-in cuTile replacement for torch_recurrent_gated_delta_rule."""
    return _RecurrentGatedDeltaRuleCuTile.apply(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        kwargs.get("persistent"),
    )
