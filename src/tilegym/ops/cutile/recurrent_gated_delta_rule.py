# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import os
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
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
    BLOCK_K: ct.Constant[int],
    BLOCK_V: ct.Constant[int],
    SMALL_TILE_USE_TMA: ct.Constant[bool],
    LARGE_TILE_USE_TMA: ct.Constant[bool],
    T: ct.Constant[int],
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
    BLOCK_K: ct.Constant[int],
    BLOCK_V: ct.Constant[int],
    SMALL_TILE_USE_TMA: ct.Constant[bool],
    LARGE_TILE_USE_TMA: ct.Constant[bool],
    T: ct.Constant[int],
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


@ct.kernel(occupancy=2)
def _recurrent_gated_delta_rule_fwd_kernel_decode_vstream(
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
    BLOCK_K: ct.Constant[int],
    BLOCK_V: ct.Constant[int],
    SMALL_TILE_USE_TMA: ct.Constant[bool],
    LARGE_TILE_USE_TMA: ct.Constant[bool],
    STREAM_V_TILE: ct.Constant[int],
):
    """Decode-only stream-V kernel. Grid: (B * HV, ceil(V / BLOCK_V), 1)."""
    idx_bhv = ct.bid(0)
    idx_v_block = ct.bid(1)

    H = Query.shape[2]
    HV = Value.shape[2]
    V = Value.shape[3]
    idx_b = idx_bhv // HV
    idx_hv = idx_bhv % HV
    idx_h = idx_hv // (HV // H)

    remaining_v = V - idx_v_block * BLOCK_V
    valid_v = min(BLOCK_V, remaining_v)
    num_stream_v_tiles = ct.cdiv(valid_v, STREAM_V_TILE)

    QueryT = ct.load(
        Query,
        index=(idx_b, 0, idx_h, 0),
        shape=(1, 1, 1, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=LARGE_TILE_USE_TMA,
    ).reshape((BLOCK_K,))
    KeyT = ct.load(
        Key,
        index=(idx_b, 0, idx_h, 0),
        shape=(1, 1, 1, BLOCK_K),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=LARGE_TILE_USE_TMA,
    ).reshape((BLOCK_K,))
    QueryT = ct.astype(QueryT, ct.float32)
    KeyT = ct.astype(KeyT, ct.float32)

    if USE_QK_L2NORM:
        QueryT = QueryT * ct.rsqrt(ct.sum(QueryT * QueryT, axis=0) + 1e-6)
        KeyT = KeyT * ct.rsqrt(ct.sum(KeyT * KeyT, axis=0) + 1e-6)
    QueryT = QueryT * scale

    gate_t = ct.astype(ct.gather(Gate, (idx_b, 0, idx_hv), check_bounds=False), ct.float32)
    beta_t = ct.astype(ct.gather(Beta, (idx_b, 0, idx_hv), check_bounds=False), ct.float32)
    gamma_t = ct.exp(gate_t)

    KeyCol = ct.expand_dims(KeyT, axis=1)
    QueryCol = ct.expand_dims(QueryT, axis=1)
    key_query_dot = ct.sum(KeyT * QueryT, axis=0)

    for idx_stream_v in range(num_stream_v_tiles):
        idx_sv = idx_v_block * (BLOCK_V // STREAM_V_TILE) + idx_stream_v
        ValueT = ct.load(
            Value,
            index=(idx_b, 0, idx_hv, idx_sv),
            shape=(1, 1, 1, STREAM_V_TILE),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=SMALL_TILE_USE_TMA,
        ).reshape((STREAM_V_TILE,))
        ValueT = ct.astype(ValueT, ct.float32)

        if HAS_INITIAL_STATE:
            State = ct.load(
                StateIn,
                index=(idx_b, idx_hv, 0, idx_sv),
                shape=(1, 1, BLOCK_K, STREAM_V_TILE),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=LARGE_TILE_USE_TMA,
            ).reshape((BLOCK_K, STREAM_V_TILE))
            State = ct.astype(State, ct.float32)
        else:
            State = ct.zeros((BLOCK_K, STREAM_V_TILE), dtype=ct.float32)

        State = State * gamma_t
        KvMemT = ct.sum(State * KeyCol, axis=0)
        OutBaseT = ct.sum(State * QueryCol, axis=0)
        DeltaT = (ValueT - KvMemT) * beta_t
        OutT = OutBaseT + DeltaT * key_query_dot
        State = State + KeyCol * ct.expand_dims(DeltaT, axis=0)

        ct.store(
            Output,
            index=(idx_b, 0, idx_hv, idx_sv),
            tile=ct.astype(ct.reshape(OutT, (1, 1, 1, STREAM_V_TILE)), Output.dtype),
            allow_tma=SMALL_TILE_USE_TMA,
        )

        if OUTPUT_FINAL_STATE:
            ct.store(
                StateOut,
                index=(idx_b, idx_hv, 0, idx_sv),
                tile=ct.reshape(State, (1, 1, BLOCK_K, STREAM_V_TILE)),
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
                    KERNEL="standard",
                    BLOCK_V=block_v,
                    occupancy=occupancy,
                    SMALL_TILE_USE_TMA=small_tile_use_tma,
                    LARGE_TILE_USE_TMA=large_tile_use_tma,
                )


def _persistent_autotune_configs(V: int, B: int, Hv: int, num_sms: int):
    for cfg in _autotune_configs(V=V, B=B, Hv=Hv, num_sms=num_sms):
        yield SimpleNamespace(
            KERNEL="persistent",
            BLOCK_V=cfg.BLOCK_V,
            occupancy=cfg.occupancy,
            SMALL_TILE_USE_TMA=cfg.SMALL_TILE_USE_TMA,
            LARGE_TILE_USE_TMA=cfg.LARGE_TILE_USE_TMA,
        )


def _decode_vstream_autotune_configs(V: int):
    # BV: [16, next_power_of_2(V)]
    block_v_candidates = [2**i for i in range(4, max(5, math.ceil(math.log2(V)) + 1))]
    use_tma_small_large_resp = [(False, True), (True, True)]
    for block_v in block_v_candidates:
        # SV: At least 2 stream-V steps.
        stream_v_candidates = [2**i for i in range(3, max(4, int(math.log2(block_v))))]
        for stream_v_tile in stream_v_candidates:
            for occupancy in (1, 2, 4):
                for small_tile_use_tma, large_tile_use_tma in use_tma_small_large_resp:
                    yield SimpleNamespace(
                        KERNEL="decode_vstream",
                        BLOCK_V=block_v,
                        STREAM_V_TILE=stream_v_tile,
                        occupancy=occupancy,
                        SMALL_TILE_USE_TMA=small_tile_use_tma,
                        LARGE_TILE_USE_TMA=large_tile_use_tma,
                    )


@dataclass(frozen=True)
class _KernelCandidate:
    kernel: ct.kernel
    configs: Sequence[SimpleNamespace]
    args_fn: Callable[[SimpleNamespace], tuple[torch.Tensor | float | bool | int, ...]]


def _grid(persistent, B, HV, V, BLOCK_V, device):
    num_v_blocks = ct.cdiv(V, BLOCK_V)
    if persistent:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        return (min(num_sms, B * HV * num_v_blocks),)
    else:
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

    def common_args_fn(cfg):
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
            BLOCK_K,
            cfg.BLOCK_V,
            cfg.SMALL_TILE_USE_TMA,
            cfg.LARGE_TILE_USE_TMA,
        )

    def grid_fn(cfg):
        return _grid(cfg.KERNEL == "persistent", B, Hv, V, cfg.BLOCK_V, device)

    def hints_fn(cfg):
        return {"occupancy": cfg.occupancy}

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    default_candidate = _KernelCandidate(
        kernel=_recurrent_gated_delta_rule_fwd_kernel,
        configs=list(_autotune_configs(V, B, Hv, num_sms)),
        args_fn=lambda cfg: common_args_fn(cfg) + (T,),
    )
    persistent_candidate = _KernelCandidate(
        kernel=_recurrent_gated_delta_rule_fwd_kernel_persistent,
        configs=list(_persistent_autotune_configs(V, B, Hv, num_sms)),
        args_fn=lambda cfg: common_args_fn(cfg) + (T,),
    )
    if persistent is None:
        kernel_candidates = [default_candidate, persistent_candidate]
    elif persistent:
        kernel_candidates = [persistent_candidate]
    else:
        kernel_candidates = [default_candidate]

    if T == 1:
        kernel_candidates.append(
            _KernelCandidate(
                kernel=_recurrent_gated_delta_rule_fwd_kernel_decode_vstream,
                configs=list(_decode_vstream_autotune_configs(V)),
                args_fn=lambda cfg: common_args_fn(cfg) + (cfg.STREAM_V_TILE,),
            )
        )

    best_kernel, best_config, best_time = None, None, float("inf")
    for candidate in kernel_candidates:
        result = ct.tune.exhaustive_search(
            candidate.configs,
            torch.cuda.current_stream(),
            grid_fn,
            candidate.kernel,
            candidate.args_fn,
            hints_fn,
            quiet=True,
        )
        if result.best.mean_us < best_time:
            best_time = result.best.mean_us
            best_kernel, best_config = candidate.kernel, result.best.config
    assert best_kernel is not None
    best_kernel = best_kernel.replace_hints(occupancy=best_config.occupancy)
    return best_kernel, best_config


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
            if T == 1 and 32 <= V:
                best_kernel = _recurrent_gated_delta_rule_fwd_kernel_decode_vstream
                best_config = SimpleNamespace(
                    KERNEL="decode_vstream",
                    BLOCK_V=32,
                    STREAM_V_TILE=16,
                    SMALL_TILE_USE_TMA=True,
                    LARGE_TILE_USE_TMA=True,
                )
            elif persistent:
                best_kernel = _recurrent_gated_delta_rule_fwd_kernel_persistent
                best_config = SimpleNamespace(
                    KERNEL="persistent",
                    BLOCK_V=64,
                    SMALL_TILE_USE_TMA=True,
                    LARGE_TILE_USE_TMA=True,
                )
            else:
                best_kernel = _recurrent_gated_delta_rule_fwd_kernel
                best_config = SimpleNamespace(
                    KERNEL="standard",
                    BLOCK_V=64,
                    SMALL_TILE_USE_TMA=True,
                    LARGE_TILE_USE_TMA=True,
                )
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
            if cache_key not in _RecurrentGatedDeltaRuleCuTile.autotune_cache:
                best_kernel, best_config = _autotune(
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
                _RecurrentGatedDeltaRuleCuTile.autotune_cache[cache_key] = best_kernel, best_config
            else:
                best_kernel, best_config = _RecurrentGatedDeltaRuleCuTile.autotune_cache[cache_key]

        grid = _grid(best_config.KERNEL == "persistent", B, HV, V, best_config.BLOCK_V, device)

        if has_initial_state and output_final_state:
            init_arg, final_arg = initial_state, final_state
        else:
            dummy = torch.empty(1, 1, 1, 1, device=device, dtype=torch.float32)
            init_arg = initial_state if has_initial_state else dummy
            final_arg = final_state if output_final_state else dummy

        common_args = (
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
            BLOCK_K,
            best_config.BLOCK_V,
            best_config.SMALL_TILE_USE_TMA,
            best_config.LARGE_TILE_USE_TMA,
        )
        if best_config.KERNEL == "decode_vstream":
            kernel_args = common_args + (best_config.STREAM_V_TILE,)
        else:
            kernel_args = common_args + (T,)

        ct.launch(torch.cuda.current_stream(), grid, best_kernel, kernel_args)

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
