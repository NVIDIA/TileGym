# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ gemma_attention (prefill FMHA with soft cap + sliding window).
"""

import math
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.autotuner import Config
from tilegym.ops.tilecpp.autotuner import TileCppAutotuner
from tilegym.ops.tilecpp.autotuner import autotune
from tilegym.ops.tilecpp.autotuner import is_autotuning_enabled
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "gemma_attention.cuh",
    kernel_name="gemma_attention_fwd_kernel",
)


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _autotune_configs():
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        return [
            Config(BLOCK_M=64, BLOCK_N=64, num_ctas=1, occupancy=2),
            Config(BLOCK_M=128, BLOCK_N=64, num_ctas=1, occupancy=2),
        ]
    else:
        # sm9+ / sm10+ (Hopper / Blackwell).
        return [
            Config(BLOCK_M=256, BLOCK_N=128, num_ctas=1, occupancy=1),
            Config(BLOCK_M=128, BLOCK_N=128, num_ctas=1, occupancy=2),
        ]


def _default_config():
    """
    `BLOCK_M=128, BLOCK_N=128` + `@ct.kernel(occupancy=2)`.
    """
    return Config(BLOCK_M=128, BLOCK_N=128, num_ctas=1, occupancy=2)


def _launch(
    cfg: Config,
    Q,
    K,
    V,
    Out,
    sm_scale,
    soft_cap,
    B,
    H,
    H_kv,
    S_qo,
    S_kv,
    block_d,
    is_causal,
    window_size,
    has_soft_cap,
):
    """Launch gemma_attention_fwd_kernel with the given config."""
    dump_kernel_types("gemma_attention_fwd_kernel", Q, K, V, Out)

    occupancy = cfg.occupancy if cfg.occupancy is not None else 2
    bool_to_str = lambda b: "true" if b else "false"
    template_params = [
        B,
        H,
        H_kv,
        S_qo,
        S_kv,
        cfg.BLOCK_M,
        cfg.BLOCK_N,
        block_d,
        bool_to_str(is_causal),
        window_size,
        bool_to_str(has_soft_cap),
        occupancy,
    ]

    kernel, _, _ = _kernel.get_kernel(
        dtype=Q.dtype,
        template_params=template_params,
        signature="const {T}*, const {T}*, const {T}*, {T}*, float, float",
    )

    grid = ((S_qo + cfg.BLOCK_M - 1) // cfg.BLOCK_M, B * H, 1)

    _kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(Q.data_ptr()),
            np.uint64(K.data_ptr()),
            np.uint64(V.data_ptr()),
            np.uint64(Out.data_ptr()),
            np.float32(sm_scale),
            np.float32(soft_cap),
        ],
    )


@autotune(search_space=_autotune_configs())
def _autotune_gemma_attention(
    Q,
    K,
    V,
    Out,
    sm_scale,
    soft_cap,
    B,
    H,
    H_kv,
    S_qo,
    S_kv,
    block_d,
    is_causal,
    window_size,
    has_soft_cap,
    autotuner: TileCppAutotuner | None = None,
):
    key = (
        "tilecpp_gemma_attention",
        str(Q.dtype),
        B,
        H,
        H_kv,
        S_qo,
        S_kv,
        block_d,
        is_causal,
        window_size,
        has_soft_cap,
    )

    def launch_fn(cfg: Config):
        _launch(
            cfg,
            Q,
            K,
            V,
            Out,
            sm_scale,
            soft_cap,
            B,
            H,
            H_kv,
            S_qo,
            S_kv,
            block_d,
            is_causal,
            window_size,
            has_soft_cap,
        )

    def grid_fn(args: dict, cfg: Config) -> tuple:
        return ((S_qo + cfg.BLOCK_M - 1) // cfg.BLOCK_M, B * H, 1)

    autotuner(
        torch.cuda.current_stream(),
        key=key,
        launch_fn=launch_fn,
        grid_fn=grid_fn,
        named_args={},
    )


# =============================================================================
# Public Interface
# =============================================================================


class _GemmaAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, window_size=0, soft_cap=None, is_causal=True, use_autotune=False):
        B, H, S_qo, D = q.shape
        _, H_kv, S_kv, _ = k.shape
        BLOCK_D = _next_power_of_2(D)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = torch.empty_like(q)

        assert H % H_kv == 0

        has_soft_cap = soft_cap is not None
        soft_cap_val = float(soft_cap) if has_soft_cap else 0.0
        win = window_size if window_size else 0

        if use_autotune or is_autotuning_enabled():
            _autotune_gemma_attention(
                q,
                k,
                v,
                o,
                sm_scale,
                soft_cap_val,
                B,
                H,
                H_kv,
                S_qo,
                S_kv,
                BLOCK_D,
                is_causal,
                win,
                has_soft_cap,
            )
        else:
            _launch(
                _default_config(),
                q,
                k,
                v,
                o,
                sm_scale,
                soft_cap_val,
                B,
                H,
                H_kv,
                S_qo,
                S_kv,
                BLOCK_D,
                is_causal,
                win,
                has_soft_cap,
            )

        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass not implemented for gemma attention")


gemma_attention = _GemmaAttention.apply


@register_impl("gemma_attention", backend="tilecpp")
def gemma_attention_tilecpp(
    q,
    k,
    v,
    scaling=None,
    window_size=0,
    soft_cap=None,
    is_causal=True,
    use_autotune=False,
    **kwargs,
):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))
    return gemma_attention(q, k, v, scaling, window_size, soft_cap, is_causal, use_autotune)
