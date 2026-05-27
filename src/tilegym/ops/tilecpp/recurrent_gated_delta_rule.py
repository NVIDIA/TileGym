# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Recurrent Gated Delta Rule (Qwen3-Next linear-attention variant).

"""

import math
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "recurrent_gated_delta_rule.cuh",
    kernel_name="recurrent_gated_delta_rule_fwd_kernel",
)


def _next_power_of_2(n):
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _launch_fwd(
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
    use_qk_l2norm,
):
    dtype = query.dtype
    dump_kernel_types("recurrent_gated_delta_rule_fwd_kernel", query, key, value)

    B, T, H, K_HEAD_DIM = query.shape
    V_HEAD_DIM = value.shape[-1]

    BLOCK_K = _next_power_of_2(K_HEAD_DIM)
    BLOCK_V = min(64, _next_power_of_2(V_HEAD_DIM))
    if BLOCK_K != K_HEAD_DIM:
        raise NotImplementedError(
            f"tilecpp recurrent_gated_delta_rule requires K head dim to be a power of two "
            f"(got K={K_HEAD_DIM}); the kernel does unmasked tile loads of size BLOCK_K=next_pow2(K)."
        )
    if V_HEAD_DIM % BLOCK_V != 0:
        raise NotImplementedError(
            f"tilecpp recurrent_gated_delta_rule requires V head dim ({V_HEAD_DIM}) to be a multiple "
            f"of BLOCK_V={BLOCK_V}; the kernel does unmasked tile loads of BLOCK_V-wide V tiles."
        )

    # Template params: BLOCK_K, BLOCK_V, HAS_INITIAL_STATE, OUTPUT_FINAL_STATE, USE_QK_L2NORM
    bool_to_str = lambda b: "true" if b else "false"
    template_params = [
        BLOCK_K,
        BLOCK_V,
        bool_to_str(has_initial_state),
        bool_to_str(output_final_state),
        bool_to_str(use_qk_l2norm),
    ]

    kernel, _, _ = _fwd_kernel.get_kernel(
        dtype=dtype,
        template_params=template_params,
        signature="const {T}*, const {T}*, const {T}*, const {T}*, const {T}*, {T}*, const float*, float*, float, int, int, int, int, int",
    )

    grid = (B * H, (V_HEAD_DIM + BLOCK_V - 1) // BLOCK_V, 1)

    init_ptr = np.uint64(initial_state.data_ptr()) if has_initial_state else np.uint64(0)
    final_ptr = np.uint64(final_state.data_ptr()) if output_final_state else np.uint64(0)

    _fwd_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(query.data_ptr()),
            np.uint64(key.data_ptr()),
            np.uint64(value.data_ptr()),
            np.uint64(g.data_ptr()),
            np.uint64(beta.data_ptr()),
            np.uint64(output.data_ptr()),
            init_ptr,
            final_ptr,
            np.float32(scale),
            np.int32(B),
            np.int32(T),
            np.int32(H),
            np.int32(K_HEAD_DIM),
            np.int32(V_HEAD_DIM),
        ],
    )


class TileCppRecurrentGatedDeltaRule(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    ):
        B, T, H, K = query.shape
        V = value.shape[-1]
        scale = 1.0 / math.sqrt(K)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()

        output = torch.empty(B, T, H, V, device=query.device, dtype=query.dtype)

        has_initial_state = initial_state is not None
        if has_initial_state:
            initial_state = initial_state.contiguous().float()

        final_state = None
        if output_final_state:
            final_state = torch.empty(B, H, K, V, device=query.device, dtype=torch.float32)

        _launch_fwd(
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
        )

        return output, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        raise NotImplementedError("Backward not implemented for TileCppRecurrentGatedDeltaRule")


@register_impl("recurrent_gated_delta_rule", backend="tilecpp")
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
    """Drop-in CUDA Tile C++ replacement for torch_recurrent_gated_delta_rule."""
    return TileCppRecurrentGatedDeltaRule.apply(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
    )
