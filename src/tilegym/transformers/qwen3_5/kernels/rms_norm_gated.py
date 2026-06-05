# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Qwen3.5 fused RMSNorm with SiLU gate cuTile kernel."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _rms_norm_gated_silu_kernel(
    hidden_states,  # (N, D)
    gate,  # (N, D)
    weight,  # (D,)
    output,  # (N, D)
    eps: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    h = ct.astype(ct.gather(hidden_states, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    g = ct.astype(ct.gather(gate, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (offs,), padding_value=0.0, check_bounds=True), ct.float32)

    variance = ct.sum(h * h) * ct.truediv(1.0, D)
    normed = h * ct.rsqrt(variance + eps)
    sig_g = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(g))))
    silu_g = g * sig_g
    result = ct.astype(w * normed * silu_g, output.dtype)

    ct.scatter(output, (bid, offs), result, check_bounds=True)


def rms_norm_gated_cutile(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm + SiLU gating as cuTile kernel."""
    D = hidden_states.shape[-1]
    h_flat = hidden_states.contiguous().view(-1, D)
    g_flat = gate.contiguous().view(-1, D)
    N = h_flat.shape[0]
    output = torch.empty_like(h_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _rms_norm_gated_silu_kernel,
        (h_flat, g_flat, weight, output, eps, D, TILE_D),
    )
    return output.view(hidden_states.shape)
