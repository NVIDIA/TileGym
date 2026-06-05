# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Qwen3.5 fused SiLU(gate) * up cuTile kernel for separate tensors."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _silu_and_mul_separate_kernel(
    gate,  # (N, D)
    up,  # (N, D)
    output,  # (N, D)
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    g = ct.astype(ct.gather(gate, (bid, offs), check_bounds=True), ct.float32)
    u = ct.astype(ct.gather(up, (bid, offs), check_bounds=True), ct.float32)

    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(g))))
    result = ct.astype(g * sig * u, output.dtype)
    ct.scatter(output, (bid, offs), result, check_bounds=True)


def silu_and_mul_separate_cutile(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute silu(gate) * up without concatenation, using a fused cuTile kernel."""
    orig_shape = gate.shape
    D = orig_shape[-1]
    gate_flat = gate.contiguous().view(-1, D)
    up_flat = up.contiguous().view(-1, D)
    N = gate_flat.shape[0]
    output = torch.empty_like(gate_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(torch.cuda.current_stream(), (N,), _silu_and_mul_separate_kernel, (gate_flat, up_flat, output, TILE_D))
    return output.view(orig_shape)
