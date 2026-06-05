# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Qwen3.5 fused sigmoid(gate) * x cuTile kernel."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _sigmoid_mul_kernel(
    x,  # (N, D)
    gate,  # (N, D)
    output,  # (N, D)
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    xv = ct.astype(ct.gather(x, (bid, offs), check_bounds=True), ct.float32)
    gv = ct.astype(ct.gather(gate, (bid, offs), check_bounds=True), ct.float32)

    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(gv))))
    result = ct.astype(xv * sig, output.dtype)
    ct.scatter(output, (bid, offs), result, check_bounds=True)


def sigmoid_mul_cutile(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute x * sigmoid(gate) using a fused cuTile kernel."""
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.contiguous().view(-1, D)
    gate_flat = gate.contiguous().view(-1, D)
    N = x_flat.shape[0]
    output = torch.empty_like(x_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(torch.cuda.current_stream(), (N,), _sigmoid_mul_kernel, (x_flat, gate_flat, output, TILE_D))
    return output.view(orig_shape)
