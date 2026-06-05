# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Qwen3.5 fused residual add and Gemma-style RMSNorm cuTile kernel."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _residual_add_rms_norm_kernel(
    residual,  # (N, D)
    x,  # (N, D)
    weight,  # (D,)
    sum_out,  # (N, D)
    normed_out,  # (N, D)
    eps: float,
    offset: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    r = ct.astype(ct.gather(residual, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    h = ct.astype(ct.gather(x, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (offs,), padding_value=0.0, check_bounds=True), ct.float32)

    s = r + h
    variance = ct.sum(s * s) * ct.truediv(1.0, D)
    normed = s * ct.rsqrt(variance + eps) * (offset + w)

    ct.scatter(sum_out, (bid, offs), ct.astype(s, sum_out.dtype), check_bounds=True)
    ct.scatter(normed_out, (bid, offs), ct.astype(normed, normed_out.dtype), check_bounds=True)


def residual_add_rms_norm_cutile(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    offset: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + Gemma-style RMSNorm. Returns (sum, normed)."""
    D = residual.shape[-1]
    r_flat = residual.contiguous().view(-1, D)
    x_flat = x.contiguous().view(-1, D)
    N = r_flat.shape[0]
    sum_out = torch.empty_like(r_flat)
    normed_out = torch.empty_like(r_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _residual_add_rms_norm_kernel,
        (r_flat, x_flat, weight, sum_out, normed_out, eps, offset, D, TILE_D),
    )
    return sum_out.view(residual.shape), normed_out.view(residual.shape)
