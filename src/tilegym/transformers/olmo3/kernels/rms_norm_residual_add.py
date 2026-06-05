# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""OLMo3 fused RMSNorm and residual add cuTile kernel."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _rms_norm_residual_add_kernel(
    x,  # (N, D)
    residual,  # (N, D)
    weight,  # (D,)
    out,  # (N, D)
    eps: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    h = ct.astype(ct.gather(x, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    r = ct.astype(ct.gather(residual, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (offs,), padding_value=0.0, check_bounds=True), ct.float32)

    variance = ct.sum(h * h) * ct.truediv(1.0, D)
    normed = h * ct.rsqrt(variance + eps) * w
    result = r + normed

    ct.scatter(out, (bid, offs), ct.astype(result, out.dtype), check_bounds=True)


def rms_norm_residual_add_cutile(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm + residual add. Returns residual + rms_norm(x)."""
    D = x.shape[-1]
    x_flat = x.contiguous().view(-1, D)
    r_flat = residual.contiguous().view(-1, D)
    N = x_flat.shape[0]
    out = torch.empty_like(x_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _rms_norm_residual_add_kernel,
        (x_flat, r_flat, weight, out, eps, D, TILE_D),
    )
    return out.view(x.shape)
