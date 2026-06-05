# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Qwen3.5 fused gated-delta-rule gate preprocessing cuTile kernel."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _gdr_preprocess_kernel(
    b_in,  # (N, H)
    a_in,  # (N, H)
    A_log,  # (H,)
    dt_bias,  # (H,)
    beta_out,  # (N, H)
    g_out,  # (N, H)
    TILE_H: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_H, dtype=ct.int32)

    b = ct.astype(ct.gather(b_in, (bid, offs), check_bounds=True), ct.float32)
    a = ct.astype(ct.gather(a_in, (bid, offs), check_bounds=True), ct.float32)
    a_log = ct.astype(ct.gather(A_log, (offs,), check_bounds=True), ct.float32)
    dt_b = ct.astype(ct.gather(dt_bias, (offs,), check_bounds=True), ct.float32)

    beta = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(b))))
    sp_arg = a + dt_b
    sp = ct.log(ct.add(1.0, ct.exp(sp_arg)))
    g = ct.negative(ct.exp(a_log) * sp)

    ct.scatter(beta_out, (bid, offs), ct.astype(beta, beta_out.dtype), check_bounds=True)
    ct.scatter(g_out, (bid, offs), ct.astype(g, g_out.dtype), check_bounds=True)


def gdr_preprocess_cutile(
    b: torch.Tensor, a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused computation of beta=sigmoid(b) and g=-exp(A_log)*softplus(a+dt_bias)."""
    orig_shape = b.shape
    H = orig_shape[-1]
    b_flat = b.contiguous().view(-1, H)
    a_flat = a.contiguous().view(-1, H)
    N = b_flat.shape[0]
    beta = torch.empty_like(b_flat)
    g = torch.empty(N, H, dtype=torch.float32, device=b.device)
    TILE_H = 1 << (H - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _gdr_preprocess_kernel,
        (b_flat, a_flat, A_log, dt_bias, beta, g, TILE_H),
    )
    return beta.view(orig_shape), g.view(orig_shape)
