# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Qwen3.5 fused causal conv1d and SiLU prefill-path cuTile kernel."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _causal_conv1d_prefill_silu_kernel(
    x,  # (D, T)
    weight,  # (D, K)
    output,  # (D, T)
    T: ConstInt,
    BLOCK_T: ConstInt,
):
    bid_d = ct.bid(0)
    bid_t = ct.bid(1)
    t_start = bid_t * BLOCK_T
    offs = ct.arange(BLOCK_T, dtype=ct.int32)
    t_idx = t_start + offs

    w0 = ct.astype(ct.gather(weight, (bid_d, 0), check_bounds=True), ct.float32)
    w1 = ct.astype(ct.gather(weight, (bid_d, 1), check_bounds=True), ct.float32)
    w2 = ct.astype(ct.gather(weight, (bid_d, 2), check_bounds=True), ct.float32)
    w3 = ct.astype(ct.gather(weight, (bid_d, 3), check_bounds=True), ct.float32)

    v0 = ct.astype(ct.gather(x, (bid_d, t_idx), padding_value=0.0, check_bounds=True), ct.float32)
    v1 = ct.astype(ct.gather(x, (bid_d, t_idx + 1), padding_value=0.0, check_bounds=True), ct.float32)
    v2 = ct.astype(ct.gather(x, (bid_d, t_idx + 2), padding_value=0.0, check_bounds=True), ct.float32)
    v3 = ct.astype(ct.gather(x, (bid_d, t_idx + 3), padding_value=0.0, check_bounds=True), ct.float32)

    dot = v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3
    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(dot))))
    result = dot * sig

    ct.scatter(output, (bid_d, t_idx), ct.astype(result, output.dtype), check_bounds=True)


def causal_conv1d_prefill_silu_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Fused causal depthwise conv1d + SiLU for prefill path."""
    B, D, _T_padded = x.shape
    assert B == 1, "causal_conv1d_prefill_silu only supports B=1"

    x_2d = x.squeeze(0).contiguous()
    w = weight.contiguous()
    output = torch.empty(D, seq_len, dtype=x.dtype, device=x.device)

    BLOCK_T = 256
    grid = (D, (seq_len + BLOCK_T - 1) // BLOCK_T)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _causal_conv1d_prefill_silu_kernel,
        (x_2d, w, output, seq_len, BLOCK_T),
    )
    return output.unsqueeze(0)
