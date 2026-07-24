# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Qwen3.5 fused causal conv1d update and SiLU decode-path cuTile kernel."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _causal_conv1d_update_silu_kernel(
    x,  # (D,)
    conv_state,  # (D, 4)
    weight,  # (D, 4)
    output,  # (D,)
    BLOCK_D: ConstInt,
):
    bid = ct.bid(0)
    d_start = bid * BLOCK_D
    offs = ct.arange(BLOCK_D, dtype=ct.int32)
    d_idx = d_start + offs

    s1 = ct.astype(ct.gather(conv_state, (d_idx, 1), check_bounds=True), ct.float32)
    s2 = ct.astype(ct.gather(conv_state, (d_idx, 2), check_bounds=True), ct.float32)
    s3 = ct.astype(ct.gather(conv_state, (d_idx, 3), check_bounds=True), ct.float32)
    xv = ct.astype(ct.gather(x, (d_idx,), check_bounds=True), ct.float32)

    w0 = ct.astype(ct.gather(weight, (d_idx, 0), check_bounds=True), ct.float32)
    w1 = ct.astype(ct.gather(weight, (d_idx, 1), check_bounds=True), ct.float32)
    w2 = ct.astype(ct.gather(weight, (d_idx, 2), check_bounds=True), ct.float32)
    w3 = ct.astype(ct.gather(weight, (d_idx, 3), check_bounds=True), ct.float32)

    dot = s1 * w0 + s2 * w1 + s3 * w2 + xv * w3
    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(dot))))
    result = dot * sig

    ct.scatter(output, (d_idx,), ct.astype(result, output.dtype), check_bounds=True)
    ct.scatter(conv_state, (d_idx, 0), ct.astype(s1, conv_state.dtype), check_bounds=True)
    ct.scatter(conv_state, (d_idx, 1), ct.astype(s2, conv_state.dtype), check_bounds=True)
    ct.scatter(conv_state, (d_idx, 2), ct.astype(s3, conv_state.dtype), check_bounds=True)
    ct.scatter(conv_state, (d_idx, 3), ct.astype(xv, conv_state.dtype), check_bounds=True)


def causal_conv1d_update_silu_cutile(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    activation=None,
) -> torch.Tensor:
    """Fused causal conv1d update + SiLU for decode path (seq_len=1)."""
    B, D, seq_len = hidden_states.shape
    assert seq_len == 1, "causal_conv1d_update_silu_cutile only supports seq_len=1"
    assert B == 1, "causal_conv1d_update_silu_cutile only supports B=1 currently"
    assert conv_state.shape == (B, D, 4), f"expected conv_state shape {(B, D, 4)}, got {conv_state.shape}"
    assert weight.shape == (D, 4), f"expected weight shape {(D, 4)}, got {weight.shape}"

    x = hidden_states.squeeze(0).squeeze(-1).contiguous()
    w = weight.contiguous()
    output = torch.empty(D, dtype=hidden_states.dtype, device=hidden_states.device)
    cs = conv_state.squeeze(0)

    BLOCK_D = 256
    grid = ((D + BLOCK_D - 1) // BLOCK_D,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _causal_conv1d_update_silu_kernel,
        (x, cs, w, output, BLOCK_D),
    )
    return output.unsqueeze(0).unsqueeze(-1)
