# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""LFM2-MoE depthwise causal conv1d prefill-path cuTile kernel.

Replacement for the module-level ``causal_conv1d_fn`` called by
``Lfm2MoeShortConv.forward``. Unlike the Qwen3.5 conv kernel this
uses ``kernel_size = conv_L_cache = 3`` and applies **no** activation — LFM2's
short conv is externally gated (``y = C * conv_out``), the SiLU is not fused in.

The stock call site passes the *unpadded* input ``Bx`` of shape ``(B, C, L)``;
the wrapper left-pads by ``K-1`` and the kernel reads the ``K``-wide causal
window, matching ``nn.Conv1d(..., padding=L_cache-1)(Bx)[..., :L]``.
"""

import cuda.tile as ct
import torch
import torch.nn.functional as F

ConstInt = ct.Constant[int]


@ct.kernel
def _causal_conv1d_prefill_kernel(
    x,  # (D, T_padded)  left-padded by K-1
    weight,  # (D, K=3)
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

    # x is left-padded, so window for output position t reads x[t], x[t+1], x[t+2].
    v0 = ct.astype(ct.gather(x, (bid_d, t_idx), padding_value=0.0, check_bounds=True), ct.float32)
    v1 = ct.astype(ct.gather(x, (bid_d, t_idx + 1), padding_value=0.0, check_bounds=True), ct.float32)
    v2 = ct.astype(ct.gather(x, (bid_d, t_idx + 2), padding_value=0.0, check_bounds=True), ct.float32)

    result = v0 * w0 + v1 * w1 + v2 * w2

    ct.scatter(output, (bid_d, t_idx), ct.astype(result, output.dtype), check_bounds=True)


def lfm2_causal_conv1d_fn_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation=None,
    seq_idx=None,
) -> torch.Tensor:
    """Depthwise causal conv1d for the prefill path (drop-in for ``causal_conv1d_fn``).

    Args:
        x: ``(B=1, D, L)`` unpadded input (``Bx`` in the LFM2 short-conv block).
        weight: ``(D, K=3)`` depthwise conv weights (``conv.weight.view(D, K)``).
        bias: optional ``(D,)`` bias (LFM2-8B-A1B uses ``conv_bias=False`` -> None).
        activation: accepted for signature compatibility; must be ``None`` (LFM2
            does not fuse an activation into the conv).
        seq_idx: accepted for signature compatibility. Packed-sequence boundaries
            are not supported by this fused path; only ``None`` is handled.

    Returns:
        ``(B=1, D, L)`` conv output.
    """
    assert activation is None, "LFM2 short-conv fuses no activation; activation must be None"
    assert seq_idx is None, "lfm2_causal_conv1d_fn_cutile does not support packed sequences (seq_idx)"

    B, D, L = x.shape
    assert B == 1, "lfm2_causal_conv1d_fn_cutile only supports B=1"
    K = weight.shape[1]
    assert K == 3, f"expected kernel_size 3, got {K}"

    x_2d = x.squeeze(0).contiguous()  # (D, L)
    x_padded = F.pad(x_2d, (K - 1, 0))  # (D, L + K - 1), left pad only
    w = weight.contiguous()
    output = torch.empty(D, L, dtype=x.dtype, device=x.device)

    BLOCK_T = 256
    grid = (D, (L + BLOCK_T - 1) // BLOCK_T)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _causal_conv1d_prefill_kernel,
        (x_padded, w, output, L, BLOCK_T),
    )

    out = output.unsqueeze(0)  # (1, D, L)
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out
