# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""LFM2-MoE depthwise causal conv1d decode-update cuTile kernel.

Replacement for the module-level ``causal_conv1d_update`` called by
``Lfm2MoeShortConv.forward`` on the single-token cached-decode path.

LFM2 stores a full ``K``-wide (``conv_L_cache = 3``) rolling window in the
conv-state cache. For a new input ``x`` and state ``[s0, s1, s2]`` the update
rolls the window (dropping the oldest ``s0``) and computes the conv over the
new window ``[s1, s2, x]``:

    out = s1 * w0 + s2 * w1 + x * w2

and writes the rolled window back into ``conv_state`` **in place**. No
activation is fused (LFM2 gates externally). This mirrors the pure-torch
``Lfm2MoeShortConv.slow_forward`` decode branch
(``sum(update_conv_state(x) * weight, dim=-1)``).
"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _causal_conv1d_update_kernel(
    x,  # (D,)
    conv_state,  # (D, K=3), updated in place
    weight,  # (D, K=3)
    output,  # (D,)
    BLOCK_D: ConstInt,
):
    bid = ct.bid(0)
    d_start = bid * BLOCK_D
    offs = ct.arange(BLOCK_D, dtype=ct.int32)
    d_idx = d_start + offs

    s0 = ct.astype(ct.gather(conv_state, (d_idx, 0), check_bounds=True), ct.float32)
    s1 = ct.astype(ct.gather(conv_state, (d_idx, 1), check_bounds=True), ct.float32)
    s2 = ct.astype(ct.gather(conv_state, (d_idx, 2), check_bounds=True), ct.float32)
    xv = ct.astype(ct.gather(x, (d_idx,), check_bounds=True), ct.float32)

    w0 = ct.astype(ct.gather(weight, (d_idx, 0), check_bounds=True), ct.float32)
    w1 = ct.astype(ct.gather(weight, (d_idx, 1), check_bounds=True), ct.float32)
    w2 = ct.astype(ct.gather(weight, (d_idx, 2), check_bounds=True), ct.float32)

    # Roll the window (drop s0) then convolve over [s1, s2, x].
    result = s1 * w0 + s2 * w1 + xv * w2

    ct.scatter(output, (d_idx,), ct.astype(result, output.dtype), check_bounds=True)
    # Shift state in place: [s0, s1, s2] -> [s1, s2, x]
    ct.scatter(conv_state, (d_idx, 0), ct.astype(s1, conv_state.dtype), check_bounds=True)
    ct.scatter(conv_state, (d_idx, 1), ct.astype(s2, conv_state.dtype), check_bounds=True)
    ct.scatter(conv_state, (d_idx, 2), ct.astype(xv, conv_state.dtype), check_bounds=True)


def lfm2_causal_conv1d_update_cutile(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation=None,
) -> torch.Tensor:
    """Depthwise causal conv1d decode-update (drop-in for ``causal_conv1d_update``).

    Matches the call contract of ``Lfm2MoeShortConv.forward``'s single-token
    cached-decode branch, which passes the *3D* ``hidden_states`` tensor and
    multiplies the result with the 3D gate ``C`` (``y = C * hidden_states``).

    Args:
        x: ``(B=1, D, L=1)`` current-timestep input (``B * x`` after the
            in-proj chunk, still 3D at the call site).
        conv_state: ``(B=1, D, K=3)`` rolling window cache, updated **in place**.
        weight: ``(D, K=3)`` depthwise conv weights.
        bias: optional ``(D,)`` bias (LFM2-8B-A1B uses ``conv_bias=False`` -> None).
        activation: accepted for signature compatibility; must be ``None``.

    Returns:
        ``(B=1, D, 1)`` conv output for the current timestep (same rank as the
        input, like the HF reference implementation).
    """
    assert activation is None, "LFM2 short-conv fuses no activation; activation must be None"

    B, D, L = x.shape
    assert B == 1, "lfm2_causal_conv1d_update_cutile only supports B=1"
    assert L == 1, f"lfm2_causal_conv1d_update_cutile is the single-token decode path, got L={L}"
    K = weight.shape[1]
    assert K == 3, f"expected kernel_size 3, got {K}"

    x_1d = x.reshape(D).contiguous()  # (D,)
    cs = conv_state.squeeze(0)  # (D, 3) view -> mutated in place
    w = weight.contiguous()
    output = torch.empty(D, dtype=x.dtype, device=x.device)

    BLOCK_D = 256
    grid = ((D + BLOCK_D - 1) // BLOCK_D,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _causal_conv1d_update_kernel,
        (x_1d, cs, w, output, BLOCK_D),
    )

    out = output.view(1, D, 1)  # (1, D, 1), matching the HF fallback's output rank
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out
