# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""OLMo3 fused in-place Q/K RMSNorm cuTile kernel."""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]


@ct.kernel
def _dual_rms_norm_kernel(
    q,  # (N, D)
    k,  # (N, D)
    q_weight,  # (D,)
    k_weight,  # (D,)
    eps: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    PAD = ct.PaddingMode.ZERO
    bid = ct.bid(0)

    q_h = ct.load(q, index=(bid, 0), shape=(1, TILE_D), padding_mode=PAD).reshape((TILE_D,)).astype(ct.float32)
    q_w = ct.load(q_weight, index=(0,), shape=(TILE_D,), padding_mode=PAD).astype(ct.float32)
    q_var = ct.sum(q_h * q_h) * ct.truediv(1.0, D)
    q_normed = q_h * ct.rsqrt(q_var + eps) * q_w
    ct.store(q, index=(bid, 0), tile=q_normed.reshape((1, TILE_D)).astype(q.dtype))

    k_h = ct.load(k, index=(bid, 0), shape=(1, TILE_D), padding_mode=PAD).reshape((TILE_D,)).astype(ct.float32)
    k_w = ct.load(k_weight, index=(0,), shape=(TILE_D,), padding_mode=PAD).astype(ct.float32)
    k_var = ct.sum(k_h * k_h) * ct.truediv(1.0, D)
    k_normed = k_h * ct.rsqrt(k_var + eps) * k_w
    ct.store(k, index=(bid, 0), tile=k_normed.reshape((1, TILE_D)).astype(k.dtype))


def dual_rms_norm_cutile(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused in-place RMSNorm for Q and K in a single kernel launch."""
    if q.shape != k.shape:
        raise ValueError(f"q and k must have identical shapes, got {q.shape} vs {k.shape}")
    D = q.shape[-1]
    if q_weight.numel() != D:
        raise ValueError(f"q_weight must have length {D}, got {q_weight.numel()}")
    if k_weight.numel() != D:
        raise ValueError(f"k_weight must have length {D}, got {k_weight.numel()}")
    q = q.contiguous()
    k = k.contiguous()
    q_flat = q.view(-1, D)
    k_flat = k.view(-1, D)
    N = q_flat.shape[0]
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _dual_rms_norm_kernel,
        (q_flat, k_flat, q_weight, k_weight, eps, D, TILE_D),
    )
    return q, k
