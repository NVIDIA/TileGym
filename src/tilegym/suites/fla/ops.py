# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Dispatch interfaces for FLA linear-attention operators."""

from typing import Optional
from typing import Union

import torch

from tilegym.backend import dispatch
from tilegym.backend import get_current_backend


@dispatch(
    "fla.linear_attention_chunk",
)
def linear_attention_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[Union[torch.Tensor, tuple]] = None,
    output_final_state: bool = False,
    normalize: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """Chunked linear attention. q/k/v: ``[B, T, H, K/K/V]``. Returns ``(o, final_state)``."""
    raise NotImplementedError(f"linear_attention_chunk not implemented for {get_current_backend()}")


@dispatch(
    "fla.linear_attention_fused_chunk",
)
def linear_attention_fused_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[Union[torch.Tensor, tuple]] = None,
    output_final_state: bool = False,
    normalize: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    """Fused-chunk linear attention. q/k/v: ``[B, T, H, K/K/V]``. Returns ``(o, final_state)``."""
    raise NotImplementedError(f"linear_attention_fused_chunk not implemented for {get_current_backend()}")


# --- Mamba-2 (SSD): simple-GLA with per-token scalar log-decay gate g [B, T, H] ---


@dispatch(
    "fla.linear_attention_mamba2_fused_chunk",
)
def linear_attention_mamba2_fused_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Fused-chunk Mamba-2 (SSD). g = log(alpha) ``[B, T, H]``. Training does not differentiate g. Returns ``(o, final_state)``."""
    raise NotImplementedError(f"linear_attention_mamba2_fused_chunk not implemented for {get_current_backend()}")


@dispatch(
    "fla.linear_attention_mamba2_fused_recurrent",
)
def linear_attention_mamba2_fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    reverse: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Fused-recurrent Mamba-2 (SSD). g = log(alpha) ``[B, T, H]``.

    The cuTile dense BF16 chunk route does not support gradients for g. Returns ``(o, final_state)``.
    """
    raise NotImplementedError(f"linear_attention_mamba2_fused_recurrent not implemented for {get_current_backend()}")


# --- DeltaNet (beta-gated delta rule): S_t = (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T ---


@dispatch(
    "fla.linear_attention_deltanet_fused_recurrent",
)
def linear_attention_deltanet_fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Fused-recurrent DeltaNet. q/k/v: ``[B, T, H, K/K/V]``, beta: ``[B, T, H]``. Returns ``(o, final_state)``."""
    raise NotImplementedError(f"linear_attention_deltanet_fused_recurrent not implemented for {get_current_backend()}")


# --- Gated DeltaNet (GDN): S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T, alpha_t = exp(g_t) ---


@dispatch(
    "fla.linear_attention_gdn_chunk",
)
def linear_attention_gdn_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Chunked Gated DeltaNet. q/k/v: ``[B, T, H, K/K/V]``, g=log(alpha)/beta: ``[B, T, H]`` (bf16). Returns ``(o, final_state)``."""
    raise NotImplementedError(f"linear_attention_gdn_chunk not implemented for {get_current_backend()}")


@dispatch(
    "fla.linear_attention_gdn_fused_recurrent",
)
def linear_attention_gdn_fused_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Fused-recurrent Gated DeltaNet. q/k/v: ``[B, T, H, K/K/V]``, g=log(alpha)/beta: ``[B, T, H]``. Returns ``(o, final_state)``."""
    raise NotImplementedError(f"linear_attention_gdn_fused_recurrent not implemented for {get_current_backend()}")
