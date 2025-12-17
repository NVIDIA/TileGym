# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from typing import Any
from typing import Dict
from typing import Optional

import torch

from tilegym.backend import *

######################################################################
################Multi-head attention interface################
######################################################################


def repeat_kv(tensor: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Repeat KV heads for grouped query attention"""
    batch_size, num_kv_heads, seq_len, head_dim = tensor.shape
    tensor = tensor.unsqueeze(2).expand(batch_size, num_kv_heads, num_groups, seq_len, head_dim)
    return tensor.reshape(batch_size, num_kv_heads * num_groups, seq_len, head_dim)


def fmha_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = True,
    scaling: float = None,
    backend: str = None,
    has_backward: bool = False,
    kernel_configs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Unified interface for Flash Multi-Head Attention (FMHA) operations.

    This is a high-level wrapper around tilegym.ops.fmha dispatch system.

    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        is_causal: Whether to apply causal masking
        scaling: Scaling factor for attention scores
        backend: Backend to use (cutile, torch)
        has_backward: Whether backward pass is needed
        kernel_configs: Kernel configuration parameters
        **kwargs: Additional arguments for specific backends

    Returns:
        Output tensor
    """
    # Use the unified dispatch system
    from tilegym.ops import fmha

    return fmha(
        q,
        k,
        v,
        scaling=scaling,
        is_causal=is_causal,
        has_backward=has_backward,
        kernel_configs=kernel_configs,
        backend=backend,
    )


def get_fmha_interface(backend=None, kernel_configs=None):
    """
    Factory function that returns a configured FMHA interface.

    Args:
        backend: Backend to use (cutile, torch)
        kernel_configs: Kernel configuration parameters
    """

    def fmha_interface_wrapper(
        module: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        has_backward: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Core FMHA implementation with minimal required parameters.
        """
        if scaling is None:
            scaling = 1.0 / math.sqrt(q.size(-1))

        if q.size(-2) == 1:
            from tilegym.ops import fmha_decode

            return fmha_decode(q, k, v, sm_scale=scaling), None

        # Set default values
        is_causal = True if is_causal is None else is_causal
        has_backward = False if has_backward is None else has_backward
        # Call fmha_interface with the given arguments
        o = fmha_interface(
            q,
            k,
            v,
            is_causal=is_causal,
            scaling=scaling,
            backend=backend,
            has_backward=has_backward,
            kernel_configs=kernel_configs,
            **kwargs,
        )
        return o.transpose(1, 2).contiguous(), None

    return fmha_interface_wrapper


######################################################################
################Multi-head linear attention interface################
######################################################################


def mla_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qpe: torch.Tensor,
    kpe: torch.Tensor,
    is_causal: bool,
    scaling: Optional[float] = None,
    kernel_configs: Optional[Dict[str, Any]] = None,
    backend: Optional[str] = None,
) -> torch.Tensor:
    """
    Unified multi-head linear attention interface

    This is a high-level wrapper around tilegym.ops.mla dispatch system.

    Args:
        q: Query tensor [batch, heads, seq_len, hidden_dim]
        k: Key tensor [batch, kv_heads, seq_len, hidden_dim]
        v: Value tensor [batch, kv_heads, seq_len, hidden_dim]
        qpe: Query positional embedding [batch, heads, seq_len, pe_dim]
        kpe: Key positional embedding [batch, 1, seq_len, pe_dim]
        is_causal: Whether to use causal mask
        scaling: Scaling factor, defaults to 1/sqrt(hidden_dim + pe_dim)
        kernel_configs: Kernel configuration parameters
        backend: Backend to use (cutile, torch)

    Returns:
        Output tensor [batch, heads, seq_len, hidden_dim]
    """
    from tilegym.ops import mla

    return mla(
        q,
        k,
        v,
        qpe,
        kpe,
        is_causal,
        scaling=scaling,
        kernel_configs=kernel_configs,
        backend=backend,
    )


def mla_decoding_interface(
    q: torch.Tensor,
    qpe: torch.Tensor,
    kv: torch.Tensor,
    kpe: torch.Tensor,
    sm_scale: Optional[float],
    transpose: Optional[bool],
    backend: Optional[str] = None,
) -> torch.Tensor:
    """Unified multi latent attention interface

    Returns:
        out: Output tensor
    """
    if transpose is None:
        transpose = False
    if sm_scale is None:
        sm_scale = 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))

    assert q.dim() == 3, "q's shape should be [b, q_head_num, q_nope_dim]"
    assert qpe.dim() == 3, "qpe's shape should be [b, q_head_num, q_pe_dim]"
    assert kv.dim() == 3, "kv's shape should be [b, kv_seqlen, kv_dim]"
    assert kpe.dim() == 3, "kpe's shape should be [b, kv_seqlen, kpe_dim]"

    if backend is None:
        backend = get_current_backend()
    assert_backend_available(backend)

    from tilegym.ops import mla_decoding_split_kv

    out = mla_decoding_split_kv(q, qpe, kv, kpe, sm_scale, kv_len_per_split=512)
    return out
