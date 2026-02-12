# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn

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


######################################################################
####Multi-head softcap & window attention interface(used in Gemma3)###
######################################################################


def gemma3_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def get_fmha_gemma3_interface(backend=None, kernel_configs=None):
    """
    Factory function that returns a configured FMHA interface for Gemma3.

    Gemma3 uses special attention features:
    - Soft cap (attn_logit_softcapping): limits attention logits range
    - Sliding window: local attention within a window
    - Mixed layer types: some layers use global, some use sliding window

    Args:
        backend: Backend to use (triton, cutile, torch)
        kernel_configs: Kernel configuration parameters

    Returns:
        Callable interface compatible with transformers' ALL_ATTENTION_FUNCTIONS
    """

    def fmha_gemma3_interface_wrapper(
        module: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        softcap: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Gemma3-specific FMHA implementation with soft cap and sliding window support.

        Args:
            module: The attention module (contains config)
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            attention_mask: Attention mask (usually None for causal)
            dropout: Dropout probability
            scaling: Attention score scaling factor
            sliding_window: Sliding window size (None for global attention)
            softcap: Soft cap value for attention logits (takes priority over module config)
            **kwargs: Additional arguments

        Returns:
            Tuple of (output, attention_weights)
            - output: [batch, seq_len, num_heads * head_dim]
            - attention_weights: None (not computed for efficiency)
        """
        # Get soft cap: prioritize parameter, then module attribute, then module.config
        soft_cap = softcap  # Use parameter if provided
        if soft_cap is None:
            if hasattr(module, "attn_logit_softcapping"):
                soft_cap = module.attn_logit_softcapping
            elif hasattr(module, "config") and hasattr(module.config, "attn_logit_softcapping"):
                soft_cap = module.config.attn_logit_softcapping

        # Set default scaling
        if scaling is None:
            scaling = 1.0 / math.sqrt(q.size(-1))

        # Determine if causal based on mask or config
        is_causal = True
        if hasattr(module, "is_causal"):
            is_causal = module.is_causal

        # Get sequence lengths
        seq_len_q = q.size(2)

        # Use specialized decode kernel when seq_len_q == 1 (decode phase)
        # This kernel is optimized for single query with long KV cache
        if seq_len_q == 1:
            # return gemma3_eager_attention_forward(module, q, k, v, attention_mask, dropout, scaling, softcap, **kwargs)
            from tilegym.ops import gemma_attention_decode

            o = gemma_attention_decode(
                q,
                k,
                v,
                scaling=scaling,
                window_size=sliding_window if sliding_window else 0,
                soft_cap=soft_cap,
                **kwargs,
            )
        else:
            # Use gemma_attention for prefill (seq_len_q > 1)
            from tilegym.ops import gemma_attention

            o = gemma_attention(
                q,
                k,
                v,
                scaling=scaling,
                is_causal=is_causal,
                window_size=sliding_window if sliding_window else 0,
                soft_cap=soft_cap,
                **kwargs,
            )

        kernel_output = o.transpose(1, 2).contiguous()

        # Return in format expected by Gemma3: (output, attn_weights)
        return kernel_output, None

    return fmha_gemma3_interface_wrapper
