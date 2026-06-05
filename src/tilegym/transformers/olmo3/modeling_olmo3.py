# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""OLMo-3-specific cuTile kernel wrappers and patched decoder layer forward."""

import torch
import torch.nn as nn

from tilegym.transformers.olmo3.kernels.dual_rms_norm import dual_rms_norm_cutile
from tilegym.transformers.olmo3.kernels.rms_norm_residual_add import rms_norm_residual_add_cutile


class FusedOlmo3MLP(nn.Module):
    """Fully fused SwiGLU MLP using linear_gluact_linear (single kernel).

    Replaces PartiallyFusedSwiGLUMLP's 3-kernel pattern (matmul + silu_and_mul + matmul)
    with a single fused kernel: silu(x @ W_gate^T) * (x @ W_up^T) @ W_down^T.
    """

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        from tilegym.ops import linear_gluact_linear

        return linear_gluact_linear(
            input=x,
            weight_act=self.gate_proj.weight,
            weight_noact=self.up_proj.weight,
            weight2=self.down_proj.weight,
            act_type="silu",
        )


def _attention_forward_tilegym(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Patched Olmo3Attention.forward with fused dual Q/K RMSNorm."""
    from transformers.models.olmo3.modeling_olmo3 import ALL_ATTENTION_FUNCTIONS
    from transformers.models.olmo3.modeling_olmo3 import apply_rotary_pos_emb
    from transformers.models.olmo3.modeling_olmo3 import eager_attention_forward

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Fused dual Q/K RMSNorm (single kernel instead of two)
    q_norm_eps = getattr(self.q_norm, "variance_epsilon", getattr(self.q_norm, "eps", 1e-6))
    query_states, key_states = dual_rms_norm_cutile(
        query_states,
        key_states,
        self.q_norm.weight,
        self.k_norm.weight,
        q_norm_eps,
    )

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = value_states.view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _decoder_layer_forward_tilegym(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=None,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
) -> torch.Tensor:
    """Patched Olmo3DecoderLayer.forward with fused RMSNorm + residual add."""
    from transformers.models.olmo3.modeling_olmo3 import apply_rotary_pos_emb

    # ---- Self-attention ----
    residual = hidden_states
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )

    # Fused: post_attention_layernorm(hidden_states) + residual
    norm = self.post_attention_layernorm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    hidden_states = rms_norm_residual_add_cutile(hidden_states, residual, norm.weight, eps)

    # ---- MLP ----
    residual = hidden_states
    hidden_states = self.mlp(hidden_states)

    # Fused: post_feedforward_layernorm(hidden_states) + residual
    norm = self.post_feedforward_layernorm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    hidden_states = rms_norm_residual_add_cutile(hidden_states, residual, norm.weight, eps)

    return hidden_states
