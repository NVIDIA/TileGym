# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
TileGym replacements for Qwen3.5 model components.

Qwen3.5 is a hybrid model with both standard (full) attention layers and
gated delta rule linear attention layers.  This module provides:

- Qwen3_5MLPTileGym       – SwiGLU MLP accelerated with TileGym silu_and_mul
- get_fmha_qwen3_5_interface – FMHA wrapper that fixes decode-path output layout
- sigmoid_mul_cutile       – Fused sigmoid(gate) * x for attention output gating
- gdr_preprocess_cutile    – Fused gate preprocessing: sigmoid(b), -exp(A)*softplus(a+dt)
- rms_norm_gated_cutile    – Fused RMSNorm with SiLU gating
- causal_conv1d_update_silu_cutile – Fused depthwise conv1d update + SiLU (decode path)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from tilegym.transformers.qwen3_5.kernels.causal_conv1d_prefill_silu import causal_conv1d_prefill_silu_cutile
from tilegym.transformers.qwen3_5.kernels.causal_conv1d_update_silu import causal_conv1d_update_silu_cutile
from tilegym.transformers.qwen3_5.kernels.gdr_preprocess import gdr_preprocess_cutile
from tilegym.transformers.qwen3_5.kernels.residual_add_rms_norm import residual_add_rms_norm_cutile
from tilegym.transformers.qwen3_5.kernels.rms_norm_gated import rms_norm_gated_cutile
from tilegym.transformers.qwen3_5.kernels.sigmoid_mul import sigmoid_mul_cutile
from tilegym.transformers.qwen3_5.kernels.silu_and_mul_separate import silu_and_mul_separate_cutile

# ──────────────────────────────────────────────────────────────────────
# Replacement Qwen3_5RMSNormGated using cuTile
# ──────────────────────────────────────────────────────────────────────


class Qwen3_5RMSNormGatedTileGym(nn.Module):
    """Drop-in cuTile replacement for Qwen3_5RMSNormGated."""

    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        return rms_norm_gated_cutile(hidden_states, gate, self.weight, self.variance_epsilon)


# ──────────────────────────────────────────────────────────────────────
# Patched forward for Qwen3_5GatedDeltaNet: uses fused preprocessing
# ──────────────────────────────────────────────────────────────────────


def _gated_delta_net_forward_tilegym(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
    """Patched forward for Qwen3_5GatedDeltaNet with fused cuTile preprocessing."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states and seq_len == 1:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if use_precomputed_states:
            mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
        if cache_params is not None:
            new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.update_conv_state(new_conv_state, self.layer_idx)
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            # Fused cuTile causal conv1d + SiLU for prefill
            padded = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
            mixed_qkv = causal_conv1d_prefill_silu_cutile(
                padded,
                self.conv1d.weight.squeeze(1),
                mixed_qkv.shape[-1],
            )
        if use_precomputed_states:
            mixed_qkv = mixed_qkv[:, :, -seq_len:]

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    # Fused gate preprocessing (cuTile)
    beta, g = gdr_preprocess_cutile(b, a, self.A_log, self.dt_bias)

    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if use_precomputed_states and seq_len == 1:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state if use_precomputed_states else None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)
    return output


# ──────────────────────────────────────────────────────────────────────
# Patched forward for Qwen3_5Attention: fused sigmoid gate
# ──────────────────────────────────────────────────────────────────────


def _attention_forward_tilegym(
    self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None, **kwargs
):
    """Patched Qwen3_5Attention.forward with fused sigmoid_mul gate."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    from transformers.models.qwen3_5.modeling_qwen3_5 import eager_attention_forward

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
        **kwargs,
    )

    # For decode, FMHA returns (B, 1, H*D) already flat; for prefill, (B, S, H, D) needs reshape
    if attn_output.dim() != 3:
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # Fused sigmoid_mul (cuTile) instead of attn_output * torch.sigmoid(gate)
    attn_output = sigmoid_mul_cutile(attn_output, gate)

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ──────────────────────────────────────────────────────────────────────
# SwiGLU MLP
# ──────────────────────────────────────────────────────────────────────


class Qwen3_5MLPTileGym(nn.Module):
    """
    TileGym-aware Qwen3.5 MLP replacement.

    Matches Qwen3_5MLP(config, intermediate_size) constructor signature to
    preserve checkpoint compatibility, while accelerating SiLU+mul with
    TileGym kernels.
    """

    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if self.config.hidden_act in ("silu", "swish"):
            hidden_states = silu_and_mul_separate_cutile(gate, up)
        else:
            hidden_states = self.act_fn(gate) * up
        return self.down_proj(hidden_states)


# ──────────────────────────────────────────────────────────────────────
# FMHA interface
# ──────────────────────────────────────────────────────────────────────
#
# Wraps the TileGym FMHA op for Qwen3.5:
#   - Transpose the decode-path output to (B, S, H, D) as HF expects.


def get_fmha_qwen3_5_interface(backend=None, kernel_configs=None):
    """Return an FMHA interface suitable for Qwen3.5 attention layers."""
    from tilegym.backend import get_current_backend
    from tilegym.ops import fmha
    from tilegym.ops import fmha_decode

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
    ):
        del attention_mask, dropout
        if scaling is None:
            scaling = 1.0 / math.sqrt(q.size(-1))

        if q.size(-2) == 1:
            # Decode path — return (B, 1, H*D) directly, avoiding transpose+contiguous copy
            o = fmha_decode(q, k, v, sm_scale=scaling)
            return o.view(o.size(0), 1, -1), None

        # Prefill path
        configs = dict(kernel_configs) if kernel_configs else {}
        is_causal = True if is_causal is None else is_causal
        has_backward = False if has_backward is None else has_backward
        use_backend = backend if backend is not None else get_current_backend()
        o = fmha(
            q,
            k,
            v,
            scaling=scaling,
            is_causal=is_causal,
            has_backward=has_backward,
            kernel_configs=configs,
            backend=use_backend,
        )
        return o.transpose(1, 2).contiguous(), None

    return fmha_interface_wrapper


# ──────────────────────────────────────────────────────────────────────
# Patched forward for Qwen3_5DecoderLayer: fused residual add + RMSNorm
# ──────────────────────────────────────────────────────────────────────


def _decoder_layer_forward_tilegym(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Patched Qwen3_5DecoderLayer.forward with fused residual add + RMSNorm."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
        )
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    # Fused residual add + RMSNorm (cuTile)
    norm_mod = self.post_attention_layernorm
    norm_eps = getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-6))
    norm_offset = getattr(norm_mod, "offset", 1.0)
    hidden_states, normed = residual_add_rms_norm_cutile(
        residual, hidden_states, norm_mod.weight, norm_eps, offset=norm_offset
    )

    # MLP
    residual = hidden_states
    hidden_states = self.mlp(normed)
    hidden_states = residual + hidden_states

    return hidden_states
