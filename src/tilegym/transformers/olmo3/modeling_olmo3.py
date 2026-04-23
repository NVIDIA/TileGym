# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""OLMo-3-specific cuTile kernel wrappers and patched decoder layer forward."""

import cuda.tile as ct
import torch
import torch.nn as nn

ConstInt = ct.Constant[int]


@ct.kernel
def _dual_rms_norm_kernel(
    q,  # (N, D) — projected Q, normalized in-place
    k,  # (N, D) — projected K, normalized in-place
    q_weight,  # (D,)
    k_weight,  # (D,)
    eps: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    """Fused in-place RMSNorm for Q and K in a single kernel launch.

    Grid: (N,). Each block normalizes both q[bid] and k[bid] in-place.
    No race conditions since each block owns a unique row pair.
    """
    PAD = ct.PaddingMode.ZERO
    bid = ct.bid(0)

    # ---- Normalize Q row ----
    q_h = ct.load(q, index=(bid, 0), shape=(1, TILE_D), padding_mode=PAD).reshape((TILE_D,)).astype(ct.float32)
    q_w = ct.load(q_weight, index=(0,), shape=(TILE_D,), padding_mode=PAD).astype(ct.float32)
    q_var = ct.sum(q_h * q_h) * ct.truediv(1.0, D)
    q_normed = q_h * ct.rsqrt(q_var + eps) * q_w
    ct.store(q, index=(bid, 0), tile=q_normed.reshape((1, TILE_D)).astype(q.dtype))

    # ---- Normalize K row ----
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
    D = q.shape[-1]
    q_flat = q.contiguous().view(-1, D)
    k_flat = k.contiguous().view(-1, D)
    N = q_flat.shape[0]
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _dual_rms_norm_kernel,
        (q_flat, k_flat, q_weight, k_weight, eps, D, TILE_D),
    )
    return q, k


@ct.kernel
def _rms_norm_residual_add_kernel(
    x,  # (N, D) — branch output (attn or mlp)
    residual,  # (N, D) — residual from before the branch
    weight,  # (D,) — RMSNorm weight
    out,  # (N, D) — residual + rms_norm(x)
    eps: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    """Fused RMSNorm + residual add for OLMo-3 post-normalization.

    Computes: out = residual + weight * x * rsqrt(mean(x^2) + eps)

    OLMo-3 uses post-norm (norm on branch output, then add residual),
    so this fuses what would be two separate ops into one kernel.
    """
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    # Load in float32 for numerical stability
    h = ct.astype(ct.gather(x, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    r = ct.astype(ct.gather(residual, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (offs,), padding_value=0.0, check_bounds=True), ct.float32)

    # RMSNorm: weight * x * rsqrt(mean(x^2) + eps)
    variance = ct.sum(h * h) * ct.truediv(1.0, D)
    normed = h * ct.rsqrt(variance + eps) * w

    # Residual add
    result = r + normed

    ct.scatter(out, (bid, offs), ct.astype(result, out.dtype), check_bounds=True)


def rms_norm_residual_add_cutile(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm + residual add. Returns residual + rms_norm(x)."""
    D = x.shape[-1]
    x_flat = x.contiguous().view(-1, D)
    r_flat = residual.contiguous().view(-1, D)
    N = x_flat.shape[0]
    out = torch.empty_like(x_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _rms_norm_residual_add_kernel,
        (x_flat, r_flat, weight, out, eps, D, TILE_D),
    )
    return out.view(x.shape)


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
