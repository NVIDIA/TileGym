# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""TileGym replacement modules for `transformers.models.olmoe.modeling_olmoe`.

Only the MoE block is replaced here; RoPE / RMSNorm / attention are patched
elsewhere via the registry-level / class-level monkey-patches.

`OlmoeSparseMoeBlockTileGym` keeps the exact same nested-parameter layout as
the stock `OlmoeSparseMoeBlock` (`self.gate = OlmoeTopKRouter(...)`,
`self.experts = OlmoeExperts(...)`) so HuggingFace `state_dict` loading works
unchanged. Forward replaces the per-expert Python loop in `OlmoeExperts` with
TileGym's batched `fused_moe` kernel.

Weight-layout compatibility notes (verified against HF OLMoE 5.x source):

- HF `self.experts.gate_up_proj`: shape ``(E, 2*I, H)``. The first ``I`` rows
  along axis 1 are the **gate** projection, the second ``I`` rows are the
  **up** projection — confirmed by HF's
  ``linear(x, gate_up_proj[e]).chunk(2, dim=-1)`` which produces
  ``(gate, up)`` in that order.
- HF `self.experts.down_proj`: shape ``(E, H, I)``.
- TileGym `fused_moe(w1, w2)` expects:
    * ``w1: (E, 2*I, H)`` and assumes the standard ``silu_and_mul`` ordering
      ``silu(x[:, :I]) * x[:, I:]``, i.e. ``[gate, up]`` along the
      output-feature axis — identical to HF.
    * ``w2: (E, H, I)`` — identical to HF.
  So we can pass the HF parameters directly with **no merge / no reorder**.

Routing semantics:

- HF `OlmoeTopKRouter` does ``softmax(logits, fp32).topk(k)`` with
  ``norm_topk_prob=False`` for the real OLMoE-1B-7B-0924 checkpoint, so the
  ``top_k`` weights sum to less than 1. TileGym's MoE kernel multiplies the
  routed weights into the down-projection output regardless of whether they
  sum to 1, so the un-normalized weights flow through correctly.
"""

import torch
import torch.nn.functional as F
from torch import nn

from tilegym.ops import fused_moe
from tilegym.ops import matmul as tilegym_matmul
from tilegym.transformers.olmoe.kernels.dual_rms_norm import dual_rms_norm_olmoe_cutile
from tilegym.transformers.olmoe.kernels.residual_add_rms_norm import residual_add_rms_norm_olmoe_cutile


def _linear_cutile(self, attr_name: str, x: torch.Tensor) -> torch.Tensor:
    """Run nn.Linear's matmul through cuTile, caching the transposed weight
    once per Linear instance. Works on flattened (M, in_features) inputs;
    reshape callers handle leading dims.
    """
    proj = getattr(self, attr_name)
    cache_attr = f"_{attr_name}_weight_t"
    wt = getattr(proj, cache_attr, None)
    if wt is None:
        wt = proj.weight.t().contiguous()
        setattr(proj, cache_attr, wt)
    return tilegym_matmul(x, wt)


def _attention_forward_tilegym(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    """Patched OlmoeAttention.forward that fuses Q/K RMSNorm into one kernel
    and routes Q/K/V/O projections through cuTile matmul.

    OLMoE has no clip_qkv (config.clip_qkv = None for the public checkpoint),
    so this path also drops that conditional. Falls back to stock semantics
    for everything else: same RoPE, same KV-cache update, same attention
    interface dispatch.
    """
    from transformers.models.olmoe import modeling_olmoe

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])

    query_states = _linear_cutile(self, "q_proj", hidden_flat).view(*input_shape, -1)
    key_states = _linear_cutile(self, "k_proj", hidden_flat).view(*input_shape, -1)
    value_states = _linear_cutile(self, "v_proj", hidden_flat).view(*input_shape, -1)

    q_norm_eps = getattr(self.q_norm, "variance_epsilon", getattr(self.q_norm, "eps", 1e-5))
    query_states, key_states = dual_rms_norm_olmoe_cutile(
        query_states,
        key_states,
        self.q_norm.weight,
        self.k_norm.weight,
        q_norm_eps,
    )

    if self.config.clip_qkv is not None:
        query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
        value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

    query_states = query_states.view(*hidden_shape).transpose(1, 2)
    key_states = key_states.view(*hidden_shape).transpose(1, 2)
    value_states = value_states.view(*hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = modeling_olmoe.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = modeling_olmoe.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modeling_olmoe.eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self.config, "sliding_window", None),
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_flat = attn_output.reshape(-1, attn_output.shape[-1])
    attn_output = _linear_cutile(self, "o_proj", attn_flat).view(*input_shape, -1)
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
    """Patched OlmoeDecoderLayer.forward that fuses the post-attention residual
    add with the post-attention RMSNorm, mirroring qwen3_5's pattern.
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
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

    # Fused: hidden_states = residual + hidden_states; normed = post_attn_norm(hidden_states)
    norm_mod = self.post_attention_layernorm
    norm_eps = getattr(norm_mod, "variance_epsilon", getattr(norm_mod, "eps", 1e-5))
    hidden_states, normed = residual_add_rms_norm_olmoe_cutile(residual, hidden_states, norm_mod.weight, norm_eps)

    # MoE
    residual = hidden_states
    hidden_states = self.mlp(normed)
    hidden_states = residual + hidden_states
    return hidden_states


class OlmoeSparseMoeBlockTileGym(nn.Module):
    """Drop-in replacement for ``OlmoeSparseMoeBlock`` that routes the expert
    compute through TileGym's batched ``fused_moe`` kernel.

    The nested submodule layout (``self.gate``, ``self.experts``) is kept
    identical to the stock class so the HuggingFace state_dict loads with
    ``strict=True``.
    """

    def __init__(self, config):
        super().__init__()
        # Import here so the module import is cheap and doesn't run HF init
        # at TileGym import time.
        from transformers.models.olmoe.modeling_olmoe import OlmoeExperts
        from transformers.models.olmoe.modeling_olmoe import OlmoeTopKRouter

        self.gate = OlmoeTopKRouter(config)
        self.experts = OlmoeExperts(config)
        # Cache router metadata for convenience.
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_size = config.hidden_size

    def _route(self, hidden_flat: torch.Tensor):
        """Reproduce ``OlmoeTopKRouter.forward`` inline to avoid a redundant
        reshape and to keep all routing tensors easy to track.

        Returns ``(topk_weights, topk_indices)`` where:
        - ``topk_weights`` is cast back to ``hidden_flat.dtype``
        - ``topk_indices`` is ``torch.long`` (output of ``torch.topk``)
        """
        # Linear with the gate weight; HF stores as (num_experts, hidden_size).
        # cuTile matmul doesn't support trans_b, so we materialize the transpose.
        # gate.weight is small ((num_experts, hidden_size)) — transpose is cheap.
        if not hasattr(self, "_gate_weight_t") or self._gate_weight_t.data_ptr() == 0:
            self._gate_weight_t = self.gate.weight.t().contiguous()
        router_logits = tilegym_matmul(hidden_flat, self._gate_weight_t)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_values, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_values = topk_values / topk_values.sum(dim=-1, keepdim=True)
        topk_weights = topk_values.to(hidden_flat.dtype)
        return topk_weights, topk_indices

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.reshape(-1, hidden_dim).contiguous()

        topk_weights, topk_indices = self._route(hidden_flat)

        # TileGym's fused_moe expects (M, H) input, (E, 2I, H) w1, (E, H, I) w2.
        # ``topk_indices`` from torch.topk is int64; cast to int32 for the
        # kernel which uses 32-bit indices internally.
        out_flat = fused_moe(
            hidden_flat,
            w1=self.experts.gate_up_proj,
            w2=self.experts.down_proj,
            topk_weights=topk_weights,
            topk_ids=topk_indices.to(torch.int32),
        )

        # Match the dtype contract of the stock block.
        out_flat = out_flat.to(hidden_states.dtype)
        return out_flat.view(batch_size, sequence_length, hidden_dim)
