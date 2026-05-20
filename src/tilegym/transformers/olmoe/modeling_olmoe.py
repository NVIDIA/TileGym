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

import cuda.tile as ct
import torch
import torch.nn.functional as F
from torch import nn

from tilegym.ops import fused_moe
from tilegym.ops import matmul as tilegym_matmul

ConstInt = ct.Constant[int]


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: fused residual add + (Llama-style) RMSNorm
#
# Computes both outputs the layer body needs:
#   sum    = residual + x
#   normed = sum * rsqrt(mean(sum**2) + eps) * weight
#
# OLMoE uses pre-norm with offset=0, so the multiplier is `weight` (not
# `1 + weight` as in Gemma3/Qwen3.5). The kernel is structurally identical
# to qwen3_5's `_residual_add_rms_norm_kernel` minus the offset.
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _olmoe_residual_add_rms_norm_kernel(
    residual,  # (N, D)
    x,  # (N, D)
    weight,  # (D,)
    sum_out,  # (N, D)
    normed_out,  # (N, D)
    eps: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    r = ct.astype(ct.gather(residual, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    h = ct.astype(ct.gather(x, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (offs,), padding_value=0.0, check_bounds=True), ct.float32)

    s = r + h
    variance = ct.sum(s * s) * ct.truediv(1.0, D)
    normed = s * ct.rsqrt(variance + eps) * w

    ct.scatter(sum_out, (bid, offs), ct.astype(s, sum_out.dtype), check_bounds=True)
    ct.scatter(normed_out, (bid, offs), ct.astype(normed, normed_out.dtype), check_bounds=True)


def residual_add_rms_norm_olmoe_cutile(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
):
    """Fused residual add + Llama-style (offset=0) RMSNorm. Returns (sum, normed)."""
    D = residual.shape[-1]
    r_flat = residual.contiguous().view(-1, D)
    x_flat = x.contiguous().view(-1, D)
    N = r_flat.shape[0]
    sum_out = torch.empty_like(r_flat)
    normed_out = torch.empty_like(r_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _olmoe_residual_add_rms_norm_kernel,
        (r_flat, x_flat, weight, sum_out, normed_out, eps, D, TILE_D),
    )
    return sum_out.view(residual.shape), normed_out.view(residual.shape)


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: fused dual RMSNorm over Q and K in one launch
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _olmoe_dual_rms_norm_kernel(
    q,  # (N, D) — projected Q; normalized in-place
    k,  # (N, D) — projected K; normalized in-place
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


def dual_rms_norm_olmoe_cutile(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
):
    """Fused in-place RMSNorm on Q and K in a single kernel launch."""
    D = q.shape[-1]
    q_flat = q.contiguous().view(-1, D)
    k_flat = k.contiguous().view(-1, D)
    N = q_flat.shape[0]
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _olmoe_dual_rms_norm_kernel,
        (q_flat, k_flat, q_weight, k_weight, eps, D, TILE_D),
    )
    return q, k


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
