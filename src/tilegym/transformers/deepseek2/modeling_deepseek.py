# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import warnings
from typing import Optional

import torch
import transformers
from packaging import version
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MLP
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MoEGate

REQUIRED_TRANSFORMERS_VERSION = "4.55.2"
current_version = transformers.__version__

if version.parse(current_version) < version.parse(REQUIRED_TRANSFORMERS_VERSION):
    raise ImportError("In new transformers version, past_key_value is named to past_key_values")

from tilegym.logger import get_logger
from tilegym.ops import fused_moe
from tilegym.ops import get_fused_swiglu_module
from tilegym.ops import group_gemm
from tilegym.ops import mla_interface
from tilegym.ops.attn_interface import mla_decoding_interface

logger = get_logger(__name__)


# in transformers latest version, the past_key_value is named to past_key_values
def tilegym_deepseek_v2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(query_shape).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_head_num = q.shape[1]

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
    # Dynamically import to get the current (possibly monkey-patched) implementation
    from transformers.models.deepseek_v2.modeling_deepseek_v2 import apply_rotary_emb

    q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, position_embeddings.to(q_pe.device))

    if past_key_values is not None:
        cache_kwargs = {"cache_position": cache_position}
        k_nope = self.kv_a_layernorm(k_nope)
        k_nope, k_pe = past_key_values.update(k_nope, k_pe, self.layer_idx, cache_kwargs)

    # prefill, normal implementation(mha)
    if seq_length != 1:
        if k_nope.shape[1] != seq_length:
            print(f"k_nope.shape: {k_nope.shape}, seq_length: {seq_length}")
            raise NotImplementedError("don't support chunk prefill now")

        attn_output, attn_weights = _forward_normal(
            self,
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            batch_size,
            q_head_num,
            attention_mask,
            **kwargs,
        )
    # decode, absorb implementation(mla)
    else:
        attn_output, attn_weights = _forward_absorb(
            self,
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            batch_size,
            q_head_num,
            attention_mask,
            **kwargs,
        )

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _forward_normal(
    self,
    q_nope,
    q_pe,
    k_nope,
    k_pe,
    batch_size,
    q_head_num,
    attention_mask,
    **kwargs,
):
    key_shape = (
        batch_size,
        k_nope.shape[1],
        -1,
        self.qk_nope_head_dim + self.v_head_dim,
    )
    k_nope = self.kv_b_proj(k_nope).view(key_shape).transpose(1, 2)
    k_nope, value_states = torch.split(k_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_pe = k_pe.expand(*k_nope.shape[:-1], -1)

    is_causal = q_nope.shape[2] > 1 and attention_mask is None and getattr(self, "is_causal", True)
    attn_output = (
        mla_interface(
            q_nope,
            k_nope,
            value_states,
            q_pe,
            k_pe,
            is_causal=is_causal,
            scaling=self.scaling,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_weights = None
    return attn_output, attn_weights


def _forward_absorb(
    self,
    q_nope,
    q_pe,
    k_nope,
    k_pe,
    batch_size,
    q_head_num,
    attention_mask,
    **kwargs,
):
    if not hasattr(self, "k_up_proj"):
        wkv_b = self.kv_b_proj.weight
        wkv_b = wkv_b.view(q_head_num, -1, self.kv_lora_rank)
        self.k_up_proj = wkv_b[:, : self.qk_nope_head_dim].contiguous()
        self.v_up_proj = wkv_b[:, -self.v_head_dim :].contiguous()

    # Use group_gemm to replace einsum operations - eliminates aten::matmul calls
    # Verified to work perfectly with bfloat16 (the model's dtype)

    batch_size_mla, num_heads_mla, seq_len_mla, qk_nope_dim = q_nope.shape

    # First group GEMM: q_nope @ k_up_proj per head
    # q_nope: (batch, heads, seq, qk_nope_head_dim)
    # k_up_proj: (heads, qk_nope_head_dim, kv_lora_rank)
    # Result: (batch, heads, seq, kv_lora_rank)

    if batch_size_mla == 1 and seq_len_mla == 1:
        # Fast path for decode (single token) - use group_gemm to eliminate aten::matmul
        group_A_list = [q_nope[0, h, 0, :].reshape(1, -1) for h in range(num_heads_mla)]
        group_B_list = [self.k_up_proj[h, :, :] for h in range(num_heads_mla)]
        group_C_list = group_gemm(group_A_list, group_B_list, transpose_b=False)
        q_nope = torch.stack([c.reshape(seq_len_mla, -1) for c in group_C_list], dim=0).unsqueeze(0)
    else:
        # Use group_gemm for prefill as well
        # einsum: "bhsd,hdc->bhsc" means for each (b, h): q_nope[b,h,:,:] @ k_up_proj[h,:,:]
        # group_gemm supports strided tensors via stride(0), so no .contiguous() needed
        group_A_list = []
        group_B_list = []
        for b in range(batch_size_mla):
            for h in range(num_heads_mla):
                # q_nope[b, h, :, :] has shape (seq_len, qk_nope_head_dim)
                group_A_list.append(q_nope[b, h, :, :])
                # k_up_proj[h, :, :] has shape (qk_nope_head_dim, kv_lora_rank)
                group_B_list.append(self.k_up_proj[h, :, :])

        group_C_list = group_gemm(group_A_list, group_B_list, transpose_b=False)

        # Reshape results back to (batch, heads, seq, kv_lora_rank)
        q_nope_list = []
        idx = 0
        for b in range(batch_size_mla):
            heads_list = []
            for h in range(num_heads_mla):
                heads_list.append(group_C_list[idx])
                idx += 1
            q_nope_list.append(torch.stack(heads_list, dim=0))
        q_nope = torch.stack(q_nope_list, dim=0)

    attn_output = mla_decoding_interface(
        q_nope.squeeze(2),
        q_pe.squeeze(2),
        k_nope,
        k_pe.squeeze(1),
        self.scaling,
        transpose=True,
    )
    attn_weights = None

    # Second group GEMM: attn_output @ v_up_proj per head
    # attn_output: (batch, heads, kv_lora_rank) after unsqueeze: (batch, heads, 1, kv_lora_rank)
    # v_up_proj: (heads, v_head_dim, kv_lora_rank)
    # Result: (batch, heads, 1, v_head_dim)

    attn_output_expanded = attn_output.unsqueeze(2)
    batch_size_attn = attn_output_expanded.shape[0]
    seq_len_attn = attn_output_expanded.shape[2]

    if batch_size_attn == 1 and seq_len_attn == 1:
        # Fast path for decode - use group_gemm to eliminate aten::matmul
        group_A_list = [attn_output_expanded[0, h, 0, :].reshape(1, -1) for h in range(num_heads_mla)]
        # v_up_proj is (heads, v_head_dim, kv_lora_rank), use transpose_b=True to avoid non-contiguous tensors
        group_B_list = [self.v_up_proj[h, :, :] for h in range(num_heads_mla)]
        group_C_list = group_gemm(group_A_list, group_B_list, transpose_b=True)
        attn_output = (
            torch.stack([c.reshape(seq_len_attn, -1) for c in group_C_list], dim=0).unsqueeze(0).transpose(1, 2)
        )
    else:
        # Use group_gemm for prefill as well
        # einsum: "bhsc,hdc->bhsd" means for each (b, h): attn_output_expanded[b,h,:,:] @ v_up_proj[h,:,:].T
        # group_gemm supports strided tensors via stride(0), so no .contiguous() needed
        group_A_list = []
        group_B_list = []
        for b in range(batch_size_attn):
            for h in range(num_heads_mla):
                # attn_output_expanded[b, h, :, :] has shape (seq_len, kv_lora_rank)
                group_A_list.append(attn_output_expanded[b, h, :, :])
                # v_up_proj[h, :, :] has shape (v_head_dim, kv_lora_rank), need transpose
                group_B_list.append(self.v_up_proj[h, :, :])

        group_C_list = group_gemm(group_A_list, group_B_list, transpose_b=True)

        # Reshape results back to (batch, heads, seq, v_head_dim)
        attn_output_list = []
        idx = 0
        for b in range(batch_size_attn):
            heads_list = []
            for h in range(num_heads_mla):
                heads_list.append(group_C_list[idx])
                idx += 1
            attn_output_list.append(torch.stack(heads_list, dim=0))
        attn_output = torch.stack(attn_output_list, dim=0).transpose(1, 2)
    return attn_output, attn_weights


class DeepseekV2MoETileGym(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV2Config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        self.experts = nn.ModuleList(
            [
                (DeepseekV2MLP(config, intermediate_size=config.moe_intermediate_size))
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV2MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # Use PartiallyFusedSwiGLUMLP for shared experts to eliminate PyTorch linear operations
            FusedSwiGLUMLP = get_fused_swiglu_module()
            self.shared_experts = FusedSwiGLUMLP(config=config, intermediate_size=intermediate_size)
        self.ep_rank = 0
        self.experts_per_rank = config.n_routed_experts
        self.init = False

    def init_merged_expert_weights(self):
        if self.init:
            return
        if not hasattr(self, "experts") or len(self.experts) == 0:
            print("experts is empty, self.w1_merged and self.w2_merged will not be initialized")
            return

        w1_merged = torch.cat(
            [expert.gate_proj.weight.unsqueeze(0) for expert in self.experts],
            dim=0,
        )
        w3_merged = torch.cat(
            [expert.up_proj.weight.unsqueeze(0) for expert in self.experts],
            dim=0,
        )
        self.w2_merged = torch.cat(
            [expert.down_proj.weight.unsqueeze(0) for expert in self.experts],
            dim=0,
        )

        self.w13_merged = torch.cat([w1_merged, w3_merged], dim=1)

        self.init = True

    def moe_infer(self, x, topk_ids, topk_weight):
        out = fused_moe(
            x,
            w1=self.w13_merged,
            w2=self.w2_merged,
            topk_weights=topk_weight,
            topk_ids=topk_ids,
        )
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.init_merged_expert_weights()
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe_infer(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    @torch.no_grad()
    def moe_infer_naive(self, x, topk_ids, topk_weight):
        # reference implementation of moe_infer
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0

        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out
