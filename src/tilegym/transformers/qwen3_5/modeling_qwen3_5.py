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

import cuda.tile as ct
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from tilegym.ops import silu_and_mul

# Type aliases for cuTile constants
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: sigmoid(gate) * x  (for full-attention output gating)
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _sigmoid_mul_kernel(
    x,  # (N, D)
    gate,  # (N, D)
    output,  # (N, D)
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    xv = ct.astype(ct.gather(x, (bid, offs), check_bounds=True), ct.float32)
    gv = ct.astype(ct.gather(gate, (bid, offs), check_bounds=True), ct.float32)

    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(gv))))
    result = ct.astype(xv * sig, output.dtype)
    ct.scatter(output, (bid, offs), result, check_bounds=True)


def sigmoid_mul_cutile(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute x * sigmoid(gate) using a fused cuTile kernel."""
    orig_shape = x.shape
    D = orig_shape[-1]
    x_flat = x.contiguous().view(-1, D)
    gate_flat = gate.contiguous().view(-1, D)
    N = x_flat.shape[0]
    output = torch.empty_like(x_flat)
    TILE_D = 1 << (D - 1).bit_length()  # next power of 2
    ct.launch(torch.cuda.current_stream(), (N,), _sigmoid_mul_kernel, (x_flat, gate_flat, output, TILE_D))
    return output.view(orig_shape)


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: fused gate preprocessing for gated delta rule
#   beta = sigmoid(b)
#   g = -exp(A_log) * softplus(a + dt_bias)
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _gdr_preprocess_kernel(
    b_in,  # (N, H)
    a_in,  # (N, H)
    A_log,  # (H,)
    dt_bias,  # (H,)
    beta_out,  # (N, H)
    g_out,  # (N, H)
    TILE_H: ConstInt,
):
    bid = ct.bid(0)  # row index in [0, N)
    offs = ct.arange(TILE_H, dtype=ct.int32)

    b = ct.astype(ct.gather(b_in, (bid, offs), check_bounds=True), ct.float32)
    a = ct.astype(ct.gather(a_in, (bid, offs), check_bounds=True), ct.float32)
    a_log = ct.astype(ct.gather(A_log, (offs,), check_bounds=True), ct.float32)
    dt_b = ct.astype(ct.gather(dt_bias, (offs,), check_bounds=True), ct.float32)

    # beta = sigmoid(b)
    beta = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(b))))

    # g = -exp(A_log) * softplus(a + dt_bias)
    # softplus(x) = log(1 + exp(x))
    sp_arg = a + dt_b
    sp = ct.log(ct.add(1.0, ct.exp(sp_arg)))
    g = ct.negative(ct.exp(a_log) * sp)

    ct.scatter(beta_out, (bid, offs), ct.astype(beta, beta_out.dtype), check_bounds=True)
    ct.scatter(g_out, (bid, offs), ct.astype(g, g_out.dtype), check_bounds=True)


def gdr_preprocess_cutile(
    b: torch.Tensor, a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused computation of beta=sigmoid(b) and g=-exp(A_log)*softplus(a+dt_bias)."""
    orig_shape = b.shape
    H = orig_shape[-1]
    b_flat = b.contiguous().view(-1, H)
    a_flat = a.contiguous().view(-1, H)
    N = b_flat.shape[0]
    beta = torch.empty_like(b_flat)
    g = torch.empty(N, H, dtype=torch.float32, device=b.device)
    TILE_H = 1 << (H - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _gdr_preprocess_kernel,
        (b_flat, a_flat, A_log, dt_bias, beta, g, TILE_H),
    )
    return beta.view(orig_shape), g.view(orig_shape)


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: fused RMSNormGated = weight * norm(x) * silu(gate)
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _rms_norm_gated_silu_kernel(
    hidden_states,  # (N, D)
    gate,  # (N, D)
    weight,  # (D,)
    output,  # (N, D)
    eps: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    h = ct.astype(ct.gather(hidden_states, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    g = ct.astype(ct.gather(gate, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (offs,), padding_value=0.0, check_bounds=True), ct.float32)

    # RMSNorm: h / sqrt(mean(h^2) + eps)
    variance = ct.sum(h * h) * ct.truediv(1.0, D)
    normed = h * ct.rsqrt(variance + eps)

    # Apply weight and SiLU gate: weight * normed * silu(gate)
    sig_g = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(g))))
    silu_g = g * sig_g
    result = ct.astype(w * normed * silu_g, output.dtype)

    ct.scatter(output, (bid, offs), result, check_bounds=True)


def rms_norm_gated_cutile(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fused RMSNorm + SiLU gating as cuTile kernel."""
    D = hidden_states.shape[-1]
    h_flat = hidden_states.contiguous().view(-1, D)
    g_flat = gate.contiguous().view(-1, D)
    N = h_flat.shape[0]
    output = torch.empty_like(h_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(
        torch.cuda.current_stream(),
        (N,),
        _rms_norm_gated_silu_kernel,
        (h_flat, g_flat, weight, output, eps, D, TILE_D),
    )
    return output.view(hidden_states.shape)


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: fused residual add + Gemma-style RMSNorm
#   Computes: sum = residual + x; normed = (1+w) * sum / sqrt(mean(sum^2) + eps)
#   Outputs both sum and normed (for residual chain + next block input)
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _residual_add_rms_norm_kernel(
    residual,  # (N, D)
    x,  # (N, D)
    weight,  # (D,)
    sum_out,  # (N, D) — residual + x
    normed_out,  # (N, D) — rms_norm(sum)
    eps: float,
    offset: float,
    D: ConstInt,
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    r = ct.astype(ct.gather(residual, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    h = ct.astype(ct.gather(x, (bid, offs), padding_value=0.0, check_bounds=True), ct.float32)
    w = ct.astype(ct.gather(weight, (offs,), padding_value=0.0, check_bounds=True), ct.float32)

    # Residual add
    s = r + h

    # Gemma-style RMSNorm: (offset + weight) * s / sqrt(mean(s^2) + eps)
    variance = ct.sum(s * s) * ct.truediv(1.0, D)
    normed = s * ct.rsqrt(variance + eps) * (offset + w)

    ct.scatter(sum_out, (bid, offs), ct.astype(s, sum_out.dtype), check_bounds=True)
    ct.scatter(normed_out, (bid, offs), ct.astype(normed, normed_out.dtype), check_bounds=True)


def residual_add_rms_norm_cutile(
    residual: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    offset: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + Gemma-style RMSNorm. Returns (sum, normed)."""
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
        _residual_add_rms_norm_kernel,
        (r_flat, x_flat, weight, sum_out, normed_out, eps, offset, D, TILE_D),
    )
    return sum_out.view(residual.shape), normed_out.view(residual.shape)


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: fused causal conv1d update + SiLU (decode path)
#   Fuses: cat + depthwise conv1d + silu + state update
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _causal_conv1d_update_silu_kernel(
    x,  # (D,) — input for current time step
    conv_state,  # (D, 3) — state buffer (kernel_size - 1)
    weight,  # (D, 4) — conv weights
    output,  # (D,) — output
    BLOCK_D: ConstInt,
):
    bid = ct.bid(0)
    d_start = bid * BLOCK_D
    offs = ct.arange(BLOCK_D, dtype=ct.int32)
    d_idx = d_start + offs

    # Load state (3 values per channel) and input
    s0 = ct.astype(ct.gather(conv_state, (d_idx, 0), check_bounds=True), ct.float32)
    s1 = ct.astype(ct.gather(conv_state, (d_idx, 1), check_bounds=True), ct.float32)
    s2 = ct.astype(ct.gather(conv_state, (d_idx, 2), check_bounds=True), ct.float32)
    xv = ct.astype(ct.gather(x, (d_idx,), check_bounds=True), ct.float32)

    # Load weights
    w0 = ct.astype(ct.gather(weight, (d_idx, 0), check_bounds=True), ct.float32)
    w1 = ct.astype(ct.gather(weight, (d_idx, 1), check_bounds=True), ct.float32)
    w2 = ct.astype(ct.gather(weight, (d_idx, 2), check_bounds=True), ct.float32)
    w3 = ct.astype(ct.gather(weight, (d_idx, 3), check_bounds=True), ct.float32)

    # Dot product: conv1d depthwise
    dot = s0 * w0 + s1 * w1 + s2 * w2 + xv * w3

    # SiLU activation
    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(dot))))
    result = dot * sig

    # Store output
    ct.scatter(output, (d_idx,), ct.astype(result, output.dtype), check_bounds=True)

    # Update conv state: shift left by 1, append new input
    ct.scatter(conv_state, (d_idx, 0), ct.astype(s1, conv_state.dtype), check_bounds=True)
    ct.scatter(conv_state, (d_idx, 1), ct.astype(s2, conv_state.dtype), check_bounds=True)
    ct.scatter(conv_state, (d_idx, 2), ct.astype(xv, conv_state.dtype), check_bounds=True)


def causal_conv1d_update_silu_cutile(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias=None,
    activation=None,
) -> torch.Tensor:
    """Fused causal conv1d update + SiLU for decode path (seq_len=1).

    Args:
        hidden_states: (B, D, 1) input
        conv_state: (B, D, kernel_size-1) state buffer, updated in-place
        weight: (D, kernel_size) conv weights
        bias: ignored (Qwen3.5 conv has no bias)
        activation: ignored (always SiLU)
    """
    B, D, seq_len = hidden_states.shape
    assert seq_len == 1, "causal_conv1d_update_silu_cutile only supports seq_len=1"
    assert B == 1, "causal_conv1d_update_silu_cutile only supports B=1 currently"

    x = hidden_states.squeeze(0).squeeze(-1).contiguous()  # (D,)
    w = weight.contiguous()  # (D, 4)
    output = torch.empty(D, dtype=hidden_states.dtype, device=hidden_states.device)
    cs = conv_state.squeeze(0)  # (D, 3)

    BLOCK_D = 256
    grid = ((D + BLOCK_D - 1) // BLOCK_D,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _causal_conv1d_update_silu_kernel,
        (x, cs, w, output, BLOCK_D),
    )
    return output.unsqueeze(0).unsqueeze(-1)  # (1, D, 1)


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: SiLU(gate) * up — takes separate inputs (no cat needed)
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _silu_and_mul_separate_kernel(
    gate,  # (N, D)
    up,  # (N, D)
    output,  # (N, D)
    TILE_D: ConstInt,
):
    bid = ct.bid(0)
    offs = ct.arange(TILE_D, dtype=ct.int32)

    g = ct.astype(ct.gather(gate, (bid, offs), check_bounds=True), ct.float32)
    u = ct.astype(ct.gather(up, (bid, offs), check_bounds=True), ct.float32)

    # SiLU(gate) * up = gate * sigmoid(gate) * up
    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(g))))
    result = ct.astype(g * sig * u, output.dtype)
    ct.scatter(output, (bid, offs), result, check_bounds=True)


def silu_and_mul_separate_cutile(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Compute silu(gate) * up without concatenation, using a fused cuTile kernel."""
    orig_shape = gate.shape
    D = orig_shape[-1]
    gate_flat = gate.contiguous().view(-1, D)
    up_flat = up.contiguous().view(-1, D)
    N = gate_flat.shape[0]
    output = torch.empty_like(gate_flat)
    TILE_D = 1 << (D - 1).bit_length()
    ct.launch(torch.cuda.current_stream(), (N,), _silu_and_mul_separate_kernel, (gate_flat, up_flat, output, TILE_D))
    return output.view(orig_shape)


# ──────────────────────────────────────────────────────────────────────
# cuTile kernel: fused depthwise conv1d + SiLU for prefill path
#   Replaces: F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
# ──────────────────────────────────────────────────────────────────────


@ct.kernel
def _causal_conv1d_prefill_silu_kernel(
    x,  # (D, T) — input channels × time
    weight,  # (D, K) — conv weights per channel, K=kernel_size
    output,  # (D, T) — output
    T: ConstInt,
    BLOCK_T: ConstInt,
):
    """Each block handles one channel across a tile of time steps."""
    bid_d = ct.bid(0)  # channel index
    bid_t = ct.bid(1)  # time tile index
    t_start = bid_t * BLOCK_T
    offs = ct.arange(BLOCK_T, dtype=ct.int32)
    t_idx = t_start + offs

    # Load 4 weights for this channel (kernel_size=4)
    w0 = ct.astype(ct.gather(weight, (bid_d, 0), check_bounds=True), ct.float32)
    w1 = ct.astype(ct.gather(weight, (bid_d, 1), check_bounds=True), ct.float32)
    w2 = ct.astype(ct.gather(weight, (bid_d, 2), check_bounds=True), ct.float32)
    w3 = ct.astype(ct.gather(weight, (bid_d, 3), check_bounds=True), ct.float32)

    # Load input values: x[d, t], x[d, t+1], x[d, t+2], x[d, t+3]
    # The conv1d with padding=3 means output[t] = sum(x[t:t+4] * w)
    v0 = ct.astype(ct.gather(x, (bid_d, t_idx), padding_value=0.0, check_bounds=True), ct.float32)
    v1 = ct.astype(ct.gather(x, (bid_d, t_idx + 1), padding_value=0.0, check_bounds=True), ct.float32)
    v2 = ct.astype(ct.gather(x, (bid_d, t_idx + 2), padding_value=0.0, check_bounds=True), ct.float32)
    v3 = ct.astype(ct.gather(x, (bid_d, t_idx + 3), padding_value=0.0, check_bounds=True), ct.float32)

    # Depthwise conv: dot product per time step
    dot = v0 * w0 + v1 * w1 + v2 * w2 + v3 * w3

    # SiLU activation
    sig = ct.truediv(1.0, ct.add(1.0, ct.exp(ct.negative(dot))))
    result = dot * sig

    ct.scatter(output, (bid_d, t_idx), ct.astype(result, output.dtype), check_bounds=True)


def causal_conv1d_prefill_silu_cutile(
    x: torch.Tensor,
    weight: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Fused causal depthwise conv1d + SiLU for prefill path.

    Args:
        x: (B, D, T_padded) input after nn.Conv1d padding
        weight: (D, kernel_size) conv weights
        seq_len: actual sequence length to slice output
    Returns:
        (B, D, seq_len) output after conv1d + SiLU
    """
    B, D, T_padded = x.shape
    assert B == 1, "causal_conv1d_prefill_silu only supports B=1"

    # The nn.Conv1d with padding=kernel_size-1=3 pads on BOTH sides.
    # For causal conv, we want output[:, :, :seq_len].
    # The padded input has T_padded = seq_len + 3 (padding on right).
    # conv1d output at position t = sum(x[t:t+4] * w) for t in [0, T_padded-3)
    # We want the first seq_len positions.

    x_2d = x.squeeze(0).contiguous()  # (D, T_padded)
    w = weight.contiguous()  # (D, 4)
    output = torch.empty(D, seq_len, dtype=x.dtype, device=x.device)

    BLOCK_T = 256
    grid = (D, (seq_len + BLOCK_T - 1) // BLOCK_T)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _causal_conv1d_prefill_silu_kernel,
        (x_2d, w, output, seq_len, BLOCK_T),
    )
    return output.unsqueeze(0)  # (1, D, seq_len)


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

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
    )

    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state_val = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state_val
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
            mixed_qkv = causal_conv1d_prefill_silu_cutile(padded, self.conv1d.weight.squeeze(1), seq_len)

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

    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
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

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

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
