# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
TileGym replacements for Qwen3.5 model components.

Qwen3.5 is a hybrid model with both standard (full) attention layers and
gated delta rule linear attention layers.  This module provides:

- Qwen3_5MLPTileGym       – SwiGLU MLP accelerated with TileGym silu_and_mul
- apply_partial_rope       – cuTile partial RoPE for partial_rotary_factor=0.25
                             (rotates first rope_dim=64 of head_dim=256)
- get_fmha_qwen3_5_interface – FMHA wrapper that fixes decode-path output layout
"""

import math
from typing import Optional

import cuda.tile as ct
import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from tilegym.ops import silu_and_mul
from tilegym.ops.cutile.utils import next_power_of_2

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
            hidden_states = silu_and_mul(torch.cat([gate, up], dim=-1))
        else:
            hidden_states = self.act_fn(gate) * up
        return self.down_proj(hidden_states)


# ──────────────────────────────────────────────────────────────────────
# Partial RoPE
# ──────────────────────────────────────────────────────────────────────
#
# Qwen3.5 uses partial_rotary_factor=0.25, meaning only the first
# rope_dim (= head_dim * 0.25 = 64 out of 256) dimensions receive
# rotary encoding.  The remaining dimensions pass through unchanged.

# Type aliases for cuTile constants
ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def rope_partial_kernel(
    q,
    k,
    cos,
    sin,
    cos_bs: ConstInt,
    seq_len: ConstInt,
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_RD: ConstInt,
):
    """
    In-place partial RoPE kernel operating directly on the full Q/K tensors.

    Indexes into the first ``rope_dim`` elements of head_dim via cuTile
    tile-space addressing, rotates them, and stores back.  Dims beyond
    ``rope_dim`` are never touched — no host-side slice, copy, or cat needed.

    q shape: (bsz, num_q_heads, seq_len, head_dim)   — full tensor, in-place
    k shape: (bsz, num_kv_heads, seq_len, head_dim)   — full tensor, in-place
    cos shape: (cos_bs, seq_len, rope_dim)             — original 3-D
    sin shape: (cos_bs, seq_len, rope_dim)             — original 3-D

    TILE_RD = next_power_of_2(rope_dim // 2).
    Tile-space index ``(batch, head_tile, seq, dim_tile)`` with tile shape
    ``TILE_RD`` on the last axis selects dims ``[dim_tile*TILE_RD : (dim_tile+1)*TILE_RD]``,
    so ``dim_tile=0`` → first half, ``dim_tile=1`` → second half of rope_dim.
    """
    cos_bs = cos.shape[0]

    bid = ct.bid(0)
    batch_idx = bid // seq_len
    row_idx = bid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    # Load cos/sin — first half of rope_dim (cos is doubled: first == second half)
    cos_row = ct.load(cos, index=(cos_batch_idx, row_idx, 0), shape=(1, 1, TILE_RD), padding_mode=PAD_ZERO).reshape(
        (1, TILE_RD)
    )
    sin_row = ct.load(sin, index=(cos_batch_idx, row_idx, 0), shape=(1, 1, TILE_RD), padding_mode=PAD_ZERO).reshape(
        (1, TILE_RD)
    )

    # ── Q: load first half [0:rope_dim//2] and second half [rope_dim//2:rope_dim] ──
    q1 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 0),
        shape=(1, TILE_QH, 1, TILE_RD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_RD))
    q2 = ct.load(
        q,
        index=(batch_idx, 0, row_idx, 1),
        shape=(1, TILE_QH, 1, TILE_RD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_QH, TILE_RD))
    # rotate_half: new1 = q1*cos - q2*sin,  new2 = q2*cos + q1*sin
    new_q1 = q1 * cos_row - q2 * sin_row
    new_q2 = q2 * cos_row + q1 * sin_row
    ct.store(q, index=(batch_idx, 0, row_idx, 0), tile=new_q1.reshape((1, TILE_QH, 1, TILE_RD)).astype(q.dtype))
    ct.store(q, index=(batch_idx, 0, row_idx, 1), tile=new_q2.reshape((1, TILE_QH, 1, TILE_RD)).astype(q.dtype))

    # ── K: same pattern ──
    k1 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 0),
        shape=(1, TILE_KH, 1, TILE_RD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_RD))
    k2 = ct.load(
        k,
        index=(batch_idx, 0, row_idx, 1),
        shape=(1, TILE_KH, 1, TILE_RD),
        padding_mode=PAD_ZERO,
    ).reshape((TILE_KH, TILE_RD))
    new_k1 = k1 * cos_row - k2 * sin_row
    new_k2 = k2 * cos_row + k1 * sin_row
    ct.store(k, index=(batch_idx, 0, row_idx, 0), tile=new_k1.reshape((1, TILE_KH, 1, TILE_RD)).astype(k.dtype))
    ct.store(k, index=(batch_idx, 0, row_idx, 1), tile=new_k2.reshape((1, TILE_KH, 1, TILE_RD)).astype(k.dtype))


def _rope_partial_forward(q, k, cos, sin):
    """
    Apply partial rotary position encoding **in-place**.

    The kernel indexes directly into the full ``(bsz, H, S, head_dim)``
    tensors and only touches the first ``rope_dim`` elements per head.
    No host-side slice, contiguous copy, reshape, or cat is needed.

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor (modified in-place)
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor (modified in-place)
        cos: [bsz, seq_len, rope_dim] or [1, seq_len, rope_dim] - Cosine values
        sin: [bsz, seq_len, rope_dim] or [1, seq_len, rope_dim] - Sine values

    Returns:
        (q, k) — same tensors, rotated in-place.
    """
    batch_size, n_q_head, seq_len, _ = q.shape
    n_kv_head = k.shape[1]
    rope_dim = cos.shape[-1]
    half_rope_dim = rope_dim // 2

    TILE_RD = next_power_of_2(half_rope_dim)
    TILE_QH = next_power_of_2(n_q_head)
    TILE_KH = next_power_of_2(n_kv_head)

    ct.launch(
        torch.cuda.current_stream(),
        (batch_size * seq_len, 1, 1),
        rope_partial_kernel,
        (q, k, cos, sin, cos.shape[0], seq_len, TILE_QH, TILE_KH, TILE_RD),
    )
    return q, k


def apply_partial_rope(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies partial Rotary Positional Embedding for Qwen3.5.

    Operates **in-place** on Q and K — only the first ``rope_dim`` dimensions
    are rotated; the remaining ``head_dim - rope_dim`` dimensions pass through
    unchanged without any memory copy.

    Args:
        q: [bsz, n_q_head, seq_len, head_dim] - Query tensor
        k: [bsz, n_kv_head, seq_len, head_dim] - Key tensor
        cos: [bsz, seq_len, rope_dim] - Cosine tensor (rope_dim < head_dim)
        sin: [bsz, seq_len, rope_dim] - Sine tensor
        unsqueeze_dim: Unused, kept for API compatibility

    Returns:
        (q, k) with partial RoPE applied in-place
    """
    return _rope_partial_forward(q, k, cos, sin)


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
            # Decode path — transpose output to (B, S, H, D)
            o = fmha_decode(q, k, v, sm_scale=scaling)
            return o.transpose(1, 2).contiguous(), None

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
