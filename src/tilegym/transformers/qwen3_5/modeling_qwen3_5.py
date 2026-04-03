# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
TileGym replacements for Qwen3.5 model components.

Qwen3.5 is a hybrid model with both standard (full) attention layers and
gated delta rule linear attention layers.  This module provides:

- Qwen3_5MLPTileGym       – SwiGLU MLP accelerated with TileGym silu_and_mul
- get_fmha_qwen3_5_interface – FMHA wrapper that fixes decode-path output layout
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from tilegym.ops import silu_and_mul

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
