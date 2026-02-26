# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from tilegym.backend import get_current_backend
from tilegym.ops import fmha
from tilegym.ops import fmha_decode
from tilegym.ops import silu_and_mul


class Phi3MLPTileGym(nn.Module):
    """
    TileGym-aware Phi-3 MLP replacement.

    Keeps Phi-3's fused gate_up_proj/down_proj parameter layout to preserve
    checkpoint compatibility, while accelerating SiLU+mul with TileGym kernels.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        up_states = self.gate_up_proj(hidden_states)
        if self.config.hidden_act in ("silu", "swish"):
            up_states = silu_and_mul(up_states)
        else:
            gate, up = up_states.chunk(2, dim=-1)
            up_states = up * self.activation_fn(gate)
        return self.down_proj(up_states)


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _pad_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    pad = target_dim - x.size(-1)
    if pad <= 0:
        return x
    return F.pad(x, (0, pad))


def get_fmha_phi3_interface(backend=None, kernel_configs=None):
    """
    FMHA interface for Phi-3.

    CuTile prefill FMHA requires power-of-two head_dim. Phi-3 uses head_dim=96,
    so we pad Q/K/V to next power-of-two for CuTile and slice output back.
    """

    def fmha_interface_wrapper(
        module: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        is_causal: bool | None = None,
        has_backward: bool | None = None,
        **kwargs,
    ):
        del attention_mask, dropout
        if scaling is None:
            scaling = 1.0 / math.sqrt(q.size(-1))

        if q.size(-2) == 1:
            # Decode path
            use_backend = backend
            if use_backend is None:
                use_backend = get_current_backend()
            orig_dim = q.size(-1)
            padded_dim = _next_power_of_2(orig_dim)
            if padded_dim != orig_dim:
                q_pad = _pad_last_dim(q, padded_dim)
                k_pad = _pad_last_dim(k, padded_dim)
                v_pad = _pad_last_dim(v, padded_dim)
                o = fmha_decode(q_pad, k_pad, v_pad, sm_scale=scaling, backend=use_backend)
                return o[..., :orig_dim], None
            return fmha_decode(q, k, v, sm_scale=scaling, backend=use_backend), None

        # Prefill path
        is_causal = True if is_causal is None else is_causal
        has_backward = False if has_backward is None else has_backward
        use_backend = backend
        if use_backend is None:
            use_backend = get_current_backend()

        orig_dim = q.size(-1)
        padded_dim = _next_power_of_2(orig_dim)
        if padded_dim != orig_dim:
            q = _pad_last_dim(q, padded_dim)
            k = _pad_last_dim(k, padded_dim)
            v = _pad_last_dim(v, padded_dim)

        o = fmha(
            q,
            k,
            v,
            scaling=scaling,
            is_causal=is_causal,
            has_backward=has_backward,
            kernel_configs=kernel_configs,
            backend=use_backend,
        )
        if o.size(-1) != orig_dim:
            o = o[..., :orig_dim]
        return o.transpose(1, 2).contiguous(), None

    return fmha_interface_wrapper
