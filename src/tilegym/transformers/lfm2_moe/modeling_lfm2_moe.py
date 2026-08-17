# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""TileGym replacement modules for `transformers.models.lfm2_moe.modeling_lfm2_moe`.

The MoE block and the dense SwiGLU MLP are replaced here; RoPE / RMSNorm /
attention are patched elsewhere (registry-level / class-level) by
`apply_tilegym_kernel_to_lfm2_moe`, and the hybrid short-convolution operator
layers are routed through the fused cuTile kernels in `kernels/` when
`use_cutile=True`.

`Lfm2MoeSparseMoeBlockTileGym` keeps the exact same nested-parameter layout as
the stock `Lfm2MoeSparseMoeBlock` (`self.experts = Lfm2MoeExperts(...)`,
`self.gate = Lfm2MoeTopKRouter(...)`, and the optional `self.expert_bias`
buffer) so HuggingFace `state_dict` loading works unchanged. Forward replaces
the per-expert Python loop in `Lfm2MoeExperts` with TileGym's batched
`fused_moe` kernel.

Weight-layout compatibility notes (verified against HF LFM2-MoE source):

- HF `self.experts.gate_up_proj`: shape ``(E, 2*I, H)``. The first ``I`` rows
  along axis 1 are the **gate** projection, the second ``I`` rows are the
  **up** projection — confirmed by HF's
  ``linear(x, gate_up_proj[e]).chunk(2, dim=-1)`` which produces
  ``(gate, up)`` in that order.
- HF `self.experts.down_proj`: shape ``(E, H, I)``.
- TileGym `fused_moe(w1, w2)` expects ``w1: (E, 2*I, H)`` with the standard
  ``silu_and_mul`` ordering ``silu(x[:, :I]) * x[:, I:]`` (i.e. ``[gate, up]``)
  and ``w2: (E, H, I)`` — identical to HF, so the parameters are passed through
  with **no merge / no reorder**.

Routing semantics (differ from OLMoE — reproduced inline from
`Lfm2MoeTopKRouter.forward`):

- LFM2-MoE routes with a **sigmoid** (not softmax): ``sigmoid(logits)``.
- When ``use_expert_bias`` is set, a per-expert bias is added *only to select*
  the top-k experts; the gathered weights are the un-biased sigmoid values
  (DeepSeek-V3 style). The bias lives in the ``expert_bias`` buffer.
- ``norm_topk_prob`` divides the top-k weights by ``(sum + 1e-6)``.
- ``routed_scaling_factor`` multiplies the weights.

All of the above is applied in the wrapper before calling ``fused_moe`` — the
kernel has no norm/scaling/bias arguments and multiplies the (already final)
routing weights into the down-projection output un-normalized.
"""

import torch
import torch.nn.functional as F
from torch import nn

from tilegym.ops import fused_moe
from tilegym.ops import silu_and_mul


class Lfm2MoeMLPTileGym(nn.Module):
    """Drop-in replacement for the dense ``Lfm2MoeMLP`` (used in the first
    ``num_dense_layers`` layers) that fuses the SiLU-and-mul activation via
    TileGym's ``silu_and_mul`` kernel.

    LFM2-MoE names the SwiGLU projections ``w1`` (gate), ``w3`` (up) and
    ``w2`` (down) — not the usual ``gate_proj``/``up_proj``/``down_proj`` — so
    the generic ``get_swiglu_module`` helpers are not state_dict compatible.
    This class keeps those exact ``nn.Linear`` attribute names so the
    HuggingFace ``state_dict`` loads with ``strict=True``. Follows the
    ``Phi3MLPTileGym`` precedent (nn.Linear projections + fused activation).
    """

    def __init__(self, config, intermediate_size: int | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # gate
        self.w3 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)  # up
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        up = self.w3(x)
        # silu_and_mul(cat([gate, up])) == silu(gate) * up, matching the stock
        # forward ``w2(F.silu(w1(x)) * w3(x))``.
        return self.w2(silu_and_mul(torch.cat([gate, up], dim=-1)))


class Lfm2MoeSparseMoeBlockTileGym(nn.Module):
    """Drop-in replacement for ``Lfm2MoeSparseMoeBlock`` that routes the expert
    compute through TileGym's batched ``fused_moe`` kernel.

    The nested submodule layout (``self.experts``, ``self.gate``) and the
    optional ``expert_bias`` buffer are kept identical to the stock class so the
    HuggingFace state_dict loads with ``strict=True``.
    """

    def __init__(self, config):
        super().__init__()
        # Import here so the module import is cheap and doesn't run HF init
        # at TileGym import time.
        from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeExperts
        from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeTopKRouter

        self.experts = Lfm2MoeExperts(config)
        self.gate = Lfm2MoeTopKRouter(config)
        self.use_expert_bias = config.use_expert_bias
        if self.use_expert_bias:
            # Match the stock buffer exactly (name / dtype / shape) so strict
            # state_dict loading succeeds.
            self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32))

        # Cache router metadata for convenience.
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.hidden_size = config.hidden_size

    def _route(self, hidden_flat: torch.Tensor):
        """Reproduce ``Lfm2MoeTopKRouter.forward`` inline.

        Returns ``(topk_weights, topk_indices)`` where:
        - ``topk_weights`` is the final (bias-selected, normalized, scaled)
          routing weight, cast back to ``hidden_flat.dtype``.
        - ``topk_indices`` is ``torch.long`` (output of ``torch.topk``).
        """
        # gate.weight is (num_experts, hidden_size); F.linear handles the
        # transpose and the matmul is tiny, so no cuTile matmul is needed.
        router_logits = F.linear(hidden_flat, self.gate.weight)
        routing_weights = router_logits.sigmoid()

        if self.use_expert_bias:
            # Bias is used only to *select* the experts; the returned weights
            # are the un-biased sigmoid values gathered at the selected indices.
            scores_for_routing = routing_weights + self.expert_bias
            _, topk_indices = torch.topk(scores_for_routing, self.top_k, dim=-1)
            topk_weights = torch.gather(routing_weights, dim=1, index=topk_indices).type_as(router_logits)
        else:
            topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights.to(hidden_flat.dtype), topk_indices

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

        # Match the dtype contract of the stock block, which returns a single
        # tensor (not a tuple).
        out_flat = out_flat.to(hidden_states.dtype)
        return out_flat.view(batch_size, sequence_length, hidden_dim)
