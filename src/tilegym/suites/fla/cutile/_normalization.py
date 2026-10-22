# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Normalized linear attention with an injected cumulative-sum implementation."""

import torch

from tilegym.suites.fla.cutile._host_utils import input_guard


class StatefulNormalizeFunction(torch.autograd.Function):
    """Normalized linear-attention divide step with a hand-written backward."""

    @staticmethod
    @input_guard
    def forward(ctx, o, q, k, scale, z_init, reverse, cu_seqlens, cumsum):
        k_cum = cumsum(k, reverse=reverse, cu_seqlens=cu_seqlens, output_dtype=k.dtype) if k.shape[1] else k.clone()
        if z_init is not None:
            if cu_seqlens is not None:
                seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
                z_init_b = z_init.squeeze(1).repeat_interleave(seq_lens, dim=0).unsqueeze(0)
            else:
                z_init_b = z_init
            k_cum = k_cum + z_init_b
        denom = (q * scale * k_cum).sum(-1, keepdim=True) + 1e-10
        o_out = o / denom
        if cu_seqlens is not None:
            nonempty = cu_seqlens[1:] > cu_seqlens[:-1]
            idx = (cu_seqlens[:-1] if reverse else cu_seqlens[1:] - 1).to(torch.long)
            z_state_out = (
                k.new_zeros((len(cu_seqlens) - 1, 1, k.shape[2], k.shape[3])) if z_init is None else z_init.clone()
            )
            z_state_out[nonempty] = k_cum[0, idx[nonempty]].unsqueeze(1)
        elif k.shape[1]:
            z_state_out = (k_cum[:, :1] if reverse else k_cum[:, -1:]).contiguous()
        else:
            z_state_out = k.new_zeros((k.shape[0], 1, k.shape[2], k.shape[3])) if z_init is None else z_init.clone()

        ctx.cumsum = cumsum
        ctx.save_for_backward(o, q, k_cum, denom)
        ctx.scale = scale
        ctx.reverse = reverse
        ctx.cu_seqlens = cu_seqlens
        ctx.has_z_init = z_init is not None
        ctx.k_dtype = k.dtype
        return o_out, z_state_out

    @staticmethod
    @input_guard
    def backward(ctx, do, dz_state):
        o, q, k_cum, denom = ctx.saved_tensors
        scale = ctx.scale
        reverse = ctx.reverse
        cu_seqlens = ctx.cu_seqlens

        d_o_in = do / denom
        ddenom = -(do * o).sum(-1, keepdim=True) / (denom * denom)

        dq = ddenom * scale * k_cum
        dk_cum = ddenom * scale * q

        if dz_state is not None:
            if cu_seqlens is not None:
                idx = (cu_seqlens[:-1] if reverse else cu_seqlens[1:] - 1).to(torch.long)
                nonempty = cu_seqlens[1:] > cu_seqlens[:-1]
                dk_cum[0].index_add_(0, idx[nonempty], dz_state.squeeze(1)[nonempty].to(dk_cum.dtype))
            elif reverse and dk_cum.shape[1]:
                dk_cum[:, :1].add_(dz_state.to(dk_cum.dtype))
            elif dk_cum.shape[1]:
                dk_cum[:, -1:].add_(dz_state.to(dk_cum.dtype))

        dz_init = None
        if ctx.has_z_init:
            if cu_seqlens is not None:
                seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)
                seg_id = torch.repeat_interleave(
                    torch.arange(seq_lens.numel(), device=dk_cum.device),
                    seq_lens,
                )
                flat = dk_cum[0]
                dz_init = torch.zeros(
                    seq_lens.numel(),
                    flat.shape[1],
                    flat.shape[2],
                    device=flat.device,
                    dtype=flat.dtype,
                )
                dz_init.index_add_(0, seg_id, flat)
                dz_init = dz_init.unsqueeze(1)
                if dz_state is not None:
                    dz_init[~nonempty] += dz_state[~nonempty]
            else:
                dz_init = dk_cum.sum(dim=1, keepdim=True)
                if not dk_cum.shape[1] and dz_state is not None:
                    dz_init = dz_init + dz_state

        dk = (
            ctx.cumsum(
                dk_cum,
                reverse=not reverse,
                cu_seqlens=cu_seqlens,
                output_dtype=ctx.k_dtype,
            )
            if dk_cum.shape[1]
            else dk_cum.to(ctx.k_dtype)
        )
        return d_o_in, dq, dk, None, dz_init, None, None, None
