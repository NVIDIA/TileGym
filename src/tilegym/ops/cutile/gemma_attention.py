# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""
gemma-specific attention implementation with soft cap and sliding window support.

Optimized for gemma's needs:
- Soft cap (attn_logit_softcapping): Apply tanh to attention logits
- Sliding window: Limit attention to a local window
- Causal masking
- Group Query Attention (GQA)
- Uses TMA (Tensor Memory Accelerator) on SM90+
"""

import math
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl

# Constants
INV_LOG_2 = 1.0 / math.log(2)
LOG_2 = math.log(2)

# Define type aliases for Constant integers and booleans
ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
ConstBool = ct.Constant[bool]


def _gemma_attn_fwd_inner(
    K,
    V,
    acc,
    l_i,
    m_i,
    q,
    batch_idx,
    kv_head_idx,
    start_m,
    qk_scale,
    sm_scale,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    STAGE: ConstInt,
    WINDOW_SIZE: ConstInt,
    SOFT_CAP: ConstFloat,
    HAS_SOFT_CAP: ConstBool,
    offs_m,
    offs_n,
    N_CTX: ConstInt,
    EVEN_K: ConstBool,
):
    """
    Inner loop for gemma attention computation.

    STAGE:
        - 1: Off-diagonal blocks for causal
        - 2: Diagonal block for causal with mask
        - 3: Full sequence for non-causal
    """
    # Calculate loop range based on STAGE
    if STAGE == 1:
        if WINDOW_SIZE > 0:
            lo = max(0, start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N * BLOCK_N
        else:
            lo = 0
        hi = start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
    else:
        if WINDOW_SIZE > 0:
            lo = max(0, start_m * BLOCK_M - WINDOW_SIZE) // BLOCK_N * BLOCK_N
            hi = min(N_CTX, (start_m + 1) * BLOCK_M + WINDOW_SIZE)
        else:
            lo = 0
            hi = N_CTX

    cnt = lo // BLOCK_N
    for curr_n in range(lo, hi, BLOCK_N):
        k = (
            ct.load(
                K,
                index=(batch_idx, kv_head_idx, cnt, 0),
                shape=(1, 1, BLOCK_N, BLOCK_D),
                order=(0, 1, 2, 3),
                latency=3,
            )
            .reshape((BLOCK_N, BLOCK_D))
            .transpose(1, 0)
        )

        qk = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if HAS_SOFT_CAP:
            qk = ct.mul(qk, sm_scale, flush_to_zero=True)
            qk = ct.truediv(qk, SOFT_CAP, flush_to_zero=True, rounding_mode=RMd.APPROX)
            # TODO: Performance will be ready once tanh approx is supported
            # Currently using exact tanh which may impact performance
            qk = ct.tanh(qk)
            qk = ct.mul(qk, SOFT_CAP, flush_to_zero=True)

            if STAGE == 2:
                causal_mask = offs_m[:, None] >= (curr_n + offs_n[None, :])
                qk = ct.where(causal_mask, qk, -1.0e6)

            if STAGE == 3 and not EVEN_K:
                boundary_mask = curr_n + offs_n[None, :] < N_CTX
                qk = ct.where(boundary_mask, qk, -1.0e6)

            if WINDOW_SIZE > 0:
                qk_offset = curr_n + offs_n[None, :] - offs_m[:, None]
                window_mask = (qk_offset >= -WINDOW_SIZE) & (qk_offset <= WINDOW_SIZE)
                qk = ct.where(window_mask, qk, -1.0e6)

            m_ij = max(m_i, ct.mul(ct.max(qk, axis=-1), INV_LOG_2, flush_to_zero=True))
            qk = ct.sub(ct.mul(qk, INV_LOG_2, flush_to_zero=True), m_ij[:, None], flush_to_zero=True)
        else:
            if STAGE == 2:
                causal_mask = offs_m[:, None] >= (curr_n + offs_n[None, :])
                qk = ct.where(causal_mask, qk, -1.0e6)

            if STAGE == 3 and not EVEN_K:
                boundary_mask = curr_n + offs_n[None, :] < N_CTX
                qk = ct.where(boundary_mask, qk, -1.0e6)

            if WINDOW_SIZE > 0:
                qk_offset = curr_n + offs_n[None, :] - offs_m[:, None]
                window_mask = (qk_offset >= -WINDOW_SIZE) & (qk_offset <= WINDOW_SIZE)
                qk = ct.where(window_mask, qk, -1.0e6)

            m_ij = max(m_i, ct.mul(ct.max(qk, axis=-1), qk_scale, flush_to_zero=True))
            qk = ct.sub(ct.mul(qk, qk_scale, flush_to_zero=True), m_ij[:, None], flush_to_zero=True)

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1)

        alpha = ct.exp2(ct.sub(m_i, m_ij, flush_to_zero=True), flush_to_zero=True)
        l_i = ct.add(ct.mul(l_i, alpha, flush_to_zero=True), l_ij, flush_to_zero=True)

        acc = ct.mul(acc, alpha[:, None], flush_to_zero=True)

        v = ct.load(V, index=(batch_idx, kv_head_idx, cnt, 0), shape=(1, 1, BLOCK_N, BLOCK_D), latency=3).reshape(
            (BLOCK_N, BLOCK_D)
        )

        p = p.astype(q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij
        cnt += 1

    return acc, l_i, m_i


@ct.kernel(occupancy=2)
def gemma_fmha_kernel(
    Q,
    K,
    V,
    Out,
    sm_scale: float,
    B: ConstInt,
    H: ConstInt,
    S_qo: ConstInt,
    S_kv: ConstInt,
    BLOCK_D: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    STAGE: ConstInt,
    WINDOW_SIZE: ConstInt,
    SOFT_CAP: ConstFloat,
    HAS_SOFT_CAP: ConstBool,
    EVEN_K: ConstBool,
):
    """
    gemma Flash Multi-Head Attention kernel using TMA.

    STAGE:
        - 1: non-causal, single loop over all KV
        - 3: causal, two loops (stage 1 for off-diagonal, stage 2 for diagonal)

    All tensors use BNSD layout: [B, H, S, D]
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    if QUERY_GROUP_SIZE > 1:
        kv_head_idx = head_idx // QUERY_GROUP_SIZE
    else:
        kv_head_idx = head_idx

    qk_scale = ct.mul(sm_scale, INV_LOG_2, flush_to_zero=True)

    offs_m = bid_x * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
    offs_n = ct.arange(BLOCK_N, dtype=ct.int32)

    m_i = ct.full((BLOCK_M,), -math.inf, dtype=ct.float32)
    l_i = ct.full((BLOCK_M,), 1.0, dtype=ct.float32)
    acc = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=ct.float32)

    q = ct.load(
        Q,
        index=(batch_idx, head_idx, bid_x, 0),
        shape=(1, 1, BLOCK_M, BLOCK_D),
    ).reshape((BLOCK_M, BLOCK_D))

    k_seqlen = K.shape[2]

    if STAGE & 1:
        inner_stage = 4 - STAGE
        acc, l_i, m_i = _gemma_attn_fwd_inner(
            K,
            V,
            acc,
            l_i,
            m_i,
            q,
            batch_idx,
            kv_head_idx,
            bid_x,
            qk_scale,
            sm_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            inner_stage,
            WINDOW_SIZE,
            SOFT_CAP,
            HAS_SOFT_CAP,
            offs_m,
            offs_n,
            k_seqlen,
            EVEN_K,
        )

    if STAGE & 2:
        acc, l_i, m_i = _gemma_attn_fwd_inner(
            K,
            V,
            acc,
            l_i,
            m_i,
            q,
            batch_idx,
            kv_head_idx,
            bid_x,
            qk_scale,
            sm_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            2,
            WINDOW_SIZE,
            SOFT_CAP,
            HAS_SOFT_CAP,
            offs_m,
            offs_n,
            k_seqlen,
            EVEN_K,
        )

    acc = ct.truediv(acc, l_i[:, None], flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, BLOCK_M, BLOCK_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def _gemma_fmha_autotune_configs():
    """Iterator of autotune configurations for gemma FMHA kernel."""
    yield SimpleNamespace(BLOCK_M=256, BLOCK_N=128, num_ctas=1, occupancy=1)
    yield SimpleNamespace(BLOCK_M=128, BLOCK_N=128, num_ctas=1, occupancy=2)


def _cutile_autotune_gemma_fmha(
    stream,
    q,
    k,
    v,
    o,
    sm_scale,
    B,
    H,
    S_qo,
    S_kv,
    BLOCK_D,
    query_group_size,
    stage,
    window_size,
    soft_cap_val,
    has_soft_cap,
):
    """Launch gemma FMHA kernel with autotune."""
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            math.ceil(S_qo / cfg.BLOCK_M),
            B * H,
            1,
        ),
        kernel=gemma_fmha_kernel,
        args_fn=lambda cfg: (
            q,
            k,
            v,
            o,
            sm_scale,
            B,
            H,
            S_qo,
            S_kv,
            BLOCK_D,
            cfg.BLOCK_M,
            cfg.BLOCK_N,
            query_group_size,
            stage,
            window_size,
            soft_cap_val,
            has_soft_cap,
            (S_kv % cfg.BLOCK_N) == 0,
        ),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=_gemma_fmha_autotune_configs,
    )
    return o


class _gemma_attention(torch.autograd.Function):
    """Autograd function for gemma attention"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sm_scale,
        window_size=0,
        soft_cap=None,
        is_causal=True,
        use_autotune=True,
    ):
        """Forward pass for gemma attention (BNSD layout)"""
        B, H, S_qo, D = q.shape
        _, num_head_kv, S_kv, _ = k.shape
        BLOCK_D = 1 << (D - 1).bit_length()

        q = q.contiguous() if not q.is_contiguous() else q
        k = k.contiguous() if not k.is_contiguous() else k
        v = v.contiguous() if not v.is_contiguous() else v

        o = torch.empty_like(q)

        assert H % num_head_kv == 0
        query_group_size = int(H / num_head_kv)

        if S_qo == 1:
            stage = 1
        else:
            stage = 3 if is_causal else 1

        has_soft_cap = soft_cap is not None
        soft_cap_val = soft_cap if soft_cap is not None else 0.0

        if use_autotune:
            return _cutile_autotune_gemma_fmha(
                torch.cuda.current_stream(),
                q,
                k,
                v,
                o,
                sm_scale,
                B,
                H,
                S_qo,
                S_kv,
                BLOCK_D,
                query_group_size,
                stage,
                window_size,
                soft_cap_val,
                has_soft_cap,
            )

        BLOCK_M = 128
        BLOCK_N = 128
        EVEN_K = (S_kv % BLOCK_N) == 0
        grid = ((S_qo + BLOCK_M - 1) // BLOCK_M, B * H, 1)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            gemma_fmha_kernel,
            (
                q,
                k,
                v,
                o,
                sm_scale,
                B,
                H,
                S_qo,
                S_kv,
                BLOCK_D,
                BLOCK_M,
                BLOCK_N,
                query_group_size,
                stage,
                window_size,
                soft_cap_val,
                has_soft_cap,
                EVEN_K,
            ),
        )

        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass not implemented for gemma attention")


gemma_attention = _gemma_attention.apply


@register_impl("gemma_attention", backend="cutile")
def gemma_attention_cutile(
    q,
    k,
    v,
    scaling=None,
    window_size=0,
    soft_cap=None,
    is_causal=True,
    use_autotune=False,
    **kwargs,
):
    """
    gemma-specific FMHA implementation (BNSD layout).

    Args:
        q: Query [B, H, S, D]
        k: Key [B, H_kv, S, D]
        v: Value [B, H_kv, S, D]
        scaling: Attention scaling (default: 1/sqrt(d))
        window_size: Sliding window size (0 for global attention)
        soft_cap: Soft cap value (None for no soft cap)
        is_causal: Whether to apply causal masking
        use_autotune: Whether to use autotune (default: True)

    Returns:
        Output tensor [B, H, S, D]
    """
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))

    return gemma_attention(
        q,
        k,
        v,
        scaling,
        window_size,
        soft_cap,
        is_causal,
        use_autotune,
    )
