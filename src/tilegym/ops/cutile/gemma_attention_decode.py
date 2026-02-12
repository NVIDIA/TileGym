# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Gemma-specific attention decode implementation with soft cap and sliding window support.

Optimized for:
- Soft cap (attn_logit_softcapping): Apply tanh to attention logits
- Sliding window: Limit attention to a local window
- Group Query Attention (GQA)
- Split-K parallelization for long KV sequences
"""

import math

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl

from .splitk_reduce import splitk_reduce
from .utils import next_power_of_2

# Constants
INV_LOG_2 = 1.0 / math.log(2)
LOG_2 = math.log(2)

# Type aliases for constants
ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]
ConstBool = ct.Constant[bool]


def _gemma_attn_decode_inner_grouped(
    K,
    V,
    acc,
    l_i,
    m_i,
    q,
    batch_idx,
    off_kv_h,
    qk_scale,
    start_n: ConstInt,
    S_kv: ConstInt,
    KV_LEN_PER_SPLIT: ConstInt,
    BLOCK_N: ConstInt,
    HEAD_DIM: ConstInt,
    QUERY_GROUP_BLOCK_SIZE: ConstInt,
    WINDOW_SIZE: ConstInt,
    SOFT_CAP: ConstFloat,
    HAS_SOFT_CAP: ConstBool,
):
    """
    Inner loop for Gemma attention decode computation.

    Computes attention for a split of the KV sequence with soft cap and sliding window support.
    """
    cnt = start_n // BLOCK_N
    offs_n = ct.arange(BLOCK_N, dtype=ct.int32)

    for curr_n in range(start_n, start_n + KV_LEN_PER_SPLIT, BLOCK_N):
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, cnt, 0),
            shape=(1, 1, BLOCK_N, HEAD_DIM),
            order=(0, 1, 2, 3),
            latency=3,
        )
        k = ct.reshape(k, (BLOCK_N, HEAD_DIM))

        qk = ct.full((BLOCK_N, QUERY_GROUP_BLOCK_SIZE), 0.0, dtype=ct.float32)
        qk = ct.mma(k, q, qk)

        if HAS_SOFT_CAP:
            # IMPORTANT: Use original sm_scale (without INV_LOG_2) for tanh
            # because tanh is non-linear: tanh(x/log2) != tanh(x)/log2
            # qk_scale = sm_scale / log(2), so sm_scale = qk_scale * log(2)
            sm_scale_orig = ct.mul(qk_scale, LOG_2, flush_to_zero=True)
            qk = ct.mul(qk, sm_scale_orig, flush_to_zero=True)
            qk = ct.truediv(qk, SOFT_CAP, flush_to_zero=True, rounding_mode=RMd.APPROX)
            # TODO: Performance will be ready once tanh approx is supported
            # Currently using exact tanh which may impact performance
            qk = ct.tanh(qk)
            qk = ct.mul(qk, SOFT_CAP, flush_to_zero=True)

            mask = curr_n + offs_n < S_kv
            qk = ct.where(mask[:, None], qk, -1.0e6)

            if WINDOW_SIZE > 0:
                query_pos = S_kv - 1
                kv_positions = curr_n + offs_n
                in_window = kv_positions >= (query_pos - WINDOW_SIZE)
                qk = ct.where(in_window[:, None], qk, -1.0e6)

            m_ij = ct.maximum(m_i, ct.mul(ct.max(qk, axis=0), INV_LOG_2, flush_to_zero=True))
            qk = ct.sub(ct.mul(qk, INV_LOG_2, flush_to_zero=True), m_ij[None, :], flush_to_zero=True)
        else:
            mask = curr_n + offs_n < S_kv
            qk = ct.where(mask[:, None], qk, -1.0e6)

            if WINDOW_SIZE > 0:
                query_pos = S_kv - 1
                kv_positions = curr_n + offs_n
                in_window = kv_positions >= (query_pos - WINDOW_SIZE)
                qk = ct.where(in_window[:, None], qk, -1.0e6)

            m_ij = ct.maximum(m_i, ct.mul(ct.max(qk, axis=0), qk_scale, flush_to_zero=True))
            qk = ct.sub(ct.mul(qk, qk_scale, flush_to_zero=True), m_ij[None, :], flush_to_zero=True)

        p = ct.exp2(qk, flush_to_zero=True)

        alpha = ct.exp2(ct.sub(m_i, m_ij, flush_to_zero=True), flush_to_zero=True)
        l_i = ct.add(ct.mul(l_i, alpha[None, :], flush_to_zero=True), p, flush_to_zero=True)

        acc = ct.mul(acc, alpha[None, :], flush_to_zero=True)

        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, cnt, 0),
            shape=(1, 1, BLOCK_N, HEAD_DIM),
            order=(0, 1, 2, 3),
            latency=3,
        )
        v = ct.reshape(v, (BLOCK_N, HEAD_DIM))
        v = ct.transpose(v)

        p = ct.astype(p, q.dtype)
        acc = ct.mma(v, p, acc)

        m_i = m_ij
        cnt += 1

    return acc, l_i, m_i


@ct.kernel
def gemma_attention_decode_kernel_grouped(
    Q,
    K,
    V,
    Att_Out,
    LSE_Out,
    softmax_scale: float,
    B: ConstInt,
    H_qo: ConstInt,
    H_kv: ConstInt,
    S_kv: ConstInt,
    KV_LEN_PER_SPLIT: ConstInt,
    HEAD_DIM: ConstInt,
    BLOCK_N: ConstInt,
    NUM_Q_HEAD_PER_KV: ConstInt,
    QUERY_GROUP_BLOCK_SIZE: ConstInt,
    NUM_KV_SPLITS: ConstInt,
    WINDOW_SIZE: ConstInt,
    SOFT_CAP: ConstFloat,
    HAS_SOFT_CAP: ConstBool,
):
    """
    Gemma attention decode kernel with soft cap and sliding window support.

    Uses split-K parallelization to handle long KV sequences efficiently.

    Grid: (batch_size, num_kv_heads, num_kv_splits)
    """
    dtype = Q.dtype

    batch_id = ct.bid(0)
    head_id = ct.bid(1)
    block_id = ct.bid(2)

    qk_scale = ct.mul(softmax_scale, INV_LOG_2, flush_to_zero=True)

    q = ct.load(
        Q,
        index=(batch_id, head_id, 0, 0),
        shape=(1, 1, QUERY_GROUP_BLOCK_SIZE, HEAD_DIM),
        order=(0, 1, 2, 3),
        latency=3,
    )
    q = ct.reshape(q, (QUERY_GROUP_BLOCK_SIZE, HEAD_DIM))
    q = ct.transpose(q)

    start_idx = block_id * KV_LEN_PER_SPLIT
    end_idx = min(start_idx + KV_LEN_PER_SPLIT, S_kv)

    m_i = ct.full((QUERY_GROUP_BLOCK_SIZE,), -math.inf, dtype=ct.float32)
    l_i = ct.full((BLOCK_N, QUERY_GROUP_BLOCK_SIZE), 1.0, dtype=ct.float32)
    acc = ct.full((HEAD_DIM, QUERY_GROUP_BLOCK_SIZE), 0.0, dtype=ct.float32)

    if end_idx > start_idx:
        acc, l_i, m_i = _gemma_attn_decode_inner_grouped(
            K,
            V,
            acc,
            l_i,
            m_i,
            q,
            batch_id,
            head_id,
            qk_scale,
            start_idx,
            S_kv,
            KV_LEN_PER_SPLIT,
            BLOCK_N,
            HEAD_DIM,
            QUERY_GROUP_BLOCK_SIZE,
            WINDOW_SIZE,
            SOFT_CAP,
            HAS_SOFT_CAP,
        )

    l = ct.sum(l_i, axis=0)
    acc = ct.truediv(acc, l[None, :], flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = ct.astype(acc, ct.float32)
    acc = ct.transpose(acc)
    acc = ct.astype(acc, dtype)
    l = ct.add(m_i, ct.astype(ct.log2(l), dtype), flush_to_zero=True)

    acc_reshaped = ct.reshape(acc, (1, 1, QUERY_GROUP_BLOCK_SIZE, 1, HEAD_DIM))

    if NUM_Q_HEAD_PER_KV == QUERY_GROUP_BLOCK_SIZE:
        ct.store(
            Att_Out,
            index=(batch_id, head_id, 0, block_id, 0),
            tile=acc_reshaped,
            order=(0, 1, 2, 3, 4),
            allow_tma=True,
        )
    else:
        offs_h = ct.arange(QUERY_GROUP_BLOCK_SIZE, dtype=ct.int32)[:, None]
        offs_d = ct.arange(HEAD_DIM, dtype=ct.int32)[None, :]
        ct.scatter(
            Att_Out,
            (batch_id, head_id, offs_h, block_id, offs_d),
            acc,
            check_bounds=True,
            latency=1,
        )

    offs_lse_h = ct.arange(QUERY_GROUP_BLOCK_SIZE, dtype=ct.int32)
    ct.scatter(
        LSE_Out,
        (batch_id, head_id, offs_lse_h, block_id),
        l,
        check_bounds=True,
        latency=1,
    )


class _gemma_attention_decode(torch.autograd.Function):
    """Autograd function for Gemma attention decode"""

    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        softmax_scale,
        window_size=0,
        soft_cap=None,
        kv_len_per_split=None,
    ):
        """
        Gemma Attention Decode with soft cap and sliding window support.
        """
        batch_size, num_q_heads = Q.shape[0], Q.shape[1]
        num_kv_heads = K.shape[1]
        seq_len, head_dim = V.shape[2], V.shape[3]

        Q = Q.view(batch_size, num_q_heads, head_dim)
        K = K.view(batch_size, num_kv_heads, seq_len, head_dim)
        V = V.view(batch_size, num_kv_heads, seq_len, head_dim)

        BLOCK_N = 128
        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            NUM_KV_SPLITS = max(1, NUM_SMS // (batch_size * num_kv_heads))
            BLOCK_SIZE = max(
                BLOCK_N,
                next_power_of_2((seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS),
            )
            NUM_KV_SPLITS = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        else:
            NUM_KV_SPLITS = (seq_len + kv_len_per_split - 1) // kv_len_per_split
            BLOCK_SIZE = kv_len_per_split

        assert BLOCK_SIZE == next_power_of_2(BLOCK_SIZE)

        device = Q.device
        Att_Mid_Out = torch.empty(
            (batch_size, num_q_heads, NUM_KV_SPLITS, head_dim),
            device=device,
            dtype=Q.dtype,
        )
        LSE_Out = torch.empty(
            (batch_size, num_q_heads, NUM_KV_SPLITS),
            device=device,
            dtype=torch.float32,
        )

        O = torch.empty_like(Q)

        assert num_q_heads % num_kv_heads == 0
        num_q_head_per_kv = num_q_heads // num_kv_heads
        query_group_block_size = max(8, next_power_of_2(num_q_head_per_kv))

        Att_Mid_Out_5D = Att_Mid_Out.view(
            batch_size,
            num_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
            head_dim,
        )
        LSE_Out_4D = LSE_Out.view(
            batch_size,
            num_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
        )

        HEAD_DIM = next_power_of_2(head_dim)

        Q_grouped = Q.view(
            batch_size,
            num_q_heads // num_q_head_per_kv,
            num_q_head_per_kv,
            head_dim,
        )

        has_soft_cap = soft_cap is not None
        soft_cap_val = soft_cap if soft_cap is not None else 0.0

        grid = (batch_size, num_kv_heads, NUM_KV_SPLITS)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            gemma_attention_decode_kernel_grouped,
            (
                Q_grouped,
                K,
                V,
                Att_Mid_Out_5D,
                LSE_Out_4D,
                softmax_scale,
                batch_size,
                num_q_heads,
                num_kv_heads,
                seq_len,
                BLOCK_SIZE,  # KV_LEN_PER_SPLIT
                HEAD_DIM,
                BLOCK_N,
                num_q_head_per_kv,
                query_group_block_size,
                NUM_KV_SPLITS,
                window_size,
                soft_cap_val,
                has_soft_cap,
            ),
        )
        splitk_reduce(Att_Mid_Out, LSE_Out, O, seq_len)
        return O.view(batch_size, num_q_heads, 1, head_dim)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Gemma attention decode backward is not implemented yet")


gemma_attention_decode = _gemma_attention_decode.apply


@register_impl("gemma_attention_decode", backend="cutile")
def gemma_fmha_decode(
    q,
    k,
    v,
    sm_scale=None,
    window_size=0,
    soft_cap=None,
    kv_len_per_split=None,
    **kwargs,
):
    """
    Gemma-specific FMHA decode implementation.

    This function provides the main interface for Gemma attention decode,
    supporting soft cap (attn_logit_softcapping) and sliding window attention.

    Args:
        q: Query tensor [B, H, 1, D]
        k: Key tensor [B, H_kv, S, D]
        v: Value tensor [B, H_kv, S, D]
        sm_scale: Attention scaling (default: 1/sqrt(d))
        window_size: Sliding window size (0 for global attention)
        soft_cap: Soft cap value (None for no soft cap)
        kv_len_per_split: Optional KV length per split for parallelization

    Returns:
        Output tensor [B, H, 1, D]
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    o = gemma_attention_decode(
        q,
        k,
        v,
        sm_scale,
        window_size,
        soft_cap,
        kv_len_per_split,
    )
    return o
