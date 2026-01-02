# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Experimental fused FMHA decode (split-kv + reduction in a single kernel).

See `flash_decode.py` for the baseline two-kernel approach.
"""

import math

import cuda.tile as ct
import torch
from cuda.tile._numeric_semantics import RoundingMode as RMd

from tilegym.backend import register_impl

from .utils import next_power_of_2

INV_LOG_2 = 1.0 / math.log(2)

# Type aliases for constants
ConstInt = ct.Constant[int]


@ct.kernel
def attention_decode_fused_kernel(
    Q,  # [B, H_kv, NUM_Q_HEAD_PER_KV, HEAD_DIM]
    K,  # [B, H_kv, S_kv, HEAD_DIM]
    V,  # [B, H_kv, S_kv, HEAD_DIM]
    Output,  # [B, H_q, HEAD_DIM]
    Partial_O,  # [B, H_kv, NUM_KV_SPLITS, NUM_Q_HEAD_PER_KV, HEAD_DIM]
    Partial_LSE,  # [B, H_kv, NUM_KV_SPLITS, NUM_Q_HEAD_PER_KV]
    Completion_Counter,  # [B, H_kv] int32
    softmax_scale: float,
    B: int,
    H_q: int,
    H_kv: int,
    S_kv: int,
    HEAD_DIM: ConstInt,
    TILE_N: ConstInt,
    KV_LEN_PER_SPLIT: ConstInt,
    NUM_Q_HEAD_PER_KV: ConstInt,
    NUM_KV_SPLITS: ConstInt,
):
    batch_id = ct.bid(0)
    kv_head_id = ct.bid(1)
    split_id = ct.bid(2)

    qk_scale = ct.mul(softmax_scale, INV_LOG_2)

    # =========================================
    # PHASE 1: Standard attention computation (local to this split)
    # =========================================
    q = ct.load(
        Q,
        index=(batch_id, kv_head_id, 0, 0),
        shape=(1, 1, NUM_Q_HEAD_PER_KV, HEAD_DIM),
        order=(0, 1, 2, 3),
        allow_tma=True,
    )
    q = ct.reshape(q, (NUM_Q_HEAD_PER_KV, HEAD_DIM))
    q = ct.transpose(q)  # (HEAD_DIM, NUM_Q_HEAD_PER_KV)

    m_i = ct.full((NUM_Q_HEAD_PER_KV,), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_N, NUM_Q_HEAD_PER_KV), 1.0, dtype=ct.float32)
    acc = ct.full((HEAD_DIM, NUM_Q_HEAD_PER_KV), 0.0, dtype=ct.float32)

    start_idx = ct.mul(split_id, KV_LEN_PER_SPLIT)
    end_idx = ct.minimum(ct.add(start_idx, KV_LEN_PER_SPLIT), S_kv)

    num_tiles = ct.cdiv(KV_LEN_PER_SPLIT, TILE_N)
    offs_n = ct.arange(TILE_N, dtype=ct.int32)

    for idx in range(num_tiles):
        cnt = (start_idx // TILE_N) + idx
        kv_pos = cnt * TILE_N

        if kv_pos >= end_idx:
            continue

        k = ct.load(
            K,
            index=(batch_id, kv_head_id, cnt, 0),
            shape=(1, 1, TILE_N, HEAD_DIM),
            order=(0, 1, 2, 3),
            allow_tma=True,
        )
        k = ct.reshape(k, (TILE_N, HEAD_DIM))
        qk = ct.matmul(k, q)  # (TILE_N, NUM_Q_HEAD_PER_KV)

        # Mask for split end
        if kv_pos + TILE_N > end_idx:
            mask = ct.less(ct.add(kv_pos, offs_n[:, None]), end_idx)
            qk = ct.where(mask, qk, -1.0e6)

        qk_scaled = ct.mul(qk, qk_scale)
        m_ij = ct.maximum(m_i, ct.max(qk_scaled, 0))
        qk_shifted = ct.sub(qk_scaled, m_ij[None, :])
        p = ct.exp2(qk_shifted)

        alpha = ct.exp2(ct.sub(m_i, m_ij))
        l_i = ct.add(ct.mul(l_i, alpha[None, :]), p)
        acc = ct.mul(acc, alpha[None, :])

        v = ct.load(
            V,
            index=(batch_id, kv_head_id, cnt, 0),
            shape=(1, 1, TILE_N, HEAD_DIM),
            order=(0, 1, 2, 3),
            allow_tma=True,
        )
        v = ct.reshape(v, (TILE_N, HEAD_DIM))
        v = ct.transpose(v)  # (HEAD_DIM, TILE_N)
        p = ct.astype(p, q.dtype)
        acc = ct.mma(v, p, acc=acc)

        m_i = m_ij

    # Finalize local results
    l = ct.sum(l_i, 0)  # (NUM_Q_HEAD_PER_KV,)
    acc = ct.truediv(acc, l[None, :], flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = ct.astype(acc, ct.float32)
    acc = ct.transpose(acc)  # (NUM_Q_HEAD_PER_KV, HEAD_DIM)
    acc = ct.astype(acc, Partial_O.dtype)
    lse = ct.add(m_i, ct.log2(l))  # log2-space LSE per q-head

    # =========================================
    # PHASE 2: Write partial results
    # =========================================
    ct.store(
        Partial_O,
        index=(batch_id, kv_head_id, split_id, 0, 0),
        tile=ct.reshape(acc, (1, 1, 1, NUM_Q_HEAD_PER_KV, HEAD_DIM)),
        order=(0, 1, 2, 3, 4),
        # Avoid async TMA stores here: we need the data to be globally visible
        # before the completion counter is incremented.
        allow_tma=False,
    )

    idx_q = ct.arange(NUM_Q_HEAD_PER_KV, dtype=ct.int32)
    ct.scatter(
        Partial_LSE,
        (batch_id, kv_head_id, split_id, idx_q),
        lse,
        check_bounds=True,
        latency=1,
    )

    # =========================================
    # PHASE 3: Atomic counter and reduction
    # =========================================
    # Publish partials, then increment completion counter.
    # Use RELEASE to prevent reordering of the stores after the atomic.
    old_count = ct.atomic_add(
        Completion_Counter,
        (batch_id, kv_head_id),
        1,
        check_bounds=True,
        memory_order=ct.MemoryOrder.RELEASE,
        memory_scope=ct.MemoryScope.DEVICE,
    )

    if old_count == (NUM_KV_SPLITS - 1):
        # Acquire fence to ensure we observe all other splits' published partials.
        # (atomic_add with update=0 acts as an atomic load + acquire barrier.)
        ct.atomic_add(
            Completion_Counter,
            (batch_id, kv_head_id),
            0,
            check_bounds=True,
            memory_order=ct.MemoryOrder.ACQUIRE,
            memory_scope=ct.MemoryScope.DEVICE,
        )

        # Reset counter for next iteration
        ct.atomic_xchg(
            Completion_Counter,
            (batch_id, kv_head_id),
            0,
            check_bounds=True,
            memory_order=ct.MemoryOrder.RELAXED,
            memory_scope=ct.MemoryScope.DEVICE,
        )

        # Load all partials
        all_partial_o = ct.load(
            Partial_O,
            index=(batch_id, kv_head_id, 0, 0, 0),
            shape=(1, 1, NUM_KV_SPLITS, NUM_Q_HEAD_PER_KV, HEAD_DIM),
            order=(0, 1, 2, 3, 4),
            allow_tma=False,
        )
        all_partial_o = ct.reshape(all_partial_o, (NUM_KV_SPLITS, NUM_Q_HEAD_PER_KV, HEAD_DIM))

        all_lse = ct.load(
            Partial_LSE,
            index=(batch_id, kv_head_id, 0, 0),
            shape=(1, 1, NUM_KV_SPLITS, NUM_Q_HEAD_PER_KV),
            order=(0, 1, 2, 3),
            allow_tma=False,
        )
        all_lse = ct.reshape(all_lse, (NUM_KV_SPLITS, NUM_Q_HEAD_PER_KV))

        # Reduce over splits for all Q heads in this kv-group (vectorized).
        # Avoid dynamic tile indexing (cuda.tile requires constant subscripts).
        lse_max = ct.max(all_lse, 0)  # (NUM_Q_HEAD_PER_KV,)
        weights = ct.exp2(ct.sub(all_lse, lse_max[None, :]))  # (NUM_KV_SPLITS, NUM_Q_HEAD_PER_KV)
        weights_sum = ct.sum(weights, 0)  # (NUM_Q_HEAD_PER_KV,)

        weights_3d = ct.reshape(weights, (NUM_KV_SPLITS, NUM_Q_HEAD_PER_KV, 1))
        weighted_sum = ct.sum(weights_3d * all_partial_o, axis=0)  # (NUM_Q_HEAD_PER_KV, HEAD_DIM)
        final_output = weighted_sum / ct.reshape(weights_sum, (NUM_Q_HEAD_PER_KV, 1))

        # Store all query heads for this kv-head group in one go.
        # IMPORTANT: `ct.store` indices are tile indices (not element indices).
        # With a tile shaped (1, NUM_Q_HEAD_PER_KV, HEAD_DIM) on Output[B, H_q, D],
        # the head dimension is tiled by NUM_Q_HEAD_PER_KV, so we index by `kv_head_id`.
        ct.store(
            Output,
            index=(batch_id, kv_head_id, 0),
            tile=ct.reshape(ct.astype(final_output, Output.dtype), (1, NUM_Q_HEAD_PER_KV, HEAD_DIM)),
            order=(0, 1, 2),
            allow_tma=True,
        )


class _attention_decode_fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale, kv_len_per_split=None):
        batch_size, num_q_heads = Q.shape[0], Q.shape[1]
        num_kv_heads = K.shape[1]
        seq_len, head_dim = V.shape[2], V.shape[3]

        # Reshape for processing
        Q = Q.view(batch_size, num_q_heads, head_dim)
        K = K.view(batch_size, num_kv_heads, seq_len, head_dim)
        V = V.view(batch_size, num_kv_heads, seq_len, head_dim)

        TILE_N = 128
        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_kv_splits_est = NUM_SMS // (batch_size * num_kv_heads)
            KV_LEN_PER_SPLIT = max(
                TILE_N,
                next_power_of_2((seq_len + num_kv_splits_est - 1) // num_kv_splits_est),
            )
            NUM_KV_SPLITS = (seq_len + KV_LEN_PER_SPLIT - 1) // KV_LEN_PER_SPLIT
        else:
            KV_LEN_PER_SPLIT = kv_len_per_split
            NUM_KV_SPLITS = (seq_len + KV_LEN_PER_SPLIT - 1) // KV_LEN_PER_SPLIT

        KV_LEN_PER_SPLIT = next_power_of_2(KV_LEN_PER_SPLIT)
        assert KV_LEN_PER_SPLIT >= TILE_N

        # Grouped-query layout (same constraints as existing decode kernel)
        assert num_q_heads % num_kv_heads == 0
        num_q_head_per_kv = num_q_heads // num_kv_heads
        assert head_dim == next_power_of_2(head_dim)
        assert num_q_head_per_kv == next_power_of_2(num_q_head_per_kv)

        HEAD_DIM = head_dim
        Q_grouped = Q.view(batch_size, num_kv_heads, num_q_head_per_kv, head_dim)

        # Workspaces + output
        Partial_O = torch.empty(
            (batch_size, num_kv_heads, NUM_KV_SPLITS, num_q_head_per_kv, head_dim),
            device=Q.device,
            dtype=Q.dtype,
        )
        Partial_LSE = torch.empty(
            (batch_size, num_kv_heads, NUM_KV_SPLITS, num_q_head_per_kv),
            device=Q.device,
            dtype=torch.float32,
        )
        Completion_Counter = torch.zeros(
            (batch_size, num_kv_heads),
            device=Q.device,
            dtype=torch.int32,
        )
        O = torch.empty((batch_size, num_q_heads, head_dim), device=Q.device, dtype=Q.dtype)

        grid = (batch_size, num_kv_heads, NUM_KV_SPLITS)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            attention_decode_fused_kernel,
            (
                Q_grouped,
                K,
                V,
                O,
                Partial_O,
                Partial_LSE,
                Completion_Counter,
                softmax_scale,
                batch_size,
                num_q_heads,
                num_kv_heads,
                seq_len,
                HEAD_DIM,
                TILE_N,
                KV_LEN_PER_SPLIT,
                num_q_head_per_kv,
                NUM_KV_SPLITS,
            ),
        )

        return O.view(batch_size, num_q_heads, 1, head_dim)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Fused decode backward is not implemented yet")


attention_decode_fused = _attention_decode_fused.apply


@register_impl("fmha_decode_fused", backend="cutile")
def fmha_decode_fused(q, k, v, sm_scale, kv_len_per_split=None, **kwargs):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return attention_decode_fused(q, k, v, sm_scale, kv_len_per_split)

