# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch
from cuda.tile._numeric_semantics import RoundingMode as RMd

from tilegym.backend import register_impl
from tilegym.ops.cutile.splitk_reduce import splitk_reduce

from .utils import next_power_of_2

# Type aliases for constants
ConstInt = ct.Constant[int]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel
def attention_decode_kernel_grouped(
    Q,
    K,
    V,  # query, key, value tensors
    Att_Out,
    LSE_Out,
    softmax_scale: float,
    stride_mid_ob: int,
    stride_mid_oh: int,
    stride_mid_os: int,
    stride_mid_lseb: int,
    stride_mid_lsem: int,
    B: int,
    H_qo: int,
    H_kv: int,
    S_kv: int,
    HEAD_DIM: ConstInt,  # head dimension
    TILE_N: ConstInt,
    KV_LEN_PER_SPLIT: ConstInt,
    NUM_Q_HEAD_PER_KV: ConstInt,
    QUERY_GROUP_TILE_SIZE: ConstInt,
    NUM_KV_SPLITS: ConstInt,
):
    # Get program IDs
    batch_id = ct.bid(0)
    head_id = ct.bid(1)
    tile_id = ct.bid(2)

    qk_scale = ct.mul(softmax_scale, INV_LOG_2)

    # Load Q with grouped query attention layout
    # Q is organized as [B, H_qo // NUM_Q_HEAD_PER_KV, NUM_Q_HEAD_PER_KV, HEAD_DIM]
    q = ct.load(
        Q,
        index=(batch_id, head_id, 0, 0),
        shape=(1, 1, QUERY_GROUP_TILE_SIZE, HEAD_DIM),
        order=(0, 1, 2, 3),
        allow_tma=True,
    )
    q = ct.reshape(q, (QUERY_GROUP_TILE_SIZE, HEAD_DIM))
    q = ct.transpose(q)  # Shape: (HEAD_DIM, QUERY_GROUP_TILE_SIZE)

    # Calculate start and end indices for this tile
    start_idx = ct.mul(tile_id, KV_LEN_PER_SPLIT)
    end_idx = ct.minimum(ct.add(start_idx, KV_LEN_PER_SPLIT), S_kv)

    # Initialize accumulators
    m_i = ct.full((QUERY_GROUP_TILE_SIZE,), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_N, QUERY_GROUP_TILE_SIZE), 1.0, dtype=ct.float32)
    acc = ct.full((HEAD_DIM, QUERY_GROUP_TILE_SIZE), 0.0, dtype=ct.float32)

    # Pre-compute variables outside conditional
    num_tiles = ct.cdiv(KV_LEN_PER_SPLIT, TILE_N)
    start_tile = start_idx // TILE_N
    offs_n = ct.arange(TILE_N, dtype=ct.int32)

    # Process keys and values in this tile
    if end_idx > start_idx:
        # Process each KV tile
        for idx in range(num_tiles):
            cnt = start_tile + idx
            curr_n = cnt * TILE_N

            # Load K unconditionally - TMA handles bounds, enables Tensor Core optimization
            # [B, H_kv, S_kv, HEAD_DIM]
            k = ct.load(
                K,
                index=(batch_id, head_id, cnt, 0),
                shape=(1, 1, TILE_N, HEAD_DIM),
                order=(0, 1, 2, 3),
                allow_tma=True,
            )
            k = ct.reshape(k, (TILE_N, HEAD_DIM))

            # Compute qk - unconditional execution enables Tensor Core usage
            # (HEAD_DIM, QUERY_GROUP_TILE_SIZE) @ (TILE_N, HEAD_DIM).T
            # Result: (TILE_N, QUERY_GROUP_TILE_SIZE)
            qk = ct.full((TILE_N, QUERY_GROUP_TILE_SIZE), 0.0, dtype=ct.float32)
            qk = ct.mma(k, q, qk)

            # Process boundary case (non-causal) - apply mask to result only
            if curr_n + TILE_N > S_kv:
                mask = ct.less(ct.add(curr_n, offs_n[:, None]), S_kv)
                qk = ct.where(mask, qk, -1.0e6)

            # Compute softmax statistics
            qk_scaled = ct.mul(qk, qk_scale)
            m_ij = ct.maximum(m_i, ct.max(qk_scaled, 0))
            qk = ct.sub(qk_scaled, m_ij[None, :])
            p = ct.exp2(qk)

            # Update m_i and l_i
            alpha = ct.exp2(ct.sub(m_i, m_ij))
            l_i = ct.add(ct.mul(l_i, alpha[None, :]), p)

            # Update output accumulator
            acc = ct.mul(acc, alpha[None, :])

            # Load V and update accumulator
            v = ct.load(
                V,
                index=(batch_id, head_id, cnt, 0),
                shape=(1, 1, TILE_N, HEAD_DIM),
                order=(0, 1, 2, 3),
                allow_tma=True,
            )
            v = ct.reshape(v, (TILE_N, HEAD_DIM))
            v = ct.transpose(v)  # (HEAD_DIM, TILE_N)
            p = ct.astype(p, q.dtype)
            acc = ct.mma(v, p, acc=acc)

            # Update m_i
            m_i = m_ij

    l = ct.sum(l_i, 0)
    acc = ct.truediv(acc, l[None, :], flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = ct.astype(acc, ct.float32)
    acc = ct.transpose(acc)
    acc = ct.astype(acc, Att_Out.dtype)
    l = ct.add(m_i, ct.log2(l))

    # Store attention output
    acc_reshaped = ct.reshape(acc, (1, 1, QUERY_GROUP_TILE_SIZE, 1, HEAD_DIM))

    if NUM_Q_HEAD_PER_KV == QUERY_GROUP_TILE_SIZE:
        # Use TMA store for optimal performance
        ct.store(
            Att_Out,
            index=(batch_id, head_id, 0, tile_id, 0),
            tile=acc_reshaped,
            order=(0, 1, 2, 3, 4),
            allow_tma=True,
        )
    else:
        # Use scatter with boundary checking for non-matching tile sizes
        idx_q_offset = ct.arange(QUERY_GROUP_TILE_SIZE, dtype=ct.int32)[:, None]
        idx_dim = ct.arange(HEAD_DIM, dtype=ct.int32)[None, :]
        ct.scatter(
            Att_Out,
            (batch_id, head_id, idx_q_offset, tile_id, idx_dim),
            acc,
            check_bounds=True,
            latency=1,
        )

    # Store log sum exp
    idx_lse_q_offset = ct.arange(QUERY_GROUP_TILE_SIZE, dtype=ct.int32)
    ct.scatter(
        LSE_Out,
        (batch_id, head_id, idx_lse_q_offset, tile_id),
        l,
        check_bounds=True,
        latency=1,
    )


class _attention_decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, softmax_scale, kv_len_per_split=None):
        """
        Grouped Query Attention implementation using attention_decode_kernel_grouped.
        Supports both standard attention (num_q_heads == num_kv_heads) and
        grouped attention (num_q_heads != num_kv_heads) cases.

        Args:
            Q: Query tensor of shape [batch_size, num_q_heads, 1, head_dim]
            K: Key tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            V: Value tensor of shape [batch_size, num_kv_heads, seq_len, head_dim]
            softmax_scale: Scale factor for attention computation
            kv_len_per_split: Optional KV length per split for parallelization

        Returns:
            O: Output tensor of shape [batch_size, num_q_heads, 1, head_dim]
        """
        # Get dimensions
        batch_size, num_q_heads = Q.shape[0], Q.shape[1]
        num_kv_heads = K.shape[1]
        seq_len, head_dim = V.shape[2], V.shape[3]

        # Reshape for processing
        Q = Q.view(batch_size, num_q_heads, head_dim)
        K = K.view(batch_size, num_kv_heads, seq_len, head_dim)
        V = V.view(batch_size, num_kv_heads, seq_len, head_dim)

        # Calculate number of tiles
        TILE_N = 128
        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            NUM_KV_SPLITS = max(1, NUM_SMS // (batch_size * num_kv_heads))
            TILE_SIZE = max(
                TILE_N,
                next_power_of_2((seq_len + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS),
            )
            NUM_KV_SPLITS = (seq_len + TILE_SIZE - 1) // TILE_SIZE
        else:
            NUM_KV_SPLITS = (seq_len + kv_len_per_split - 1) // kv_len_per_split
            TILE_SIZE = kv_len_per_split

        assert TILE_SIZE == next_power_of_2(TILE_SIZE)

        # Allocate intermediate results
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

        # Prepare output
        O = torch.empty_like(Q)

        # Calculate grouped attention parameters
        assert num_q_heads % num_kv_heads == 0
        num_q_head_per_kv = num_q_heads // num_kv_heads
        query_group_tile_size = max(8, next_power_of_2(num_q_head_per_kv))

        # Create grouped views for kernel
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

        # Calculate strides
        stride_mid_ob, stride_mid_oh, stride_mid_os = (
            Att_Mid_Out.stride(0),
            Att_Mid_Out.stride(1),
            Att_Mid_Out.stride(2),
        )
        stride_mid_lseb, stride_mid_lsem = (
            LSE_Out.stride(0),
            LSE_Out.stride(1),
        )

        # Round up head_dim to next power of 2
        HEAD_DIM = next_power_of_2(head_dim)

        # Reshape Q for grouped query attention
        Q_grouped = Q.view(
            batch_size,
            num_q_heads // num_q_head_per_kv,
            num_q_head_per_kv,
            head_dim,
        )

        # Launch kernel
        grid = (batch_size, num_kv_heads, NUM_KV_SPLITS)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            attention_decode_kernel_grouped,
            (
                Q_grouped,
                K,
                V,
                Att_Mid_Out_5D,
                LSE_Out_4D,
                softmax_scale,
                stride_mid_ob,
                stride_mid_oh,
                stride_mid_os,
                stride_mid_lseb,
                stride_mid_lsem,
                batch_size,
                num_q_heads,
                num_kv_heads,
                seq_len,
                HEAD_DIM,
                TILE_N,
                TILE_SIZE,
                num_q_head_per_kv,
                query_group_tile_size,
                NUM_KV_SPLITS,
            ),
        )

        # Reduce kernel splitk results
        splitk_reduce(Att_Mid_Out, LSE_Out, O, seq_len)

        return O.view(batch_size, num_q_heads, 1, head_dim)

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Attention backward is not implemented yet")


attention_decode = _attention_decode.apply


@register_impl("fmha_decode", backend="cutile")
def fmha_decode(q, k, v, sm_scale, kv_len_per_split=None, **kwargs):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    o = attention_decode(q, k, v, sm_scale, kv_len_per_split)
    return o
