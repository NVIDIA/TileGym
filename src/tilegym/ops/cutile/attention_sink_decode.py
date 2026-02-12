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
ConstFloat = ct.Constant[float]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel
def attention_sink_decode_kernel(
    Q,
    K,
    V,
    Sinks,  # Sink token logits per head [H_qo] or dummy tensor if None
    Att_Out,
    LSE_Out,
    Start_q,  # Query position in KV cache [1]
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
    KV_LEN_PER_SPLIT: ConstInt,
    HEAD_DIM: ConstInt,
    BLOCK_N: ConstInt,
    NUM_Q_HEAD_PER_KV: ConstInt,
    QUERY_GROUP_BLOCK_SIZE: ConstInt,
    NUM_KV_SPLITS: ConstInt,
    BANDWIDTH: ConstInt,
    HAS_SINKS: ConstBool,
):
    """
    Split-KV attention kernel with sink token support for decode phase.

    This kernel processes a portion of the KV cache (split) and incorporates
    sink token contribution in the first split. Supports causal masking and
    sliding window attention.
    """
    # Get program IDs
    batch_id = ct.bid(0)
    head_id = ct.bid(1)  # KV head index
    split_id = ct.bid(2)

    qk_scale = ct.mul(softmax_scale, INV_LOG_2)

    # Load start_q value - shape (1,), extract scalar
    start_q_tile = ct.load(
        Start_q,
        index=(0,),
        shape=(1,),
        order=(0,),
        allow_tma=True,
    )
    start_q_val = start_q_tile.item()

    # Load Q with grouped query attention layout
    # Q is organized as [B, H_kv, NUM_Q_HEAD_PER_KV, HEAD_DIM]
    q = ct.load(
        Q,
        index=(batch_id, head_id, 0, 0),
        shape=(1, 1, QUERY_GROUP_BLOCK_SIZE, HEAD_DIM),
        order=(0, 1, 2, 3),
        allow_tma=True,
    )
    q = ct.reshape(q, (QUERY_GROUP_BLOCK_SIZE, HEAD_DIM))
    q = ct.transpose(q)  # Shape: (HEAD_DIM, QUERY_GROUP_BLOCK_SIZE)

    # Calculate start and end indices for this split
    start_idx = ct.mul(split_id, KV_LEN_PER_SPLIT)
    end_idx = ct.minimum(ct.add(start_idx, KV_LEN_PER_SPLIT), S_kv)

    # For causal attention, limit to positions <= start_q
    end_idx = ct.minimum(end_idx, ct.add(start_q_val, 1))

    # For sliding window, also limit start
    if BANDWIDTH > 0:
        window_start = ct.maximum(0, ct.sub(start_q_val, ct.sub(BANDWIDTH, 1)))
        start_idx = ct.maximum(start_idx, window_start)

    # Align start_idx to BLOCK_N
    start_idx = ct.mul(ct.floordiv(start_idx, BLOCK_N), BLOCK_N)

    # Load sink values for first split
    # When HAS_SINKS is False, sink_scaled = -inf so sink_exp = 0 (no contribution)
    sink_scaled = ct.full((QUERY_GROUP_BLOCK_SIZE,), -math.inf, dtype=ct.float32)
    if HAS_SINKS:
        # For grouped attention, we need per-head sinks
        # Sinks shape: [H_qo] - one sink per query head
        offs_h = ct.add(ct.mul(head_id, NUM_Q_HEAD_PER_KV), ct.arange(QUERY_GROUP_BLOCK_SIZE, dtype=ct.int32))
        sink_vals = ct.gather(Sinks, (offs_h,), padding_value=0)
        sink_vals = ct.astype(sink_vals, ct.float32)
        sink_scaled = ct.mul(sink_vals, INV_LOG_2)

    # Initialize accumulators
    m_i = ct.full((QUERY_GROUP_BLOCK_SIZE,), -math.inf, dtype=ct.float32)
    l_i = ct.full((BLOCK_N, QUERY_GROUP_BLOCK_SIZE), 1.0, dtype=ct.float32)
    acc = ct.full((HEAD_DIM, QUERY_GROUP_BLOCK_SIZE), 0.0, dtype=ct.float32)

    # Pre-compute variables
    num_tiles = ct.cdiv(KV_LEN_PER_SPLIT, BLOCK_N)
    start_tile = ct.floordiv(start_idx, BLOCK_N)
    offs_n = ct.arange(BLOCK_N, dtype=ct.int32)

    # Process keys and values in this split
    if end_idx > start_idx:
        for idx in range(num_tiles):
            cnt = ct.add(start_tile, idx)
            curr_n = ct.mul(cnt, BLOCK_N)

            # Load K - [B, H_kv, S_kv, HEAD_DIM]
            k = ct.load(
                K,
                index=(batch_id, head_id, cnt, 0),
                shape=(1, 1, BLOCK_N, HEAD_DIM),
                order=(0, 1, 2, 3),
                allow_tma=True,
            )
            k = ct.reshape(k, (BLOCK_N, HEAD_DIM))

            # Compute qk: (BLOCK_N, HEAD_DIM) @ (HEAD_DIM, QUERY_GROUP_BLOCK_SIZE)
            # Result: (BLOCK_N, QUERY_GROUP_BLOCK_SIZE)
            qk = ct.full((BLOCK_N, QUERY_GROUP_BLOCK_SIZE), 0.0, dtype=ct.float32)
            qk = ct.mma(k, q, qk)

            # Build combined mask: boundary + causal + sliding window
            kv_positions = ct.add(curr_n, offs_n)

            # Boundary mask: positions beyond valid KV
            mask = ct.greater_equal(kv_positions, S_kv)

            # Causal mask: cannot attend to future positions
            causal_mask = ct.greater(kv_positions, start_q_val)
            mask = mask | causal_mask

            # Sliding window mask
            if BANDWIDTH > 0:
                window_limit = ct.sub(ct.add(start_q_val, 1), BANDWIDTH)
                too_old = ct.less(kv_positions, window_limit)
                mask = mask | too_old

            # Apply mask - expand mask for broadcasting
            qk = ct.where(mask[:, None], -1.0e6, qk)

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
                shape=(1, 1, BLOCK_N, HEAD_DIM),
                order=(0, 1, 2, 3),
                allow_tma=True,
            )
            v = ct.reshape(v, (BLOCK_N, HEAD_DIM))
            v = ct.transpose(v)  # (HEAD_DIM, BLOCK_N)
            p = ct.astype(p, q.dtype)
            acc = ct.mma(v, p, acc=acc)

            # Update m_i
            m_i = m_ij

    # Compute l (sum of attention weights)
    l = ct.sum(l_i, 0)

    # For first split, incorporate sink contribution AFTER inner loop
    if split_id == 0:
        # Update m_i to consider sink
        m_i_with_sink = ct.maximum(m_i, sink_scaled)
        # Rescale existing l and acc by the max change
        alpha = ct.exp2(ct.sub(m_i, m_i_with_sink))
        l = ct.mul(l, alpha)
        acc = ct.mul(acc, alpha[None, :])
        # Add sink contribution to denominator
        sink_exp = ct.exp2(ct.sub(sink_scaled, m_i_with_sink))
        l = ct.add(l, sink_exp)
        # Update m_i for LSE calculation
        m_i = m_i_with_sink

    # Normalize output
    acc = ct.truediv(acc, l[None, :], flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = ct.astype(acc, ct.float32)
    acc = ct.transpose(acc)  # (QUERY_GROUP_BLOCK_SIZE, HEAD_DIM)
    acc = ct.astype(acc, Att_Out.dtype)

    # Compute LSE = m + log2(l)
    lse = ct.add(m_i, ct.log2(l))
    lse = ct.astype(lse, Att_Out.dtype)

    # Store attention output
    acc_reshaped = ct.reshape(acc, (1, 1, QUERY_GROUP_BLOCK_SIZE, 1, HEAD_DIM))

    if NUM_Q_HEAD_PER_KV == QUERY_GROUP_BLOCK_SIZE:
        # Use TMA store for optimal performance
        ct.store(
            Att_Out,
            index=(batch_id, head_id, 0, split_id, 0),
            tile=acc_reshaped,
            order=(0, 1, 2, 3, 4),
            allow_tma=True,
        )
    else:
        # Use scatter with boundary checking for non-matching tile sizes
        idx_q_offset = ct.arange(QUERY_GROUP_BLOCK_SIZE, dtype=ct.int32)[:, None]
        idx_dim = ct.arange(HEAD_DIM, dtype=ct.int32)[None, :]
        ct.scatter(
            Att_Out,
            (batch_id, head_id, idx_q_offset, split_id, idx_dim),
            acc,
            check_bounds=True,
            latency=1,
        )

    # Store LSE
    idx_lse_q_offset = ct.arange(QUERY_GROUP_BLOCK_SIZE, dtype=ct.int32)
    ct.scatter(
        LSE_Out,
        (batch_id, head_id, idx_lse_q_offset, split_id),
        lse,
        check_bounds=True,
        latency=1,
    )


class _attention_sink_decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, sm_scale, bandwidth, start_q, kv_len_per_split=None):
        """
        Attention with sink tokens using split-KV algorithm for decode phase.

        Args:
            q: [B, 1, num_kv_heads, num_kv_groups, head_dim] - single token query
            k: [B, N_KV, num_kv_heads, head_dim]
            v: [B, N_KV, num_kv_heads, head_dim]
            sinks: [num_heads] - sink token logits per head (or None)
            sm_scale: softmax scale factor
            bandwidth: sliding window size (or None/0 for no sliding window)
            start_q: query position in KV cache
            kv_len_per_split: optional, KV length per split (power of 2)

        Returns:
            Output tensor of shape [B, 1, num_heads * head_dim]
        """
        assert len(start_q) == 1
        bs, n_ctx, n_kv_heads, repeat_kv, head_dim = q.shape
        bs, n_kv_ctx, n_kv_heads_k, head_dim_k = k.shape
        bs, n_kv_ctx, n_kv_heads_v, head_dim_v = v.shape
        n_heads = n_kv_heads * repeat_kv

        assert n_ctx == 1, "Split-KV decode kernel only supports single token query"
        assert head_dim == head_dim_k and head_dim_k == head_dim_v
        assert n_kv_heads == n_kv_heads_k and n_kv_heads == n_kv_heads_v
        assert head_dim in {16, 32, 64, 128, 256}

        # Reshape Q to [B, n_heads, head_dim] (remove seq dim since it's 1)
        q_reshaped = q.view(bs, n_heads, head_dim)
        # K, V to [B, n_kv_heads, n_kv_ctx, head_dim]
        k_reshaped = k.transpose(1, 2).contiguous()
        v_reshaped = v.transpose(1, 2).contiguous()

        # Determine split configuration
        BLOCK_N = 128
        if kv_len_per_split is None:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            NUM_KV_SPLITS = NUM_SMS // (bs * n_kv_heads)
            BLOCK_SIZE = max(
                BLOCK_N,
                next_power_of_2((n_kv_ctx + NUM_KV_SPLITS - 1) // NUM_KV_SPLITS),
            )
            NUM_KV_SPLITS = (n_kv_ctx + BLOCK_SIZE - 1) // BLOCK_SIZE
        else:
            NUM_KV_SPLITS = (n_kv_ctx + kv_len_per_split - 1) // kv_len_per_split
            BLOCK_SIZE = kv_len_per_split

        assert BLOCK_SIZE == next_power_of_2(BLOCK_SIZE)

        # Allocate intermediate results
        Att_Mid_Out = torch.empty(
            (bs, n_heads, NUM_KV_SPLITS, head_dim),
            device=q.device,
            dtype=q.dtype,
        )
        LSE_Out = torch.empty(
            (bs, n_heads, NUM_KV_SPLITS),
            device=q.device,
            dtype=torch.float32,
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

        # Calculate grouped attention parameters
        num_q_head_per_kv = repeat_kv
        query_group_block_size = max(8, next_power_of_2(num_q_head_per_kv))

        # Handle bandwidth
        bandwidth_val = bandwidth if bandwidth else 0

        # Handle sinks - pass dummy tensor if None
        HAS_SINKS = sinks is not None
        sinks_arg = sinks if sinks is not None else torch.zeros(1, device=q.device, dtype=q.dtype)

        # Reshape Q for grouped query attention
        Q_grouped = q_reshaped.view(
            bs,
            n_kv_heads,
            num_q_head_per_kv,
            head_dim,
        )

        # Create grouped views for kernel
        Att_Mid_Out_5D = Att_Mid_Out.view(
            bs,
            n_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
            head_dim,
        )
        LSE_Out_4D = LSE_Out.view(
            bs,
            n_kv_heads,
            num_q_head_per_kv,
            NUM_KV_SPLITS,
        )

        # Launch kernel
        grid = (bs, n_kv_heads, NUM_KV_SPLITS)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            attention_sink_decode_kernel,
            (
                Q_grouped,
                k_reshaped,
                v_reshaped,
                sinks_arg,
                Att_Mid_Out_5D,
                LSE_Out_4D,
                start_q,
                sm_scale,
                stride_mid_ob,
                stride_mid_oh,
                stride_mid_os,
                stride_mid_lseb,
                stride_mid_lsem,
                bs,
                n_heads,
                n_kv_heads,
                n_kv_ctx,
                BLOCK_SIZE,
                HEAD_DIM,
                BLOCK_N,
                num_q_head_per_kv,
                query_group_block_size,
                NUM_KV_SPLITS,
                bandwidth_val,
                HAS_SINKS,
            ),
        )

        # Allocate final output [B, n_heads, head_dim]
        o = torch.empty(
            (bs, n_heads, head_dim),
            device=q.device,
            dtype=q.dtype,
        )

        # Use shared splitk_reduce kernel
        splitk_reduce(Att_Mid_Out, LSE_Out, o, n_kv_ctx)

        # Reshape to expected output format [B, 1, n_heads * head_dim]
        o = o.unsqueeze(1).contiguous()
        o = o.view(bs, n_ctx, n_heads * head_dim)
        return o


attention_splitkv = _attention_sink_decode.apply


@register_impl("attention_sink_decode", backend="cutile")
def attention_sink_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float = 0.125,
    sliding_window: int | None = None,
    start_q: torch.LongTensor = 0,
    kv_len_per_split: int | None = None,
    **kwargs,
):
    """Attention with sink tokens using split-KV algorithm for decode phase.

    This implementation splits the KV cache into multiple chunks and processes
    them in parallel, then reduces the partial results using the shared
    splitk_reduce kernel. This is more efficient for long KV caches during decode.

    The sink token contribution is incorporated into the first split's LSE,
    making it compatible with the standard splitk_reduce.

    The number of splits is dynamically determined based on GPU SM count to
    maximize parallelism and SM utilization.

    Args:
        query: [B, 1, num_kv_heads, num_kv_groups, head_dim] - single token query
        key: [B, N_KV, num_kv_heads, head_dim]
        value: [B, N_KV, num_kv_heads, head_dim]
        sinks: [num_heads] - sink token logits per head
        sm_scale: softmax scale factor
        sliding_window: optional sliding window size
        start_q: query position in KV cache
        kv_len_per_split: optional, KV length per split (power of 2, >= BLOCK_N)
    """
    return attention_splitkv(query, key, value, sinks, sm_scale, sliding_window, start_q, kv_len_per_split)
