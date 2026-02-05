# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl
from tilegym.logger import get_logger
from .utils import next_power_of_2

logger = get_logger(__name__)

INV_LOG_2 = 1.0 / math.log(2)
LN2 = math.log(2)

# Define type aliases for Constant integers and booleans
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


# --- FMHA Forward Kernel (for inference, no backward) ---
@ct.kernel(occupancy=2)
def fmha_kernel(
    Q,
    K,
    V,
    Out,
    qk_scale: float,
    input_pos: int,
    TILE_D: ConstInt,  # TILE_D = hidden_size
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
    EVEN_K: ConstBool,
):
    """
    cuTile kernel for Fused Multi-Head Attention (FMHA).
    Computes attention output for a specific batch item and head, using tiling and online softmax.
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)  # [TILE_M]
    offs_m += input_pos
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)  # [TILE_N]
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load query tile for this batch, head, and M-chunk
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape(
        (TILE_M, TILE_D)
    )  # [TILE_M, TILE_D]

    # Loop over k, v and update accumulator
    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        # When kv pos could exceed q pos
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        # When kv pos could exceed k_seqlen
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        # --- Compute QK product ---
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]

        # --- Apply Causal Masking ---
        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            # Out of bound mask
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            # Causal mask
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)  # [TILE_M, TILE_N]
            mask = ct.where(mask, 0.0, -math.inf)  # [TILE_M, TILE_N]
            qk += mask

        # --- Online Softmax Update ---
        # Moving qk_scale multiplication after reduce_max is to improve performance.
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [TILE_M, TILE_N]

        # Attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]
        # Update m_i and l_i
        l_i = l_i * alpha + l_ij  # [TILE_M, 1]
        # Scale acc
        acc = acc * alpha  # [TILE_M, TILE_N]

        # --- Compute PV product ---
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))  # [TILE_N, TILE_D]
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, TILE_N]
        m_i = m_ij  # [TILE_M, 1]

    # --- Final Normalization and Store ---
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


# --- FMHA Forward Kernel with LSE (for training, with backward support) ---
@ct.kernel(occupancy=2)
def fmha_fwd_kernel_with_lse(
    Q,
    K,
    V,
    Out,
    LSE,  # Log-Sum-Exp output: flattened to 1D [batch * num_heads * seq_len]
    qk_scale: float,
    input_pos: int,
    TILE_D: ConstInt,
    H: ConstInt,
    B: ConstInt,  # Batch size
    SEQ_LEN: ConstInt,  # Sequence length
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
    EVEN_K: ConstBool,
):
    """
    FMHA forward kernel that also saves the log-sum-exp (LSE) for backward pass.
    LSE = m + log2(l) where m is the running max and l is the running sum of exp.
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    qk_scale = qk_scale * INV_LOG_2

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m += input_pos
    offs_m = offs_m[:, None]

    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]

    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape(
        (TILE_M, TILE_D)
    )

    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:
        mask_start = (input_pos + bid_x * TILE_M) // TILE_N
        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    for j in range(0, Tc):
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)
            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)
            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            mask = ct.where(mask, 0.0, -math.inf)
            qk += mask

        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Final normalization
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)

    # Store LSE = m + log2(l) for backward pass
    # This is used to reconstruct softmax probabilities in backward
    lse_tile = m_i + ct.log2(l_i)  # [TILE_M, 1]
    lse_tile = lse_tile.reshape((TILE_M,))  # [TILE_M]

    # Use scatter for 1D-like storage since ct.store with tile indexing doesn't work for 3D tensors
    # LSE is passed as 1D flattened array [batch * heads * seq_len]
    # Compute linear index for this batch/head/tile
    lse_offsets = ct.arange(TILE_M, dtype=ct.int32)
    lse_indices = batch_idx * (H * SEQ_LEN) + head_idx * SEQ_LEN + bid_x * TILE_M + lse_offsets
    ct.scatter(LSE, lse_indices, lse_tile)


# --- Backward Pass Kernels ---

@ct.kernel(occupancy=2)
def fmha_bwd_preprocess_kernel(
    O,
    dO,
    Delta,  # Output: row-wise dot product of O and dO (flattened 1D)
    TILE_M: ConstInt,
    TILE_D: ConstInt,
    H: ConstInt,
    SEQ_LEN: ConstInt,
):
    """
    Preprocess for backward: compute Delta[i] = sum(O[i] * dO[i]) for each query position.
    This is needed for the backward pass gradient computation.
    """
    bid_m = ct.bid(0)
    bid_hz = ct.bid(1)
    batch_idx = bid_hz // H
    head_idx = bid_hz % H

    # Load O and dO tiles using tile-based indexing (bid_m is tile index)
    o_tile = ct.load(O, index=(batch_idx, head_idx, bid_m, 0), shape=(1, 1, TILE_M, TILE_D),
                     padding_mode=ct.PaddingMode.ZERO)
    o_tile = o_tile.reshape((TILE_M, TILE_D)).astype(ct.float32)

    do_tile = ct.load(dO, index=(batch_idx, head_idx, bid_m, 0), shape=(1, 1, TILE_M, TILE_D),
                      padding_mode=ct.PaddingMode.ZERO)
    do_tile = do_tile.reshape((TILE_M, TILE_D)).astype(ct.float32)

    # Compute row-wise dot product: Delta[i] = sum_d(O[i,d] * dO[i,d])
    delta = ct.sum(o_tile * do_tile, axis=-1, keepdims=False)  # [TILE_M]

    # Store Delta using scatter (Delta is passed as 1D flattened)
    delta_offsets = ct.arange(TILE_M, dtype=ct.int32)
    delta_indices = batch_idx * (H * SEQ_LEN) + head_idx * SEQ_LEN + bid_m * TILE_M + delta_offsets
    ct.scatter(Delta, delta_indices, delta)


@ct.kernel(occupancy=2)
def fmha_bwd_dkdv_kernel(
    Q,
    K,
    V,
    dO,
    dK,
    dV,
    LSE,  # Passed as 1D flattened
    Delta,  # Passed as 1D flattened
    qk_scale: float,
    TILE_D: ConstInt,
    H_Q: ConstInt,  # Number of query heads
    H_KV: ConstInt,  # Number of KV heads
    SEQ_LEN: ConstInt,
    TILE_M: ConstInt,  # Query tile size for inner loop
    TILE_N: ConstInt,  # K/V tile size (this block's tile)
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
):
    """
    Compute dK and dV gradients.
    Each block handles one K/V tile for one KV head and iterates over all Q tiles
    from all query heads that share this KV head.

    For GQA (Grouped Query Attention), multiple query heads share the same KV head.
    This kernel accumulates gradients from all query heads in the group.
    """
    bid_n = ct.bid(0)  # K/V tile index
    bid_hz = ct.bid(1)  # batch_idx * H_KV + kv_head_idx
    batch_idx = bid_hz // H_KV
    kv_head_idx = bid_hz % H_KV

    q_seqlen = Q.shape[2]
    k_seqlen = K.shape[2]

    # Scale for exp2
    qk_scale_log2 = qk_scale * INV_LOG_2

    # Initialize accumulators for dK and dV
    dk_acc = ct.full((TILE_N, TILE_D), 0.0, dtype=ct.float32)
    dv_acc = ct.full((TILE_N, TILE_D), 0.0, dtype=ct.float32)

    # Load K and V for this block (same for all query heads in the group)
    # Use latency hint for TMA optimization
    k = ct.load(K, index=(batch_idx, kv_head_idx, bid_n, 0), shape=(1, 1, TILE_N, TILE_D),
                padding_mode=ct.PaddingMode.ZERO, latency=2)
    k = k.reshape((TILE_N, TILE_D))
    v = ct.load(V, index=(batch_idx, kv_head_idx, bid_n, 0), shape=(1, 1, TILE_N, TILE_D),
                padding_mode=ct.PaddingMode.ZERO, latency=2)
    v = v.reshape((TILE_N, TILE_D))

    # Offsets for this K/V tile
    offs_n = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    offs_n = offs_n[:, None]  # [TILE_N, 1]

    # Determine loop bounds based on causal masking
    if CAUSAL:
        start_m = bid_n * TILE_N // TILE_M
    else:
        start_m = 0
    num_m_tiles = ct.cdiv(q_seqlen, TILE_M)

    lse_delta_offsets = ct.arange(TILE_M, dtype=ct.int32)

    # Loop over all query heads that share this KV head
    for qh_offset in range(QUERY_GROUP_SIZE):
        q_head_idx = kv_head_idx * QUERY_GROUP_SIZE + qh_offset

        # Precompute base index for LSE/Delta gather for this query head
        lse_delta_base = batch_idx * (H_Q * SEQ_LEN) + q_head_idx * SEQ_LEN

        # Loop over Q tiles for this query head
        for m_idx in range(start_m, num_m_tiles):
            offs_m = m_idx * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
            offs_m = offs_m[None, :]  # [1, TILE_M]

            # Load Q tile for this query head (with TMA latency hints)
            q = ct.load(Q, index=(batch_idx, q_head_idx, m_idx, 0), shape=(1, 1, TILE_M, TILE_D),
                        padding_mode=ct.PaddingMode.ZERO, latency=2)
            q = q.reshape((TILE_M, TILE_D))

            # Load dO tile for this query head
            do = ct.load(dO, index=(batch_idx, q_head_idx, m_idx, 0), shape=(1, 1, TILE_M, TILE_D),
                         padding_mode=ct.PaddingMode.ZERO, latency=3)
            do = do.reshape((TILE_M, TILE_D))

            # Load LSE and Delta for this query head
            lse_indices = lse_delta_base + m_idx * TILE_M + lse_delta_offsets
            lse = ct.gather(LSE, lse_indices)  # [TILE_M]
            delta = ct.gather(Delta, lse_indices)  # [TILE_M]

            # Compute K @ Q^T: [TILE_N, TILE_M]
            qk = ct.full((TILE_N, TILE_M), 0.0, dtype=ct.float32)
            q_t = q.permute((1, 0))  # [TILE_D, TILE_M]
            qk = ct.mma(k, q_t, qk)  # [TILE_N, TILE_M]

            # Compute P = softmax(QK^T) = exp2(QK * scale - LSE)
            lse_broadcast = lse[None, :]  # [1, TILE_M]
            p_t = ct.exp2(qk * qk_scale_log2 - lse_broadcast, flush_to_zero=True)  # [TILE_N, TILE_M]

            # Apply causal mask
            if CAUSAL:
                mask = offs_n <= offs_m  # [TILE_N, TILE_M]: k_pos <= q_pos
                p_t = ct.where(mask, p_t, 0.0)

            # Compute dV += P^T @ dO
            p_t_cast = p_t.astype(Q.dtype)
            dv_acc = ct.mma(p_t_cast, do, dv_acc)  # [TILE_N, TILE_D]

            # Compute dP = dO @ V^T: we need dP^T = V @ dO^T
            do_t = do.permute((1, 0))  # [TILE_D, TILE_M]
            dp_t = ct.full((TILE_N, TILE_M), 0.0, dtype=ct.float32)
            dp_t = ct.mma(v, do_t, dp_t)  # [TILE_N, TILE_M]

            # Compute dS = P * (dP - Delta)
            delta_broadcast = delta[None, :]  # [1, TILE_M]
            ds_t = p_t * (dp_t - delta_broadcast)  # [TILE_N, TILE_M]

            # Compute dK += dS^T @ Q
            ds_t_cast = ds_t.astype(Q.dtype)
            dk_acc = ct.mma(ds_t_cast, q, dk_acc)  # [TILE_N, TILE_D]

    # Apply scale to dK
    dk_acc = dk_acc * qk_scale

    # Store dK and dV (now properly accumulated from all query heads in the group)
    dk_store = dk_acc.reshape((1, 1, TILE_N, TILE_D)).astype(dK.dtype)
    dv_store = dv_acc.reshape((1, 1, TILE_N, TILE_D)).astype(dV.dtype)
    ct.store(dK, index=(batch_idx, kv_head_idx, bid_n, 0), tile=dk_store)
    ct.store(dV, index=(batch_idx, kv_head_idx, bid_n, 0), tile=dv_store)


@ct.kernel(occupancy=2)
def fmha_bwd_dq_kernel(
    Q,
    K,
    V,
    dO,
    dQ,
    LSE,  # Passed as 1D flattened
    Delta,  # Passed as 1D flattened
    qk_scale: float,
    TILE_D: ConstInt,
    H: ConstInt,
    SEQ_LEN: ConstInt,
    TILE_M: ConstInt,  # Q tile size (this block's tile)
    TILE_N: ConstInt,  # K/V tile size for inner loop
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
):
    """
    Compute dQ gradient.
    Each block handles one Q tile and iterates over all K/V tiles.
    """
    bid_m = ct.bid(0)  # Q tile index
    bid_hz = ct.bid(1)
    batch_idx = bid_hz // H
    head_idx = bid_hz % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE

    k_seqlen = K.shape[2]

    # Scale for exp2
    qk_scale_log2 = qk_scale * INV_LOG_2

    # Initialize accumulator for dQ
    dq_acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load Q, dO for this block using tile-based indexing (with TMA latency hints)
    q = ct.load(Q, index=(batch_idx, head_idx, bid_m, 0), shape=(1, 1, TILE_M, TILE_D),
                padding_mode=ct.PaddingMode.ZERO, latency=2)
    q = q.reshape((TILE_M, TILE_D))
    do = ct.load(dO, index=(batch_idx, head_idx, bid_m, 0), shape=(1, 1, TILE_M, TILE_D),
                 padding_mode=ct.PaddingMode.ZERO, latency=2)
    do = do.reshape((TILE_M, TILE_D))

    # Load LSE and Delta using gather (they're 1D flattened)
    lse_delta_indices = batch_idx * (H * SEQ_LEN) + head_idx * SEQ_LEN + bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    lse = ct.gather(LSE, lse_delta_indices).reshape((TILE_M, 1))  # [TILE_M, 1]
    delta = ct.gather(Delta, lse_delta_indices).reshape((TILE_M, 1))  # [TILE_M, 1]

    # Offsets for this Q tile
    offs_m = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]  # [TILE_M, 1]

    # Determine loop bounds based on causal masking
    if CAUSAL:
        # Only process K/V tiles where k_pos <= q_pos (at least partially)
        end_n = ct.cdiv((bid_m + 1) * TILE_M, TILE_N)
        end_n = min(end_n, ct.cdiv(k_seqlen, TILE_N))
    else:
        end_n = ct.cdiv(k_seqlen, TILE_N)

    # Loop over K/V tiles
    for n_idx in range(0, end_n):
        offs_n = n_idx * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
        offs_n = offs_n[None, :]  # [1, TILE_N]

        # Load K and V tiles using tile-based indexing (with TMA latency hints)
        k = ct.load(K, index=(batch_idx, off_kv_h, n_idx, 0), shape=(1, 1, TILE_N, TILE_D),
                    padding_mode=ct.PaddingMode.ZERO, latency=2)
        k = k.reshape((TILE_N, TILE_D))
        v = ct.load(V, index=(batch_idx, off_kv_h, n_idx, 0), shape=(1, 1, TILE_N, TILE_D),
                    padding_mode=ct.PaddingMode.ZERO, latency=4)
        v = v.reshape((TILE_N, TILE_D))

        # Compute Q @ K^T: [TILE_M, TILE_N]
        k_t = k.permute((1, 0))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k_t, qk)  # [TILE_M, TILE_N]

        # Compute P = softmax(QK^T * scale)
        p = ct.exp2(qk * qk_scale_log2 - lse, flush_to_zero=True)  # [TILE_M, TILE_N]

        # Apply causal mask
        if CAUSAL:
            mask = offs_m >= offs_n  # [TILE_M, TILE_N]: q_pos >= k_pos
            p = ct.where(mask, p, 0.0)

        # Compute dP = dO @ V^T: [TILE_M, TILE_N]
        v_t = v.permute((1, 0))  # [TILE_D, TILE_N]
        dp = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        dp = ct.mma(do, v_t, dp)  # [TILE_M, TILE_N]

        # Compute dS = P * (dP - Delta)
        ds = p * (dp - delta)  # [TILE_M, TILE_N]

        # Compute dQ += dS @ K
        ds_cast = ds.astype(Q.dtype)
        dq_acc = ct.mma(ds_cast, k, dq_acc)  # [TILE_M, TILE_D]

    # Apply scale to dQ
    dq_acc = dq_acc * qk_scale

    # Store dQ using tile-based indexing
    dq_store = dq_acc.reshape((1, 1, TILE_M, TILE_D)).astype(dQ.dtype)
    ct.store(dQ, index=(batch_idx, head_idx, bid_m, 0), tile=dq_store)


_FMHA_FWD_TILE_CONFIGS_BY_D = {
    64: ([64, 128, 256], [32, 64, 128]),
    128: ([64, 128, 256], [32, 64, 128]),
    256: ([64, 128], [32, 64]),
}

_FMHA_BWD_DKDV_TILE_CONFIGS_BY_D = {
    64: ([32, 64, 128], [64, 128]),
    128: ([16, 32, 64], [32, 64]),
    256: ([32], [32, 64]),
}

_FMHA_BWD_DQ_TILE_CONFIGS_BY_D = {
    64: ([64, 128], [32, 64, 128]),
    128: ([32, 64], [16, 32, 64]),
    256: ([64], [32, 64]),
}


def _head_dim_key(head_dim: int | None) -> int | None:
    if head_dim is None:
        return None
    return next_power_of_2(head_dim)


def _iter_tile_configs(tile_ms: list[int], tile_ns: list[int]):
    for tm in tile_ms:
        for tn in tile_ns:
            yield SimpleNamespace(TILE_M=tm, TILE_N=tn)


def _fmha_autotune_configs(head_dim: int | None = None):
    """
    Iterator of autotune configurations for FMHA forward kernel.

    Only tunes tile sizes; num_ctas and occupancy are left to the compiler.
    """
    key = _head_dim_key(head_dim)
    tile_ms, tile_ns = _FMHA_FWD_TILE_CONFIGS_BY_D.get(
        key, ([64, 128, 256], [32, 64, 128])
    )
    yield from _iter_tile_configs(tile_ms, tile_ns)


def cutile_autotune_fmha(
    stream,
    q,
    k,
    v,
    o,
    sm_scale,
    input_pos,
    hidden_size,
    num_heads,
    query_group_size,
    is_causal,
    EVEN_K,
):
    batch_size, _, q_len, _ = q.shape
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            math.ceil(q_len / cfg.TILE_M),
            batch_size * num_heads,
            1,
        ),
        kernel=fmha_kernel,
        args_fn=lambda cfg: (
            q,
            k,
            v,
            o,
            sm_scale,
            input_pos,
            hidden_size,
            num_heads,
            cfg.TILE_M,
            cfg.TILE_N,
            query_group_size,
            is_causal,
            EVEN_K,
        ),
        search_space=lambda: _fmha_autotune_configs(hidden_size),
        max_iter=20,
    )
    return o


def tile_prefill_fmha(q, k, v, sm_scale, is_causal=True, kernel_configs=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    batch_size, num_heads, q_len, hidden_size = q.shape
    _, num_head_kv, k_len, _ = k.shape

    assert num_heads % num_head_kv == 0
    query_group_size = num_heads // num_head_kv

    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    o = torch.empty_like(q)

    input_pos = 0  # prefill, causal

    max_tile_n = max(cfg.TILE_N for cfg in _fmha_autotune_configs(hidden_size))
    EVEN_K = (k_len % max_tile_n) == 0
    return cutile_autotune_fmha(
        torch.cuda.current_stream(),
        q,
        k,
        v,
        o,
        sm_scale,
        input_pos,
        hidden_size,
        num_heads,
        query_group_size,
        is_causal,
        EVEN_K,
    )


def tile_fmha(
    q,
    k,
    v,
    scaling=None,
    is_causal=True,
    **kwargs,
):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))
    kernel_configs = kwargs.get("kernel_configs", None)
    o = tile_prefill_fmha(q, k, v, scaling, is_causal, kernel_configs)
    return o


# --- Backward Pass Autotune Configs ---
def _fmha_bwd_autotune_configs(head_dim: int | None = None):
    """Reference configs for FMHA backward (used for padding/preprocess only).

    The actual autotuning is done by _fmha_bwd_dkdv_autotune_configs and
    _fmha_bwd_dq_autotune_configs. This function provides the superset of
    tile sizes for padding calculation.
    """
    key = _head_dim_key(head_dim)
    dkdv_ms, dkdv_ns = _FMHA_BWD_DKDV_TILE_CONFIGS_BY_D.get(
        key, ([32, 64, 128], [64, 128])
    )
    dq_ms, dq_ns = _FMHA_BWD_DQ_TILE_CONFIGS_BY_D.get(
        key, ([64, 128], [32, 64, 128])
    )
    tile_ms = sorted(set(dkdv_ms + dq_ms))
    tile_ns = sorted(set(dkdv_ns + dq_ns))
    yield from _iter_tile_configs(tile_ms, tile_ns)


def _fmha_bwd_dkdv_autotune_configs(head_dim: int | None = None):
    """Autotune configurations for dK/dV kernel.

    Only tunes tile sizes; num_ctas and occupancy are left to the compiler.

    Search dimensions:
    - TILE_M: [32, 64, 128] (Q tile, inner loop)
    - TILE_N: [64, 128] (K/V tile, this block's tile)
    """
    key = _head_dim_key(head_dim)
    tile_ms, tile_ns = _FMHA_BWD_DKDV_TILE_CONFIGS_BY_D.get(
        key, ([32, 64, 128], [64, 128])
    )
    yield from _iter_tile_configs(tile_ms, tile_ns)


def _fmha_bwd_dq_autotune_configs(head_dim: int | None = None):
    """Autotune configurations for dQ kernel.

    Only tunes tile sizes; num_ctas and occupancy are left to the compiler.

    Search dimensions:
    - TILE_M: [64, 128] (Q tile, this block's tile)
    - TILE_N: [32, 64, 128] (K/V tile, inner loop)
    """
    key = _head_dim_key(head_dim)
    tile_ms, tile_ns = _FMHA_BWD_DQ_TILE_CONFIGS_BY_D.get(
        key, ([64, 128], [32, 64, 128])
    )
    yield from _iter_tile_configs(tile_ms, tile_ns)


# --- Backward Pass Python Functions ---

def fmha_forward_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass that saves LSE for backward.

    Returns:
        output: Attention output
        lse: Log-sum-exp tensor for backward
    """
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch_size, num_heads, q_len, hidden_size = q.shape
    _, num_head_kv, k_len, _ = k.shape

    assert num_heads % num_head_kv == 0
    query_group_size = num_heads // num_head_kv

    o = torch.empty_like(q)

    # Get tile sizes from config
    configs = list(_fmha_autotune_configs(hidden_size))
    cfg = configs[0]  # Use first config for now
    TILE_M = cfg.TILE_M
    TILE_N = cfg.TILE_N

    # Pad seq_len to multiple of TILE_M to avoid out-of-bounds in backward gather
    padded_q_len = math.ceil(q_len / TILE_M) * TILE_M

    # Allocate LSE as 1D flattened for scatter operation in kernel (with padding)
    lse_flat = torch.zeros((batch_size * num_heads * padded_q_len,), dtype=torch.float32, device=q.device)

    max_tile_n = max(c.TILE_N for c in configs)
    EVEN_K = (k_len % max_tile_n) == 0

    grid = (
        math.ceil(q_len / TILE_M),
        batch_size * num_heads,
        1,
    )

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fmha_fwd_kernel_with_lse,
        (
            q,
            k,
            v,
            o,
            lse_flat,
            sm_scale,
            0,  # input_pos
            hidden_size,
            num_heads,
            batch_size,
            padded_q_len,  # Use padded length for kernel's SEQ_LEN
            TILE_M,
            TILE_N,
            query_group_size,
            is_causal,
            EVEN_K,
        ),
    )

    # Return only the valid portion of LSE (extract from padded array properly)
    # lse_flat is [batch * heads * padded_q_len], reshape to [batch, heads, padded_q_len]
    # then slice to [batch, heads, q_len]
    lse = lse_flat.view(batch_size, num_heads, padded_q_len)[:, :, :q_len].contiguous()
    return o, lse


def fmha_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    sm_scale: float,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass for Flash Attention with autotuning.

    Args:
        q, k, v: Forward inputs
        o: Forward output
        do: Gradient of output
        lse: Log-sum-exp from forward
        sm_scale: Softmax scale
        is_causal: Whether causal masking is applied

    Returns:
        dq, dk, dv: Gradients for q, k, v
    """
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = o.contiguous()
    do = do.contiguous()
    lse = lse.contiguous()

    batch_size, num_heads, q_len, hidden_size = q.shape
    _, num_head_kv, k_len, _ = k.shape

    assert num_heads % num_head_kv == 0
    query_group_size = num_heads // num_head_kv

    # Allocate outputs
    dq = torch.empty_like(q)
    dk = torch.zeros_like(k)  # Use zeros for GQA accumulation
    dv = torch.zeros_like(v)

    # Ensure TILE_D is power of 2
    TILE_D = next_power_of_2(hidden_size)

    # Get max tile size from configs for padding calculation
    all_configs = list(_fmha_bwd_autotune_configs(hidden_size))
    max_tile_m = max(cfg.TILE_M for cfg in all_configs)
    max_tile_n = max(cfg.TILE_N for cfg in all_configs)

    # Pad seq_len to multiple of max tile size to avoid out-of-bounds in gather
    padded_q_len = math.ceil(q_len / max_tile_m) * max_tile_m

    # Allocate Delta buffer as 1D flattened for scatter/gather (with padding)
    delta_flat = torch.zeros((batch_size * num_heads * padded_q_len,), dtype=torch.float32, device=q.device)

    # Pad and flatten LSE for gather operations
    if q_len != padded_q_len:
        lse_padded = torch.full((batch_size, num_heads, padded_q_len), float('inf'), dtype=torch.float32, device=q.device)
        lse_padded[:, :, :q_len] = lse
        lse_flat = lse_padded.view(-1).contiguous()
    else:
        lse_flat = lse.view(-1).contiguous()

    stream = torch.cuda.current_stream()

    # Step 1: Compute Delta = rowsum(O * dO)
    # Use first config's TILE_M for preprocess (doesn't need autotuning)
    preprocess_tile_m = all_configs[0].TILE_M
    grid_preprocess = (
        math.ceil(q_len / preprocess_tile_m),
        batch_size * num_heads,
        1,
    )
    ct.launch(
        stream,
        grid_preprocess,
        fmha_bwd_preprocess_kernel,
        (o, do, delta_flat, preprocess_tile_m, TILE_D, num_heads, padded_q_len),
    )

    # Step 2: Compute dK and dV with autotuning
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            math.ceil(k_len / cfg.TILE_N),
            batch_size * num_head_kv,
            1,
        ),
        kernel=fmha_bwd_dkdv_kernel,
        args_fn=lambda cfg: (
            q, k, v, do, dk, dv, lse_flat, delta_flat,
            sm_scale, TILE_D, num_heads, num_head_kv, padded_q_len,
            cfg.TILE_M, cfg.TILE_N, query_group_size, is_causal,
        ),
        search_space=lambda: _fmha_bwd_dkdv_autotune_configs(hidden_size),
        max_iter=20,
    )

    # Step 3: Compute dQ with autotuning
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            math.ceil(q_len / cfg.TILE_M),
            batch_size * num_heads,
            1,
        ),
        kernel=fmha_bwd_dq_kernel,
        args_fn=lambda cfg: (
            q, k, v, do, dq, lse_flat, delta_flat,
            sm_scale, TILE_D, num_heads, padded_q_len,
            cfg.TILE_M, cfg.TILE_N, query_group_size, is_causal,
        ),
        search_space=lambda: _fmha_bwd_dq_autotune_configs(hidden_size),
        max_iter=20,
    )

    return dq, dk, dv


# --- Autograd Function ---

class FlashAttentionFunction(torch.autograd.Function):
    """
    Autograd function for Flash Attention with backward support.
    """

    @staticmethod
    def forward(ctx, q, k, v, sm_scale, is_causal):
        """
        Forward pass for Flash Attention.

        Args:
            q: Query tensor [batch, num_heads, seq_len, head_dim]
            k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
            v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
            sm_scale: Softmax scale (typically 1/sqrt(head_dim))
            is_causal: Whether to apply causal masking

        Returns:
            output: Attention output [batch, num_heads, seq_len, head_dim]
        """
        o, lse = fmha_forward_with_lse(q, k, v, sm_scale, is_causal)

        # Save tensors for backward
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal

        return o

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass for Flash Attention.

        Args:
            do: Gradient of output

        Returns:
            dq, dk, dv: Gradients for q, k, v
            None, None: For sm_scale and is_causal (non-differentiable)
        """
        q, k, v, o, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        dq, dk, dv = fmha_backward(q, k, v, o, do, lse, sm_scale, is_causal)

        return dq, dk, dv, None, None


# --- Public API ---

def tile_fmha_with_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Flash Multi-Head Attention with backward support.

    This function supports autograd and can be used in training.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        scaling: Softmax scale (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking (default: True)

    Returns:
        output: Attention output [batch, num_heads, seq_len, head_dim]
    """
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))

    return FlashAttentionFunction.apply(q, k, v, scaling, is_causal)


def tile_fmha_functional(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: float = None,
    is_causal: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Flash Multi-Head Attention that automatically selects inference or training mode.

    - In inference mode (torch.no_grad() or torch.inference_mode()): uses faster forward-only kernel
    - In training mode: uses forward kernel with LSE saving and supports backward

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        v: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        scaling: Softmax scale (default: 1/sqrt(head_dim))
        is_causal: Whether to apply causal masking (default: True)

    Returns:
        output: Attention output [batch, num_heads, seq_len, head_dim]
    """
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))

    # Check if we're in inference mode (no gradients needed)
    if not torch.is_grad_enabled() or not (q.requires_grad or k.requires_grad or v.requires_grad):
        # Use fast inference path
        return tile_fmha(q, k, v, scaling=scaling, is_causal=is_causal, **kwargs)
    else:
        # Use training path with backward support
        return FlashAttentionFunction.apply(q, k, v, scaling, is_causal)


# Register cutile implementation for fmha
register_impl("fmha", "cutile")(tile_fmha)
register_impl("fmha_backward", "cutile")(tile_fmha_with_backward)
