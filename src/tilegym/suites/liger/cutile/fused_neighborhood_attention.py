# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# cuTile kernels for fused neighborhood attention (forward + backward).

import functools
import math
import os
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)


# cuTile kernel: Fused flash-style forward (online softmax, no O(S^2) HBM)
# NOTE: occupancy is NOT hard-coded in the decorator; it is injected via
# replace_hints() at launch time so that exhaustive_search can sweep it.
@ct.kernel
def _fna_fused_forward_kernel(
    query,
    key,
    value,
    output,
    lse_cache,  # 1-D flattened [B*H*padded_seq], log-sum-exp in log2 space for backward
    qk_scale: float,
    NUM_HEADS: int,
    SEQ_LEN: int,
    HEAD_DIM: ConstInt,
    KERNEL_SIZE: ConstInt,
    DILATION: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    LSE_STRIDE: int,  # padded_seq_len = cdiv(seq_len, BLOCK_M) * BLOCK_M
):
    """
    Fused flash-style neighborhood attention forward pass with LSE output.

    Grid: (batch*heads, cdiv(seq_len, BLOCK_M), 1)
    Each block handles BLOCK_M query rows with online softmax — never
    materialises the full [B, H, S, S] attention matrix.

    Q, K, V: [B, H, S, D]   output: [B, H, S, D]
    LSE:     [B*H*S] (1-D flattened), log-sum-exp = m_i + log2(l_i)
    """
    batch_head = ct.bid(0)
    tile_m = ct.bid(1)

    batch_id = batch_head // NUM_HEADS
    head_id = batch_head % NUM_HEADS

    # Adjust scale for exp2-based softmax (multiply by 1/log(2))
    scale_log2 = qk_scale * INV_LOG_2

    # Absolute row indices for this tile: [BLOCK_M, 1]
    rows = tile_m * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)  # [BLOCK_M]
    rows = rows[:, None]  # [BLOCK_M, 1]

    # Online softmax running state — all in float32 for stability.
    # Use a large-but-finite -1e30 instead of -inf for m_i so that
    # alpha = exp2(m_i - m_ij) never evaluates as exp2(-inf - (-inf)) = NaN.
    _M_INIT = -1e30
    m_i = ct.full((BLOCK_M, 1), _M_INIT, dtype=ct.float32)  # row max
    l_i = ct.full((BLOCK_M, 1), 0.0, dtype=ct.float32)  # row sum
    acc = ct.full((BLOCK_M, HEAD_DIM), 0.0, dtype=ct.float32)  # output accumulator

    # Load Q tile once: [BLOCK_M, head_dim]. latency= hints let the compiler overlap the TMA loads
    # with the MMA (K=2, V=4, Q=2, following the TileGym attention pattern); loads stay inside the
    # ct.mma loop to avoid the warp-specialization hang.
    q = ct.load(query, index=(batch_id, head_id, tile_m, 0), shape=(1, 1, BLOCK_M, HEAD_DIM), latency=2).reshape(
        (BLOCK_M, HEAD_DIM)
    )

    half_k = KERNEL_SIZE // 2

    # The neighborhood window only touches columns in [m0 - band_w, m0 + BLOCK_M - 1 + band_w],
    # so restrict the key/value loop to the band of tiles that intersect that window instead of
    # scanning the whole sequence. Loads stay inside the loop and the in-kernel mask below
    # remains the correctness guard for partially-covered edge tiles.
    band_w = half_k * DILATION
    m0 = tile_m * BLOCK_M
    band_lo = max(0, m0 - band_w)
    band_hi = min(SEQ_LEN, m0 + BLOCK_M + band_w)
    n_start = band_lo // BLOCK_N
    n_end = ct.cdiv(band_hi, BLOCK_N)
    for j in range(n_start, n_end):
        # 1. Compute Q @ K[j]^T  ->  qk: [BLOCK_M, BLOCK_N]
        # Load K tile transposed: shape (1,1, head_dim, BLOCK_N) to get K^T
        k = ct.load(
            key,
            index=(batch_id, head_id, 0, j),
            shape=(1, 1, HEAD_DIM, BLOCK_N),
            order=(0, 1, 3, 2),
            latency=2,
        ).reshape((HEAD_DIM, BLOCK_N))  # [head_dim, BLOCK_N]

        q_mma = ct.astype(q, ct.tfloat32) if q.dtype == ct.float32 else q
        k_mma = ct.astype(k, ct.tfloat32) if k.dtype == ct.float32 else k

        qk = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
        qk = ct.mma(q_mma, k_mma, qk)  # [BLOCK_M, BLOCK_N]
        # 2. Apply neighborhood mask inline
        cols = j * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)  # [BLOCK_N]
        cols = cols[None, :]  # [1, BLOCK_N]

        # Neighborhood window: |col - row| <= half_k * dilation
        col_lo = rows - half_k * DILATION  # [BLOCK_M, 1]
        col_hi = rows + half_k * DILATION  # [BLOCK_M, 1]

        in_range = (cols >= col_lo) & (cols <= col_hi)  # [BLOCK_M, BLOCK_N]

        if DILATION > 1:
            # Also require (col - row) % dilation == 0
            rel = cols - rows  # [BLOCK_M, BLOCK_N]
            valid_dilation = (rel % DILATION) == 0
            in_range = in_range & valid_dilation

        # Out-of-bounds columns (col >= seq_len) are always masked
        in_bounds = cols < SEQ_LEN  # [1, BLOCK_N]
        in_range = in_range & in_bounds

        # Mask out-of-neighborhood entries with -inf (they become 0 after exp).
        neg_inf_tile = ct.full((BLOCK_M, BLOCK_N), -math.inf, dtype=ct.float32)
        qk = ct.where(in_range, qk, neg_inf_tile)  # [BLOCK_M, BLOCK_N]
        # 3. Online softmax update (exp2 trick for efficiency)
        qk_scaled = qk * scale_log2  # [BLOCK_M, BLOCK_N]

        # m_ij: row-wise max of scaled logits.
        # When all entries are masked (-inf), max gives -inf, but m_i starts at -1e30
        # (not -inf), so max(m_i=-1e30, -inf) = -1e30, keeping m_ij finite.
        m_ij = max(m_i, ct.max(qk_scaled, axis=-1, keepdims=True))  # [BLOCK_M, 1]

        # p = exp2(qk_scaled - m_ij): masked entries give exp2(-inf - finite) = 0
        p = ct.exp2(qk_scaled - m_ij, flush_to_zero=True)  # [BLOCK_M, BLOCK_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [BLOCK_M, 1]

        # alpha = exp2(m_i - m_ij): both are finite (m_i starts at -1e30), no NaN.
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [BLOCK_M, 1]
        l_i = l_i * alpha + l_ij  # [BLOCK_M, 1]
        acc = acc * alpha  # [BLOCK_M, head_dim]

        # Update m_i
        m_i = m_ij
        # 4. acc += p @ V[j]
        v = ct.load(
            value,
            index=(batch_id, head_id, j, 0),
            shape=(1, 1, BLOCK_N, HEAD_DIM),
            latency=4,
        ).reshape((BLOCK_N, HEAD_DIM))  # [BLOCK_N, head_dim]

        p_cast = p.astype(query.dtype)
        p_mma = ct.astype(p_cast, ct.tfloat32) if p_cast.dtype == ct.float32 else p_cast
        v_mma = ct.astype(v, ct.tfloat32) if v.dtype == ct.float32 else v

        acc = ct.mma(p_mma, v_mma, acc)  # [BLOCK_M, head_dim]
    # 5. Final normalisation and store
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    ct.store(
        output,
        index=(batch_id, head_id, tile_m, 0),
        tile=acc.reshape((1, 1, BLOCK_M, HEAD_DIM)).astype(output.dtype),
    )
    # 6. Store LSE = m_i + log2(l_i) for backward (in log2 space)
    #    Scatter into 1-D padded LSE array of size B*H*padded_seq.
    #    padded_seq = cdiv(seq_len, BLOCK_M) * BLOCK_M  (passed via lse_stride).
    #    For rows beyond seq_len (padding), write +1e30 so that backward
    #    reconstructs p ~ 0 for those ghost rows, keeping gradients clean.
    lse_tile = m_i + ct.log2(l_i)  # [BLOCK_M, 1]
    lse_tile = lse_tile.reshape((BLOCK_M,))  # [BLOCK_M]
    lse_offsets = ct.arange(BLOCK_M, dtype=ct.int32)
    row_ids = tile_m * BLOCK_M + lse_offsets  # [BLOCK_M] — always < padded_seq
    # Out-of-bounds rows get +1e30 so backward reconstructs p ~ 0
    lse_safe = ct.where(
        row_ids < SEQ_LEN,
        lse_tile,
        ct.full((BLOCK_M,), 1e30, dtype=ct.float32),
    )
    # lse_stride = padded_seq_len (passed as a ConstInt), guaranteed >= row_ids
    lse_indices = batch_head * LSE_STRIDE + row_ids
    ct.scatter(lse_cache, lse_indices, lse_safe)


# cuTile kernel: backward preprocess — delta_cache[i] = rowsum(O * grad_output)
@ct.kernel(occupancy=1)
def _fna_bwd_preprocess_kernel(
    output,
    grad_output,
    delta_cache,  # 1-D padded [B*H*padded_seq]
    NUM_HEADS: int,
    SEQ_LEN: int,
    HEAD_DIM: ConstInt,
    BLOCK_M: ConstInt,
    LSE_STRIDE: int,  # padded_seq_len = cdiv(seq_len, BLOCK_M) * BLOCK_M
):
    """
    Compute delta_cache[i] = sum_d(O[i,d] * grad_output[i,d]) for each position i.

    Grid: (batch*heads, cdiv(seq_len, BLOCK_M), 1)
    O, grad_output: [B, H, S, D]   delta_cache: [B*H*padded_seq] 1-D padded float32
    """
    batch_head = ct.bid(0)
    tile_m = ct.bid(1)

    batch_id = batch_head // NUM_HEADS
    head_id = batch_head % NUM_HEADS

    o_tile = (
        ct.load(output, index=(batch_id, head_id, tile_m, 0), shape=(1, 1, BLOCK_M, HEAD_DIM))
        .reshape((BLOCK_M, HEAD_DIM))
        .astype(ct.float32)
    )

    do_tile = (
        ct.load(grad_output, index=(batch_id, head_id, tile_m, 0), shape=(1, 1, BLOCK_M, HEAD_DIM))
        .reshape((BLOCK_M, HEAD_DIM))
        .astype(ct.float32)
    )

    delta = ct.sum(o_tile * do_tile, axis=-1, keepdims=False)  # [BLOCK_M]

    # Zero out contributions from rows beyond seq_len (padding rows)
    delta_offsets = ct.arange(BLOCK_M, dtype=ct.int32)
    row_ids = tile_m * BLOCK_M + delta_offsets  # [BLOCK_M]
    delta_safe = ct.where(
        row_ids < SEQ_LEN,
        delta,
        ct.zeros((BLOCK_M,), dtype=ct.float32),
    )
    delta_indices = batch_head * LSE_STRIDE + row_ids
    ct.scatter(delta_cache, delta_indices, delta_safe)


# cuTile kernel: fused backward grad_key + grad_value
#   Each block owns one K/V tile (tile_n) and loops over all Q tiles,
#   reconstructing P from Q, K, and LSE (no stored attn_weights).
@ct.kernel(occupancy=1)
def _fna_bwd_dkdv_kernel(
    query,
    key,
    value,
    grad_output,
    grad_key,
    grad_value,
    lse_cache,  # 1-D padded [B*H*padded_seq] float32
    delta_cache,  # 1-D padded [B*H*padded_seq] float32
    qk_scale: float,
    NUM_HEADS: int,
    SEQ_LEN: int,
    HEAD_DIM: ConstInt,
    KERNEL_SIZE: ConstInt,
    DILATION: ConstInt,
    BLOCK_M: ConstInt,  # Q-tile size (inner loop)
    BLOCK_N: ConstInt,  # K/V-tile size (this block)
    LSE_STRIDE: int,  # padded_seq_len = cdiv(seq_len, BLOCK_M) * BLOCK_M
):
    """
    Fused backward: grad_key and grad_value for one K/V tile.

    Grid: (batch*heads, cdiv(seq_len, BLOCK_N), 1)

    For each KV tile n:
      Loop over Q tiles m:
        reconstruct P[m, n] = exp2(QK[m,n] * scale * INV_LOG2 - LSE[m])
        apply neighborhood mask (set P = 0 where masked)
        grad_value[n] += P^T @ grad_output[m]
        dP[m,n] = grad_output[m] @ V[n]^T
        dS[m,n] = P * (dP - delta_cache[m])   (softmax bwd, masked positions stay 0)
        grad_key[n] += dS[m,n]^T @ Q[m] * scale
    """
    batch_head = ct.bid(0)
    tile_n = ct.bid(1)  # this block's KV tile

    batch_id = batch_head // NUM_HEADS
    head_id = batch_head % NUM_HEADS

    scale_log2 = qk_scale * INV_LOG_2
    half_k = KERNEL_SIZE // 2

    # Accumulate grad_key and grad_value in float32
    dk_acc = ct.full((BLOCK_N, HEAD_DIM), 0.0, dtype=ct.float32)
    dv_acc = ct.full((BLOCK_N, HEAD_DIM), 0.0, dtype=ct.float32)

    # Load K and V tiles for this block (reused across all Q-tile iterations)
    k = ct.load(key, index=(batch_id, head_id, tile_n, 0), shape=(1, 1, BLOCK_N, HEAD_DIM), latency=2).reshape(
        (BLOCK_N, HEAD_DIM)
    )
    v = ct.load(value, index=(batch_id, head_id, tile_n, 0), shape=(1, 1, BLOCK_N, HEAD_DIM), latency=4).reshape(
        (BLOCK_N, HEAD_DIM)
    )

    # Column indices for this KV tile: [1, BLOCK_N]
    cols = tile_n * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)  # [BLOCK_N]
    cols_bcast = cols[None, :]  # [1, BLOCK_N]

    # Scatter indices for LSE / delta_cache lookup
    lse_delta_offsets = ct.arange(BLOCK_M, dtype=ct.int32)

    # This block owns one KV tile; only Q tiles whose neighborhood window reaches columns
    # [n0, n0 + BLOCK_N - 1] contribute, i.e. rows in [n0 - band_w, n0 + BLOCK_N - 1 + band_w].
    # Band the Q-tile loop to that row range; the in-loop mask still guards edge columns.
    band_w = half_k * DILATION
    n0 = tile_n * BLOCK_N
    band_lo = max(0, n0 - band_w)
    band_hi = min(SEQ_LEN, n0 + BLOCK_N + band_w)
    m_start = band_lo // BLOCK_M
    m_end = ct.cdiv(band_hi, BLOCK_M)
    for m_idx in range(m_start, m_end):
        # Row indices for this Q tile: [BLOCK_M, 1]
        rows = m_idx * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)  # [BLOCK_M]
        rows_bcast = rows[:, None]  # [BLOCK_M, 1]

        # Load Q and grad_output tiles
        q = ct.load(query, index=(batch_id, head_id, m_idx, 0), shape=(1, 1, BLOCK_M, HEAD_DIM), latency=2).reshape(
            (BLOCK_M, HEAD_DIM)
        )
        do = ct.load(
            grad_output, index=(batch_id, head_id, m_idx, 0), shape=(1, 1, BLOCK_M, HEAD_DIM), latency=2
        ).reshape((BLOCK_M, HEAD_DIM))

        # Gather LSE and delta_cache for this Q tile (using padded stride lse_stride)
        lse_indices = batch_head * LSE_STRIDE + m_idx * BLOCK_M + lse_delta_offsets
        lse = ct.gather(lse_cache, lse_indices)  # [BLOCK_M]
        delta = ct.gather(delta_cache, lse_indices)  # [BLOCK_M]
        lse = lse[:, None]  # [BLOCK_M, 1]
        delta = delta[:, None]  # [BLOCK_M, 1]

        # Compute QK^T: [BLOCK_M, BLOCK_N]
        k_t = k.permute((1, 0))  # [head_dim, BLOCK_N]
        q_cast = q.astype(ct.float32)
        k_t_cast = k_t.astype(ct.float32)
        qk = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
        qk = ct.mma(
            ct.astype(q_cast, ct.tfloat32) if q_cast.dtype == ct.float32 else q_cast,
            ct.astype(k_t_cast, ct.tfloat32) if k_t_cast.dtype == ct.float32 else k_t_cast,
            qk,
        )

        # Reconstruct P = exp2(QK * scale_log2 - LSE)
        p = ct.exp2(qk * scale_log2 - lse, flush_to_zero=True)  # [BLOCK_M, BLOCK_N]

        # Apply neighborhood mask: positions outside window become P=0
        col_lo = rows_bcast - half_k * DILATION  # [BLOCK_M, 1]
        col_hi = rows_bcast + half_k * DILATION  # [BLOCK_M, 1]
        in_range = (cols_bcast >= col_lo) & (cols_bcast <= col_hi)  # [BLOCK_M, BLOCK_N]
        if DILATION > 1:
            rel = cols_bcast - rows_bcast
            in_range = in_range & ((rel % DILATION) == 0)
        in_bounds = cols_bcast < SEQ_LEN
        in_range = in_range & in_bounds
        p = ct.where(in_range, p, ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))

        # grad_value += P^T @ grad_output   (P: [BLOCK_M, BLOCK_N], grad_output: [BLOCK_M, head_dim])
        p_t = p.permute((1, 0))  # [BLOCK_N, BLOCK_M]
        do_cast = do.astype(ct.float32)
        p_t_mma = ct.astype(p_t, ct.tfloat32) if p_t.dtype == ct.float32 else p_t
        do_mma = ct.astype(do_cast, ct.tfloat32) if do_cast.dtype == ct.float32 else do_cast
        dv_acc = ct.mma(p_t_mma, do_mma, dv_acc)  # [BLOCK_N, head_dim]

        # dP = grad_output @ V^T: [BLOCK_M, BLOCK_N]
        v_t = v.permute((1, 0))  # [head_dim, BLOCK_N]
        v_t_cast = v_t.astype(ct.float32)
        dp = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
        dp = ct.mma(
            ct.astype(do_cast, ct.tfloat32) if do_cast.dtype == ct.float32 else do_cast,
            ct.astype(v_t_cast, ct.tfloat32) if v_t_cast.dtype == ct.float32 else v_t_cast,
            dp,
        )

        # dS = P * (dP - delta_cache):  softmax backward, masked positions stay 0
        ds = p * (dp - delta)  # [BLOCK_M, BLOCK_N]
        # Re-apply mask: out-of-bounds V/K loads can return NaN, giving 0*NaN=NaN.
        # Explicitly zero masked positions to prevent NaN from propagating into grad_key/grad_value.
        ds = ct.where(in_range, ds, ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))

        # grad_key += dS^T @ Q * scale
        ds_t = ds.permute((1, 0))  # [BLOCK_N, BLOCK_M]
        q_cast2 = q.astype(ct.float32)
        ds_t_mma = ct.astype(ds_t, ct.tfloat32) if ds_t.dtype == ct.float32 else ds_t
        q_mma2 = ct.astype(q_cast2, ct.tfloat32) if q_cast2.dtype == ct.float32 else q_cast2
        dk_acc = ct.mma(ds_t_mma, q_mma2, dk_acc)  # [BLOCK_N, head_dim]

    dk_acc = dk_acc * qk_scale

    ct.store(
        grad_key,
        index=(batch_id, head_id, tile_n, 0),
        tile=dk_acc.reshape((1, 1, BLOCK_N, HEAD_DIM)).astype(grad_key.dtype),
    )
    ct.store(
        grad_value,
        index=(batch_id, head_id, tile_n, 0),
        tile=dv_acc.reshape((1, 1, BLOCK_N, HEAD_DIM)).astype(grad_value.dtype),
    )


# cuTile kernel: fused backward grad_query
#   Each block owns one Q tile (tile_m) and loops over all K/V tiles,
#   reconstructing P from Q, K, and LSE (no stored attn_weights).
@ct.kernel(occupancy=1)
def _fna_bwd_dq_kernel(
    query,
    key,
    value,
    grad_output,
    grad_query,
    lse_cache,  # 1-D padded [B*H*padded_seq] float32
    delta_cache,  # 1-D padded [B*H*padded_seq] float32
    qk_scale: float,
    NUM_HEADS: int,
    SEQ_LEN: int,
    HEAD_DIM: ConstInt,
    KERNEL_SIZE: ConstInt,
    DILATION: ConstInt,
    BLOCK_M: ConstInt,  # Q-tile size (this block)
    BLOCK_N: ConstInt,  # K/V-tile size (inner loop)
    LSE_STRIDE: int,  # padded_seq_len = cdiv(seq_len, BLOCK_M) * BLOCK_M
):
    """
    Fused backward: grad_query for one Q tile.

    Grid: (batch*heads, cdiv(seq_len, BLOCK_M), 1)

    For each Q tile m:
      Load Q[m], grad_output[m], LSE[m], delta_cache[m]
      Loop over K/V tiles n:
        reconstruct P[m,n] + apply neighborhood mask
        dP[m,n] = grad_output[m] @ V[n]^T
        dS[m,n] = P * (dP - delta_cache[m])
        grad_query[m] += dS[m,n] @ K[n] * scale
    """
    batch_head = ct.bid(0)
    tile_m = ct.bid(1)

    batch_id = batch_head // NUM_HEADS
    head_id = batch_head % NUM_HEADS

    scale_log2 = qk_scale * INV_LOG_2
    half_k = KERNEL_SIZE // 2

    # Accumulate grad_query in float32
    dq_acc = ct.full((BLOCK_M, HEAD_DIM), 0.0, dtype=ct.float32)

    # Pre-compute indices (arithmetic only, no memory access)
    lse_delta_offsets = ct.arange(BLOCK_M, dtype=ct.int32)
    lse_indices = batch_head * LSE_STRIDE + tile_m * BLOCK_M + lse_delta_offsets

    # Row indices for this Q tile: [BLOCK_M, 1]
    rows = tile_m * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)  # [BLOCK_M]
    rows_bcast = rows[:, None]  # [BLOCK_M, 1]

    # Only KV tiles intersecting this Q tile's neighborhood window contribute; band the loop
    # like the forward pass. The in-loop mask still zeroes out-of-window entries.
    band_w = half_k * DILATION
    m0 = tile_m * BLOCK_M
    band_lo = max(0, m0 - band_w)
    band_hi = min(SEQ_LEN, m0 + BLOCK_M + band_w)
    n_start = band_lo // BLOCK_N
    n_end = ct.cdiv(band_hi, BLOCK_N)
    for n_idx in range(n_start, n_end):
        # All tensor loads are inside the loop to avoid a cuTile warp-specialization
        # deadlock: having TMA loads (Q, grad_output) outside a ct.mma accumulation loop
        # causes a kernel hang for loop counts >= 4 (seq_len >= 512 with BLOCK_N=128).
        # Q, grad_output, LSE, delta_cache are per-Q-tile constants reloaded redundantly each iteration.
        q = ct.load(query, index=(batch_id, head_id, tile_m, 0), shape=(1, 1, BLOCK_M, HEAD_DIM), latency=2).reshape(
            (BLOCK_M, HEAD_DIM)
        )
        do = ct.load(
            grad_output, index=(batch_id, head_id, tile_m, 0), shape=(1, 1, BLOCK_M, HEAD_DIM), latency=2
        ).reshape((BLOCK_M, HEAD_DIM))
        lse = ct.gather(lse_cache, lse_indices)[:, None]  # [BLOCK_M, 1]
        delta = ct.gather(delta_cache, lse_indices)[:, None]  # [BLOCK_M, 1]

        # Column indices for this KV tile: [1, BLOCK_N]
        cols = n_idx * BLOCK_N + ct.arange(BLOCK_N, dtype=ct.int32)  # [BLOCK_N]
        cols_bcast = cols[None, :]  # [1, BLOCK_N]

        # Load K and V for this KV tile
        k = ct.load(key, index=(batch_id, head_id, n_idx, 0), shape=(1, 1, BLOCK_N, HEAD_DIM), latency=2).reshape(
            (BLOCK_N, HEAD_DIM)
        )
        v = ct.load(value, index=(batch_id, head_id, n_idx, 0), shape=(1, 1, BLOCK_N, HEAD_DIM), latency=4).reshape(
            (BLOCK_N, HEAD_DIM)
        )

        # Compute QK^T: [BLOCK_M, BLOCK_N]
        k_t = k.permute((1, 0))  # [head_dim, BLOCK_N]
        q_cast = q.astype(ct.float32)
        k_t_cast = k_t.astype(ct.float32)
        qk = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
        qk = ct.mma(
            ct.astype(q_cast, ct.tfloat32) if q_cast.dtype == ct.float32 else q_cast,
            ct.astype(k_t_cast, ct.tfloat32) if k_t_cast.dtype == ct.float32 else k_t_cast,
            qk,
        )

        # Reconstruct P = exp2(QK * scale_log2 - LSE)
        p = ct.exp2(qk * scale_log2 - lse, flush_to_zero=True)  # [BLOCK_M, BLOCK_N]

        # Apply neighborhood mask: positions outside window become P=0
        col_lo = rows_bcast - half_k * DILATION
        col_hi = rows_bcast + half_k * DILATION
        in_range = (cols_bcast >= col_lo) & (cols_bcast <= col_hi)
        if DILATION > 1:
            rel = cols_bcast - rows_bcast
            in_range = in_range & ((rel % DILATION) == 0)
        in_bounds = cols_bcast < SEQ_LEN
        in_range = in_range & in_bounds
        p = ct.where(in_range, p, ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))

        # dP = grad_output @ V^T: [BLOCK_M, BLOCK_N]
        v_t = v.permute((1, 0))  # [head_dim, BLOCK_N]
        do_cast = do.astype(ct.float32)
        v_t_cast = v_t.astype(ct.float32)
        dp = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
        dp = ct.mma(
            ct.astype(do_cast, ct.tfloat32) if do_cast.dtype == ct.float32 else do_cast,
            ct.astype(v_t_cast, ct.tfloat32) if v_t_cast.dtype == ct.float32 else v_t_cast,
            dp,
        )

        # dS = P * (dP - delta_cache): softmax backward, masked positions stay 0
        ds = p * (dp - delta)  # [BLOCK_M, BLOCK_N]
        # Re-apply mask: out-of-bounds K/V loads can return NaN, giving 0*NaN=NaN.
        # Explicitly zero masked positions to prevent NaN from propagating into grad_query.
        ds = ct.where(in_range, ds, ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32))

        # grad_query += dS @ K * scale
        k_cast = k.astype(ct.float32)
        ds_mma = ct.astype(ds, ct.tfloat32) if ds.dtype == ct.float32 else ds
        k_mma = ct.astype(k_cast, ct.tfloat32) if k_cast.dtype == ct.float32 else k_cast
        dq_acc = ct.mma(ds_mma, k_mma, dq_acc)  # [BLOCK_M, head_dim]

    dq_acc = dq_acc * qk_scale

    ct.store(
        grad_query,
        index=(batch_id, head_id, tile_m, 0),
        tile=dq_acc.reshape((1, 1, BLOCK_M, HEAD_DIM)).astype(grad_query.dtype),
    )


# ---------------------------------------------------------------------------
# Autotuning helpers for the fused forward kernel
# ---------------------------------------------------------------------------


def _fused_fwd_autotune_configs():
    """
    Search space for the fused forward kernel.

    Sweep BLOCK_M in {64, 128} x BLOCK_N in {32, 64, 128} x occupancy in {1, 2}.
    Total: 2 x 3 x 2 = 12 configs.

    Design notes:
    - BLOCK_M=256 removed: too many registers at occupancy=2; keep at most 128.
    - BLOCK_N=32 included: for a ~7-key window most of BLOCK_N=128 is masked,
      so a narrower N-tile reduces useless MMA work.
    - BLOCK_N < 32 not supported by ct.mma on sm_100.
    - occupancy=1 allows larger tiles without register spill;
      occupancy=2 increases parallelism for small shapes.
    """
    for bm in [64, 128]:
        for bn in [32, 64, 128]:
            for occ in [1, 2]:
                yield SimpleNamespace(BLOCK_M=bm, BLOCK_N=bn, occupancy=occ)


# Module-level cache: (batch*heads, seq_len, head_dim, dtype, device_str) -> (cfg, tuned_kernel)
_fwd_autotune_cache: dict = {}


def _fused_fwd_autotune(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    kernel_size: int,
    dilation: int,
    stream,
) -> SimpleNamespace:
    """
    Run exhaustive_search once per problem shape, return the best config.
    The chosen config (BLOCK_M, BLOCK_N, occupancy) and the associated
    replace_hints()-patched kernel are cached at module level so subsequent
    calls go straight to ct.launch.
    """
    batch_heads = query.shape[0] * query.shape[1]
    cache_key = (batch_heads, head_dim, query.dtype, query.device)

    if cache_key not in _fwd_autotune_cache:
        configs = list(_fused_fwd_autotune_configs())
        if os.environ.get("DISABLE_AUTOTUNE", "0") == "1":
            configs = configs[:1]

        def grid_fn(cfg):
            return (
                batch_heads,
                (seq_len + cfg.BLOCK_M - 1) // cfg.BLOCK_M,
                1,
            )

        def args_fn(cfg):
            padded_seq = ((seq_len + cfg.BLOCK_M - 1) // cfg.BLOCK_M) * cfg.BLOCK_M
            # Resize lse to match BLOCK_M for this config (autotuner calls args_fn per config).
            # No pre-fill: the kernel scatters every slot (real rows = m+log2(l); ghost/padding
            # rows = 1e30 via the in-kernel ct.where), so torch.empty is safe and drops a
            # redundant device-side fill kernel. Timing-only buffer; full coverage either way.
            lse_cfg = torch.empty(
                (batch_heads * padded_seq,),
                device=query.device,
                dtype=torch.float32,
            )
            return (
                query,
                key,
                value,
                output,
                lse_cfg,
                scale,
                num_heads,
                seq_len,
                head_dim,
                kernel_size,
                dilation,
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                padded_seq,  # lse_stride
            )

        def hints_fn(cfg):
            return {"occupancy": cfg.occupancy}

        with ct.compiler_timeout(30):
            result = exhaustive_search(configs, stream, grid_fn, _fna_fused_forward_kernel, args_fn, hints_fn)
        best_cfg = result.best.config
        tuned_kernel = _fna_fused_forward_kernel.replace_hints(occupancy=best_cfg.occupancy)
        _fwd_autotune_cache[cache_key] = (best_cfg, tuned_kernel)

    return _fwd_autotune_cache[cache_key]


def _fused_neighborhood_attention_fused_forward_ct(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int = 7,
    dilation: int = 1,
    scale: float = None,
) -> tuple:
    """Flash-style cuTile forward pass. Returns ``(output, lse)`` for the backward pass."""
    batch_size, num_heads, seq_len, head_dim = query.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    output = torch.empty_like(query)
    stream = torch.cuda.current_stream()

    # padded_seq depends on the tuned BLOCK_M; allocate a placeholder lse
    # that the autotuner args_fn will resize per config.
    placeholder_lse = torch.empty(0, device=query.device, dtype=torch.float32)
    best_cfg, tuned_kernel = _fused_fwd_autotune(
        query,
        key,
        value,
        output,
        placeholder_lse,
        scale,
        num_heads,
        seq_len,
        head_dim,
        kernel_size,
        dilation,
        stream,
    )

    fwd_block_m = best_cfg.BLOCK_M
    fwd_block_n = best_cfg.BLOCK_N
    padded_seq = ((seq_len + fwd_block_m - 1) // fwd_block_m) * fwd_block_m
    # LSE: 1-D padded [B*H*padded_seq] in float32 (m + log2(l) from online softmax).
    # No pre-fill: the forward kernel scatters EVERY slot (real rows = m+log2(l); ghost/padding
    # rows = 1e30 via the in-kernel ct.where, so backward still reads 1e30 => p ~ 0). torch.empty
    # is safe and drops a redundant device-side fill kernel on every forward call.
    lse = torch.empty(
        (batch_size * num_heads * padded_seq,),
        device=query.device,
        dtype=torch.float32,
    )

    grid = (batch_size * num_heads, (seq_len + fwd_block_m - 1) // fwd_block_m, 1)
    ct.launch(
        stream,
        grid,
        tuned_kernel,
        (
            query,
            key,
            value,
            output,
            lse,
            scale,
            num_heads,
            seq_len,
            head_dim,
            kernel_size,
            dilation,
            fwd_block_m,
            fwd_block_n,
            padded_seq,
        ),
    )
    return output, lse


# Autograd Function: cuTile forward + cuTile backward
def _ensure_contiguous_ct(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def _c(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        return fn(ctx, *[_c(a) for a in args], **{k: _c(v) for k, v in kwargs.items()})

    return wrapper


class _FusedNeighborhoodAttentionFunctionCT(torch.autograd.Function):
    @staticmethod
    @_ensure_contiguous_ct
    def forward(ctx, query, key, value, kernel_size=7, dilation=1, scale=None):
        # Use the fused flash-style forward (no O(S^2) HBM materialisation).
        # Also saves LSE (log-sum-exp, shape [B*H*S]) for the backward pass.
        output, lse = _fused_neighborhood_attention_fused_forward_ct(query, key, value, kernel_size, dilation, scale)
        # Save Q, K, V, and LSE — no O(S^2) attn_weights saved.
        ctx.save_for_backward(query, key, value, output, lse)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.scale = scale

        # Record which BLOCK_M was actually used in forward so the backward can
        # reconstruct the correct LSE stride (padded_seq = cdiv(seq_len, fwd_block_m)*fwd_block_m).
        # _fused_neighborhood_attention_fused_forward_ct always populates _fwd_autotune_cache.
        batch_size, num_heads, seq_len, head_dim = query.shape
        cache_key = (batch_size * num_heads, head_dim, query.dtype, query.device)
        cfg, _ = _fwd_autotune_cache[cache_key]
        ctx.fwd_block_m = cfg.BLOCK_M
        ctx.fwd_block_n = cfg.BLOCK_N

        return output

    @staticmethod
    @_ensure_contiguous_ct
    def backward(ctx, grad_output):
        query, key, value, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = ctx.scale if ctx.scale is not None else 1.0 / math.sqrt(head_dim)
        kernel_size = ctx.kernel_size
        dilation = ctx.dilation

        # Use the BLOCK_M/N that forward actually used so LSE strides are consistent.
        fwd_block_m = ctx.fwd_block_m
        fwd_block_n = ctx.fwd_block_n

        # padded_seq must match the stride used in forward to store LSE
        padded_seq = ((seq_len + fwd_block_m - 1) // fwd_block_m) * fwd_block_m

        stream = torch.cuda.current_stream()
        # delta_cache uses the same padded stride as LSE (ghost-row slots = 0)
        delta = torch.zeros(batch_size * num_heads * padded_seq, device=query.device, dtype=torch.float32)
        grid_pre = (
            batch_size * num_heads,
            (seq_len + fwd_block_m - 1) // fwd_block_m,
            1,
        )
        ct.launch(
            stream,
            grid_pre,
            _fna_bwd_preprocess_kernel,
            (output, grad_output, delta, num_heads, seq_len, head_dim, fwd_block_m, padded_seq),
        )
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        grid_dkdv = (
            batch_size * num_heads,
            (seq_len + fwd_block_n - 1) // fwd_block_n,
            1,
        )
        ct.launch(
            stream,
            grid_dkdv,
            _fna_bwd_dkdv_kernel,
            (
                query,
                key,
                value,
                grad_output,
                grad_key,
                grad_value,
                lse,
                delta,
                scale,
                num_heads,
                seq_len,
                head_dim,
                kernel_size,
                dilation,
                fwd_block_m,  # BLOCK_M for inner Q-tile loop
                fwd_block_n,  # BLOCK_N for this block's KV tile
                padded_seq,  # lse_stride
            ),
        )
        grad_query = torch.zeros_like(query)
        grid_dq = (
            batch_size * num_heads,
            (seq_len + fwd_block_m - 1) // fwd_block_m,
            1,
        )
        ct.launch(
            stream,
            grid_dq,
            _fna_bwd_dq_kernel,
            (
                query,
                key,
                value,
                grad_output,
                grad_query,
                lse,
                delta,
                scale,
                num_heads,
                seq_len,
                head_dim,
                kernel_size,
                dilation,
                fwd_block_m,  # BLOCK_M for this block's Q tile
                fwd_block_n,  # BLOCK_N for inner KV-tile loop
                padded_seq,  # lse_stride
            ),
        )

        return grad_query, grad_key, grad_value, None, None, None


# TileGym registered implementation
@register_impl("liger.fused_neighborhood_attention", backend="cutile")
def fused_neighborhood_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int = 7,
    dilation: int = 1,
    scale: float = None,
) -> torch.Tensor:
    """
    Fused neighborhood attention — full cuTile forward and backward.

    Args:
        query: [batch, heads, seq_len, head_dim]
        key:   [batch, heads, seq_len, head_dim]
        value: [batch, heads, seq_len, head_dim]
        kernel_size: neighborhood window size (must be odd)
        dilation: dilation factor for neighborhood window
        scale: attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        output tensor of shape [batch, heads, seq_len, head_dim]
    """
    inference = (not torch.is_grad_enabled()) or not (query.requires_grad or key.requires_grad or value.requires_grad)
    if inference:
        output, _ = _fused_neighborhood_attention_fused_forward_ct(query, key, value, kernel_size, dilation, scale)
        return output
    return _FusedNeighborhoodAttentionFunctionCT.apply(query, key, value, kernel_size, dilation, scale)
