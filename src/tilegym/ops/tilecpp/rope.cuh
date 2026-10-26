// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ RoPE (Rotary Position Embedding) Kernel
 * Implements RoPE forward and backward passes.
 *
 * Forward:  y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
 * Backward: dy = [dx1, dx2] * [cos, cos] + [dx2, -dx1] * [sin, sin]
 *
 * Tensor layouts (no host-side copy of q/k/cos/sin needed):
 *   q: [batch, n_q_heads, seq_len, head_dim]
 *   k: [batch, n_kv_heads, seq_len, head_dim]
 *   cos/sin: [cos_batch, seq_len, rope_dim] or [cos_batch, seq_len, rope_dim / 2]
 *
 * The first rope_dim elements of q/k are viewed as [..., 2, half_rope_dim], so
 * index 0 / 1 on the half axis selects exactly [0:half_rope_dim) /
 * [half_rope_dim:rope_dim) for any half_rope_dim. BLOCK_HD / BLOCK_QH / BLOCK_KH
 * are padded tile sizes, so every view is accessed with load_masked /
 * store_masked: out-of-bounds lanes are zero-filled on load and dropped on store.
 * Only the first half_rope_dim elements of cos/sin are read, which covers both
 * the duplicate-frequency (rope_dim) and the half-frequency (rope_dim / 2) layout.
 */

#pragma once

#include <cuda_tile.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * RoPE forward kernel - processes all heads at once using 2D tiles.
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BATCH: Batch size
 *   Q_HEADS: Number of query heads
 *   K_HEADS: Number of key heads
 *   BLOCK_QH: Tile size for query heads (power of 2, >= Q_HEADS)
 *   BLOCK_KH: Tile size for key heads (power of 2, >= K_HEADS)
 *   BLOCK_HD: Tile size for half rope dimension (power of 2, >= HALF_ROPE_DIM)
 *   HALF_ROPE_DIM: half of the rope dimension (rope_dim / 2)
 *   HEAD_DIM: actual last dim of q/k (>= 2*HALF_ROPE_DIM)
 *   COS_BS: Cos/sin batch size (1 for broadcast, BATCH otherwise)
 *   SEQ_LEN: Sequence length
 *   Q_STRIDE_* / K_STRIDE_*: q/k strides (head_dim stride is 1)
 *   COS_STRIDE_* / SIN_STRIDE_*: cos/sin strides
 *   Q_ALIGN / K_ALIGN / COS_ALIGN / SIN_ALIGN: byte alignment of the q/k/cos/sin
 *     pointers (at most 16)
 *
 * Each program handles one (batch, seq) position, processing all heads in parallel.
 */
template<typename T, typename CT, typename ST, int BATCH, int Q_HEADS, int K_HEADS,
         int BLOCK_QH, int BLOCK_KH, int BLOCK_HD,
         int HALF_ROPE_DIM, int HEAD_DIM, int COS_BS, int SEQ_LEN,
         int Q_STRIDE_B, int Q_STRIDE_H, int Q_STRIDE_S,
         int K_STRIDE_B, int K_STRIDE_H, int K_STRIDE_S,
         int COS_STRIDE_B, int COS_STRIDE_S, int COS_STRIDE_D,
         int SIN_STRIDE_B, int SIN_STRIDE_S, int SIN_STRIDE_D,
         int Q_ALIGN, int K_ALIGN, int COS_ALIGN, int SIN_ALIGN>
__tile_global__ void rope_kernel(
    T* __restrict__ q,    // [batch, n_q_heads, seq_len, head_dim]
    T* __restrict__ k,    // [batch, n_kv_heads, seq_len, head_dim]
    const CT* __restrict__ cos,  // [cos_batch, seq_len, rope_dim or half_rope_dim]
    const ST* __restrict__ sin   // [cos_batch, seq_len, rope_dim or half_rope_dim]
) {
    namespace ct = cuda::tiles;
    using AuxT  = std::conditional_t<(sizeof(ST) > sizeof(CT)), ST, CT>;
    using CompT = std::conditional_t<(sizeof(AuxT) > sizeof(T)), AuxT, T>;

    q = ct::assume_aligned<Q_ALIGN>(q);
    k = ct::assume_aligned<K_ALIGN>(k);
    cos = ct::assume_aligned<COS_ALIGN>(cos);
    sin = ct::assume_aligned<SIN_ALIGN>(sin);

    int pid = ct::bid().x;
    int batch_idx = pid / SEQ_LEN;
    int row_idx = pid % SEQ_LEN;
    int cos_batch_idx = (COS_BS == 1) ? 0 : batch_idx;

    // Load cos and sin values (first half only; the second half, if present, is
    // identical under the duplicate-frequency layout). The (batch, seq) row is
    // addressed by pointer arithmetic, so broadcast (stride-0) dims need no copy.
    auto pCos = ct::partition_view(
        ct::tensor_span{cos + cos_batch_idx * COS_STRIDE_B + row_idx * COS_STRIDE_S, ct::layout_strided_mapping{
            ct::extents<uint32_t, HALF_ROPE_DIM>{}, ct::extents<uint32_t, COS_STRIDE_D>{}}},
        ct::shape<BLOCK_HD>{});
    auto cos_loaded = pCos.load_masked(0);
    auto cos_row = ct::element_cast<CompT>(ct::reshape<ct::shape<1, BLOCK_HD>>(cos_loaded));

    auto pSin = ct::partition_view(
        ct::tensor_span{sin + cos_batch_idx * SIN_STRIDE_B + row_idx * SIN_STRIDE_S, ct::layout_strided_mapping{
            ct::extents<uint32_t, HALF_ROPE_DIM>{}, ct::extents<uint32_t, SIN_STRIDE_D>{}}},
        ct::shape<BLOCK_HD>{});
    auto sin_loaded = pSin.load_masked(0);
    auto sin_row = ct::element_cast<CompT>(ct::reshape<ct::shape<1, BLOCK_HD>>(sin_loaded));

    // Process Q tensor - 5D [..., 2, HALF_ROPE_DIM] view over the first rope_dim
    // elements of head_dim; any elements at [rope_dim:HEAD_DIM) are not accessed
    // and pass through unchanged (in-place).
    auto pQ = ct::partition_view(
        ct::tensor_span{q, ct::layout_strided_mapping{
            ct::extents<uint32_t, BATCH, Q_HEADS, SEQ_LEN, 2, HALF_ROPE_DIM>{},
            ct::extents<uint32_t, Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_S, HALF_ROPE_DIM, 1>{}}},
        ct::shape<1, BLOCK_QH, 1, 1, BLOCK_HD>{});
    auto q_tile_1_loaded = pQ.load_masked(batch_idx, 0, row_idx, 0, 0);
    auto q_tile_1 = ct::element_cast<CompT>(ct::reshape<ct::shape<BLOCK_QH, BLOCK_HD>>(q_tile_1_loaded));

    auto q_tile_2_loaded = pQ.load_masked(batch_idx, 0, row_idx, 1, 0);
    auto q_tile_2 = ct::element_cast<CompT>(ct::reshape<ct::shape<BLOCK_QH, BLOCK_HD>>(q_tile_2_loaded));

    auto cos_bcast_q = ct::broadcast<ct::shape<BLOCK_QH, BLOCK_HD>>(cos_row);
    auto sin_bcast_q = ct::broadcast<ct::shape<BLOCK_QH, BLOCK_HD>>(sin_row);

    // y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    auto new_q_tile_1 = q_tile_1 * cos_bcast_q - q_tile_2 * sin_bcast_q;
    auto new_q_tile_2 = q_tile_2 * cos_bcast_q + q_tile_1 * sin_bcast_q;

    auto new_q_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_QH, 1, 1, BLOCK_HD>>(ct::element_cast<T>(new_q_tile_1));
    auto new_q_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_QH, 1, 1, BLOCK_HD>>(ct::element_cast<T>(new_q_tile_2));

    pQ.store_masked(new_q_tile_1_reshaped, batch_idx, 0, row_idx, 0, 0);
    pQ.store_masked(new_q_tile_2_reshaped, batch_idx, 0, row_idx, 1, 0);

    // Process K tensor
    auto pK = ct::partition_view(
        ct::tensor_span{k, ct::layout_strided_mapping{
            ct::extents<uint32_t, BATCH, K_HEADS, SEQ_LEN, 2, HALF_ROPE_DIM>{},
            ct::extents<uint32_t, K_STRIDE_B, K_STRIDE_H, K_STRIDE_S, HALF_ROPE_DIM, 1>{}}},
        ct::shape<1, BLOCK_KH, 1, 1, BLOCK_HD>{});
    auto k_tile_1_loaded = pK.load_masked(batch_idx, 0, row_idx, 0, 0);
    auto k_tile_1 = ct::element_cast<CompT>(ct::reshape<ct::shape<BLOCK_KH, BLOCK_HD>>(k_tile_1_loaded));

    auto k_tile_2_loaded = pK.load_masked(batch_idx, 0, row_idx, 1, 0);
    auto k_tile_2 = ct::element_cast<CompT>(ct::reshape<ct::shape<BLOCK_KH, BLOCK_HD>>(k_tile_2_loaded));

    auto cos_bcast_k = ct::broadcast<ct::shape<BLOCK_KH, BLOCK_HD>>(cos_row);
    auto sin_bcast_k = ct::broadcast<ct::shape<BLOCK_KH, BLOCK_HD>>(sin_row);

    auto new_k_tile_1 = k_tile_1 * cos_bcast_k - k_tile_2 * sin_bcast_k;
    auto new_k_tile_2 = k_tile_2 * cos_bcast_k + k_tile_1 * sin_bcast_k;

    auto new_k_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_KH, 1, 1, BLOCK_HD>>(ct::element_cast<T>(new_k_tile_1));
    auto new_k_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_KH, 1, 1, BLOCK_HD>>(ct::element_cast<T>(new_k_tile_2));

    pK.store_masked(new_k_tile_1_reshaped, batch_idx, 0, row_idx, 0, 0);
    pK.store_masked(new_k_tile_2_reshaped, batch_idx, 0, row_idx, 1, 0);
}


/**
 * RoPE backward kernel - processes all heads at once using 2D tiles.
 */
template<typename T, typename CT, typename ST, int BATCH, int Q_HEADS, int K_HEADS,
         int BLOCK_QH, int BLOCK_KH, int BLOCK_HD,
         int HALF_ROPE_DIM, int HEAD_DIM, int COS_BS, int SEQ_LEN,
         int COS_STRIDE_B, int COS_STRIDE_S, int COS_STRIDE_D,
         int SIN_STRIDE_B, int SIN_STRIDE_S, int SIN_STRIDE_D,
         int Q_ALIGN, int K_ALIGN, int COS_ALIGN, int SIN_ALIGN>
__tile_global__ void rope_backward_kernel(
    T* __restrict__ dq,   // [batch, n_q_heads, seq_len, head_dim] (contiguous)
    T* __restrict__ dk,   // [batch, n_kv_heads, seq_len, head_dim] (contiguous)
    const CT* __restrict__ cos,  // [cos_batch, seq_len, rope_dim or half_rope_dim]
    const ST* __restrict__ sin   // [cos_batch, seq_len, rope_dim or half_rope_dim]
) {
    namespace ct = cuda::tiles;
    using AuxT  = std::conditional_t<(sizeof(ST) > sizeof(CT)), ST, CT>;
    using CompT = std::conditional_t<(sizeof(AuxT) > sizeof(T)), AuxT, T>;

    dq = ct::assume_aligned<Q_ALIGN>(dq);
    dk = ct::assume_aligned<K_ALIGN>(dk);
    cos = ct::assume_aligned<COS_ALIGN>(cos);
    sin = ct::assume_aligned<SIN_ALIGN>(sin);

    int pid = ct::bid().x;
    int batch_idx = pid / SEQ_LEN;
    int row_idx = pid % SEQ_LEN;
    int cos_batch_idx = (COS_BS == 1) ? 0 : batch_idx;

    // The (batch, seq) row is addressed by pointer arithmetic, so broadcast
    // (stride-0) cos/sin batch or seq dims need no copy.
    auto pCos = ct::partition_view(
        ct::tensor_span{cos + cos_batch_idx * COS_STRIDE_B + row_idx * COS_STRIDE_S, ct::layout_strided_mapping{
            ct::extents<uint32_t, HALF_ROPE_DIM>{}, ct::extents<uint32_t, COS_STRIDE_D>{}}},
        ct::shape<BLOCK_HD>{});
    auto cos_loaded = pCos.load_masked(0);
    auto cos_row = ct::element_cast<CompT>(ct::reshape<ct::shape<1, BLOCK_HD>>(cos_loaded));

    auto pSin = ct::partition_view(
        ct::tensor_span{sin + cos_batch_idx * SIN_STRIDE_B + row_idx * SIN_STRIDE_S, ct::layout_strided_mapping{
            ct::extents<uint32_t, HALF_ROPE_DIM>{}, ct::extents<uint32_t, SIN_STRIDE_D>{}}},
        ct::shape<BLOCK_HD>{});
    auto sin_loaded = pSin.load_masked(0);
    auto sin_row = ct::element_cast<CompT>(ct::reshape<ct::shape<1, BLOCK_HD>>(sin_loaded));

    // Process dQ tensor
    auto pDQ = ct::partition_view(
        ct::tensor_span{dq, ct::layout_strided_mapping{
            ct::extents<uint32_t, BATCH, Q_HEADS, SEQ_LEN, 2, HALF_ROPE_DIM>{},
            ct::extents<uint32_t, Q_HEADS * SEQ_LEN * HEAD_DIM, SEQ_LEN * HEAD_DIM, HEAD_DIM, HALF_ROPE_DIM, 1>{}}},
        ct::shape<1, BLOCK_QH, 1, 1, BLOCK_HD>{});
    auto dq_tile_1_loaded = pDQ.load_masked(batch_idx, 0, row_idx, 0, 0);
    auto dq_tile_1 = ct::element_cast<CompT>(ct::reshape<ct::shape<BLOCK_QH, BLOCK_HD>>(dq_tile_1_loaded));

    auto dq_tile_2_loaded = pDQ.load_masked(batch_idx, 0, row_idx, 1, 0);
    auto dq_tile_2 = ct::element_cast<CompT>(ct::reshape<ct::shape<BLOCK_QH, BLOCK_HD>>(dq_tile_2_loaded));

    auto cos_bcast_q = ct::broadcast<ct::shape<BLOCK_QH, BLOCK_HD>>(cos_row);
    auto sin_bcast_q = ct::broadcast<ct::shape<BLOCK_QH, BLOCK_HD>>(sin_row);

    // Backward: dy = [dx1, dx2] * [cos, cos] + [dx2, -dx1] * [sin, sin]
    auto new_dq_tile_1 = dq_tile_1 * cos_bcast_q + dq_tile_2 * sin_bcast_q;
    auto new_dq_tile_2 = dq_tile_2 * cos_bcast_q - dq_tile_1 * sin_bcast_q;

    auto new_dq_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_QH, 1, 1, BLOCK_HD>>(ct::element_cast<T>(new_dq_tile_1));
    auto new_dq_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_QH, 1, 1, BLOCK_HD>>(ct::element_cast<T>(new_dq_tile_2));

    pDQ.store_masked(new_dq_tile_1_reshaped, batch_idx, 0, row_idx, 0, 0);
    pDQ.store_masked(new_dq_tile_2_reshaped, batch_idx, 0, row_idx, 1, 0);

    // Process dK tensor
    auto pDK = ct::partition_view(
        ct::tensor_span{dk, ct::layout_strided_mapping{
            ct::extents<uint32_t, BATCH, K_HEADS, SEQ_LEN, 2, HALF_ROPE_DIM>{},
            ct::extents<uint32_t, K_HEADS * SEQ_LEN * HEAD_DIM, SEQ_LEN * HEAD_DIM, HEAD_DIM, HALF_ROPE_DIM, 1>{}}},
        ct::shape<1, BLOCK_KH, 1, 1, BLOCK_HD>{});
    auto dk_tile_1_loaded = pDK.load_masked(batch_idx, 0, row_idx, 0, 0);
    auto dk_tile_1 = ct::element_cast<CompT>(ct::reshape<ct::shape<BLOCK_KH, BLOCK_HD>>(dk_tile_1_loaded));

    auto dk_tile_2_loaded = pDK.load_masked(batch_idx, 0, row_idx, 1, 0);
    auto dk_tile_2 = ct::element_cast<CompT>(ct::reshape<ct::shape<BLOCK_KH, BLOCK_HD>>(dk_tile_2_loaded));

    auto cos_bcast_k = ct::broadcast<ct::shape<BLOCK_KH, BLOCK_HD>>(cos_row);
    auto sin_bcast_k = ct::broadcast<ct::shape<BLOCK_KH, BLOCK_HD>>(sin_row);

    auto new_dk_tile_1 = dk_tile_1 * cos_bcast_k + dk_tile_2 * sin_bcast_k;
    auto new_dk_tile_2 = dk_tile_2 * cos_bcast_k - dk_tile_1 * sin_bcast_k;

    auto new_dk_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_KH, 1, 1, BLOCK_HD>>(ct::element_cast<T>(new_dk_tile_1));
    auto new_dk_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_KH, 1, 1, BLOCK_HD>>(ct::element_cast<T>(new_dk_tile_2));

    pDK.store_masked(new_dk_tile_1_reshaped, batch_idx, 0, row_idx, 0, 0);
    pDK.store_masked(new_dk_tile_2_reshaped, batch_idx, 0, row_idx, 1, 0);
}
