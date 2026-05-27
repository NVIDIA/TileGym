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
 * Tensor layouts (no host-side reshape of q/k needed):
 *   q: [batch, n_q_heads, seq_len, head_dim]
 *   k: [batch, n_kv_heads, seq_len, head_dim]
 *   cos/sin: [cos_batch, seq_len, 2, half_rope_dim]  (host wrapper reshapes)
 */

#pragma once

#include <cuda_tile.h>
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
 *
 * Each program handles one (batch, seq) position, processing all heads in parallel.
 */
template<typename T, int BATCH, int Q_HEADS, int K_HEADS,
         int BLOCK_QH, int BLOCK_KH, int BLOCK_HD,
         int HALF_ROPE_DIM, int HEAD_DIM, int COS_BS, int SEQ_LEN>
__tile_global__ void rope_kernel(
    T* __restrict__ q,    // [batch, n_q_heads, seq_len, head_dim]
    T* __restrict__ k,    // [batch, n_kv_heads, seq_len, head_dim]
    T* __restrict__ cos,  // [cos_batch, seq_len, 2, half_rope_dim]
    T* __restrict__ sin   // [cos_batch, seq_len, 2, half_rope_dim]
) {
    namespace ct = cuda::tiles;

    q = ct::assume_aligned<16>(q);
    k = ct::assume_aligned<16>(k);
    cos = ct::assume_aligned<16>(cos);
    sin = ct::assume_aligned<16>(sin);

    int pid = ct::bid().x;
    int batch_idx = pid / SEQ_LEN;
    int row_idx = pid % SEQ_LEN;
    int cos_batch_idx = (COS_BS == 1) ? 0 : batch_idx;

    // Load cos and sin values (first half only; second half is identical under
    // the duplicate-frequency layout).
    auto pCos = ct::partition_view(ct::tensor_span{cos, ct::extents{COS_BS, SEQ_LEN, 2, HALF_ROPE_DIM}},
                                   ct::shape<1, 1, 1, BLOCK_HD>{});
    auto cos_loaded = pCos.load(cos_batch_idx, row_idx, 0, 0);
    auto cos_row = ct::reshape<ct::shape<1, BLOCK_HD>>(cos_loaded);

    auto pSin = ct::partition_view(ct::tensor_span{sin, ct::extents{COS_BS, SEQ_LEN, 2, HALF_ROPE_DIM}},
                                   ct::shape<1, 1, 1, BLOCK_HD>{});
    auto sin_loaded = pSin.load(cos_batch_idx, row_idx, 0, 0);
    auto sin_row = ct::reshape<ct::shape<1, BLOCK_HD>>(sin_loaded);

    // Process Q tensor - 4D view over head_dim. Tile indices 0 and 1 along the
    // head_dim axis cover [0:2*BLOCK_HD) == [0:rope_dim); any elements at
    // [rope_dim:HEAD_DIM) are not accessed and pass through unchanged (in-place).
    auto pQ = ct::partition_view(ct::tensor_span{q, ct::extents{BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM}},
                                 ct::shape<1, BLOCK_QH, 1, BLOCK_HD>{});
    auto q_tile_1_loaded = pQ.load(batch_idx, 0, row_idx, 0);
    auto q_tile_1 = ct::reshape<ct::shape<BLOCK_QH, BLOCK_HD>>(q_tile_1_loaded);

    auto q_tile_2_loaded = pQ.load(batch_idx, 0, row_idx, 1);
    auto q_tile_2 = ct::reshape<ct::shape<BLOCK_QH, BLOCK_HD>>(q_tile_2_loaded);

    auto cos_bcast_q = ct::broadcast<ct::shape<BLOCK_QH, BLOCK_HD>>(cos_row);
    auto sin_bcast_q = ct::broadcast<ct::shape<BLOCK_QH, BLOCK_HD>>(sin_row);

    // y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    auto new_q_tile_1 = q_tile_1 * cos_bcast_q - q_tile_2 * sin_bcast_q;
    auto new_q_tile_2 = q_tile_2 * cos_bcast_q + q_tile_1 * sin_bcast_q;

    auto new_q_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_QH, 1, BLOCK_HD>>(new_q_tile_1);
    auto new_q_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_QH, 1, BLOCK_HD>>(new_q_tile_2);

    pQ.store(new_q_tile_1_reshaped, batch_idx, 0, row_idx, 0);
    pQ.store(new_q_tile_2_reshaped, batch_idx, 0, row_idx, 1);

    // Process K tensor
    auto pK = ct::partition_view(ct::tensor_span{k, ct::extents{BATCH, K_HEADS, SEQ_LEN, HEAD_DIM}},
                                 ct::shape<1, BLOCK_KH, 1, BLOCK_HD>{});
    auto k_tile_1_loaded = pK.load(batch_idx, 0, row_idx, 0);
    auto k_tile_1 = ct::reshape<ct::shape<BLOCK_KH, BLOCK_HD>>(k_tile_1_loaded);

    auto k_tile_2_loaded = pK.load(batch_idx, 0, row_idx, 1);
    auto k_tile_2 = ct::reshape<ct::shape<BLOCK_KH, BLOCK_HD>>(k_tile_2_loaded);

    auto cos_bcast_k = ct::broadcast<ct::shape<BLOCK_KH, BLOCK_HD>>(cos_row);
    auto sin_bcast_k = ct::broadcast<ct::shape<BLOCK_KH, BLOCK_HD>>(sin_row);

    auto new_k_tile_1 = k_tile_1 * cos_bcast_k - k_tile_2 * sin_bcast_k;
    auto new_k_tile_2 = k_tile_2 * cos_bcast_k + k_tile_1 * sin_bcast_k;

    auto new_k_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_KH, 1, BLOCK_HD>>(new_k_tile_1);
    auto new_k_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_KH, 1, BLOCK_HD>>(new_k_tile_2);

    pK.store(new_k_tile_1_reshaped, batch_idx, 0, row_idx, 0);
    pK.store(new_k_tile_2_reshaped, batch_idx, 0, row_idx, 1);
}


/**
 * RoPE backward kernel - processes all heads at once using 2D tiles.
 */
template<typename T, int BATCH, int Q_HEADS, int K_HEADS,
         int BLOCK_QH, int BLOCK_KH, int BLOCK_HD,
         int HALF_ROPE_DIM, int HEAD_DIM, int COS_BS, int SEQ_LEN>
__tile_global__ void rope_backward_kernel(
    T* __restrict__ dq,   // [batch, n_q_heads, seq_len, head_dim]
    T* __restrict__ dk,   // [batch, n_kv_heads, seq_len, head_dim]
    T* __restrict__ cos,  // [cos_batch, seq_len, 2, half_rope_dim]
    T* __restrict__ sin   // [cos_batch, seq_len, 2, half_rope_dim]
) {
    namespace ct = cuda::tiles;

    dq = ct::assume_aligned<16>(dq);
    dk = ct::assume_aligned<16>(dk);
    cos = ct::assume_aligned<16>(cos);
    sin = ct::assume_aligned<16>(sin);

    int pid = ct::bid().x;
    int batch_idx = pid / SEQ_LEN;
    int row_idx = pid % SEQ_LEN;
    int cos_batch_idx = (COS_BS == 1) ? 0 : batch_idx;

    auto pCos = ct::partition_view(ct::tensor_span{cos, ct::extents{COS_BS, SEQ_LEN, 2, HALF_ROPE_DIM}},
                                   ct::shape<1, 1, 1, BLOCK_HD>{});
    auto cos_loaded = pCos.load(cos_batch_idx, row_idx, 0, 0);
    auto cos_row = ct::reshape<ct::shape<1, BLOCK_HD>>(cos_loaded);

    auto pSin = ct::partition_view(ct::tensor_span{sin, ct::extents{COS_BS, SEQ_LEN, 2, HALF_ROPE_DIM}},
                                   ct::shape<1, 1, 1, BLOCK_HD>{});
    auto sin_loaded = pSin.load(cos_batch_idx, row_idx, 0, 0);
    auto sin_row = ct::reshape<ct::shape<1, BLOCK_HD>>(sin_loaded);

    // Process dQ tensor
    auto pDQ = ct::partition_view(ct::tensor_span{dq, ct::extents{BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM}},
                                  ct::shape<1, BLOCK_QH, 1, BLOCK_HD>{});
    auto dq_tile_1_loaded = pDQ.load(batch_idx, 0, row_idx, 0);
    auto dq_tile_1 = ct::reshape<ct::shape<BLOCK_QH, BLOCK_HD>>(dq_tile_1_loaded);

    auto dq_tile_2_loaded = pDQ.load(batch_idx, 0, row_idx, 1);
    auto dq_tile_2 = ct::reshape<ct::shape<BLOCK_QH, BLOCK_HD>>(dq_tile_2_loaded);

    auto cos_bcast_q = ct::broadcast<ct::shape<BLOCK_QH, BLOCK_HD>>(cos_row);
    auto sin_bcast_q = ct::broadcast<ct::shape<BLOCK_QH, BLOCK_HD>>(sin_row);

    // Backward: dy = [dx1, dx2] * [cos, cos] + [dx2, -dx1] * [sin, sin]
    auto new_dq_tile_1 = dq_tile_1 * cos_bcast_q + dq_tile_2 * sin_bcast_q;
    auto new_dq_tile_2 = dq_tile_2 * cos_bcast_q - dq_tile_1 * sin_bcast_q;

    auto new_dq_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_QH, 1, BLOCK_HD>>(new_dq_tile_1);
    auto new_dq_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_QH, 1, BLOCK_HD>>(new_dq_tile_2);

    pDQ.store(new_dq_tile_1_reshaped, batch_idx, 0, row_idx, 0);
    pDQ.store(new_dq_tile_2_reshaped, batch_idx, 0, row_idx, 1);

    // Process dK tensor
    auto pDK = ct::partition_view(ct::tensor_span{dk, ct::extents{BATCH, K_HEADS, SEQ_LEN, HEAD_DIM}},
                                  ct::shape<1, BLOCK_KH, 1, BLOCK_HD>{});
    auto dk_tile_1_loaded = pDK.load(batch_idx, 0, row_idx, 0);
    auto dk_tile_1 = ct::reshape<ct::shape<BLOCK_KH, BLOCK_HD>>(dk_tile_1_loaded);

    auto dk_tile_2_loaded = pDK.load(batch_idx, 0, row_idx, 1);
    auto dk_tile_2 = ct::reshape<ct::shape<BLOCK_KH, BLOCK_HD>>(dk_tile_2_loaded);

    auto cos_bcast_k = ct::broadcast<ct::shape<BLOCK_KH, BLOCK_HD>>(cos_row);
    auto sin_bcast_k = ct::broadcast<ct::shape<BLOCK_KH, BLOCK_HD>>(sin_row);

    auto new_dk_tile_1 = dk_tile_1 * cos_bcast_k + dk_tile_2 * sin_bcast_k;
    auto new_dk_tile_2 = dk_tile_2 * cos_bcast_k - dk_tile_1 * sin_bcast_k;

    auto new_dk_tile_1_reshaped = ct::reshape<ct::shape<1, BLOCK_KH, 1, BLOCK_HD>>(new_dk_tile_1);
    auto new_dk_tile_2_reshaped = ct::reshape<ct::shape<1, BLOCK_KH, 1, BLOCK_HD>>(new_dk_tile_2);

    pDK.store(new_dk_tile_1_reshaped, batch_idx, 0, row_idx, 0);
    pDK.store(new_dk_tile_2_reshaped, batch_idx, 0, row_idx, 1);
}
