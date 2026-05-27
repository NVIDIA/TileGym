// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Multi-head Latent Attention (MLA) - Tile C++ Implementation
 *
 *
 * MLA is a specialized attention variant used in DeepSeek models that uses
 * compressed key-value projections with separate position embeddings.
 *
 * Computation:
 *   QK = Q @ K^T + QPE @ KPE^T  (attention scores with position contribution)
 *   P = softmax(QK * scale)
 *   Out = P @ V
 *
 * Key optimizations:
 * - Uses tensor_span + partition_view for all memory access
 * - All dimensions are template parameters
 * - Uses ct::irange for loops
 * - FP32 accumulator with native type inputs
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// 1/ln(2) for exp2 optimization
constexpr float MLA_INV_LOG_2 = 1.442695040888963f;
constexpr float MLA_NEG_INF = -1.0e6f;

/**
 * Prefill MLA Forward Kernel
 */
template<typename T, int B, int H, int H_KV, int S_QO, int S_KV,
         int TILE_D, int TILE_KPE, int TILE_M, int TILE_N, int QUERY_GROUP_SIZE, bool IS_CAUSAL>
__tile_global__ void prefill_mla_kernel(
    const T* __restrict__ _Q_ptr,
    const T* __restrict__ _QPE_ptr,
    const T* __restrict__ _K_ptr,
    const T* __restrict__ _KPE_ptr,
    const T* __restrict__ _V_ptr,
    T* __restrict__ _Out_ptr,
    float sm_scale
) {
    namespace ct = cuda::tiles;

    // Apply alignment hints
    const T* Q_ptr = ct::assume_aligned<16>(_Q_ptr);
    const T* QPE_ptr = ct::assume_aligned<16>(_QPE_ptr);
    const T* K_ptr = ct::assume_aligned<16>(_K_ptr);
    const T* KPE_ptr = ct::assume_aligned<16>(_KPE_ptr);
    const T* V_ptr = ct::assume_aligned<16>(_V_ptr);
    T* Out_ptr = ct::assume_aligned<16>(_Out_ptr);

    int bid_x = ct::bid().x;  // Query block index
    int bid_y = ct::bid().y;  // Batch * Head index
    int batch_idx = bid_y / H;
    int head_idx = bid_y % H;

    // For grouped-query attention
    int off_kv_h;
    if constexpr (QUERY_GROUP_SIZE > 0) {
        off_kv_h = head_idx / QUERY_GROUP_SIZE;
    } else {
        off_kv_h = head_idx;
    }

    // Scale for exp2 optimization
    float qk_scale = sm_scale * MLA_INV_LOG_2;

    using i32xM = ct::tile<int, ct::shape<TILE_M>>;
    using i32xN = ct::tile<int, ct::shape<TILE_N>>;
    auto offs_m_base = ct::iota<i32xM>();  // [0, 1, 2, ..., TILE_M-1]
    auto offs_n_base = ct::iota<i32xN>();  // [0, 1, 2, ..., TILE_N-1]

    // Add block offset to get actual positions
    auto offs_m = offs_m_base + ct::full<i32xM>(bid_x * TILE_M);

    // Reshape for 2D masking: offs_m is [TILE_M, 1], offs_n is [1, TILE_N]
    auto offs_m_2d = ct::reshape(offs_m, ct::shape<TILE_M, 1>{});
    auto offs_n_2d = ct::reshape(offs_n_base, ct::shape<1, TILE_N>{});

    // Initialize online softmax state
    using f32xM = ct::tile<float, ct::shape<TILE_M>>;
    using f32xMxD = ct::tile<float, ct::shape<TILE_M, TILE_D>>;
    using f32xMxN = ct::tile<float, ct::shape<TILE_M, TILE_N>>;

    auto m_i = ct::full<f32xM>(-INFINITY);
    auto l_i = ct::full<f32xM>(1.0f);
    auto acc = ct::zeros<f32xMxD>();

    auto Q_span = ct::tensor_span{Q_ptr,  ct::extents<uint32_t, B, H, S_QO, TILE_D>{}};
    auto Q_view = ct::partition_view(Q_span, ct::shape<1, 1, TILE_M, TILE_D>{});

    auto QPE_span = ct::tensor_span{QPE_ptr, ct::extents<uint32_t, B, H, S_QO, TILE_KPE>{}};
    auto QPE_view = ct::partition_view(QPE_span, ct::shape<1, 1, TILE_M, TILE_KPE>{});

    // K: [B, H_KV, S_KV, TILE_D]
    auto K_span = ct::tensor_span{K_ptr,  ct::extents<uint32_t, B, H_KV, S_KV, TILE_D>{}};
    auto K_view = ct::partition_view(K_span, ct::shape<1, 1, TILE_N, TILE_D>{});

    // KPE: [B, 1, S_KV, TILE_KPE]
    auto KPE_span = ct::tensor_span{KPE_ptr, ct::extents<uint32_t, B, 1, S_KV, TILE_KPE>{}};
    auto KPE_view = ct::partition_view(KPE_span, ct::shape<1, 1, TILE_N, TILE_KPE>{});

    auto V_span = ct::tensor_span{V_ptr,  ct::extents<uint32_t, B, H_KV, S_KV, TILE_D>{}};
    auto V_view = ct::partition_view(V_span, ct::shape<1, 1, TILE_N, TILE_D>{});

    // Load Q and QPE once (reused across all K blocks)
    auto q_4d = Q_view.load(batch_idx, head_idx, bid_x, 0);
    auto q = ct::reshape(q_4d, ct::shape<TILE_M, TILE_D>{});

    auto qpe_4d = QPE_view.load(batch_idx, head_idx, bid_x, 0);
    auto qpe = ct::reshape(qpe_4d, ct::shape<TILE_M, TILE_KPE>{});

    // Compute loop bounds based on causal/non-causal mode
    constexpr int num_k_blocks = (S_KV + TILE_N - 1) / TILE_N;
    int hi;
    int mask_start;
    if constexpr (IS_CAUSAL) {
        int start_m = bid_x;
        mask_start = (start_m * TILE_M) / TILE_N;
        hi = ((start_m + 1) * TILE_M + TILE_N - 1) / TILE_N;
        if (hi > num_k_blocks) hi = num_k_blocks;
    } else {
        // Non-causal: process all K/V blocks
        mask_start = num_k_blocks;  // Never trigger causal mask
        hi = num_k_blocks;
    }

    // ============================================================
    // Body loop (unmasked).  For non-causal mask_start == num_k_blocks
    // so this loop covers all K/V blocks; for causal it stops at the
    // first row that needs masking.
    // ============================================================
    for (auto j : ct::irange(0, mask_start)) {
        auto k_4d_T = ct::permute(K_view.load(batch_idx, off_kv_h, j, 0),
                                  ct::dimension_map<0, 1, 3, 2>{});
        auto k = ct::reshape(k_4d_T, ct::shape<TILE_D, TILE_N>{});

        auto qk = ct::mma(q, k, ct::zeros<f32xMxN>());

        auto kpe_4d_T = ct::permute(KPE_view.load(batch_idx, 0, j, 0),
                                    ct::dimension_map<0, 1, 3, 2>{});
        auto kpe_tile = ct::reshape(kpe_4d_T, ct::shape<TILE_KPE, TILE_N>{});

        qk = ct::mma(qpe, kpe_tile, qk);

        // NO mask in the body loop.

        auto qk_max    = ct::reduce_max<1>(qk);
        auto qk_max_1d = ct::reshape(qk_max, ct::shape<TILE_M>{});
        auto m_ij      = ct::max(m_i, qk_max_1d * qk_scale);

        auto m_ij_2d   = ct::reshape(m_ij, ct::shape<TILE_M, 1>{});
        auto qk_scaled = qk * qk_scale - m_ij_2d;

        auto p = ct::exp2(qk_scaled, ct::round_subnormals_to_zero_t{});

        auto p_sum = ct::sum<1>(p);
        auto l_ij  = ct::reshape(p_sum, ct::shape<TILE_M>{});

        auto alpha = ct::exp2(m_i - m_ij, ct::round_subnormals_to_zero_t{});
        l_i = l_i * alpha + l_ij;

        auto alpha_2d = ct::reshape(alpha, ct::shape<TILE_M, 1>{});
        acc = acc * alpha_2d;

        auto v_4d = V_view.load(batch_idx, off_kv_h, j, 0);
        auto v    = ct::reshape(v_4d, ct::shape<TILE_N, TILE_D>{});

        auto p_cast = ct::element_cast<T>(p);
        acc  = ct::mma(p_cast, v, acc);
        m_i  = m_ij;
    }

    // ============================================================
    // Tail loop (masked when IS_CAUSAL).
    // ============================================================
    for (auto j : ct::irange(mask_start, hi)) {
        int curr_n = j * TILE_N;

        auto k_4d_T = ct::permute(K_view.load(batch_idx, off_kv_h, j, 0),
                                  ct::dimension_map<0, 1, 3, 2>{});
        auto k = ct::reshape(k_4d_T, ct::shape<TILE_D, TILE_N>{});

        auto qk = ct::mma(q, k, ct::zeros<f32xMxN>());

        auto kpe_4d_T = ct::permute(KPE_view.load(batch_idx, 0, j, 0),
                                    ct::dimension_map<0, 1, 3, 2>{});
        auto kpe_tile = ct::reshape(kpe_4d_T, ct::shape<TILE_KPE, TILE_N>{});

        qk = ct::mma(qpe, kpe_tile, qk);

        if constexpr (IS_CAUSAL) {
            auto curr_n_tile    = ct::full<i32xN>(curr_n);
            auto n_positions    = curr_n_tile + offs_n_base;
            auto n_positions_2d = ct::reshape(n_positions, ct::shape<1, TILE_N>{});
            auto mask           = offs_m_2d >= n_positions_2d;
            qk = ct::select(mask, qk, ct::full<f32xMxN>(MLA_NEG_INF));
        }

        auto qk_max    = ct::reduce_max<1>(qk);
        auto qk_max_1d = ct::reshape(qk_max, ct::shape<TILE_M>{});
        auto m_ij      = ct::max(m_i, qk_max_1d * qk_scale);

        auto m_ij_2d   = ct::reshape(m_ij, ct::shape<TILE_M, 1>{});
        auto qk_scaled = qk * qk_scale - m_ij_2d;

        auto p = ct::exp2(qk_scaled, ct::round_subnormals_to_zero_t{});

        auto p_sum = ct::sum<1>(p);
        auto l_ij  = ct::reshape(p_sum, ct::shape<TILE_M>{});

        auto alpha = ct::exp2(m_i - m_ij, ct::round_subnormals_to_zero_t{});
        l_i = l_i * alpha + l_ij;

        auto alpha_2d = ct::reshape(alpha, ct::shape<TILE_M, 1>{});
        acc = acc * alpha_2d;

        auto v_4d = V_view.load(batch_idx, off_kv_h, j, 0);
        auto v    = ct::reshape(v_4d, ct::shape<TILE_N, TILE_D>{});

        auto p_cast = ct::element_cast<T>(p);
        acc  = ct::mma(p_cast, v, acc);
        m_i  = m_ij;
    }

    auto l_i_2d = ct::reshape(l_i, ct::shape<TILE_M, 1>{});
    acc = ct::div(acc, l_i_2d,
                  ct::round_approximate_t{},
                  ct::round_subnormals_to_zero_t{});

    // Convert to output type and store
    auto result = ct::element_cast<T>(acc);
    auto result_4d = ct::reshape(result, ct::shape<1, 1, TILE_M, TILE_D>{});

    auto Out_span = ct::tensor_span{Out_ptr, ct::extents<uint32_t, B, H, S_QO, TILE_D>{}};
    auto Out_view = ct::partition_view(Out_span, ct::shape<1, 1, TILE_M, TILE_D>{});
    Out_view.store(result_4d, batch_idx, head_idx, bid_x, 0);
}
