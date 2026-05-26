// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr float FLASH_DECODE_INV_LOG_2 = 1.0f / 0.693147180559945309417232121458176568f;

/**
 * Optimized Attention Decode Kernel
 *
 * Key optimizations:
 * - BLOCK_N=256 tile size for better memory efficiency
 * - Native type MMA for tensor core utilization
 */
template<typename T, int B, int H_KV, int S_KV, int Q_PER_KV, int QUERY_GROUP_BLOCK_SIZE,
         int KV_LEN_PER_SPLIT, int HEAD_DIM, int BLOCK_N, int NUM_KV_SPLITS>
__tile_global__ void attention_decode_kernel_optimized(
    const T* __restrict__ Q_ptr,      // [B, H_KV, Q_PER_KV, HEAD_DIM]
    const T* __restrict__ K_ptr,      // [B, H_KV, S_KV, HEAD_DIM]
    const T* __restrict__ V_ptr,      // [B, H_KV, S_KV, HEAD_DIM]
    T* __restrict__ Out_ptr,          // [B, H_KV, Q_PER_KV, NUM_KV_SPLITS, HEAD_DIM]
    float* __restrict__ LSE_ptr,      // [B, H_KV, Q_PER_KV, NUM_KV_SPLITS]
    float softmax_scale
) {
    namespace ct = cuda::tiles;

    // Apply alignment hints
    Q_ptr = ct::assume_aligned<16>(Q_ptr);
    K_ptr = ct::assume_aligned<16>(K_ptr);
    V_ptr = ct::assume_aligned<16>(V_ptr);
    Out_ptr = ct::assume_aligned<16>(Out_ptr);
    LSE_ptr = ct::assume_aligned<16>(LSE_ptr);

    // Tile types - native type for tensor core MMA
    using T_QGxD = ct::tile<T, ct::shape<QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>>;
    using T_NxQG = ct::tile<T, ct::shape<BLOCK_N, QUERY_GROUP_BLOCK_SIZE>>;
    using T_DxN = ct::tile<T, ct::shape<HEAD_DIM, BLOCK_N>>;
    using T_NxD = ct::tile<T, ct::shape<BLOCK_N, HEAD_DIM>>;
    using T_4D_Tile = ct::tile<T, ct::shape<1, 1, BLOCK_N, HEAD_DIM>>;

    using f32_QG = ct::tile<float, ct::shape<QUERY_GROUP_BLOCK_SIZE>>;
    using f32_DxQG = ct::tile<float, ct::shape<HEAD_DIM, QUERY_GROUP_BLOCK_SIZE>>;
    using f32_NxQG = ct::tile<float, ct::shape<BLOCK_N, QUERY_GROUP_BLOCK_SIZE>>;
    using f32_NxD = ct::tile<float, ct::shape<BLOCK_N, HEAD_DIM>>;
    using i32_QG = ct::tile<int, ct::shape<QUERY_GROUP_BLOCK_SIZE>>;

    using i32_N = ct::tile<int, ct::shape<BLOCK_N>>;

    // Get program IDs
    int batch_id = ct::bid().x;
    int head_id = ct::bid().y;
    int split_id = ct::bid().z;

    float qk_scale = softmax_scale * FLASH_DECODE_INV_LOG_2;

    // Create tensor spans for structured memory access
    auto Q_span = ct::tensor_span{Q_ptr, ct::extents{B, H_KV, Q_PER_KV, HEAD_DIM}};
    auto Q_view = ct::partition_view(Q_span, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>{});

    auto K_span = ct::tensor_span{K_ptr, ct::extents{B, H_KV, S_KV, HEAD_DIM}};
    auto K_view = ct::partition_view(K_span, ct::shape<1, 1, BLOCK_N, HEAD_DIM>{});

    auto V_span = ct::tensor_span{V_ptr, ct::extents{B, H_KV, S_KV, HEAD_DIM}};
    auto V_view = ct::partition_view(V_span, ct::shape<1, 1, BLOCK_N, HEAD_DIM>{});

    // Load Q and transpose to [HEAD_DIM, QUERY_GROUP_BLOCK_SIZE]
    using Q_4D_Tile = ct::tile<T, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>>;
    using T_DxQG = ct::tile<T, ct::shape<HEAD_DIM, QUERY_GROUP_BLOCK_SIZE>>;
    Q_4D_Tile q_4d = Q_view.load(batch_id, head_id, 0, 0);
    auto q_2d = ct::reshape(q_4d, ct::shape<QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>{});
    // Keep Q in native type for tensor core K@Q MMA
    auto q_native = ct::transpose(q_2d);  // [HEAD_DIM, QUERY_GROUP_BLOCK_SIZE]

    // Calculate split bounds
    int start_idx = split_id * KV_LEN_PER_SPLIT;
    int end_idx = start_idx + KV_LEN_PER_SPLIT;
    if (end_idx > S_KV) end_idx = S_KV;

    auto m_i = ct::full<f32_QG>(-INFINITY);
    auto l_i = ct::full<f32_NxQG>(1.0f);
    auto acc = ct::zeros<f32_DxQG>();

    // Calculate iteration bounds
    int num_blocks = (end_idx - start_idx + BLOCK_N - 1) / BLOCK_N;
    int start_block = start_idx / BLOCK_N;

    // Offset generator for boundary masking
    auto offs_n = ct::iota<i32_N>();

    // Main attention loop
    for (auto block_idx : ct::irange(0, num_blocks)) {
        int kv_block = start_block + block_idx;
        int curr_n = kv_block * BLOCK_N;

        // Load K
        T_4D_Tile k_4d = K_view.load(batch_id, head_id, kv_block, 0);
        auto k = ct::reshape(k_4d, ct::shape<BLOCK_N, HEAD_DIM>{});

        // Compute qk = K @ Q -> [BLOCK_N, QUERY_GROUP_BLOCK_SIZE]
        // Use MMA with float32 accumulation for best tensor core utilization
        auto qk = ct::mma(k, q_native, ct::zeros<f32_NxQG>());

        // Apply boundary mask ONLY when necessary
        if (curr_n + BLOCK_N > S_KV) {
            auto n_indices = ct::full<i32_N>(curr_n) + offs_n;
            auto n_mask = n_indices < S_KV;
            auto n_mask_2d = ct::reshape(n_mask, ct::shape<BLOCK_N, 1>{});
            qk = ct::select(n_mask_2d, qk, ct::full<f32_NxQG>(-1.0e6f));
        }

        // Softmax computation
        auto qk_scaled = qk * qk_scale;
        auto qk_max_reduced = ct::reduce_max(qk_scaled, ct::integral_constant<0>{});
        auto m_ij = ct::max(m_i, ct::reshape(qk_max_reduced, ct::shape<QUERY_GROUP_BLOCK_SIZE>{}));

        // Compute p with exp2(qk_scaled - m_ij)
        auto m_ij_2d = ct::reshape(m_ij, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
        auto p = ct::exp2(qk_scaled - m_ij_2d);  // [BLOCK_N, QUERY_GROUP_BLOCK_SIZE]

        // Update l_i with 2D accumulation and scale acc
        auto alpha = ct::exp2(m_i - m_ij);
        auto alpha_2d = ct::reshape(alpha, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
        l_i = l_i * alpha_2d + p;
        acc = acc * alpha_2d;

        // Load V after softmax to interleave with compute
        T_4D_Tile v_4d = V_view.load(batch_id, head_id, kv_block, 0);
        auto v_2d = ct::reshape(v_4d, ct::shape<BLOCK_N, HEAD_DIM>{});
        auto v_t = ct::transpose(v_2d);  // [HEAD_DIM, BLOCK_N]

        // Cast to native type for tensor core MMA
        auto p_T = ct::element_cast<T>(p);
        auto v_t_T = ct::element_cast<T>(v_t);

        // Accumulate: acc += V^T @ p
        acc = ct::mma(v_t_T, p_T, acc);

        m_i = m_ij;
    }

    // Reduce l_i from 2D to 1D by summing over the N dimension
    auto l = ct::sum(l_i, ct::integral_constant<0>{});
    auto l_1d = ct::reshape(l, ct::shape<QUERY_GROUP_BLOCK_SIZE>{});

    auto l_2d = ct::reshape(l_1d, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
    auto acc_normalized = ct::div(acc, l_2d, ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});

    // Compute LSE
    auto lse = m_i + ct::log2(l_1d);

    // Transpose and cast for output
    auto acc_out = ct::transpose(acc_normalized);  // Use transpose instead of permute
    auto acc_out_T = ct::element_cast<T>(acc_out);

    // Store output using partition_view
    auto Out_span = ct::tensor_span{Out_ptr, ct::extents{B, H_KV, Q_PER_KV, NUM_KV_SPLITS, HEAD_DIM}};
    auto Out_view = ct::partition_view(Out_span, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1, HEAD_DIM>{});
    auto acc_out_5d = ct::reshape(acc_out_T, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1, HEAD_DIM>{});
    Out_view.store(acc_out_5d, batch_id, head_id, 0, split_id, 0);

    // Store LSE
    auto LSE_span = ct::tensor_span{LSE_ptr, ct::extents{B, H_KV, Q_PER_KV, NUM_KV_SPLITS}};
    auto LSE_view = ct::partition_view(LSE_span, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1>{});
    auto lse_4d = ct::reshape(lse, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1>{});
    LSE_view.store(lse_4d, batch_id, head_id, 0, split_id);
}
