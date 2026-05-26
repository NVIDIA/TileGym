// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Gemma Attention Decode (with soft cap + sliding window) + split-K reduce.
 *
 *
 * Layout mirrors flash_decode.cuh, adding:
 *   - runtime soft_cap (enabled by HAS_SOFT_CAP compile-time flag),
 *   - compile-time WINDOW_SIZE (>0 means sliding window; 0 means global).
 *
 * Grid: (batch, num_kv_heads, num_kv_splits)
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr float GEMMA_DECODE_INV_LOG_2 = 1.0f / 0.693147180559945309417232121458176568f;
constexpr float GEMMA_DECODE_LOG_2     = 0.693147180559945309417232121458176568f;

template<typename T,
         int B,
         int H_KV,
         int S_KV,
         int Q_PER_KV,
         int QUERY_GROUP_BLOCK_SIZE,
         int KV_LEN_PER_SPLIT,
         int HEAD_DIM,
         int BLOCK_N,
         int NUM_KV_SPLITS,
         int WINDOW_SIZE,
         bool HAS_SOFT_CAP,
         int occupancy>
[[ using cutile : hint(1000, occupancy=occupancy) ]]
__tile_global__ void gemma_attention_decode_kernel(
    const T* __restrict__ Q_ptr,      // [B, H_KV, Q_PER_KV, HEAD_DIM]
    const T* __restrict__ K_ptr,      // [B, H_KV, S_KV, HEAD_DIM]
    const T* __restrict__ V_ptr,      // [B, H_KV, S_KV, HEAD_DIM]
    T* __restrict__ Out_ptr,          // [B, H_KV, Q_PER_KV, NUM_KV_SPLITS, HEAD_DIM]
    float* __restrict__ LSE_ptr,      // [B, H_KV, Q_PER_KV, NUM_KV_SPLITS]
    float softmax_scale,
    float soft_cap
) {
    namespace ct = cuda::tiles;

    Q_ptr   = ct::assume_aligned<16>(Q_ptr);
    K_ptr   = ct::assume_aligned<16>(K_ptr);
    V_ptr   = ct::assume_aligned<16>(V_ptr);
    Out_ptr = ct::assume_aligned<16>(Out_ptr);
    LSE_ptr = ct::assume_aligned<16>(LSE_ptr);

    using T_NxD      = ct::tile<T, ct::shape<BLOCK_N, HEAD_DIM>>;
    using T_4D_Tile  = ct::tile<T, ct::shape<1, 1, BLOCK_N, HEAD_DIM>>;
    using Q_4D_Tile  = ct::tile<T, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>>;

    using f32_QG     = ct::tile<float, ct::shape<QUERY_GROUP_BLOCK_SIZE>>;
    using f32_DxQG   = ct::tile<float, ct::shape<HEAD_DIM, QUERY_GROUP_BLOCK_SIZE>>;
    using f32_NxQG   = ct::tile<float, ct::shape<BLOCK_N, QUERY_GROUP_BLOCK_SIZE>>;
    using i32_N      = ct::tile<int, ct::shape<BLOCK_N>>;

    int batch_id = ct::bid().x;
    int head_id  = ct::bid().y;
    int split_id = ct::bid().z;

    float qk_scale = softmax_scale * GEMMA_DECODE_INV_LOG_2;
    // Original sm_scale (pre INV_LOG_2) for the soft-cap branch.
    float sm_scale_orig = softmax_scale;
    float inv_soft_cap  = HAS_SOFT_CAP ? (1.0f / soft_cap) : 0.0f;

    auto Q_view = ct::partition_view(
        ct::tensor_span{Q_ptr, ct::extents{B, H_KV, Q_PER_KV, HEAD_DIM}},
        ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>{});
    auto K_view = ct::partition_view(
        ct::tensor_span{K_ptr, ct::extents{B, H_KV, S_KV, HEAD_DIM}},
        ct::shape<1, 1, BLOCK_N, HEAD_DIM>{});
    auto V_view = ct::partition_view(
        ct::tensor_span{V_ptr, ct::extents{B, H_KV, S_KV, HEAD_DIM}},
        ct::shape<1, 1, BLOCK_N, HEAD_DIM>{});

    // Load Q -> [HEAD_DIM, QUERY_GROUP_BLOCK_SIZE] in native T for tensor-core MMA.
    Q_4D_Tile q_4d = Q_view.load(batch_id, head_id, 0, 0);
    auto q_2d   = ct::reshape(q_4d, ct::shape<QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>{});
    auto q_nat  = ct::transpose(q_2d);  // [HEAD_DIM, QUERY_GROUP_BLOCK_SIZE]

    int start_idx = split_id * KV_LEN_PER_SPLIT;
    int end_idx   = start_idx + KV_LEN_PER_SPLIT;
    if (end_idx > S_KV) end_idx = S_KV;

    auto m_i = ct::full<f32_QG>(-INFINITY);
    auto l_i = ct::full<f32_NxQG>(1.0f);
    auto acc = ct::zeros<f32_DxQG>();

    int num_blocks  = (end_idx - start_idx + BLOCK_N - 1) / BLOCK_N;
    int start_block = start_idx / BLOCK_N;
    auto offs_n = ct::iota<i32_N>();

    // Sliding-window limit (query_pos is S_KV-1 for decode).
    int window_lo = (WINDOW_SIZE > 0) ? (S_KV - 1 - WINDOW_SIZE) : 0;

    for (auto block_idx : ct::irange(0, num_blocks)) {
        int kv_block = start_block + block_idx;
        int curr_n   = kv_block * BLOCK_N;

        T_4D_Tile k_4d = K_view.load(batch_id, head_id, kv_block, 0);
        auto k = ct::reshape(k_4d, ct::shape<BLOCK_N, HEAD_DIM>{});

        // qk = K @ Q  ->  (BLOCK_N, QUERY_GROUP_BLOCK_SIZE) in fp32.
        auto qk = ct::mma(k, q_nat, ct::zeros<f32_NxQG>());

        auto n_indices = ct::full<i32_N>(curr_n) + offs_n;
        auto n_mask_b  = n_indices < S_KV;
        auto n_mask_b_2d = ct::reshape(n_mask_b, ct::shape<BLOCK_N, 1>{});

        f32_NxQG qk_softmax;
        if constexpr (HAS_SOFT_CAP) {
            // Apply original scale, soft cap via tanh, then convert to log2 domain.
            auto qk_scaled = qk * sm_scale_orig;
            auto qk_cap    = qk_scaled * inv_soft_cap;
            auto qk_tanh   = ct::tanh(qk_cap);
            auto qk_out    = qk_tanh * soft_cap;

            // Boundary mask.
            auto qk_masked = ct::select(n_mask_b_2d, qk_out,
                                         ct::full<f32_NxQG>(-1.0e6f));
            if constexpr (WINDOW_SIZE > 0) {
                auto in_win  = n_indices >= ct::full<i32_N>(window_lo);
                auto in_win_2d = ct::reshape(in_win, ct::shape<BLOCK_N, 1>{});
                qk_masked = ct::select(in_win_2d, qk_masked,
                                        ct::full<f32_NxQG>(-1.0e6f));
            }

            qk_softmax = qk_masked * GEMMA_DECODE_INV_LOG_2;
        } else {
            // Fold scale and INV_LOG_2 together.
            auto qk_scaled = qk * qk_scale;
            auto qk_masked = ct::select(n_mask_b_2d, qk_scaled,
                                         ct::full<f32_NxQG>(-1.0e6f));
            if constexpr (WINDOW_SIZE > 0) {
                auto in_win  = n_indices >= ct::full<i32_N>(window_lo);
                auto in_win_2d = ct::reshape(in_win, ct::shape<BLOCK_N, 1>{});
                qk_masked = ct::select(in_win_2d, qk_masked,
                                        ct::full<f32_NxQG>(-1.0e6f));
            }
            qk_softmax = qk_masked;
        }

        auto qk_max_reduced = ct::reduce_max(qk_softmax, ct::integral_constant<0>{});
        auto m_ij = ct::max(m_i, ct::reshape(qk_max_reduced,
                                              ct::shape<QUERY_GROUP_BLOCK_SIZE>{}));
        auto m_ij_2d = ct::reshape(m_ij, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
        auto p = ct::exp2(qk_softmax - m_ij_2d);

        auto alpha     = ct::exp2(m_i - m_ij);
        auto alpha_2d  = ct::reshape(alpha, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
        l_i = l_i * alpha_2d + p;
        acc = acc * alpha_2d;

        T_4D_Tile v_4d = V_view.load(batch_id, head_id, kv_block, 0);
        auto v_2d = ct::reshape(v_4d, ct::shape<BLOCK_N, HEAD_DIM>{});
        auto v_t  = ct::transpose(v_2d);

        auto p_T   = ct::element_cast<T>(p);
        auto v_t_T = ct::element_cast<T>(v_t);
        acc = ct::mma(v_t_T, p_T, acc);

        m_i = m_ij;
    }

    // Reduce l_i across BLOCK_N and normalize accumulator.
    auto l    = ct::sum(l_i, ct::integral_constant<0>{});
    auto l_1d = ct::reshape(l, ct::shape<QUERY_GROUP_BLOCK_SIZE>{});
    auto l_2d = ct::reshape(l_1d, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
    auto acc_norm = ct::div(acc, l_2d, ct::round_approximate_t{},
                             ct::round_subnormals_to_zero_t{});
    auto lse = m_i + ct::log2(l_1d);

    auto acc_out   = ct::transpose(acc_norm);
    auto acc_out_T = ct::element_cast<T>(acc_out);

    auto Out_view = ct::partition_view(
        ct::tensor_span{Out_ptr, ct::extents{B, H_KV, Q_PER_KV, NUM_KV_SPLITS, HEAD_DIM}},
        ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1, HEAD_DIM>{});
    auto acc_out_5d = ct::reshape(acc_out_T,
                                    ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1, HEAD_DIM>{});
    Out_view.store(acc_out_5d, batch_id, head_id, 0, split_id, 0);

    auto LSE_view = ct::partition_view(
        ct::tensor_span{LSE_ptr, ct::extents{B, H_KV, Q_PER_KV, NUM_KV_SPLITS}},
        ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1>{});
    auto lse_4d = ct::reshape(lse, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1>{});
    LSE_view.store(lse_4d, batch_id, head_id, 0, split_id);
}
