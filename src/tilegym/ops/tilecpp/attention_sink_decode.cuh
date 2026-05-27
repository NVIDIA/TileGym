// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Attention Sink Decode split-K kernel.
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr float ASD_INV_LOG_2 = 1.0f / 0.693147180559945309417232121458176568f;

template<typename T,
         int B,
         int H_KV,
         int H_QO,                    // == H_KV * NUM_Q_HEAD_PER_KV
         int S_KV,
         int NUM_Q_HEAD_PER_KV,
         int QUERY_GROUP_BLOCK_SIZE,
         int KV_LEN_PER_SPLIT,
         int HEAD_DIM,
         int BLOCK_N,
         int NUM_KV_SPLITS,
         int BANDWIDTH,
         bool HAS_SINKS,
         int occupancy>
[[ using cutile : hint(1000, occupancy=occupancy) ]]
__tile_global__ void attention_sink_decode_kernel(
    const T*   __restrict__ Q_ptr,        // [B, H_KV, NUM_Q_HEAD_PER_KV, HEAD_DIM]
    const T*   __restrict__ K_ptr,        // [B, H_KV, S_KV, HEAD_DIM]
    const T*   __restrict__ V_ptr,        // [B, H_KV, S_KV, HEAD_DIM]
    const T*   __restrict__ Sinks_ptr,    // [H_QO]  (or nullptr when !HAS_SINKS)
    T*         __restrict__ Out_ptr,      // [B, H_KV, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS, HEAD_DIM]
    float*     __restrict__ LSE_ptr,      // [B, H_KV, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS]
    const int* __restrict__ Start_q_ptr,  // [1]
    float softmax_scale
) {
    namespace ct = cuda::tiles;

    Q_ptr       = ct::assume_aligned<16>(Q_ptr);
    K_ptr       = ct::assume_aligned<16>(K_ptr);
    V_ptr       = ct::assume_aligned<16>(V_ptr);
    Out_ptr     = ct::assume_aligned<16>(Out_ptr);
    LSE_ptr     = ct::assume_aligned<16>(LSE_ptr);
    Start_q_ptr = ct::assume_aligned<16>(Start_q_ptr);

    using T_4D_Tile  = ct::tile<T, ct::shape<1, 1, BLOCK_N, HEAD_DIM>>;
    using Q_4D_Tile  = ct::tile<T, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>>;

    using f32_QG     = ct::tile<float, ct::shape<QUERY_GROUP_BLOCK_SIZE>>;
    using f32_DxQG   = ct::tile<float, ct::shape<HEAD_DIM, QUERY_GROUP_BLOCK_SIZE>>;
    using f32_NxQG   = ct::tile<float, ct::shape<BLOCK_N, QUERY_GROUP_BLOCK_SIZE>>;
    using i32_N      = ct::tile<int, ct::shape<BLOCK_N>>;
    using i32_QG     = ct::tile<int, ct::shape<QUERY_GROUP_BLOCK_SIZE>>;

    int batch_id = ct::bid().x;
    int head_id  = ct::bid().y;
    int split_id = ct::bid().z;

    float qk_scale = softmax_scale * ASD_INV_LOG_2;

    int start_q_val = Start_q_ptr[0];

    constexpr auto zero_pad = ct::view_padding::zero;

    auto Q_view = ct::partition_view(
        ct::tensor_span{Q_ptr, ct::extents{B, H_KV, NUM_Q_HEAD_PER_KV, HEAD_DIM}},
        ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>{});
    auto K_view = ct::partition_view(
        ct::tensor_span{K_ptr, ct::extents{B, H_KV, S_KV, HEAD_DIM}},
        ct::shape<1, 1, BLOCK_N, HEAD_DIM>{});
    auto V_view = ct::partition_view(
        ct::tensor_span{V_ptr, ct::extents{B, H_KV, S_KV, HEAD_DIM}},
        ct::shape<1, 1, BLOCK_N, HEAD_DIM>{});

    Q_4D_Tile q_4d = Q_view.load(batch_id, head_id, 0, 0);
    auto q_2d  = ct::reshape(q_4d, ct::shape<QUERY_GROUP_BLOCK_SIZE, HEAD_DIM>{});
    auto q_nat = ct::transpose(q_2d);  // [HEAD_DIM, QUERY_GROUP_BLOCK_SIZE]

    // Split bounds: causal (<= start_q) + optional sliding window.
    int start_idx = split_id * KV_LEN_PER_SPLIT;
    int end_idx   = start_idx + KV_LEN_PER_SPLIT;
    if (end_idx > S_KV)           end_idx = S_KV;
    if (end_idx > start_q_val + 1) end_idx = start_q_val + 1;
    if constexpr (BANDWIDTH > 0) {
        int ws = start_q_val - (BANDWIDTH - 1);
        if (ws < 0) ws = 0;
        if (start_idx < ws) start_idx = ws;
    }
    // Align start_idx down to BLOCK_N.
    start_idx = (start_idx / BLOCK_N) * BLOCK_N;

    // Load sinks (QUERY_GROUP_BLOCK_SIZE,) from Sinks[head_id*NUM_Q_HEAD_PER_KV : ...].
    // When HAS_SINKS is false we initialise sink_scaled to -inf so sink_exp = 0.
    auto sink_scaled = ct::full<f32_QG>(-INFINITY);
    if constexpr (HAS_SINKS) {
        auto Sinks_aligned = ct::assume_aligned<16>(Sinks_ptr);
        auto idx_qg  = ct::iota<i32_QG>();
        auto base    = ct::full<i32_QG>(head_id * NUM_Q_HEAD_PER_KV);
        auto indices = base + idx_qg;
        auto in_bounds = indices < ct::full<i32_QG>(H_QO);
        auto sink_T  = ct::load_masked(Sinks_aligned + indices, in_bounds, T(0));
        auto sink_f  = ct::element_cast<float>(sink_T);
        sink_scaled  = sink_f * ct::full<f32_QG>(ASD_INV_LOG_2);
    }

    auto m_i = ct::full<f32_QG>(-INFINITY);
    auto l_i = ct::full<f32_NxQG>(1.0f);
    auto acc = ct::zeros<f32_DxQG>();

    auto offs_n = ct::iota<i32_N>();
    int num_blocks = (KV_LEN_PER_SPLIT + BLOCK_N - 1) / BLOCK_N;
    int start_block = start_idx / BLOCK_N;

    if (end_idx > start_idx) {
        for (auto block_idx : ct::irange(0, num_blocks)) {
            int kv_block = start_block + block_idx;
            int curr_n   = kv_block * BLOCK_N;
            if (curr_n >= end_idx) break;

            T_4D_Tile k_4d = K_view.template load_masked<zero_pad>(batch_id, head_id, kv_block, 0);
            auto k = ct::reshape(k_4d, ct::shape<BLOCK_N, HEAD_DIM>{});

            auto qk = ct::mma(k, q_nat, ct::zeros<f32_NxQG>());

            // Build combined mask: boundary (>= S_KV), causal (> start_q), sliding window.
            auto kv_pos    = ct::full<i32_N>(curr_n) + offs_n;
            auto beyond_s  = kv_pos >= ct::full<i32_N>(S_KV);
            auto beyond_q  = kv_pos >  ct::full<i32_N>(start_q_val);
            auto mask_b    = beyond_s | beyond_q;
            if constexpr (BANDWIDTH > 0) {
                int wl = start_q_val + 1 - BANDWIDTH;
                auto too_old = kv_pos < ct::full<i32_N>(wl);
                mask_b = mask_b | too_old;
            }
            auto mask_b_2d = ct::reshape(mask_b, ct::shape<BLOCK_N, 1>{});
            auto qk_m = ct::select(mask_b_2d, ct::full<f32_NxQG>(-1.0e6f), qk);

            auto qk_scaled = qk_m * qk_scale;
            auto qk_max    = ct::reduce_max(qk_scaled, ct::integral_constant<0>{});
            auto m_ij      = ct::max(m_i, ct::reshape(qk_max,
                                                       ct::shape<QUERY_GROUP_BLOCK_SIZE>{}));
            auto m_ij_2d   = ct::reshape(m_ij, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
            auto p         = ct::exp2(qk_scaled - m_ij_2d);

            auto alpha     = ct::exp2(m_i - m_ij);
            auto alpha_2d  = ct::reshape(alpha, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
            l_i = l_i * alpha_2d + p;
            acc = acc * alpha_2d;

            T_4D_Tile v_4d = V_view.template load_masked<zero_pad>(batch_id, head_id, kv_block, 0);
            auto v_2d = ct::reshape(v_4d, ct::shape<BLOCK_N, HEAD_DIM>{});
            auto v_t  = ct::transpose(v_2d);

            auto p_T   = ct::element_cast<T>(p);
            auto v_t_T = ct::element_cast<T>(v_t);
            acc = ct::mma(v_t_T, p_T, acc);
            m_i = m_ij;
        }
    }

    // Reduce l_i to 1D.
    auto l    = ct::sum(l_i, ct::integral_constant<0>{});
    auto l_1d = ct::reshape(l, ct::shape<QUERY_GROUP_BLOCK_SIZE>{});

    // For the first split, fold the sink contribution into (m_i, l, acc).
    if (split_id == 0) {
        auto m_i_ws = ct::max(m_i, sink_scaled);
        auto alpha  = ct::exp2(m_i - m_i_ws);
        l_1d        = l_1d * alpha;
        auto alpha_2d = ct::reshape(alpha, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
        acc         = acc * alpha_2d;
        auto sink_exp = ct::exp2(sink_scaled - m_i_ws);
        l_1d        = l_1d + sink_exp;
        m_i         = m_i_ws;
    }

    auto l_2d = ct::reshape(l_1d, ct::shape<1, QUERY_GROUP_BLOCK_SIZE>{});
    auto acc_norm = ct::div(acc, l_2d, ct::round_approximate_t{},
                             ct::round_subnormals_to_zero_t{});
    auto lse = m_i + ct::log2(l_1d);

    auto acc_out   = ct::transpose(acc_norm);
    auto acc_out_T = ct::element_cast<T>(acc_out);

    auto Out_view = ct::partition_view(
        ct::tensor_span{Out_ptr, ct::extents{B, H_KV, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS, HEAD_DIM}},
        ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1, HEAD_DIM>{});
    auto acc_out_5d = ct::reshape(acc_out_T,
                                    ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1, HEAD_DIM>{});
    Out_view.store(acc_out_5d, batch_id, head_id, 0, split_id, 0);

    auto LSE_view = ct::partition_view(
        ct::tensor_span{LSE_ptr, ct::extents{B, H_KV, NUM_Q_HEAD_PER_KV, NUM_KV_SPLITS}},
        ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1>{});
    auto lse_4d = ct::reshape(lse, ct::shape<1, 1, QUERY_GROUP_BLOCK_SIZE, 1>{});
    LSE_view.store(lse_4d, batch_id, head_id, 0, split_id);
}
