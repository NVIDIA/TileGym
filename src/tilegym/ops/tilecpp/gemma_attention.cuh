// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Gemma Prefill FMHA with soft cap + sliding window.
 *
 * flash-attention prefill kernel:
 *   - compile-time WINDOW_SIZE (>0 limits attention to a local window)
 *   - compile-time HAS_SOFT_CAP + runtime soft_cap (applies tanh softcap)
 *
 * Grid: (cdiv(S_QO, BLOCK_M), B * H, 1), BNSD layout.
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr float GEMMA_ATTN_INV_LOG_2 = 1.442695040888963f;

template<typename T,
         int B,
         int H,
         int H_KV,
         int S_QO,
         int S_KV,
         int BLOCK_M,
         int BLOCK_N,
         int BLOCK_D,
         bool IS_CAUSAL,
         int WINDOW_SIZE,
         bool HAS_SOFT_CAP,
         int occupancy>
[[ using cutile : hint(1000, occupancy=occupancy) ]]
__tile_global__ void gemma_attention_fwd_kernel(
    const T* __restrict__ Q_ptr,
    const T* __restrict__ K_ptr,
    const T* __restrict__ V_ptr,
    T* __restrict__ Out_ptr,
    float sm_scale,
    float soft_cap
) {
    namespace ct = cuda::tiles;

    Q_ptr   = ct::assume_aligned<16>(Q_ptr);
    K_ptr   = ct::assume_aligned<16>(K_ptr);
    V_ptr   = ct::assume_aligned<16>(V_ptr);
    Out_ptr = ct::assume_aligned<16>(Out_ptr);

    using T_4D_M = ct::tile<T, ct::shape<1, 1, BLOCK_M, BLOCK_D>>;
    using T_4D_N = ct::tile<T, ct::shape<1, 1, BLOCK_N, BLOCK_D>>;

    using f32_MxD = ct::tile<float, ct::shape<BLOCK_M, BLOCK_D>>;
    using f32_MxN = ct::tile<float, ct::shape<BLOCK_M, BLOCK_N>>;
    using f32_M   = ct::tile<float, ct::shape<BLOCK_M>>;
    using i32_M   = ct::tile<int, ct::shape<BLOCK_M>>;
    using i32_N   = ct::tile<int, ct::shape<BLOCK_N>>;

    constexpr float NEG_INF = -INFINITY;

    int pid_x = ct::bid().x;
    int pid_y = ct::bid().y;
    int batch_idx = pid_y / H;
    int head_idx  = pid_y % H;

    constexpr int QUERY_GROUP_SIZE = (H == H_KV) ? 1 : (H / H_KV);
    int off_kv_h = (H == H_KV) ? head_idx : (head_idx / QUERY_GROUP_SIZE);

    float qk_scale      = sm_scale * GEMMA_ATTN_INV_LOG_2;
    float inv_soft_cap  = HAS_SOFT_CAP ? (1.0f / soft_cap) : 0.0f;

    auto Q_view = ct::partition_view(
        ct::tensor_span{Q_ptr, ct::extents<uint32_t, B, H, S_QO, BLOCK_D>{}},
        ct::shape<1, 1, BLOCK_M, BLOCK_D>{});
    auto K_view = ct::partition_view(
        ct::tensor_span{K_ptr, ct::extents<uint32_t, B, H_KV, S_KV, BLOCK_D>{}},
        ct::shape<1, 1, BLOCK_N, BLOCK_D>{});
    auto V_view = ct::partition_view(
        ct::tensor_span{V_ptr, ct::extents<uint32_t, B, H_KV, S_KV, BLOCK_D>{}},
        ct::shape<1, 1, BLOCK_N, BLOCK_D>{});
    auto Out_view = ct::partition_view(
        ct::tensor_span{Out_ptr, ct::extents<uint32_t, B, H, S_QO, BLOCK_D>{}},
        ct::shape<1, 1, BLOCK_M, BLOCK_D>{});

    T_4D_M q_4d = Q_view.load(batch_idx, head_idx, pid_x, 0);
    auto q = ct::reshape(q_4d, ct::shape<BLOCK_M, BLOCK_D>{});

    auto m_i = ct::full<f32_M>(NEG_INF);
    auto l_i = ct::full<f32_M>(1.0f);
    auto acc = ct::zeros<f32_MxD>();

    auto offs_m = ct::iota<i32_M>();
    auto offs_n = ct::iota<i32_N>();
    auto q_pos  = ct::full<i32_M>(pid_x * BLOCK_M) + offs_m;

    int hi;
    int mask_start;
    if constexpr (IS_CAUSAL) {
        // Causal: process blocks in [0, (pid_x+1)*BLOCK_M) clipped to S_KV.
        hi = (pid_x + 1) * BLOCK_M;
        if (hi > S_KV) hi = S_KV;
        mask_start = (pid_x * BLOCK_M) / BLOCK_N;          // diagonal start
    } else {
        hi = S_KV;
        mask_start = (S_KV + BLOCK_N - 1) / BLOCK_N;       // never enter mask
    }
    int num_kv_blocks = (hi + BLOCK_N - 1) / BLOCK_N;

    // Window-size narrowing affects both STAGE 1 (lo) and (non-causal) hi.
    int start_block = 0;
    if constexpr (WINDOW_SIZE > 0) {
        int ws_lo = pid_x * BLOCK_M - WINDOW_SIZE;
        if (ws_lo < 0) ws_lo = 0;
        start_block = ws_lo / BLOCK_N;
        if constexpr (!IS_CAUSAL) {
            int ws_hi = (pid_x + 1) * BLOCK_M + WINDOW_SIZE;
            if (ws_hi < hi) {
                hi = ws_hi;
                num_kv_blocks = (hi + BLOCK_N - 1) / BLOCK_N;
            }
        }
    }

    // STAGE 1 (off-diagonal, no causal mask).  In non-causal mode this loop
    // covers the entire range; in causal mode it stops at the diagonal.
    for (auto kv_block : ct::irange(start_block, mask_start)) {
        int curr_n = kv_block * BLOCK_N;

        T_4D_N k_raw;
        [[ using cutile : hint(1000, latency=3) ]]
        k_raw = K_view.load(batch_idx, off_kv_h, kv_block, 0);
        auto k_4d_T = ct::permute(k_raw, ct::dimension_map<0, 1, 3, 2>{});
        auto k_t    = ct::reshape(k_4d_T, ct::shape<BLOCK_D, BLOCK_N>{});

        auto qk = ct::mma(q, k_t, ct::zeros<f32_MxN>());

        if constexpr (HAS_SOFT_CAP) {
            auto sm_tile  = ct::full<f32_MxN>(sm_scale);
            auto cap_tile = ct::full<f32_MxN>(soft_cap);
            qk = ct::mul(qk, sm_tile, ct::round_ties_to_even_t{},
                         ct::round_subnormals_to_zero_t{});
            qk = ct::div(qk, cap_tile, ct::round_approximate_t{},
                         ct::round_subnormals_to_zero_t{});
            qk = ct::tanh(qk);
            qk = ct::mul(qk, cap_tile, ct::round_ties_to_even_t{},
                         ct::round_subnormals_to_zero_t{});
        }

        // No causal mask in STAGE 1. For non-causal calls with S_KV not a
        // multiple of BLOCK_N, the last block has K positions past S_KV
        // whose qk values come from zero-padded K loads and would otherwise
        // contribute exp(0) = 1 to the softmax denominator.
        if constexpr (!IS_CAUSAL && (S_KV % BLOCK_N != 0)) {
            auto n_indices_2d = ct::reshape(ct::full<i32_N>(curr_n) + offs_n,
                                            ct::shape<1, BLOCK_N>{});
            auto in_bounds = n_indices_2d < S_KV;
            qk = ct::select(in_bounds, qk, ct::full<f32_MxN>(-1.0e6f));
        }

        if constexpr (WINDOW_SIZE > 0) {
            auto n_indices_2d = ct::reshape(ct::full<i32_N>(curr_n) + offs_n,
                                            ct::shape<1, BLOCK_N>{});
            auto q_pos_2d     = ct::reshape(q_pos, ct::shape<BLOCK_M, 1>{});
            using i32_MxN = ct::tile<int, ct::shape<BLOCK_M, BLOCK_N>>;
            auto diff = ct::zeros<i32_MxN>() + n_indices_2d - q_pos_2d;
            auto lb   = diff >= -WINDOW_SIZE;
            auto ub   = diff <=  WINDOW_SIZE;
            auto win  = lb & ub;
            qk = ct::select(win, qk, ct::full<f32_MxN>(-1.0e6f));
        }

        const float scale_log2 = HAS_SOFT_CAP ? GEMMA_ATTN_INV_LOG_2 : qk_scale;

        auto qk_max    = ct::reduce_max(qk, ct::integral_constant<1>{});
        auto qk_max_1d = ct::reshape(qk_max, ct::shape<BLOCK_M>{});
        auto scale_1d  = ct::full<f32_M>(scale_log2);
        auto qk_max_scaled = ct::mul(qk_max_1d, scale_1d,
                                     ct::round_ties_to_even_t{},
                                     ct::round_subnormals_to_zero_t{});
        auto m_ij      = ct::max(m_i, qk_max_scaled);
        auto m_ij_2d   = ct::reshape(m_ij, ct::shape<BLOCK_M, 1>{});

        auto scale_2d    = ct::full<f32_MxN>(scale_log2);
        auto neg_m_ij_2d = ct::sub(ct::zeros<f32_MxN>(), m_ij_2d,
                                   ct::round_ties_to_even_t{},
                                   ct::round_subnormals_to_zero_t{});
        auto qk_minus_m  = ct::fma(qk, scale_2d, neg_m_ij_2d,
                                   ct::round_ties_to_even_t{},
                                   ct::round_subnormals_to_zero_t{});
        auto p     = ct::exp2(qk_minus_m, ct::round_subnormals_to_zero_t{});
        auto p_sum = ct::sum(p, ct::integral_constant<1>{});
        auto l_ij  = ct::reshape(p_sum, ct::shape<BLOCK_M>{});

        auto m_minus = ct::sub(m_i, m_ij, ct::round_ties_to_even_t{},
                               ct::round_subnormals_to_zero_t{});
        auto alpha    = ct::exp2(m_minus, ct::round_subnormals_to_zero_t{});
        l_i = ct::fma(l_i, alpha, l_ij, ct::round_ties_to_even_t{},
                      ct::round_subnormals_to_zero_t{});
        auto alpha_2d = ct::reshape(alpha, ct::shape<BLOCK_M, 1>{});
        acc = ct::mul(acc, alpha_2d, ct::round_ties_to_even_t{},
                      ct::round_subnormals_to_zero_t{});

        T_4D_N v_4d;
        [[ using cutile : hint(1000, latency=3) ]]
        v_4d = V_view.load(batch_idx, off_kv_h, kv_block, 0);
        auto v   = ct::reshape(v_4d, ct::shape<BLOCK_N, BLOCK_D>{});
        auto p_T = ct::element_cast<T>(p);
        acc = ct::mma(p_T, v, acc);

        m_i = m_ij;
    }

    // STAGE 2 (diagonal blocks, causal mask).  Empty for non-causal.
    for (auto kv_block : ct::irange(mask_start, num_kv_blocks)) {
        int curr_n = kv_block * BLOCK_N;

        T_4D_N k_raw;
        [[ using cutile : hint(1000, latency=3) ]]
        k_raw = K_view.load(batch_idx, off_kv_h, kv_block, 0);
        auto k_4d_T = ct::permute(k_raw, ct::dimension_map<0, 1, 3, 2>{});
        auto k_t    = ct::reshape(k_4d_T, ct::shape<BLOCK_D, BLOCK_N>{});

        auto qk = ct::mma(q, k_t, ct::zeros<f32_MxN>());

        if constexpr (HAS_SOFT_CAP) {
            auto sm_tile  = ct::full<f32_MxN>(sm_scale);
            auto cap_tile = ct::full<f32_MxN>(soft_cap);
            qk = ct::mul(qk, sm_tile, ct::round_ties_to_even_t{},
                         ct::round_subnormals_to_zero_t{});
            qk = ct::div(qk, cap_tile, ct::round_approximate_t{},
                         ct::round_subnormals_to_zero_t{});
            qk = ct::tanh(qk);
            qk = ct::mul(qk, cap_tile, ct::round_ties_to_even_t{},
                         ct::round_subnormals_to_zero_t{});
        }

        auto n_indices_2d = ct::reshape(ct::full<i32_N>(curr_n) + offs_n,
                                        ct::shape<1, BLOCK_N>{});
        auto q_pos_2d     = ct::reshape(q_pos, ct::shape<BLOCK_M, 1>{});

        if constexpr (IS_CAUSAL) {
            auto causal_mask = q_pos_2d >= n_indices_2d;
            qk = ct::select(causal_mask, qk, ct::full<f32_MxN>(-1.0e6f));
        }
        if constexpr (WINDOW_SIZE > 0) {
            using i32_MxN = ct::tile<int, ct::shape<BLOCK_M, BLOCK_N>>;
            auto diff = ct::zeros<i32_MxN>() + n_indices_2d - q_pos_2d;
            auto lb   = diff >= -WINDOW_SIZE;
            auto ub   = diff <=  WINDOW_SIZE;
            auto win  = lb & ub;
            qk = ct::select(win, qk, ct::full<f32_MxN>(-1.0e6f));
        }

        const float scale_log2 = HAS_SOFT_CAP ? GEMMA_ATTN_INV_LOG_2 : qk_scale;

        auto qk_max    = ct::reduce_max(qk, ct::integral_constant<1>{});
        auto qk_max_1d = ct::reshape(qk_max, ct::shape<BLOCK_M>{});
        auto scale_1d  = ct::full<f32_M>(scale_log2);
        auto qk_max_scaled = ct::mul(qk_max_1d, scale_1d,
                                     ct::round_ties_to_even_t{},
                                     ct::round_subnormals_to_zero_t{});
        auto m_ij      = ct::max(m_i, qk_max_scaled);
        auto m_ij_2d   = ct::reshape(m_ij, ct::shape<BLOCK_M, 1>{});

        auto scale_2d    = ct::full<f32_MxN>(scale_log2);
        auto neg_m_ij_2d = ct::sub(ct::zeros<f32_MxN>(), m_ij_2d,
                                   ct::round_ties_to_even_t{},
                                   ct::round_subnormals_to_zero_t{});
        auto qk_minus_m  = ct::fma(qk, scale_2d, neg_m_ij_2d,
                                   ct::round_ties_to_even_t{},
                                   ct::round_subnormals_to_zero_t{});
        auto p     = ct::exp2(qk_minus_m, ct::round_subnormals_to_zero_t{});
        auto p_sum = ct::sum(p, ct::integral_constant<1>{});
        auto l_ij  = ct::reshape(p_sum, ct::shape<BLOCK_M>{});

        auto m_minus = ct::sub(m_i, m_ij, ct::round_ties_to_even_t{},
                               ct::round_subnormals_to_zero_t{});
        auto alpha    = ct::exp2(m_minus, ct::round_subnormals_to_zero_t{});
        l_i = ct::fma(l_i, alpha, l_ij, ct::round_ties_to_even_t{},
                      ct::round_subnormals_to_zero_t{});
        auto alpha_2d = ct::reshape(alpha, ct::shape<BLOCK_M, 1>{});
        acc = ct::mul(acc, alpha_2d, ct::round_ties_to_even_t{},
                      ct::round_subnormals_to_zero_t{});

        T_4D_N v_4d;
        [[ using cutile : hint(1000, latency=3) ]]
        v_4d = V_view.load(batch_idx, off_kv_h, kv_block, 0);
        auto v   = ct::reshape(v_4d, ct::shape<BLOCK_N, BLOCK_D>{});
        auto p_T = ct::element_cast<T>(p);
        acc = ct::mma(p_T, v, acc);

        m_i = m_ij;
    }

    auto l_i_2d = ct::reshape(l_i, ct::shape<BLOCK_M, 1>{});
    acc = ct::div(acc, l_i_2d, ct::round_approximate_t{},
                  ct::round_subnormals_to_zero_t{});

    auto acc_T  = ct::element_cast<T>(acc);
    auto acc_4d = ct::reshape(acc_T, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});
    Out_view.store(acc_4d, batch_idx, head_idx, pid_x, 0);
}
