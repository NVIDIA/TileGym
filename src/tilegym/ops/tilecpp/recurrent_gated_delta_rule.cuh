// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Recurrent Gated Delta Rule (Qwen3-Next linear-attention variant).
 *
 * Grid: (batch_size * num_heads, ceil(v_head_dim / BLOCK_V), 1)
 *
 * Tensor layouts:
 *   Q, K:         (B, T, H, K_HEAD_DIM)   - input dtype T
 *   V, Output:    (B, T, H, V_HEAD_DIM)   - input dtype T
 *   G, Beta:      (B, T, H)               - input dtype T (loaded as scalar)
 *   InitState:    (B, H, K_HEAD_DIM, V_HEAD_DIM)  - float32
 *   FinalState:   (B, H, K_HEAD_DIM, V_HEAD_DIM)  - float32
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

/**
 * Forward kernel.
 *
 * Template params:
 *   T:                   element type for Q/K/V/G/Beta/Output
 *   BLOCK_K, BLOCK_V:    power-of-2 tile sizes (>= K_HEAD_DIM, BLOCK_V <= V_HEAD_DIM)
 *   HAS_INITIAL_STATE:   seed the recurrent state from InitState if true
 *   OUTPUT_FINAL_STATE:  write state back to FinalState if true
 *   USE_QK_L2NORM:       L2-normalize q_t and k_t per timestep
 *
 * Runtime args include B, SEQ_LEN, NUM_HEADS, K_HEAD_DIM, V_HEAD_DIM so the
 * same cubin can serve every shape (avoids recompile per shape).
 */
template<typename T,
         int BLOCK_K,
         int BLOCK_V,
         bool HAS_INITIAL_STATE,
         bool OUTPUT_FINAL_STATE,
         bool USE_QK_L2NORM>
__tile_global__ void recurrent_gated_delta_rule_fwd_kernel(
    const T* __restrict__ Q,               // (B, T, H, K_HEAD_DIM)
    const T* __restrict__ K,               // (B, T, H, K_HEAD_DIM)
    const T* __restrict__ V,               // (B, T, H, V_HEAD_DIM)
    const T* __restrict__ G,               // (B, T, H)
    const T* __restrict__ Beta,            // (B, T, H)
    T* __restrict__ Output,                // (B, T, H, V_HEAD_DIM)
    const float* __restrict__ InitState,   // (B, H, K_HEAD_DIM, V_HEAD_DIM) or nullptr
    float* __restrict__ FinalState,        // (B, H, K_HEAD_DIM, V_HEAD_DIM) or nullptr
    float scale,
    int B,
    int SEQ_LEN,
    int NUM_HEADS,
    int K_HEAD_DIM,
    int V_HEAD_DIM
) {
    namespace ct = cuda::tiles;

    using f32_KxV = ct::tile<float, ct::shape<BLOCK_K, BLOCK_V>>;
    using f32_K   = ct::tile<float, ct::shape<BLOCK_K>>;
    using f32_V   = ct::tile<float, ct::shape<BLOCK_V>>;

    Q          = ct::assume_aligned<16>(Q);
    K          = ct::assume_aligned<16>(K);
    V          = ct::assume_aligned<16>(V);
    G          = ct::assume_aligned<16>(G);
    Beta       = ct::assume_aligned<16>(Beta);
    Output     = ct::assume_aligned<16>(Output);
    InitState  = ct::assume_aligned<16>(InitState);
    FinalState = ct::assume_aligned<16>(FinalState);

    int pid_bh = ct::bid().x;
    int pid_v  = ct::bid().y;
    int b = pid_bh / NUM_HEADS;
    int h = pid_bh % NUM_HEADS;

    // Partitioned views for TMA-style block loads/stores.
    auto pQ = ct::partition_view(
        ct::tensor_span{Q, ct::extents{B, SEQ_LEN, NUM_HEADS, K_HEAD_DIM}},
        ct::shape<1, 1, 1, BLOCK_K>{});
    auto pK = ct::partition_view(
        ct::tensor_span{K, ct::extents{B, SEQ_LEN, NUM_HEADS, K_HEAD_DIM}},
        ct::shape<1, 1, 1, BLOCK_K>{});
    auto pV = ct::partition_view(
        ct::tensor_span{V, ct::extents{B, SEQ_LEN, NUM_HEADS, V_HEAD_DIM}},
        ct::shape<1, 1, 1, BLOCK_V>{});
    auto pO = ct::partition_view(
        ct::tensor_span{Output, ct::extents{B, SEQ_LEN, NUM_HEADS, V_HEAD_DIM}},
        ct::shape<1, 1, 1, BLOCK_V>{});
    auto pG = ct::partition_view(
        ct::tensor_span{G, ct::extents{B, SEQ_LEN, NUM_HEADS}},
        ct::shape<1, 1, 1>{});
    auto pBeta = ct::partition_view(
        ct::tensor_span{Beta, ct::extents{B, SEQ_LEN, NUM_HEADS}},
        ct::shape<1, 1, 1>{});

    // ---- Initialize state ----
    auto state = ct::zeros<f32_KxV>();
    if constexpr (HAS_INITIAL_STATE) {
        auto pInit = ct::partition_view(
            ct::tensor_span{InitState, ct::extents{B, NUM_HEADS, K_HEAD_DIM, V_HEAD_DIM}},
            ct::shape<1, 1, BLOCK_K, BLOCK_V>{});
        auto init_loaded = pInit.load(b, h, 0, pid_v);
        state = ct::reshape<ct::shape<BLOCK_K, BLOCK_V>>(init_loaded);
    }

    for (auto t : ct::irange(0, SEQ_LEN)) {
        // Load q_t, k_t as BLOCK_K tiles.
        auto q_t_loaded = pQ.load(b, t, h, 0);
        auto q_t = ct::element_cast<float>(ct::reshape<ct::shape<BLOCK_K>>(q_t_loaded));

        auto k_t_loaded = pK.load(b, t, h, 0);
        auto k_t = ct::element_cast<float>(ct::reshape<ct::shape<BLOCK_K>>(k_t_loaded));

        if constexpr (USE_QK_L2NORM) {
            // rsqrt of a scalar: stage into a 1-element tile and let ct::rsqrt handle it.
            using f32_1 = ct::tile<float, ct::shape<1>>;
            float q_sq = static_cast<float>(ct::sum<0>(q_t * q_t));
            float k_sq = static_cast<float>(ct::sum<0>(k_t * k_t));
            float q_inv = static_cast<float>(ct::rsqrt(ct::full<f32_1>(q_sq + 1e-6f)));
            float k_inv = static_cast<float>(ct::rsqrt(ct::full<f32_1>(k_sq + 1e-6f)));
            q_t = q_t * q_inv;
            k_t = k_t * k_inv;
        }

        q_t = q_t * scale;

        // Load v_t as BLOCK_V tile.
        auto v_t_loaded = pV.load(b, t, h, pid_v);
        auto v_t = ct::element_cast<float>(ct::reshape<ct::shape<BLOCK_V>>(v_t_loaded));

        auto g_t_tile    = ct::element_cast<float>(pG.load(b, t, h));
        auto beta_t_tile = ct::element_cast<float>(pBeta.load(b, t, h));
        float g_t    = static_cast<float>(
            ct::sum<0>(ct::reshape<ct::shape<1>>(g_t_tile)));
        float beta_t = static_cast<float>(
            ct::sum<0>(ct::reshape<ct::shape<1>>(beta_t_tile)));

        // 1. Decay state: state *= exp(g_t)
        state = state * ct::exp(g_t);

        // 2. kv_mem[v] = sum_k state[k, v] * k_t[k]
        auto k_col = ct::reshape<ct::shape<BLOCK_K, 1>>(k_t);
        auto kv_mem_2d = ct::sum<0>(state * k_col);
        auto kv_mem    = ct::reshape<ct::shape<BLOCK_V>>(kv_mem_2d);

        // 3. delta = (v_t - kv_mem) * beta_t
        auto delta = (v_t - kv_mem) * beta_t;

        // 4. Rank-1 state update: state += outer(k_t, delta)
        auto delta_row = ct::reshape<ct::shape<1, BLOCK_V>>(delta);
        state = state + k_col * delta_row;

        // 5. out_t[v] = sum_k state[k, v] * q_t[k]
        auto q_col    = ct::reshape<ct::shape<BLOCK_K, 1>>(q_t);
        auto out_t_2d = ct::sum<0>(state * q_col);
        auto out_t    = ct::reshape<ct::shape<BLOCK_V>>(out_t_2d);

        // Store output (cast back to T).
        auto out_tile = ct::reshape<ct::shape<1, 1, 1, BLOCK_V>>(
            ct::element_cast<T>(out_t));
        pO.store(out_tile, b, t, h, pid_v);
    }

    // ---- Final state ----
    if constexpr (OUTPUT_FINAL_STATE) {
        auto pFinal = ct::partition_view(
            ct::tensor_span{FinalState, ct::extents{B, NUM_HEADS, K_HEAD_DIM, V_HEAD_DIM}},
            ct::shape<1, 1, BLOCK_K, BLOCK_V>{});
        auto state_tile = ct::reshape<ct::shape<1, 1, BLOCK_K, BLOCK_V>>(state);
        pFinal.store(state_tile, b, h, 0, pid_v);
    }
}
