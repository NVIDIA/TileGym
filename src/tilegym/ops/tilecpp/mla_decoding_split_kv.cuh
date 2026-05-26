// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * MLA (Multi-head Latent Attention) Decoding with Split-KV
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

constexpr float MLA_INV_LOG_2 = 1.0f / 0.693147180559945309417232121458176568f;

/**
 * MLA Decoding Split-KV Kernel - Optimized
 *
 * Uses tensor_span + partition_view with contiguous tensor layouts.
 * All dimensions are template parameters.
 */
template<typename T, int B, int NUM_HEADS, int S_KV,
         int TILE_D, int TILE_H, int TILE_N, int TILE_KPE,
         int NUM_KV_SPLITS, int KV_LEN_PER_SPLIT, bool EVEN_N>
[[ using cutile : hint(1000, occupancy=2) ]]
__tile_global__ void naive_absorb_mla_transpose(
    const T* __restrict__ Q_ptr,       // [B, NUM_HEADS, TILE_D]
    const T* __restrict__ QPE_ptr,     // [B, NUM_HEADS, TILE_KPE]
    const T* __restrict__ K_ptr,       // [B, S_KV, TILE_D]
    const T* __restrict__ V_ptr,       // [B, S_KV, TILE_D] (same as K for MLA)
    const T* __restrict__ KPE_ptr,     // [B, S_KV, TILE_KPE]
    T* __restrict__ Att_Out_ptr,       // [B, NUM_HEADS, NUM_KV_SPLITS, TILE_D]
    float* __restrict__ LSE_Out_ptr,   // [B, NUM_HEADS, NUM_KV_SPLITS]
    float sm_scale
) {
    namespace ct = cuda::tiles;

    Q_ptr = ct::assume_aligned<16>(Q_ptr);
    QPE_ptr = ct::assume_aligned<16>(QPE_ptr);
    K_ptr = ct::assume_aligned<16>(K_ptr);
    V_ptr = ct::assume_aligned<16>(V_ptr);
    KPE_ptr = ct::assume_aligned<16>(KPE_ptr);
    Att_Out_ptr = ct::assume_aligned<16>(Att_Out_ptr);
    LSE_Out_ptr = ct::assume_aligned<16>(LSE_Out_ptr);

    // Native tile types for tensor core MMA
    using T_HxD = ct::tile<T, ct::shape<TILE_H, TILE_D>>;
    using T_DxH = ct::tile<T, ct::shape<TILE_D, TILE_H>>;
    using T_HxKPE = ct::tile<T, ct::shape<TILE_H, TILE_KPE>>;
    using T_KPExH = ct::tile<T, ct::shape<TILE_KPE, TILE_H>>;
    using T_NxD = ct::tile<T, ct::shape<TILE_N, TILE_D>>;
    using T_DxN = ct::tile<T, ct::shape<TILE_D, TILE_N>>;
    using T_NxKPE = ct::tile<T, ct::shape<TILE_N, TILE_KPE>>;
    using T_NxH = ct::tile<T, ct::shape<TILE_N, TILE_H>>;

    // Float types for accumulation
    using f32_H = ct::tile<float, ct::shape<TILE_H>>;
    using f32_DxH = ct::tile<float, ct::shape<TILE_D, TILE_H>>;
    using f32_NxH = ct::tile<float, ct::shape<TILE_N, TILE_H>>;

    using i32_N = ct::tile<int, ct::shape<TILE_N>>;
    using i32_H = ct::tile<int, ct::shape<TILE_H>>;

    int pid_x = ct::bid().x;
    int batch_idx = ct::bid().y;
    int tile_idx = ct::bid().z;  // Split dimension

    float qk_scale = sm_scale * MLA_INV_LOG_2;

    auto scale_h_pre   = ct::full<f32_H>  (qk_scale);
    auto scale_nxh_pre = ct::full<f32_NxH>(qk_scale);

    // Initialize accumulation variables
    auto m_prev = ct::full<f32_H>(-INFINITY);
    auto l_prev = ct::full<f32_NxH>(1.0f);
    auto acc = ct::zeros<f32_DxH>();

    // Load q with latency hints for pipeline optimization
    constexpr auto zero_pad = ct::view_padding::zero;
    using Q_3D_Tile = ct::tile<T, ct::shape<1, TILE_H, TILE_D>>;
    auto Q_view = ct::partition_view(
        ct::tensor_span{Q_ptr, ct::extents<uint32_t, B, NUM_HEADS, TILE_D>{}},
        ct::shape<1, TILE_H, TILE_D>{});
    Q_3D_Tile q_loaded;
    [[ using cutile : hint(1000, latency=2) ]]
    q_loaded = Q_view.template load_masked<zero_pad>(batch_idx, pid_x, 0);
    auto q_3d = ct::permute(q_loaded, ct::dimension_map<0, 2, 1>{});
    auto q = ct::reshape(q_3d, ct::shape<TILE_D, TILE_H>{});

    using QPE_3D_Tile = ct::tile<T, ct::shape<1, TILE_H, TILE_KPE>>;
    auto QPE_view = ct::partition_view(
        ct::tensor_span{QPE_ptr, ct::extents<uint32_t, B, NUM_HEADS, TILE_KPE>{}},
        ct::shape<1, TILE_H, TILE_KPE>{});
    QPE_3D_Tile qpe_loaded;
    [[ using cutile : hint(1000, latency=2) ]]
    qpe_loaded = QPE_view.template load_masked<zero_pad>(batch_idx, pid_x, 0);
    auto qpe_3d = ct::permute(qpe_loaded, ct::dimension_map<0, 2, 1>{});
    auto qpe = ct::reshape(qpe_3d, ct::shape<TILE_KPE, TILE_H>{});

    // Calculate split range (split-specific logic)
    int split_kv_start = KV_LEN_PER_SPLIT * tile_idx;
    int split_kv_end = split_kv_start + KV_LEN_PER_SPLIT;
    if (split_kv_end > S_KV) {
        split_kv_end = S_KV;
    }

    // Loop over key-value pairs and update accumulator
    int cnt = split_kv_start / TILE_N;
    constexpr int mask_start = (S_KV / TILE_N) * TILE_N;
    auto offs_n = ct::iota<i32_N>();
    for (auto curr_n : ct::irange(split_kv_start, split_kv_end, TILE_N)) {
        // Load key and compute Q@K^T with latency hints
        using K_3D_Tile = ct::tile<T, ct::shape<1, TILE_N, TILE_D>>;
        auto K_view = ct::partition_view(
            ct::tensor_span{K_ptr, ct::extents<uint32_t, B, S_KV, TILE_D>{}},
            ct::shape<1, TILE_N, TILE_D>{});
        K_3D_Tile k_loaded;
        [[ using cutile : hint(1000, latency=2) ]]
        k_loaded = K_view.template load_masked<zero_pad>(batch_idx, cnt, 0);
        auto k = ct::reshape(k_loaded, ct::shape<TILE_N, TILE_D>{});
        auto qk = ct::mma(k, q, ct::zeros<f32_NxH>());

        // Load key position encoding and compute QPE@KPE^T with latency hints
        using KPE_3D_Tile = ct::tile<T, ct::shape<1, TILE_N, TILE_KPE>>;
        auto KPE_view = ct::partition_view(
            ct::tensor_span{KPE_ptr, ct::extents<uint32_t, B, S_KV, TILE_KPE>{}},
            ct::shape<1, TILE_N, TILE_KPE>{});
        KPE_3D_Tile kpe_loaded;
        [[ using cutile : hint(1000, latency=2) ]]
        kpe_loaded = KPE_view.template load_masked<zero_pad>(batch_idx, cnt, 0);
        auto kpe = ct::reshape(kpe_loaded, ct::shape<TILE_N, TILE_KPE>{});
        qk = ct::mma(kpe, qpe, qk);

        // Apply mask if needed
        if constexpr (!EVEN_N) {
            if (curr_n >= mask_start) {
                auto mask = ct::reshape(ct::full<i32_N>(curr_n) + offs_n < S_KV,
                                        ct::shape<TILE_N, 1>{});
                qk = ct::select(mask, qk, ct::full<f32_NxH>(-1.0e6f));
            }
        }

        // Apply scaling and compute attention scores
        auto qk_max_reduced = ct::reduce_max(qk, ct::integral_constant<0>{});
        auto qk_max = ct::reshape(qk_max_reduced, ct::shape<TILE_H>{});
        auto qk_max_scaled = qk_max * scale_h_pre;
        auto m_ij = ct::max(m_prev, qk_max_scaled);

        auto m_ij_2d = ct::reshape(m_ij, ct::shape<1, TILE_H>{});
        auto neg_m_ij_2d = ct::zeros<f32_NxH>() - m_ij_2d;
        qk = ct::fma(qk, scale_nxh_pre, neg_m_ij_2d);

        // Compute attention weights and update running statistics
        auto p = ct::exp2(qk);
        auto alpha = ct::exp2(m_prev - m_ij);
        auto alpha_2d = ct::reshape(alpha, ct::shape<1, TILE_H>{});
        l_prev = l_prev * alpha_2d + p;
        acc = acc * alpha_2d;

        // Load value and compute attention @ value with latency hints
        using V_3D_Tile = ct::tile<T, ct::shape<1, TILE_N, TILE_D>>;
        auto V_view = ct::partition_view(
            ct::tensor_span{V_ptr, ct::extents<uint32_t, B, S_KV, TILE_D>{}},
            ct::shape<1, TILE_N, TILE_D>{});
        V_3D_Tile v_loaded;
        [[ using cutile : hint(1000, latency=2) ]]
        v_loaded = V_view.template load_masked<zero_pad>(batch_idx, cnt, 0);
        auto v_3d = ct::permute(v_loaded, ct::dimension_map<0, 2, 1>{});
        auto v_t = ct::reshape(v_3d, ct::shape<TILE_D, TILE_N>{});
        auto p_T = ct::element_cast<T>(p);
        acc = ct::mma(v_t, p_T, acc);
        m_prev = m_ij;
        cnt += 1;
    }

    // Finalize attention computation
    auto l_sum = ct::reshape(ct::sum(l_prev, ct::integral_constant<0>{}),
                             ct::shape<TILE_H>{});
    auto acc_normalized = ct::div(acc,
                                  ct::reshape(l_sum, ct::shape<1, TILE_H>{}),
                                  ct::round_approximate_t{},
                                  ct::round_subnormals_to_zero_t{});
    l_sum = m_prev + ct::log2(l_sum);

    // Store results (adapted for split-kv format) with latency hints
    auto acc_out = ct::transpose(acc_normalized);
    auto acc_out_T = ct::element_cast<T>(acc_out);
    auto Out_span = ct::tensor_span{Att_Out_ptr, ct::extents<uint32_t, B, NUM_HEADS, NUM_KV_SPLITS, TILE_D>{}};
    auto Out_view = ct::partition_view(Out_span, ct::shape<1, TILE_H, 1, TILE_D>{});
    auto acc_out_4d = ct::reshape(acc_out_T, ct::shape<1, TILE_H, 1, TILE_D>{});
    [[ using cutile : hint(1000, latency=2) ]]
    Out_view.store(acc_out_4d, batch_idx, pid_x, tile_idx, 0);

    // Store log sum exp for this tile with latency hint
    auto idx_head = pid_x * TILE_H + ct::iota<i32_H>();
    auto lse_offsets = (batch_idx * NUM_HEADS + idx_head) * NUM_KV_SPLITS + tile_idx;
    auto lse_mask = idx_head < NUM_HEADS;
    auto lse_ptrs = LSE_Out_ptr + lse_offsets;
    [[ using cutile : hint(1000, latency=2) ]]
    ct::store_masked(lse_ptrs, l_sum, lse_mask);
}
