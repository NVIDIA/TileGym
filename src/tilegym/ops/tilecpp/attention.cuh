// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Flash Attention Kernel - Optimized Tile C++ implementation
 *
 * Key optimizations:
 * - tensor_span + partition_view for structured memory access
 * - ct::irange for compiler-optimized loops
 * - All dimensions as template parameters
 * - Native type MMA for tensor core utilization
 *
 * Features:
 * - Flash attention with online softmax (prefill)
 * - Causal masking support (IS_CAUSAL template parameter)
 * - BNSD layout (batch, heads, seq, dim)
 * - Uses exp2 for numerical stability (base-2 exponential)
 * - FP16/BF16 inputs with FP32 accumulation
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 1/ln(2) for exp2 optimization
constexpr float ATTENTION_INV_LOG_2 = 1.442695040888963f;

/**
 * Optimized Prefill Flash Multi-Head Attention Forward Kernel
 *
 * Template Parameters:
 *   T: Element type (__half or __nv_bfloat16)
 *   B: Batch size
 *   H: Number of heads (query heads)
 *   H_KV: Number of KV heads
 *   S_QO: Query/Output sequence length
 *   S_KV: Key/Value sequence length
 *   BLOCK_M: Query block size (typically 64 or 128)
 *   BLOCK_N: Key/Value block size (typically 64 or 128)
 *   BLOCK_D: Head dimension (e.g., 64, 128)
 *   IS_CAUSAL: Whether to apply causal masking
 *   HAS_BACKWARD: Whether to store log-sum-exp for backward pass
 *
 * Grid: (cdiv(S_QO, BLOCK_M), B * H, 1)
 */
template<typename T, int B, int H, int H_KV, int S_QO, int S_KV,
         int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL, bool HAS_BACKWARD,
         int occupancy, int NUM_CTAS = 1>
[[ using cutile : hint(1000, num_cta_in_cga=NUM_CTAS, occupancy=occupancy)]]
__tile_global__ void prefill_fmha_fwd_kernel(
    const T* __restrict__ Q_ptr,
    const T* __restrict__ K_ptr,
    const T* __restrict__ V_ptr,
    T* __restrict__ Out_ptr,
    float* __restrict__ L_ptr,
    float sm_scale
) {
    namespace ct = cuda::tiles;

    // Apply alignment hints
    Q_ptr = ct::assume_aligned<16>(Q_ptr);
    K_ptr = ct::assume_aligned<16>(K_ptr);
    V_ptr = ct::assume_aligned<16>(V_ptr);
    Out_ptr = ct::assume_aligned<16>(Out_ptr);
    L_ptr = ct::assume_aligned<16>(L_ptr);

    // Tile type aliases
    using T_MxD = ct::tile<T, ct::shape<BLOCK_M, BLOCK_D>>;
    using T_NxD = ct::tile<T, ct::shape<BLOCK_N, BLOCK_D>>;
    using T_MxN = ct::tile<T, ct::shape<BLOCK_M, BLOCK_N>>;
    using T_4D_M = ct::tile<T, ct::shape<1, 1, BLOCK_M, BLOCK_D>>;
    using T_4D_N = ct::tile<T, ct::shape<1, 1, BLOCK_N, BLOCK_D>>;

    using f32_MxD = ct::tile<float, ct::shape<BLOCK_M, BLOCK_D>>;
    using f32_NxD = ct::tile<float, ct::shape<BLOCK_N, BLOCK_D>>;
    using f32_MxN = ct::tile<float, ct::shape<BLOCK_M, BLOCK_N>>;
    using f32_DxN = ct::tile<float, ct::shape<BLOCK_D, BLOCK_N>>;
    using f32_M = ct::tile<float, ct::shape<BLOCK_M>>;
    using i32_M = ct::tile<int, ct::shape<BLOCK_M>>;
    using i32_N = ct::tile<int, ct::shape<BLOCK_N>>;

    constexpr float NEG_INF = -INFINITY;

    int pid_x = ct::bid().x;  // Query block index
    int pid_y = ct::bid().y;  // Batch * Head index
    int batch_idx = pid_y / H;
    int head_idx = pid_y % H;

    // For grouped-query attention
    constexpr int QUERY_GROUP_SIZE = (H == H_KV) ? 1 : (H / H_KV);
    int off_kv_h = (H == H_KV) ? head_idx : (head_idx / QUERY_GROUP_SIZE);

    float qk_scale = sm_scale * ATTENTION_INV_LOG_2;

    auto Q_span = ct::tensor_span{Q_ptr, ct::extents<uint32_t, B, H, S_QO, BLOCK_D>{}};
    auto Q_view = ct::partition_view(Q_span, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});

    auto K_span = ct::tensor_span{K_ptr, ct::extents<uint32_t, B, H_KV, S_KV, BLOCK_D>{}};
    auto K_view = ct::partition_view(K_span, ct::shape<1, 1, BLOCK_N, BLOCK_D>{});

    auto V_span = ct::tensor_span{V_ptr, ct::extents<uint32_t, B, H_KV, S_KV, BLOCK_D>{}};
    auto V_view = ct::partition_view(V_span, ct::shape<1, 1, BLOCK_N, BLOCK_D>{});

    auto Out_span = ct::tensor_span{Out_ptr, ct::extents<uint32_t, B, H, S_QO, BLOCK_D>{}};
    auto Out_view = ct::partition_view(Out_span, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});

    // Load Q block: [BLOCK_M, BLOCK_D]
    T_4D_M q_4d = Q_view.load(batch_idx, head_idx, pid_x, 0);
    auto q = ct::reshape(q_4d, ct::shape<BLOCK_M, BLOCK_D>{});

    using f32_Mx1 = ct::tile<float, ct::shape<BLOCK_M, 1>>;
    auto m_i = ct::full<f32_Mx1>(NEG_INF);
    auto l_i = ct::full<f32_Mx1>(0.0f);
    auto acc = ct::zeros<f32_MxD>();

    // Compute loop bounds for causal/non-causal
    int lo = 0;
    int hi;
    if constexpr (IS_CAUSAL) {
        hi = (pid_x + 1) * BLOCK_M;
        if (hi > S_KV) hi = S_KV;
    } else {
        hi = S_KV;
    }

    // Number of K/V blocks to process
    int num_kv_blocks = (hi - lo + BLOCK_N - 1) / BLOCK_N;
    constexpr bool EVEN_K = (S_KV % BLOCK_N) == 0;
    int mask_start = IS_CAUSAL ? ((pid_x * BLOCK_M) / BLOCK_N) : (S_KV / BLOCK_N);

    using i32_Mx1 = ct::tile<int, ct::shape<BLOCK_M, 1>>;
    using i32_1xN = ct::tile<int, ct::shape<1, BLOCK_N>>;
    auto offs_m_base = ct::iota<i32_M>();
    auto offs_m_2d   = ct::reshape(offs_m_base + ct::full<i32_M>(pid_x * BLOCK_M),
                                   ct::shape<BLOCK_M, 1>{});
    auto offs_n_1d   = ct::iota<i32_N>();
    auto offs_n_2d   = ct::reshape(offs_n_1d, ct::shape<1, BLOCK_N>{});

    auto neg_inf_2d = ct::full<f32_MxN>(-INFINITY);
    auto zero_2d    = ct::full<f32_MxN>(0.0f);

    constexpr bool NEEDS_MASK = IS_CAUSAL || !EVEN_K;
    int unmasked_end = NEEDS_MASK ? (mask_start < num_kv_blocks ? mask_start : num_kv_blocks)
                                  : num_kv_blocks;
    int masked_start = NEEDS_MASK ? (mask_start > 0 ? mask_start : 0)
                                  : num_kv_blocks;

    // ---------------------------------------------------------------------
    // Loop 1: unmasked region [0, unmasked_end)
    // ---------------------------------------------------------------------
    for (auto kv_block : ct::irange(0, unmasked_end)) {
        T_4D_N k_raw;
        [[ using cutile : hint(1000, latency=2) ]]
        k_raw = K_view.load(batch_idx, off_kv_h, kv_block, 0);
        auto k_4d_T = ct::permute(k_raw, ct::dimension_map<0, 1, 3, 2>{});
        auto k_t    = ct::reshape(k_4d_T, ct::shape<BLOCK_D, BLOCK_N>{});

        auto qk = ct::mma(q, k_t, ct::zeros<f32_MxN>());

        auto qk_max_2d = ct::reduce_max(qk, ct::integral_constant<1>{});
        auto m_ij      = ct::max(m_i, qk_max_2d * qk_scale);
        qk             = qk * qk_scale - m_ij;

        auto p     = ct::exp2(qk, ct::round_subnormals_to_zero_t{});
        auto l_ij  = ct::sum(p, ct::integral_constant<1>{});
        auto alpha = ct::exp2(m_i - m_ij, ct::round_subnormals_to_zero_t{});

        l_i = l_i * alpha + l_ij;
        acc = acc * alpha;

        T_4D_N v_4d;
        [[ using cutile : hint(1000, latency=4) ]]
        v_4d = V_view.load(batch_idx, off_kv_h, kv_block, 0);
        auto v = ct::reshape(v_4d, ct::shape<BLOCK_N, BLOCK_D>{});

        auto p_T = ct::element_cast<T>(p);
        acc = ct::mma(p_T, v, acc);
        m_i = m_ij;
    }

    // ---------------------------------------------------------------------
    // Loop 2: masked region [masked_start, num_kv_blocks)
    // ---------------------------------------------------------------------
    if constexpr (NEEDS_MASK) {
        for (auto kv_block : ct::irange(masked_start, num_kv_blocks)) {
            int curr_n = kv_block * BLOCK_N;

            T_4D_N k_raw;
            [[ using cutile : hint(1000, latency=2) ]]
            k_raw = K_view.load(batch_idx, off_kv_h, kv_block, 0);
            auto k_4d_T = ct::permute(k_raw, ct::dimension_map<0, 1, 3, 2>{});
            auto k_t    = ct::reshape(k_4d_T, ct::shape<BLOCK_D, BLOCK_N>{});

            auto qk = ct::mma(q, k_t, ct::zeros<f32_MxN>());

            auto n_pos = ct::reshape(offs_n_1d + ct::full<i32_N>(curr_n),
                                     ct::shape<1, BLOCK_N>{});
            if constexpr (IS_CAUSAL && !EVEN_K) {
                auto valid_bool = (n_pos < ct::full<i32_1xN>(S_KV)) & (offs_m_2d >= n_pos);
                auto mask_val = ct::select(valid_bool, zero_2d, neg_inf_2d);
                qk = qk + mask_val;
            } else if constexpr (IS_CAUSAL) {
                auto valid_bool = (offs_m_2d >= n_pos);
                auto mask_val = ct::select(valid_bool, zero_2d, neg_inf_2d);
                qk = qk + mask_val;
            } else {
                auto valid_bool = n_pos < ct::full<i32_1xN>(S_KV);
                auto mask_val = ct::select(valid_bool, zero_2d, neg_inf_2d);
                qk = qk + mask_val;
            }

            auto qk_max_2d = ct::reduce_max(qk, ct::integral_constant<1>{});
            auto m_ij      = ct::max(m_i, qk_max_2d * qk_scale);
            qk             = qk * qk_scale - m_ij;

            auto p     = ct::exp2(qk, ct::round_subnormals_to_zero_t{});
            auto l_ij  = ct::sum(p, ct::integral_constant<1>{});
            auto alpha = ct::exp2(m_i - m_ij, ct::round_subnormals_to_zero_t{});

            l_i = l_i * alpha + l_ij;
            acc = acc * alpha;

            T_4D_N v_4d;
            [[ using cutile : hint(1000, latency=4) ]]
            v_4d = V_view.load(batch_idx, off_kv_h, kv_block, 0);
            auto v = ct::reshape(v_4d, ct::shape<BLOCK_N, BLOCK_D>{});

            auto p_T = ct::element_cast<T>(p);
            acc = ct::mma(p_T, v, acc);
            m_i = m_ij;
        }
    }

    acc = ct::div(acc, l_i, ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});

    // Convert back to input type and store output
    auto acc_T = ct::element_cast<T>(acc);
    auto acc_4d = ct::reshape(acc_T, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});
    Out_view.store(acc_4d, batch_idx, head_idx, pid_x, 0);

    if constexpr (HAS_BACKWARD) {
        auto L_span = ct::tensor_span{L_ptr, ct::extents<uint32_t, B, H, S_QO>{}};
        auto L_view = ct::partition_view(L_span, ct::shape<1, 1, BLOCK_M>{});

        auto lse_2d = m_i + ct::log2(l_i);                                 // (TILE_M, 1)
        auto lse_1d = ct::reshape(lse_2d, ct::shape<BLOCK_M>{});
        auto lse_3d = ct::reshape(lse_1d, ct::shape<1, 1, BLOCK_M>{});
        L_view.store(lse_3d, batch_idx, head_idx, pid_x);
    }
}


/**
 * Backward Preprocess Kernel
 *
 * Computes:
 *   delta = -sum(o * do, axis=D) * softmax_scale
 *   minus_L = -L
 */
template<typename T, int B, int H, int S_QO, int BLOCK_M, int BLOCK_D, int occupancy>
[[ using cutile : hint(1000, occupancy=occupancy)]]
__tile_global__ void fmha_bwd_preprocess_kernel(
    const T* __restrict__ Out_ptr,
    const T* __restrict__ dO_ptr,
    const float* __restrict__ L_ptr,
    float* __restrict__ minus_Delta_ptr,
    float* __restrict__ minus_L_ptr,
    float softmax_scale
) {
    namespace ct = cuda::tiles;

    Out_ptr = ct::assume_aligned<16>(Out_ptr);
    dO_ptr = ct::assume_aligned<16>(dO_ptr);
    L_ptr = ct::assume_aligned<16>(L_ptr);
    minus_Delta_ptr = ct::assume_aligned<16>(minus_Delta_ptr);
    minus_L_ptr = ct::assume_aligned<16>(minus_L_ptr);

    using T_MxD = ct::tile<T, ct::shape<BLOCK_M, BLOCK_D>>;
    using T_4D_M = ct::tile<T, ct::shape<1, 1, BLOCK_M, BLOCK_D>>;
    using f32_MxD = ct::tile<float, ct::shape<BLOCK_M, BLOCK_D>>;
    using f32_M = ct::tile<float, ct::shape<BLOCK_M>>;
    using f32_3D = ct::tile<float, ct::shape<1, 1, BLOCK_M>>;

    int pid_x = ct::bid().x;  // Block index
    int pid_y = ct::bid().y;  // Batch * Head index
    int batch_idx = pid_y / H;
    int head_idx = pid_y % H;

    // Create tensor spans
    auto Out_span = ct::tensor_span{Out_ptr, ct::extents{B, H, S_QO, BLOCK_D}};
    auto Out_view = ct::partition_view(Out_span, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});

    auto dO_span = ct::tensor_span{dO_ptr, ct::extents{B, H, S_QO, BLOCK_D}};
    auto dO_view = ct::partition_view(dO_span, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});

    auto L_span = ct::tensor_span{L_ptr, ct::extents{B, H, S_QO}};
    auto L_view = ct::partition_view(L_span, ct::shape<1, 1, BLOCK_M>{});

    auto Delta_span = ct::tensor_span{minus_Delta_ptr, ct::extents{B, H, S_QO}};
    auto Delta_view = ct::partition_view(Delta_span, ct::shape<1, 1, BLOCK_M>{});

    auto ML_span = ct::tensor_span{minus_L_ptr, ct::extents{B, H, S_QO}};
    auto ML_view = ct::partition_view(ML_span, ct::shape<1, 1, BLOCK_M>{});

    // Load O and dO: [BLOCK_M, BLOCK_D]
    T_4D_M o_4d = Out_view.load(batch_idx, head_idx, pid_x, 0);
    T_4D_M do_4d = dO_view.load(batch_idx, head_idx, pid_x, 0);
    auto o = ct::element_cast<float>(ct::reshape(o_4d, ct::shape<BLOCK_M, BLOCK_D>{}));
    auto do_tile = ct::element_cast<float>(ct::reshape(do_4d, ct::shape<BLOCK_M, BLOCK_D>{}));

    // Load L: [BLOCK_M]
    f32_3D l_3d = L_view.load(batch_idx, head_idx, pid_x);
    auto l = ct::reshape(l_3d, ct::shape<BLOCK_M>{});

    // Compute delta = -sum(o * do, axis=D) * softmax_scale
    auto o_do = o * do_tile;
    auto sum_2d = ct::sum(o_do, ct::integral_constant<1>{});  // [BLOCK_M, 1]
    auto sum_1d = ct::reshape(sum_2d, ct::shape<BLOCK_M>{});
    auto delta = sum_1d * (-softmax_scale);

    // Compute minus_L = -L
    auto ml = -l;

    // Store results
    auto delta_3d = ct::reshape(delta, ct::shape<1, 1, BLOCK_M>{});
    auto ml_3d = ct::reshape(ml, ct::shape<1, 1, BLOCK_M>{});
    Delta_view.store(delta_3d, batch_idx, head_idx, pid_x);
    ML_view.store(ml_3d, batch_idx, head_idx, pid_x);
}


/**
 * Backward Kernel
 *
 * Computes gradients for Q, K, V in the attention backward pass.
 */
template<typename T, int B, int H, int S_QO, int S_KV, int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL, int occupancy>
[[ using cutile : hint(1000, occupancy=occupancy)]]
__tile_global__ void fmha_bwd_main_kernel(
    const T* __restrict__ Q_ptr,
    const T* __restrict__ K_ptr,
    const T* __restrict__ V_ptr,
    const T* __restrict__ dO_ptr,
    const float* __restrict__ L_ptr,        // This is minus_L from preprocess
    const float* __restrict__ Delta_ptr,    // This is minus_Delta from preprocess
    float* __restrict__ dQ_ptr,       // Output: gradient w.r.t Q (accumulated atomically, float32)
    T* __restrict__ dK_ptr,           // Output: gradient w.r.t K
    T* __restrict__ dV_ptr,           // Output: gradient w.r.t V
    float softmax_scale
) {
    namespace ct = cuda::tiles;

    Q_ptr = ct::assume_aligned<16>(Q_ptr);
    K_ptr = ct::assume_aligned<16>(K_ptr);
    V_ptr = ct::assume_aligned<16>(V_ptr);
    dO_ptr = ct::assume_aligned<16>(dO_ptr);
    L_ptr = ct::assume_aligned<16>(L_ptr);
    Delta_ptr = ct::assume_aligned<16>(Delta_ptr);
    dQ_ptr = ct::assume_aligned<16>(dQ_ptr);
    dK_ptr = ct::assume_aligned<16>(dK_ptr);
    dV_ptr = ct::assume_aligned<16>(dV_ptr);

    using T_MxD = ct::tile<T, ct::shape<BLOCK_M, BLOCK_D>>;
    using T_NxD = ct::tile<T, ct::shape<BLOCK_N, BLOCK_D>>;
    using T_4D_M = ct::tile<T, ct::shape<1, 1, BLOCK_M, BLOCK_D>>;
    using T_4D_N = ct::tile<T, ct::shape<1, 1, BLOCK_N, BLOCK_D>>;
    using f32_MxD = ct::tile<float, ct::shape<BLOCK_M, BLOCK_D>>;
    using f32_NxD = ct::tile<float, ct::shape<BLOCK_N, BLOCK_D>>;
    using f32_NxM = ct::tile<float, ct::shape<BLOCK_N, BLOCK_M>>;
    using f32_MxN = ct::tile<float, ct::shape<BLOCK_M, BLOCK_N>>;
    using f32_M = ct::tile<float, ct::shape<BLOCK_M>>;
    using f32_N = ct::tile<float, ct::shape<BLOCK_N>>;
    using f32_3D_M = ct::tile<float, ct::shape<1, 1, BLOCK_M>>;
    using i32_M = ct::tile<int, ct::shape<BLOCK_M>>;
    using i32_N = ct::tile<int, ct::shape<BLOCK_N>>;

    float softmax_scale_inv_ln2 = softmax_scale * ATTENTION_INV_LOG_2;

    int pid_x = ct::bid().x;  // K/V block index
    int pid_y = ct::bid().y;  // Batch * Head index
    int batch_idx = pid_y / H;
    int head_idx = pid_y % H;

    int offset_skv = pid_x * BLOCK_N;

    // For causal, determine starting Q block
    int start_m;
    if constexpr (IS_CAUSAL) {
        start_m = ((offset_skv) / BLOCK_M) * BLOCK_M;
    } else {
        start_m = 0;
    }

    // Create tensor spans
    auto Q_span = ct::tensor_span{Q_ptr, ct::extents{B, H, S_QO, BLOCK_D}};
    auto Q_view = ct::partition_view(Q_span, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});

    auto K_span = ct::tensor_span{K_ptr, ct::extents{B, H, S_KV, BLOCK_D}};
    auto K_view = ct::partition_view(K_span, ct::shape<1, 1, BLOCK_N, BLOCK_D>{});

    auto V_span = ct::tensor_span{V_ptr, ct::extents{B, H, S_KV, BLOCK_D}};
    auto V_view = ct::partition_view(V_span, ct::shape<1, 1, BLOCK_N, BLOCK_D>{});

    auto dO_span = ct::tensor_span{dO_ptr, ct::extents{B, H, S_QO, BLOCK_D}};
    auto dO_view = ct::partition_view(dO_span, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});

    auto L_span = ct::tensor_span{L_ptr, ct::extents{B, H, S_QO}};
    auto L_view = ct::partition_view(L_span, ct::shape<1, 1, BLOCK_M>{});

    auto Delta_span = ct::tensor_span{Delta_ptr, ct::extents{B, H, S_QO}};
    auto Delta_view = ct::partition_view(Delta_span, ct::shape<1, 1, BLOCK_M>{});

    auto dQ_span = ct::tensor_span{dQ_ptr, ct::extents{B, H, S_QO, BLOCK_D}};
    auto dQ_view = ct::partition_view(dQ_span, ct::shape<1, 1, BLOCK_M, BLOCK_D>{});

    auto dK_span = ct::tensor_span{dK_ptr, ct::extents{B, H, S_KV, BLOCK_D}};
    auto dK_view = ct::partition_view(dK_span, ct::shape<1, 1, BLOCK_N, BLOCK_D>{});

    auto dV_span = ct::tensor_span{dV_ptr, ct::extents{B, H, S_KV, BLOCK_D}};
    auto dV_view = ct::partition_view(dV_span, ct::shape<1, 1, BLOCK_N, BLOCK_D>{});

    // Load K and V for this block: [BLOCK_N, BLOCK_D]
    T_4D_N k_4d = K_view.load(batch_idx, head_idx, pid_x, 0);
    T_4D_N v_4d = V_view.load(batch_idx, head_idx, pid_x, 0);
    auto k = ct::element_cast<float>(ct::reshape(k_4d, ct::shape<BLOCK_N, BLOCK_D>{}));  // [BLOCK_N, BLOCK_D]
    auto v = ct::element_cast<float>(ct::reshape(v_4d, ct::shape<BLOCK_N, BLOCK_D>{}));  // [BLOCK_N, BLOCK_D]

    // Initialize gradient accumulators
    auto dk = ct::zeros<f32_NxD>();
    auto dv = ct::zeros<f32_NxD>();

    // Offset generators for causal masking
    auto offs_m = ct::iota<i32_M>();
    auto offs_n = ct::iota<i32_N>();
    auto n_pos = ct::full<i32_N>(offset_skv) + offs_n;

    // Number of Q blocks to process
    int num_q_blocks = (S_QO - start_m + BLOCK_M - 1) / BLOCK_M;
    int start_block = start_m / BLOCK_M;

    // Loop over Q blocks
    for (auto q_block_idx : ct::irange(0, num_q_blocks)) {
        int curr_m = start_m + q_block_idx * BLOCK_M;
        int q_block = start_block + q_block_idx;

        // Load Q: [BLOCK_M, BLOCK_D]
        T_4D_M q_4d = Q_view.load(batch_idx, head_idx, q_block, 0);
        auto q = ct::element_cast<float>(ct::reshape(q_4d, ct::shape<BLOCK_M, BLOCK_D>{}));  // [BLOCK_M, BLOCK_D]

        // Compute S_T = K @ Q^T: [BLOCK_N, BLOCK_D] @ [BLOCK_D, BLOCK_M] = [BLOCK_N, BLOCK_M]
        auto q_t = ct::transpose(q);  // [BLOCK_D, BLOCK_M]
        auto s_t = ct::mma(k, q_t, ct::zeros<f32_NxM>());

        // Load dO: [BLOCK_M, BLOCK_D]
        T_4D_M do_4d = dO_view.load(batch_idx, head_idx, q_block, 0);
        auto do_tile = ct::element_cast<float>(ct::reshape(do_4d, ct::shape<BLOCK_M, BLOCK_D>{}));  // [BLOCK_M, BLOCK_D]

        // Compute dP_T = V @ dO^T: [BLOCK_N, BLOCK_D] @ [BLOCK_D, BLOCK_M] = [BLOCK_N, BLOCK_M]
        auto do_t = ct::transpose(do_tile);  // [BLOCK_D, BLOCK_M]
        auto dp_t = ct::mma(v, do_t, ct::zeros<f32_NxM>());

        // Load L (minus_L) and Delta (minus_Delta): [BLOCK_M]
        // Use pointer-based loading with masking to handle boundary conditions
        auto l_base_ptr = L_ptr + batch_idx * (H * S_QO) + head_idx * S_QO + curr_m;
        auto delta_base_ptr = Delta_ptr + batch_idx * (H * S_QO) + head_idx * S_QO + curr_m;

        // Create position offsets and mask for valid positions
        auto m_offs = ct::iota<i32_M>();
        auto m_pos_abs = ct::full<i32_M>(curr_m) + m_offs;
        auto valid_mask = m_pos_abs < S_QO;

        // Create pointer tiles
        auto l_ptrs = l_base_ptr + m_offs;
        auto delta_ptrs = delta_base_ptr + m_offs;

        // Load with masking: use 0 for out-of-bounds positions
        auto l = ct::select(valid_mask, ct::load(l_ptrs), ct::zeros<f32_M>());       // [BLOCK_M] - this is -L from preprocess
        auto delta = ct::select(valid_mask, ct::load(delta_ptrs), ct::zeros<f32_M>()); // [BLOCK_M] - this is -Delta from preprocess

        // Apply boundary mask for out-of-bounds Q positions (when curr_m + i >= S_QO)
        auto valid_mask_2d = ct::reshape(valid_mask, ct::shape<1, BLOCK_M>{});
        s_t = ct::select(valid_mask_2d, s_t, ct::full<f32_NxM>(-INFINITY));

        // Apply causal mask if needed
        if constexpr (IS_CAUSAL) {
            // s_t is [BLOCK_N, BLOCK_M]: mask where m >= n
            auto m_pos_2d = ct::reshape(m_pos_abs, ct::shape<1, BLOCK_M>{});
            auto n_pos_2d = ct::reshape(n_pos, ct::shape<BLOCK_N, 1>{});
            auto causal_mask = m_pos_2d >= n_pos_2d;
            s_t = ct::select(causal_mask, s_t, ct::full<f32_NxM>(-INFINITY));
        }

        // Recompute P: P = exp2(S * scale_inv_ln2 + l_t).
        // Note: l is -L, so s_t * scale_inv_ln2 + l = s_t * scale_inv_ln2 - L
        auto l_t = ct::reshape(l, ct::shape<1, BLOCK_M>{});  // [1, BLOCK_M]
        auto s_t_scaled = s_t * softmax_scale_inv_ln2 + l_t;
        auto p_t = ct::exp2(s_t_scaled, ct::round_subnormals_to_zero_t{});  // [BLOCK_N, BLOCK_M]

        // Compute dS = P * (dP * scale + delta)
        // delta is -Delta from preprocess, so we add it
        auto delta_t = ct::reshape(delta, ct::shape<1, BLOCK_M>{});
        auto dp_t_new = dp_t * softmax_scale + delta_t;
        auto ds_t = p_t * dp_t_new;  // [BLOCK_N, BLOCK_M]

        // Accumulate dK: dK += dS_T @ Q = [BLOCK_N, BLOCK_M] @ [BLOCK_M, BLOCK_D]
        auto ds_t_T = ct::element_cast<T>(ds_t);
        auto q_T = ct::element_cast<T>(q);
        dk = ct::mma(ds_t_T, q_T, dk);

        // Accumulate dV: dV += P_T @ dO = [BLOCK_N, BLOCK_M] @ [BLOCK_M, BLOCK_D]
        auto p_t_T = ct::element_cast<T>(p_t);
        auto do_T = ct::element_cast<T>(do_tile);
        dv = ct::mma(p_t_T, do_T, dv);

        // Compute dQ and atomic add: dQ += dS^T @ K
        auto ds = ct::transpose(ds_t);  // [BLOCK_M, BLOCK_N]
        auto dq = ct::mma(ds, k, ct::zeros<f32_MxD>());  // [BLOCK_M, BLOCK_D]

        // Atomic add to dQ (float32)
        // Get pointer to dQ block and atomic add
        auto dq_block_ptr = dQ_ptr + batch_idx * (H * S_QO * BLOCK_D) + head_idx * (S_QO * BLOCK_D) + q_block * BLOCK_M * BLOCK_D;
        auto dq_2d = ct::reshape(dq, ct::shape<BLOCK_M, BLOCK_D>{});

        // Create row and column offset tiles for 2D pointer arithmetic
        using i32_Mx1 = ct::tile<int, ct::shape<BLOCK_M, 1>>;
        using i32_1xD = ct::tile<int, ct::shape<1, BLOCK_D>>;
        auto row_offsets = ct::iota<i32_Mx1>() * static_cast<int>(BLOCK_D);
        auto col_offsets = ct::iota<i32_1xD>();
        auto offsets_2d = row_offsets + col_offsets;  // [BLOCK_M, BLOCK_D]
        auto dq_ptrs = dq_block_ptr + offsets_2d;
        ct::atomic_add(dq_ptrs, dq_2d, ct::memory_order_relaxed_t{}, ct::thread_scope_device_t{});
    }

    // Store dK and dV
    auto dk_T = ct::element_cast<T>(dk);
    auto dv_T = ct::element_cast<T>(dv);
    auto dk_4d = ct::reshape(dk_T, ct::shape<1, 1, BLOCK_N, BLOCK_D>{});
    auto dv_4d = ct::reshape(dv_T, ct::shape<1, 1, BLOCK_N, BLOCK_D>{});
    dK_view.store(dk_4d, batch_idx, head_idx, pid_x, 0);
    dV_view.store(dv_4d, batch_idx, head_idx, pid_x, 0);
}
