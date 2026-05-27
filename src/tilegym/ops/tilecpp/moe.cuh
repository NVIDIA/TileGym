// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ Mixture of Experts (MOE) Kernels
 * Implements fused computation for MOE using token and expert matrices.
 *
 *
 * Features:
 * - FP32 accumulator for precision
 * - Templated for float16/bfloat16 support
 * - Support for token routing via sorted_token_ids and expert_ids
 * - Optional multiplication with routed weights (topk_weights)
 * - Grouped scheduling for L2 cache reuse
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

/**
 * Helper to get a zero scalar value of type T using cuda::tiles::zeros
 * This avoids calling T constructors which may not be allowed in tile kernels
 * A 1-element tile is treated as a scalar in CUDA Tile C++.
 */
 template<typename T>
 __tile__ auto zero_scalar() {
     namespace ct = cuda::tiles;
     return ct::zeros<ct::tile<T, ct::shape<1>>>();
 }


/**
 * Fused MOE Kernel
 *
 * Key Parameters:
 * - A: Input tensor (tokens) with shape (M, K), where M is total tokens, K is feature dim
 * - B: Stacked expert weights with shape (E, N, K), E=experts, N=output dim, K=input dim
 * - C: Output tensor with shape (num_valid_tokens, N)
 * - sorted_token_ids: Sorted indices of tokens, repeated topk times, arranged by expert
 * - expert_ids: Expert index for each block
 * - topk_weights: Routing weights for each token (optional)
 * - num_tokens_post_padded: Total number of tokens after padding
 * - num_valid_tokens: Number of valid tokens (before padding)
 * Template Parameters
 *   OUT_T: Output element type (__half or __nv_bfloat16)
 *   IN_T: Input element type (__half, __nv_bfloat16, or __nv_fp8_e4m3)
 *   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions
 *   GROUP_SIZE_M: Number of M tiles to group for L2 reuse
 *   MUL_ROUTED_WEIGHT: Whether to multiply by topk_weights
 *   USE_FP8_SCALES: Whether to use FP8 block-wise scaling
 *   N: Output dimension (constant per model layer)
 *   K: Input dimension (constant per model layer)
 *   GROUP_N: N group size for FP8 quantization
 *   GROUP_K: K group size for FP8 quantization
 *   TOP_K: Number of experts per token
 *   STRIDE_AM: A row stride (A shape[1] before flatten)
 *   STRIDE_AK: A col stride (typically 1)
 *   STRIDE_BE: B expert stride
 *   STRIDE_BK: B K stride
 *   STRIDE_BN: B N stride
 *   STRIDE_CM: C row stride (C shape[1] before flatten)
 *   STRIDE_CN: C col stride (typically 1)
 *   STRIDE_ASM, STRIDE_ASK: A scale strides
 *   STRIDE_BSE, STRIDE_BSK, STRIDE_BSN: B scale strides
 */
template<typename OUT_T, typename IN_T,
         int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int GROUP_SIZE_M,
         bool MUL_ROUTED_WEIGHT, bool USE_FP8_SCALES,
         int N, int K, int GROUP_N, int GROUP_K, int TOP_K,
         int STRIDE_AM, int STRIDE_AK, int STRIDE_BE, int STRIDE_BK, int STRIDE_BN,
         int STRIDE_CM, int STRIDE_CN,
         int STRIDE_ASM, int STRIDE_ASK, int STRIDE_BSE, int STRIDE_BSK, int STRIDE_BSN,
         int EM>
__tile_global__ void fused_moe_kernel(
    const IN_T* __restrict__ a_ptr,      // Input tokens (M, K)
    const IN_T* __restrict__ b_ptr,      // Expert weights (E, N, K)
    OUT_T* __restrict__ c_ptr,           // Output (num_valid_tokens, N)
    const float* __restrict__ a_scale_ptr,      // A scales [M, K/BLOCK_K] (optional, for FP8)
    const float* __restrict__ b_scale_ptr,      // B scales [E, N/BLOCK_N, K/BLOCK_K] (optional, for FP8)
    const float* __restrict__ topk_weights_ptr, // Routing weights (num_valid_tokens,) - always FP32
    const int* __restrict__ sorted_token_ids_ptr,  // Sorted token indices
    const int* __restrict__ expert_ids_ptr,   // Expert index per M-block
    const int* __restrict__ num_tokens_post_padded_ptr,
    // Runtime dimensions (vary per batch)
    int num_valid_tokens  // Valid tokens before padding
) {
    namespace ct = cuda::tiles;

    using f32_MxN = ct::tile<float, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_N>>;
    using f32_MxK = ct::tile<float, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_K>>;
    using f32_KxN = ct::tile<float, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_N>>;
    using IN_MxK = ct::tile<IN_T, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_K>>;
    using IN_KxN = ct::tile<IN_T, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_N>>;
    using OUT_MxN = ct::tile<OUT_T, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_N>>;
    using i32_M = ct::tile<int, ct::shape<BLOCK_SIZE_M>>;
    using i32_N = ct::tile<int, ct::shape<BLOCK_SIZE_N>>;
    using i32_K = ct::tile<int, ct::shape<BLOCK_SIZE_K>>;
    using f32_M = ct::tile<float, ct::shape<BLOCK_SIZE_M>>;

    // Assume alignment for all pointers
    a_ptr = ct::assume_aligned<16>(a_ptr);
    b_ptr = ct::assume_aligned<16>(b_ptr);
    c_ptr = ct::assume_aligned<16>(c_ptr);
    a_scale_ptr = ct::assume_aligned<16>(a_scale_ptr);
    b_scale_ptr = ct::assume_aligned<16>(b_scale_ptr);
    topk_weights_ptr = ct::assume_aligned<16>(topk_weights_ptr);
    sorted_token_ids_ptr = ct::assume_aligned<16>(sorted_token_ids_ptr);
    expert_ids_ptr = ct::assume_aligned<16>(expert_ids_ptr);
    num_tokens_post_padded_ptr = ct::assume_aligned<16>(num_tokens_post_padded_ptr);

    // Load num_tokens_post_padded
    auto num_tokens_post_padded_tile = ct::load(num_tokens_post_padded_ptr);
    int num_tokens_post_padded = (int)num_tokens_post_padded_tile;

    // Grouped ordering for L2 reuse
    int bid = ct::bid().x;
    constexpr int num_bid_m = (EM + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int num_bid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    int num_bid_in_group = GROUP_SIZE_M * num_bid_n;
    int group_id = bid / num_bid_in_group;
    int first_bid_m = group_id * GROUP_SIZE_M;
    int group_size_m = (num_bid_m - first_bid_m < GROUP_SIZE_M)
                     ? (num_bid_m - first_bid_m) : GROUP_SIZE_M;
    int bid_m = first_bid_m + ((bid % num_bid_in_group) % group_size_m);
    int bid_n = (bid % num_bid_in_group) / group_size_m;

    if (bid_m * BLOCK_SIZE_M < num_tokens_post_padded) {
        auto offs_token_id = ct::full<i32_M>(bid_m * BLOCK_SIZE_M) + ct::iota<i32_M>();

        auto sorted_ids_ptrs = sorted_token_ids_ptr + offs_token_id;
        auto offs_token = ct::load(sorted_ids_ptrs);

        // token_mask = offs_token < num_valid_tokens
        auto token_mask = offs_token < num_valid_tokens;

        auto off_experts_tile = ct::load(expert_ids_ptr + bid_m);
        int off_experts = (int)off_experts_tile;

        auto row_indices = offs_token / TOP_K;
        auto a_row_offset = ct::reshape(row_indices, ct::shape<BLOCK_SIZE_M, 1>{}) * STRIDE_AM;

        // Precompute A base pointers: row_offset + initial K=0 column offset (hoisted outside K loop)
        auto offs_k_init = ct::iota<i32_K>();
        auto offs_k_init_2d = ct::reshape(offs_k_init, ct::shape<1, BLOCK_SIZE_K>{});
        auto a_base_ptrs = a_ptr + a_row_offset + offs_k_init_2d * STRIDE_AK;

        auto token_mask_2d = ct::reshape(token_mask, ct::shape<BLOCK_SIZE_M, 1>{});

        // K-stride for advancing A pointers each iteration (compile-time scalar)
        constexpr int a_k_stride = BLOCK_SIZE_K * STRIDE_AK;

        // Initialize accumulator
        auto accumulator = ct::zeros<f32_MxN>();

        // K loop
        constexpr int num_k_blocks = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
        for (auto k_block : ct::irange(0, num_k_blocks)) {
            auto a_ptrs = a_base_ptrs + k_block * a_k_stride;
            auto a = ct::load_masked(a_ptrs, token_mask_2d, zero_scalar<IN_T>());

            // Load B using partition_view: b_ptr shape (E, N, K), load (off_experts, bid_n, k_block)
            auto b_layout = ct::layout_right_mapping{ct::extents<uint32_t, EM, N, K>{}};
            auto pB = ct::partition_view{ct::tensor_span{b_ptr, b_layout}, ct::shape<1, BLOCK_SIZE_N, BLOCK_SIZE_K>{}};
            auto b_3d = pB.load_masked(off_experts, bid_n, k_block);
            auto b_perm = ct::permute(b_3d, ct::dimension_map<0, 2, 1>{});
            auto b = ct::reshape(b_perm, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_N>{});

            // Apply FP8 scales if enabled; otherwise accumulate directly via mma
            if constexpr (USE_FP8_SCALES) {
                // FP8 path: mma into zeros, then scale before adding to accumulator
                auto dot_result = ct::mma(a, b, ct::zeros<f32_MxN>());

                // Load A scale: one per row, per K-block (indexed by original token)
                int k_scale_idx = (k_block * BLOCK_SIZE_K) / GROUP_K;
                auto a_scale_ptrs = a_scale_ptr + (offs_token / TOP_K) * STRIDE_ASM + k_scale_idx * STRIDE_ASK;
                auto a_scale = ct::load_masked(a_scale_ptrs, token_mask, 0.0f);

                // Load B scale: one per expert, per N-block, per K-block
                int n_scale_idx = (bid_n * BLOCK_SIZE_N) / GROUP_N;
                float b_scale_val = b_scale_ptr[off_experts * STRIDE_BSE + n_scale_idx * STRIDE_BSN + k_scale_idx * STRIDE_BSK];

                // Fuse scales: a_scale[M] * b_scale_scalar -> ab_scale[M], then one broadcast multiply
                auto ab_scale = a_scale * b_scale_val;
                auto ab_scale_2d = ct::reshape(ab_scale, ct::shape<BLOCK_SIZE_M, 1>{});
                accumulator = accumulator + dot_result * ab_scale_2d;
            } else {
                // Non-FP8 path: accumulate directly into accumulator
                accumulator = ct::mma(a, b, accumulator);
            }
        }

        // Optional multiplication with routed weights. Padding rows have
        // offs_token == num_valid_tokens (one-past-end), so we must mask.
        if constexpr (MUL_ROUTED_WEIGHT) {
            using f32_M = ct::tile<float, ct::shape<BLOCK_SIZE_M>>;
            auto weight_ptrs = topk_weights_ptr + offs_token;
            auto zero_pad = ct::zeros<f32_M>();
            auto moe_weight = ct::load_masked(weight_ptrs, token_mask, zero_pad);
            auto moe_weight_2d = ct::reshape(moe_weight, ct::shape<BLOCK_SIZE_M, 1>{});
            accumulator = accumulator * moe_weight_2d;
        }

        // Convert to output type
        auto result = ct::element_cast<OUT_T>(accumulator);

        // Write output
        auto offs_cn = ct::iota<i32_N>() + bid_n * BLOCK_SIZE_N;
        auto offs_token_out = ct::reshape(offs_token, ct::shape<BLOCK_SIZE_M, 1>{});
        auto offs_cn_out = ct::reshape(offs_cn, ct::shape<1, BLOCK_SIZE_N>{});
        auto c_offsets = offs_token_out * STRIDE_CM + offs_cn_out * STRIDE_CN;
        auto c_ptrs = c_ptr + c_offsets;

        auto token_mask_out = ct::reshape(token_mask, ct::shape<BLOCK_SIZE_M, 1>{});
        auto n_mask_out = offs_cn_out < N;
        auto c_mask = token_mask_out & n_mask_out;

        ct::store_masked(c_ptrs, result, c_mask);
    }
}

/**
 * SiLU activation function: x * sigmoid(x)
 */
template<typename T>
__tile__ T silu(T x) {
    return x / (T(1.0f) + ct::exp(-x));
}


/**
 * FP8 Fused MOE FC1 Layer Kernel
 *
 * Computes: output = silu(A @ B1) * (A @ B2) for the first layer of gated MLP
 * With FP8 quantization and block-wise scaling.
 *
 * Template Parameters:
 *   T: Output type (__half or __nv_bfloat16) - {T} placeholder from Python (MUST BE FIRST)
 *   INPUT_T: FP8 input type (__nv_fp8_e4m3)
 *   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions
 *   GROUP_SIZE_M: Group size for L2 cache optimization
 *   MUL_ROUTED_WEIGHT: Whether to multiply by topk_weights
 */
template<typename T, typename INPUT_T,
         int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int GROUP_SIZE_M,
         bool MUL_ROUTED_WEIGHT>
__tile_global__ void fused_moe_fc1_layer_kernel(
    const INPUT_T* __restrict__ a_ptr,           // Input tokens (M, K) in FP8
    const INPUT_T* __restrict__ b1_ptr,          // Expert weights 1 (E, N, K) in FP8
    const INPUT_T* __restrict__ b2_ptr,          // Expert weights 2 (E, N, K) in FP8
    T* __restrict__ c_ptr,                       // Output (num_tokens, 2*N) - combined gate and up
    const float* __restrict__ a_scale_ptr,       // A scales [M, K/BLOCK_K]
    const float* __restrict__ b1_scale_ptr,      // B1 scales [E, N/BLOCK_N, K/BLOCK_K]
    const float* __restrict__ b2_scale_ptr,      // B2 scales [E, N/BLOCK_N, K/BLOCK_K]
    const T* __restrict__ topk_weights_ptr,      // Routing weights
    const int* __restrict__ sorted_token_ids_ptr,
    const int* __restrict__ expert_ids_ptr,
    const int* __restrict__ num_tokens_post_padded_ptr,
    // Dimensions
    int M,                          // Number of input rows
    int N,                          // Output dimension (per B matrix)
    int K,                          // Input dimension
    int EM,                         // Total sorted token count
    int num_valid_tokens,
    // Strides
    int stride_am, int stride_ak,
    int stride_b1e, int stride_b1k, int stride_b1n,
    int stride_b2e, int stride_b2k, int stride_b2n,
    int stride_cm, int stride_cn,
    int stride_asm, int stride_ask,
    int stride_b1se, int stride_b1sk, int stride_b1sn,
    int stride_b2se, int stride_b2sk, int stride_b2sn,
    int group_n, int group_k,
    int top_k
) {
    namespace ct = cuda::tiles;

    using f32_MxN = ct::tile<float, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_N>>;
    using f32_M = ct::tile<float, ct::shape<BLOCK_SIZE_M>>;
    using i32_M = ct::tile<int, ct::shape<BLOCK_SIZE_M>>;
    using i32_N = ct::tile<int, ct::shape<BLOCK_SIZE_N>>;
    using i32_K = ct::tile<int, ct::shape<BLOCK_SIZE_K>>;
    using TxMxN = ct::tile<T, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_N>>;

    // Assume alignment for all pointers
    a_ptr = ct::assume_aligned<16>(a_ptr);
    b1_ptr = ct::assume_aligned<16>(b1_ptr);
    b2_ptr = ct::assume_aligned<16>(b2_ptr);
    c_ptr = ct::assume_aligned<16>(c_ptr);
    a_scale_ptr = ct::assume_aligned<16>(a_scale_ptr);
    b1_scale_ptr = ct::assume_aligned<16>(b1_scale_ptr);
    b2_scale_ptr = ct::assume_aligned<16>(b2_scale_ptr);
    topk_weights_ptr = ct::assume_aligned<16>(topk_weights_ptr);
    sorted_token_ids_ptr = ct::assume_aligned<16>(sorted_token_ids_ptr);
    expert_ids_ptr = ct::assume_aligned<16>(expert_ids_ptr);
    num_tokens_post_padded_ptr = ct::assume_aligned<16>(num_tokens_post_padded_ptr);

    auto num_tokens_post_padded_tile = ct::load(num_tokens_post_padded_ptr);
    int num_tokens_post_padded = int(ct::reshape(num_tokens_post_padded_tile, ct::shape<>{}));

    // Compute program ID with grouped ordering
    int pid = ct::bid().x;
    int num_pid_m = (EM + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
    int group_id = pid / num_pid_in_group;
    int first_pid_m = group_id * GROUP_SIZE_M;
    int group_size_m = (num_pid_m - first_pid_m < GROUP_SIZE_M)
                     ? (num_pid_m - first_pid_m) : GROUP_SIZE_M;
    int pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
    int pid_n = (pid % num_pid_in_group) / group_size_m;

    if (pid_m * BLOCK_SIZE_M >= num_tokens_post_padded) {
        return;
    }

    // Load sorted token IDs
    auto offs_token_id = ct::full<i32_M>(pid_m * BLOCK_SIZE_M) + ct::iota<i32_M>();
    auto sorted_ids_ptrs = sorted_token_ids_ptr + offs_token_id;
    auto offs_token = ct::load(sorted_ids_ptrs);
    auto token_mask = offs_token < num_valid_tokens;

    // Get expert for this block
    auto off_experts_tile = ct::load(expert_ids_ptr + pid_m);
    int off_experts = int(ct::reshape(off_experts_tile, ct::shape<>{}));

    auto offs_token_2d = ct::reshape(offs_token / top_k, ct::shape<BLOCK_SIZE_M, 1>{});
    auto token_mask_2d = ct::reshape(token_mask, ct::shape<BLOCK_SIZE_M, 1>{});

    // Initialize accumulators for both B1 and B2
    auto accumulator1 = ct::zeros<f32_MxN>();
    auto accumulator2 = ct::zeros<f32_MxN>();

    // K loop
    int num_k_blocks = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for (auto k_block : ct::irange(0, num_k_blocks)) {
        int k_start = k_block * BLOCK_SIZE_K;
        auto offs_k = ct::iota<i32_K>() + k_start;

        // Load A tile [BLOCK_SIZE_M, BLOCK_SIZE_K]
        auto offs_k_2d = ct::reshape(offs_k, ct::shape<1, BLOCK_SIZE_K>{});
        auto a_offsets = offs_token_2d * stride_am + offs_k_2d * stride_ak;
        auto a_ptrs = a_ptr + a_offsets;
        auto k_mask = offs_k_2d < K;
        auto a_mask = token_mask_2d & k_mask;
        auto a = ct::load_masked(a_ptrs, a_mask, zero_scalar<INPUT_T>());

        // Load A scale [BLOCK_SIZE_M] - one scale per row per k-block
        int k_scale_idx = k_start / group_k;
        auto a_scale_ptrs = a_scale_ptr + (offs_token / top_k) * stride_asm + k_scale_idx * stride_ask;
        auto a_scale = ct::load_masked(a_scale_ptrs, token_mask, 0.0f);

        // Load B1 and B2 using partition_view: shape (E, N, K), load (off_experts, pid_n, k_block)
        auto b1_layout = ct::layout_right_mapping{ct::extents{M, N, K}};
        auto b2_layout = ct::layout_right_mapping{ct::extents{M, N, K}};
        auto pB1 = ct::partition_view{ct::tensor_span{b1_ptr, b1_layout}, ct::shape<1, BLOCK_SIZE_N, BLOCK_SIZE_K>{}};
        auto pB2 = ct::partition_view{ct::tensor_span{b2_ptr, b2_layout}, ct::shape<1, BLOCK_SIZE_N, BLOCK_SIZE_K>{}};
        auto b1_3d = pB1.load_masked(off_experts, pid_n, k_block);
        auto b2_3d = pB2.load_masked(off_experts, pid_n, k_block);
        auto b1_raw = ct::reshape(b1_3d, ct::shape<BLOCK_SIZE_N, BLOCK_SIZE_K>{});
        auto b2_raw = ct::reshape(b2_3d, ct::shape<BLOCK_SIZE_N, BLOCK_SIZE_K>{});
        auto b1 = ct::transpose(b1_raw);
        auto b2 = ct::transpose(b2_raw);

        // Load B scales (per block)
        int n_scale_idx = (pid_n * BLOCK_SIZE_N) / group_n;
        float b1_scale_val = b1_scale_ptr[off_experts * stride_b1se + n_scale_idx * stride_b1sn + k_scale_idx * stride_b1sk];
        float b2_scale_val = b2_scale_ptr[off_experts * stride_b2se + n_scale_idx * stride_b2sn + k_scale_idx * stride_b2sk];

        // Matrix multiply with FP8->F32 (mma handles conversion)
        auto dot1 = ct::mma(a, b1, ct::zeros<f32_MxN>());
        auto dot2 = ct::mma(a, b2, ct::zeros<f32_MxN>());

        // Apply scales: dot * a_scale[:, None] * b_scale
        auto a_scale_2d = ct::reshape(a_scale, ct::shape<BLOCK_SIZE_M, 1>{});
        accumulator1 = accumulator1 + dot1 * a_scale_2d * b1_scale_val;
        accumulator2 = accumulator2 + dot2 * a_scale_2d * b2_scale_val;
    }

    // Apply SiLU to accumulator1 and multiply with accumulator2
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    auto silu_result = accumulator1 / (ct::full<f32_MxN>(1.0f) + ct::exp(-accumulator1));
    auto output = silu_result * accumulator2;

    // Convert to output type
    auto result = ct::element_cast<T>(output);

    // Store output
    auto offs_cn = ct::iota<i32_N>() + pid_n * BLOCK_SIZE_N;
    auto offs_token_out = ct::reshape(offs_token, ct::shape<BLOCK_SIZE_M, 1>{});
    auto offs_cn_out = ct::reshape(offs_cn, ct::shape<1, BLOCK_SIZE_N>{});
    auto c_offsets = offs_token_out * stride_cm + offs_cn_out * stride_cn;
    auto c_ptrs = c_ptr + c_offsets;

    auto token_mask_out = ct::reshape(token_mask, ct::shape<BLOCK_SIZE_M, 1>{});
    auto n_mask_out = offs_cn_out < N;
    auto c_mask = token_mask_out & n_mask_out;

    ct::store_masked(c_ptrs, result, c_mask);
}

/**
 * FP8 Fused MOE FC2 Layer Kernel
 *
 * Computes: output = A @ B for the second layer of MLP
 * With FP8 quantization and block-wise scaling.
 *
 * Template Parameters:
 *   T: Output type (__half or __nv_bfloat16) - {T} placeholder from Python (MUST BE FIRST)
 *   INPUT_T: FP8 input type (__nv_fp8_e4m3)
 */
template<typename T, typename INPUT_T,
         int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int GROUP_SIZE_M,
         bool MUL_ROUTED_WEIGHT>
__tile_global__ void fused_moe_fc2_layer_kernel(
    const INPUT_T* __restrict__ a_ptr,           // Input (sorted_tokens, K) in FP8
    const INPUT_T* __restrict__ b_ptr,           // Expert weights (E, N, K) in FP8
    T* __restrict__ c_ptr,                       // Output (num_valid_tokens, N)
    const float* __restrict__ a_scale_ptr,       // A scales
    const float* __restrict__ b_scale_ptr,       // B scales
    const T* __restrict__ topk_weights_ptr,
    const int* __restrict__ sorted_token_ids_ptr,
    const int* __restrict__ expert_ids_ptr,
    const int* __restrict__ num_tokens_post_padded_ptr,
    // Dimensions
    int M,
    int N,
    int K,
    int EM,
    int num_valid_tokens,
    // Strides
    int stride_am, int stride_ak,
    int stride_be, int stride_bk, int stride_bn,
    int stride_cm, int stride_cn,
    int stride_asm, int stride_ask,
    int stride_bse, int stride_bsk, int stride_bsn,
    int group_n, int group_k,
    int top_k
) {
    namespace ct = cuda::tiles;

    using f32_MxN = ct::tile<float, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_N>>;
    using f32_M = ct::tile<float, ct::shape<BLOCK_SIZE_M>>;
    using i32_M = ct::tile<int, ct::shape<BLOCK_SIZE_M>>;
    using i32_N = ct::tile<int, ct::shape<BLOCK_SIZE_N>>;
    using i32_K = ct::tile<int, ct::shape<BLOCK_SIZE_K>>;
    using TxMxN = ct::tile<T, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_N>>;

    // Assume alignment for all pointers
    a_ptr = ct::assume_aligned<16>(a_ptr);
    b_ptr = ct::assume_aligned<16>(b_ptr);
    c_ptr = ct::assume_aligned<16>(c_ptr);
    a_scale_ptr = ct::assume_aligned<16>(a_scale_ptr);
    b_scale_ptr = ct::assume_aligned<16>(b_scale_ptr);
    topk_weights_ptr = ct::assume_aligned<16>(topk_weights_ptr);
    sorted_token_ids_ptr = ct::assume_aligned<16>(sorted_token_ids_ptr);
    expert_ids_ptr = ct::assume_aligned<16>(expert_ids_ptr);
    num_tokens_post_padded_ptr = ct::assume_aligned<16>(num_tokens_post_padded_ptr);

    auto num_tokens_post_padded_tile = ct::load(num_tokens_post_padded_ptr);
    int num_tokens_post_padded = int(ct::reshape(num_tokens_post_padded_tile, ct::shape<>{}));

    // Compute program ID with grouped ordering
    int pid = ct::bid().x;
    int num_pid_m = (EM + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
    int group_id = pid / num_pid_in_group;
    int first_pid_m = group_id * GROUP_SIZE_M;
    int group_size_m = (num_pid_m - first_pid_m < GROUP_SIZE_M)
                     ? (num_pid_m - first_pid_m) : GROUP_SIZE_M;
    int pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
    int pid_n = (pid % num_pid_in_group) / group_size_m;

    if (pid_m * BLOCK_SIZE_M >= num_tokens_post_padded) {
        return;
    }

    // Load sorted token IDs
    auto offs_token_id = ct::full<i32_M>(pid_m * BLOCK_SIZE_M) + ct::iota<i32_M>();
    auto sorted_ids_ptrs = sorted_token_ids_ptr + offs_token_id;
    auto offs_token = ct::load(sorted_ids_ptrs);
    auto token_mask = offs_token < num_valid_tokens;

    // Get expert for this block
    auto off_experts_tile = ct::load(expert_ids_ptr + pid_m);
    int off_experts = int(ct::reshape(off_experts_tile, ct::shape<>{}));

    auto offs_row = ct::reshape(offs_token, ct::shape<BLOCK_SIZE_M, 1>{});
    auto row_mask = ct::reshape(token_mask, ct::shape<BLOCK_SIZE_M, 1>{});

    // Initialize accumulator
    auto accumulator = ct::zeros<f32_MxN>();

    // K loop
    int num_k_blocks = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for (auto k_block : ct::irange(0, num_k_blocks)) {
        int k_start = k_block * BLOCK_SIZE_K;
        auto offs_k = ct::iota<i32_K>() + k_start;

        // For FC2, A is indexed by offs_token (sorted_id) since that's where FC1 wrote the data
        auto offs_k_2d = ct::reshape(offs_k, ct::shape<1, BLOCK_SIZE_K>{});
        auto a_offsets = offs_row * stride_am + offs_k_2d * stride_ak;
        auto a_ptrs = a_ptr + a_offsets;
        auto k_mask = offs_k_2d < K;
        auto a_mask = row_mask & k_mask;
        auto a = ct::load_masked(a_ptrs, a_mask, zero_scalar<INPUT_T>());

        // Load A scale - indexed by original token (offs_token // top_k)
        int k_scale_idx = k_start / group_k;
        auto original_token = offs_token / top_k;
        auto a_scale_ptrs = a_scale_ptr + original_token * stride_asm + k_scale_idx * stride_ask;
        auto a_scale = ct::load_masked(a_scale_ptrs, token_mask, 0.0f);

        // Load B using partition_view: shape (E, N, K), load (off_experts, pid_n, k_block)
        auto b_layout = ct::layout_right_mapping{ct::extents{M, N, K}};
        auto pB = ct::partition_view{ct::tensor_span{b_ptr, b_layout}, ct::shape<1, BLOCK_SIZE_N, BLOCK_SIZE_K>{}};
        auto b_3d = pB.load_masked(off_experts, pid_n, k_block);
        auto b_raw = ct::reshape(b_3d, ct::shape<BLOCK_SIZE_N, BLOCK_SIZE_K>{});
        auto b = ct::transpose(b_raw);

        // Load B scale
        int n_scale_idx = (pid_n * BLOCK_SIZE_N) / group_n;
        float b_scale_val = b_scale_ptr[off_experts * stride_bse + n_scale_idx * stride_bsn + k_scale_idx * stride_bsk];

        // Matrix multiply
        auto dot = ct::mma(a, b, ct::zeros<f32_MxN>());

        // Apply scales
        auto a_scale_2d = ct::reshape(a_scale, ct::shape<BLOCK_SIZE_M, 1>{});
        accumulator = accumulator + dot * a_scale_2d * b_scale_val;
    }

    // Optional multiplication with routed weights
    if constexpr (MUL_ROUTED_WEIGHT) {
        auto weight_ptrs = topk_weights_ptr + offs_token;
        auto moe_weight = ct::load_masked(weight_ptrs, token_mask, zero_scalar<T>());
        auto moe_weight_f32 = ct::element_cast<float>(moe_weight);
        auto moe_weight_2d = ct::reshape(moe_weight_f32, ct::shape<BLOCK_SIZE_M, 1>{});
        accumulator = accumulator * moe_weight_2d;
    }

    // Convert to output type
    auto result = ct::element_cast<T>(accumulator);

    // Store output - scatter to original token positions
    auto offs_cn = ct::iota<i32_N>() + pid_n * BLOCK_SIZE_N;
    auto offs_token_out = ct::reshape(offs_token, ct::shape<BLOCK_SIZE_M, 1>{});
    auto offs_cn_out = ct::reshape(offs_cn, ct::shape<1, BLOCK_SIZE_N>{});
    auto c_offsets = offs_token_out * stride_cm + offs_cn_out * stride_cn;
    auto c_ptrs = c_ptr + c_offsets;

    auto token_mask_out = ct::reshape(token_mask, ct::shape<BLOCK_SIZE_M, 1>{});
    auto n_mask_out = offs_cn_out < N;
    auto c_mask = token_mask_out & n_mask_out;

    ct::store_masked(c_ptrs, result, c_mask);
}
