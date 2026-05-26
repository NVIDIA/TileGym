// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Split-K Reduce Kernel
 *
 * Reduces intermediate attention results from multiple K splits into final output.
 * This is used in attention decode with split-K optimization.
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr float SPLITK_NEG_INF = -1e30f; // Use large negative instead of -INFINITY

/**
 * Split-K Reduce Kernel - Optimized version
 *
 * Uses ct::irange for loops, __restrict__ pointers, and alignment hints.
 * Note: Tensor dimensions match actual allocation (NUM_KV_SPLITS not padded).
 * Template Parameters:
 *   T: Element type for attention tensors (float, __half, __nv_bfloat16)
 *   B, NUM_HEADS, HEAD_DIM: tensor shape constants (assume contiguous layout
 *     so strides are derived from these).
 *   NUM_KV_SPLITS: Actual number of KV splits
 *   NUM_KV_SPLITS_POW2: Next power of 2 >= NUM_KV_SPLITS
 *   BLOCK_D: Block size for head dimension
 *
 * Parameters:
 *   attn_splitk_out: [B, NUM_HEADS, NUM_KV_SPLITS, HEAD_DIM] - intermediate attention
 *   lse_splitk_out: [B, NUM_HEADS, NUM_KV_SPLITS] - log-sum-exp values (always float32)
 *   attn_out: [B, NUM_HEADS, HEAD_DIM] - output
 */
template<typename T,
         int B, int NUM_HEADS, int HEAD_DIM,
         int NUM_KV_SPLITS, int NUM_KV_SPLITS_POW2, int BLOCK_D, bool USE_DOT>
[[ using cutile : hint(1000, occupancy=4) ]]
__tile_global__ void splitk_reduce_kernel(
    const T* __restrict__ attn_splitk_out,
    const float* __restrict__ lse_splitk_out,
    T* __restrict__ attn_out
) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    attn_splitk_out = ct::assume_aligned<16>(attn_splitk_out);
    lse_splitk_out  = ct::assume_aligned<16>(lse_splitk_out);
    attn_out        = ct::assume_aligned<16>(attn_out);

    // Strides derived from template-baked shapes (contiguous layout).
    constexpr int stride_lse_rb = NUM_HEADS * NUM_KV_SPLITS;
    constexpr int stride_lse_rm = NUM_KV_SPLITS;

    // Tile type definitions
    using f32xS   = ct::tile<float, ct::shape<NUM_KV_SPLITS_POW2>>;
    using f32xD   = ct::tile<float, ct::shape<BLOCK_D>>;
    using f32x1xS = ct::tile<float, ct::shape<1, NUM_KV_SPLITS_POW2>>;
    using f32xSxD = ct::tile<float, ct::shape<NUM_KV_SPLITS_POW2, BLOCK_D>>;
    using f32x1xD = ct::tile<float, ct::shape<1, BLOCK_D>>;
    using TxD     = ct::tile<T, ct::shape<BLOCK_D>>;
    using TxSxD   = ct::tile<T, ct::shape<NUM_KV_SPLITS_POW2, BLOCK_D>>;
    using i32xS   = ct::tile<int, ct::shape<NUM_KV_SPLITS_POW2>>;
    using i32xD   = ct::tile<int, ct::shape<BLOCK_D>>;

    int batch_id = ct::bid().x;
    int head_id = ct::bid().y;
    int block_id = ct::bid().z;

    // Generate offset for splits dimension
    auto offs_s = ct::iota<i32xS>();  // [0, 1, ..., NUM_KV_SPLITS_POW2-1]
    auto offs_d = ct::iota<i32xD>();  // [0, 1, ..., BLOCK_D-1]

    constexpr auto zero_pad = ct::view_padding::zero;

    auto Att_span = ct::tensor_span{
        attn_splitk_out,
        ct::extents<uint32_t, B, NUM_HEADS, NUM_KV_SPLITS, HEAD_DIM>{}};
    auto Att_view = ct::partition_view(
        Att_span,
        ct::shape<1, 1, NUM_KV_SPLITS_POW2, BLOCK_D>{});

    auto Out_span = ct::tensor_span{
        attn_out,
        ct::extents<uint32_t, B, NUM_HEADS, HEAD_DIM>{}};
    auto Out_view = ct::partition_view(
        Out_span,
        ct::shape<1, 1, BLOCK_D>{});

    int lse_base = batch_id * stride_lse_rb + head_id * stride_lse_rm;
    auto lse_offs = lse_base + offs_s;
    auto lse_mask = offs_s < NUM_KV_SPLITS;
    auto lse_ptrs = lse_splitk_out + lse_offs;
    auto lse_splitk = ct::load_masked(
        lse_ptrs,
        lse_mask,
        ct::full<f32xS>(SPLITK_NEG_INF));

    auto lse_max_1d  = ct::reshape(ct::reduce_max(lse_splitk, 0_ic),
                                   ct::shape<1>{});
    auto lse_max_bcS = ct::broadcast(lse_max_1d, ct::shape<NUM_KV_SPLITS_POW2>{});

    // Compute sumexp_normalized_splitk = exp2(lse_splitk - lse_max).
    auto sumexp_normalized_splitk = ct::exp2(lse_splitk - lse_max_bcS);

    // Sum of sumexp_normalized — also keep in the tile world.
    auto sum_1d = ct::reshape(ct::sum(sumexp_normalized_splitk, 0_ic),
                              ct::shape<1>{});
    auto sum_bcD = ct::broadcast(sum_1d, ct::shape<BLOCK_D>{});

    using Tx4D = ct::tile<T, ct::shape<1, 1, NUM_KV_SPLITS_POW2, BLOCK_D>>;
    Tx4D out_splitk_raw_4d;
    [[ using cutile : hint(1000, latency=2) ]]
    out_splitk_raw_4d = Att_view.template load_masked<zero_pad>(
        batch_id, head_id, 0, block_id);
    auto out_splitk_raw = ct::reshape(
        out_splitk_raw_4d,
        ct::shape<NUM_KV_SPLITS_POW2, BLOCK_D>{});
    auto out_splitk = ct::element_cast<float>(out_splitk_raw);

    f32xD numerator_normalized;
    if constexpr (USE_DOT) {
        auto weights_2d = ct::reshape(sumexp_normalized_splitk,
                                      ct::shape<1, NUM_KV_SPLITS_POW2>{});
        auto mma_result = ct::mma(weights_2d, out_splitk, ct::zeros<f32x1xD>());
        numerator_normalized = ct::reshape(mma_result, ct::shape<BLOCK_D>{});
    } else {
        auto weights_col = ct::reshape(sumexp_normalized_splitk,
                                       ct::shape<NUM_KV_SPLITS_POW2, 1>{});
        auto sum_result = ct::sum(out_splitk * weights_col,
                                  ct::integral_constant<0>{});
        numerator_normalized = ct::reshape(sum_result, ct::shape<BLOCK_D>{});
    }

    // Compute final result: acc = numerator / sumexp_normalized
    auto acc = numerator_normalized / sum_bcD;

    // Convert back to output type and store via partition_view.
    auto acc_out = ct::element_cast<T>(acc);
    auto acc_out_3d = ct::reshape(acc_out, ct::shape<1, 1, BLOCK_D>{});
    [[ using cutile : hint(1000, latency=2) ]]
    Out_view.store_masked(acc_out_3d, batch_id, head_id, block_id);
}
