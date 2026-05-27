// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * CUDA Tile C++ Matrix Multiplication Kernel
 *
 * C = A @ B (with optional transposes)
 * - A is [M, K] or [K, M] if transposed
 * - B is [K, N] or [N, K] if transposed
 * - C is [M, N]
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

/**
 * Matrix Multiplication Kernel with transpose support
 *
 * Template Parameters:
 *   T: Element type (__half, __nv_bfloat16, float)
 *   M, N, K: Matrix dimensions (promoted to template params for static shapes)
 *   TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K: Tile dimensions
 *   GROUP_SIZE_M: Number of M tiles to group for L2 reuse
 *   num_tiles_k: Number of K tiles (for compile-time loop optimization)
 *   TRANSPOSE_A, TRANSPOSE_B: Whether to transpose inputs
 */
template<typename T, int M, int N, int K,
         int TILE_SIZE_M, int TILE_SIZE_N, int TILE_SIZE_K, int GROUP_SIZE_M, int num_tiles_k,
         bool TRANSPOSE_A, bool TRANSPOSE_B,
         int num_ctas, int occupancy>
[[ using cutile :
    hint(1000, num_cta_in_cga=num_ctas),
    hint(1000, occupancy=occupancy)
]]
__tile_global__ void matmul_kernel(
    const T* __restrict__ _A,
    const T* __restrict__ _B,
    T* __restrict__ _C
) {
    namespace ct = cuda::tiles;

    // Apply alignment hints (128-byte = cache line)
    const T* A = ct::assume_aligned<16>(_A);
    const T* B = ct::assume_aligned<16>(_B);
    T* C = ct::assume_aligned<16>(_C);

    // Tile counts
    constexpr int num_bid_m = (M + TILE_SIZE_M - 1) / TILE_SIZE_M;
    constexpr int num_bid_n = (N + TILE_SIZE_N - 1) / TILE_SIZE_N;
    constexpr int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

    // 2D swizzle for L2 cache reuse
    int bidval = ct::bid().x;
    int group_id = bidval / num_bid_in_group;
    int first_bid_m = group_id * GROUP_SIZE_M;
    int group_size_m = ct::min(num_bid_m - first_bid_m, GROUP_SIZE_M);
    int bidx = first_bid_m + (bidval % group_size_m);
    int bidy = (bidval % num_bid_in_group) / group_size_m;


    // Type aliases for tiles
    using ATile = ct::tile<T, ct::shape<TILE_SIZE_M, TILE_SIZE_K>>;
    using BTile = ct::tile<T, ct::shape<TILE_SIZE_K, TILE_SIZE_N>>;
    using AccTile = ct::tile<float, ct::shape<TILE_SIZE_M, TILE_SIZE_N>>;
    using CTile = ct::tile<T, ct::shape<TILE_SIZE_M, TILE_SIZE_N>>;

    // Initialize accumulator
    AccTile acc = ct::zeros<AccTile>();

    // K-dimension accumulation loop with transpose handling
    constexpr auto zero_pad = ct::view_padding::zero;

    if constexpr (!TRANSPOSE_A && !TRANSPOSE_B) {
        // A is [M, K], B is [K, N] - most common case
        auto pA = ct::partition_view{ct::tensor_span{A, ct::extents<uint32_t, M, K>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_K>{}};
        auto pB = ct::partition_view{ct::tensor_span{B, ct::extents<uint32_t, K, N>{}}, ct::shape<TILE_SIZE_K, TILE_SIZE_N>{}};

        for (auto k : ct::irange(0, num_tiles_k)) {
            auto a = pA.template load_masked<zero_pad>(bidx, k);
            auto b = pB.template load_masked<zero_pad>(k, bidy);
            acc = ct::mma(a, b, acc);
        }
    } else if constexpr (TRANSPOSE_A && !TRANSPOSE_B) {
        // A is stored as [K, M], B is [K, N]
        auto pA = ct::partition_view{ct::tensor_span{A, ct::extents<uint32_t, K,M>{}}, ct::shape<TILE_SIZE_K, TILE_SIZE_M>{}};
        auto pB = ct::partition_view{ct::tensor_span{B, ct::extents<uint32_t, K, N>{}}, ct::shape<TILE_SIZE_K, TILE_SIZE_N>{}};

        for (auto k : ct::irange(0, num_tiles_k)) {
            auto a_raw = pA.template load_masked<zero_pad>(k, bidx);
            auto b = pB.template load_masked<zero_pad>(k, bidy);
            auto a = ct::transpose(a_raw);  // [TILE_K, TILE_M] -> [TILE_M, TILE_K]
            acc = ct::mma(a, b, acc);
        }
    } else if constexpr (!TRANSPOSE_A && TRANSPOSE_B) {
        // A is [M, K], B is stored as [N, K]
        auto pA = ct::partition_view{ct::tensor_span{A, ct::extents<uint32_t, M, K>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_K>{}};
        auto pB = ct::partition_view{ct::tensor_span{B, ct::extents<uint32_t, N, K>{}}, ct::shape<TILE_SIZE_N, TILE_SIZE_K>{}};

        for (auto k : ct::irange(0, num_tiles_k)) {
            auto a = pA.template load_masked<zero_pad>(bidx, k);
            auto b_raw = pB.template load_masked<zero_pad>(bidy, k);
            auto b = ct::transpose(b_raw);  // [TILE_N, TILE_K] -> [TILE_K, TILE_N]
            acc = ct::mma(a, b, acc);
        }
    } else {
        // A is stored as [K, M], B is stored as [N, K]
        auto pA = ct::partition_view{ct::tensor_span{A, ct::extents<uint32_t, K, M>{}}, ct::shape<TILE_SIZE_K, TILE_SIZE_M>{}};
        auto pB = ct::partition_view{ct::tensor_span{B, ct::extents<uint32_t, N, K>{}}, ct::shape<TILE_SIZE_N, TILE_SIZE_K>{}};

        for (auto k : ct::irange(0, num_tiles_k)) {
            auto a_raw = pA.template load_masked<zero_pad>(k, bidx);
            auto b_raw = pB.template load_masked<zero_pad>(bidy, k);
            auto a = ct::transpose(a_raw);  // [TILE_K, TILE_M] -> [TILE_M, TILE_K]
            auto b = ct::transpose(b_raw);  // [TILE_N, TILE_K] -> [TILE_K, TILE_N]
            acc = ct::mma(a, b, acc);
        }
    }

    // Create output tensor span and partition view
    auto pC = ct::partition_view{ct::tensor_span{C, ct::extents<uint32_t, M, N>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_N>{}};

    // Cast and store result
    pC.store(ct::element_cast<T>(acc), bidx, bidy);
}
