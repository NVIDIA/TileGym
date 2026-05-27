// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * CUDA Tile C++ Static-Persistent Matrix Multiplication Kernel.
 *
 * Grid (set at launch): min(NUM_SMS // num_ctas, num_tiles) * occupancy.
 * Each CTA walks `tile_id in irange(start_bid, num_tiles, num_programs)` and
 * does the full K-reduction per tile, emitting one output tile each.
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

template<typename T,
         int M, int N, int K,
         int TILE_SIZE_M, int TILE_SIZE_N, int TILE_SIZE_K,
         int GROUP_SIZE_M,
         bool TRANSPOSE_A, bool TRANSPOSE_B,
         int num_ctas, int occupancy>
[[ using cutile : hint(1000, num_cta_in_cga=num_ctas, occupancy=occupancy) ]]
__tile_global__ void static_persistent_matmul_kernel(
    const T* __restrict__ _A,
    const T* __restrict__ _B,
    T* __restrict__ _C
) {
    namespace ct = cuda::tiles;

    const T* A = ct::assume_aligned<16>(_A);
    const T* B = ct::assume_aligned<16>(_B);
    T* C       = ct::assume_aligned<16>(_C);

    int start_bid    = ct::bid().x;
    int num_programs = ct::num_blocks().x;

    constexpr int num_bid_m        = (M + TILE_SIZE_M - 1) / TILE_SIZE_M;
    constexpr int num_bid_n        = (N + TILE_SIZE_N - 1) / TILE_SIZE_N;
    constexpr int k_tiles          = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;
    constexpr int num_tiles        = num_bid_m * num_bid_n;
    constexpr int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

    constexpr auto zero_pad = ct::view_padding::zero;

    using AccTile = ct::tile<float, ct::shape<TILE_SIZE_M, TILE_SIZE_N>>;

    if constexpr (!TRANSPOSE_A && !TRANSPOSE_B) {
        auto pA = ct::partition_view{ct::tensor_span{A, ct::extents<uint32_t, M, K>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_K>{}};
        auto pB = ct::partition_view{ct::tensor_span{B, ct::extents<uint32_t, K, N>{}}, ct::shape<TILE_SIZE_K, TILE_SIZE_N>{}};
        auto pC = ct::partition_view{ct::tensor_span{C, ct::extents<uint32_t, M, N>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_N>{}};

        for (auto tile_id : ct::irange(start_bid, num_tiles, num_programs)) {
            int group_id     = tile_id / num_bid_in_group;
            int first_bid_m  = group_id * GROUP_SIZE_M;
            int group_size_m = ct::min(num_bid_m - first_bid_m, GROUP_SIZE_M);
            int bid_m        = first_bid_m + (tile_id % group_size_m);
            int bid_n        = (tile_id % num_bid_in_group) / group_size_m;

            AccTile acc = ct::zeros<AccTile>();
            for (auto k : ct::irange(0, k_tiles)) {
                auto a = pA.template load_masked<zero_pad>(bid_m, k);
                auto b = pB.template load_masked<zero_pad>(k, bid_n);
                acc = ct::mma(a, b, acc);
            }
            pC.store(ct::element_cast<T>(acc), bid_m, bid_n);
        }
    } else if constexpr (TRANSPOSE_A && !TRANSPOSE_B) {
        auto pA = ct::partition_view{ct::tensor_span{A, ct::extents<uint32_t, K, M>{}}, ct::shape<TILE_SIZE_K, TILE_SIZE_M>{}};
        auto pB = ct::partition_view{ct::tensor_span{B, ct::extents<uint32_t, K, N>{}}, ct::shape<TILE_SIZE_K, TILE_SIZE_N>{}};
        auto pC = ct::partition_view{ct::tensor_span{C, ct::extents<uint32_t, M, N>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_N>{}};

        for (auto tile_id : ct::irange(start_bid, num_tiles, num_programs)) {
            int group_id     = tile_id / num_bid_in_group;
            int first_bid_m  = group_id * GROUP_SIZE_M;
            int group_size_m = ct::min(num_bid_m - first_bid_m, GROUP_SIZE_M);
            int bid_m        = first_bid_m + (tile_id % group_size_m);
            int bid_n        = (tile_id % num_bid_in_group) / group_size_m;

            AccTile acc = ct::zeros<AccTile>();
            for (auto k : ct::irange(0, k_tiles)) {
                auto a_raw = pA.template load_masked<zero_pad>(k, bid_m);
                auto b     = pB.template load_masked<zero_pad>(k, bid_n);
                auto a     = ct::transpose(a_raw);
                acc = ct::mma(a, b, acc);
            }
            pC.store(ct::element_cast<T>(acc), bid_m, bid_n);
        }
    } else if constexpr (!TRANSPOSE_A && TRANSPOSE_B) {
        auto pA = ct::partition_view{ct::tensor_span{A, ct::extents<uint32_t, M, K>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_K>{}};
        auto pB = ct::partition_view{ct::tensor_span{B, ct::extents<uint32_t, N, K>{}}, ct::shape<TILE_SIZE_N, TILE_SIZE_K>{}};
        auto pC = ct::partition_view{ct::tensor_span{C, ct::extents<uint32_t, M, N>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_N>{}};

        for (auto tile_id : ct::irange(start_bid, num_tiles, num_programs)) {
            int group_id     = tile_id / num_bid_in_group;
            int first_bid_m  = group_id * GROUP_SIZE_M;
            int group_size_m = ct::min(num_bid_m - first_bid_m, GROUP_SIZE_M);
            int bid_m        = first_bid_m + (tile_id % group_size_m);
            int bid_n        = (tile_id % num_bid_in_group) / group_size_m;

            AccTile acc = ct::zeros<AccTile>();
            for (auto k : ct::irange(0, k_tiles)) {
                auto a     = pA.template load_masked<zero_pad>(bid_m, k);
                auto b_raw = pB.template load_masked<zero_pad>(bid_n, k);
                auto b     = ct::transpose(b_raw);
                acc = ct::mma(a, b, acc);
            }
            pC.store(ct::element_cast<T>(acc), bid_m, bid_n);
        }
    } else {
        auto pA = ct::partition_view{ct::tensor_span{A, ct::extents<uint32_t, K, M>{}}, ct::shape<TILE_SIZE_K, TILE_SIZE_M>{}};
        auto pB = ct::partition_view{ct::tensor_span{B, ct::extents<uint32_t, N, K>{}}, ct::shape<TILE_SIZE_N, TILE_SIZE_K>{}};
        auto pC = ct::partition_view{ct::tensor_span{C, ct::extents<uint32_t, M, N>{}}, ct::shape<TILE_SIZE_M, TILE_SIZE_N>{}};

        for (auto tile_id : ct::irange(start_bid, num_tiles, num_programs)) {
            int group_id     = tile_id / num_bid_in_group;
            int first_bid_m  = group_id * GROUP_SIZE_M;
            int group_size_m = ct::min(num_bid_m - first_bid_m, GROUP_SIZE_M);
            int bid_m        = first_bid_m + (tile_id % group_size_m);
            int bid_n        = (tile_id % num_bid_in_group) / group_size_m;

            AccTile acc = ct::zeros<AccTile>();
            for (auto k : ct::irange(0, k_tiles)) {
                auto a_raw = pA.template load_masked<zero_pad>(k, bid_m);
                auto b_raw = pB.template load_masked<zero_pad>(bid_n, k);
                auto a     = ct::transpose(a_raw);
                auto b     = ct::transpose(b_raw);
                acc = ct::mma(a, b, acc);
            }
            pC.store(ct::element_cast<T>(acc), bid_m, bid_n);
        }
    }
}
