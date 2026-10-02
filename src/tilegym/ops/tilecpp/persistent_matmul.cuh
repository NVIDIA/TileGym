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
#include <cuda_tf32.h>
#include <type_traits>

template<int LATENCY, typename View, typename... Indices>
__tile__ inline auto persistent_matmul_load(View& view, Indices... indices) {
    typename View::view_tile_type value;
    if constexpr (LATENCY > 0) {
        [[cutile::hint(0, latency=LATENCY)]]
        value = view.template load_masked<cuda::tiles::view_padding::zero>(indices...);
    } else {
        value = view.template load_masked<cuda::tiles::view_padding::zero>(indices...);
    }
    return value;
}

template<typename T,
         int M, int N, int K,
         int TILE_SIZE_M, int TILE_SIZE_N, int TILE_SIZE_K,
         int GROUP_SIZE_M,
         bool TRANSPOSE_A, bool TRANSPOSE_B,
         int num_ctas, int occupancy, int LOAD_LATENCY>
[[ using cutile : hint(0, num_cta_in_cga=num_ctas, occupancy=occupancy) ]]
__tile_global__ void static_persistent_matmul_kernel(
    const T* __restrict__ _A,
    const T* __restrict__ _B,
    T* __restrict__ _C,
    int runtime_k
) {
    namespace ct = cuda::tiles;
    runtime_k = ct::assume_bounded_below<0>(runtime_k);

    const T* A = ct::assume_aligned<16>(_A);
    const T* B = ct::assume_aligned<16>(_B);
    T* C       = ct::assume_aligned<16>(_C);

    int start_bid    = ct::bid().x;
    int num_programs = ct::num_blocks().x;

    constexpr int num_bid_m        = (M + TILE_SIZE_M - 1) / TILE_SIZE_M;
    constexpr int num_bid_n        = (N + TILE_SIZE_N - 1) / TILE_SIZE_N;
    int k_tiles = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, __nv_bfloat16>) {
        k_tiles = (runtime_k + TILE_SIZE_K - 1) / TILE_SIZE_K;
    }
    constexpr int num_tiles        = num_bid_m * num_bid_n;
    constexpr int num_bid_in_group = GROUP_SIZE_M * num_bid_n;
    constexpr bool output_tiles_are_full = (M % TILE_SIZE_M == 0) && (N % TILE_SIZE_N == 0);


    using AccType = std::conditional_t<std::is_same_v<T, double>, double, float>;
    using AccTile = ct::tile<AccType, ct::shape<TILE_SIZE_M, TILE_SIZE_N>>;
    using MmaType = std::conditional_t<std::is_same_v<T, float>, __nv_tf32, T>;

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
                auto a = ct::element_cast<MmaType>(persistent_matmul_load<LOAD_LATENCY>(pA, bid_m, k));
                auto b = ct::element_cast<MmaType>(persistent_matmul_load<LOAD_LATENCY>(pB, k, bid_n));
                acc = ct::mma(a, b, acc);
            }
            auto result = ct::element_cast<T>(acc);
            if constexpr (output_tiles_are_full) {
                pC.store(result, bid_m, bid_n);
            } else {
                pC.store_masked(result, bid_m, bid_n);
            }
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
                auto a_raw = persistent_matmul_load<LOAD_LATENCY>(pA, k, bid_m);
                auto a     = ct::element_cast<MmaType>(ct::transpose(a_raw));
                auto b     = ct::element_cast<MmaType>(persistent_matmul_load<LOAD_LATENCY>(pB, k, bid_n));
                acc = ct::mma(a, b, acc);
            }
            auto result = ct::element_cast<T>(acc);
            if constexpr (output_tiles_are_full) {
                pC.store(result, bid_m, bid_n);
            } else {
                pC.store_masked(result, bid_m, bid_n);
            }
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
                auto a     = ct::element_cast<MmaType>(persistent_matmul_load<LOAD_LATENCY>(pA, bid_m, k));
                auto b_raw = persistent_matmul_load<LOAD_LATENCY>(pB, bid_n, k);
                auto b     = ct::element_cast<MmaType>(ct::transpose(b_raw));
                acc = ct::mma(a, b, acc);
            }
            auto result = ct::element_cast<T>(acc);
            if constexpr (output_tiles_are_full) {
                pC.store(result, bid_m, bid_n);
            } else {
                pC.store_masked(result, bid_m, bid_n);
            }
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
                auto a_raw = persistent_matmul_load<LOAD_LATENCY>(pA, k, bid_m);
                auto b_raw = persistent_matmul_load<LOAD_LATENCY>(pB, bid_n, k);
                auto a     = ct::element_cast<MmaType>(ct::transpose(a_raw));
                auto b     = ct::element_cast<MmaType>(ct::transpose(b_raw));
                acc = ct::mma(a, b, acc);
            }
            auto result = ct::element_cast<T>(acc);
            if constexpr (output_tiles_are_full) {
                pC.store(result, bid_m, bid_n);
            } else {
                pC.store_masked(result, bid_m, bid_n);
            }
        }
    }
}
