// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ Batch Matrix Multiplication Kernels
 *
 * Computes C = A x B for batched 3D tensors using 3D tensor_span + partition_view.
 * A has shape (Q, M, K), B has shape (Q, K, N), C has shape (Q, M, N)
 *
 * Supports transpose of A and/or B.
 * Templated on element type for float16/bfloat16/float32 support.
 *
 * NOTE: The K-dimension accumulation loop cannot be eliminated - it's mathematically
 * required for matrix multiplication: C[m,n] = sum(A[m,k] * B[k,n] for all k).
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * Batch Matrix Multiplication Kernel
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions
 *   GROUP_SIZE_M: Number of M-tiles to group for L2 reuse
 *   TRANSPOSE_A, TRANSPOSE_B: Whether to transpose inputs
 */
template<typename T, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int GROUP_SIZE_M, bool TRANSPOSE_A, bool TRANSPOSE_B>
__tile_global__ void bmm_kernel(
    const T* __restrict__ _a_ptr,
    const T* __restrict__ _b_ptr,
    T* __restrict__ _c_ptr,
    int Q,  // Batch size
    int M,
    int N,
    int K
) {
    namespace ct = cuda::tiles;
    const T* a_ptr = ct::assume_aligned<16>(_a_ptr);
    const T* b_ptr = ct::assume_aligned<16>(_b_ptr);
    T* c_ptr = ct::assume_aligned<16>(_c_ptr);

    // Tile type definitions
    using AccTile = ct::tile<float, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_N>>;

    int pid = ct::bid().x;
    int pid_q = ct::bid().y;  // Batch index

    // Grouped ordering for L2 reuse
    int num_pid_m = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int num_pid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    int num_pid_in_group = GROUP_SIZE_M * num_pid_n;
    int group_id = pid / num_pid_in_group;
    int first_pid_m = group_id * GROUP_SIZE_M;
    int group_size_m = (num_pid_m - first_pid_m < GROUP_SIZE_M)
                     ? (num_pid_m - first_pid_m) : GROUP_SIZE_M;
    int pid_m = first_pid_m + (pid % group_size_m);
    int pid_n = (pid % num_pid_in_group) / group_size_m;

    // Early exit for out-of-bounds blocks
    if (pid_m >= num_pid_m || pid_n >= num_pid_n) {
        return;
    }

    // Initialize accumulator in float32 for precision
    auto accumulator = ct::zeros<AccTile>();

    // K-dimension loop
    int num_k_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
    for (auto k_tile : ct::irange(0, num_k_tiles)) {
        // Load A tile
        ct::tile<T, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_K>> a_tile;
        if constexpr (TRANSPOSE_A) {
            // A is transposed: physical layout (Q, K, M), load as (1, TILE_K, TILE_M)
            auto a_layout = ct::layout_right_mapping{ct::extents{Q, K, M}};
            auto pA = ct::partition_view{ct::tensor_span{a_ptr, a_layout}, ct::shape<1, BLOCK_SIZE_K, BLOCK_SIZE_M>{}};
            auto a_tile_3d = pA.load_masked(pid_q, k_tile, pid_m);
            // Reshape and transpose to get (TILE_M, TILE_K)
            auto a_raw = ct::reshape(a_tile_3d, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_M>{});
            a_tile = ct::transpose(a_raw);
        } else {
            // A is normal: physical layout (Q, M, K)
            auto a_layout = ct::layout_right_mapping{ct::extents{Q, M, K}};
            auto pA = ct::partition_view{ct::tensor_span{a_ptr, a_layout}, ct::shape<1, BLOCK_SIZE_M, BLOCK_SIZE_K>{}};
            auto a_tile_3d = pA.load_masked(pid_q, pid_m, k_tile);
            a_tile = ct::reshape(a_tile_3d, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_K>{});
        }

        // Load B tile
        ct::tile<T, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_N>> b_tile;
        if constexpr (TRANSPOSE_B) {
            // B is transposed: physical layout (Q, N, K), load and transpose
            auto b_layout = ct::layout_right_mapping{ct::extents{Q, N, K}};
            auto pB = ct::partition_view{ct::tensor_span{b_ptr, b_layout}, ct::shape<1, BLOCK_SIZE_N, BLOCK_SIZE_K>{}};
            auto b_tile_3d = pB.load_masked(pid_q, pid_n, k_tile);
            // Reshape and transpose to get (TILE_K, TILE_N)
            auto b_raw = ct::reshape(b_tile_3d, ct::shape<BLOCK_SIZE_N, BLOCK_SIZE_K>{});
            b_tile = ct::transpose(b_raw);
        } else {
            // B is normal: physical layout (Q, K, N)
            auto b_layout = ct::layout_right_mapping{ct::extents{Q, K, N}};
            auto pB = ct::partition_view{ct::tensor_span{b_ptr, b_layout}, ct::shape<1, BLOCK_SIZE_K, BLOCK_SIZE_N>{}};
            auto b_tile_3d = pB.load_masked(pid_q, k_tile, pid_n);
            b_tile = ct::reshape(b_tile_3d, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_N>{});
        }

        // Matrix multiplication and accumulation
        accumulator = ct::mma(a_tile, b_tile, accumulator);
    }

    // Convert to output dtype and store result
    auto result = ct::element_cast<T>(accumulator);
    auto c_layout = ct::layout_right_mapping{ct::extents{Q, M, N}};
    auto pC = ct::partition_view{ct::tensor_span{c_ptr, c_layout}, ct::shape<1, BLOCK_SIZE_M, BLOCK_SIZE_N>{}};
    auto result_3d = ct::reshape(result, ct::shape<1, BLOCK_SIZE_M, BLOCK_SIZE_N>{});
    pC.store_masked(result_3d, pid_q, pid_m, pid_n);
}

/**
 * Static Persistent Batch Matrix Multiplication Kernel
 *
 * This kernel uses static persistent scheduling where each thread block
 * processes multiple tiles in a loop, improving GPU utilization and
 * reducing kernel launch overhead.
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K: Tile dimensions
 *   GROUP_SIZE_M: Number of M-tiles to group for L2 reuse
 *   TRANSPOSE_A, TRANSPOSE_B: Whether to transpose inputs
 *   Q: Batch size
 *   M, N, K: Matrix dimensions
 */
template<typename T, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int GROUP_SIZE_M, bool TRANSPOSE_A, bool TRANSPOSE_B, int Q, int M, int N, int K, int num_ctas, int occupancy>
[[ using cutile :
    hint(1000, num_cta_in_cga=num_ctas),
    hint(1000, occupancy=occupancy)
]]
__tile_global__ void bmm_static_persistent_kernel(
    const T* __restrict__ _a_ptr,
    const T* __restrict__ _b_ptr,
    T* __restrict__ _c_ptr
) {
    namespace ct = cuda::tiles;
    const T* a_ptr = ct::assume_aligned<16>(_a_ptr);
    const T* b_ptr = ct::assume_aligned<16>(_b_ptr);
    T* c_ptr = ct::assume_aligned<16>(_c_ptr);

    // Tile type definitions
    using AccTile = ct::tile<float, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_N>>;

    int bid = ct::bid().x;  // Current program ID
    int num_programs = ct::num_blocks().x;

    // Calculate total number of tiles
    int num_tiles_m = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
    int num_tiles_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
    int total_tiles = num_tiles_m * num_tiles_n * Q;

    // Static persistent scheduling loop using irange with stride
    for (auto current_bid : ct::irange(bid, total_tiles, num_programs)) {
        // Calculate BID coordinates using grouped ordering
        int num_bid_m = (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M;
        int num_bid_n = (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N;
        int bid_q = current_bid / (num_bid_m * num_bid_n);
        int num_bid_in_group = GROUP_SIZE_M * num_bid_n;

        int current_bid_2d = current_bid % (num_bid_m * num_bid_n);
        int group_id = current_bid_2d / num_bid_in_group;
        int first_bid_m = group_id * GROUP_SIZE_M;
        int group_size_m_temp = num_bid_m - first_bid_m;
        int group_size_m = (group_size_m_temp < GROUP_SIZE_M) ? group_size_m_temp : GROUP_SIZE_M;
        int bid_m = first_bid_m + (current_bid_2d % group_size_m);
        int bid_n = (current_bid_2d % num_bid_in_group) / group_size_m;

        // Initialize accumulator (2D to avoid reshape overhead)
        auto accumulator = ct::zeros<AccTile>();

        // K-dimension loop
        int num_k_tiles = (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K;
        for (auto k_tile : ct::irange(0, num_k_tiles)) {
            // Load A tile
            ct::tile<T, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_K>> a_tile;
            if constexpr (TRANSPOSE_A) {
                // A is transposed: physical layout (Q, K, M), load as (1, TILE_K, TILE_M)
                auto a_layout = ct::layout_right_mapping{ct::extents{Q, K, M}};
                auto pA = ct::partition_view{ct::tensor_span{a_ptr, a_layout}, ct::shape<1, BLOCK_SIZE_K, BLOCK_SIZE_M>{}};
                auto a_tile_3d = pA.load_masked(bid_q, k_tile, bid_m);
                // Reshape and transpose to get (TILE_M, TILE_K)
                auto a_raw = ct::reshape(a_tile_3d, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_M>{});
                a_tile = ct::transpose(a_raw);
            } else {
                // A is normal: physical layout (Q, M, K)
                auto a_layout = ct::layout_right_mapping{ct::extents{Q, M, K}};
                auto pA = ct::partition_view{ct::tensor_span{a_ptr, a_layout}, ct::shape<1, BLOCK_SIZE_M, BLOCK_SIZE_K>{}};
                auto a_tile_3d = pA.load_masked(bid_q, bid_m, k_tile);
                a_tile = ct::reshape(a_tile_3d, ct::shape<BLOCK_SIZE_M, BLOCK_SIZE_K>{});
            }

            // Load B tile
            ct::tile<T, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_N>> b_tile;
            if constexpr (TRANSPOSE_B) {
                // B is transposed: physical layout (Q, N, K), load and transpose
                auto b_layout = ct::layout_right_mapping{ct::extents{Q, N, K}};
                auto pB = ct::partition_view{ct::tensor_span{b_ptr, b_layout}, ct::shape<1, BLOCK_SIZE_N, BLOCK_SIZE_K>{}};
                auto b_tile_3d = pB.load_masked(bid_q, bid_n, k_tile);
                // Reshape and transpose to get (TILE_K, TILE_N)
                auto b_raw = ct::reshape(b_tile_3d, ct::shape<BLOCK_SIZE_N, BLOCK_SIZE_K>{});
                b_tile = ct::transpose(b_raw);
            } else {
                // B is normal: physical layout (Q, K, N)
                auto b_layout = ct::layout_right_mapping{ct::extents{Q, K, N}};
                auto pB = ct::partition_view{ct::tensor_span{b_ptr, b_layout}, ct::shape<1, BLOCK_SIZE_K, BLOCK_SIZE_N>{}};
                auto b_tile_3d = pB.load_masked(bid_q, k_tile, bid_n);
                b_tile = ct::reshape(b_tile_3d, ct::shape<BLOCK_SIZE_K, BLOCK_SIZE_N>{});
            }

            // Matrix multiplication and accumulation
            accumulator = ct::mma(a_tile, b_tile, accumulator);
        }

        // Convert to output dtype and store result
        auto result = ct::element_cast<T>(accumulator);
        auto c_layout = ct::layout_right_mapping{ct::extents{Q, M, N}};
        auto pC = ct::partition_view{ct::tensor_span{c_ptr, c_layout}, ct::shape<1, BLOCK_SIZE_M, BLOCK_SIZE_N>{}};
        auto result_3d = ct::reshape(result, ct::shape<1, BLOCK_SIZE_M, BLOCK_SIZE_N>{});
        pC.store_masked(result_3d, bid_q, bid_m, bid_n);
    }
}
