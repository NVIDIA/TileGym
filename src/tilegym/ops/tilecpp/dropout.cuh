// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ Dropout Kernel
 * Seeded dropout with deterministic random mask generation.
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * Seeded dropout kernel with one CTA per block.
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_SIZE: Number of elements processed per block (must be power of 2)
 *   N_ELEMENTS: Number of elements in the tensor
 *   P: Dropout probability
 *   SEED: Pre-mixed random seed
 *
 * Parameters:
 *   x_ptr: Pointer to input tensor
 *   output_ptr: Pointer to output tensor
 */
template<typename T, int BLOCK_SIZE, int N_ELEMENTS, float P, uint32_t SEED>
__tile_global__ void seeded_dropout_kernel(
    const T* __restrict__ x_ptr,
    T* __restrict__ output_ptr
) {
    namespace ct = cuda::tiles;

    // Add alignment hints for better memory access
    x_ptr = ct::assume_aligned<16>(x_ptr);
    output_ptr = ct::assume_aligned<16>(output_ptr);

    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;
    using u32xN = ct::tile<uint32_t, ct::shape<BLOCK_SIZE>>;

    int bid = ct::bid().x;

    // p == 1 drops everything; 1/(1-p) would be inf and poison the kept lanes.
    constexpr float scale = (P >= 1.0f) ? 0.0f : 1.0f / (1.0f - P);
    // Python passes seed already mixed into a 32-bit space via _mix_seed; keep
    // the full 32 bits here. A modulo by a Mersenne prime would alias distinct
    // 32-bit seeds and shrink the effective seed space.
    constexpr int seed_i32 = static_cast<int>(SEED);

    int tile_start = bid * BLOCK_SIZE;
    auto offsets = ct::iota<i32xN>() + tile_start;

    // gather_scatter_view is available starting with CUDA 13.4.
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
    auto x_span = ct::tensor_span{x_ptr, ct::extents<uint32_t, N_ELEMENTS>{}};
    auto x_view = ct::gather_scatter_view{
        x_span, ct::shape<BLOCK_SIZE>{}, ct::integral_constant<0>{}};
    auto x = x_view.load_masked(offsets);
#else
    auto mask = offsets < ct::full<i32xN>(N_ELEMENTS);
    auto x = ct::load_masked(x_ptr + offsets, mask, T(0));
#endif

    // combined = offsets * 1103515245 + seed, computed in uint32 so the
    // multiply wraps; signed overflow would be undefined behaviour.
    auto combined_u32 = ct::element_cast<uint32_t>(offsets) * 1103515245u
                      + ct::full<u32xN>(static_cast<uint32_t>(seed_i32));
    auto combined = ct::element_cast<int>(combined_u32);

    auto hash_val = combined ^ (combined >> 16);
    hash_val = hash_val ^ (hash_val << 8);
    hash_val = hash_val ^ (hash_val >> 4);

    // Convert to float in [0, 1): clear sign bit, cast, normalize
    auto hash_positive = hash_val & 0x7FFFFFFF;
    auto hash_float = ct::element_cast<float>(hash_positive);
    auto random = hash_float / 2147483647.0f;

    // x_keep = random > p
    auto x_keep = random > P;

    auto scaled_x = x * ct::full<TxN>(scale);
    auto output = ct::select(x_keep, scaled_x, ct::zeros<TxN>());
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
    auto output_span = ct::tensor_span{output_ptr, ct::extents<uint32_t, N_ELEMENTS>{}};
    auto output_view = ct::gather_scatter_view{
        output_span, ct::shape<BLOCK_SIZE>{}, ct::integral_constant<0>{}};
    output_view.store_masked(output, offsets);
#else
    ct::store_masked(output_ptr + offsets, output, mask);
#endif
}
