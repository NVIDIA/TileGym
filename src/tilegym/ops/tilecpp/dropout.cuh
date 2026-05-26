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
 *   NUM_BLOCKS: Total number of blocks (N_ELEMENTS / BLOCK_SIZE)
 *
 * Parameters:
 *   x_ptr: Pointer to input tensor
 *   output_ptr: Pointer to output tensor
 *   p: Dropout probability (0.0 to 1.0)
 *   seed: Random seed for deterministic dropout mask
 */
template<typename T, int BLOCK_SIZE, int NUM_BLOCKS>
__tile_global__ void seeded_dropout_kernel(
    const T* __restrict__ x_ptr,
    T* __restrict__ output_ptr,
    float p,
    uint64_t seed
) {
    namespace ct = cuda::tiles;

    // Add alignment hints for better memory access
    x_ptr = ct::assume_aligned<16>(x_ptr);
    output_ptr = ct::assume_aligned<16>(output_ptr);

    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using f32xN = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    int bid = ct::bid().x;

    float scale = 1.0f / (1.0f - p);
    // Python passes seed already mixed into a 32-bit space via _mix_seed; keep
    // the full 32 bits here. A modulo by a Mersenne prime would alias distinct
    // 32-bit seeds and shrink the effective seed space.
    int seed_i32 = static_cast<int>(seed);

    int tile_start = bid * BLOCK_SIZE;
    auto offsets = ct::iota<i32xN>() + tile_start;

    auto x_ptrs = x_ptr + offsets;
    auto x_raw = ct::load(x_ptrs);
    auto x = ct::element_cast<float>(x_raw);

    // combined = offsets * 1103515245 + seed
    auto combined = offsets * 1103515245 + ct::full<i32xN>(seed_i32);

    auto hash_val = combined ^ (combined >> 16);
    hash_val = hash_val ^ (hash_val << 8);
    hash_val = hash_val ^ (hash_val >> 4);

    // Convert to float in [0, 1): clear sign bit, cast, normalize
    auto hash_positive = hash_val & 0x7FFFFFFF;
    auto hash_float = ct::element_cast<float>(hash_positive);
    auto random = hash_float * (1.0f / 2147483647.0f);

    // x_keep = random > p
    auto x_keep = random > p;

    auto scaled_x = x * scale;
    auto output_f32 = ct::select(x_keep, scaled_x, ct::zeros<f32xN>());

    // Convert back to T and store using pointer + scatter pattern
    auto output = ct::element_cast<T>(output_f32);
    auto out_ptrs = output_ptr + offsets;
    ct::store(out_ptrs, output);
}
