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
 *
 * Parameters:
 *   x_ptr: Pointer to input tensor
 *   output_ptr: Pointer to output tensor
 *   p: Dropout probability (0.0 to 1.0)
 *   seed: Random seed for deterministic dropout mask
 *   n_elements: Number of elements in the tensor
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void seeded_dropout_kernel(
    const T* __restrict__ x_ptr,
    T* __restrict__ output_ptr,
    float p,
    uint64_t seed,
    int n_elements
) {
    namespace ct = cuda::tiles;

    // Add alignment hints for better memory access
    x_ptr = ct::assume_aligned<16>(x_ptr);
    output_ptr = ct::assume_aligned<16>(output_ptr);

    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using f32xN = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;
    using u32xN = ct::tile<uint32_t, ct::shape<BLOCK_SIZE>>;

    int bid = ct::bid().x;

    // p == 1 drops everything; 1/(1-p) would be inf and poison the kept lanes.
    float scale = (p >= 1.0f) ? 0.0f : 1.0f / (1.0f - p);
    // Python passes seed already mixed into a 32-bit space via _mix_seed; keep
    // the full 32 bits here. A modulo by a Mersenne prime would alias distinct
    // 32-bit seeds and shrink the effective seed space.
    int seed_i32 = static_cast<int>(seed);

    int tile_start = bid * BLOCK_SIZE;
    auto offsets = ct::iota<i32xN>() + tile_start;

    auto mask = offsets < ct::full<i32xN>(n_elements);

    TxN x_raw = ct::load_masked(x_ptr + offsets, mask, T(0));
    auto x = ct::element_cast<float>(x_raw);

    // combined = offsets * 1103515245 + seed, computed in uint32 so the
    // multiply wraps; signed overflow would be undefined behaviour.
    auto combined = ct::element_cast<uint32_t>(offsets) * 1103515245u
                  + ct::full<u32xN>(static_cast<uint32_t>(seed_i32));

    auto hash_val = combined ^ (combined >> 16);
    hash_val = hash_val ^ (hash_val << 8);
    hash_val = hash_val ^ (hash_val >> 4);

    // Convert to float in [0, 1): clear sign bit, cast, normalize
    auto hash_positive = hash_val & 0x7FFFFFFFu;
    auto hash_float = ct::element_cast<float>(hash_positive);
    auto random = hash_float / 2147483647.0f;

    // x_keep = random > p
    auto x_keep = random > p;

    auto scaled_x = x * scale;
    auto output_f32 = ct::select(x_keep, scaled_x, ct::zeros<f32xN>());

    // Convert back to T and store using pointer + scatter pattern
    auto output = ct::element_cast<T>(output_f32);
    ct::store_masked(output_ptr + offsets, output, mask);
}
