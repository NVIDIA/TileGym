// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ SiLU and Mul Kernel
 * Fused SiLU activation with element-wise multiplication.
 *
 * Computes: silu(input[..., :hidden_size]) * input[..., hidden_size:]
 * where silu(x) = x * sigmoid(x)
 *
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * Fused SiLU and multiplication kernel.
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_SIZE: Tile size (power of 2, must match hidden_size)
 *
 * Parameters:
 *   input: Pointer to input data (flattened, stride = 2 * hidden_size per row)
 *   output: Pointer to output data (flattened, stride = hidden_size per row)
 *   stride: Input stride (2 * hidden_size for row-major)
 *   hidden_size: Size of each half (must equal BLOCK_SIZE)
 *
 * BLOCK_SIZE may be larger than hidden_size (the launcher rounds up to the
 * next power of two); a column mask guards loads and stores so that out-of-
 * range lanes do not read or write past the row.
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void silu_and_mul_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int stride,
    int hidden_size
) {
    namespace ct = cuda::tiles;

    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using i64xN = ct::tile<int64_t, ct::shape<BLOCK_SIZE>>;

    int64_t pid = ct::bid().x;

    auto input_aligned = ct::assume_aligned<16>(input);
    auto output_aligned = ct::assume_aligned<16>(output);

    auto input_row = input_aligned + pid * stride;
    auto output_row = output_aligned + pid * (stride / 2);

    auto col_offsets = ct::iota<i64xN>();
    auto mask = col_offsets < static_cast<int64_t>(hidden_size);
    auto zero_pad = ct::full<TxN>(static_cast<T>(0));

    auto a_ptrs = input_row + col_offsets;
    auto a_T = ct::load_masked(a_ptrs, mask, zero_pad);
    auto a = ct::element_cast<float>(a_T);

    auto b_ptrs = input_row + hidden_size + col_offsets;
    auto b_T = ct::load_masked(b_ptrs, mask, zero_pad);
    auto b = ct::element_cast<float>(b_T);

    auto exp_neg_a = ct::exp(-a);
    auto silu_a = a / (1.0f + exp_neg_a);

    auto result = silu_a * b;

    auto result_T = ct::element_cast<T>(result);
    auto out_ptrs = output_row + col_offsets;
    ct::store_masked(out_ptrs, result_T, mask);
}


/**
 * Row-wise SiLU and multiplication kernel using tensor_span.
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   M: Number of rows (batch_size)
 *   N: Number of columns (2 * hidden_size)
 *   HIDDEN_SIZE: Hidden size (N / 2) - compile-time constant
 *   TILE_SIZE: Tile size (power of 2, >= HIDDEN_SIZE)
 *
 * Parameters:
 *   input: Pointer to input data (M, N) where N = 2 * HIDDEN_SIZE
 *   output: Pointer to output data (M, HIDDEN_SIZE)
 */
template<typename T, int M, int N, int HIDDEN_SIZE, int TILE_SIZE, int INPUT_STRIDE, int OUTPUT_STRIDE>
__tile_global__ void silu_and_mul_kernel_row_wise(
    T* __restrict__ input,
    T* __restrict__ output
) {
    namespace ct = cuda::tiles;

    using TxTS = ct::tile<T, ct::shape<TILE_SIZE>>;
    using f32xTS = ct::tile<float, ct::shape<TILE_SIZE>>;
    using i32xTS = ct::tile<int32_t, ct::shape<TILE_SIZE>>;
    using i64xTS = ct::tile<int64_t, ct::shape<TILE_SIZE>>;

    // Alignment hints for pointers
    input = ct::assume_aligned<16>(input);
    output = ct::assume_aligned<16>(output);

    // Each block handles one row
    int32_t row_i32 = ct::bid().x;
    int64_t row = static_cast<int64_t>(row_i32);

    // Use template parameters for strides
    constexpr int64_t input_stride_i64 = static_cast<int64_t>(INPUT_STRIDE);
    constexpr int64_t output_stride_i64 = static_cast<int64_t>(OUTPUT_STRIDE);

    auto offsets = ct::iota<i32xTS>();

    // Create constant tiles for reuse
    auto row_offset_input = ct::full<i64xTS>(row * input_stride_i64);
    auto row_offset_output = ct::full<i64xTS>(row * output_stride_i64);

    // Statically determine if masking is needed based on TILE_SIZE and HIDDEN_SIZE
    if constexpr (TILE_SIZE == HIDDEN_SIZE) {
        // Optimized path: No masking needed when TILE_SIZE == HIDDEN_SIZE
        // All accesses are guaranteed to be in bounds

        // Broadcast hidden_size to vector tile
        auto hidden_size_tile = ct::full<i32xTS>(HIDDEN_SIZE);

        // Calculate column indices for a and b (i32)
        auto a_col_idx = offsets;
        auto b_col_idx = offsets + hidden_size_tile;

        // Extend to i64 for pointer arithmetic
        auto a_col_idx_i64 = ct::element_cast<int64_t>(a_col_idx);
        auto b_col_idx_i64 = ct::element_cast<int64_t>(b_col_idx);

        // Compute flat indices: row * stride + col_idx
        auto a_indices = row_offset_input + a_col_idx_i64;
        auto b_indices = row_offset_input + b_col_idx_i64;

        // Load without masking (all accesses in bounds)
        auto a_T = ct::load(input + a_indices);
        auto b_T = ct::load(input + b_indices);

        // Convert to float32 for computation
        auto a = ct::element_cast<float>(a_T);
        auto b = ct::element_cast<float>(b_T);

        // Compute sigmoid for SiLU: sigmoid(x) = 1 / (1 + exp(-x))
        // Use optimized operations with flush_to_zero and rounding<approx> flags
        auto exp_neg_a = ct::exp(-a);
        auto denom = ct::add(1.0f, exp_neg_a, ct::round_ties_to_even_t{}, ct::round_subnormals_to_zero_t{});
        auto sigmoid_a = ct::div(1.0f, denom, ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});

        // Compute SiLU(a) = a * sigmoid(a)
        auto silu_a = ct::mul(a, sigmoid_a, ct::round_ties_to_even_t{}, ct::round_subnormals_to_zero_t{});

        // Multiply: result = silu(a) * b
        auto result_f32 = ct::mul(silu_a, b, ct::round_ties_to_even_t{}, ct::round_subnormals_to_zero_t{});

        // Convert back to output type
        auto result = ct::element_cast<T>(result_f32);

        // Store without masking
        auto offsets_i64 = ct::element_cast<int64_t>(offsets);
        auto out_indices = row_offset_output + offsets_i64;
        ct::store(output + out_indices, result);

    } else {
        // General path: Use masking when TILE_SIZE != HIDDEN_SIZE

        // Broadcast template parameters to vector tiles
        auto hidden_size_tile = ct::full<i32xTS>(HIDDEN_SIZE);
        auto n_tile = ct::full<i32xTS>(N);
        auto zero_pad = ct::zeros<TxTS>();

        // Calculate column indices for a and b (i32)
        auto a_col_idx = offsets;
        auto b_col_idx = offsets + hidden_size_tile;

        // Create masks for bounds checking
        auto a_mask = a_col_idx < hidden_size_tile;
        auto b_mask = b_col_idx < n_tile;
        auto out_mask = offsets < hidden_size_tile;

        // Extend to i64 for pointer arithmetic
        auto a_col_idx_i64 = ct::element_cast<int64_t>(a_col_idx);
        auto b_col_idx_i64 = ct::element_cast<int64_t>(b_col_idx);
        auto offsets_i64 = ct::element_cast<int64_t>(offsets);

        // Compute flat indices: row * stride + col_idx
        auto a_indices = row_offset_input + a_col_idx_i64;
        auto b_indices = row_offset_input + b_col_idx_i64;

        // Load with masking
        auto a_T = ct::load_masked(input + a_indices, a_mask, zero_pad);
        auto b_T = ct::load_masked(input + b_indices, b_mask, zero_pad);

        // Convert to float32 for computation
        auto a = ct::element_cast<float>(a_T);
        auto b = ct::element_cast<float>(b_T);

        // Compute sigmoid for SiLU: sigmoid(x) = 1 / (1 + exp(-x))
        // Use optimized operations with flush_to_zero and rounding<approx> flags
        auto exp_neg_a = ct::exp(-a);
        auto denom = ct::add(1.0f, exp_neg_a, ct::round_ties_to_even_t{}, ct::round_subnormals_to_zero_t{});
        auto sigmoid_a = ct::div(1.0f, denom, ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});

        // Compute SiLU(a) = a * sigmoid(a)
        auto silu_a = ct::mul(a, sigmoid_a, ct::round_ties_to_even_t{}, ct::round_subnormals_to_zero_t{});

        // Multiply: result = silu(a) * b
        auto result_f32 = ct::mul(silu_a, b, ct::round_ties_to_even_t{}, ct::round_subnormals_to_zero_t{});

        // Convert back to output type
        auto result = ct::element_cast<T>(result_f32);

        // Store with masking
        auto out_indices = row_offset_output + offsets_i64;
        ct::store_masked(output + out_indices, result, out_mask);
    }
}
