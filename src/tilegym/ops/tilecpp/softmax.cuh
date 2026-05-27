// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ Softmax Kernel
 * Computes softmax along the last dimension.
 *
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * Softmax forward kernel.
 *
 * Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_SIZE: Tile size (power of 2, must equal n_cols)
 *
 * Parameters:
 *   output: Pointer to output tensor (n_rows, n_cols)
 *   input: Pointer to input tensor (n_rows, n_cols)
 *   input_row_stride: Stride for input rows
 *   output_row_stride: Stride for output rows
 *   n_rows: Number of rows
 *   n_cols: Number of columns (power of 2, must equal BLOCK_SIZE)
 *   num_programs: Total number of CTA programs for persistent scheduling
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void softmax_kernel(
    T* __restrict__ _output,
    const T* __restrict__ _input,
    int input_row_stride,
    int output_row_stride,
    int n_rows,
    int n_cols,
    int num_programs
) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    // Apply alignment hints for better memory access
    const T* input = ct::assume_aligned<16>(_input);
    T* output = ct::assume_aligned<16>(_output);

    using f32xN = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    // Persistent scheduling: each program handles multiple rows
    int row_start = ct::bid().x;
    int row_step = num_programs;

    for (auto row_idx : ct::irange(row_start, n_rows, row_step)) {
        // row_start_ptr = input + row_idx * input_row_stride
        const T* row_start_ptr = input + row_idx * input_row_stride;

        auto col_offsets = ct::iota<i32xN>();

        // Create mask for valid columns (handles non-power-of-2 n_cols)
        auto mask = col_offsets < n_cols;

        // input_ptrs = row_start_ptr + col_offsets
        auto input_ptrs = row_start_ptr + col_offsets;

        // Load with mask, padding invalid elements with -infinity
        // so they don't affect max/sum calculations
        auto neg_inf_pad = ct::full<TxN>(T(-INFINITY));
        auto row_T = ct::load_masked(input_ptrs, mask, neg_inf_pad);
        auto row = ct::element_cast<float>(row_T);

        float row_max = static_cast<float>(ct::reduce_max(row, 0_ic));

        // row_minus_max = row - row_max
        auto row_minus_max = row - row_max;

        // exp(-inf - max) = 0, so masked elements contribute 0
        auto numerator = ct::exp(row_minus_max);

        float denominator = static_cast<float>(ct::sum(numerator, 0_ic));

        // softmax_output = numerator / denominator
        auto softmax_output = numerator / denominator;

        // Convert back to output type and store (only valid columns)
        auto softmax_output_T = ct::element_cast<T>(softmax_output);
        auto output_ptrs = output + row_idx * output_row_stride + col_offsets;
        ct::store_masked(output_ptrs, softmax_output_T, mask);
    }
}

/**
 * Online softmax forward kernel.
 * Handles cases where n_cols > BLOCK_SIZE using two-pass algorithm.
 *
 * Template Parameters:
 *   T: Element type
 *   BLOCK_SIZE: Block size for processing columns (power of 2)
 *
 * Handles arbitrary n_cols by ceiling-dividing the column count by BLOCK_SIZE
 * and masking the tail block's loads with -INFINITY (so exp(...) = 0 contribution)
 * and stores so out-of-bounds lanes are not written.
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void online_softmax_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    int input_row_stride,
    int output_row_stride,
    int n_cols
) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    int row_idx = ct::bid().x;

    auto input_aligned = ct::assume_aligned<16>(input);
    auto output_aligned = ct::assume_aligned<16>(output);

    auto row_ptr = input_aligned + row_idx * input_row_stride;
    auto output_row_ptr = output_aligned + row_idx * output_row_stride;

    auto neg_inf_pad = ct::full<TxN>(static_cast<T>(-INFINITY));

    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    int num_blocks = (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (auto block_idx : ct::irange(0, num_blocks)) {
        int start_col = block_idx * BLOCK_SIZE;

        auto col_offsets = ct::full<i32xN>(start_col) + ct::iota<i32xN>();
        auto mask = col_offsets < n_cols;

        auto row_T = ct::load_masked(row_ptr + col_offsets, mask, neg_inf_pad);
        auto row = ct::element_cast<float>(row_T);

        float block_max = static_cast<float>(ct::reduce_max(row, 0_ic));
        float m_curr = (block_max > m_prev) ? block_max : m_prev;

        l_prev *= ct::exp(m_prev - m_curr);

        auto p = ct::exp(row - m_curr);

        float l_block = static_cast<float>(ct::sum(p, 0_ic));

        l_prev += l_block;
        m_prev = m_curr;
    }

    for (auto block_idx : ct::irange(0, num_blocks)) {
        int start_col = block_idx * BLOCK_SIZE;

        auto col_offsets = ct::full<i32xN>(start_col) + ct::iota<i32xN>();
        auto mask = col_offsets < n_cols;

        auto row_T = ct::load_masked(row_ptr + col_offsets, mask, neg_inf_pad);
        auto row = ct::element_cast<float>(row_T);

        auto row_minus_max = row - m_prev;
        auto numerator = ct::exp(row_minus_max);
        auto softmax_output = numerator / l_prev;

        auto softmax_output_T = ct::element_cast<T>(softmax_output);
        ct::store_masked(output_row_ptr + col_offsets, softmax_output_T, mask);
    }
}

/**
 * Softmax backward kernel.
 *
 * Given the softmax output y = softmax(x) and upstream gradient dy,
 * computes dx = y * (dy - sum(y * dy))
 *
 * Template Parameters:
 *   T: Element type
 *   BLOCK_SIZE: Tile size (power of 2, must equal n_cols)
 *
 * Note: Assumes input buffers are power-of-2 sized and tiles match dimensions.
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void softmax_kernel_backward(
    T* __restrict__ dx_ptr,
    const T* __restrict__ y_ptr,
    const T* __restrict__ dy_ptr,
    int dy_row_stride,
    int y_row_stride,
    int dx_row_stride,
    int n_cols
) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    using f32xN = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    int row_idx = ct::bid().x;

    auto y_aligned = ct::assume_aligned<16>(y_ptr);
    auto dy_aligned = ct::assume_aligned<16>(dy_ptr);
    auto dx_aligned = ct::assume_aligned<16>(dx_ptr);

    // Pointers to this row
    auto y_row_ptr = y_aligned + row_idx * y_row_stride;
    auto dy_row_ptr = dy_aligned + row_idx * dy_row_stride;
    auto dx_row_ptr = dx_aligned + row_idx * dx_row_stride;

    auto col_offsets = ct::iota<i32xN>();

    // Create mask for valid columns (handles non-power-of-2 n_cols)
    auto mask = col_offsets < n_cols;
    auto zero_pad = ct::zeros<TxN>();

    // Load probs and gradient with masking, convert to float
    auto probs_T = ct::load_masked(y_row_ptr + col_offsets, mask, zero_pad);
    auto dy_T = ct::load_masked(dy_row_ptr + col_offsets, mask, zero_pad);
    auto probs = ct::element_cast<float>(probs_T);
    auto dy = ct::element_cast<float>(dy_T);

    // dxhat = probs * dy
    auto dxhat = probs * dy;

    // sum(dxhat)
    float dxhat_sum = static_cast<float>(ct::sum(dxhat, 0_ic));

    // softmax_grad = dxhat - probs * sum(dxhat)
    auto dx = dxhat - probs * dxhat_sum;

    // Convert and store (only valid columns)
    auto dx_T = ct::element_cast<T>(dx);
    ct::store_masked(dx_row_ptr + col_offsets, dx_T, mask);
}

/**
 * Online softmax backward kernel.
 * Handles cases where n_cols > BLOCK_SIZE using multiple passes.
 *
 * Template Parameters:
 *   T: Element type
 *   BLOCK_SIZE: Block size for processing columns (power of 2)
 *
 * Handles arbitrary n_cols by ceiling-dividing the column count by BLOCK_SIZE
 * and masking the tail block's loads with 0 (zero contribution to the
 * dxhat reduction) and stores so out-of-bounds lanes are not written.
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void online_softmax_kernel_backward(
    T* __restrict__ dx_ptr,
    const T* __restrict__ y_ptr,
    const T* __restrict__ dy_ptr,
    int dy_row_stride,
    int y_row_stride,
    int dx_row_stride,
    int n_cols
) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;

    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    int row_idx = ct::bid().x;

    auto y_aligned = ct::assume_aligned<16>(y_ptr);
    auto dy_aligned = ct::assume_aligned<16>(dy_ptr);
    auto dx_aligned = ct::assume_aligned<16>(dx_ptr);

    auto y_row_ptr = y_aligned + row_idx * y_row_stride;
    auto dy_row_ptr = dy_aligned + row_idx * dy_row_stride;
    auto dx_row_ptr = dx_aligned + row_idx * dx_row_stride;

    auto zero_pad = ct::full<TxN>(static_cast<T>(0));

    float dxhat_sum = 0.0f;
    int num_blocks = (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (auto block_idx : ct::irange(0, num_blocks)) {
        int start_col = block_idx * BLOCK_SIZE;
        auto col_offsets = ct::full<i32xN>(start_col) + ct::iota<i32xN>();
        auto mask = col_offsets < n_cols;

        auto probs_T = ct::load_masked(y_row_ptr + col_offsets, mask, zero_pad);
        auto dy_T = ct::load_masked(dy_row_ptr + col_offsets, mask, zero_pad);
        auto probs = ct::element_cast<float>(probs_T);
        auto dy = ct::element_cast<float>(dy_T);

        auto dxhat = probs * dy;
        dxhat_sum += static_cast<float>(ct::sum(dxhat, 0_ic));
    }

    for (auto block_idx : ct::irange(0, num_blocks)) {
        int start_col = block_idx * BLOCK_SIZE;
        auto col_offsets = ct::full<i32xN>(start_col) + ct::iota<i32xN>();
        auto mask = col_offsets < n_cols;

        auto probs_T = ct::load_masked(y_row_ptr + col_offsets, mask, zero_pad);
        auto dy_T = ct::load_masked(dy_row_ptr + col_offsets, mask, zero_pad);
        auto probs = ct::element_cast<float>(probs_T);
        auto dy = ct::element_cast<float>(dy_T);

        auto dxhat = probs * dy;
        auto dx = dxhat - probs * dxhat_sum;

        auto dx_T = ct::element_cast<T>(dx);
        ct::store_masked(dx_row_ptr + col_offsets, dx_T, mask);
    }
}
