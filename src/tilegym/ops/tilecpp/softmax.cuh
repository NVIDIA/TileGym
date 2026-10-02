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

template<typename T, int BLOCK_SIZE, int TMA_ROWS, int ALIGNED_ROWS, int ROWS, int COLS>
__tile__ void softmax_body(
    T* __restrict__ _output, const T* __restrict__ _input,
    int input_row_stride, int output_row_stride, int n_rows, int n_cols,
    int num_programs
) {
    namespace ct = cuda::tiles;
    using namespace ct::literals;
    using f32xN = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;
    const T* input = ct::assume_aligned<16>(_input);
    T* output = ct::assume_aligned<16>(_output);
    n_rows = ct::assume_bounded_below<0>(n_rows);
    n_cols = ct::assume_bounded_below<0>(n_cols);
    input_row_stride = ct::assume_bounded_below<0>(input_row_stride);
    output_row_stride = ct::assume_bounded_below<0>(output_row_stride);
    if constexpr (ALIGNED_ROWS) {
        using alignment = ct::integral_constant<16 / sizeof(T)>;
        input_row_stride = ct::assume_divisible(input_row_stride, alignment{});
        output_row_stride = ct::assume_divisible(output_row_stride, alignment{});
        n_cols = ct::assume_divisible(n_cols, alignment{});
    }
    int row_start = ct::bid().x;
    if constexpr (TMA_ROWS) {
        auto in_layout = ct::layout_strided_mapping{
            ct::extents{n_rows, n_cols}, ct::extents{input_row_stride, ct::integral_constant<1>{}}};
        auto out_layout = ct::layout_strided_mapping{
            ct::extents{n_rows, n_cols}, ct::extents{output_row_stride, ct::integral_constant<1>{}}};
        auto pIn = ct::partition_view{ct::tensor_span{input, in_layout}, ct::shape<1, BLOCK_SIZE>{}};
        auto pOut = ct::partition_view{ct::tensor_span{output, out_layout}, ct::shape<1, BLOCK_SIZE>{}};
        for (auto row_idx : ct::irange(row_start, ROWS, num_programs)) {
            auto row = ct::element_cast<float>(ct::reshape(
                pIn.template load_masked<ct::view_padding::negative_inf>(row_idx, 0),
                ct::shape<BLOCK_SIZE>{}));
            auto maximum = ct::reduce_max(row, 0_ic);
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
            auto numerator = ct::exp(row - maximum, ct::round_approximate_t{});
#else
            auto numerator = ct::exp(row - maximum);
#endif
            auto denominator = ct::sum(numerator, 0_ic);
            auto result = ct::div(numerator, denominator,
                                  ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});
            pOut.store_masked(ct::reshape(ct::element_cast<T>(result), ct::shape<1, BLOCK_SIZE>{}),
                              row_idx, 0);
        }
    } else {
        for (auto row_idx : ct::irange(row_start, ROWS, num_programs)) {
            const T* row_ptr = input + row_idx * input_row_stride;
            T* out_ptr = output + row_idx * output_row_stride;
            if constexpr (ALIGNED_ROWS) {
                row_ptr = ct::assume_aligned<16>(row_ptr);
                out_ptr = ct::assume_aligned<16>(out_ptr);
            }
            auto offsets = ct::iota<i32xN>();
            auto mask = offsets < n_cols;
            auto row = ct::element_cast<float>(ct::load_masked(
                row_ptr + offsets, mask, ct::full<TxN>(T(-INFINITY))));
            auto maximum = ct::reduce_max(row, 0_ic);
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
            auto numerator = ct::exp(row - maximum, ct::round_approximate_t{});
#else
            auto numerator = ct::exp(row - maximum);
#endif
            auto denominator = ct::sum(numerator, 0_ic);
            auto result = ct::div(numerator, denominator,
                                  ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});
            ct::store_masked(out_ptr + offsets, ct::element_cast<T>(result), mask);
        }
    }
}

template<typename T, int BLOCK_SIZE, int TMA_ROWS, int ALIGNED_ROWS, int OCCUPANCY, int ROWS, int COLS>
[[cutile::hint(0, occupancy=OCCUPANCY)]]
__tile_global__ void softmax_kernel(
    T* __restrict__ output, const T* __restrict__ input,
    int input_row_stride, int output_row_stride, int n_rows, int n_cols, int num_programs
) {
    softmax_body<T, BLOCK_SIZE, TMA_ROWS, ALIGNED_ROWS, ROWS, COLS>(
        output, input, input_row_stride, output_row_stride, n_rows, n_cols, num_programs);
}

template<typename T, int BLOCK_SIZE, int TMA_ROWS, int ALIGNED_ROWS, int OCCUPANCY, int WORKER_WARPS, int ROWS, int COLS>
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
[[cutile::hint(0, occupancy=OCCUPANCY, num_worker_warps_per_cta=WORKER_WARPS)]]
#else
[[cutile::hint(0, occupancy=OCCUPANCY)]]
#endif
__tile_global__ void softmax_kernel_warps(
    T* __restrict__ output, const T* __restrict__ input,
    int input_row_stride, int output_row_stride, int n_rows, int n_cols, int num_programs
) {
    softmax_body<T, BLOCK_SIZE, TMA_ROWS, ALIGNED_ROWS, ROWS, COLS>(
        output, input, input_row_stride, output_row_stride, n_rows, n_cols, num_programs);
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
template<typename T, int BLOCK_SIZE, int ALIGNED_ROWS, int COLS>
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
[[ using cutile : hint(0, occupancy=2, num_worker_warps_per_cta=8) ]]
#else
[[ using cutile : hint(0, occupancy=2) ]]
#endif
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
    if constexpr (ALIGNED_ROWS) {
        row_ptr = ct::assume_aligned<16>(row_ptr);
        output_row_ptr = ct::assume_aligned<16>(output_row_ptr);
        n_cols = ct::assume_divisible(n_cols, ct::integral_constant<16 / sizeof(T)>{});
    }

    auto neg_inf_pad = ct::full<TxN>(static_cast<T>(-INFINITY));

    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    constexpr int num_blocks = (COLS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (auto block_idx : ct::irange(0, num_blocks)) {
        int start_col = block_idx * BLOCK_SIZE;

        auto col_offsets = ct::full<i32xN>(start_col) + ct::iota<i32xN>();
        auto mask = col_offsets < COLS;

        auto row_T = ct::load_masked(row_ptr + col_offsets, mask, neg_inf_pad);
        auto row = ct::element_cast<float>(row_T);

        float block_max = static_cast<float>(ct::reduce_max(row, 0_ic));
        float m_curr = (block_max > m_prev) ? block_max : m_prev;

#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
        l_prev *= ct::exp(m_prev - m_curr, ct::round_approximate_t{});
#else
        l_prev *= ct::exp(m_prev - m_curr);
#endif

#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
        auto p = ct::exp(row - m_curr, ct::round_approximate_t{});
#else
        auto p = ct::exp(row - m_curr);
#endif

        float l_block = static_cast<float>(ct::sum(p, 0_ic));

        l_prev += l_block;
        m_prev = m_curr;
    }

    float inv_denominator = ct::div(1.0f, l_prev, ct::round_approximate_t{},
                                      ct::round_subnormals_to_zero_t{});

    for (auto block_idx : ct::irange(0, num_blocks)) {
        int start_col = (num_blocks - 1 - block_idx) * BLOCK_SIZE;

        auto col_offsets = ct::full<i32xN>(start_col) + ct::iota<i32xN>();
        auto mask = col_offsets < COLS;

        auto row_T = ct::load_masked(row_ptr + col_offsets, mask, neg_inf_pad);
        auto row = ct::element_cast<float>(row_T);

        auto row_minus_max = row - m_prev;
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)
        auto numerator = ct::exp(row_minus_max, ct::round_approximate_t{});
#else
        auto numerator = ct::exp(row_minus_max);
#endif
        auto softmax_output = numerator * inv_denominator;

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
