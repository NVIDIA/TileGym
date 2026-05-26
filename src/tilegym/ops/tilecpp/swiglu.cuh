// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ SwiGLU Kernel
 * Implements SwiGLU activation function.
 *
 * SwiGLU: c = silu(a) * b where silu(x) = x * sigmoid(x)
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * SwiGLU backward kernel.
 *
 * Given upstream gradient dc and saved inputs a, b, computes gradients da and db.
 *
 * Forward: c = silu(a) * b where silu(x) = x * sigmoid(x)
 *
 * Backward:
 *   db = dc * silu(a)
 *   da = dc * (silu(a) * (1 - sigmoid(a)) + sigmoid(a)) * b
 *      = dc * sig(a) * (a * (1 - sig(a)) + 1) * b
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_SIZE: Must be >= n_cols (power of 2)
 *
 * Parameters:
 *   dc_ptr: Pointer to upstream gradient (n_rows, n_cols)
 *   a_ptr:  Pointer to first input  (n_rows, n_cols)
 *   b_ptr:  Pointer to second input (n_rows, n_cols)
 *   da_ptr: Pointer to first output gradient (n_rows, n_cols)
 *   db_ptr: Pointer to second output gradient (n_rows, n_cols)
 *   stride: Row stride (typically n_cols)
 *   n_cols: Number of columns
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void swiglu_backward_kernel(
    const T* __restrict__ dc_ptr,
    const T* __restrict__ a_ptr,
    const T* __restrict__ b_ptr,
    T* __restrict__ da_ptr,
    T* __restrict__ db_ptr,
    int stride,
    int n_cols
) {
    namespace ct = cuda::tiles;

    using f32xN = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    int64_t program_id = ct::bid().x;

    auto dc_aligned = ct::assume_aligned<16>(dc_ptr);
    auto a_aligned = ct::assume_aligned<16>(a_ptr);
    auto b_aligned = ct::assume_aligned<16>(b_ptr);
    auto da_aligned = ct::assume_aligned<16>(da_ptr);
    auto db_aligned = ct::assume_aligned<16>(db_ptr);

    auto dc = dc_aligned + program_id * stride;
    auto a = a_aligned + program_id * stride;
    auto b = b_aligned + program_id * stride;
    auto da = da_aligned + program_id * stride;
    auto db = db_aligned + program_id * stride;

    auto col_offsets = ct::iota<i32xN>();
    auto mask = col_offsets < n_cols;

    auto dc_row = ct::load_masked(dc + col_offsets, mask, T(0));
    auto a_row_t = ct::load_masked(a + col_offsets, mask, T(0));
    auto b_row = ct::load_masked(b + col_offsets, mask, T(0));

    auto a_row = ct::element_cast<float>(a_row_t);
    auto dc_row_f32 = ct::element_cast<float>(dc_row);
    auto b_row_f32 = ct::element_cast<float>(b_row);

    auto neg_a = -a_row;
    auto exp_neg_a = ct::exp(neg_a);
    auto sig_a = ct::ones<f32xN>() / (ct::ones<f32xN>() + exp_neg_a);
    auto silu_a = a_row * sig_a;

    auto db_row_f32 = dc_row_f32 * silu_a;
    auto da_row_f32 = dc_row_f32 * (silu_a * (ct::ones<f32xN>() - sig_a) + sig_a) * b_row_f32;

    auto da_row = ct::element_cast<T>(da_row_f32);
    auto db_row = ct::element_cast<T>(db_row_f32);

    ct::store_masked(da + col_offsets, da_row, mask);
    ct::store_masked(db + col_offsets, db_row, mask);
}

/**
 * SwiGLU forward kernel using tensor_span + partition_view with unmasked loads.
 *
 * This is the primary optimized kernel for the forward pass.
 *
 * Requires BLOCK_SIZE to divide N_COLS evenly.
 *
 * Template Parameters:
 *   T: Element type
 *   N_ROWS: Number of rows (template for compile-time shape)
 *   N_COLS: Number of columns (template for compile-time shape)
 *   BLOCK_SIZE: Tile size for columns (must divide N_COLS)
 *   OCCUPANCY: Occupancy hint (scaled based on BLOCK_SIZE for better utilization)
 */
template<typename T, int N_ROWS, int N_COLS, int BLOCK_SIZE, int OCCUPANCY = 4>
__tile_global__ void swiglu_forward_kernel_pv(
    T* __restrict__ a_ptr,
    T* __restrict__ b_ptr,
    T* __restrict__ c_ptr
) {
    namespace ct = cuda::tiles;

    // Alignment hints
    a_ptr = ct::assume_aligned<16>(a_ptr);
    b_ptr = ct::assume_aligned<16>(b_ptr);
    c_ptr = ct::assume_aligned<16>(c_ptr);

    // 2D tile types for partition_view: (1, BLOCK_SIZE)
    using Tile1xN = ct::tile<T, ct::shape<1, BLOCK_SIZE>>;
    using Tile1xN_f32 = ct::tile<float, ct::shape<1, BLOCK_SIZE>>;

    // 2D grid: row = bid.x, col_tile = bid.y
    int row = ct::bid().x;
    int col = ct::bid().y;

    // Create tensor_span with compile-time extents
    auto a_span = ct::tensor_span{a_ptr, ct::extents{N_ROWS, N_COLS}};
    auto b_span = ct::tensor_span{b_ptr, ct::extents{N_ROWS, N_COLS}};
    auto c_span = ct::tensor_span{c_ptr, ct::extents{N_ROWS, N_COLS}};

    // Create partition views with compile-time tile shape
    auto a_view = ct::partition_view(a_span, ct::shape<1, BLOCK_SIZE>{});
    auto b_view = ct::partition_view(b_span, ct::shape<1, BLOCK_SIZE>{});
    auto c_view = ct::partition_view(c_span, ct::shape<1, BLOCK_SIZE>{});

    // Unmasked loads - high latency hint for memory-bound kernel
    Tile1xN a_tile, b_tile;
    a_tile = a_view.load(row, col);
    b_tile = b_view.load(row, col);

    // Convert to float32 for silu computation
    auto a_tile_f32 = ct::element_cast<float>(a_tile);

    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    auto exp_neg_a = ct::exp(-a_tile_f32);
    auto one = ct::full<Tile1xN_f32>(1.0f);
    auto sigmoid_a = one / (one + exp_neg_a);
    auto silu_result_f32 = a_tile_f32 * sigmoid_a;

    // Cast back to T and multiply by b
    auto silu_result = ct::element_cast<T>(silu_result_f32);
    auto c_tile = silu_result * b_tile;

    // Store with TMA hint - low latency for writes
    c_view.store(c_tile, row, col);
}


/**
 *
 * Grid: (n_rows,).  TILE_SIZE = next_power_of_2(n_cols).
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void swiglu_forward_kernel_gather(
    const T* __restrict__ a_ptr,
    const T* __restrict__ b_ptr,
    T* __restrict__ c_ptr,
    int n_cols,
    int row_stride
) {
    namespace ct = cuda::tiles;

    using TxN   = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using f32xN = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    a_ptr = ct::assume_aligned<16>(a_ptr);
    b_ptr = ct::assume_aligned<16>(b_ptr);
    c_ptr = ct::assume_aligned<16>(c_ptr);

    n_cols     = ct::assume_bounded_below(n_cols,     ct::integral_constant<0>{});
    n_cols     = ct::assume_divisible(n_cols,         ct::integral_constant<8>{});
    row_stride = ct::assume_bounded_below(row_stride, ct::integral_constant<0>{});
    row_stride = ct::assume_divisible(row_stride,     ct::integral_constant<16>{});

    int row = ct::bid().x;

    const T* a_row = a_ptr + row * row_stride;
    const T* b_row = b_ptr + row * row_stride;
    T*       c_row = c_ptr + row * row_stride;

    auto col_offs = ct::iota<i32xN>();
    auto mask     = col_offs < n_cols;

    TxN a_val, b_val;
    [[ using cutile : hint(1000, latency=1) ]]
    a_val = ct::load_masked(a_row + col_offs, mask, T(0));
    [[ using cutile : hint(1000, latency=1) ]]
    b_val = ct::load_masked(b_row + col_offs, mask, T(0));

    auto a_f32 = ct::element_cast<float>(a_val);

    auto one       = ct::full<f32xN>(1.0f);
    auto denom     = ct::add(one, ct::exp(-a_f32),
                             ct::round_ties_to_even_t{},
                             ct::round_subnormals_to_zero_t{});
    auto sigmoid_a = ct::div(one, denom,
                             ct::round_approximate_t{},
                             ct::round_subnormals_to_zero_t{});
    auto silu_f32  = ct::mul(a_f32, sigmoid_a,
                             ct::round_ties_to_even_t{},
                             ct::round_subnormals_to_zero_t{});

    auto silu_T = ct::element_cast<T>(silu_f32);
    auto c_val  = silu_T * b_val;

    [[ using cutile : hint(1000, latency=1) ]]
    ct::store_masked(c_row + col_offs, c_val, mask);
}
