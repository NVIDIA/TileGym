// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ RMS Normalization Kernel
 * Implements RMSNorm with optional weight scaling.
 *
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * RMSNorm kernel - optimized version.
 *
 * Computes: y = x * rsqrt(mean(x^2) + eps) * weight
 *
 * Optimizations:
 * - __restrict__ pointers for alias optimization
 * - Alignment hints for vectorized loads
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_SIZE: Processing tile size (power of 2)
 *
 * Parameters:
 *   X: Pointer to input tensor (M, N)
 *   W: Pointer to weight tensor (N,)
 *   Y: Pointer to output tensor (M, N)
 *   rstd_ptr: Pointer to store 1/std per row (M,)
 *   stride: Row stride (typically N)
 *   N: Number of columns (hidden size)
 *   eps: Small epsilon for numerical stability
 */
template<typename T, int BLOCK_SIZE, int N, float EPS>
__tile_global__ void rms_norm_kernel(
    const T* __restrict__ X,
    const T* __restrict__ W,
    T* __restrict__ Y,
    float* __restrict__ rstd_ptr,
    int stride
) {
    namespace ct = cuda::tiles;

    using TxBS = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using f32xBS = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using i32xBS = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    // Each program handles one row
    int row = ct::bid().x;

    // Aligned pointers for this row
    auto X_row = ct::assume_aligned<16>(X + row * stride);
    auto Y_row = ct::assume_aligned<16>(Y + row * stride);
    auto W_aligned = ct::assume_aligned<16>(W);
    auto rstd_aligned = ct::assume_aligned<16>(rstd_ptr);

    auto zero_pad = ct::zeros<TxBS>();

    constexpr bool EVEN_N = (N % BLOCK_SIZE) == 0;
    constexpr int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto rms_acc = ct::zeros<f32xBS>();

    for (auto j_idx : ct::irange(0, num_blocks)) {
        int j = j_idx * BLOCK_SIZE;
        auto cols = ct::iota<i32xBS>() + j;
        TxBS xj_raw;
        if constexpr (EVEN_N) {
            [[ using cutile : hint(1000, latency=1) ]]
            xj_raw = ct::load(X_row + cols);
        } else {
            auto mask = cols < N;
            [[ using cutile : hint(1000, latency=1) ]]
            xj_raw = ct::load_masked(X_row + cols, mask, zero_pad);
        }
        auto xj = ct::element_cast<float>(xj_raw);
        rms_acc = rms_acc + xj * xj;
    }

    float sum_sq = static_cast<float>(ct::sum<0>(rms_acc));
    constexpr float inv_N = 1.0f / static_cast<float>(N);
    float rms = ct::rsqrt(sum_sq * inv_N + EPS);

    rstd_aligned[row] = rms;

    // Second pass: normalize and apply linear transformation.
    for (auto j_idx : ct::irange(0, num_blocks)) {
        int j = j_idx * BLOCK_SIZE;
        auto cols = ct::iota<i32xBS>() + j;
        TxBS wj_raw;
        TxBS xj_raw;
        if constexpr (EVEN_N) {
            [[ using cutile : hint(1000, latency=1) ]]
            wj_raw = ct::load(W_aligned + cols);
            [[ using cutile : hint(1000, latency=1) ]]
            xj_raw = ct::load(X_row + cols);
        } else {
            auto mask = cols < N;
            [[ using cutile : hint(1000, latency=1) ]]
            wj_raw = ct::load_masked(W_aligned + cols, mask, zero_pad);
            [[ using cutile : hint(1000, latency=1) ]]
            xj_raw = ct::load_masked(X_row + cols, mask, zero_pad);
        }
        auto wj = ct::element_cast<float>(wj_raw);
        auto xj = ct::element_cast<float>(xj_raw);
        auto yj_f32 = xj * rms * wj;
        auto yj = ct::element_cast<T>(yj_f32);
        if constexpr (EVEN_N) {
            [[ using cutile : hint(1000, latency=1) ]]
            ct::store(Y_row + cols, yj);
        } else {
            auto mask = cols < N;
            [[ using cutile : hint(1000, latency=1) ]]
            ct::store_masked(Y_row + cols, yj, mask);
        }
    }
}


/**
 * RMSNorm kernel — one CTA per row, unmasked partition_view loads/stores.
 * Template Parameters:
 *   T, M, N, BLOCK_SIZE (= N).
 */
template<typename T, int M, int N, int BLOCK_SIZE>
[[ using cutile : hint(1000, num_cta_in_cga=1) ]]
__tile_global__ void rms_norm_kernel_pv(
    const T* __restrict__ X,
    const T* __restrict__ W,
    T* __restrict__ Y,
    float* __restrict__ rstd_ptr,
    float eps
) {
    namespace ct = cuda::tiles;

    using TxBS    = ct::tile<T, ct::shape<1, BLOCK_SIZE>>;
    using TxBS_1d = ct::tile<T, ct::shape<BLOCK_SIZE>>;

    X = ct::assume_aligned<16>(X);
    W = ct::assume_aligned<16>(W);
    Y = ct::assume_aligned<16>(Y);
    auto rstd_aligned = ct::assume_aligned<16>(rstd_ptr);

    int row = ct::bid().x;

    auto X_view = ct::partition_view(
        ct::tensor_span{X, ct::extents<uint32_t, M, N>{}},
        ct::shape<1, BLOCK_SIZE>{});
    auto Y_view = ct::partition_view(
        ct::tensor_span{Y, ct::extents<uint32_t, M, N>{}},
        ct::shape<1, BLOCK_SIZE>{});
    auto W_view = ct::partition_view(
        ct::tensor_span{W, ct::extents<uint32_t, N>{}},
        ct::shape<BLOCK_SIZE>{});

    TxBS_1d w_loaded;
    [[ using cutile : hint(1000, latency=1) ]]
    w_loaded = W_view.load(0);
    auto w = ct::element_cast<float>(w_loaded);

    TxBS x_loaded;
    [[ using cutile : hint(1000, latency=1) ]]
    x_loaded = X_view.load(row, 0);
    auto x = ct::element_cast<float>(x_loaded);

    auto x_squared = x * x;
    auto sum_row   = ct::sum(x_squared, ct::integral_constant<1>{});   // (1, 1)
    float sum_sq   = static_cast<float>(ct::sum(sum_row, ct::integral_constant<0>{}));
    float rms      = ct::rsqrt(sum_sq / static_cast<float>(N) + eps);

    if (rstd_ptr != nullptr) {
        rstd_aligned[row] = rms;
    }

    auto x_1d  = ct::reshape<ct::shape<BLOCK_SIZE>>(x);
    auto y_1d  = x_1d * rms * w;
    auto y_2d  = ct::reshape<ct::shape<1, BLOCK_SIZE>>(y_1d);
    auto y_out = ct::element_cast<T>(y_2d);

    [[ using cutile : hint(1000, allow_tma=false) ]]
    Y_view.store(y_out, row, 0);
}


/**
 *
 * Grid: (NUM_SMS, 1, 1) — persistent grid-stride.
 * Each block processes TILE_SIZE_M rows x TILE_SIZE_N cols per iteration and
 * strides by NUM_SMS * TILE_SIZE_M across M.
 *
 * Template Parameters:
 *   T:            element type
 *   TILE_SIZE_M:  rows per tile (2, 4, 8, 16)
 *   TILE_SIZE_N:  cols per tile; power of 2 >= N (the full row is loaded once)
 *
 * Runtime Parameters:
 *   X:     (M, N)
 *   Y:     (M, N)
 *   W:     (N,)
 *   Rstd:  (M,)           fp32 — saved for backward
 *   eps:   epsilon
 *   offset: added to weight (0.0 for Llama, 1.0 for Gemma3)
 *   M, N:  runtime row/col counts
 *   NUM_SMS: persistent grid size (== grid x-dim)
 */
template<typename T,
         int TILE_SIZE_M, int TILE_SIZE_N,
         int occupancy,
         int M,
         int N,
         int NUM_SMS,   // ← compile-time so the persistent for-loop step is constant.
         float EPS,     // ← compile-time so (var + eps) hoists out of the loop.
         float OFFSET>
[[ using cutile : hint(1000, num_cta_in_cga=1, occupancy=occupancy) ]]
__tile_global__ void rms_norm_static_persistent_kernel(
    const T* __restrict__ X,
    T* __restrict__ Y,
    const T* __restrict__ W,
    float* __restrict__ Rstd
) {
    namespace ct = cuda::tiles;

    using f32_MxN = ct::tile<float, ct::shape<TILE_SIZE_M, TILE_SIZE_N>>;
    using TileMxN = ct::tile<T,     ct::shape<TILE_SIZE_M, TILE_SIZE_N>>;
    using f32_Mx1 = ct::tile<float, ct::shape<TILE_SIZE_M, 1>>;
    using f32_M   = ct::tile<float, ct::shape<TILE_SIZE_M>>;

    X = ct::assume_aligned<16>(X);
    Y = ct::assume_aligned<16>(Y);
    W = ct::assume_aligned<16>(W);
    Rstd = ct::assume_aligned<16>(Rstd);

    int pid                   = ct::bid().x;
    constexpr int upper_bound = (M + TILE_SIZE_M - 1) / TILE_SIZE_M;

    // Static extents → no runtime stride machinery, no in-loop tensor_view.
    using ExtMN = ct::extents<uint32_t, static_cast<uint32_t>(M), static_cast<uint32_t>(N)>;
    using ExtN  = ct::extents<uint32_t, static_cast<uint32_t>(N)>;
    using ExtM  = ct::extents<uint32_t, static_cast<uint32_t>(M)>;
    auto pX    = ct::partition_view(ct::tensor_span{X,    ExtMN{}},
                                    ct::shape<TILE_SIZE_M, TILE_SIZE_N>{});
    auto pY    = ct::partition_view(ct::tensor_span{Y,    ExtMN{}},
                                    ct::shape<TILE_SIZE_M, TILE_SIZE_N>{});
    auto pW    = ct::partition_view(ct::tensor_span{W,    ExtN{}},
                                    ct::shape<TILE_SIZE_N>{});
    auto pRstd = ct::partition_view(ct::tensor_span{Rstd, ExtM{}},
                                    ct::shape<TILE_SIZE_M>{});

    // Load W once, apply offset in fp32 (y = x_hat * (offset + w)).
    auto w       = ct::element_cast<float>(pW.load(0));            // (TILE_N,)
    auto w_with  = w + ct::full<ct::tile<float, ct::shape<TILE_SIZE_N>>>(OFFSET);
    auto w_bcast = ct::reshape<ct::shape<1, TILE_SIZE_N>>(w_with);

    constexpr float inv_N = 1.0f / static_cast<float>(N);

    for (auto current_bid : ct::irange(pid, upper_bound, NUM_SMS)) {
        TileMxN x_tile;
        [[ using cutile : hint(1000, latency=10) ]]
        x_tile = pX.load(current_bid, 0);
        auto x = ct::element_cast<float>(x_tile);

        // Row-wise sum(x^2) -> (TILE_M, 1) -> divide by N -> +eps -> rsqrt
        auto x_sq     = x * x;
        auto sq_sum   = ct::sum<1>(x_sq);                            // (TILE_M, 1)
        auto variance = sq_sum * inv_N;
        auto rstd_col = ct::rsqrt(variance + EPS);                   // (TILE_M, 1)

        [[ using cutile : hint(1000, allow_tma=false) ]]
        pRstd.store(ct::reshape<ct::shape<TILE_SIZE_M>>(rstd_col), current_bid);

        auto x_norm = x * rstd_col;                                  // broadcast (TILE_M,1)
        auto y_f32  = x_norm * w_bcast;                              // broadcast (1,TILE_N)
        auto y_T    = ct::element_cast<T>(y_f32);

        [[ using cutile : hint(1000, allow_tma=false, latency=3) ]]
        pY.store(y_T, current_bid, 0);
    }
}


/**
 * RMSNorm backward kernel - computes dx and stores dy*x*rstd into temp_buffer for dw.
 *
 * Formula: dx_{m,i} = dy_{m,i} * w_i / r_m - x_{m,i} / (N * r_m^3) * sum_j(dy_{m,j} * w_j * x_{m,j})
 * where r_m = 1/rstd[m] (the RMS for row m), so rstd_m = 1/r_m.
 *
 * temp_buffer stores: dy_{m,j} * x_{m,j} * rstd_m (for computing dw later by summing over rows).
 *
 * Each CTA handles one row and processes all columns via a loop.
 * BLOCK_SIZE should be >= N.
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_SIZE: Processing tile size (power of 2, >= N)
 *
 * Parameters:
 *   DX: Output gradient w.r.t. input (M, N)
 *   DY: Upstream gradient (M, N)
 *   X: Original input (M, N)
 *   W: Weight (N,)
 *   Rstd: Reciprocal std per row (M,), i.e., rstd[m] = rsqrt(mean(x_m^2) + eps)
 *   temp_buffer: scratch buffer (M, N) for storing dy*x*rstd per element (float32)
 *   stride: Row stride of X (typically N)
 *   N: Number of columns
 */
template<typename T, int BLOCK_SIZE>
__tile_global__ void rms_norm_backward_dx_kernel(
    T* __restrict__ DX,
    const T* __restrict__ DY,
    const T* __restrict__ X,
    const T* __restrict__ W,
    const float* __restrict__ Rstd,
    float* __restrict__ temp_buffer,
    int stride,
    int N
) {
    namespace ct = cuda::tiles;

    using TxBS = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using f32xBS = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using i32xBS = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    int row = ct::bid().x;

    auto X_row = ct::assume_aligned<16>(X + row * stride);
    auto DY_row = ct::assume_aligned<16>(DY + row * stride);
    auto DX_row = ct::assume_aligned<16>(DX + row * stride);
    auto W_aligned = ct::assume_aligned<16>(W);
    auto temp_row = ct::assume_aligned<16>(temp_buffer + row * stride);
    auto Rstd_aligned = ct::assume_aligned<16>(Rstd);

    auto zero_pad = ct::zeros<TxBS>();

    // Load rstd for this row (scalar)
    float inv_std = Rstd_aligned[row];

    // First pass: compute sum_j(dy_j * w_j * x_j) and store dy*x*rstd into temp_buffer
    auto weighted_grad_sum_acc = ct::zeros<f32xBS>();

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (auto j_idx : ct::irange(0, num_blocks)) {
        int j = j_idx * BLOCK_SIZE;
        auto cols = ct::iota<i32xBS>() + j;
        auto mask = cols < N;

        auto x_raw = ct::load_masked(X_row + cols, mask, zero_pad);
        auto dy_raw = ct::load_masked(DY_row + cols, mask, zero_pad);
        auto w_raw = ct::load_masked(W_aligned + cols, mask, zero_pad);

        // Upcast to float32 before any multiply so dy*x is computed in fp32.
        // Doing the multiply in native fp16/bf16 would lose precision.
        auto x = ct::element_cast<float>(x_raw);
        auto dy = ct::element_cast<float>(dy_raw);
        auto w = ct::element_cast<float>(w_raw);
        auto dy_x = dy * x;
        auto dy_x_rstd = dy_x * inv_std;
        ct::store_masked(temp_row + cols, dy_x_rstd, mask);

        weighted_grad_sum_acc = weighted_grad_sum_acc + dy_x * w;
    }

    float weighted_grad_sum = static_cast<float>(ct::sum<0>(weighted_grad_sum_acc));

    // Precompute: inv_std^3 / N * weighted_grad_sum
    float correction_coeff = inv_std * inv_std * inv_std / static_cast<float>(N) * weighted_grad_sum;

    // Second pass: compute dx = dy * w * rstd - x * correction_coeff
    for (auto j_idx : ct::irange(0, num_blocks)) {
        int j = j_idx * BLOCK_SIZE;
        auto cols = ct::iota<i32xBS>() + j;
        auto mask = cols < N;

        auto x_raw = ct::load_masked(X_row + cols, mask, zero_pad);
        auto dy_raw = ct::load_masked(DY_row + cols, mask, zero_pad);
        auto w_raw = ct::load_masked(W_aligned + cols, mask, zero_pad);

        auto x = ct::element_cast<float>(x_raw);
        auto dy = ct::element_cast<float>(dy_raw);
        auto w = ct::element_cast<float>(w_raw);

        // Direct term: dy * w * rstd
        auto scaled_grad = dy * w * inv_std;

        // Correction term: x * correction_coeff
        auto correction = x * correction_coeff;

        // dx = direct - correction
        auto dx = scaled_grad - correction;

        auto dx_out = ct::element_cast<T>(dx);
        ct::store_masked(DX_row + cols, dx_out, mask);
    }
}
