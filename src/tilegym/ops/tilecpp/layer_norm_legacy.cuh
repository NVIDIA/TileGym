// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ Layer Normalization Legacy Kernel
 * Implements simple row-wise LayerNorm with forward pass.
 *
 * This version processes one row per program and supports arbitrary N < 64KB.
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

/**
 * Layer Normalization Forward Fused Kernel.
 *
 * Computes: y = (x - mean) / sqrt(var + eps) * (weight + weight_shift) + bias
 *
 * Each program handles one row. Uses loop over blocks if N > BLOCK_SIZE.
 */
template<typename T, int N, int BLOCK_SIZE>
__tile_global__ void layer_norm_fwd_fused_kernel(
    T* __restrict__ X,
    T* __restrict__ Y,
    T* __restrict__ W,
    T* __restrict__ B,
    float* __restrict__ Mean,
    float* __restrict__ Rstd,
    float eps,
    float weight_shift
) {
    namespace ct = cuda::tiles;

    using TxN = ct::tile<T, ct::shape<BLOCK_SIZE>>;
    using f32xN = ct::tile<float, ct::shape<BLOCK_SIZE>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_SIZE>>;

    X = ct::assume_aligned<16>(X);
    Y = ct::assume_aligned<16>(Y);
    W = ct::assume_aligned<16>(W);
    B = ct::assume_aligned<16>(B);
    Mean = ct::assume_aligned<16>(Mean);
    Rstd = ct::assume_aligned<16>(Rstd);

    int row = ct::bid().x;
    T* X_row = X + row * N;
    T* Y_row = Y + row * N;

    constexpr int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Pass 1: mean.  All loop bounds + mask comparisons use compile-time N
    // so the optimizer can DCE the mask when num_blocks == 1.
    auto _mean = ct::zeros<f32xN>();
    for (auto block_idx : ct::irange(0, num_blocks)) {
        int off = block_idx * BLOCK_SIZE;
        auto cols = ct::full<i32xN>(off) + ct::iota<i32xN>();
        auto mask = cols < ct::full<i32xN>(N);
        auto a_t = ct::load_masked(X_row + cols, mask, T(0));
        auto a = ct::element_cast<float>(a_t);
        _mean = _mean + a;
    }
    float mean = static_cast<float>(ct::sum<0>(_mean)) / static_cast<float>(N);

    // Pass 2: variance.
    auto _var = ct::zeros<f32xN>();
    for (auto block_idx : ct::irange(0, num_blocks)) {
        int off = block_idx * BLOCK_SIZE;
        auto cols = ct::full<i32xN>(off) + ct::iota<i32xN>();
        auto mask = cols < ct::full<i32xN>(N);
        auto x_t = ct::load_masked(X_row + cols, mask, T(0));
        auto x = ct::element_cast<float>(x_t);
        auto x_centered = ct::select(mask, x - mean, ct::zeros<f32xN>());
        _var = _var + x_centered * x_centered;
    }
    float var = static_cast<float>(ct::sum<0>(_var)) / static_cast<float>(N);
    float rstd = ct::rsqrt(var + eps);

    Mean[row] = mean;
    Rstd[row] = rstd;

    // Pass 3: normalise + linear transform + store.
    for (auto block_idx : ct::irange(0, num_blocks)) {
        int off = block_idx * BLOCK_SIZE;
        auto cols = ct::full<i32xN>(off) + ct::iota<i32xN>();
        auto mask = cols < ct::full<i32xN>(N);

        auto w_t = ct::load_masked(W + cols, mask, T(0));
        auto b_t = ct::load_masked(B + cols, mask, T(0));
        auto w = ct::element_cast<float>(w_t) + weight_shift;
        auto b = ct::element_cast<float>(b_t);

        auto x_t = ct::load_masked(X_row + cols, mask, T(0));
        auto x = ct::element_cast<float>(x_t);

        auto x_hat = (x - mean) * rstd;
        auto y_f32 = x_hat * w + b;
        auto y = ct::element_cast<T>(y_f32);

        ct::store_masked(Y_row + cols, y, mask);
    }
}
