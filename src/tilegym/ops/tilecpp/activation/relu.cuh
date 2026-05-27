// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


template<typename T, int BLOCK_SIZE>
using tile_t = cuda::tiles::tile<T, cuda::tiles::shape<BLOCK_SIZE>>;

template<typename T, int BLOCK_SIZE, int OP>
__tile_global__ void relu_activation_fwd_kernel(const T* __restrict__ x, T* __restrict__ y, int n_elements, float alpha, float lower, float upper, bool training) {
    namespace ct = cuda::tiles;
    x = ct::assume_aligned<16>(x);
    y = ct::assume_aligned<16>(y);
    using TxN = tile_t<T, BLOCK_SIZE>;
    using f32xN = tile_t<float, BLOCK_SIZE>;
    using i32xN = tile_t<int32_t, BLOCK_SIZE>;
    int base = ct::bid().x * BLOCK_SIZE;
    auto offsets = ct::full<i32xN>(base) + ct::iota<i32xN>();
    auto mask = offsets < ct::full<i32xN>(n_elements);
    auto x_T = ct::load_masked(x + offsets, mask, ct::zeros<TxN>());
    auto xf = ct::element_cast<float>(x_T);
    auto zero = ct::zeros<f32xN>();
    f32xN out;
    if constexpr (OP == 0) { // relu
        out = ct::select(xf > zero, xf, zero);
    } else if constexpr (OP == 1) { // elu
        out = ct::select(xf > zero, xf, alpha * (ct::exp(xf) - 1.0f));
    } else if constexpr (OP == 2) { // leaky_relu
        out = ct::select(xf > zero, xf, alpha * xf);
    } else if constexpr (OP == 3) { // selu
        constexpr float scale = 1.0507009873554805f;
        constexpr float alpha_selu = 1.6732632423543772f;
        out = scale * ct::select(xf > zero, xf, alpha_selu * (ct::exp(xf) - 1.0f));
    } else if constexpr (OP == 4) { // celu
        out = ct::select(xf > zero, xf, alpha * (ct::exp(xf / alpha) - 1.0f));
    } else { // rrelu
        float avg_alpha = (lower + upper) * 0.5f;
        auto alpha_tile = ct::full<f32xN>(avg_alpha);
        // Use deterministic average slope. Existing tests use inference-style API compatibility.
        out = ct::select(xf > zero, xf, alpha_tile * xf);
    }
    ct::store_masked(y + offsets, ct::element_cast<T>(out), mask);
}

template<typename T, int BLOCK_SIZE, int OP>
__tile_global__ void relu_activation_bwd_kernel(const T* __restrict__ dy, const T* __restrict__ x, T* __restrict__ dx, int n_elements, float alpha, float lower, float upper, bool training) {
    namespace ct = cuda::tiles;
    dy = ct::assume_aligned<16>(dy);
    x = ct::assume_aligned<16>(x);
    dx = ct::assume_aligned<16>(dx);
    using TxN = tile_t<T, BLOCK_SIZE>;
    using f32xN = tile_t<float, BLOCK_SIZE>;
    using i32xN = tile_t<int32_t, BLOCK_SIZE>;
    int base = ct::bid().x * BLOCK_SIZE;
    auto offsets = ct::full<i32xN>(base) + ct::iota<i32xN>();
    auto mask = offsets < ct::full<i32xN>(n_elements);
    auto dy_T = ct::load_masked(dy + offsets, mask, ct::zeros<TxN>());
    auto x_T = ct::load_masked(x + offsets, mask, ct::zeros<TxN>());
    auto dyf = ct::element_cast<float>(dy_T);
    auto xf = ct::element_cast<float>(x_T);
    auto zero = ct::zeros<f32xN>();
    auto one = ct::full<f32xN>(1.0f);
    f32xN factor;
    if constexpr (OP == 0) {
        factor = ct::select(xf > zero, one, zero);
    } else if constexpr (OP == 1) {
        factor = ct::select(xf > zero, one, alpha * ct::exp(xf));
    } else if constexpr (OP == 2) {
        factor = ct::select(xf > zero, one, ct::full<f32xN>(alpha));
    } else if constexpr (OP == 3) {
        constexpr float scale = 1.0507009873554805f;
        constexpr float alpha_selu = 1.6732632423543772f;
        factor = scale * ct::select(xf > zero, one, alpha_selu * ct::exp(xf));
    } else if constexpr (OP == 4) {
        factor = ct::select(xf > zero, one, ct::exp(xf / alpha));
    } else {
        float avg_alpha = (lower + upper) * 0.5f;
        factor = ct::select(xf > zero, one, ct::full<f32xN>(avg_alpha));
    }
    auto out = dyf * factor;
    ct::store_masked(dx + offsets, ct::element_cast<T>(out), mask);
}
