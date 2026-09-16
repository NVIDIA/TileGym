// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


template<typename T, int BLOCK_SIZE>
using tile_t = cuda::tiles::tile<T, cuda::tiles::shape<BLOCK_SIZE>>;

template<int BLOCK_SIZE>
__tile__ auto sigmoid_f32(tile_t<float, BLOCK_SIZE> x) {
    namespace ct = cuda::tiles;
    return 1.0f / (1.0f + ct::exp(-x));
}

template<int BLOCK_SIZE>
__tile__ auto tanh_approx_f32(tile_t<float, BLOCK_SIZE> x) {
    return cuda::tiles::tanh(x);
}

template<int BLOCK_SIZE>
__tile__ auto erf_f32(tile_t<float, BLOCK_SIZE> x) {
    namespace ct = cuda::tiles;
    using f32xN = tile_t<float, BLOCK_SIZE>;
    constexpr float p  = 0.3275911f;
    constexpr float a1 = 0.254829592f;
    constexpr float a2 = -0.284496736f;
    constexpr float a3 = 1.421413741f;
    constexpr float a4 = -1.453152027f;
    constexpr float a5 = 1.061405429f;

    auto zero = ct::zeros<f32xN>();
    auto neg = x < zero;
    auto ax = ct::select(neg, -x, x);
    auto t = 1.0f / (1.0f + p * ax);
    auto poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
    auto r = 1.0f - poly * ct::exp(-ax * ax);
    return ct::select(neg, -r, r);
}

template<int BLOCK_SIZE>
__tile__ auto normal_cdf_f32(tile_t<float, BLOCK_SIZE> x) {
    constexpr float inv_sqrt_2 = 0.7071067811865476f;
    return 0.5f * (1.0f + erf_f32<BLOCK_SIZE>(x * inv_sqrt_2));
}

template<int BLOCK_SIZE>
__tile__ auto normal_pdf_f32(tile_t<float, BLOCK_SIZE> x) {
    namespace ct = cuda::tiles;
    constexpr float inv_sqrt_2pi = 0.3989422804014327f;
    return inv_sqrt_2pi * ct::exp(-0.5f * x * x);
}

template<typename T, int BLOCK_SIZE, int OP>
__tile_global__ void gelu_fwd_kernel(const T* __restrict__ x, T* __restrict__ y, int n_elements) {
    namespace ct = cuda::tiles;
    x = ct::assume_aligned<16>(x);
    y = ct::assume_aligned<16>(y);
    using TxN = tile_t<T, BLOCK_SIZE>;
    using f32xN = tile_t<float, BLOCK_SIZE>;
    using i32xN = tile_t<int32_t, BLOCK_SIZE>;

    int base = ct::bid().x * BLOCK_SIZE;
    auto offsets = ct::full<i32xN>(base) + ct::iota<i32xN>();
    auto mask = offsets < ct::full<i32xN>(n_elements);
    auto zero_T = ct::zeros<TxN>();
    auto x_T = ct::load_masked(x + offsets, mask, zero_T);
    auto xf = ct::element_cast<float>(x_T);
    f32xN out;

    if constexpr (OP == 0) {
        out = xf * normal_cdf_f32<BLOCK_SIZE>(xf);
    } else if constexpr (OP == 1) {
        out = 0.5f * xf * (1.0f + tanh_approx_f32<BLOCK_SIZE>(0.7978845608028654f * (xf + 0.044715f * xf * xf * xf)));
    } else {
        out = xf * normal_cdf_f32<BLOCK_SIZE>(xf);
    }

    ct::store_masked(y + offsets, ct::element_cast<T>(out), mask);
}

template<typename T, int BLOCK_SIZE, int OP>
__tile_global__ void gelu_bwd_kernel(const T* __restrict__ dy, const T* __restrict__ x, T* __restrict__ dx, int n_elements) {
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
    auto zero_T = ct::zeros<TxN>();
    auto dy_T = ct::load_masked(dy + offsets, mask, zero_T);
    auto x_T = ct::load_masked(x + offsets, mask, zero_T);
    auto dyf = ct::element_cast<float>(dy_T);
    auto xf = ct::element_cast<float>(x_T);
    f32xN grad;

    grad = dyf * (normal_cdf_f32<BLOCK_SIZE>(xf) + xf * normal_pdf_f32<BLOCK_SIZE>(xf));

    ct::store_masked(dx + offsets, ct::element_cast<T>(grad), mask);
}
