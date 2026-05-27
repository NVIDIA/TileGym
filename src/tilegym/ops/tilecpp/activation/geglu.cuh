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
    return 2.0f * sigmoid_f32<BLOCK_SIZE>(2.0f * x) - 1.0f;
}

template<int BLOCK_SIZE>
__tile__ auto normal_cdf_f32(tile_t<float, BLOCK_SIZE> x) {
    constexpr float sqrt_2_div_pi = 0.7978845608028654f;
    constexpr float coeff_044715 = 0.044715f;
    auto x3 = x * x * x;
    return 0.5f * (1.0f + tanh_approx_f32<BLOCK_SIZE>(sqrt_2_div_pi * (x + coeff_044715 * x3)));
}

template<int BLOCK_SIZE>
__tile__ auto normal_pdf_f32(tile_t<float, BLOCK_SIZE> x) {
    namespace ct = cuda::tiles;
    constexpr float inv_sqrt_2pi = 0.3989422804014327f;
    return inv_sqrt_2pi * ct::exp(-0.5f * x * x);
}

template<typename T, int BLOCK_SIZE, int APPROXIMATE>
__tile_global__ void geglu_fwd_kernel(const T* __restrict__ x, T* __restrict__ y, int N, int m_stride, int my_stride, int n_elements) {
    namespace ct = cuda::tiles;
    x = ct::assume_aligned<16>(x);
    y = ct::assume_aligned<16>(y);
    using TxN = tile_t<T, BLOCK_SIZE>;
    using i32xN = tile_t<int32_t, BLOCK_SIZE>;

    int base = ct::bid().x * BLOCK_SIZE;
    auto gid = ct::full<i32xN>(base) + ct::iota<i32xN>();
    auto mask = gid < ct::full<i32xN>(n_elements);
    auto m_id = gid / ct::full<i32xN>(N);
    auto n_offs = gid - m_id * ct::full<i32xN>(N);
    auto left_offsets = m_id * ct::full<i32xN>(m_stride) + n_offs;
    auto right_offsets = left_offsets + ct::full<i32xN>(N);
    auto out_offsets = m_id * ct::full<i32xN>(my_stride) + n_offs;
    auto zero_T = ct::zeros<TxN>();
    auto a_T = ct::load_masked(x + left_offsets, mask, zero_T);
    auto b_T = ct::load_masked(x + right_offsets, mask, zero_T);
    auto a = ct::element_cast<float>(a_T);
    auto b = ct::element_cast<float>(b_T);
    auto gelu_b = b * normal_cdf_f32<BLOCK_SIZE>(b);
    if constexpr (APPROXIMATE == 1) {
        gelu_b = 0.5f * b * (1.0f + tanh_approx_f32<BLOCK_SIZE>(0.7978845608028654f * (b + 0.044715f * b * b * b)));
    }
    ct::store_masked(y + out_offsets, ct::element_cast<T>(a * gelu_b), mask);
}

template<typename T, int BLOCK_SIZE, int APPROXIMATE>
__tile_global__ void geglu_bwd_kernel(T* __restrict__ dx, const T* __restrict__ dy, const T* __restrict__ x, int N, int m_stride, int my_stride, int n_elements) {
    namespace ct = cuda::tiles;
    dx = ct::assume_aligned<16>(dx);
    dy = ct::assume_aligned<16>(dy);
    x = ct::assume_aligned<16>(x);
    using TxN = tile_t<T, BLOCK_SIZE>;
    using i32xN = tile_t<int32_t, BLOCK_SIZE>;

    int base = ct::bid().x * BLOCK_SIZE;
    auto gid = ct::full<i32xN>(base) + ct::iota<i32xN>();
    auto mask = gid < ct::full<i32xN>(n_elements);
    auto m_id = gid / ct::full<i32xN>(N);
    auto n_offs = gid - m_id * ct::full<i32xN>(N);
    auto left_offsets = m_id * ct::full<i32xN>(m_stride) + n_offs;
    auto right_offsets = left_offsets + ct::full<i32xN>(N);
    auto out_offsets = m_id * ct::full<i32xN>(my_stride) + n_offs;
    auto zero_T = ct::zeros<TxN>();
    auto a_T = ct::load_masked(x + left_offsets, mask, zero_T);
    auto b_T = ct::load_masked(x + right_offsets, mask, zero_T);
    auto dy_T = ct::load_masked(dy + out_offsets, mask, zero_T);
    auto a = ct::element_cast<float>(a_T);
    auto b = ct::element_cast<float>(b_T);
    auto dyf = ct::element_cast<float>(dy_T);
    auto gelu_b = b * normal_cdf_f32<BLOCK_SIZE>(b);
    if constexpr (APPROXIMATE == 1) {
        gelu_b = 0.5f * b * (1.0f + tanh_approx_f32<BLOCK_SIZE>(0.7978845608028654f * (b + 0.044715f * b * b * b)));
    }
    auto da = dyf * gelu_b;
    auto db = dyf * a * (normal_cdf_f32<BLOCK_SIZE>(b) + b * normal_pdf_f32<BLOCK_SIZE>(b));
    ct::store_masked(dx + left_offsets, ct::element_cast<T>(da), mask);
    ct::store_masked(dx + right_offsets, ct::element_cast<T>(db), mask);
}
