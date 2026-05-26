// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Persistent Layer Normalization forward kernel.
 *
 *
 * Grid: (NUM_SMS, 1, 1) - persistent grid-stride pattern.
 * Each block processes BLOCK_N rows x BLOCK_D cols per iteration and strides
 * by NUM_SMS * BLOCK_N across the N dimension.
 *
 *   X:    (N, D)         - input dtype T
 *   Y:    (N, D)         - output dtype T
 *   W:    (D,)           - input dtype T (weight)
 *   B:    (D,)           - input dtype T (bias)
 *   Mean: (N,)           - float32
 *   Rstd: (N,)           - float32
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template<typename T,
         int BLOCK_N,
         int BLOCK_D,
         bool IS_SWISH,
         bool TRAINING,
         bool COMPUTE_MEAN_AND_RSTD,
         int N,
         int D,        // ← compile-time so `(... / D)` and partition_view extents fold.
         int NUM_SMS,  // ← compile-time so the persistent for-loop step is constant.
         float EPS>    // ← compile-time so the (var + eps) reshape/broadcast hoists out of the for-loop
[[ using cutile : hint(1000, num_cta_in_cga=1) ]]
__tile_global__ void persistent_layer_norm_fwd_kernel(
    const T* __restrict__ X,       // (N, D)
    T* __restrict__ Y,             // (N, D)
    const T* __restrict__ W,       // (D,)
    const T* __restrict__ B,       // (D,)
    float* __restrict__ Mean,      // (N,)
    float* __restrict__ Rstd       // (N,)
) {
    namespace ct = cuda::tiles;

    using f32_NxD   = ct::tile<float, ct::shape<BLOCK_N, BLOCK_D>>;
    using f32_N     = ct::tile<float, ct::shape<BLOCK_N>>;

    X    = ct::assume_aligned<16>(X);
    Y    = ct::assume_aligned<16>(Y);
    W    = ct::assume_aligned<16>(W);
    B    = ct::assume_aligned<16>(B);
    Mean = ct::assume_aligned<16>(Mean);
    Rstd = ct::assume_aligned<16>(Rstd);

    int pid = ct::bid().x;
    constexpr int upper_bound = (N + BLOCK_N - 1) / BLOCK_N;

    // Partitioned views with compile-time extents (N, D are template NTTPs).
    // Static extents → TMA-friendly codegen + no runtime stride machinery.
    using ExtND = ct::extents<uint32_t, static_cast<uint32_t>(N), static_cast<uint32_t>(D)>;
    using ExtD  = ct::extents<uint32_t, static_cast<uint32_t>(D)>;
    using ExtN  = ct::extents<uint32_t, static_cast<uint32_t>(N)>;
    auto pX = ct::partition_view(
        ct::tensor_span{X, ExtND{}},
        ct::shape<BLOCK_N, BLOCK_D>{});
    auto pY = ct::partition_view(
        ct::tensor_span{Y, ExtND{}},
        ct::shape<BLOCK_N, BLOCK_D>{});
    auto pW = ct::partition_view(
        ct::tensor_span{W, ExtD{}},
        ct::shape<BLOCK_D>{});
    auto pB = ct::partition_view(
        ct::tensor_span{B, ExtD{}},
        ct::shape<BLOCK_D>{});
    auto pMean = ct::partition_view(
        ct::tensor_span{Mean, ExtN{}},
        ct::shape<BLOCK_N>{});
    auto pRstd = ct::partition_view(
        ct::tensor_span{Rstd, ExtN{}},
        ct::shape<BLOCK_N>{});

    // Load weights once (hoisted out of the grid-stride loop).
    auto w = ct::element_cast<float>(pW.load(0));  // (BLOCK_D,)
    auto b = ct::element_cast<float>(pB.load(0));  // (BLOCK_D,)

    // Broadcast weights into (BLOCK_N, BLOCK_D) by reshape to (1, BLOCK_D).
    auto w_bcast = ct::reshape<ct::shape<1, BLOCK_D>>(w);
    auto b_bcast = ct::reshape<ct::shape<1, BLOCK_D>>(b);

    constexpr float inv_D_scalar = 1.0f / static_cast<float>(D);
    auto inv_D_tile = ct::full<f32_N>(inv_D_scalar);
    auto eps_tile   = ct::full<f32_N>(EPS);

    using TileXNxD = ct::tile<T, ct::shape<BLOCK_N, BLOCK_D>>;

    for (auto current_pid : ct::irange(pid, upper_bound, NUM_SMS)) {
        TileXNxD x_tile;
        [[ using cutile : hint(1000, latency=4) ]]
        x_tile = pX.load(current_pid, 0);
        auto x = ct::element_cast<float>(x_tile);

        f32_N mean;
        f32_N rstd;

        if constexpr (COMPUTE_MEAN_AND_RSTD) {
            // Step 1: Compute x^2 then sum/mean.  Use the loop-invariant
            // `inv_D_tile` and `eps_tile` (built outside the loop) so the
            // reshape + broadcast of those scalars stays hoisted.
            auto x_squared = x * x;
            auto avg_square_2d = ct::sum<1>(x_squared);
            auto avg_square = ct::reshape<ct::shape<BLOCK_N>>(avg_square_2d) * inv_D_tile;
            auto mean_2d = ct::sum<1>(x);
            mean = ct::reshape<ct::shape<BLOCK_N>>(mean_2d) * inv_D_tile;
            auto var = avg_square - mean * mean;

            rstd = ct::rsqrt(var + eps_tile);

            if constexpr (TRAINING) {
                [[ using cutile : hint(1000, allow_tma=false) ]]
                pMean.store(mean, current_pid);
                [[ using cutile : hint(1000, allow_tma=false) ]]
                pRstd.store(rstd, current_pid);
            }
        } else {
            mean = pMean.load(current_pid);
            rstd = pRstd.load(current_pid);
        }

        // Broadcast mean/rstd to (BLOCK_N, 1) then rely on implicit broadcast
        // against (BLOCK_N, BLOCK_D).
        auto mean_col = ct::reshape<ct::shape<BLOCK_N, 1>>(mean);
        auto rstd_col = ct::reshape<ct::shape<BLOCK_N, 1>>(rstd);

        auto x_hat = (x - mean_col) * rstd_col;
        auto y_f32 = x_hat * w_bcast + b_bcast;

        if constexpr (IS_SWISH) {
            auto one = ct::full<f32_NxD>(1.0f);
            auto sig = one / (one + ct::exp(-y_f32));
            y_f32 = sig * x;
        }

        auto y_T = ct::element_cast<T>(y_f32);
        [[ using cutile : hint(1000, allow_tma=false) ]]
        pY.store(y_T, current_pid, 0);
    }
}
