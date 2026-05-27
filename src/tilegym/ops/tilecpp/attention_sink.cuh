// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Attention Sink Kernel
 *
 * Implements attention with attention sinks for streaming/infinite context.
 * Supports sliding window attention with configurable bandwidth.
 *
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

/**
 * Attention Forward Kernel with Attention Sinks
 *
 * Supports sliding window attention with configurable bandwidth and
 * attention sinks for streaming/infinite context.
 *
 * Template parameters:
 *   T: Element type (__half or __nv_bfloat16)
 *   HEAD_DIM: Head dimension
 *   BLOCK_M: Block size for queries
 *   BLOCK_N: Block size for keys/values
 *   HAS_BANDWIDTH: Whether to use sliding window (true) or full attention (false)
 *
 * Grid: (cdiv(N_Q_CTX, BLOCK_M), Z * H, 1)
 */
template<typename T, int HEAD_DIM, int BLOCK_M, int BLOCK_N, bool HAS_BANDWIDTH>
__tile_global__ void attention_sink_fwd_kernel(
    T* __restrict__ Q_ptr,            // Query [Z, H, N_Q_CTX, HEAD_DIM]
    T* __restrict__ K_ptr,            // Key [Z, H, N_KV_CTX, HEAD_DIM]
    T* __restrict__ V_ptr,            // Value [Z, H, N_KV_CTX, HEAD_DIM]
    T* __restrict__ Sinks_ptr,        // Attention sinks per head [H] (can be nullptr), same dtype as Q/K/V
    float sm_scale,      // Softmax scale
    float* __restrict__ M_out_ptr,    // Max values output [Z, H, N_Q_CTX]
    T* __restrict__ Out_ptr,          // Output [Z, H, N_Q_CTX, HEAD_DIM]
    int start_q,         // Starting position for queries
    int Z,               // Batch size
    int H,               // Number of heads
    int N_Q_CTX,         // Query context length (padded)
    int N_KV_CTX,        // KV context length (padded)
    int bandwidth        // Sliding window bandwidth (0 for full attention)
) {
    namespace ct = cuda::tiles;

    // Apply alignment hints to tile-data pointers (Sinks_ptr is only used
    // for a scalar lookup and may be nullptr, so it is intentionally skipped).
    Q_ptr = ct::assume_aligned<16>(Q_ptr);
    K_ptr = ct::assume_aligned<16>(K_ptr);
    V_ptr = ct::assume_aligned<16>(V_ptr);
    M_out_ptr = ct::assume_aligned<16>(M_out_ptr);
    Out_ptr = ct::assume_aligned<16>(Out_ptr);

    // Tile type definitions
    using TxMxD = ct::tile<T, ct::shape<BLOCK_M, HEAD_DIM>>;
    using TxNxD = ct::tile<T, ct::shape<BLOCK_N, HEAD_DIM>>;
    using f32xMxD = ct::tile<float, ct::shape<BLOCK_M, HEAD_DIM>>;
    using f32xNxD = ct::tile<float, ct::shape<BLOCK_N, HEAD_DIM>>;
    using f32xMxN = ct::tile<float, ct::shape<BLOCK_M, BLOCK_N>>;
    using f32xM = ct::tile<float, ct::shape<BLOCK_M>>;
    using i32xM = ct::tile<int, ct::shape<BLOCK_M>>;
    using i32xN = ct::tile<int, ct::shape<BLOCK_N>>;
    using i32xD = ct::tile<int, ct::shape<HEAD_DIM>>;

    static_assert(BLOCK_N <= HEAD_DIM, "BLOCK_N must be <= HEAD_DIM");

    int start_m = ct::bid().x;
    int off_hz = ct::bid().y;
    int off_z = off_hz / H;
    int off_h = off_hz % H;

    // When no sink tensor is provided (Sinks_ptr == nullptr) we want the kernel
    // to behave exactly like a regular softmax: m_i starts at -INFINITY (so
    // m_i = max(qk) ends up the true row max) and sink_exp = exp(-INF - m_i) = 0
    // so the sink contribution drops out of the denominator z = l_i + sink_exp.
    float sink = -INFINITY;
    if (Sinks_ptr != nullptr) {
        sink = static_cast<float>(Sinks_ptr[off_h]);
    }

    // Generate offset tiles
    auto offs_m = ct::iota<i32xM>();  // [0, 1, ..., BLOCK_M-1]
    auto offs_n = ct::iota<i32xN>();  // [0, 1, ..., BLOCK_N-1]
    auto offs_d = ct::iota<i32xD>();  // [0, 1, ..., HEAD_DIM-1]

    // Reshape for 2D indexing
    auto offs_m_2d = ct::reshape(offs_m, ct::shape<BLOCK_M, 1>{});
    auto offs_d_2d = ct::reshape(offs_d, ct::shape<1, HEAD_DIM>{});
    auto offs_n_2d = ct::reshape(offs_n, ct::shape<BLOCK_N, 1>{});

    // Initialize accumulators
    auto m_i = ct::full<f32xM>(sink);  // Initialize max with sink value
    auto l_i = ct::zeros<f32xM>();
    auto acc = ct::zeros<f32xMxD>();

    // Calculate Q indices and mask
    auto q_m_indices = ct::full<i32xM>(start_m * BLOCK_M) + offs_m;
    auto q_m_mask = q_m_indices < N_Q_CTX;

    // Base pointers
    int q_batch_head_stride = H * N_Q_CTX * HEAD_DIM;
    int q_head_stride = N_Q_CTX * HEAD_DIM;
    T* Q_base = Q_ptr + off_z * q_batch_head_stride + off_h * q_head_stride;

    int kv_batch_head_stride = H * N_KV_CTX * HEAD_DIM;
    int kv_head_stride = N_KV_CTX * HEAD_DIM;
    T* K_base = K_ptr + off_z * kv_batch_head_stride + off_h * kv_head_stride;
    T* V_base = V_ptr + off_z * kv_batch_head_stride + off_h * kv_head_stride;

    // Load Q block: [BLOCK_M, HEAD_DIM]
    auto q_m_offs = ct::reshape(q_m_indices, ct::shape<BLOCK_M, 1>{}) * HEAD_DIM;
    auto q_flat_offs = q_m_offs + offs_d_2d;
    auto q_ptrs = Q_base + q_flat_offs;
    auto q_mask_2d = ct::reshape(q_m_mask, ct::shape<BLOCK_M, 1>{});
    auto q = ct::element_cast<float>(ct::load_masked(q_ptrs, q_mask_2d, T(0)));

    // Calculate KV range to process
    int lo, hi;
    if constexpr (HAS_BANDWIDTH) {
        int q_start = start_q + start_m * BLOCK_M - bandwidth;
        lo = (q_start > 0) ? q_start : 0;
        hi = start_q + (start_m + 1) * BLOCK_M;
    } else {
        lo = 0;
        hi = start_q + (start_m + 1) * BLOCK_M;
    }

    // Make lo a multiple of BLOCK_N
    lo = (lo / BLOCK_N) * BLOCK_N;

    // Process KV blocks
    for (auto start_n : ct::irange(lo, hi, BLOCK_N)) {
        auto k_n_indices = ct::full<i32xN>(start_n) + offs_n;
        auto k_n_mask = k_n_indices < N_KV_CTX;

        // Load K block: [BLOCK_N, HEAD_DIM]
        auto k_n_offs = ct::reshape(k_n_indices, ct::shape<BLOCK_N, 1>{}) * HEAD_DIM;
        auto k_flat_offs = k_n_offs + offs_d_2d;
        auto k_ptrs = K_base + k_flat_offs;
        auto k_mask_2d = ct::reshape(k_n_mask, ct::shape<BLOCK_N, 1>{});
        auto k = ct::element_cast<float>(ct::load_masked(k_ptrs, k_mask_2d, T(0)));

        // Compute QK = Q @ K^T -> [BLOCK_M, BLOCK_N]
        auto k_t = ct::transpose(k);
        auto qk = ct::matmul(q, k_t);
        qk = qk * sm_scale;

        // Apply causal mask and optional sliding window mask
        auto q_pos = ct::reshape(ct::full<i32xM>(start_q) + q_m_indices, ct::shape<BLOCK_M, 1>{});
        auto k_pos = ct::reshape(k_n_indices, ct::shape<1, BLOCK_N>{});

        // Causal mask: mask if key position > query position
        auto causal_mask = q_pos >= k_pos;
        qk = ct::select(causal_mask, qk, ct::full<f32xMxN>(-1.0e6f));

        // Sliding window mask: mask if key is too old
        if constexpr (HAS_BANDWIDTH) {
            auto window_start = q_pos - bandwidth + 1;
            auto window_mask = k_pos >= window_start;
            qk = ct::select(window_mask, qk, ct::full<f32xMxN>(-1.0e6f));
        }

        // Apply length masking
        auto n_valid = ct::reshape(k_n_mask, ct::shape<1, BLOCK_N>{});
        qk = ct::select(n_valid, qk, ct::full<f32xMxN>(-1.0e6f));

        // Compute row-wise max
        auto qk_max = ct::reduce_max<1>(qk);
        auto m_ij = ct::max(m_i, ct::reshape(qk_max, ct::shape<BLOCK_M>{}));

        // Compute alpha = exp(m_i - m_ij) for rescaling
        auto alpha = ct::exp(m_i - m_ij);

        // Compute exp(qk - m_ij) -> p
        auto m_ij_2d = ct::reshape(m_ij, ct::shape<BLOCK_M, 1>{});
        auto p = ct::exp(qk - m_ij_2d);

        // Compute row-wise sum of p
        auto p_sum = ct::sum<1>(p);
        auto l_ij = ct::reshape(p_sum, ct::shape<BLOCK_M>{});

        // Scale accumulator by alpha
        auto alpha_2d = ct::reshape(alpha, ct::shape<BLOCK_M, 1>{});
        acc = acc * alpha_2d;

        // Load V block: [BLOCK_N, HEAD_DIM]
        auto v_n_offs = ct::reshape(k_n_indices, ct::shape<BLOCK_N, 1>{}) * HEAD_DIM;
        auto v_flat_offs = v_n_offs + offs_d_2d;
        auto v_ptrs = V_base + v_flat_offs;
        auto v = ct::element_cast<float>(ct::load_masked(v_ptrs, k_mask_2d, T(0)));

        // Accumulate: acc += p @ v
        acc = ct::mma(p, v, acc);

        // Update l_i and m_i
        l_i = l_i * alpha + l_ij;
        m_i = m_ij;
    }

    // Finalize output
    // sink_scaled = exp(sink - m_i)
    // z = l_i + sink_scaled
    // acc = acc / z
    // m_i += log(l_i)
    auto sink_scaled = ct::exp(ct::full<f32xM>(sink) - m_i);
    auto z = l_i + sink_scaled;

    // Normalize accumulator
    auto z_safe = ct::select(z > ct::full<f32xM>(0.0f), z, ct::full<f32xM>(1.0f));
    auto z_inv = ct::full<f32xM>(1.0f) / z_safe;
    auto z_inv_2d = ct::reshape(z_inv, ct::shape<BLOCK_M, 1>{});
    auto output = acc * z_inv_2d;

    // Update m_i for output
    auto l_i_safe = ct::select(l_i > ct::full<f32xM>(1e-10f), l_i, ct::full<f32xM>(1e-10f));
    auto m_final = m_i + ct::log(l_i_safe);

    // Store M output
    float* M_base = M_out_ptr + off_hz * N_Q_CTX;
    ct::store_masked(M_base + q_m_indices, m_final, q_m_mask);

    // Store output
    T* O_base = Out_ptr + off_z * q_batch_head_stride + off_h * q_head_stride;
    auto o_m_offs = ct::reshape(q_m_indices, ct::shape<BLOCK_M, 1>{}) * HEAD_DIM;
    auto o_flat_offs = o_m_offs + offs_d_2d;
    auto o_ptrs = O_base + o_flat_offs;
    ct::store_masked(o_ptrs, ct::element_cast<T>(output), q_mask_2d);
}
