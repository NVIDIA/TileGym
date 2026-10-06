// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_tile.h>

template <typename T, typename S, int HEAD_DIM, int H, int N_KV_CTX, int QUERY_GROUP_SIZE, int BANDWIDTH, int BLOCK_M,
          int BLOCK_N, int NUM_CTAS, int OCCUPANCY, bool ALIGNED_BASES>
[[cutile::hint(0, num_cta_in_cga = NUM_CTAS, occupancy = OCCUPANCY)]]
__tile_global__ void attention_sink_fwd_kernel(const T *__restrict__ Q, const T *__restrict__ K,
                                               const T *__restrict__ V, const S *__restrict__ Sinks,
                                               T *__restrict__ Out, const int *__restrict__ Start_q, float sm_scale,
                                               int Z, int N_Q_CTX) {
    namespace ct = cuda::tiles;
    if constexpr (ALIGNED_BASES) {
        Q = ct::assume_aligned<16>(Q);
        K = ct::assume_aligned<16>(K);
        V = ct::assume_aligned<16>(V);
        Out = ct::assume_aligned<16>(Out);
    }
    Z = ct::assume_bounded_below<0>(Z);
    N_Q_CTX = ct::assume_bounded_below<0>(N_Q_CTX);
    constexpr int KV_HEADS = H / QUERY_GROUP_SIZE;
    constexpr float LOG2E = 1.4426950408889634f;
    using QExt = ct::extents<int, ct::dynamic_extent, H, ct::dynamic_extent, HEAD_DIM>;
    using KExt = ct::extents<int, ct::dynamic_extent, KV_HEADS, HEAD_DIM, N_KV_CTX>;
    using KStride = ct::extents<int, KV_HEADS * N_KV_CTX * HEAD_DIM, N_KV_CTX * HEAD_DIM, 1, HEAD_DIM>;
    using VExt = ct::extents<int, ct::dynamic_extent, KV_HEADS, N_KV_CTX, HEAD_DIM>;
    auto q_view = ct::partition_view{ct::tensor_span{Q, QExt{Z, N_Q_CTX}}, ct::shape<1, 1, BLOCK_M, HEAD_DIM>{}};
    auto k_view = ct::partition_view{ct::tensor_span{K, ct::layout_strided_mapping{KExt{Z}, KStride{}}},
                                     ct::shape<1, 1, HEAD_DIM, BLOCK_N>{}};
    auto v_view = ct::partition_view{ct::tensor_span{V, VExt{Z}}, ct::shape<1, 1, BLOCK_N, HEAD_DIM>{}};
    auto out_view = ct::partition_view{ct::tensor_span{Out, QExt{Z, N_Q_CTX}}, ct::shape<1, 1, BLOCK_M, HEAD_DIM>{}};
    using F32M1 = ct::tile<float, ct::shape<BLOCK_M, 1>>;
    using F32MN = ct::tile<float, ct::shape<BLOCK_M, BLOCK_N>>;
    using F32MD = ct::tile<float, ct::shape<BLOCK_M, HEAD_DIM>>;
    int block_m = ct::bid().x;
    int batch = ct::bid().y / H;
    int head = ct::bid().y % H;
    int kv_head = head / QUERY_GROUP_SIZE;
    int start_q = Start_q[0];
    float sink = -INFINITY;
    if (Sinks != nullptr)
        sink = static_cast<float>(Sinks[head]);
    float qk_scale = sm_scale * LOG2E;
    float sink_scaled = sink * LOG2E;
    auto offs_m = block_m * BLOCK_M + ct::iota<ct::tile<int, ct::shape<BLOCK_M, 1>>>();
    auto offs_n = ct::iota<ct::tile<int, ct::shape<1, BLOCK_N>>>();
    auto m_i = ct::full<F32M1>(sink_scaled);
    auto l_i = ct::zeros<F32M1>();
    auto acc = ct::zeros<F32MD>();
    auto q = ct::reshape<ct::shape<BLOCK_M, HEAD_DIM>>(
        q_view.template load_masked<ct::view_padding::zero>(batch, head, block_m, 0));
    int lo = 0;
    if constexpr (BANDWIDTH > 0)
        lo = ct::max(0, start_q + block_m * BLOCK_M - BANDWIDTH);
    int hi = ct::min(start_q + (block_m + 1) * BLOCK_M, N_KV_CTX);
    for (auto j : ct::irange(lo / BLOCK_N, (hi + BLOCK_N - 1) / BLOCK_N)) {
        typename decltype(k_view)::view_tile_type k_raw;
        [[cutile::hint(0, latency = 6)]] k_raw =
            k_view.template load_masked<ct::view_padding::zero>(batch, kv_head, 0, j);
        auto k = ct::reshape<ct::shape<HEAD_DIM, BLOCK_N>>(k_raw);
        auto qk = ct::mma(q, k, ct::zeros<F32MN>());
        auto key_pos = j * BLOCK_N + offs_n;
        auto query_pos = start_q + offs_m;
        auto mask = (key_pos > query_pos) | (key_pos >= N_KV_CTX);
        if constexpr (BANDWIDTH > 0)
            mask = mask | (key_pos < query_pos - BANDWIDTH + 1);
        qk = qk + ct::select(mask, ct::full<F32MN>(-1.0e6f), ct::zeros<F32MN>());
        auto m_ij = ct::max(m_i, ct::reduce_max(qk, ct::integral_constant<1>{}) * qk_scale);
        auto p = ct::exp2(qk * qk_scale - m_ij, ct::round_subnormals_to_zero_t{});
        auto l_ij = ct::sum(p, ct::integral_constant<1>{});
        auto alpha = ct::exp2(m_i - m_ij, ct::round_subnormals_to_zero_t{});
        l_i = l_i * alpha + l_ij;
        acc = acc * alpha;
        typename decltype(v_view)::view_tile_type v_raw;
        [[cutile::hint(0, latency = 6)]] v_raw =
            v_view.template load_masked<ct::view_padding::zero>(batch, kv_head, j, 0);
        auto v = ct::reshape<ct::shape<BLOCK_N, HEAD_DIM>>(v_raw);
        acc = ct::mma(ct::element_cast<T>(p), v, acc);
        m_i = m_ij;
    }
    auto z = l_i + ct::exp2(sink_scaled - m_i, ct::round_subnormals_to_zero_t{});
    acc = ct::div(acc, z, ct::round_approximate_t{}, ct::round_subnormals_to_zero_t{});
    out_view.store_masked(ct::reshape<ct::shape<1, 1, BLOCK_M, HEAD_DIM>>(ct::element_cast<T>(acc)), batch, head,
                          block_m, 0);
}
