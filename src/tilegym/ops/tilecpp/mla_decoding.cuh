// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

constexpr float INV_LOG_2 = 1.442695040888963f;  // 1/ln(2)

/**
 * MLA Decoding: QK = Q @ K^T + QPE @ KPE^T, then softmax and V matmul
 * Inputs: Q [B, num_head, D], QPE [B, num_head, KPE], KV [B, S_kv, D], KPE_in [B, S_kv, KPE]
 *
 * Template Parameters:
 *   T: Element type (float, __half, __nv_bfloat16)
 *   BLOCK_D: Hidden dimension block size
 *   BLOCK_H: Head block size
 *   BLOCK_N: Sequence block size
 *   BLOCK_KPE: Position embedding dimension
 */
template<typename T, int BLOCK_D, int BLOCK_H, int BLOCK_N, int BLOCK_KPE>
__tile_global__ void naive_absorb_mla(
    T* __restrict__ Q, T* __restrict__ QPE,
    T* __restrict__ KV, T* __restrict__ KPE_in,
    T* __restrict__ Out, float* __restrict__ L,
    float sm_scale,
    int stride_qb, int stride_qm, int stride_qpeb, int stride_qpem,
    int stride_kvb, int stride_kvn, int stride_kpeb, int stride_kpem,
    int stride_ob, int stride_om, int B, int num_head, int S_kv) {
    namespace ct = cuda::tiles;

    Q      = ct::assume_aligned<16>(Q);
    QPE    = ct::assume_aligned<16>(QPE);
    KV     = ct::assume_aligned<16>(KV);
    KPE_in = ct::assume_aligned<16>(KPE_in);
    Out    = ct::assume_aligned<16>(Out);
    L      = ct::assume_aligned<16>(L);

    // Tile type definitions - use float for accumulation
    using f32_HxD = ct::tile<float, ct::shape<BLOCK_H, BLOCK_D>>;
    using f32_HxKPE = ct::tile<float, ct::shape<BLOCK_H, BLOCK_KPE>>;
    using f32_NxD = ct::tile<float, ct::shape<BLOCK_N, BLOCK_D>>;
    using f32_NxKPE = ct::tile<float, ct::shape<BLOCK_N, BLOCK_KPE>>;
    using f32_DxN = ct::tile<float, ct::shape<BLOCK_D, BLOCK_N>>;
    using f32_KPExN = ct::tile<float, ct::shape<BLOCK_KPE, BLOCK_N>>;
    using f32_HxN = ct::tile<float, ct::shape<BLOCK_H, BLOCK_N>>;
    using f32_H = ct::tile<float, ct::shape<BLOCK_H>>;
    using TxHxD = ct::tile<T, ct::shape<BLOCK_H, BLOCK_D>>;
    using TxHxKPE = ct::tile<T, ct::shape<BLOCK_H, BLOCK_KPE>>;
    using TxNxD = ct::tile<T, ct::shape<BLOCK_N, BLOCK_D>>;
    using TxNxKPE = ct::tile<T, ct::shape<BLOCK_N, BLOCK_KPE>>;
    using i32_H = ct::tile<int, ct::shape<BLOCK_H>>;
    using i32_N = ct::tile<int, ct::shape<BLOCK_N>>;
    using i32_D = ct::tile<int, ct::shape<BLOCK_D>>;
    using i32_KPE = ct::tile<int, ct::shape<BLOCK_KPE>>;

    int pid_x = ct::bid().x;
    int batch_idx = ct::bid().y;
    float qk_scale = sm_scale * INV_LOG_2;

    // Base pointers
    T* Q_base = Q + batch_idx * stride_qb + pid_x * BLOCK_H * stride_qm;
    T* QPE_base = QPE + batch_idx * stride_qpeb + pid_x * BLOCK_H * stride_qpem;
    T* KV_base = KV + batch_idx * stride_kvb;
    T* KPE_base = KPE_in + batch_idx * stride_kpeb;

    // Generate 1D offset tiles
    auto offs_h = ct::iota<i32_H>();
    auto offs_d = ct::iota<i32_D>();
    auto offs_kpe = ct::iota<i32_KPE>();
    auto offs_n = ct::iota<i32_N>();

    // Reshape to 2D for broadcasting
    auto offs_h_2d = ct::reshape(offs_h, ct::shape<BLOCK_H, 1>{});
    auto offs_d_2d = ct::reshape(offs_d, ct::shape<1, BLOCK_D>{});
    auto offs_kpe_2d = ct::reshape(offs_kpe, ct::shape<1, BLOCK_KPE>{});

    // Load Q: [BLOCK_H, BLOCK_D]
    auto q_flat_offs = offs_h_2d * stride_qm + offs_d_2d;
    auto q_ptrs = Q_base + q_flat_offs;
    auto q_mask = ct::reshape(offs_h + ct::full<i32_H>(pid_x * BLOCK_H) < num_head, ct::shape<BLOCK_H, 1>{});
    auto q_t = ct::load_masked(q_ptrs, q_mask, T(0));
    auto q = ct::element_cast<float>(q_t);

    // Load QPE: [BLOCK_H, BLOCK_KPE]
    auto qpe_flat_offs = offs_h_2d * stride_qpem + offs_kpe_2d;
    auto qpe_ptrs = QPE_base + qpe_flat_offs;
    auto qpe_t = ct::load_masked(qpe_ptrs, q_mask, T(0));
    auto qpe = ct::element_cast<float>(qpe_t);

    // Initialize accumulators
    auto m_i = ct::full<f32_H>(-1e30f);
    auto l_i = ct::full<f32_H>(1.0f);
    auto acc = ct::zeros<f32_HxD>();

    int num_kv_blocks = (S_kv + BLOCK_N - 1) / BLOCK_N;
    for (auto block_idx : ct::irange(0, num_kv_blocks)) {
        int curr_n = block_idx * BLOCK_N;

        auto k_row_base = ct::full<i32_N>(curr_n) + offs_n;
        auto k_row_2d = ct::reshape(k_row_base, ct::shape<BLOCK_N, 1>{});
        auto k_flat_offs = k_row_2d * stride_kvn + offs_d_2d;
        auto k_ptrs = KV_base + k_flat_offs;
        auto k_mask = ct::reshape(k_row_base < S_kv, ct::shape<BLOCK_N, 1>{});
        auto k_t = ct::load_masked(k_ptrs, k_mask, T(0));
        auto k = ct::element_cast<float>(k_t);  // [BLOCK_N, BLOCK_D]

        auto kpe_flat_offs = k_row_2d * stride_kpem + offs_kpe_2d;
        auto kpe_ptrs = KPE_base + kpe_flat_offs;
        auto kpe_block_t = ct::load_masked(kpe_ptrs, k_mask, T(0));
        auto kpe_block = ct::element_cast<float>(kpe_block_t);  // [BLOCK_N, BLOCK_KPE]

        // Compute QK = Q @ K^T: [BLOCK_H, BLOCK_D] @ [BLOCK_D, BLOCK_N] = [BLOCK_H, BLOCK_N]
        auto k_trans = ct::transpose(k);  // [BLOCK_D, BLOCK_N]
        auto qk = ct::mma(q, k_trans, ct::zeros<f32_HxN>());

        // Add QPE @ KPE^T: [BLOCK_H, BLOCK_KPE] @ [BLOCK_KPE, BLOCK_N]
        auto kpe_trans = ct::transpose(kpe_block);  // [BLOCK_KPE, BLOCK_N]
        qk = ct::mma(qpe, kpe_trans, qk);

        auto n_valid = ct::reshape(k_row_base < S_kv, ct::shape<1, BLOCK_N>{});
        qk = ct::select(n_valid, qk, ct::full<f32_HxN>(-1e6f));

        qk = qk * qk_scale;

        auto m_ij_2d = ct::reduce_max(qk, ct::integral_constant<1>{});
        auto m_ij = ct::reshape(m_ij_2d, ct::shape<BLOCK_H>{});
        m_ij = ct::max(m_i, m_ij);

        auto m_ij_bcast = ct::reshape(m_ij, ct::shape<BLOCK_H, 1>{});
        auto p = ct::exp2(qk - m_ij_bcast);

        auto l_curr_2d = ct::sum(p, ct::integral_constant<1>{});
        auto l_curr = ct::reshape(l_curr_2d, ct::shape<BLOCK_H>{});

        auto alpha = ct::exp2(m_i - m_ij);
        l_i = l_i * alpha + l_curr;

        auto alpha_2d = ct::reshape(alpha, ct::shape<BLOCK_H, 1>{});
        acc = acc * alpha_2d;

        auto v_t = ct::load_masked(k_ptrs, k_mask, T(0));
        auto v = ct::element_cast<float>(v_t);

        // Accumulate: acc += P @ V: [BLOCK_H, BLOCK_N] @ [BLOCK_N, BLOCK_D]
        acc = ct::mma(p, v, acc);

        m_i = m_ij;
    }

    // Finalize: output = acc / l_i
    auto l_i_2d = ct::reshape(l_i, ct::shape<BLOCK_H, 1>{});
    auto output = acc / l_i_2d;

    // Store output: [BLOCK_H, BLOCK_D]
    T* O_base = Out + batch_idx * stride_ob + pid_x * BLOCK_H * stride_om;
    auto o_flat_offs = offs_h_2d * stride_om + offs_d_2d;
    auto o_ptrs = O_base + o_flat_offs;
    ct::store_masked(o_ptrs, ct::element_cast<T>(output), q_mask);

    // Store L = m_i + log2(l_i): [BLOCK_H]
    float* L_base = L + batch_idx * num_head + pid_x * BLOCK_H;
    auto l_result = m_i + ct::log2(l_i);
    auto l_ptrs = L_base + offs_h;
    auto l_mask = offs_h + ct::full<i32_H>(pid_x * BLOCK_H) < num_head;
    ct::store_masked(l_ptrs, l_result, l_mask);
}


/**
 * MLA Decoding Transpose Kernel
 *
 * A transpose variant of the MLA decoding kernel that computes attention
 * in a transposed layout for potentially better cache efficiency.
 *
 * Computes: QK = K @ Q^T + KPE @ QPE^T (in transposed [N, H] order)
 * Then transposes back for final output.
 */
template<typename T, int BLOCK_D, int BLOCK_H, int BLOCK_N, int BLOCK_KPE>
__tile_global__ void naive_absorb_mla_transpose(
    T* __restrict__ Q, T* __restrict__ QPE,
    T* __restrict__ KV, T* __restrict__ KPE_in,
    T* __restrict__ Out, float* __restrict__ L,
    float sm_scale,
    int stride_qb, int stride_qm, int stride_qpeb, int stride_qpem,
    int stride_kvb, int stride_kvn, int stride_kpeb, int stride_kpem,
    int stride_ob, int stride_om, int B, int num_head, int S_kv) {
    namespace ct = cuda::tiles;

    Q      = ct::assume_aligned<16>(Q);
    QPE    = ct::assume_aligned<16>(QPE);
    KV     = ct::assume_aligned<16>(KV);
    KPE_in = ct::assume_aligned<16>(KPE_in);
    Out    = ct::assume_aligned<16>(Out);
    L      = ct::assume_aligned<16>(L);

    // Tile type definitions - use float for accumulation
    using f32_HxD = ct::tile<float, ct::shape<BLOCK_H, BLOCK_D>>;
    using f32_DxH = ct::tile<float, ct::shape<BLOCK_D, BLOCK_H>>;
    using f32_HxKPE = ct::tile<float, ct::shape<BLOCK_H, BLOCK_KPE>>;
    using f32_KPExH = ct::tile<float, ct::shape<BLOCK_KPE, BLOCK_H>>;
    using f32_NxD = ct::tile<float, ct::shape<BLOCK_N, BLOCK_D>>;
    using f32_DxN = ct::tile<float, ct::shape<BLOCK_D, BLOCK_N>>;
    using f32_NxKPE = ct::tile<float, ct::shape<BLOCK_N, BLOCK_KPE>>;
    using f32_NxH = ct::tile<float, ct::shape<BLOCK_N, BLOCK_H>>;
    using f32_H = ct::tile<float, ct::shape<BLOCK_H>>;
    using TxHxD = ct::tile<T, ct::shape<BLOCK_H, BLOCK_D>>;
    using TxHxKPE = ct::tile<T, ct::shape<BLOCK_H, BLOCK_KPE>>;
    using TxNxD = ct::tile<T, ct::shape<BLOCK_N, BLOCK_D>>;
    using TxNxKPE = ct::tile<T, ct::shape<BLOCK_N, BLOCK_KPE>>;
    using i32_H = ct::tile<int, ct::shape<BLOCK_H>>;
    using i32_N = ct::tile<int, ct::shape<BLOCK_N>>;
    using i32_D = ct::tile<int, ct::shape<BLOCK_D>>;
    using i32_KPE = ct::tile<int, ct::shape<BLOCK_KPE>>;

    int pid_x = ct::bid().x;
    int batch_idx = ct::bid().y;
    float qk_scale = sm_scale * INV_LOG_2;

    // Base pointers
    T* Q_base = Q + batch_idx * stride_qb + pid_x * BLOCK_H * stride_qm;
    T* QPE_base = QPE + batch_idx * stride_qpeb + pid_x * BLOCK_H * stride_qpem;
    T* KV_base = KV + batch_idx * stride_kvb;
    T* KPE_base = KPE_in + batch_idx * stride_kpeb;

    // Generate 1D offset tiles
    auto offs_h = ct::iota<i32_H>();
    auto offs_d = ct::iota<i32_D>();
    auto offs_kpe = ct::iota<i32_KPE>();
    auto offs_n = ct::iota<i32_N>();

    // Reshape to 2D for broadcasting
    auto offs_h_2d = ct::reshape(offs_h, ct::shape<BLOCK_H, 1>{});
    auto offs_d_2d = ct::reshape(offs_d, ct::shape<1, BLOCK_D>{});
    auto offs_kpe_2d = ct::reshape(offs_kpe, ct::shape<1, BLOCK_KPE>{});

    // Load Q: [BLOCK_H, BLOCK_D] and transpose to [BLOCK_D, BLOCK_H]
    auto q_flat_offs = offs_h_2d * stride_qm + offs_d_2d;
    auto q_ptrs = Q_base + q_flat_offs;
    auto q_mask = ct::reshape(offs_h + ct::full<i32_H>(pid_x * BLOCK_H) < num_head, ct::shape<BLOCK_H, 1>{});
    auto q_hd_t = ct::load_masked(q_ptrs, q_mask, T(0));
    auto q_hd = ct::element_cast<float>(q_hd_t);  // [BLOCK_H, BLOCK_D]
    auto q = ct::transpose(q_hd);  // [BLOCK_D, BLOCK_H]

    // Load QPE: [BLOCK_H, BLOCK_KPE] and transpose to [BLOCK_KPE, BLOCK_H]
    auto qpe_flat_offs = offs_h_2d * stride_qpem + offs_kpe_2d;
    auto qpe_ptrs = QPE_base + qpe_flat_offs;
    auto qpe_hk_t = ct::load_masked(qpe_ptrs, q_mask, T(0));
    auto qpe_hk = ct::element_cast<float>(qpe_hk_t);  // [BLOCK_H, BLOCK_KPE]
    auto qpe = ct::transpose(qpe_hk);  // [BLOCK_KPE, BLOCK_H]

    // Initialize accumulators in transposed layout [BLOCK_D, BLOCK_H]
    auto m_i = ct::full<f32_H>(-1e30f);
    auto l_i = ct::full<f32_NxH>(1.0f);  // [BLOCK_N, BLOCK_H] for intermediate sum
    auto acc = ct::zeros<f32_DxH>();  // [BLOCK_D, BLOCK_H]

    // Loop over K/V blocks — use ct::irange to keep `m_i`, `l_i`, `acc`
    // accumulator tiles in shared memory (avoids local-memory spills).
    int num_kv_blocks = (S_kv + BLOCK_N - 1) / BLOCK_N;
    for (auto block_idx : ct::irange(0, num_kv_blocks)) {
        int curr_n = block_idx * BLOCK_N;

        auto k_row_base = ct::full<i32_N>(curr_n) + offs_n;
        auto k_row_2d = ct::reshape(k_row_base, ct::shape<BLOCK_N, 1>{});
        auto k_flat_offs = k_row_2d * stride_kvn + offs_d_2d;
        auto k_ptrs = KV_base + k_flat_offs;
        auto k_mask = ct::reshape(k_row_base < S_kv, ct::shape<BLOCK_N, 1>{});
        auto k_t = ct::load_masked(k_ptrs, k_mask, T(0));
        auto k = ct::element_cast<float>(k_t);  // [BLOCK_N, BLOCK_D]

        auto kpe_flat_offs = k_row_2d * stride_kpem + offs_kpe_2d;
        auto kpe_ptrs = KPE_base + kpe_flat_offs;
        auto kpe_block_t = ct::load_masked(kpe_ptrs, k_mask, T(0));
        auto kpe_block = ct::element_cast<float>(kpe_block_t);  // [BLOCK_N, BLOCK_KPE]

        // Compute QK = K @ Q^T: [BLOCK_N, BLOCK_D] @ [BLOCK_D, BLOCK_H] = [BLOCK_N, BLOCK_H]
        auto qk = ct::mma(k, q, ct::zeros<f32_NxH>());  // [BLOCK_N, BLOCK_H]

        // Add KPE @ QPE^T: [BLOCK_N, BLOCK_KPE] @ [BLOCK_KPE, BLOCK_H]
        qk = ct::mma(kpe_block, qpe, qk);

        // Apply mask for out-of-bounds positions
        auto n_valid = ct::reshape(k_row_base < S_kv, ct::shape<BLOCK_N, 1>{});
        qk = ct::select(n_valid, qk, ct::full<f32_NxH>(-1e6f));

        // Scale by qk_scale
        qk = qk * qk_scale;

        // Compute column-wise max (dim 0): [BLOCK_H] - max over N dimension
        auto m_ij_2d = ct::reduce_max(qk, ct::integral_constant<0>{});  // [1, BLOCK_H]
        auto m_ij = ct::reshape(m_ij_2d, ct::shape<BLOCK_H>{});
        m_ij = ct::max(m_i, m_ij);

        // Compute exp2(qk - m_ij): [BLOCK_N, BLOCK_H]
        auto m_ij_bcast = ct::reshape(m_ij, ct::shape<1, BLOCK_H>{});
        auto p = ct::exp2(qk - m_ij_bcast);

        // Update running statistics
        auto alpha = ct::exp2(m_i - m_ij);
        auto alpha_bcast = ct::reshape(alpha, ct::shape<1, BLOCK_H>{});
        l_i = l_i * alpha_bcast + p;  // [BLOCK_N, BLOCK_H]

        // Scale accumulator [BLOCK_D, BLOCK_H]
        acc = acc * alpha_bcast;

        // Load V: [BLOCK_N, BLOCK_D] and transpose to [BLOCK_D, BLOCK_N]
        auto v_t = ct::load_masked(k_ptrs, k_mask, T(0));
        auto v = ct::element_cast<float>(v_t);  // [BLOCK_N, BLOCK_D]
        auto v_trans = ct::transpose(v);  // [BLOCK_D, BLOCK_N]

        // Accumulate: acc += V^T @ P: [BLOCK_D, BLOCK_N] @ [BLOCK_N, BLOCK_H] = [BLOCK_D, BLOCK_H]
        acc = ct::mma(v_trans, p, acc);

        m_i = m_ij;
    }

    // Sum l_i over N dimension: [BLOCK_N, BLOCK_H] -> [BLOCK_H]
    auto l_sum_2d = ct::sum(l_i, ct::integral_constant<0>{});  // [1, BLOCK_H]
    auto l_sum = ct::reshape(l_sum_2d, ct::shape<BLOCK_H>{});

    // Finalize: output = acc / l_sum, transposed back to [BLOCK_H, BLOCK_D]
    auto l_sum_bcast = ct::reshape(l_sum, ct::shape<1, BLOCK_H>{});
    acc = acc / l_sum_bcast;  // [BLOCK_D, BLOCK_H]
    auto output = ct::transpose(acc);  // [BLOCK_H, BLOCK_D]

    // Store output: [BLOCK_H, BLOCK_D]
    T* O_base = Out + batch_idx * stride_ob + pid_x * BLOCK_H * stride_om;
    auto o_flat_offs = offs_h_2d * stride_om + offs_d_2d;
    auto o_ptrs = O_base + o_flat_offs;
    ct::store_masked(o_ptrs, ct::element_cast<T>(output), q_mask);

    // Store L = m_i + log2(l_sum): [BLOCK_H]
    float* L_base = L + batch_idx * num_head + pid_x * BLOCK_H;
    auto l_result = m_i + ct::log2(l_sum);
    auto l_ptrs = L_base + offs_h;
    auto l_mask = offs_h + ct::full<i32_H>(pid_x * BLOCK_H) < num_head;
    ct::store_masked(l_ptrs, l_result, l_mask);
}
