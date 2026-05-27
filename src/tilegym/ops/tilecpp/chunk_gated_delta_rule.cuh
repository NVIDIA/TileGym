// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT

/**
 * Chunk Gated Delta Rule (Qwen3-Next linear-attention variant, chunked form).
 *
 *
 * The algorithm splits the sequence into chunks of size CHUNK_SIZE and:
 *   1) (intra) For each (b, h, chunk), precompute
 *        - chunked Q, K (optionally L2-normalized and scaled)
 *        - per-chunk g_cum = cumsum(g)
 *        - attn = (I + A)^{-1} where A is the strict-lower-tri part of
 *          -((k*beta) @ k^T) * decay_mask(exp(g_cum[i] - g_cum[j]))
 *        - v_corrected[chunk] = attn @ (v * beta)
 *        - k_cumdecay[chunk]  = attn @ (k_beta * exp(g_cum))
 *   2) (inter) For each (b, h, v_tile), run the chunk recurrence with a
 *      (BLOCK_K, BLOCK_V) running state, emitting (B, H, num_chunks, CS, V)
 *      output.
 */

#pragma once

#include <cuda_tile.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

template<int CS, typename TileType>
__tile__ inline TileType cgdr_solve_tril(TileType A) {
    namespace ct = cuda::tiles;
    using i32_CS    = ct::tile<int,   ct::shape<CS>>;
    using i32_CSx1  = ct::tile<int,   ct::shape<CS, 1>>;
    using f32_CSx1  = ct::tile<float, ct::shape<CS, 1>>;
    using f32_1xCS  = ct::tile<float, ct::shape<1, CS>>;
    using f32_CS    = ct::tile<float, ct::shape<CS>>;

    auto offs = ct::iota<i32_CS>();

    for (auto i : ct::irange(1, CS)) {
        auto is_row = offs == ct::full<i32_CS>(i);                       // (CS,)
        auto is_row_col = ct::reshape<ct::shape<CS, 1>>(is_row);          // (CS, 1)

        // Extract row i of A:  sum_axis0( where(is_row_col, A, 0) )  ->  (CS,).
        auto masked_A = ct::select(is_row_col, A, ct::zeros<TileType>());
        auto row_i    = ct::sum<0>(masked_A);                             // (CS,)

        // Correction: corr = row_i @ A  ->  (CS,).
        auto row_i_col = ct::reshape<ct::shape<CS, 1>>(row_i);
        auto prod      = row_i_col * A;                                    // broadcast (CS,1)*(CS,CS)
        auto corr      = ct::sum<0>(prod);                                 // (CS,)

        // A[i, :] += corr, expressed as masked add of a rank-1 update.
        auto corr_row  = ct::reshape<ct::shape<1, CS>>(corr);              // (1, CS)
        // cast bool (is_row_col) to float for broadcast multiplication
        auto mask_f    = ct::select(is_row_col,
                                     ct::full<f32_CSx1>(1.0f),
                                     ct::zeros<f32_CSx1>());
        auto update    = mask_f * corr_row;                                 // (CS, CS)
        A = A + update;
    }
    // Add identity diagonal.
    using f32_CSxCS = TileType;
    auto offs_r = ct::reshape<ct::shape<CS, 1>>(offs);
    auto offs_c = ct::reshape<ct::shape<1, CS>>(offs);
    auto diag   = offs_r == offs_c;                                         // bool (CS, CS)
    auto eye    = ct::select(diag, ct::full<f32_CSxCS>(1.0f),
                              ct::zeros<f32_CSxCS>());
    return A + eye;
}


// ============================================================================
// Intra-chunk prepare kernel.
// Grid: (B * NUM_HEADS, num_chunks, 1).
// ============================================================================
template<typename T,
         int CHUNK_SIZE,
         int BLOCK_K,
         bool USE_QK_L2NORM,
         int occupancy>
[[ using cutile : hint(1000, occupancy=occupancy) ]]
__tile_global__ void chunk_gated_delta_rule_intra_kernel(
    const T* __restrict__ Q,           // (B, T, H, K)
    const T* __restrict__ K,           // (B, T, H, K)
    const T* __restrict__ V,           // (B, T, H, V)
    const T* __restrict__ Beta,        // (B, T, H)
    const T* __restrict__ G,           // (B, T, H)
    float* __restrict__ Q_out,         // (B, H, num_chunks, CHUNK_SIZE, K)
    float* __restrict__ K_out,         // (B, H, num_chunks, CHUNK_SIZE, K)
    float* __restrict__ V_corr,        // (B, H, num_chunks, CHUNK_SIZE, V)
    float* __restrict__ K_cumdecay,    // (B, H, num_chunks, CHUNK_SIZE, K)
    float* __restrict__ G_cum_out,     // (B, H, num_chunks, CHUNK_SIZE)
    float scale,
    int B,
    int seq_len,
    int NUM_HEADS,
    int num_chunks,
    int K_dim,
    int V_dim
) {
    namespace ct = cuda::tiles;

    Q = ct::assume_aligned<16>(Q);
    K = ct::assume_aligned<16>(K);
    V = ct::assume_aligned<16>(V);
    Beta = ct::assume_aligned<16>(Beta);
    G = ct::assume_aligned<16>(G);
    Q_out      = ct::assume_aligned<16>(Q_out);
    K_out      = ct::assume_aligned<16>(K_out);
    V_corr     = ct::assume_aligned<16>(V_corr);
    K_cumdecay = ct::assume_aligned<16>(K_cumdecay);
    G_cum_out  = ct::assume_aligned<16>(G_cum_out);

    int pid_bh    = ct::bid().x;
    int pid_chunk = ct::bid().y;
    int b = pid_bh / NUM_HEADS;
    int h = pid_bh % NUM_HEADS;

    using f32_CSxK   = ct::tile<float, ct::shape<CHUNK_SIZE, BLOCK_K>>;
    using f32_CS     = ct::tile<float, ct::shape<CHUNK_SIZE>>;
    using f32_CSx1   = ct::tile<float, ct::shape<CHUNK_SIZE, 1>>;
    using f32_1xCS   = ct::tile<float, ct::shape<1, CHUNK_SIZE>>;
    using f32_CSxCS  = ct::tile<float, ct::shape<CHUNK_SIZE, CHUNK_SIZE>>;
    using i32_CS     = ct::tile<int,   ct::shape<CHUNK_SIZE>>;

    // Views. Q/K have shape (B, T, H, K) partitioned along T into CHUNK_SIZE blocks.
    auto pQ = ct::partition_view(
        ct::tensor_span{Q, ct::extents{B, seq_len, NUM_HEADS, K_dim}},
        ct::shape<1, CHUNK_SIZE, 1, BLOCK_K>{});
    auto pK = ct::partition_view(
        ct::tensor_span{K, ct::extents{B, seq_len, NUM_HEADS, K_dim}},
        ct::shape<1, CHUNK_SIZE, 1, BLOCK_K>{});
    auto pV = ct::partition_view(
        ct::tensor_span{V, ct::extents{B, seq_len, NUM_HEADS, V_dim}},
        ct::shape<1, CHUNK_SIZE, 1, BLOCK_K>{});  // iterate V in BLOCK_K-wide stripes
    auto pBeta = ct::partition_view(
        ct::tensor_span{Beta, ct::extents{B, seq_len, NUM_HEADS}},
        ct::shape<1, CHUNK_SIZE, 1>{});
    auto pG = ct::partition_view(
        ct::tensor_span{G, ct::extents{B, seq_len, NUM_HEADS}},
        ct::shape<1, CHUNK_SIZE, 1>{});

    // Output views.
    auto pQout = ct::partition_view(
        ct::tensor_span{Q_out, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, K_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>{});
    auto pKout = ct::partition_view(
        ct::tensor_span{K_out, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, K_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>{});
    auto pVcorr = ct::partition_view(
        ct::tensor_span{V_corr, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, V_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>{});
    auto pKcd = ct::partition_view(
        ct::tensor_span{K_cumdecay, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, K_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>{});
    auto pGcum = ct::partition_view(
        ct::tensor_span{G_cum_out, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE}},
        ct::shape<1, 1, 1, CHUNK_SIZE>{});

    // Load chunk of K (CHUNK_SIZE, BLOCK_K).
    auto k_4d = pK.load(b, pid_chunk, h, 0);
    auto k    = ct::element_cast<float>(ct::reshape<ct::shape<CHUNK_SIZE, BLOCK_K>>(k_4d));
    if constexpr (USE_QK_L2NORM) {
        auto k_sq       = k * k;
        auto k_norm2_2d = ct::sum<1>(k_sq);                                  // (CS, 1)
        auto k_norm2    = ct::reshape<ct::shape<CHUNK_SIZE>>(k_norm2_2d);    // (CS,)
        auto inv        = ct::rsqrt(k_norm2 + ct::full<f32_CS>(1e-6f));
        auto inv_col    = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(inv);
        k = k * inv_col;
    }

    // Load beta, g (CHUNK_SIZE,).
    auto beta_3d = pBeta.load(b, pid_chunk, h);
    auto beta    = ct::element_cast<float>(ct::reshape<ct::shape<CHUNK_SIZE>>(beta_3d));
    auto g_3d    = pG.load(b, pid_chunk, h);
    auto g_raw   = ct::element_cast<float>(ct::reshape<ct::shape<CHUNK_SIZE>>(g_3d));

    // Cumulative sum of g along chunk (sequential, small CHUNK_SIZE).
    auto g_cum = ct::zeros<f32_CS>();
    {
        // We implement cumsum as a sequential scan by reshaping into a (CS,1)
        // and iterating. Produce g_cum via the tile's sum-broadcast identity.
        // Simpler: load into a mutable tile, compute prefix sum.
        // Use a (CS, CS) lower triangular mask and matmul-like sum.
        auto offs     = ct::iota<i32_CS>();
        auto offs_r   = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(offs);
        auto offs_c   = ct::reshape<ct::shape<1, CHUNK_SIZE>>(offs);
        auto lower_eq = offs_r >= offs_c;                              // (CS, CS)
        auto g_row    = ct::reshape<ct::shape<1, CHUNK_SIZE>>(g_raw);   // (1, CS)
        auto g_masked = ct::select(lower_eq,
                                     ct::full<f32_CSxCS>(0.0f) + g_row,
                                     ct::zeros<f32_CSxCS>());
        // sum<1> on (CS, CS) returns (CS, 1); flatten to (CS,).
        auto g_cum_2d = ct::sum<1>(g_masked);
        g_cum = ct::reshape<ct::shape<CHUNK_SIZE>>(g_cum_2d);
    }

    // k_beta = k * beta[:, None]
    auto beta_col = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(beta);
    auto k_beta   = k * beta_col;

    auto k_T  = ct::transpose(k);
    auto base_attn = ct::matmul(k_beta, k_T);  // (CS, CS) fp32

    auto gc_r   = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(g_cum);
    auto gc_c   = ct::reshape<ct::shape<1, CHUNK_SIZE>>(g_cum);
    auto decay  = ct::exp(gc_r - gc_c);

    auto offs      = ct::iota<i32_CS>();
    auto offs_r    = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(offs);
    auto offs_c    = ct::reshape<ct::shape<1, CHUNK_SIZE>>(offs);
    auto strict_lo = offs_r > offs_c;                                 // (CS, CS)
    auto attn      = ct::select(strict_lo, -(base_attn * decay),
                                   ct::zeros<f32_CSxCS>());

    // Solve (I + attn)^{-1} - I, then add I at the end.
    attn = cgdr_solve_tril<CHUNK_SIZE, f32_CSxCS>(attn);

    // --- Store Q_out (with optional l2norm + scale), K_out ---
    auto q_4d = pQ.load(b, pid_chunk, h, 0);
    auto q    = ct::element_cast<float>(ct::reshape<ct::shape<CHUNK_SIZE, BLOCK_K>>(q_4d));
    if constexpr (USE_QK_L2NORM) {
        auto q_sq       = q * q;
        auto q_norm2_2d = ct::sum<1>(q_sq);                                  // (CS, 1)
        auto q_norm2    = ct::reshape<ct::shape<CHUNK_SIZE>>(q_norm2_2d);    // (CS,)
        auto inv        = ct::rsqrt(q_norm2 + ct::full<f32_CS>(1e-6f));
        auto inv_col    = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(inv);
        q = q * inv_col;
    }
    q = q * ct::full<f32_CSxK>(scale);

    pQout.store(ct::reshape<ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>>(q),
                b, h, pid_chunk, 0, 0);
    pKout.store(ct::reshape<ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>>(k),
                b, h, pid_chunk, 0, 0);

    // --- Store g_cum ---
    pGcum.store(ct::reshape<ct::shape<1, 1, 1, CHUNK_SIZE>>(g_cum),
                 b, h, pid_chunk, 0);

    // --- k_cumdecay = attn @ (k_beta * exp(g_cum[:, None])) ---
    auto eg       = ct::exp(g_cum);
    auto eg_col   = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(eg);
    auto kbe      = k_beta * eg_col;
    auto kcd      = ct::matmul(attn, kbe);
    pKcd.store(ct::reshape<ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>>(kcd),
               b, h, pid_chunk, 0, 0);

    // --- v_corrected = attn @ (v * beta[:, None]) : iterate V tiles ---
    int num_v_tiles = (V_dim + BLOCK_K - 1) / BLOCK_K;
    for (auto vt : ct::irange(0, num_v_tiles)) {
        auto v_4d = pV.load(b, pid_chunk, h, vt);
        auto v    = ct::element_cast<float>(ct::reshape<ct::shape<CHUNK_SIZE, BLOCK_K>>(v_4d));
        auto vb   = v * beta_col;
        auto vc   = ct::matmul(attn, vb);
        pVcorr.store(ct::reshape<ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>>(vc),
                     b, h, pid_chunk, 0, vt);
    }
}


// ============================================================================
// Inter-chunk recurrence kernel.
// Grid: (B * NUM_HEADS, cdiv(V, BLOCK_V), 1).
// Each program owns a (BLOCK_K, BLOCK_V) tile of running state and sweeps all
// chunks sequentially.
// ============================================================================
template<typename T,
         int CHUNK_SIZE,
         int BLOCK_K,
         int BLOCK_V,
         bool HAS_INITIAL_STATE,
         bool OUTPUT_FINAL_STATE,
         int occupancy>
[[ using cutile : hint(1000, occupancy=occupancy) ]]
__tile_global__ void chunk_gated_delta_rule_inter_kernel(
    const float* __restrict__ Q_ch,        // (B, H, num_chunks, CS, K)
    const float* __restrict__ K_ch,        // (B, H, num_chunks, CS, K)
    const float* __restrict__ V_corr,      // (B, H, num_chunks, CS, V)
    const float* __restrict__ K_cumdecay,  // (B, H, num_chunks, CS, K)
    const float* __restrict__ G_cum_in,    // (B, H, num_chunks, CS)
    float* __restrict__ Output,            // (B, H, num_chunks, CS, V)
    const float* __restrict__ InitState,   // (B, H, K, V) or nullptr
    float* __restrict__ FinalState,        // (B, H, K, V) or nullptr
    int B,
    int num_chunks,
    int NUM_HEADS,
    int K_dim,
    int V_dim
) {
    namespace ct = cuda::tiles;

    Q_ch       = ct::assume_aligned<16>(Q_ch);
    K_ch       = ct::assume_aligned<16>(K_ch);
    V_corr     = ct::assume_aligned<16>(V_corr);
    K_cumdecay = ct::assume_aligned<16>(K_cumdecay);
    G_cum_in   = ct::assume_aligned<16>(G_cum_in);
    Output     = ct::assume_aligned<16>(Output);
    InitState  = ct::assume_aligned<16>(InitState);
    FinalState = ct::assume_aligned<16>(FinalState);

    int pid_bh = ct::bid().x;
    int pid_v  = ct::bid().y;
    int b = pid_bh / NUM_HEADS;
    int h = pid_bh % NUM_HEADS;

    using f32_KxV   = ct::tile<float, ct::shape<BLOCK_K, BLOCK_V>>;
    using f32_CSxK  = ct::tile<float, ct::shape<CHUNK_SIZE, BLOCK_K>>;
    using f32_CSxV  = ct::tile<float, ct::shape<CHUNK_SIZE, BLOCK_V>>;
    using f32_CSxCS = ct::tile<float, ct::shape<CHUNK_SIZE, CHUNK_SIZE>>;
    using f32_CS    = ct::tile<float, ct::shape<CHUNK_SIZE>>;
    using f32_CSx1  = ct::tile<float, ct::shape<CHUNK_SIZE, 1>>;
    using f32_1xCS  = ct::tile<float, ct::shape<1, CHUNK_SIZE>>;
    using i32_CS    = ct::tile<int,   ct::shape<CHUNK_SIZE>>;

    auto pQch = ct::partition_view(
        ct::tensor_span{Q_ch, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, K_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>{});
    auto pKch = ct::partition_view(
        ct::tensor_span{K_ch, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, K_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>{});
    auto pVcorr = ct::partition_view(
        ct::tensor_span{V_corr, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, V_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_V>{});
    auto pKcd = ct::partition_view(
        ct::tensor_span{K_cumdecay, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, K_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_K>{});
    auto pGcum = ct::partition_view(
        ct::tensor_span{G_cum_in, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE}},
        ct::shape<1, 1, 1, CHUNK_SIZE>{});
    auto pOut = ct::partition_view(
        ct::tensor_span{Output, ct::extents{B, NUM_HEADS, num_chunks, CHUNK_SIZE, V_dim}},
        ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_V>{});

    // Initialize state.
    auto state = ct::zeros<f32_KxV>();
    if constexpr (HAS_INITIAL_STATE) {
        auto pInit = ct::partition_view(
            ct::tensor_span{InitState, ct::extents{B, NUM_HEADS, K_dim, V_dim}},
            ct::shape<1, 1, BLOCK_K, BLOCK_V>{});
        auto init_4d = pInit.load(b, h, 0, pid_v);
        state = ct::reshape<ct::shape<BLOCK_K, BLOCK_V>>(init_4d);
    }

    // Causal mask (lower triangular, includes diagonal).
    auto offs    = ct::iota<i32_CS>();
    auto offs_r  = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(offs);
    auto offs_c  = ct::reshape<ct::shape<1, CHUNK_SIZE>>(offs);
    auto causal  = offs_r >= offs_c;

    for (auto ci : ct::irange(0, num_chunks)) {
        // v' = k_cumdecay @ state  (CS, V)
        auto kcd_5d = pKcd.load(b, h, ci, 0, 0);
        auto kcd    = ct::reshape<ct::shape<CHUNK_SIZE, BLOCK_K>>(kcd_5d);
        auto vprime = ct::matmul(kcd, state);

        auto vcorr_5d = pVcorr.load(b, h, ci, 0, pid_v);
        auto vcorr    = ct::reshape<ct::shape<CHUNK_SIZE, BLOCK_V>>(vcorr_5d);
        auto vnew     = vcorr - vprime;

        auto q_5d = pQch.load(b, h, ci, 0, 0);
        auto q    = ct::reshape<ct::shape<CHUNK_SIZE, BLOCK_K>>(q_5d);

        auto g_4d = pGcum.load(b, h, ci, 0);
        auto g    = ct::reshape<ct::shape<CHUNK_SIZE>>(g_4d);

        // q_weighted = q * exp(g_cum)[:, None], attn_inter = q_weighted @ state.
        auto eg     = ct::exp(g);
        auto eg_col = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(eg);
        auto qw     = q * eg_col;
        auto inter  = ct::matmul(qw, state);

        // intra-chunk: QK^T with causal + decay mask.
        auto k_5d = pKch.load(b, h, ci, 0, 0);
        auto k_tile = ct::reshape<ct::shape<CHUNK_SIZE, BLOCK_K>>(k_5d);
        auto k_T    = ct::transpose(k_tile);
        auto qk     = ct::matmul(q, k_T);

        auto gc_r   = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(g);
        auto gc_c   = ct::reshape<ct::shape<1, CHUNK_SIZE>>(g);
        auto decay  = ct::exp(gc_r - gc_c);
        auto qk_d   = qk * decay;
        auto qk_m   = ct::select(causal, qk_d, ct::zeros<f32_CSxCS>());

        auto intra  = ct::matmul(qk_m, vnew);
        auto out    = inter + intra;

        pOut.store(ct::reshape<ct::shape<1, 1, 1, CHUNK_SIZE, BLOCK_V>>(out),
                    b, h, ci, 0, pid_v);

        auto is_last   = offs == ct::full<i32_CS>(CHUNK_SIZE - 1);
        auto masked_g  = ct::select(is_last, g, ct::zeros<f32_CS>());
        float g_last   = static_cast<float>(ct::sum<0>(masked_g));

        auto kw_col    = ct::reshape<ct::shape<CHUNK_SIZE, 1>>(
            ct::exp(ct::full<f32_CS>(g_last) - g));
        auto k_weighted = k_tile * kw_col;
        auto kw_T       = ct::transpose(k_weighted);

        state = state * ct::exp(g_last) + ct::matmul(kw_T, vnew);
    }

    if constexpr (OUTPUT_FINAL_STATE) {
        auto pFinal = ct::partition_view(
            ct::tensor_span{FinalState, ct::extents{B, NUM_HEADS, K_dim, V_dim}},
            ct::shape<1, 1, BLOCK_K, BLOCK_V>{});
        pFinal.store(ct::reshape<ct::shape<1, 1, BLOCK_K, BLOCK_V>>(state),
                      b, h, 0, pid_v);
    }
}
