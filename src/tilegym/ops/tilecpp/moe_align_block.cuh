// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: MIT

/**
 * Standalone Tile C++ MoE Align Block Size Kernel
 * Aligns token distribution across experts for block matrix multiplication.
 *
 *
 * This kernel has 4 stages:
 * 1. Count tokens per expert (scalar operations with data-dependent indexing)
 * 2. Compute cumulative sum of token counts (tile operations with inclusive_scan)
 * 3. Compute padded cumsum for block alignment (single program, sequential)
 * 4. Assign tokens to sorted positions (scalar operations with data-dependent indexing)
 */

#pragma once

#include <cuda_tile.h>

/**
 * Stage 1: Count tokens per expert.
 * Each program counts tokens for a subset of the input.
 *
 * This stage uses scalar operations because expert_id (loaded from topk_ids)
 * determines the write location - a scatter pattern.
 */
// Constants (num_experts) as NTTPs; numel and tokens_per_thread are runtime parameters.
template<typename T, int BLOCK_SIZE, int NUM_EXPERTS, int SUB_CHUNK, int EXPERTS_POW2, int STATIC_NUMEL, int STATIC_TOKENS_PER_THREAD>
__tile_global__ void moe_align_block_size_stage1(
    const T* __restrict__ topk_ids,
    int* __restrict__ tokens_cnts,
    int NUMEL,
    int TOKENS_PER_THREAD
) {
    namespace ct = cuda::tiles;
    NUMEL = STATIC_NUMEL;
    TOKENS_PER_THREAD = STATIC_TOKENS_PER_THREAD;

    topk_ids = ct::assume_aligned<16>(topk_ids);
    tokens_cnts = ct::assume_aligned<16>(tokens_cnts);

    int pid = ct::bid().x;
    int start_idx = pid * TOKENS_PER_THREAD;
    int off_c = (pid + 1) * NUM_EXPERTS;

    using Experts = ct::tile<int, ct::shape<EXPERTS_POW2>>;
    using Tokens = ct::tile<int, ct::shape<SUB_CHUNK>>;
    auto experts = ct::iota<Experts>();
    auto offsets = ct::iota<Tokens>();
    auto counts = ct::zeros<Experts>();
    for (auto sub : ct::irange(0, TOKENS_PER_THREAD, SUB_CHUNK)) {
        auto positions = start_idx + sub + offsets;
        auto valid = (positions < NUMEL) & (sub + offsets < TOKENS_PER_THREAD);
        auto ids = ct::element_cast<int>(ct::load_masked(topk_ids + positions, valid, T(NUM_EXPERTS)));
        auto matches = ct::element_cast<int>(
            ct::reshape(ids, ct::shape<SUB_CHUNK, 1>{}) ==
            ct::reshape(experts, ct::shape<1, EXPERTS_POW2>{}));
        counts = counts + ct::reshape(ct::sum(matches, ct::integral_constant<0>{}),
                                      ct::shape<EXPERTS_POW2>{});
    }
    ct::store_masked(tokens_cnts + off_c + experts, counts, experts < NUM_EXPERTS);

}

/**
 * Stage 2: Compute cumulative sum of token counts across programs.
 * Each program computes cumsum for one expert column.
 */
template<typename T, int NUM_EXPERTS, int NUM_PROGRAMS, int PADDED_PROGRAMS>
__tile_global__ void moe_align_block_size_stage2(
    int* __restrict__ tokens_cnts
) {
    namespace ct = cuda::tiles;

    // PADDED_PROGRAMS must be next power of 2 >= NUM_PROGRAMS (enforced by Python)
    using i32xP = ct::tile<int32_t, ct::shape<PADDED_PROGRAMS>>;

    tokens_cnts = ct::assume_aligned<16>(tokens_cnts);

    int pid = ct::bid().x;

    // base_offset = num_experts + pid
    int base_offset = NUM_EXPERTS + pid;

    auto offsets = ct::iota<i32xP>() * NUM_EXPERTS + base_offset;

    // Mask: only first NUM_PROGRAMS rows are valid
    auto mask = ct::iota<i32xP>() < NUM_PROGRAMS;

    // Gather-load with mask (zero-fill padding positions)
    auto token_cnts_vec = ct::load_masked(tokens_cnts + offsets, mask,
        ct::zeros<ct::tile<int32_t, ct::shape<1>>>());

    // cumsum on padded tile (power-of-2 size)
    auto cumsum_result = ct::partial_sum(token_cnts_vec, ct::integral_constant<0>{});

    // Store back only the first NUM_EXPERTS elements
    ct::store_masked(tokens_cnts + offsets, cumsum_result, mask);
}

/**
 * Stage 3: Compute padded cumsum for block alignment.
 * Single program - computes cumulative padded counts and max expert count.
 */
template<typename T, int NUM_EXPERTS, int BLOCK_SIZE, int NUM_PROGRAMS>
__tile_global__ void moe_align_block_size_stage3(
    int* __restrict__ total_tokens_post_pad,
    int* __restrict__ max_expert_cnt,
    const int* __restrict__ tokens_cnts,
    int* __restrict__ cumsum
) {
    namespace ct = cuda::tiles;

    using i32x1 = ct::tile<int32_t, ct::shape<1>>;

    total_tokens_post_pad = ct::assume_aligned<16>(total_tokens_post_pad);
    max_expert_cnt = ct::assume_aligned<16>(max_expert_cnt);
    tokens_cnts = ct::assume_aligned<16>(tokens_cnts);
    cumsum = ct::assume_aligned<16>(cumsum);

    auto last_cumsum = ct::zeros<i32x1>();
    int off_cnt = NUM_PROGRAMS * NUM_EXPERTS;
    auto token_cnt = ct::zeros<i32x1>();
    auto padded_cnt = ct::zeros<i32x1>();
    auto max_cnt = ct::zeros<i32x1>();

    for (auto i : ct::irange(1, NUM_EXPERTS + 1)) {
        auto cnt_offset = ct::full<i32x1>(off_cnt + i - 1) + ct::iota<i32x1>();

        token_cnt = ct::load(tokens_cnts + cnt_offset);

        max_cnt = ct::max(max_cnt, token_cnt);

        // padded_cnt = ceil(token_cnt / block_size) * block_size
        auto block_size_tile = ct::full<i32x1>(BLOCK_SIZE);
        auto ones_tile = ct::ones<i32x1>();
        auto div_result = (token_cnt + block_size_tile - ones_tile) / block_size_tile;
        padded_cnt = div_result * block_size_tile;

        last_cumsum = last_cumsum + padded_cnt;

        auto cumsum_offset = ct::full<i32x1>(i);
        ct::store(cumsum + cumsum_offset, last_cumsum);
    }

    auto zero_offset = ct::zeros<i32x1>();
    ct::store(total_tokens_post_pad + zero_offset, last_cumsum);

    ct::store(max_expert_cnt + zero_offset, max_cnt);
}

/**
 * Stage 4: Assign tokens to sorted positions.
 * Each program handles expert_id assignment and token placement.
 *
 * Uses scalar operations due to data-dependent indexing.
 */
// Constants (num_experts, block_size) as NTTPs; numel and tokens_per_thread are runtime parameters.
template<typename T, int NUM_EXPERTS, int BLOCK_SIZE, int SUB_CHUNK, int EXPERTS_POW2, int STATIC_NUMEL, int STATIC_TOKENS_PER_THREAD>
__tile_global__ void moe_align_block_size_stage4(
    const T* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ tokens_cnts,
    const int* __restrict__ cumsum,
    int NUMEL,
    int TOKENS_PER_THREAD
) {
    namespace ct = cuda::tiles;
    NUMEL = STATIC_NUMEL;
    TOKENS_PER_THREAD = STATIC_TOKENS_PER_THREAD;

    using i32x1 = ct::tile<int32_t, ct::shape<1>>;

    topk_ids = ct::assume_aligned<16>(topk_ids);
    sorted_token_ids = ct::assume_aligned<16>(sorted_token_ids);
    expert_ids = ct::assume_aligned<16>(expert_ids);
    tokens_cnts = ct::assume_aligned<16>(tokens_cnts);
    cumsum = ct::assume_aligned<16>(cumsum);

    int bid = ct::bid().x;

    int off_t = bid * NUM_EXPERTS;

    auto cumsum_limit = ct::full<i32x1>(NUM_EXPERTS + 1);
    auto start_idx_off = ct::full<i32x1>(bid);
    auto end_idx_off = ct::full<i32x1>(bid + 1);
    auto zero_cnt = ct::zeros<i32x1>();
    auto start_idx_cumsum_tile =
        ct::load_masked(cumsum + start_idx_off, start_idx_off < cumsum_limit, zero_cnt);
    auto end_idx_cumsum_tile =
        ct::load_masked(cumsum + end_idx_off, end_idx_off < cumsum_limit, zero_cnt);
    int start_idx_cumsum = static_cast<int>(start_idx_cumsum_tile);
    int end_idx_cumsum = static_cast<int>(end_idx_cumsum_tile);

    int start_block = start_idx_cumsum / BLOCK_SIZE;
    int end_block = (end_idx_cumsum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks = ct::max(0, end_block - start_block);
    auto bid_tile = ct::full<i32x1>(bid);
    for (auto i : ct::irange(0, num_blocks)) {
        int block_idx = start_block + i;
        auto block_idx_tile = ct::full<i32x1>(block_idx);
        ct::store(expert_ids + block_idx_tile, bid_tile);
    }

    using Experts = ct::tile<int, ct::shape<EXPERTS_POW2>>;
    using Tokens = ct::tile<int, ct::shape<SUB_CHUNK>>;
    auto experts = ct::iota<Experts>();
    auto expert_valid = experts < NUM_EXPERTS;
    auto before = ct::load_masked(tokens_cnts + off_t + experts, expert_valid, 0);
    auto bases = ct::load_masked(cumsum + experts, expert_valid, 0);
    auto running = before + bases;
    auto offsets = ct::iota<Tokens>();
    int start_idx_tokens = bid * TOKENS_PER_THREAD;
    for (auto sub : ct::irange(0, TOKENS_PER_THREAD, SUB_CHUNK)) {
        auto positions = start_idx_tokens + sub + offsets;
        auto valid = (positions < NUMEL) & (sub + offsets < TOKENS_PER_THREAD);
        auto ids = ct::element_cast<int>(ct::load_masked(topk_ids + positions, valid, T(NUM_EXPERTS)));
        auto matches = ct::element_cast<int>(
            ct::reshape(ids, ct::shape<SUB_CHUNK, 1>{}) ==
            ct::reshape(experts, ct::shape<1, EXPERTS_POW2>{}));
        auto ranks = ct::partial_sum(matches, ct::integral_constant<0>{}) - matches;
        auto per_token = ct::reshape(ct::sum(ranks * matches, ct::integral_constant<1>{}),
                                     ct::shape<SUB_CHUNK>{});
        auto base = ct::reshape(ct::sum(
            ct::reshape(running, ct::shape<1, EXPERTS_POW2>{}) * matches,
            ct::integral_constant<1>{}), ct::shape<SUB_CHUNK>{});
        ct::store_masked(sorted_token_ids + base + per_token, positions, valid);
        running = running + ct::reshape(ct::sum(matches, ct::integral_constant<0>{}),
                                        ct::shape<EXPERTS_POW2>{});
    }

}
