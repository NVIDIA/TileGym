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
template<typename T, int BLOCK_SIZE, int NUM_EXPERTS>
__tile_global__ void moe_align_block_size_stage1(
    const int* __restrict__ topk_ids,
    int* __restrict__ tokens_cnts,
    int NUMEL,
    int TOKENS_PER_THREAD
) {
    namespace ct = cuda::tiles;

    topk_ids = ct::assume_aligned<16>(topk_ids);
    tokens_cnts = ct::assume_aligned<16>(tokens_cnts);

    int pid = ct::bid().x;
    int start_idx = pid * TOKENS_PER_THREAD;
    int off_c = (pid + 1) * NUM_EXPERTS;

    int limit = ct::max(0, ct::min(TOKENS_PER_THREAD, NUMEL - start_idx));

    for (auto i : ct::irange(0, limit)) {
        int current_idx = start_idx + i;
        auto idx_tile = ct::load(topk_ids + current_idx);
        int idx = static_cast<int>(idx_tile);
        auto token_cnt_tile = ct::load(tokens_cnts + (off_c + idx));
        auto new_cnt_tile = token_cnt_tile + 1;
        ct::store(tokens_cnts + (off_c + idx), new_cnt_tile);
    }
}

/**
 * Stage 2: Compute cumulative sum of token counts across programs.
 * Each program computes cumsum for one expert column.
 */
template<typename T, int NUM_EXPERTS, int PADDED_EXPERTS>
__tile_global__ void moe_align_block_size_stage2(
    int* __restrict__ tokens_cnts
) {
    namespace ct = cuda::tiles;

    // PADDED_EXPERTS must be next power of 2 >= NUM_EXPERTS (enforced by Python)
    using i32xP = ct::tile<int32_t, ct::shape<PADDED_EXPERTS>>;

    tokens_cnts = ct::assume_aligned<16>(tokens_cnts);

    int pid = ct::bid().x;

    // base_offset = num_experts + pid
    int base_offset = NUM_EXPERTS + pid;

    auto offsets = ct::iota<i32xP>() * NUM_EXPERTS + base_offset;

    // Mask: only first NUM_EXPERTS elements are valid
    auto mask = ct::iota<i32xP>() < NUM_EXPERTS;

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
template<typename T, int NUM_EXPERTS, int BLOCK_SIZE>
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
    int off_cnt = NUM_EXPERTS * NUM_EXPERTS;
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
template<typename T, int NUM_EXPERTS, int BLOCK_SIZE>
__tile_global__ void moe_align_block_size_stage4(
    const int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ expert_ids,
    int* __restrict__ tokens_cnts,
    const int* __restrict__ cumsum,
    int NUMEL,
    int TOKENS_PER_THREAD
) {
    namespace ct = cuda::tiles;

    using i32x1 = ct::tile<int32_t, ct::shape<1>>;

    topk_ids = ct::assume_aligned<16>(topk_ids);
    sorted_token_ids = ct::assume_aligned<16>(sorted_token_ids);
    expert_ids = ct::assume_aligned<16>(expert_ids);
    tokens_cnts = ct::assume_aligned<16>(tokens_cnts);
    cumsum = ct::assume_aligned<16>(cumsum);

    int bid = ct::bid().x;

    int off_t = bid * NUM_EXPERTS;

    auto start_idx_cumsum_tile = ct::load(cumsum + bid);
    auto end_idx_cumsum_tile = ct::load(cumsum + bid + 1);
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

    int start_idx_tokens = bid * TOKENS_PER_THREAD;

    int limit = ct::max(0, ct::min(TOKENS_PER_THREAD, NUMEL - start_idx_tokens));
    for (auto i : ct::irange(0, limit)) {
        int current_idx = start_idx_tokens + i;

        // Load expert_id for current token
        auto current_idx_tile = ct::full<i32x1>(current_idx);
        auto expert_id_tile = ct::load(topk_ids + current_idx_tile);
        int expert_id = static_cast<int>(expert_id_tile);

        // Load token count
        auto off_t_tile = ct::full<i32x1>(off_t);
        auto cnt_offset_tile = off_t_tile + expert_id_tile;
        auto token_cnt_tile = ct::load(tokens_cnts + cnt_offset_tile);
        int token_cnt = static_cast<int>(token_cnt_tile);

        // Load cumsum value
        auto cumsum_val_tile = ct::load(cumsum + expert_id_tile);
        int cumsum_val = static_cast<int>(cumsum_val_tile);

        int rank_post_pad = token_cnt + cumsum_val;

        // Store token ID at sorted position
        auto rank_post_pad_tile = ct::full<i32x1>(rank_post_pad);
        auto current_idx_store_tile = ct::full<i32x1>(current_idx);
        ct::store(sorted_token_ids + rank_post_pad_tile, current_idx_store_tile);

        // Increment token count for this expert
        auto new_cnt_tile = token_cnt_tile + 1;
        ct::store(tokens_cnts + cnt_offset_tile, new_cnt_tile);
    }
}
