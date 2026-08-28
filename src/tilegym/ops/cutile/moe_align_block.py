# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
from typing import Tuple

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.ops.cutile.utils import next_power_of_2


@ct.kernel
def _moe_align_block_size_stage1_kernel(
    topk_ids,
    tokens_cnts,
    NUM_EXPERTS: ct.Constant[int],
    NUMEL: ct.Constant[int],
    TOKENS_PER_THREAD: ct.Constant[int],
    SUB_CHUNK: ct.Constant[int],
    NUM_EXPERTS_POW2: ct.Constant[int],
):
    bid = ct.bid(0)

    start_idx = bid * TOKENS_PER_THREAD
    off_c = (bid + 1) * NUM_EXPERTS

    # Vectorized histogram: process this program's token chunk in sub-chunks of
    # SUB_CHUNK tokens.
    expert_idx = ct.arange(NUM_EXPERTS_POW2, dtype=ct.int32)
    expert_mask = expert_idx < NUM_EXPERTS
    counts = ct.zeros((NUM_EXPERTS_POW2,), dtype=ct.int32)

    sub_arange = ct.arange(SUB_CHUNK, dtype=ct.int32)
    for sub_start in range(0, TOKENS_PER_THREAD, SUB_CHUNK):
        tok_offs = start_idx + sub_start + sub_arange
        tok_mask = (tok_offs < NUMEL) & ((sub_start + sub_arange) < TOKENS_PER_THREAD)
        ids = ct.gather(topk_ids, tok_offs, mask=tok_mask, padding_value=NUM_EXPERTS)
        match = (ids[:, None] == expert_idx[None, :]).astype(ct.int32)
        counts = counts + ct.sum(match, axis=0)

    ct.scatter(tokens_cnts, off_c + expert_idx, counts, mask=expert_mask)


@ct.kernel
def _moe_align_block_size_stage2_kernel(
    tokens_cnts,
    NUM_EXPERTS: ct.Constant[int],
    NUM_EXPERTS_POW2: ct.Constant[int],
):
    bid = ct.bid(0)

    base_offset = NUM_EXPERTS + bid
    offsets = ct.arange(NUM_EXPERTS_POW2, dtype=ct.int32) * NUM_EXPERTS + base_offset
    token_cnts_vec = ct.gather(tokens_cnts, offsets, padding_value=0)
    cumsum = ct.cumsum(token_cnts_vec, axis=0)
    ct.scatter(tokens_cnts, offsets, cumsum)


@ct.kernel
def _moe_align_block_size_stage3_kernel(
    total_tokens_post_pad,
    max_expert_cnt,
    tokens_cnts,
    cumsum,
    NUM_EXPERTS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    NUM_PROGRAMS: ct.Constant[int],
):
    last_cumsum = ct.zeros((1,), dtype=ct.int32)
    # tokens_cnts has NUM_PROGRAMS + 1 rows; row NUM_PROGRAMS holds the total
    # per-expert count (inclusive prefix over all token programs) after stage2.
    off_cnt = NUM_PROGRAMS * NUM_EXPERTS
    max_cnt = ct.zeros((1,), dtype=ct.int32)

    for i in range(1, NUM_EXPERTS + 1):
        cnt_offset = off_cnt + i - 1 + ct.arange(1, dtype=ct.int32)
        token_cnt = ct.gather(tokens_cnts, cnt_offset, padding_value=0)
        max_cnt = ct.maximum(max_cnt, token_cnt)

        block_size_tile = ct.full((1,), BLOCK_SIZE, dtype=token_cnt.dtype)
        div_result = token_cnt + (block_size_tile - ct.ones((1,), dtype=token_cnt.dtype))
        ceiled_div = div_result // block_size_tile
        padded_cnt = ceiled_div * block_size_tile
        last_cumsum = last_cumsum + padded_cnt

        cumsum_offset = ct.full((1,), i, dtype=ct.int32)
        ct.scatter(cumsum, cumsum_offset, last_cumsum)

    zero_offset = ct.zeros((1,), dtype=ct.int32)
    ct.scatter(total_tokens_post_pad, zero_offset, last_cumsum)
    ct.scatter(max_expert_cnt, zero_offset, max_cnt)


@ct.kernel
def _moe_align_block_size_stage4b_scatter_kernel(
    topk_ids,
    sorted_token_ids,
    expert_ids,
    tokens_cnts,
    cumsum,
    NUM_EXPERTS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    NUMEL: ct.Constant[int],
    TOKENS_PER_THREAD: ct.Constant[int],
    SUB_CHUNK: ct.Constant[int],
    NUM_EXPERTS_POW2: ct.Constant[int],
):
    # One program per token range: scatter its tokens into sorted positions.
    bid = ct.bid(0)
    off_t = bid * NUM_EXPERTS

    # Stamp expert_ids. Programs 0..NUM_EXPERTS-1 own expert `bid`'s blocks;
    # for programs with bid >= NUM_EXPERTS the cumsum gathers fall out of bounds
    # (padding 0), so num_blocks == 0 and nothing is written.
    start_idx_cumsum = ct.gather(cumsum, bid, padding_value=0)
    end_idx_cumsum = ct.gather(cumsum, bid + 1, padding_value=0)
    start_block = start_idx_cumsum // BLOCK_SIZE
    end_block = (end_idx_cumsum + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max(0, end_block - start_block)
    for i in range(num_blocks):
        block_idx = start_block + i
        ct.scatter(expert_ids, block_idx, bid)

    expert_idx = ct.arange(NUM_EXPERTS_POW2, dtype=ct.int32)
    expert_mask = expert_idx < NUM_EXPERTS
    pre_cnt = ct.gather(tokens_cnts, off_t + expert_idx, mask=expert_mask, padding_value=0)
    cum_off = ct.gather(cumsum, expert_idx, mask=expert_mask, padding_value=0)
    running_off = pre_cnt + cum_off  # per-expert global write start for this program

    start_idx_tokens = bid * TOKENS_PER_THREAD
    sub_arange = ct.arange(SUB_CHUNK, dtype=ct.int32)
    for sub_start in range(0, TOKENS_PER_THREAD, SUB_CHUNK):
        tok_offs = start_idx_tokens + sub_start + sub_arange
        in_range = (tok_offs < NUMEL) & ((sub_start + sub_arange) < TOKENS_PER_THREAD)
        ids = ct.gather(topk_ids, tok_offs, mask=in_range, padding_value=NUM_EXPERTS)

        match = (ids[:, None] == expert_idx[None, :]).astype(ct.int32)
        rank_2d = ct.cumsum(match, axis=0) - match  # exclusive prefix per expert column
        rank_per_token = ct.sum(rank_2d * match, axis=1)  # [SUB_CHUNK]
        base_per_token = ct.sum(running_off[None, :] * match, axis=1)  # [SUB_CHUNK]
        rank_post_pad = base_per_token + rank_per_token

        ct.scatter(sorted_token_ids, rank_post_pad, tok_offs, mask=in_range)

        running_off = running_off + ct.sum(match, axis=0)


def _ceil_div(a, b):
    return (a + b - 1) // b


# Cap on the vectorized sub-chunk width (tokens processed per inner iteration).
# Bounds the [SUB_CHUNK, NUM_EXPERTS_POW2] match-matrix working set while still
# vectorizing the histogram (stage1) and scatter (stage4b) hot loops.
_SUB_CHUNK_CAP = 256

# Target tokens processed per program. The launcher derives the number of
# token-processing programs as ceil(numel / this), so large token counts spread
# across more CTAs / SMs instead of being pinned to num_experts programs. 128
# keeps each program to a single vectorized sub-chunk iteration (tokens_per_thread
# <= this <= SUB_CHUNK) while maximizing CTA-level parallelism.
_TARGET_TOKENS_PER_PROGRAM = 128


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    max_expert_cnt: torch.Tensor,
) -> torch.Tensor:
    # Flatten topk_ids and tokens_cnts to 1D for gather/scatter operations
    topk_ids_flat = topk_ids.reshape(-1)

    numel = topk_ids.numel()

    # Number of token-processing programs (stage1 / stage4b). Decoupled from
    # num_experts so large token counts spread across more CTAs (more SMs, fewer
    # serial sub-chunk iterations per program) instead of only num_experts CTAs.
    num_programs = max(num_experts, _ceil_div(numel, _TARGET_TOKENS_PER_PROGRAM))
    tokens_per_thread = _ceil_div(numel, num_programs)

    tokens_cnts = torch.zeros(
        (num_programs + 1, num_experts),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    tokens_cnts_flat = tokens_cnts.reshape(-1)
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)

    num_experts_pow2 = next_power_of_2(num_experts)
    num_programs_pow2 = next_power_of_2(num_programs)
    # Vectorized sub-chunk width: power-of-two <= tokens_per_thread, capped so the
    # match-matrix tile stays bounded for large chunks.
    sub_chunk = min(next_power_of_2(tokens_per_thread), _SUB_CHUNK_CAP)

    # Launch stage 1 (histogram): one program per token range.
    ct.launch(
        torch.cuda.current_stream(),
        (num_programs,),
        _moe_align_block_size_stage1_kernel,
        (topk_ids_flat, tokens_cnts_flat, num_experts, numel, tokens_per_thread, sub_chunk, num_experts_pow2),
    )

    # Launch stage 2 (per-expert prefix over programs): one program per expert.
    ct.launch(
        torch.cuda.current_stream(),
        (num_experts,),
        _moe_align_block_size_stage2_kernel,
        (tokens_cnts_flat, num_experts, num_programs_pow2),
    )

    # Launch stage 3 (per-expert padding + totals): single program.
    ct.launch(
        torch.cuda.current_stream(),
        (1,),
        _moe_align_block_size_stage3_kernel,
        (num_tokens_post_pad, max_expert_cnt, tokens_cnts_flat, cumsum, num_experts, block_size, num_programs),
    )

    # Launch stage 4 (expert_ids stamp + token scatter): one program per token
    # range; programs 0..num_experts-1 additionally stamp their expert's blocks.
    ct.launch(
        torch.cuda.current_stream(),
        (num_programs,),
        _moe_align_block_size_stage4b_scatter_kernel,
        (
            topk_ids_flat,
            sorted_token_ids,
            expert_ids,
            tokens_cnts_flat,
            cumsum,
            num_experts,
            block_size,
            numel,
            tokens_per_thread,
            sub_chunk,
            num_experts_pow2,
        ),
    )

    return cumsum


@register_impl("moe_align_block_size", backend="cutile")
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.
    - cumsum: The exclusive prefix sums of token counts per expert, used to
        compute per-expert write offsets into the sorted token buffer.
    - max_expert_cnt: The maximum token count per expert before padding.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    - With 3 tokens per expert, the padded counts are 4 each, so
        cumsum = [0, 4, 8, 12, 16] and num_tokens_post_padded = 16.
    - max_expert_cnt is 3 since the maximum pre-padding token count is 3.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = _ceil_div(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    max_expert_cnt = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum = _moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        max_expert_cnt,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad, cumsum, max_expert_cnt
