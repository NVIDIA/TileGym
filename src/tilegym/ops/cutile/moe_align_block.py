# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from typing import Tuple

import cuda.tile as ct
import torch

from tilegym.backend import register_impl


def ceil_div(a, b):
    return (a + b - 1) // b


@ct.kernel
def moe_align_block_size_stage1(
    topk_ids,
    tokens_cnts,
    num_experts: ct.Constant[int],
    numel: ct.Constant[int],
    tokens_per_thread: ct.Constant[int],
):
    bid = ct.bid(0)

    start_idx = bid * tokens_per_thread
    off_c = (bid + 1) * num_experts
    current_idx = 0
    token_cnt = ct.zeros((1,), dtype=ct.int32)

    # Convert loop to grid-based processing
    # Each thread processes tokens_per_thread items
    limit = max(0, min(tokens_per_thread, numel - start_idx))
    for i in range(limit):
        current_idx = start_idx + i
        # Convert current_idx to tile for gather
        current_idx_tile = ct.full((1,), current_idx, dtype=ct.int32)
        # Load expert index for current token
        idx = ct.gather(topk_ids, current_idx_tile, padding_value=0)

        # Load current count for this expert - ensure Tile types
        off_c_tile = ct.full((1,), off_c, dtype=ct.int32)
        cnt_offset = ct.add(off_c_tile, idx)
        token_cnt = ct.gather(tokens_cnts, cnt_offset, padding_value=0)

        # Increment and store back
        new_cnt = ct.add(token_cnt, ct.ones((1,), dtype=ct.int32))
        ct.scatter(tokens_cnts, cnt_offset, new_cnt)


@ct.kernel
def moe_align_block_size_stage2(
    tokens_cnts,
    num_experts: ct.Constant[int],
):
    bid = ct.bid(0)
    last_cnt = 0

    # Load all values at once
    base_offset = num_experts + bid
    offsets = ct.arange(num_experts, dtype=ct.int32) * num_experts + base_offset

    token_cnts_vec = ct.gather(tokens_cnts, offsets, padding_value=0)

    cumsum = ct.cumsum(token_cnts_vec, axis=0)

    ct.scatter(tokens_cnts, offsets, cumsum)


@ct.kernel
def moe_align_block_size_stage3(
    total_tokens_post_pad,
    tokens_cnts,
    cumsum,
    num_experts: ct.Constant[int],
    block_size: ct.Constant[int],
):
    last_cumsum = ct.zeros((1,), dtype=ct.int32)
    off_cnt = num_experts * num_experts
    token_cnt = ct.zeros((1,), dtype=ct.int32)
    padded_cnt = ct.zeros((1,), dtype=ct.int32)

    # Convert loop to sequential processing
    for i in range(1, num_experts + 1):
        cnt_offset = off_cnt + i - 1 + ct.arange(1, dtype=ct.int32)
        token_cnt = ct.gather(tokens_cnts, cnt_offset, padding_value=0)

        block_size_tile = ct.full((1,), block_size, dtype=token_cnt.dtype)
        div_result = ct.add(
            token_cnt,
            ct.sub(block_size_tile, ct.ones((1,), dtype=token_cnt.dtype)),
        )
        ceiled_div = ct.floordiv(div_result, block_size_tile)
        padded_cnt = ct.mul(ceiled_div, block_size_tile)
        last_cumsum = ct.add(last_cumsum, padded_cnt)

        cumsum_offset = ct.full((1,), i, dtype=ct.int32)
        ct.scatter(cumsum, cumsum_offset, last_cumsum)

    zero_offset = ct.zeros((1,), dtype=ct.int32)
    ct.scatter(total_tokens_post_pad, zero_offset, last_cumsum)


@ct.kernel
def moe_align_block_size_stage4(
    topk_ids,
    sorted_token_ids,
    expert_ids,
    tokens_cnts,
    cumsum,
    num_experts: ct.Constant[int],
    block_size: ct.Constant[int],
    numel: ct.Constant[int],
    tokens_per_thread: ct.Constant[int],
):
    bid = ct.bid(0)

    # Define essential variables upfront
    off_t = bid * num_experts

    # Load start and end indices from cumsum
    start_idx_cumsum = ct.gather(cumsum, bid, padding_value=0)
    end_idx_cumsum = ct.gather(cumsum, bid + 1, padding_value=0)

    # First loop: fill expert_ids array
    # Compute exact block range and iterate without an inner condition
    start_block = start_idx_cumsum // block_size
    end_block = (end_idx_cumsum + block_size - 1) // block_size
    num_blocks = max(0, end_block - start_block)
    for i in range(num_blocks):
        block_idx = start_block + i
        ct.scatter(expert_ids, block_idx, bid)

    # Second loop: process tokens
    start_idx_tokens = bid * tokens_per_thread

    limit = max(0, min(tokens_per_thread, numel - start_idx_tokens))
    for i in range(limit):
        current_idx = start_idx_tokens + i
        # Convert current_idx to tile for gather
        current_idx_tile = ct.full((1,), current_idx, dtype=ct.int32)
        # Load expert_id for current token
        expert_id = ct.gather(topk_ids, current_idx_tile, padding_value=0)

        # Load token count
        off_t_tile = ct.full((1,), off_t, dtype=ct.int32)
        cnt_offset = ct.add(off_t_tile, expert_id)
        token_cnt = ct.gather(tokens_cnts, cnt_offset, padding_value=0)

        # Load cumsum value
        cumsum_val = ct.gather(cumsum, expert_id, padding_value=0)

        # Calculate rank_post_pad
        rank_post_pad = ct.add(token_cnt, cumsum_val)

        # Store sorted token id and token count (reuse current_idx_tile from above)
        ct.scatter(sorted_token_ids, rank_post_pad, current_idx_tile)
        new_token_cnt = ct.add(token_cnt, ct.ones((1,), dtype=ct.int32))
        ct.scatter(tokens_cnts, cnt_offset, new_token_cnt)


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    # Flatten topk_ids and tokens_cnts to 1D for gather/scatter operations
    topk_ids_flat = topk_ids.reshape(-1)

    numel = topk_ids.numel()
    grid = (num_experts,)
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    tokens_cnts_flat = tokens_cnts.reshape(-1)

    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    # Launch stage 1
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        moe_align_block_size_stage1,
        (topk_ids_flat, tokens_cnts_flat, num_experts, numel, tokens_per_thread),
    )

    # Launch stage 2
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        moe_align_block_size_stage2,
        (tokens_cnts_flat, num_experts),
    )

    # Launch stage 3
    ct.launch(
        torch.cuda.current_stream(),
        (1,),
        moe_align_block_size_stage3,
        (num_tokens_post_pad, tokens_cnts_flat, cumsum, num_experts, block_size),
    )

    # Launch stage 4
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        moe_align_block_size_stage4,
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
        ),
    )
    return cumsum


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = math.ceil(max_num_tokens_padded / block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    cumsum = _moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad, cumsum


register_impl("moe_align_block_size", "cutile")(moe_align_block_size)
