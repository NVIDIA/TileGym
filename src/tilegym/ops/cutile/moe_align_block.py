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
    token_counts,
    NUM_EXPERTS: ct.Constant[int],
    NUMEL: ct.Constant[int],
):
    """Stage 1: Count tokens per expert using parallel atomic increments.
    Grid: (NUMEL,) — one CTA per token for maximum parallelism.
    Replaces the serial gather-modify-scatter loop (was O(NUMEL/E) per CTA).
    """
    bid = ct.bid(0)  # token index in [0, NUMEL)
    bid_tile = ct.full((1,), bid, dtype=ct.int32)
    expert = ct.gather(topk_ids, bid_tile, padding_value=0)
    ct.atomic_add(token_counts, expert, ct.ones((1,), dtype=ct.int32))


@ct.kernel
def _moe_align_block_size_stage2_kernel(
    total_tokens_post_pad,
    max_expert_cnt,
    token_counts,
    cumsum,
    NUM_EXPERTS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    """Stage 2: Compute padded cumulative sums and metadata.
    Grid: (1,) — single CTA, O(NUM_EXPERTS) work (small).
    Combines old stage2 (histogram prefix sum) + stage3 (cumsum/total/max).
    """
    last_cumsum = ct.zeros((1,), dtype=ct.int32)
    max_cnt = ct.zeros((1,), dtype=ct.int32)

    for i in range(NUM_EXPERTS):
        i_tile = ct.full((1,), i, dtype=ct.int32)
        token_cnt = ct.gather(token_counts, i_tile, padding_value=0)
        max_cnt = ct.maximum(max_cnt, token_cnt)

        block_size_tile = ct.full((1,), BLOCK_SIZE, dtype=ct.int32)
        div_result = token_cnt + (block_size_tile - ct.ones((1,), dtype=token_cnt.dtype))
        ceiled_div = div_result // block_size_tile
        padded_cnt = ceiled_div * block_size_tile
        last_cumsum = last_cumsum + padded_cnt

        cumsum_offset = ct.full((1,), i + 1, dtype=ct.int32)
        ct.scatter(cumsum, cumsum_offset, last_cumsum)

    zero_offset = ct.zeros((1,), dtype=ct.int32)
    ct.scatter(total_tokens_post_pad, zero_offset, last_cumsum)
    ct.scatter(max_expert_cnt, zero_offset, max_cnt)


@ct.kernel
def _moe_align_block_size_stage3_kernel(
    expert_ids,
    cumsum,
    BLOCK_SIZE: ct.Constant[int],
):
    """Stage 3: Fill expert_ids array.
    Grid: (NUM_EXPERTS,) — one CTA per expert.
    """
    bid = ct.bid(0)  # expert index

    start_idx_cumsum = ct.gather(cumsum, bid, padding_value=0)
    end_idx_cumsum = ct.gather(cumsum, bid + 1, padding_value=0)

    start_block = start_idx_cumsum // BLOCK_SIZE
    end_block = (end_idx_cumsum + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks = max(0, end_block - start_block)
    for i in range(num_blocks):
        block_idx = start_block + i
        ct.scatter(expert_ids, block_idx, bid)


@ct.kernel
def _moe_align_block_size_stage4_kernel(
    topk_ids,
    sorted_token_ids,
    cumsum,
    NUMEL: ct.Constant[int],
):
    """Stage 4: Fill sorted_token_ids by scanning tokens sequentially per expert.
    Grid: (NUM_EXPERTS,) — one CTA per expert.
    Scans tokens 0..NUMEL-1 in order, so tokens within each expert block are
    written in ascending token-index order — matching the reference implementation.
    """
    bid = ct.bid(0)  # expert index
    bid_tile = ct.full((1,), bid, dtype=ct.int32)
    base = ct.gather(cumsum, bid_tile, padding_value=0)
    slot = ct.zeros((1,), dtype=ct.int32)

    for i in range(NUMEL):
        i_tile = ct.full((1,), i, dtype=ct.int32)
        token_expert = ct.gather(topk_ids, i_tile, padding_value=-1)
        is_mine = token_expert == bid_tile
        pos = base + slot
        ct.scatter(sorted_token_ids, pos, i_tile, mask=is_mine)
        slot = slot + ct.astype(is_mine, ct.int32)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    max_expert_cnt: torch.Tensor,
) -> torch.Tensor:
    # Flatten topk_ids to 1D for gather/scatter operations
    topk_ids_flat = topk_ids.reshape(-1)

    numel = topk_ids.numel()
    stream = torch.cuda.current_stream()

    # Flat count array: token_counts[e] = number of tokens routed to expert e
    token_counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)

    # Stage 1: Count tokens per expert (fully parallel, O(1) per CTA, grid=NUMEL)
    ct.launch(
        stream,
        (numel,),
        _moe_align_block_size_stage1_kernel,
        (topk_ids_flat, token_counts, num_experts, numel),
    )

    # Stage 2: Compute padded cumsum, total tokens post pad, max expert count
    ct.launch(
        stream,
        (1,),
        _moe_align_block_size_stage2_kernel,
        (num_tokens_post_pad, max_expert_cnt, token_counts, cumsum, num_experts, block_size),
    )

    # Stage 3: Fill expert_ids (one CTA per expert, O(tokens_per_expert/block_size) loop)
    ct.launch(
        stream,
        (num_experts,),
        _moe_align_block_size_stage3_kernel,
        (expert_ids, cumsum, block_size),
    )

    # Stage 4: Fill sorted_token_ids (sequential per expert, O(NUMEL) per CTA, grid=NUM_EXPERTS)
    # Processes tokens 0..NUMEL-1 in order so output is sorted within each expert block.
    ct.launch(
        stream,
        (num_experts,),
        _moe_align_block_size_stage4_kernel,
        (topk_ids_flat, sorted_token_ids, cumsum, numel),
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
