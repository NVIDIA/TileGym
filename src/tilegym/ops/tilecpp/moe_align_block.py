# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ MoE Align Block Size Operation
Aligns token distribution across experts for block matrix multiplication
using CUDA C++ tile kernels compiled with nvcc.

"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


# =============================================================================
# Kernel Definitions
# =============================================================================


_stage1_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "moe_align_block.cuh",
    kernel_name="moe_align_block_size_stage1",
)

_stage2_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "moe_align_block.cuh",
    kernel_name="moe_align_block_size_stage2",
)

_stage3_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "moe_align_block.cuh",
    kernel_name="moe_align_block_size_stage3",
)

_stage4_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "moe_align_block.cuh",
    kernel_name="moe_align_block_size_stage4",
)


# =============================================================================
# Kernel Launchers
# =============================================================================


def _launch_stage1(
    topk_ids: torch.Tensor,
    tokens_cnts: torch.Tensor,
    num_experts: int,
    numel: int,
    tokens_per_thread: int,
    grid: int,
):
    """Launch stage 1 kernel."""
    dump_kernel_types("moe_align_block_size_stage1", topk_ids, tokens_cnts)

    # Template params: T, BLOCK_SIZE (dummy), NUM_EXPERTS
    # NUMEL and TOKENS_PER_THREAD are runtime parameters (variable per batch)
    kernel, _, _ = _stage1_kernel.get_kernel(
        dtype=torch.int32,
        template_params=[1, num_experts],
        signature="const int*, int*, int, int",
    )

    _stage1_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(topk_ids.data_ptr()),
            np.uint64(tokens_cnts.data_ptr()),
            np.int32(numel),
            np.int32(tokens_per_thread),
        ],
    )


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    v = 1
    while v < n:
        v <<= 1
    return v


def _launch_stage2(
    tokens_cnts: torch.Tensor,
    num_experts: int,
    grid: int,
):
    """Launch stage 2 kernel."""
    dump_kernel_types("moe_align_block_size_stage2", tokens_cnts)

    padded_experts = _next_power_of_2(num_experts)

    # Template params: T (int), NUM_EXPERTS, PADDED_EXPERTS (next power of 2)
    kernel, _, _ = _stage2_kernel.get_kernel(
        dtype=torch.int32,
        template_params=[num_experts, padded_experts],
        signature="int*",
    )

    _stage2_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(tokens_cnts.data_ptr()),
        ],
    )


def _launch_stage3(
    total_tokens_post_pad: torch.Tensor,
    max_expert_cnt: torch.Tensor,
    tokens_cnts: torch.Tensor,
    cumsum: torch.Tensor,
    num_experts: int,
    block_size: int,
):
    dump_kernel_types("moe_align_block_size_stage3", tokens_cnts, cumsum)

    kernel, _, _ = _stage3_kernel.get_kernel(
        dtype=torch.int32,
        template_params=[num_experts, block_size],
        signature="int*, int*, const int*, int*",
    )

    _stage3_kernel.launch(
        grid=1,
        kernel=kernel,
        args=[
            np.uint64(total_tokens_post_pad.data_ptr()),
            np.uint64(max_expert_cnt.data_ptr()),
            np.uint64(tokens_cnts.data_ptr()),
            np.uint64(cumsum.data_ptr()),
        ],
    )


def _launch_stage4(
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    tokens_cnts: torch.Tensor,
    cumsum: torch.Tensor,
    num_experts: int,
    block_size: int,
    numel: int,
    tokens_per_thread: int,
    grid: int,
):
    """Launch stage 4 kernel."""
    dump_kernel_types("moe_align_block_size_stage4", topk_ids, sorted_token_ids, expert_ids)

    # Template params: T, NUM_EXPERTS, BLOCK_SIZE
    # NUMEL and TOKENS_PER_THREAD are runtime parameters (variable per batch)
    kernel, _, _ = _stage4_kernel.get_kernel(
        dtype=torch.int32,
        template_params=[num_experts, block_size],
        signature="const int*, int*, int*, int*, const int*, int, int",
    )

    _stage4_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(topk_ids.data_ptr()),
            np.uint64(sorted_token_ids.data_ptr()),
            np.uint64(expert_ids.data_ptr()),
            np.uint64(tokens_cnts.data_ptr()),
            np.uint64(cumsum.data_ptr()),
            np.int32(numel),
            np.int32(tokens_per_thread),
        ],
    )


# =============================================================================
# Internal Function
# =============================================================================


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    max_expert_cnt: torch.Tensor,
) -> torch.Tensor:
    """
    Internal implementation of MoE align block size.

    Returns:
        cumsum: Cumulative sum tensor for block alignment.
    """
    numel = topk_ids.numel()
    grid = num_experts
    tokens_cnts = torch.zeros(
        (num_experts + 1, num_experts),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=topk_ids.device)
    tokens_per_thread = ceil_div(numel, num_experts)

    _launch_stage1(
        topk_ids,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
        grid,
    )

    _launch_stage2(
        tokens_cnts,
        num_experts,
        grid,
    )

    _launch_stage3(
        num_tokens_post_pad,
        max_expert_cnt,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
    )

    # Launch stage 4: Assign tokens to sorted positions
    _launch_stage4(
        topk_ids,
        sorted_token_ids,
        expert_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        block_size,
        numel,
        tokens_per_thread,
        grid,
    )

    return cumsum


# =============================================================================
# Public Function
# =============================================================================


@register_impl("moe_align_block_size", "tilecpp")
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block

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
    - cumsum: Cumulative sum tensor for block alignment.
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
    """
    # Ensure topk_ids is int32 (kernel expects const int*)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    sorted_ids.fill_(topk_ids.numel())
    max_num_m_blocks = ceil_div(max_num_tokens_padded, block_size)
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
