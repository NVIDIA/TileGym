# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT


import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

# Type aliases for constants
ConstInt = ct.Constant[int]


@ct.kernel(occupancy=4)
def softmax_kernel(
    output,
    input,
    n_rows: ConstInt,
    TILE_SIZE: ConstInt,
    DIM_COLS: ConstInt,
):
    # Static persistent scheduling: each block processes multiple rows
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    for row_idx in range(pid, n_rows, num_programs):
        # Load the row tile using index-based access
        row = ct.gather(input, (row_idx, offsets), check_bounds=True, padding_value=-math.inf)
        # Convert to float32 for computation
        row = ct.astype(row, ct.float32)

        # Subtract maximum for numerical stability
        row_max = ct.max(row, 0, keepdims=True)
        row_minus_max = ct.sub(row, row_max)

        # Compute exponential
        numerator = ct.exp(row_minus_max)

        # Compute sum for normalization
        denominator = ct.sum(numerator, 0, keepdims=True)

        # Final softmax computation
        softmax_output = ct.truediv(numerator, denominator)

        # Convert back to original dtype
        softmax_output = ct.astype(softmax_output, input.dtype)

        # Store result using index-based access
        ct.scatter(output, (row_idx, offsets), softmax_output, check_bounds=True)


# TMA version with static persistent scheduling
@ct.kernel(occupancy=2)
def softmax_kernel_tma(
    output,
    input,
    n_rows: ConstInt,
    n_cols: ConstInt,
    TILE_SIZE: ConstInt,
):
    # Static persistent scheduling: each block processes multiple rows
    pid = ct.bid(0)
    num_programs = ct.num_blocks(0)

    for row_idx in range(pid, n_rows, num_programs):
        # Load the entire row in one tile (TILE_SIZE >= n_cols by design)
        row = ct.load(input, index=(row_idx, 0), shape=(1, TILE_SIZE), padding_mode=ct.PaddingMode.NEG_INF)

        # Convert to float32 for computation
        row = ct.astype(row, ct.float32)

        # Subtract maximum for numerical stability
        row_max = ct.max(row, 1, keepdims=True)
        row_minus_max = ct.sub(row, row_max)

        # Compute exponential
        numerator = ct.exp(row_minus_max)

        # Compute sum for normalization
        denominator = ct.sum(numerator, 1, keepdims=True)

        # Final softmax computation
        softmax_output = ct.truediv(numerator, denominator)

        # Convert back to original dtype and store
        softmax_output = ct.astype(softmax_output, input.dtype)
        ct.store(output, index=(row_idx, 0), tile=softmax_output)


# Launch patterns for the kernels:
def launch_softmax_kernel(input, output, TILE_SIZE=1024):
    """
    Launch the basic cuTile softmax kernel with static persistent scheduling

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
        TILE_SIZE: Tile size for processing
    """
    n_rows, n_cols = input.shape
    original_n_cols = n_cols

    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    occupancy = 4  # Match @ct.kernel(occupancy=4)
    num_programs = min(NUM_SM * occupancy, n_rows)
    grid = (num_programs, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        softmax_kernel,
        (
            output,
            input,
            n_rows,
            TILE_SIZE,
            original_n_cols,
        ),
    )


def launch_softmax_kernel_tma(
    input,
    output,
):
    """
    Launch the TMA cuTile softmax kernel

    Args:
        input: Input tensor of shape (n_rows, n_cols)
        output: Output tensor of shape (n_rows, n_cols)
    """
    # Ensure input is 2D
    original_shape = input.shape
    if input.dim() == 1:
        input = input.unsqueeze(0)
        output = output.unsqueeze(0)
    elif input.dim() > 2:
        input = input.view(-1, input.shape[-1])
        output = output.view(-1, output.shape[-1])

    n_rows, n_cols = input.shape

    TILE_SIZE = next_power_of_2(n_cols)
    original_n_cols = n_cols

    # Regular TMA path (single tile per row, persistent scheduling)
    softmax_kernel_forward = softmax_kernel_tma

    # Ensure tensors are contiguous
    input = input.contiguous()
    output = output.contiguous()

    NUM_SM = torch.cuda.get_device_properties(input.device).multi_processor_count
    occupancy = 2  # Match @ct.kernel(occupancy=2)
    num_programs = min(NUM_SM * occupancy, n_rows)
    grid = (num_programs, 1, 1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        softmax_kernel_forward,
        (
            output,
            input,
            n_rows,
            original_n_cols,
            TILE_SIZE,
        ),
    )


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        use_tma=False,
    ):
        n_rows, n_cols = x.shape

        # Create output tensor
        y = torch.empty_like(x)

        if use_tma:
            # Use TMA implementation
            launch_softmax_kernel_tma(
                x,
                y,
            )
        else:
            # Use grid-based implementation - ensure single tile per row for correctness
            TILE_SIZE = next_power_of_2(n_cols)
            launch_softmax_kernel(x, y, TILE_SIZE=TILE_SIZE)
        return y


@register_impl("softmax", backend="cutile")
def softmax(
    x,
    use_tma=False,
    **kwargs,
):
    """
    Performs softmax using cuTile kernels with automatic gradient support

    Args:
        x: Input tensor of shape (M, N)
        use_tma: Whether to use TMA (Tensor Memory Accelerator) implementation.
                Requires H100+ GPU (compute capability >= 9.0)
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Softmax output tensor with gradient support
    """
    return Softmax.apply(
        x,
        use_tma,
    )
