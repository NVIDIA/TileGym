# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools

import cuda.tile as ct
import torch
from cuda.tile._numeric_semantics import RoundingMode as RMd

from tilegym.backend import register_impl

# Type aliases for constants
ConstInt = ct.Constant[int]


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


# To be launched with grid = number of rows (batch_size)
# each "block" computes an entire row of the ouptut
@ct.kernel
def silu_and_mul_kernel_row_wise(
    input,
    output,
    TILE_SIZE: ConstInt,
    hidden_size: ConstInt,
):
    bid = ct.bid(0)  # this gives us our row
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)

    # For 2D input (batch_size, 2*hidden_size), we need 2D indices
    # Row index is just bid (scalar), column indices are offsets-based
    row_idx = bid
    a_col_idx = offsets  # First half: [0, hidden_size)
    b_col_idx = offsets + hidden_size  # Second half: [hidden_size, 2*hidden_size)

    # Load tiles using gather with 2D indices
    # gather broadcasts (scalar, tile) to (tile,)
    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    # Implement sigmoid for SiLU
    denom = ct.add(1, ct.exp(-a_tile), flush_to_zero=True)
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)

    # Perform SiLU(a) * b
    silu_a = ct.mul(a_tile, sigmoid_a, flush_to_zero=True)
    result = ct.mul(silu_a, b_tile, flush_to_zero=True)
    result = ct.astype(result, input.dtype)

    # Store result using scatter with 2D indices
    # output is also 2D: (batch_size, hidden_size)
    out_col_idx = offsets
    ct.scatter(output, (row_idx, out_col_idx), result, check_bounds=True)


@register_impl("silu_and_mul", backend="cutile")
@ensure_contiguous
def silu_and_mul(
    input: torch.Tensor,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """
    Fused SiLU and Mul operation implemented with Cutile.

    Computes: silu(input[..., :hidden_size]) * input[..., hidden_size:]

    Args:
        input (torch.Tensor): Input tensor of shape (..., 2 * hidden_size)
        out (Optional[torch.Tensor]): Output tensor, if specified kernel will update in-place
    Returns:
        torch.Tensor: Output tensor of shape (..., hidden_size)
    """
    # Save original shape and flatten input for simpler processing
    original_shape = input.shape
    hidden_size = original_shape[-1] // 2

    # Flatten input to 2D: (batch_size, 2 * hidden_size)
    input_flat = input.view(-1, original_shape[-1])
    batch_size = input_flat.shape[0]

    # Get final output shape
    output_shape = list(original_shape)
    output_shape[-1] = hidden_size
    # Prepare output tensor
    if out is not None:
        # Ensure out shape is correct
        if out.shape != tuple(output_shape):
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {tuple(output_shape)}")
        output = out.view(-1, hidden_size)
    else:
        output = torch.empty(
            (batch_size, hidden_size),
            dtype=input.dtype,
            device=input.device,
        )

    from .utils import next_power_of_2

    TILE_SIZE = next_power_of_2(hidden_size)
    grid = (batch_size,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        silu_and_mul_kernel_row_wise,
        (input_flat, output, TILE_SIZE, hidden_size),
    )
    return output.reshape(*output_shape)
