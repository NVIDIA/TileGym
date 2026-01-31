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


# Backward kernel for silu_and_mul
# Computes gradients using recomputation (no saved activations)
# Forward: c = silu(a) * b = a * sigmoid(a) * b
# da = dc * b * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
#    = dc * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
# db = dc * silu(a)
@ct.kernel
def silu_and_mul_backward_kernel_row_wise(
    grad_output,  # dc: (batch_size, hidden_size)
    input,  # original input: (batch_size, 2*hidden_size)
    grad_a,  # output: (batch_size, hidden_size)
    grad_b,  # output: (batch_size, hidden_size)
    TILE_SIZE: ConstInt,
    hidden_size: ConstInt,
):
    bid = ct.bid(0)  # row index
    offsets = ct.arange(TILE_SIZE, dtype=torch.int32)

    row_idx = bid
    a_col_idx = offsets  # First half: [0, hidden_size)
    b_col_idx = offsets + hidden_size  # Second half: [hidden_size, 2*hidden_size)

    # Load grad_output (dc)
    dc_tile = ct.gather(grad_output, (row_idx, offsets), check_bounds=True)
    dc_tile = ct.astype(dc_tile, torch.float32)

    # Recompute a and b from input (saves memory vs saving in forward)
    a_tile = ct.gather(input, (row_idx, a_col_idx), check_bounds=True)
    b_tile = ct.gather(input, (row_idx, b_col_idx), check_bounds=True)
    a_tile = ct.astype(a_tile, torch.float32)
    b_tile = ct.astype(b_tile, torch.float32)

    # Recompute sigmoid(a) and silu(a)
    denom = ct.add(1, ct.exp(-a_tile), flush_to_zero=True)
    sigmoid_a = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)
    silu_a = ct.mul(a_tile, sigmoid_a, flush_to_zero=True)

    # Compute db = dc * silu(a)
    db_tile = ct.mul(dc_tile, silu_a, flush_to_zero=True)
    db_tile = ct.astype(db_tile, input.dtype)
    ct.scatter(grad_b, (row_idx, offsets), db_tile, check_bounds=True)

    # Compute da = dc * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    # = dc * b * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
    one_minus_sigmoid = ct.add(1.0, -sigmoid_a, flush_to_zero=True)
    silu_grad = ct.add(sigmoid_a, ct.mul(silu_a, one_minus_sigmoid, flush_to_zero=True), flush_to_zero=True)
    da_tile = ct.mul(dc_tile, ct.mul(b_tile, silu_grad, flush_to_zero=True), flush_to_zero=True)
    da_tile = ct.astype(da_tile, input.dtype)
    ct.scatter(grad_a, (row_idx, offsets), da_tile, check_bounds=True)


def silu_and_mul_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for fused SiLU and Mul operation.

    Args:
        grad_output: Gradient w.r.t. output, shape (..., hidden_size)
        input: Original input tensor, shape (..., 2*hidden_size)

    Returns:
        Tuple of (grad_a, grad_b) each with shape (..., hidden_size)
    """
    original_output_shape = grad_output.shape
    hidden_size = original_output_shape[-1]

    # Flatten to 2D
    grad_output_flat = grad_output.contiguous().view(-1, hidden_size)
    input_flat = input.contiguous().view(-1, input.shape[-1])
    batch_size = grad_output_flat.shape[0]

    # Allocate output gradients
    grad_a = torch.empty_like(grad_output_flat)
    grad_b = torch.empty_like(grad_output_flat)

    from .utils import next_power_of_2

    TILE_SIZE = next_power_of_2(hidden_size)
    grid = (batch_size,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        silu_and_mul_backward_kernel_row_wise,
        (grad_output_flat, input_flat, grad_a, grad_b, TILE_SIZE, hidden_size),
    )

    return grad_a.view(*original_output_shape), grad_b.view(*original_output_shape)


class SiLUAndMulFunction(torch.autograd.Function):
    """Autograd function for silu_and_mul with backward support."""

    @staticmethod
    def forward(ctx, input: torch.Tensor):
        # Save input for backward (used in recomputation)
        ctx.save_for_backward(input)

        # Get shape info
        original_shape = input.shape
        hidden_size = original_shape[-1] // 2

        # Flatten input to 2D
        input_flat = input.view(-1, original_shape[-1])
        batch_size = input_flat.shape[0]

        # Allocate output
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

        # Reshape output
        output_shape = list(original_shape)
        output_shape[-1] = hidden_size
        return output.view(*output_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_a, grad_b = silu_and_mul_backward(grad_output, input)

        # Concatenate gradients for the original input layout
        grad_input = torch.cat([grad_a, grad_b], dim=-1)
        return grad_input


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
    # Use autograd wrapper when backward is needed
    if input.requires_grad:
        if out is not None:
            raise ValueError("out parameter not supported when requires_grad=True")
        return SiLUAndMulFunction.apply(input)

    # Direct kernel call for inference (no backward needed)
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
