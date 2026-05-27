# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ SiLU and Mul
Fused SiLU activation with element-wise multiplication using CUDA C++ tile kernels.

Computes: silu(input[..., :hidden_size]) * input[..., hidden_size:]
where silu(x) = x * sigmoid(x)
"""

import functools
import logging
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import make_kernel_args
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)

# =============================================================================
# Kernel Definition
# =============================================================================

_silu_and_mul_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "silu_and_mul.cuh",
    kernel_name="silu_and_mul_kernel",
)

_silu_and_mul_kernel_row_wise = TileCppKernel(
    source_path=Path(__file__).parent / "silu_and_mul.cuh",
    kernel_name="silu_and_mul_kernel_row_wise",
)


def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def maybe_to_contiguous(x):
            return x.contiguous() if isinstance(x, torch.Tensor) else x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(*args, **kwargs)

    return wrapper


def _next_power_of_2(n):
    """Returns the next power of 2 for n."""
    if n == 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def calculate_settings(n):
    """Choose appropriate block size based on hidden_size."""
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = _next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch kernel since n = {n} exceeds max block size = {MAX_FUSED_SIZE}.")
    return BLOCK_SIZE


# =============================================================================
# Kernel Launch Function
# =============================================================================


def _launch_silu_and_mul_kernel(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    stride: int,
    hidden_size: int,
    n_rows: int,
    block_size: int,
):
    """Launch the SiLU and Mul kernel."""
    dump_kernel_types("silu_and_mul_kernel", input_tensor, output_tensor)
    dtype = input_tensor.dtype

    kernel, _, _ = _silu_and_mul_kernel.get_kernel(
        dtype=dtype,
        template_params=[block_size],
        signature="const {T}*, {T}*, int, int",
    )

    # Grid: one block per row
    grid = (n_rows,)

    _silu_and_mul_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(input_tensor.data_ptr()),
            np.uint64(output_tensor.data_ptr()),
            np.int32(stride),
            np.int32(hidden_size),
        ],
    )


def _launch_silu_and_mul_kernel_row_wise(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    hidden_size: int,
    n_rows: int,
    tile_size: int,
):
    """Launch the row-wise SiLU and Mul kernel using tensor_span."""
    dump_kernel_types("silu_and_mul_kernel_row_wise", input_tensor, output_tensor)
    dtype = input_tensor.dtype

    # Template params: M, N, HIDDEN_SIZE, TILE_SIZE, INPUT_STRIDE, OUTPUT_STRIDE
    M = n_rows
    N = input_tensor.shape[1]  # 2 * hidden_size
    INPUT_STRIDE = input_tensor.stride(0)
    OUTPUT_STRIDE = output_tensor.stride(0)

    kernel, _, _ = _silu_and_mul_kernel_row_wise.get_kernel(
        dtype=dtype,
        template_params=[M, N, hidden_size, tile_size, INPUT_STRIDE, OUTPUT_STRIDE],
        signature="{T}*, {T}*",
    )

    # Grid: one block per row
    grid = (n_rows,)

    # Simple 2-parameter signature (dimensions are template params)
    _silu_and_mul_kernel_row_wise.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(input_tensor.data_ptr()),
            np.uint64(output_tensor.data_ptr()),
        ],
    )


# =============================================================================
# Public Interface
# =============================================================================


class _SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, out=None, kernel_type="row_wise", **kwargs):
        return _silu_and_mul_impl(input, out=out, kernel_type=kernel_type, **kwargs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Backward pass for tilecpp silu_and_mul is not implemented.")


@register_impl("silu_and_mul", backend="tilecpp")
@ensure_contiguous
def silu_and_mul(
    input: torch.Tensor, out: torch.Tensor = None, kernel_type: str = "row_wise", **kwargs
) -> torch.Tensor:
    """
    Fused SiLU and Mul operation implemented with CUDA C++ tile kernels.

    Computes: silu(input[..., :hidden_size]) * input[..., hidden_size:]

    Args:
        input (torch.Tensor): Input tensor of shape (..., 2 * hidden_size)
        out (Optional[torch.Tensor]): Output tensor, if specified kernel will update in-place
        kernel_type (str): Kernel implementation to use. Options: "row_wise", "default". Default is "row_wise"
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        torch.Tensor: Output tensor of shape (..., hidden_size)

    Note:
        Backward pass is not implemented and will raise NotImplementedError on autograd.
    """
    if input.requires_grad:
        return _SiluAndMul.apply(input, out, kernel_type, **kwargs)
    return _silu_and_mul_impl(input, out=out, kernel_type=kernel_type, **kwargs)


def _silu_and_mul_impl(
    input: torch.Tensor, out: torch.Tensor = None, kernel_type: str = "row_wise", **kwargs
) -> torch.Tensor:
    """
    Fused SiLU and Mul operation implemented with CUDA C++ tile kernels.

    Computes: silu(input[..., :hidden_size]) * input[..., hidden_size:]

    Args:
        input (torch.Tensor): Input tensor of shape (..., 2 * hidden_size)
        out (Optional[torch.Tensor]): Output tensor, if specified kernel will update in-place
        kernel_type (str): Kernel implementation to use. Options: "row_wise", "default". Default is "row_wise"
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        torch.Tensor: Output tensor of shape (..., hidden_size)
    """
    # Save original shape
    original_shape = input.shape

    # Calculate hidden_size
    hidden_size = original_shape[-1] // 2

    # Get final output shape
    output_shape = list(original_shape)
    output_shape[-1] = hidden_size

    # Reshape input to 2D for processing
    input = input.view(-1, original_shape[-1])
    n_rows = input.shape[0]

    # Prepare output tensor
    if out is not None:
        # Ensure out shape is correct
        if out.shape != tuple(output_shape):
            raise ValueError(f"Output tensor shape {out.shape} does not match expected shape {tuple(output_shape)}")
        output = out.view(-1, hidden_size)
    else:
        output = torch.empty((n_rows, hidden_size), dtype=input.dtype, device=input.device)

    # Calculate block/tile size
    tile_size = calculate_settings(hidden_size)

    # Validate kernel_type
    valid_kernel_types = ["default", "row_wise"]
    if kernel_type not in valid_kernel_types:
        raise ValueError(f"Invalid kernel_type '{kernel_type}'. Must be one of {valid_kernel_types}")

    # Launch appropriate kernel
    if kernel_type == "row_wise":
        _launch_silu_and_mul_kernel_row_wise(
            input,
            output,
            hidden_size,
            n_rows,
            tile_size,
        )
    else:  # default
        _launch_silu_and_mul_kernel(
            input,
            output,
            input.stride(0),
            hidden_size,
            n_rows,
            tile_size,
        )

    return output.view(*output_shape)
