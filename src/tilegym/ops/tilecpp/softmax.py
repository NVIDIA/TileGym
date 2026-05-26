# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Softmax
Computes softmax along the last dimension using CUDA C++ tile kernels.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Threshold for using online algorithm (large column counts)
COL_THRESHOLD = 16384 * 2

# Block size for online algorithm
ONLINE_BLOCK_SIZE = 8192

# =============================================================================
# Kernel Definitions
# =============================================================================

_softmax_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "softmax.cuh",
    kernel_name="softmax_kernel",
)

_online_softmax_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "softmax.cuh",
    kernel_name="online_softmax_kernel",
)

_softmax_kernel_backward = TileCppKernel(
    source_path=Path(__file__).parent / "softmax.cuh",
    kernel_name="softmax_kernel_backward",
)

_online_softmax_kernel_backward = TileCppKernel(
    source_path=Path(__file__).parent / "softmax.cuh",
    kernel_name="online_softmax_kernel_backward",
)


# =============================================================================
# Helper Functions
# =============================================================================


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


def _get_num_sm():
    """Get number of SMs on current device."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.multi_processor_count


# =============================================================================
# Kernel Launch Functions
# =============================================================================


def _launch_softmax_forward(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    n_rows: int,
    n_cols: int,
    block_size: int,
):
    """Launch the softmax forward kernel."""
    dump_kernel_types("softmax_kernel", input_tensor, output_tensor)
    dtype = input_tensor.dtype

    kernel, _, _ = _softmax_kernel.get_kernel(
        dtype=dtype,
        template_params=[block_size],
        signature="{T}*, const {T}*, int, int, int, int, int",
    )

    num_sm = _get_num_sm()
    occupancy = 4
    num_programs = min(num_sm * occupancy, n_rows)

    grid = (num_programs,)

    _softmax_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(output_tensor.data_ptr()),
            np.uint64(input_tensor.data_ptr()),
            np.int32(input_tensor.stride(0)),
            np.int32(output_tensor.stride(0)),
            np.int32(n_rows),
            np.int32(n_cols),
            np.int32(num_programs),
        ],
    )


def _launch_online_softmax_forward(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    n_rows: int,
    n_cols: int,
    block_size: int,
):
    """Launch the online softmax forward kernel."""
    dump_kernel_types("online_softmax_kernel", input_tensor, output_tensor)
    dtype = input_tensor.dtype

    kernel, _, _ = _online_softmax_kernel.get_kernel(
        dtype=dtype,
        template_params=[block_size],
        signature="{T}*, const {T}*, int, int, int",
    )

    grid = (n_rows,)

    _online_softmax_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(output_tensor.data_ptr()),
            np.uint64(input_tensor.data_ptr()),
            np.int32(input_tensor.stride(0)),
            np.int32(output_tensor.stride(0)),
            np.int32(n_cols),
        ],
    )


def _launch_softmax_backward(
    dx_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    dy_tensor: torch.Tensor,
    n_rows: int,
    n_cols: int,
    block_size: int,
):
    """Launch the softmax backward kernel."""
    dump_kernel_types("softmax_kernel_backward", y_tensor, dy_tensor, dx_tensor)
    dtype = y_tensor.dtype

    kernel, _, _ = _softmax_kernel_backward.get_kernel(
        dtype=dtype,
        template_params=[block_size],
        signature="{T}*, const {T}*, const {T}*, int, int, int, int",
    )

    grid = (n_rows,)

    _softmax_kernel_backward.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(dx_tensor.data_ptr()),
            np.uint64(y_tensor.data_ptr()),
            np.uint64(dy_tensor.data_ptr()),
            np.int32(dy_tensor.stride(0)),
            np.int32(y_tensor.stride(0)),
            np.int32(dx_tensor.stride(0)),
            np.int32(n_cols),
        ],
    )


def _launch_online_softmax_backward(
    dx_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    dy_tensor: torch.Tensor,
    n_rows: int,
    n_cols: int,
    block_size: int,
):
    """Launch the online softmax backward kernel."""
    dump_kernel_types("online_softmax_kernel_backward", y_tensor, dy_tensor, dx_tensor)
    dtype = y_tensor.dtype

    kernel, _, _ = _online_softmax_kernel_backward.get_kernel(
        dtype=dtype,
        template_params=[block_size],
        signature="{T}*, const {T}*, const {T}*, int, int, int, int",
    )

    grid = (n_rows,)

    _online_softmax_kernel_backward.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(dx_tensor.data_ptr()),
            np.uint64(y_tensor.data_ptr()),
            np.uint64(dy_tensor.data_ptr()),
            np.int32(dy_tensor.stride(0)),
            np.int32(y_tensor.stride(0)),
            np.int32(dx_tensor.stride(0)),
            np.int32(n_cols),
        ],
    )


# =============================================================================
# Autograd Functions
# =============================================================================


class Softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = _next_power_of_2(n_cols)

        y = torch.empty_like(x)
        _launch_softmax_forward(x, y, n_rows, n_cols, BLOCK_SIZE)

        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        n_rows, n_cols = dy.shape
        BLOCK_SIZE = _next_power_of_2(n_cols)

        dx = torch.empty_like(dy)
        _launch_softmax_backward(dx, y, dy, n_rows, n_cols, BLOCK_SIZE)

        return dx


class OnlineSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        n_rows, n_cols = x.shape
        BLOCK_SIZE = ONLINE_BLOCK_SIZE

        y = torch.empty_like(x)
        _launch_online_softmax_forward(x, y, n_rows, n_cols, BLOCK_SIZE)

        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        (y,) = ctx.saved_tensors
        n_rows, n_cols = dy.shape
        BLOCK_SIZE = ONLINE_BLOCK_SIZE

        dx = torch.empty_like(dy)
        _launch_online_softmax_backward(dx, y, dy, n_rows, n_cols, BLOCK_SIZE)

        return dx


# =============================================================================
# Public Interface
# =============================================================================


@register_impl("softmax", backend="tilecpp")
def softmax(
    x: torch.Tensor,
    use_tma: bool = False,
    use_online: bool = None,
    **kwargs,
):
    r"""
    Performs Softmax on a Tensor of shape (M, N) along the N axis.

    Args:
        x: Tensor of shape (M, N)
        use_tma: Whether to use TMA.
        use_online: Whether to use online softmax implementation.
                   If None, automatically chooses based on tensor size.
    """
    n_rows, n_cols = x.shape

    # Automatically choose between regular and online algorithms based on tensor size
    if use_online is None:
        use_online = n_cols >= COL_THRESHOLD

    if use_online:
        return OnlineSoftmax.apply(x)
    else:
        return Softmax.apply(x)
