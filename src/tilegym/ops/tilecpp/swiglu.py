# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ SwiGLU Operation
SwiGLU activation using CUDA C++ tile kernels compiled with nvcc.

SwiGLU: c = silu(a) * b where silu(x) = x * sigmoid(x)
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def calculate_settings(n: int) -> int:
    """Calculate block size for the kernel.

    Reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    Returns:
        BLOCK_SIZE (power of 2)
    """
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch kernel since n = {n} exceeds the recommended blocksize = {MAX_FUSED_SIZE}.")
    return BLOCK_SIZE


# =============================================================================
# Kernel Definitions
# =============================================================================


_swiglu_backward_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "swiglu.cuh",
    kernel_name="swiglu_backward_kernel",
)

_swiglu_forward_gather_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "swiglu.cuh",
    kernel_name="swiglu_forward_kernel_gather",
)


def _get_num_sm():
    """Get number of SMs on the current GPU."""
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def _ceildiv(a, b):
    return -(a // -b)


# =============================================================================
# Kernel Launchers
# =============================================================================


def _launch_swiglu_backward_kernel(
    dc: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    da: torch.Tensor,
    db: torch.Tensor,
    stride: int,
    n_cols: int,
    block_size: int,
):
    """Launch the swiglu_backward_kernel CUDA kernel."""
    dump_kernel_types("swiglu_backward_kernel", dc, a, b)
    dtype = dc.dtype

    kernel, _, _ = _swiglu_backward_kernel.get_kernel(
        dtype=dtype,
        template_params=[block_size],
        signature="const {T}*, const {T}*, const {T}*, {T}*, {T}*, int, int",
    )

    n_rows = dc.shape[0]

    _swiglu_backward_kernel.launch(
        grid=n_rows,
        kernel=kernel,
        args=[
            np.uint64(dc.data_ptr()),
            np.uint64(a.data_ptr()),
            np.uint64(b.data_ptr()),
            np.uint64(da.data_ptr()),
            np.uint64(db.data_ptr()),
            np.int32(stride),
            np.int32(n_cols),
        ],
    )


def _launch_swiglu_forward_gather_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    n_rows: int,
    n_cols: int,
    block_size: int,
):
    """
    Grid: (n_rows,).  Each CTA gathers a (BLOCK_SIZE,) slice of one row and
    writes back via scatter.  BLOCK_SIZE = next_po2(n_cols).
    """
    dump_kernel_types("swiglu_forward_kernel_gather", a, b, c)
    dtype = a.dtype

    kernel, _, _ = _swiglu_forward_gather_kernel.get_kernel(
        dtype=dtype,
        template_params=[block_size],
        signature="const {T}*, const {T}*, {T}*, int, int",
    )

    _swiglu_forward_gather_kernel.launch(
        grid=n_rows,
        kernel=kernel,
        args=[
            np.uint64(a.data_ptr()),
            np.uint64(b.data_ptr()),
            np.uint64(c.data_ptr()),
            np.int32(n_cols),
            np.int32(a.stride(0)),
        ],
    )


# =============================================================================
# Public Functions
# =============================================================================


def swiglu_backward(a: torch.Tensor, b: torch.Tensor, dc: torch.Tensor) -> tuple:
    """
    SwiGLU backward pass.

    Args:
        a: First saved input tensor (not mutated)
        b: Second saved input tensor (not mutated)
        dc: Upstream gradient

    Returns:
        Tuple of (da, db) gradients
    """
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)

    da = torch.empty_like(a)
    db = torch.empty_like(b)

    BLOCK_SIZE = calculate_settings(n_cols)

    _launch_swiglu_backward_kernel(
        dc,
        a,
        b,
        da,
        db,
        stride=dc.stride(-2),
        n_cols=n_cols,
        block_size=BLOCK_SIZE,
    )

    return da.view(*ori_shape), db.view(*ori_shape)


def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.reshape(-1, n_cols)
    b = b.reshape(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    if n_cols % 8 != 0:
        raise NotImplementedError(f"tilecpp swiglu_forward requires n_cols ({n_cols}) to be a multiple of 8.")
    row_stride = a.stride(0)
    if row_stride % 16 != 0:
        raise NotImplementedError(f"tilecpp swiglu_forward requires row stride ({row_stride}) to be a multiple of 16.")

    BLOCK_SIZE = next_power_of_2(n_cols)
    _launch_swiglu_forward_gather_kernel(a, b, c, n_rows, n_cols, BLOCK_SIZE)
    return c.view(*ori_shape)


# =============================================================================
# Autograd Function
# =============================================================================


class SiLUMulFunction(torch.autograd.Function):
    """Autograd function for SwiGLU (SiLU * Mul)."""

    @staticmethod
    def forward(ctx, a, b):
        c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = swiglu_backward(a, b, dc)
        return a, b


# =============================================================================
# SwiGLU MLP Module
# =============================================================================


class SwiGLUMLP(nn.Module):
    """SwiGLU-based MLP layer."""

    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(SiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


# =============================================================================
# Registered Implementations
# =============================================================================


@register_impl("get_swiglu_module", backend="tilecpp")
def get_swiglu_module():
    """Get the SwiGLU MLP module class."""
    return SwiGLUMLP


def swiglu(a, b):
    return SiLUMulFunction.apply(a, b)


@register_impl("get_swiglu", backend="tilecpp")
def get_swiglu():
    """Return the autograd-aware SwiGLU function so .backward() works."""
    return swiglu
