# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel

_BLOCK_SIZE = 1024

_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "gelu.cuh",
    kernel_name="gelu_fwd_kernel",
    include_paths=[Path(__file__).parent],
)

_bwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "gelu.cuh",
    kernel_name="gelu_bwd_kernel",
    include_paths=[Path(__file__).parent],
)

_MODES = {"none": 0, "tanh": 1}


class _GeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op_id=0):
        y = torch.empty_like(x)
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        kernel, _, _ = _fwd_kernel.get_kernel(
            dtype=x.dtype,
            template_params=[_BLOCK_SIZE, op_id],
            signature="const {T}*, {T}*, int",
        )
        _fwd_kernel.launch(
            grid=(math.ceil(x_flat.numel() / _BLOCK_SIZE),),
            kernel=kernel,
            args=[np.uint64(x_flat.data_ptr()), np.uint64(y_flat.data_ptr()), np.int32(x_flat.numel())],
        )
        ctx.save_for_backward(x_flat)
        ctx.op_id = op_id
        ctx.shape = x.shape
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dy_flat = dy.contiguous().view(-1)
        dx = torch.empty_like(dy_flat)
        kernel, _, _ = _bwd_kernel.get_kernel(
            dtype=dy.dtype,
            template_params=[_BLOCK_SIZE, ctx.op_id],
            signature="const {T}*, const {T}*, {T}*, int",
        )
        _bwd_kernel.launch(
            grid=(math.ceil(dy_flat.numel() / _BLOCK_SIZE),),
            kernel=kernel,
            args=[
                np.uint64(dy_flat.data_ptr()),
                np.uint64(x.data_ptr()),
                np.uint64(dx.data_ptr()),
                np.int32(dy_flat.numel()),
            ],
        )
        return dx.view(ctx.shape), None


@register_impl("gelu", backend="tilecpp")
def gelu(input: torch.Tensor, approximate="none"):
    """Returns GELU activation of input."""
    op_id = _MODES[approximate]
    return _GeluFunction.apply(input.view(-1), op_id).view(input.shape)
