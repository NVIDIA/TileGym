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

_OP_RELU = 0
_OP_ELU = 1
_OP_LEAKY_RELU = 2
_OP_SELU = 3
_OP_CELU = 4
_OP_RRELU = 5

_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "relu.cuh",
    kernel_name="relu_activation_fwd_kernel",
    include_paths=[Path(__file__).parent],
)

_bwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "relu.cuh",
    kernel_name="relu_activation_bwd_kernel",
    include_paths=[Path(__file__).parent],
)


class _ActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op_id, alpha=1.0, lower=1.0 / 8, upper=1.0 / 3, training=False):
        y = torch.empty_like(x)
        x_flat = x.view(-1)
        y_flat = y.view(-1)
        kernel, _, _ = _fwd_kernel.get_kernel(
            dtype=x.dtype,
            template_params=[_BLOCK_SIZE, op_id],
            signature="const {T}*, {T}*, int, float, float, float, bool",
        )
        _fwd_kernel.launch(
            grid=(math.ceil(x_flat.numel() / _BLOCK_SIZE),),
            kernel=kernel,
            args=[
                np.uint64(x_flat.data_ptr()),
                np.uint64(y_flat.data_ptr()),
                np.int32(x_flat.numel()),
                np.float32(alpha),
                np.float32(lower),
                np.float32(upper),
                np.bool_(training),
            ],
        )
        ctx.save_for_backward(x_flat)
        ctx.shape = x.shape
        ctx.op_id = op_id
        ctx.alpha = alpha
        ctx.lower = lower
        ctx.upper = upper
        ctx.training = training
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Backward pass is not implemented")


@register_impl("relu", backend="tilecpp")
def relu(x):
    return _ActivationFunction.apply(x, _OP_RELU, 1.0, 1.0 / 8, 1.0 / 3, False)
