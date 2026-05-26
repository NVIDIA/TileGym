# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import operator
from functools import reduce
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel

_BLOCK_SIZE = 256

_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "geglu.cuh",
    kernel_name="geglu_fwd_kernel",
    include_paths=[Path(__file__).parent],
)

_bwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "geglu.cuh",
    kernel_name="geglu_bwd_kernel",
    include_paths=[Path(__file__).parent],
)


class GegluTileCpp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, approximate="none"):
        assert approximate == "none" or approximate == "tanh", "Only `none` or `tanh` activations are supported"
        assert input.is_contiguous()
        assert input.shape[dim] % 2 == 0
        x_shape = input.shape
        dim = dim % len(x_shape)
        y_shape = list(x_shape)
        y_shape[dim] = y_shape[dim] // 2
        x_flat = input.view(-1)
        y_flat = torch.empty(reduce(operator.mul, y_shape, 1), device=input.device, dtype=input.dtype)
        m_stride = 0 if dim == 0 else input.stride(dim - 1)
        my_stride = 0 if dim == 0 else reduce(operator.mul, y_shape[dim:], 1)
        N = reduce(operator.mul, x_shape[dim:], 1) // 2
        n_elements = reduce(operator.mul, x_shape, 1) // 2
        approximate_mode = 1 if approximate == "tanh" else 0
        kernel, _, _ = _fwd_kernel.get_kernel(
            dtype=input.dtype,
            template_params=[_BLOCK_SIZE, approximate_mode],
            signature="const {T}*, {T}*, int, int, int, int",
        )
        _fwd_kernel.launch(
            grid=(math.ceil(n_elements / _BLOCK_SIZE),),
            kernel=kernel,
            args=[
                np.uint64(x_flat.data_ptr()),
                np.uint64(y_flat.data_ptr()),
                np.int32(N),
                np.int32(m_stride),
                np.int32(my_stride),
                np.int32(n_elements),
            ],
        )
        y = y_flat.view(y_shape)
        ctx.save_for_backward(input)
        ctx.dim = dim
        ctx.N = N
        ctx.n_elements = n_elements
        ctx.m_stride = m_stride
        ctx.my_stride = my_stride
        ctx.approximate_mode = approximate_mode
        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.is_contiguous()
        (input,) = ctx.saved_tensors
        dx_flat = torch.empty_like(input.view(-1))
        kernel, _, _ = _bwd_kernel.get_kernel(
            dtype=dy.dtype,
            template_params=[_BLOCK_SIZE, ctx.approximate_mode],
            signature="{T}*, const {T}*, const {T}*, int, int, int, int",
        )
        _bwd_kernel.launch(
            grid=(math.ceil(ctx.n_elements / _BLOCK_SIZE),),
            kernel=kernel,
            args=[
                np.uint64(dx_flat.data_ptr()),
                np.uint64(dy.view(-1).data_ptr()),
                np.uint64(input.view(-1).data_ptr()),
                np.int32(ctx.N),
                np.int32(ctx.m_stride),
                np.int32(ctx.my_stride),
                np.int32(ctx.n_elements),
            ],
        )
        return dx_flat.view(input.shape), None, None


@register_impl("geglu", backend="tilecpp")
def geglu(input: torch.Tensor, dim=-1, approximate="none"):
    return GegluTileCpp.apply(input, dim, approximate)
