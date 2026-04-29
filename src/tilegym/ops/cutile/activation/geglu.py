# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import operator
from functools import reduce

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .gelu import GELU_EXACT
from .gelu import GELU_TANH
from .gelu import _gelu_fwd
from .gelu import _gelu_tanh_fwd
from .gelu import _normal_cdf
from .gelu import _normal_pdf


def _gelu_bwd(x_val, dy_val, BLOCK_SIZE: ct.Constant[int]):
    # dy * (Φ(x) + x * φ(x))
    return dy_val * (_normal_cdf(x_val, BLOCK_SIZE) + x_val * _normal_pdf(x_val, BLOCK_SIZE))


@ct.kernel
def geglu_fwd_kernel(
    y,
    x,
    N: ct.Constant[int],
    m_stride: ct.Constant[int],
    my_stride: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    APPROXIMATE: ct.Constant[int],
):
    """
    Forward kernel for GEGLU activation: output = a * GELU(b)
    where a is the left half and b is the right half of the input.
    """
    bid = ct.bid(0)

    global_id = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    m_id = global_id // N
    n_offs = global_id % N

    left_ptr_offsets = m_id * m_stride + n_offs
    right_ptr_offsets = m_id * m_stride + n_offs + N
    out_ptr_offsets = m_id * my_stride + n_offs

    a = ct.gather(x, (left_ptr_offsets,))
    b = ct.gather(x, (right_ptr_offsets,))

    if APPROXIMATE == GELU_TANH:
        out = a * _gelu_tanh_fwd(b, BLOCK_SIZE)
    else:
        out = a * _gelu_fwd(b, BLOCK_SIZE)

    ct.scatter(y, (out_ptr_offsets,), out)


@ct.kernel
def geglu_bwd_kernel(
    dx,
    dy,
    x,
    N: ct.Constant[int],
    m_stride: ct.Constant[int],
    my_stride: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    APPROXIMATE: ct.Constant[int],
):
    """
    Backward kernel for GEGLU: da = dy * GELU(b),  db = dy * a * GELU'(b).
    """
    bid = ct.bid(0)

    global_id = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    m_id = global_id // N
    n_offs = global_id % N

    left_ptr_offsets = m_id * m_stride + n_offs
    right_ptr_offsets = m_id * m_stride + n_offs + N
    out_ptr_offsets = m_id * my_stride + n_offs

    a = ct.gather(x, (left_ptr_offsets,))
    b = ct.gather(x, (right_ptr_offsets,))
    dy_val = ct.gather(dy, (out_ptr_offsets,))

    if APPROXIMATE == GELU_TANH:
        gelu_b = _gelu_tanh_fwd(b, BLOCK_SIZE)
    else:
        gelu_b = _gelu_fwd(b, BLOCK_SIZE)

    da = dy_val * gelu_b
    db = a * _gelu_bwd(b, dy_val, BLOCK_SIZE)

    ct.scatter(dx, (left_ptr_offsets,), da)
    ct.scatter(dx, (right_ptr_offsets,), db)


class GegluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, approximate):
        assert approximate == "none" or approximate == "tanh", "Only `none` or `tanh` activations are supported"
        assert x.is_contiguous()
        assert x.shape[dim] % 2 == 0

        x_shape = x.shape
        dim = dim % len(x_shape)
        y_shape = list(x_shape)
        y_shape[dim] = y_shape[dim] // 2

        x_flat = x.view(-1)
        y_flat = torch.empty(reduce(operator.mul, y_shape, 1), device=x.device, dtype=x.dtype)

        if dim == 0:
            m_stride = 0
            my_stride = 0
        else:
            m_stride = x.stride(dim - 1)
            my_stride = reduce(operator.mul, y_shape[dim:], 1)

        M = reduce(operator.mul, x_shape[:dim], 1)
        N2 = reduce(operator.mul, x_shape[dim:], 1) // 2
        n_elements = reduce(operator.mul, x_shape, 1) // 2

        BLOCK_SIZE = 256
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
        approximate_mode = GELU_TANH if approximate == "tanh" else GELU_EXACT

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            geglu_fwd_kernel,
            (y_flat, x_flat, N2, m_stride, my_stride, BLOCK_SIZE, approximate_mode),
        )

        y = y_flat.view(y_shape)
        ctx.save_for_backward(x, y)
        ctx.M = M
        ctx.N2 = N2
        ctx.dim = dim
        ctx.approximate = approximate
        ctx.n_elements = n_elements

        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.is_contiguous()
        x, y = ctx.saved_tensors
        dim = ctx.dim
        approximate = ctx.approximate
        M = ctx.M
        N2 = ctx.N2
        n_elements = ctx.n_elements

        x_shape = x.shape
        dx_flat = torch.empty_like(x.view(-1))

        if dim == 0:
            m_stride = 0
            my_stride = 0
        else:
            m_stride = x.stride(dim - 1)
            my_stride = dy.stride(dim - 1)

        BLOCK_SIZE = 256
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
        approximate_mode = GELU_TANH if approximate == "tanh" else GELU_EXACT

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            geglu_bwd_kernel,
            (dx_flat, dy.view(-1), x.view(-1), N2, m_stride, my_stride, BLOCK_SIZE, approximate_mode),
        )

        return dx_flat.view(x_shape), None, None


@register_impl("geglu", backend="cutile")
def geglu(input: torch.Tensor, dim=-1, approximate="none"):
    r"""
    Returns GEGLU activation of input.
    $f(x) = a \otimes GELU(b)$
    Where $a$ is the first half of the input matrices and $b$ is the second half.
    ```dim``` is the dimension on which to split the input.
    If approximate is ``'tanh'`` then
    $GELU(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))$
    Else if approximate is ``'none'`` then
    $GELU(x) = x * \Phi(x)$
    Where $Phi(x)$ is the Cumulative Distribution Function for Gaussian Distribution.
    Args:
        input: Tensor
        dim: int
        approximate: ``'none'`` or ``'tanh'``
    """
    return GegluFunction.apply(input, dim, approximate)
