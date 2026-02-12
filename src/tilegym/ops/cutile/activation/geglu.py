# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import operator
from functools import reduce

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .gelu import gelu_fwd_ct
from .gelu import gelu_tanh_fwd_ct
from .gelu import standard_normal_cdf_ct
from .gelu import standard_normal_pdf_ct

# Approximation mode constants
GELU_EXACT = 0
GELU_TANH = 1


def gelu_bwd_ct(x_val, dy_val, BLOCK_SIZE: ct.Constant[int]):
    """
    Compute GELU backward gradient: dy * (Φ(x) + x * φ(x))

    Args:
        x_val: Input value tile
        dy_val: Output gradient tile
        BLOCK_SIZE: Block size constant

    Returns:
        Gradient with respect to input
    """
    cdf_val = standard_normal_cdf_ct(x_val, BLOCK_SIZE)
    pdf_val = standard_normal_pdf_ct(x_val, BLOCK_SIZE)
    x_pdf = x_val * pdf_val
    grad_factor = cdf_val + x_pdf
    return dy_val * grad_factor


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
    Forward kernel for GEGLU activation.

    Computes: output = a * GELU(b)
    where a is the left half and b is the right half of the input.
    """
    bid = ct.bid(0)

    # Compute global indices for this block
    global_id = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Compute m_id (batch/row index) and n_offs (column offset)
    # m_id = global_id // N
    m_id = global_id // N
    n_offs = global_id % N

    # Compute strides for input and output
    m_offs = m_id * m_stride
    my_offs = m_id * my_stride

    # Calculate pointer offsets for left and right halves
    left_ptr_offsets = m_offs + n_offs
    right_ptr_offsets = m_offs + n_offs + N
    out_ptr_offsets = my_offs + n_offs

    # Load left and right halves using gather
    a = ct.gather(x, (left_ptr_offsets,))
    b = ct.gather(x, (right_ptr_offsets,))

    # Compute a * GELU(b)
    if APPROXIMATE == GELU_TANH:
        geglu_output = a * gelu_tanh_fwd_ct(b, BLOCK_SIZE)
    else:
        geglu_output = a * gelu_fwd_ct(b, BLOCK_SIZE)

    # Store output using scatter
    ct.scatter(y, (out_ptr_offsets,), geglu_output)


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
    Backward kernel for GEGLU activation.

    Computes gradients with respect to a and b:
    - da = dy * GELU(b)
    - db = dy * a * GELU'(b)
    """
    bid = ct.bid(0)

    # Compute global indices for this block
    global_id = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Compute m_id (batch/row index) and n_offs (column offset)
    m_id = global_id // N
    n_offs = global_id % N

    # Compute strides for input and output
    m_offs = m_id * m_stride
    my_offs = m_id * my_stride

    # Calculate pointer offsets
    left_ptr_offsets = m_offs + n_offs
    right_ptr_offsets = m_offs + n_offs + N
    out_ptr_offsets = my_offs + n_offs

    # Load input splits and output gradient
    a = ct.gather(x, (left_ptr_offsets,))
    b = ct.gather(x, (right_ptr_offsets,))
    dy_val = ct.gather(dy, (out_ptr_offsets,))

    # Compute GELU(b) for gradient of a
    if APPROXIMATE == GELU_TANH:
        dy_da = gelu_tanh_fwd_ct(b, BLOCK_SIZE)
    else:
        dy_da = gelu_fwd_ct(b, BLOCK_SIZE)

    # Compute gradients
    da = dy_val * dy_da
    db = a * gelu_bwd_ct(b, dy_val, BLOCK_SIZE)

    # Store gradients
    ct.scatter(dx, (left_ptr_offsets,), da)
    ct.scatter(dx, (right_ptr_offsets,), db)


class GEGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, approximate):
        assert approximate == "none" or approximate == "tanh", "Only `none` or `tanh` activations are supported"
        # Process input
        assert x.is_contiguous()
        assert x.shape[dim] % 2 == 0

        x_shape = x.shape
        dim = dim % len(x_shape)
        y_shape = list(x_shape)
        y_shape[dim] = y_shape[dim] // 2

        # Flatten input and output for kernel processing
        x_flat = x.view(-1)
        y_flat = torch.empty(reduce(operator.mul, y_shape, 1), device=x.device, dtype=x.dtype)

        # Compute strides
        if dim == 0:
            m_stride = 0
            my_stride = 0
        else:
            m_stride = x.stride(dim - 1)
            my_stride = reduce(operator.mul, y_shape[dim:], 1)  # Stride for flattened y

        # Compute dimensions
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

        # Reshape output back to expected shape
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

        # Flatten tensors for kernel processing
        x_flat = x.view(-1)
        dy_flat = dy.view(-1)
        dx_flat = torch.empty_like(x_flat)

        # Compute strides
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
            (dx_flat, dy_flat, x_flat, N2, m_stride, my_stride, BLOCK_SIZE, approximate_mode),
        )

        # Reshape output back to expected shape
        dx = dx_flat.view(x_shape)

        return dx, None, None


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
    return GEGLU.apply(input, dim, approximate)
