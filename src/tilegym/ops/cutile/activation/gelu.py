# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import numpy as np
import torch

from tilegym.backend import register_impl

# Approximation mode constants
GELU_EXACT = 0
GELU_TANH = 1


def sigmoid_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # sigmoid(x) = 1 / (1 + exp(-x))
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)  # new var
    neg_x = ct.negative(x_val)  # new var
    exp_neg_x = ct.exp(neg_x)  # new var
    denom = ct.add(one, exp_neg_x)  # new var
    return ct.truediv(one, denom)


def tanh_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # tanh(x) = 2 * sigmoid(2*x) - 1
    two = ct.full((BLOCK_SIZE,), 2.0, dtype=x_val.dtype)  # new var
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)  # new var
    two_x = ct.mul(two, x_val)  # new var
    sigmoid_2x = sigmoid_ct(two_x, BLOCK_SIZE)  # new var
    two_sigmoid = ct.mul(two, sigmoid_2x)  # new var
    return ct.sub(two_sigmoid, one)


def standard_normal_cdf_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # cdf = 0.5 * (1 + erf(x / sqrt(2)))
    # Using tanh approximation for erf: erf(x) ≈ tanh(sqrt(2/π) * (x + 0.044715 * x^3))
    sqrt_2_div_pi = 0.7978845608028654  # new var
    coeff_044715 = 0.044715  # new var
    half = ct.full((BLOCK_SIZE,), 0.5, dtype=x_val.dtype)  # new var
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)  # new var
    sqrt_2_div_pi_tensor = ct.full((BLOCK_SIZE,), sqrt_2_div_pi, dtype=x_val.dtype)  # new var
    coeff_tensor = ct.full((BLOCK_SIZE,), coeff_044715, dtype=x_val.dtype)  # new var

    # Compute erf approximation
    x_cubed = ct.mul(ct.mul(x_val, x_val), x_val)  # new var
    coeff_x_cubed = ct.mul(coeff_tensor, x_cubed)  # new var
    inner_sum = ct.add(x_val, coeff_x_cubed)  # new var
    scaled_inner = ct.mul(sqrt_2_div_pi_tensor, inner_sum)  # new var
    erf_approx = tanh_ct(scaled_inner, BLOCK_SIZE)  # new var

    # Compute CDF
    one_plus_erf = ct.add(one, erf_approx)  # new var
    return ct.mul(half, one_plus_erf)


def gelu_tanh_fwd_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    sqrt_2_div_pi = 0.7978845608028654  # new var
    coeff_044715 = 0.044715  # new var
    half = ct.full((BLOCK_SIZE,), 0.5, dtype=x_val.dtype)  # new var
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)  # new var
    sqrt_2_div_pi_tensor = ct.full((BLOCK_SIZE,), sqrt_2_div_pi, dtype=x_val.dtype)  # new var
    coeff_tensor = ct.full((BLOCK_SIZE,), coeff_044715, dtype=x_val.dtype)  # new var

    x_cubed = ct.mul(ct.mul(x_val, x_val), x_val)  # new var
    coeff_x_cubed = ct.mul(coeff_tensor, x_cubed)  # new var
    inner_sum = ct.add(x_val, coeff_x_cubed)  # new var
    scaled_inner = ct.mul(sqrt_2_div_pi_tensor, inner_sum)  # new var
    tanh_val = tanh_ct(scaled_inner, BLOCK_SIZE)  # new var
    one_plus_tanh = ct.add(one, tanh_val)  # new var
    half_x = ct.mul(half, x_val)  # new var

    return ct.mul(half_x, one_plus_tanh)


def gelu_fwd_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # f(x) = x * Φ(x)
    cdf_val = standard_normal_cdf_ct(x_val, BLOCK_SIZE)  # new var
    return ct.mul(x_val, cdf_val)


@ct.kernel
def gelu_kernel_ct(
    y,
    x,
    n_elements: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    approximate: ct.Constant[int],
):
    """
    cuTile GELU activation kernel supporting both exact and tanh approximation modes.

    Args:
        y: Output tensor
        x: Input tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for computation
        approximate: 0 for exact GELU, 1 for tanh approximation
    """

    # Main kernel computation
    pid = ct.bid(0)  # new var
    block_start = pid * BLOCK_SIZE  # new var

    # Create offset tile
    offsets = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), block_start)  # new var
    mask = ct.less(offsets, n_elements)  # new var

    # Load input data
    x_tile = ct.gather(x, offsets)  # new var

    # Compute GELU based on approximation mode
    if approximate == GELU_TANH:
        gelu_output = gelu_tanh_fwd_ct(x_tile, BLOCK_SIZE)
    else:  # GELU_EXACT
        gelu_output = gelu_fwd_ct(x_tile, BLOCK_SIZE)

    # Store result
    ct.scatter(y, offsets, gelu_output)


# Wrapper class for autograd integration
class GeLU_CT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, approximate):
        """
        Forward pass for GELU activation.

        Args:
            x: Input tensor
            approximate: 'none' for exact, 'tanh' for approximation

        Returns:
            Output tensor with GELU applied
        """
        # Convert string to integer enum
        approx_mode = GELU_TANH if approximate == "tanh" else GELU_EXACT

        # Allocate output
        y = torch.empty_like(x)
        n_elements = y.numel()

        # Launch kernel
        BLOCK_SIZE = 1024
        grid = (math.ceil(n_elements / BLOCK_SIZE), 1, 1)

        # Reshape to 1D for processing
        x_flat = x.view(-1)
        y_flat = y.view(-1)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            gelu_kernel_ct,
            (y_flat, x_flat, n_elements, BLOCK_SIZE, approx_mode),
        )

        ctx.x = x
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Backward pass for GELU activation is not implemented")


@register_impl("gelu", backend="cutile")
def gelu(input: torch.Tensor, approximate="none"):
    """
    cuTile implementation of GELU activation function.

    Args:
        input: Input tensor
        approximate: 'none' for exact GELU, 'tanh' for tanh approximation

    Returns:
        Tensor with GELU activation applied
    """
    return GeLU_CT.apply(input, approximate)
