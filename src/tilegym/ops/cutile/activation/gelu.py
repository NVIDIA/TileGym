# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

# Approximation mode constants
GELU_EXACT = 0
GELU_TANH = 1


def _sigmoid_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # sigmoid(x) = 1 / (1 + exp(-x))
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)
    neg_x = -x_val
    exp_neg_x = ct.exp(neg_x)
    denom = one + exp_neg_x
    return one / denom


def _tanh_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # tanh(x) = 2 * sigmoid(2*x) - 1
    two = ct.full((BLOCK_SIZE,), 2.0, dtype=x_val.dtype)
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)
    two_x = two * x_val
    sigmoid_2x = _sigmoid_ct(two_x, BLOCK_SIZE)
    two_sigmoid = two * sigmoid_2x
    return two_sigmoid - one


def standard_normal_cdf_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # cdf = 0.5 * (1 + erf(x / sqrt(2)))
    # Using tanh approximation for erf: erf(x) ≈ tanh(sqrt(2/π) * (x + 0.044715 * x^3))
    sqrt_2_div_pi = 0.7978845608028654
    coeff_044715 = 0.044715
    half = ct.full((BLOCK_SIZE,), 0.5, dtype=x_val.dtype)
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)
    sqrt_2_div_pi_tensor = ct.full((BLOCK_SIZE,), sqrt_2_div_pi, dtype=x_val.dtype)
    coeff_tensor = ct.full((BLOCK_SIZE,), coeff_044715, dtype=x_val.dtype)

    # Compute erf approximation
    x_cubed = x_val * x_val * x_val
    coeff_x_cubed = coeff_tensor * x_cubed
    inner_sum = x_val + coeff_x_cubed
    scaled_inner = sqrt_2_div_pi_tensor * inner_sum
    erf_approx = _tanh_ct(scaled_inner, BLOCK_SIZE)

    # Compute CDF
    one_plus_erf = one + erf_approx
    return half * one_plus_erf


def standard_normal_pdf_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # pdf = (1/√(2π)) * exp(-0.5 * x²)
    inverse_sqrt_2_pi = 0.3989422804014327
    half = ct.full((BLOCK_SIZE,), 0.5, dtype=x_val.dtype)
    inverse_sqrt_2_pi_tensor = ct.full((BLOCK_SIZE,), inverse_sqrt_2_pi, dtype=x_val.dtype)

    x_squared = x_val * x_val
    neg_half_x_squared = -(half * x_squared)
    # Convert to float32 for exp computation, then back
    neg_half_x_squared_f32 = ct.astype(neg_half_x_squared, ct.float32)
    exp_val = ct.exp(neg_half_x_squared_f32)
    exp_val = ct.astype(exp_val, x_val.dtype)

    return inverse_sqrt_2_pi_tensor * exp_val


def gelu_tanh_forward_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    sqrt_2_div_pi = 0.7978845608028654
    coeff_044715 = 0.044715
    half = ct.full((BLOCK_SIZE,), 0.5, dtype=x_val.dtype)
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)
    sqrt_2_div_pi_tensor = ct.full((BLOCK_SIZE,), sqrt_2_div_pi, dtype=x_val.dtype)
    coeff_tensor = ct.full((BLOCK_SIZE,), coeff_044715, dtype=x_val.dtype)

    x_cubed = x_val * x_val * x_val
    coeff_x_cubed = coeff_tensor * x_cubed
    inner_sum = x_val + coeff_x_cubed
    scaled_inner = sqrt_2_div_pi_tensor * inner_sum
    tanh_val = _tanh_ct(scaled_inner, BLOCK_SIZE)
    one_plus_tanh = one + tanh_val
    half_x = half * x_val

    return half_x * one_plus_tanh


def gelu_forward_ct(x_val, BLOCK_SIZE: ct.Constant[int]):
    # f(x) = x * Φ(x)
    cdf_val = standard_normal_cdf_ct(x_val, BLOCK_SIZE)
    return x_val * cdf_val


@ct.kernel
def _gelu_kernel(
    y,
    x,
    N_ELEMENTS: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    APPROXIMATE: ct.Constant[int],
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
    pid = ct.bid(0)
    block_start = pid * BLOCK_SIZE
    offsets = ct.arange(BLOCK_SIZE, dtype=ct.int32) + block_start

    # Load input data with padding_value to handle out-of-bounds reads safely
    x_tile = ct.gather(x, offsets, padding_value=0)

    # Compute GELU based on approximation mode
    if APPROXIMATE == GELU_TANH:
        gelu_output = gelu_tanh_forward_ct(x_tile, BLOCK_SIZE)
    else:  # GELU_EXACT
        gelu_output = gelu_forward_ct(x_tile, BLOCK_SIZE)

    # Store result with check_bounds to prevent out-of-bounds writes
    ct.scatter(y, offsets, gelu_output, check_bounds=True)


# Wrapper class for autograd integration
class _GeluCuTileFunction(torch.autograd.Function):
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
        approx_mode = GELU_TANH if approximate == "tanh" else GELU_EXACT
        y = torch.empty_like(x)
        n_elements = y.numel()
        BLOCK_SIZE = 1024
        grid = (math.ceil(n_elements / BLOCK_SIZE), 1, 1)
        x_flat = x.view(-1)
        y_flat = y.view(-1)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _gelu_kernel,
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
    return _GeluCuTileFunction.apply(input, approximate)
