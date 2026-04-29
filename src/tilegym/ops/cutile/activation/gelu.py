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


def _sigmoid(x_val, BLOCK_SIZE: ct.Constant[int]):
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)
    denom = one + ct.exp(-x_val)
    return one / denom


def _tanh(x_val, BLOCK_SIZE: ct.Constant[int]):
    # tanh(x) = 2 * sigmoid(2*x) - 1
    two = ct.full((BLOCK_SIZE,), 2.0, dtype=x_val.dtype)
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)
    return two * _sigmoid(two * x_val, BLOCK_SIZE) - one


def _normal_cdf(x_val, BLOCK_SIZE: ct.Constant[int]):
    # cdf = 0.5 * (1 + erf(x / sqrt(2)))
    # erf(x) ≈ tanh(sqrt(2/π) * (x + 0.044715 * x^3))
    sqrt_2_div_pi = 0.7978845608028654
    coeff_044715 = 0.044715
    half = ct.full((BLOCK_SIZE,), 0.5, dtype=x_val.dtype)
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)
    c1 = ct.full((BLOCK_SIZE,), sqrt_2_div_pi, dtype=x_val.dtype)
    c2 = ct.full((BLOCK_SIZE,), coeff_044715, dtype=x_val.dtype)

    erf_approx = _tanh(c1 * (x_val + c2 * x_val * x_val * x_val), BLOCK_SIZE)
    return half * (one + erf_approx)


def _normal_pdf(x_val, BLOCK_SIZE: ct.Constant[int]):
    # pdf = (1/√(2π)) * exp(-0.5 * x²)
    inv_sqrt_2pi = 0.3989422804014327
    half = ct.full((BLOCK_SIZE,), 0.5, dtype=x_val.dtype)
    c = ct.full((BLOCK_SIZE,), inv_sqrt_2pi, dtype=x_val.dtype)

    # Cast to float32 for exp, then back to input dtype
    exp_val = ct.astype(ct.exp(ct.astype(-(half * x_val * x_val), ct.float32)), x_val.dtype)
    return c * exp_val


def _gelu_tanh_fwd(x_val, BLOCK_SIZE: ct.Constant[int]):
    # f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    sqrt_2_div_pi = 0.7978845608028654
    coeff_044715 = 0.044715
    half = ct.full((BLOCK_SIZE,), 0.5, dtype=x_val.dtype)
    one = ct.ones((BLOCK_SIZE,), dtype=x_val.dtype)
    c1 = ct.full((BLOCK_SIZE,), sqrt_2_div_pi, dtype=x_val.dtype)
    c2 = ct.full((BLOCK_SIZE,), coeff_044715, dtype=x_val.dtype)

    tanh_val = _tanh(c1 * (x_val + c2 * x_val * x_val * x_val), BLOCK_SIZE)
    return half * x_val * (one + tanh_val)


def _gelu_fwd(x_val, BLOCK_SIZE: ct.Constant[int]):
    # f(x) = x * Φ(x)
    return x_val * _normal_cdf(x_val, BLOCK_SIZE)


@ct.kernel
def gelu_fwd_kernel(
    y,
    x,
    n_elements: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    APPROXIMATE: ct.Constant[int],
):
    pid = ct.bid(0)
    offsets = ct.arange(BLOCK_SIZE, dtype=ct.int32) + pid * BLOCK_SIZE
    x_tile = ct.gather(x, offsets, padding_value=0)

    if APPROXIMATE == GELU_TANH:
        out = _gelu_tanh_fwd(x_tile, BLOCK_SIZE)
    else:
        out = _gelu_fwd(x_tile, BLOCK_SIZE)

    ct.scatter(y, offsets, out, check_bounds=True)


class GeluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, approximate):
        approx_mode = GELU_TANH if approximate == "tanh" else GELU_EXACT

        y = torch.empty_like(x)
        n_elements = y.numel()
        BLOCK_SIZE = 1024
        grid = (math.ceil(n_elements / BLOCK_SIZE), 1, 1)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            gelu_fwd_kernel,
            (y.view(-1), x.view(-1), n_elements, BLOCK_SIZE, approx_mode),
        )

        ctx.x = x
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Backward pass for GELU activation is not implemented")


@register_impl("gelu", backend="cutile")
def gelu(input: torch.Tensor, approximate="none"):
    r"""
    Returns GELU activation of input.

    $GELU(x) = x * \Phi(x)$ (``approximate='none'``)
    $GELU(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))$ (``approximate='tanh'``)

    Args:
        input: Tensor
        approximate: ``'none'`` or ``'tanh'``
    """
    return GeluFunction.apply(input, approximate)
