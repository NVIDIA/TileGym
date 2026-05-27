# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Layer Normalization Legacy
Simple row-wise LayerNorm with forward pass using CUDA C++ tile kernels.
"""

from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types


def _largest_power_of_2_le(n):
    return 1 << (n.bit_length() - 1) if n >= 1 else 1


# Define kernel
_layer_norm_fwd_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "layer_norm_legacy.cuh",
    kernel_name="layer_norm_fwd_fused_kernel",
)


def _launch_layer_norm_fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    N: int,
    eps: float,
    weight_shift: float,
    BLOCK_SIZE: int,
    M: int,
):
    """Launch the layer_norm_fwd_fused_kernel CUDA kernel."""
    dump_kernel_types("layer_norm_fwd_fused_kernel", x, y, weight, bias)
    dtype = x.dtype

    kernel, _, _ = _layer_norm_fwd_kernel.get_kernel(
        dtype=dtype,
        template_params=[N, BLOCK_SIZE],
        signature="{T}*, {T}*, {T}*, {T}*, float*, float*, float, float",
    )

    _layer_norm_fwd_kernel.launch(
        grid=(M, 1, 1),
        kernel=kernel,
        args=[
            np.uint64(x.data_ptr()),
            np.uint64(y.data_ptr()),
            np.uint64(weight.data_ptr()),
            np.uint64(bias.data_ptr()),
            np.uint64(mean.data_ptr()),
            np.uint64(rstd.data_ptr()),
            np.float32(eps),
            np.float32(weight_shift),
        ],
    )


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, weight_shift=0.0):
        # allocate output
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, _largest_power_of_2_le(N))
        if N > MAX_FUSED_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        _launch_layer_norm_fwd(
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,
            N,
            eps,
            weight_shift,
            BLOCK_SIZE,
            M,
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.eps = eps
        ctx.weight_shift = weight_shift
        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("LayerNorm backward is not implemented for tilecpp backend")


@register_impl("layer_norm_legacy", backend="tilecpp")
def layer_norm(input, normalized_shape, weight, bias, eps, weight_shift=0.0, **kwargs):
    r"""
    Returns the LayerNorm of input along dimension N

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Unused
        weight: Tensor of shape (N,)
        bias: Tensor of shape (N,)
        eps: small scaler to be added to
            variance calculation prior to division.
        weight_shift: float value to be added to the weight
        **kwargs: Additional arguments for backend-specific configurations
    """
    return LayerNorm.apply(input, normalized_shape, weight, bias, eps, weight_shift)
