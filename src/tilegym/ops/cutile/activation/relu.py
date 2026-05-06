# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import cuda.tile as ct
import torch

from tilegym.backend import register_impl


@ct.kernel
def _relu_fwd_kernel(x, y, N_ELEMENTS: ct.Constant[int], BLOCK_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    block_start = pid * BLOCK_SIZE
    offsets_base = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    offsets = block_start + offsets_base
    # For 1D arrays, indices are passed directly (not as tuple)
    # Use padding_value=0 (int) to avoid dtype mismatch with float16
    x_tile = ct.gather(x, offsets, padding_value=0)

    # Convert to float32 for computation
    x_f32 = ct.astype(x_tile, ct.float32)
    zeros = ct.zeros((BLOCK_SIZE,), dtype=ct.float32)

    # Compute ReLU: max(0, x)
    y_f32 = ct.maximum(x_f32, zeros)
    y_tile = ct.astype(y_f32, x_tile.dtype)
    ct.scatter(y, offsets, y_tile)


class _ReluCuTileFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.x = x
        y = torch.empty_like(x)

        assert x.is_contiguous()

        # Flatten to 1D for gather/scatter operations
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)

        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _relu_fwd_kernel,
            (x_flat, y_flat, n_elements, BLOCK_SIZE),
        )

        return y

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Backward pass for ReLU activation is not implemented")


@register_impl("relu", backend="cutile")
def relu(x):
    """Returns ReLU activation of x using cuTile kernels."""
    return _ReluCuTileFunction.apply(x)
