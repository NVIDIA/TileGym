# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
import cuda.tile as ct
import torch

from tilegym.backend import register_impl

# ReLU


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


@ct.kernel
def _relu_bwd_kernel(dy, dx, x, N_ELEMENTS: ct.Constant[int], BLOCK_SIZE: ct.Constant[int]):
    pid = ct.bid(0)
    block_start = pid * BLOCK_SIZE
    offsets_base = ct.arange(BLOCK_SIZE, dtype=ct.int32)
    offsets = block_start + offsets_base
    # For 1D arrays, indices are passed directly (not as tuple)
    # Use padding_value=0 (int) to avoid dtype mismatch with float16
    dy_tile = ct.gather(dy, offsets, padding_value=0)
    x_tile = ct.gather(x, offsets, padding_value=0)

    # Convert to float32 for computation
    dy_f32 = ct.astype(dy_tile, ct.float32)
    x_f32 = ct.astype(x_tile, ct.float32)
    zeros = ct.zeros((BLOCK_SIZE,), dtype=ct.float32)
    ones = ct.ones((BLOCK_SIZE,), dtype=ct.float32)

    # Compute gradient: dy * (x > 0 ? 1 : 0)
    pos_mask = x_f32 > zeros
    dydx = ct.where(pos_mask, ones, zeros)
    dx_f32 = dydx * dy_f32
    dx_tile = ct.astype(dx_f32, dy_tile.dtype)
    ct.scatter(dx, offsets, dx_tile)


# Wrapper Classes
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
        dy_flat = dy.contiguous().view(-1)
        x_flat = ctx.x.contiguous().view(-1)
        dx_flat = torch.empty_like(dy_flat)

        n_elements = dy_flat.numel()
        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _relu_bwd_kernel,
            (dy_flat, dx_flat, x_flat, n_elements, BLOCK_SIZE),
        )
        return dx_flat.view(ctx.x.shape)


# Public API Functions


@register_impl("relu", backend="cutile")
def relu(x):
    """Returns ReLU activation of x using cuTile kernels."""
    return _ReluCuTileFunction.apply(x)
