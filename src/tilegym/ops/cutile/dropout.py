# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl


@ct.kernel
def dropout_kernel_ct(
    x,
    output,
    p: ct.Constant[float],
    seed: ct.Constant[int],
    TILE_SIZE: ct.Constant[int],
    training: ct.Constant[bool],
):
    """
    cuTile kernel for dropout operation.

    Args:
        x: Input tensor
        output: Output tensor
        p: Dropout probability
        seed: Random seed
        TILE_SIZE: Tile size for computation
        training: Whether in training mode
    """
    # Get program ID
    bid = ct.bid(0)
    tile_start = bid * TILE_SIZE

    # Create offset tile
    offsets = ct.add(ct.arange(TILE_SIZE, dtype=ct.int32), tile_start)
    # Load input data using gather
    # For 1D arrays, indices are passed directly (not as tuple)
    # Use padding_value=0 (int) to avoid dtype mismatch with float16
    x_tile = ct.gather(x, offsets, padding_value=0)

    # Initialize output tile
    output_tile = ct.zeros((TILE_SIZE,), dtype=x_tile.dtype)

    # Only apply dropout if training
    if training:
        # Generate pseudo-random numbers using a simple hash function
        # This is a deterministic approximation since cuTile doesn't have tl.rand
        # Use a simple hash based on offsets and seed
        # Combine seed and offsets with a simple formula
        combined = ct.add(
            ct.mul(offsets, 1103515245),  # Large prime number
            ct.full((TILE_SIZE,), seed, dtype=ct.int32),
        )

        # Apply a simple hash function using available bitwise operations
        hash_val = ct.bitwise_xor(combined, ct.bitwise_rshift(combined, 16))
        hash_val = ct.bitwise_xor(hash_val, ct.bitwise_lshift(hash_val, 8))
        hash_val = ct.bitwise_xor(hash_val, ct.bitwise_rshift(hash_val, 4))

        # Convert to float and normalize to [0, 1)
        hash_float = ct.truediv(
            ct.astype(ct.bitwise_and(hash_val, 0x7FFFFFFF), ct.float32),
            2147483647.0,  # 2^31 - 1
        )

        # Create mask for elements to keep
        keep_mask = ct.greater(hash_float, p)

        # Apply dropout: x / (1-p) if kept, 0 otherwise
        scale = ct.full((TILE_SIZE,), 1.0 / (1.0 - p), dtype=x_tile.dtype)
        scaled_x = ct.mul(x_tile, scale)
        output_tile = ct.where(keep_mask, scaled_x, output_tile)
    else:
        # In inference mode, just copy input to output
        output_tile = x_tile

    # Store result
    ct.scatter(output, offsets, output_tile)


class Dropout_CT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, seed, p=0.5, training=True, inplace=False):
        """
        Forward pass for dropout.

        Args:
            x: Input tensor
            seed: Random seed
            p: Dropout probability
            training: Whether in training mode
            inplace: Whether to perform operation in-place

        Returns:
            Output tensor with dropout applied
        """
        if not training:
            ctx.mark_dirty(x)
            return x

        if inplace:
            ctx.mark_dirty(x)
            output = x
        else:
            output = torch.empty_like(x)

        assert x.is_contiguous()

        n_elements = x.numel()

        # Launch kernel
        TILE_SIZE = 1024
        grid = (math.ceil(n_elements / TILE_SIZE), 1, 1)

        # Reshape to 1D for processing
        x_flat = x.view(-1)
        output_flat = output.view(-1)

        # Convert seed to int32 to avoid overflow
        seed_int32 = int(seed) % 2147483647  # Convert to int32 range

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            dropout_kernel_ct,
            (
                x_flat,
                output_flat,
                p,
                seed_int32,
                TILE_SIZE,
                training,
            ),
        )

        ctx.p = p
        ctx.seed = seed
        return output

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Backward pass for dropout is not implemented")


@register_impl("dropout", backend="cutile")
def dropout(x, seed, p=0.5, training=True, inplace=False, **kwargs):
    """
    cuTile implementation of dropout.

    Performs dropout on x.

    Args:
        x: Input tensor
        seed: Integer value for initializing random mask
        p: Dropout probability, default is 0.5
        training: If True perform dropout, else return x
        inplace: If True, modify x directly with dropout
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Tensor with dropout applied
    """
    return Dropout_CT.apply(x, seed, p, training, inplace)
