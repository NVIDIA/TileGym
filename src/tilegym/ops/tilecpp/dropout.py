# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Dropout Operation
Seeded dropout with deterministic random mask generation using CUDA C++ tile kernels.
"""

from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

# Block size for kernel
BLOCK_SIZE = 1024

# Define kernel
_seeded_dropout_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "dropout.cuh",
    kernel_name="seeded_dropout_kernel",
)


def _mix_seed(seed: int) -> int:
    """Pre-mix a user-provided seed into a well-spread int32.

    The kernel's internal hash (3 rounds of XOR-shift) does not amplify small
    seed deltas. Multiplying the seed by Knuth's constant 0x9E3779B1
    (== floor(2^32 / phi)) spreads the bits evenly across all 32 positions
    before the XOR-shift so distinct seeds produce well-separated masks.
    """
    return (int(seed) * 2654435761) & 0xFFFFFFFF


def _launch_dropout_kernel(
    x: torch.Tensor,
    output: torch.Tensor,
    n_elements: int,
    p: float,
    seed: int,
    block_size: int = BLOCK_SIZE,
):
    """Launch the seeded_dropout_kernel CUDA kernel."""
    dump_kernel_types("seeded_dropout_kernel", x, output)
    dtype = x.dtype

    if n_elements % block_size != 0:
        raise ValueError(
            f"dropout requires n_elements ({n_elements}) to be a multiple of block_size ({block_size}); "
            f"the kernel does unmasked tile loads/stores of BLOCK_SIZE elements per block."
        )

    num_blocks = (n_elements + block_size - 1) // block_size

    # Template params: BLOCK_SIZE, NUM_BLOCKS
    kernel, _, _ = _seeded_dropout_kernel.get_kernel(
        dtype=dtype,
        template_params=[block_size, num_blocks],
        signature="const {T}*, {T}*, float, uint64_t",
    )

    _seeded_dropout_kernel.launch(
        grid=num_blocks,
        kernel=kernel,
        args=[
            np.uint64(x.data_ptr()),
            np.uint64(output.data_ptr()),
            np.float32(p),
            np.uint64(_mix_seed(seed)),
        ],
    )


class Dropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, seed, p=0.5, training=True, inplace=False):
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
        _launch_dropout_kernel(x, output, n_elements, p, seed)

        ctx.p = p
        ctx.seed = seed
        return output

    @staticmethod
    def backward(ctx, dy):
        p = ctx.p
        seed = ctx.seed
        dx = torch.empty_like(dy)
        n_elements = dy.numel()

        _launch_dropout_kernel(dy, dx, n_elements, p, seed)

        return dx, None, None, None, None


@register_impl("dropout", backend="tilecpp")
def dropout(x, seed, p=0.5, training=True, inplace=False, **kwargs):
    r"""
    Performs dropout on x

    Args:
        seed: Integer value for initializing
            random mask
        training: If True perform dropout, else
            return x
        inplace: If True, modify x directly with
            dropout
        **kwargs: Additional arguments for backend-specific configurations
    """
    return Dropout.apply(x, seed, p, training, inplace)
