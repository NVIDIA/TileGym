# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl


@ct.kernel
def _transpose_kernel(
    input,
    output,
    M: ct.Constant[int],
    N: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
):
    """CuTile kernel for transposing a 2D matrix."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    # Load a tile from input at position (pid_m, pid_n)
    # with padding for out-of-bounds accesses
    input_tile = ct.load(input, index=(pid_m, pid_n), shape=(BLOCK_M, BLOCK_N), padding_mode=ct.PaddingMode.ZERO)

    # Transpose the loaded tile
    transposed_tile = ct.transpose(input_tile)

    # Store to output at transposed position (pid_n, pid_m)
    # The transposed tile has shape (BLOCK_N, BLOCK_M)
    ct.store(output, index=(pid_n, pid_m), tile=transposed_tile)


class _Transpose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, out=None):
        M, N = inp.shape
        if out is None:
            out = torch.empty(N, M, device=inp.device, dtype=inp.dtype)

        BLOCK_M, BLOCK_N = 64, 64
        if inp.dtype in (torch.float16, torch.bfloat16) and M >= 4096 and N >= 4096:
            BLOCK_M, BLOCK_N = 128, 128
        grid = (math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N), 1)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _transpose_kernel,
            (
                inp,
                out,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
            ),
        )
        return out

    @staticmethod
    def backward(ctx, dy):
        dx = _Transpose.apply(dy)
        return dx, None


@register_impl("transpose", backend="cutile")
def transpose(inp, out=None, static_persistent=None, **kwargs):
    return _Transpose.apply(inp, out)
