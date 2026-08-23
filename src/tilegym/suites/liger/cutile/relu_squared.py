# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
ReLU-squared activation kernel (CuTile backend).

Formula: y = relu(x)^2 = max(x, 0)^2

Row-parallel: grid = (n_rows, 1, 1). Each block handles one row, looping over
column chunks of BLOCK_SIZE.

PERF NOTES
==========
- Pure element-wise, memory-bandwidth-bound. No exp/reduction → no occupancy=1
  or exp2 tricks needed (unlike swiglu). Backward writes a fresh DX (not in-place),
  so there is no scatter-in-loop hang hazard.
- Two kernel variants per direction:
    *_aligned: check_bounds=False — used when n_cols % BLOCK_SIZE == 0 (all power-of-2
               n_cols up to MAX_FUSED_SIZE). Faster: no per-lane bounds masking.
    *_ct:      check_bounds=True  — fallback for non-aligned n_cols.
- Compute in float32, cast to the output dtype on scatter.
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

MAX_FUSED_SIZE = 4096  # tile cap; relu is register-light so a large tile is safe


@ct.kernel
def _relu_squared_fwd_ct_aligned(
    X,  # (n_rows, n_cols) input
    Y,  # (n_rows, n_cols) output
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    """ReLU-squared forward — aligned fast path (check_bounds=False)."""
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        x = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=False, padding_value=0.0), ct.float32)
        relu_x = ct.maximum(x, 0.0)
        y = relu_x * relu_x
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y, Y.dtype), check_bounds=False)


@ct.kernel
def _relu_squared_fwd_ct(
    X,  # (n_rows, n_cols) input
    Y,  # (n_rows, n_cols) output
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    """ReLU-squared forward — general path (check_bounds=True) for arbitrary n_cols."""
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        x = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        relu_x = ct.maximum(x, 0.0)
        y = relu_x * relu_x
        ct.scatter(Y, (row_idx, col_idx), ct.astype(y, Y.dtype), check_bounds=True)


@ct.kernel
def _relu_squared_bwd_ct_aligned(
    DY,  # (n_rows, n_cols) upstream gradient
    X,  # (n_rows, n_cols) saved input
    DX,  # (n_rows, n_cols) output gradient
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    """ReLU-squared backward — aligned fast path (check_bounds=False).

    d/dx[relu(x)^2] = 2 * relu(x)  →  dx = dy * 2 * max(x, 0)
    """
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        dy = ct.astype(ct.gather(DY, (row_idx, col_idx), check_bounds=False, padding_value=0.0), ct.float32)
        x = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=False, padding_value=0.0), ct.float32)
        relu_x = ct.maximum(x, 0.0)
        dx = dy * 2.0 * relu_x
        ct.scatter(DX, (row_idx, col_idx), ct.astype(dx, DX.dtype), check_bounds=False)


@ct.kernel
def _relu_squared_bwd_ct(
    DY,  # (n_rows, n_cols) upstream gradient
    X,  # (n_rows, n_cols) saved input
    DX,  # (n_rows, n_cols) output gradient
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
):
    """ReLU-squared backward — general path (check_bounds=True) for arbitrary n_cols."""
    row_idx = ct.bid(0)
    n_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    for ci in range(n_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        dy = ct.astype(ct.gather(DY, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        x = ct.astype(ct.gather(X, (row_idx, col_idx), check_bounds=True, padding_value=0.0), ct.float32)
        relu_x = ct.maximum(x, 0.0)
        dx = dy * 2.0 * relu_x
        ct.scatter(DX, (row_idx, col_idx), ct.astype(dx, DX.dtype), check_bounds=True)


def _calculate_block_size(n_cols):
    # Cap the tile at MAX_FUSED_SIZE (or next_pow2(n_cols) if smaller).
    block = max(min(next_power_of_2(n_cols), MAX_FUSED_SIZE), 128)
    # Largest power-of-2 tile <= block that evenly divides n_cols → enables the
    # check_bounds=False aligned fast path (dispatch selects it when n_cols % block == 0).
    aligned = block
    while aligned > 128 and n_cols % aligned != 0:
        aligned //= 2
    # Prefer the aligned block only when it stays large (>= half the cap); otherwise keep the
    # full block and let the masked (check_bounds=True) kernel cover the remainder in fewer chunks.
    if n_cols % aligned == 0 and aligned >= block // 2:
        return aligned
    return block


class ReLUSquaredCuTileFunction(torch.autograd.Function):
    """CuTile autograd wrapper for ReLU-squared: y = relu(x)^2."""

    @staticmethod
    def forward(ctx, x):
        ori_shape = x.shape
        n_cols = ori_shape[-1]
        x = x.contiguous().view(-1, n_cols)
        n_rows = x.shape[0]

        y = torch.empty_like(x)
        BLOCK_SIZE = _calculate_block_size(n_cols)
        fwd_kernel = _relu_squared_fwd_ct_aligned if n_cols % BLOCK_SIZE == 0 else _relu_squared_fwd_ct

        ct.launch(
            torch.cuda.current_stream(),
            (n_rows, 1, 1),
            fwd_kernel,
            (x, y, int(n_cols), int(BLOCK_SIZE)),
        )
        ctx.save_for_backward(x)
        ctx.ori_shape = ori_shape
        return y.view(*ori_shape)

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        ori_shape = ctx.ori_shape
        n_cols = ori_shape[-1]
        dy = dy.contiguous().view(-1, n_cols)
        n_rows = dy.shape[0]

        dx = torch.empty_like(dy)
        BLOCK_SIZE = _calculate_block_size(n_cols)
        bwd_kernel = _relu_squared_bwd_ct_aligned if n_cols % BLOCK_SIZE == 0 else _relu_squared_bwd_ct

        ct.launch(
            torch.cuda.current_stream(),
            (n_rows, 1, 1),
            bwd_kernel,
            (dy, x, dx, int(n_cols), int(BLOCK_SIZE)),
        )
        return dx.view(*ori_shape)


@register_impl("liger.relu_squared", backend="cutile")
def relu_squared(input: torch.Tensor) -> torch.Tensor:
    """
    ReLU-squared activation: y = relu(x)^2 = max(x, 0)^2.

    Args:
        input: Input tensor of shape (*, N).

    Returns:
        Output tensor of same shape as input.
    """
    return ReLUSquaredCuTileFunction.apply(input)
