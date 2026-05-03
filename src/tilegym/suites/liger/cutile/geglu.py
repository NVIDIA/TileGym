# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
GEGLU activation kernel (CuTile backend).

Computes: c = GELU(a) * b  using the tanh approximation of GELU:
  GELU(a) = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3)))

Row-parallel: grid = (n_rows, 1, 1). Each block handles one row,
looping over column chunks of size BLOCK_SIZE.
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

MAX_FUSED_SIZE_FWD = 4096  # Forward: tile size cap; aligned path activates when n_cols % BLOCK_SIZE == 0
MAX_FUSED_SIZE_BWD = 512  # Backward: recomputes tanh + two gradient terms — larger tile causes register spill
SQRT_2_OVER_PI = 0.7978845608028654


@ct.kernel(occupancy=1)
def _geglu_fwd_ct(
    A,  # (n_rows, n_cols) input a
    B,  # (n_rows, n_cols) input b
    C,  # (n_rows, n_cols) output c
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    CHECK_BOUNDS: ct.Constant[bool],
):
    """GEGLU forward. CHECK_BOUNDS=False (aligned path) is ~17-20% faster on B200."""
    row_idx = ct.bid(0)

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        a = ct.astype(ct.gather(A, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0), ct.float32)
        b = ct.gather(B, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0)

        a_sq = a * a
        tanh_arg = SQRT_2_OVER_PI * (a + 0.044715 * a_sq * a)
        tanh_result = ct.tanh(tanh_arg)
        geglu_a = 0.5 * a * (1.0 + tanh_result)

        c = ct.astype(geglu_a, b.dtype) * b
        ct.scatter(C, (row_idx, col_idx), c, check_bounds=CHECK_BOUNDS)


@ct.kernel
def _geglu_bwd_ct(
    DC,  # (n_rows, n_cols) upstream gradient
    A,  # (n_rows, n_cols) saved input a — DA written in-place
    B,  # (n_rows, n_cols) saved input b — DB written in-place
    n_cols: ct.Constant[int],
    BLOCK_SIZE: ct.Constant[int],
    CHECK_BOUNDS: ct.Constant[bool],
):
    """
    GEGLU backward. CHECK_BOUNDS=False (aligned path) is faster on B200.
    """
    row_idx = ct.bid(0)

    num_chunks = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    for ci in range(num_chunks):
        col_idx = ct.add(ct.arange(BLOCK_SIZE, dtype=ct.int32), ci * BLOCK_SIZE)

        dc = ct.astype(ct.gather(DC, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0), ct.float32)
        a = ct.astype(ct.gather(A, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0), ct.float32)
        b = ct.astype(ct.gather(B, (row_idx, col_idx), check_bounds=CHECK_BOUNDS, padding_value=0.0), ct.float32)

        # CSE: term1 = 0.5*(1+tanh), geglu_a = term1*a — shared between db and da
        a_sq = a * a
        tanh_arg = SQRT_2_OVER_PI * (a + 0.044715 * a_sq * a)
        tanh_result = ct.tanh(tanh_arg, rounding_mode=ct.RoundingMode.APPROX)
        term1 = 0.5 * (1.0 + tanh_result)
        geglu_a = term1 * a

        db = dc * geglu_a

        # da = dc * b * (term1 + 0.5*a*(1-tanh^2)*sqrt(2/pi)*(1+3*0.044715*a^2))
        tanh_sq = tanh_result * tanh_result
        term2 = 0.5 * a * (1.0 - tanh_sq) * (SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * a_sq))
        da = dc * b * (term1 + term2)

        ct.scatter(A, (row_idx, col_idx), ct.astype(da, A.dtype), check_bounds=CHECK_BOUNDS)
        ct.scatter(B, (row_idx, col_idx), ct.astype(db, B.dtype), check_bounds=CHECK_BOUNDS)


def _calculate_block_size(n_cols, max_fused_size):
    BLOCK_SIZE = next_power_of_2(n_cols)
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, max_fused_size)
    # If BLOCK_SIZE divides n_cols exactly, all chunks are full → check_bounds=False (fast path).
    # Otherwise use it as-is with check_bounds=True for the partial last chunk.
    return BLOCK_SIZE


class GEGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ori_shape = a.shape
        n_cols = ori_shape[-1]
        a = a.view(-1, n_cols)
        b = b.view(-1, n_cols)

        if not a.is_contiguous():
            a = a.contiguous()
        if not b.is_contiguous():
            b = b.contiguous()

        n_rows = a.shape[0]
        c = torch.empty_like(a)
        BLOCK_SIZE = _calculate_block_size(n_cols, MAX_FUSED_SIZE_FWD)
        aligned = n_cols % BLOCK_SIZE == 0

        grid = (n_rows, 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _geglu_fwd_ct,
            (a, b, c, int(n_cols), int(BLOCK_SIZE), not aligned),
        )
        ctx.save_for_backward(a, b)
        ctx.ori_shape = ori_shape
        return c.view(*ori_shape)

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        ori_shape = ctx.ori_shape
        n_cols = ori_shape[-1]
        dc = dc.view(-1, n_cols).contiguous()
        n_rows = dc.shape[0]
        BLOCK_SIZE = _calculate_block_size(n_cols, MAX_FUSED_SIZE_BWD)
        aligned = n_cols % BLOCK_SIZE == 0

        grid = (n_rows, 1, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _geglu_bwd_ct,
            (dc, a, b, int(n_cols), int(BLOCK_SIZE), not aligned),
        )
        return a.view(*ori_shape), b.view(*ori_shape)


@register_impl("liger.geglu", backend="cutile")
def geglu(
    a: torch.Tensor,
    b: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    return GEGLUFunction.apply(a, b)
