# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Split-K Reduce Operation
Reduces intermediate attention results from multiple K splits into final output.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import get_dtype_info
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)

# Define kernel
_splitk_reduce_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "splitk_reduce.cuh",
    kernel_name="splitk_reduce_kernel",
)


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def _launch_splitk_reduce_kernel(
    attn_splitk_out: torch.Tensor,
    lse_splitk_out: torch.Tensor,
    attn_out: torch.Tensor,
    B: int,
    num_heads: int,
    head_dim: int,
    num_kv_splits: int,
    num_kv_splits_pow2: int,
    block_d: int,
):
    """Launch the splitk_reduce_kernel CUDA kernel."""
    dump_kernel_types("splitk_reduce_kernel", attn_splitk_out, lse_splitk_out, attn_out)
    dtype = attn_splitk_out.dtype

    use_dot = num_kv_splits_pow2 >= 16

    # Template params: T, B, NUM_HEADS, HEAD_DIM, NUM_KV_SPLITS, NUM_KV_SPLITS_POW2, BLOCK_D, USE_DOT
    kernel, _, _ = _splitk_reduce_kernel.get_kernel(
        dtype=dtype,
        template_params=[
            B,
            num_heads,
            head_dim,
            num_kv_splits,
            num_kv_splits_pow2,
            block_d,
            "true" if use_dot else "false",
        ],
        signature="const {T}*, const float*, {T}*",
    )

    # 3D grid: (B, num_heads, ceil(head_dim / BLOCK_D))
    grid_z = (head_dim + block_d - 1) // block_d
    grid = (B, num_heads, grid_z)

    _splitk_reduce_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(attn_splitk_out.data_ptr()),
            np.uint64(lse_splitk_out.data_ptr()),
            np.uint64(attn_out.data_ptr()),
        ],
    )


@register_impl("splitk_reduce", backend="tilecpp")
def splitk_reduce(attn_splitk_out, lse_splitk_out, attn_out, S_kv, **kwargs):
    """
    Reduce the intermediate attention results and lse results into the final output for attention decode.

    Args:
        attn_splitk_out: intermediate attention results [B, num_heads, NUM_KV_SPLITS, head_dim]
        lse_splitk_out: intermediate lse results [B, num_heads, NUM_KV_SPLITS] (float32)
        attn_out: final output [B, num_heads, head_dim]
        S_kv: sequence length of the key-value tensor, used for boundary check

    Returns:
        attn_out tensor with reduced attention results
    """
    B, num_heads, NUM_KV_SPLITS, head_dim = attn_splitk_out.shape

    # Calculate block size for head dimension
    BLOCK_D = min(128, _next_power_of_2(head_dim))

    # Calculate next power of 2 for NUM_KV_SPLITS
    NUM_KV_SPLITS_POW2 = _next_power_of_2(NUM_KV_SPLITS)

    _launch_splitk_reduce_kernel(
        attn_splitk_out,
        lse_splitk_out,
        attn_out,
        B,
        num_heads,
        head_dim,
        NUM_KV_SPLITS,
        NUM_KV_SPLITS_POW2,
        BLOCK_D,
    )

    return attn_out
