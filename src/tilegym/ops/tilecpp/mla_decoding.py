# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ MLA (Multi-head Latent Attention) Decoding Operation
Implements MLA decoding forward pass using CUDA C++ tile kernels.
"""

import math
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def _cdiv(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


# Define kernels
_naive_absorb_mla_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "mla_decoding.cuh",
    kernel_name="naive_absorb_mla",
)

_naive_absorb_mla_transpose_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "mla_decoding.cuh",
    kernel_name="naive_absorb_mla_transpose",
)


def _launch_mla_decoding_kernel(
    q: torch.Tensor,
    qpe: torch.Tensor,
    kv: torch.Tensor,
    kpe: torch.Tensor,
    out: torch.Tensor,
    l: torch.Tensor,
    sm_scale: float,
    transpose: bool,
    BLOCK_D: int,
    BLOCK_H: int,
    BLOCK_N: int,
    BLOCK_KPE: int,
):
    """Launch the MLA decoding CUDA kernel."""
    dtype = q.dtype
    B, num_head, _ = q.shape
    S_kv = kv.shape[1]

    dump_kernel_types("naive_absorb_mla" if not transpose else "naive_absorb_mla_transpose", q, qpe, kv, kpe, out)

    # Select kernel
    kernel_wrapper = _naive_absorb_mla_transpose_kernel if transpose else _naive_absorb_mla_kernel

    # Get kernel with template parameters
    kernel, _, _ = kernel_wrapper.get_kernel(
        dtype=dtype,
        template_params=[BLOCK_D, BLOCK_H, BLOCK_N, BLOCK_KPE],
        signature="{T}*, {T}*, {T}*, {T}*, {T}*, float*, float, int, int, int, int, int, int, int, int, int, int, int, int, int",
    )

    # Calculate grid
    grid_x = _cdiv(num_head, BLOCK_H)
    grid_y = B

    # Launch kernel with 2D grid
    kernel_wrapper.launch(
        grid=(grid_x, grid_y),
        kernel=kernel,
        args=[
            np.uint64(q.data_ptr()),
            np.uint64(qpe.data_ptr()),
            np.uint64(kv.data_ptr()),
            np.uint64(kpe.data_ptr()),
            np.uint64(out.data_ptr()),
            np.uint64(l.data_ptr()),
            np.float32(sm_scale),
            np.int32(q.stride(0)),
            np.int32(q.stride(1)),
            np.int32(qpe.stride(0)),
            np.int32(qpe.stride(1)),
            np.int32(kv.stride(0)),
            np.int32(kv.stride(1)),
            np.int32(kpe.stride(0)),
            np.int32(kpe.stride(1)),
            np.int32(out.stride(0)),
            np.int32(out.stride(1)),
            np.int32(B),
            np.int32(num_head),
            np.int32(S_kv),
        ],
    )


class _mla_decoding(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        qpe,
        kv,
        kpe,
        sm_scale,
        transpose,
    ):
        # Setup stride and shape
        B, num_head, BLOCK_D = q.shape
        BLOCK_KPE = kpe.shape[2]
        S_kv = kv.shape[1]

        o = torch.empty_like(q)
        l = torch.empty((B, num_head), device=q.device, dtype=torch.float32).contiguous()

        # Choose block sizes - use power of 2 values
        # BLOCK_D and BLOCK_KPE are determined by input dimensions
        # BLOCK_H and BLOCK_N are tuning parameters
        BLOCK_H = min(32, _next_power_of_2(num_head))
        BLOCK_N = 64  # Default block size for sequence dimension

        _launch_mla_decoding_kernel(
            q=q,
            qpe=qpe,
            kv=kv,
            kpe=kpe,
            out=o,
            l=l,
            sm_scale=sm_scale,
            transpose=transpose,
            BLOCK_D=BLOCK_D,
            BLOCK_H=BLOCK_H,
            BLOCK_N=BLOCK_N,
            BLOCK_KPE=BLOCK_KPE,
        )
        return o, l

    @staticmethod
    def backward(ctx, do, dl):
        raise NotImplementedError("MLA decoding backward is not implemented")


class MLADecoding:
    def __init__(self, transpose):
        self.transpose = transpose

    def __call__(self, q, qpe, kv, kpe, sm_scale):
        if sm_scale is None:
            sm_scale = 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))
        o, l = _mla_decoding.apply(
            q,
            qpe,
            kv,
            kpe,
            sm_scale,
            self.transpose,
        )
        return o, l


@register_impl("mla_decoding", backend="tilecpp")
def mla_decoding(
    q: torch.Tensor,
    qpe: torch.Tensor,
    kv: torch.Tensor,
    kpe: torch.Tensor,
    sm_scale: float,
    transpose: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Multi-head Latent Attention (MLA) Decoding.

    Computes: QK = Q @ K^T + QPE @ KPE^T, then softmax and V matmul.

    Args:
        q: Query tensor [B, num_head, D]
        qpe: Query position encoding [B, num_head, KPE]
        kv: Key-Value tensor [B, S_kv, D]
        kpe: Key position encoding [B, S_kv, KPE]
        sm_scale: Softmax scale factor; if None, defaults to 1/sqrt(D + KPE)
        transpose: Whether to use transposed kernel variant

    Returns:
        Output tensor and log-sum-exp values
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1) + qpe.size(-1))
    return _mla_decoding.apply(q, qpe, kv, kpe, sm_scale, transpose)
