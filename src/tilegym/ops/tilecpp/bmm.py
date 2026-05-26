# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Batch Matrix Multiplication
Computes C = A x B for batched 3D tensors using CUDA C++ tile kernels.

A has shape (Q, M, K), B has shape (Q, K, N), C has shape (Q, M, N)
Supports transpose of A and/or B.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import get_cpp_type
from tilegym.ops.tilecpp.utils._cuda_utils import make_kernel_args
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)

# =============================================================================
# Kernel Definition
# =============================================================================

_bmm_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "bmm.cuh",
    kernel_name="bmm_kernel",
)

_bmm_static_persistent_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "bmm.cuh",
    kernel_name="bmm_static_persistent_kernel",
)

# Default block sizes
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
GROUP_SIZE_M = 8

# =============================================================================
# Kernel Launch Function
# =============================================================================


def _get_bmm_kernel(
    dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    transpose_a: bool,
    transpose_b: bool,
):
    """Get compiled kernel for specific configuration."""
    bool_to_str = lambda b: "true" if b else "false"

    kernel, mangled_name, _ = _bmm_kernel.get_kernel(
        dtype=dtype,
        template_params=[
            block_m,
            block_n,
            block_k,
            group_m,
            bool_to_str(transpose_a),
            bool_to_str(transpose_b),
        ],
        signature=("const {T}*, const {T}*, {T}*, int, int, int, int"),
    )
    return kernel


def _launch_bmm_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    Q: int,
    M: int,
    N: int,
    K: int,
    transpose_a: bool,
    transpose_b: bool,
    block_m: int = BLOCK_SIZE_M,
    block_n: int = BLOCK_SIZE_N,
    block_k: int = BLOCK_SIZE_K,
    group_m: int = GROUP_SIZE_M,
):
    """Launch the BMM kernel."""
    dump_kernel_types("bmm_kernel", a, b, c, M, N)
    dtype = a.dtype

    kernel = _get_bmm_kernel(
        dtype,
        block_m,
        block_n,
        block_k,
        group_m,
        transpose_a,
        transpose_b,
    )

    # Grid dimensions: (num_m_blocks * num_n_blocks, Q)
    num_pid_m = (M + block_m - 1) // block_m
    num_pid_n = (N + block_n - 1) // block_n
    grid = (num_pid_m * num_pid_n, Q)

    _bmm_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(a.data_ptr()),
            np.uint64(b.data_ptr()),
            np.uint64(c.data_ptr()),
            np.int32(Q),
            np.int32(M),
            np.int32(N),
            np.int32(K),
        ],
    )


def _get_bmm_static_persistent_kernel(
    dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    transpose_a: bool,
    transpose_b: bool,
    Q: int,
    M: int,
    N: int,
    K: int,
    num_ctas: int,
    occupancy: int,
):
    """Get compiled static persistent kernel for specific configuration."""
    bool_to_str = lambda b: "true" if b else "false"

    kernel, mangled_name, _ = _bmm_static_persistent_kernel.get_kernel(
        dtype=dtype,
        template_params=[
            block_m,
            block_n,
            block_k,
            group_m,
            bool_to_str(transpose_a),
            bool_to_str(transpose_b),
            Q,
            M,
            N,
            K,
            num_ctas,
            occupancy,
        ],
        signature=("const {T}*, const {T}*, {T}*"),
    )
    return kernel


def _launch_bmm_static_persistent_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    Q: int,
    M: int,
    N: int,
    K: int,
    transpose_a: bool,
    transpose_b: bool,
    block_m: int = BLOCK_SIZE_M,
    block_n: int = BLOCK_SIZE_N,
    block_k: int = BLOCK_SIZE_K,
    group_m: int = GROUP_SIZE_M,
    occupancy: int = 1,
    num_ctas: int = 1,
):
    """Launch the static persistent BMM kernel."""
    dump_kernel_types("bmm_static_persistent_kernel", a, b, c, M, N)
    dtype = a.dtype

    kernel = _get_bmm_static_persistent_kernel(
        dtype,
        block_m,
        block_n,
        block_k,
        group_m,
        transpose_a,
        transpose_b,
        Q,
        M,
        N,
        K,
        num_ctas,
        occupancy,
    )

    # Calculate grid size for static persistent scheduling
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_tiles_m = (M + block_m - 1) // block_m
    num_tiles_n = (N + block_n - 1) // block_n
    total_tiles = num_tiles_m * num_tiles_n * Q

    # min(NUM_SMS // num_ctas, total_tiles) * occupancy
    base_programs = NUM_SMS // num_ctas
    grid_size = min(base_programs, total_tiles) * occupancy
    grid = (grid_size,)

    _bmm_static_persistent_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(a.data_ptr()),
            np.uint64(b.data_ptr()),
            np.uint64(c.data_ptr()),
        ],
    )


# =============================================================================
# Public Interface
# =============================================================================


def bmm_fn(a, b, transpose_a=False, transpose_b=False):
    """
    Non-TMA batch matrix multiplication.

    """
    if transpose_a:
        Q_A, K_A, M = a.shape
    else:
        Q_A, M, K_A = a.shape

    if transpose_b:
        Q_B, N, K_B = b.shape
    else:
        Q_B, K_B, N = b.shape

    assert K_A == K_B, "incompatible dimensions"
    assert Q_A == Q_B, "incompatible dimensions"
    K = K_A
    Q = Q_A

    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    # Allocate output
    c = torch.empty((Q, M, N), device=a.device, dtype=a.dtype)

    _launch_bmm_kernel(
        a,
        b,
        c,
        Q,
        M,
        N,
        K,
        transpose_a,
        transpose_b,
    )

    return c


def bmm_memref(a, b, transpose_a=False, transpose_b=False, static_persistent=True, **kwargs):
    """
    TMA-style batch matrix multiplication (memref interface).

    By default, uses static persistent scheduling for better performance.

    Args:
        a: Input tensor A with shape (Q, M, K) or (Q, K, M) if transposed
        b: Input tensor B with shape (Q, K, N) or (Q, N, K) if transposed
        transpose_a: Whether to transpose A
        transpose_b: Whether to transpose B
        static_persistent: Whether to use static persistent scheduling (default: True)
        **kwargs: Additional kernel configuration parameters passed to bmm_static_persistent

    Returns:
        Output tensor C with shape (Q, M, N)
    """
    if static_persistent:
        # Use static persistent kernel by default
        return bmm_static_persistent(a, b, transpose_a=transpose_a, transpose_b=transpose_b, **kwargs)
    else:
        # Fall back to non-persistent kernel
        if transpose_a:
            Q_A, K_A, M = a.shape
        else:
            Q_A, M, K_A = a.shape

        if transpose_b:
            Q_B, N, K_B = b.shape
        else:
            Q_B, K_B, N = b.shape

        assert K_A == K_B, "incompatible dimensions"
        assert Q_A == Q_B, "incompatible dimensions"
        K = K_A
        Q = Q_A

        assert a.is_contiguous(), "matrix A must be contiguous"
        assert b.is_contiguous(), "matrix B must be contiguous"

        c = torch.empty((Q, M, N), device=a.device, dtype=a.dtype)

        _launch_bmm_kernel(
            a,
            b,
            c,
            Q,
            M,
            N,
            K,
            transpose_a,
            transpose_b,
        )

        return c


class BMM(torch.autograd.Function):
    """Autograd function for BMM with backward pass."""

    @staticmethod
    def forward(ctx, a, b):
        c = bmm_memref(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        da = bmm_memref(dy, b, transpose_b=True)
        db = bmm_memref(a, dy, transpose_a=True)
        return da, db


def bmm_memref_fwd_bwd(a, b):
    """BMM with forward and backward pass support."""
    return BMM.apply(a, b)


def bmm_static_persistent(a, b, transpose_a=False, transpose_b=False, **kwargs):
    """
    Static persistent batch matrix multiplication.

    Uses static persistent scheduling with GPU-specific default configurations:
    - B200 (sm_120/121): TILE_M=128, TILE_N=128, TILE_K=64, occupancy=2, num_ctas=1
    - H100 (sm_90): TILE_M=128, TILE_N=128, TILE_K=64, occupancy=1, num_ctas=1
    - GB100 (sm_100): TILE_M=256, TILE_N=256, TILE_K=64, occupancy=1, num_ctas=2

    Args:
        a: Input tensor A with shape (Q, M, K) or (Q, K, M) if transposed
        b: Input tensor B with shape (Q, K, N) or (Q, N, K) if transposed
        transpose_a: Whether to transpose A
        transpose_b: Whether to transpose B
        **kwargs: Optional kernel configuration parameters:
            - TILE_M: M-dimension tile size (default: GPU-dependent)
            - TILE_N: N-dimension tile size (default: GPU-dependent)
            - TILE_K: K-dimension tile size (default: 64)
            - GROUP_SIZE_M: Number of M-tiles to group (default: 8)
            - occupancy: Occupancy multiplier (default: GPU-dependent)
            - num_ctas: Number of CTAs per kernel (default: GPU-dependent)

    Returns:
        Output tensor C with shape (Q, M, N)
    """
    if transpose_a:
        Q_A, K_A, M = a.shape
    else:
        Q_A, M, K_A = a.shape

    if transpose_b:
        Q_B, N, K_B = b.shape
    else:
        Q_B, K_B, N = b.shape

    assert K_A == K_B, "incompatible dimensions"
    assert Q_A == Q_B, "incompatible dimensions"
    K = K_A
    Q = Q_A

    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    capability = torch.cuda.get_device_capability()
    if capability in [(12, 0), (12, 1)]:
        # B200: Smaller tiles, num_ctas=1, higher occupancy
        default_config = {
            "TILE_M": 128,
            "TILE_N": 128,
            "TILE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 2,
        }
    elif capability == (9, 0):
        # H100: Medium tiles
        default_config = {
            "TILE_M": 128,
            "TILE_N": 128,
            "TILE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
        }
    else:
        # Other GPUs (e.g., GB100): Larger tiles, num_ctas=2
        default_config = {
            "TILE_M": 256,
            "TILE_N": 256,
            "TILE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 2,
            "occupancy": 1,
        }

    # Override defaults with any user-provided configs
    config = {**default_config, **kwargs}

    c = torch.empty((Q, M, N), device=a.device, dtype=a.dtype)

    _launch_bmm_static_persistent_kernel(
        a,
        b,
        c,
        Q,
        M,
        N,
        K,
        transpose_a,
        transpose_b,
        block_m=config["TILE_M"],
        block_n=config["TILE_N"],
        block_k=config["TILE_K"],
        group_m=config["GROUP_SIZE_M"],
        occupancy=config["occupancy"],
        num_ctas=config["num_ctas"],
    )

    return c


# Register the implementation
register_impl("bmm", "tilecpp")(bmm_memref)
