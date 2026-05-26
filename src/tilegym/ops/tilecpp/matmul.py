# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Matrix Multiplication with transpose support.

C = A @ B where:
- A is [M, K] or [K, M] if transpose_a
- B is [K, N] or [N, K] if transpose_b
- C is [M, N]

Autotuning:
- Controlled by TILECPP_AUTOTUNE environment variable
- When enabled, searches over tile size configurations
"""

import logging
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.autotuner import Config
from tilegym.ops.tilecpp.autotuner import TileCppAutotuner
from tilegym.ops.tilecpp.autotuner import autotune
from tilegym.ops.tilecpp.autotuner import is_autotuning_enabled
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import make_kernel_args
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration - Best found through sweeping
# =============================================================================

# Default tile sizes (best for 2048x2048+)
DEFAULT_TILE_SIZE_M = 256
DEFAULT_TILE_SIZE_N = 128
DEFAULT_TILE_SIZE_K = 128
DEFAULT_GROUP_SIZE_M = 8


def _get_best_tile_sizes(M: int, N: int, K: int) -> tuple[int, int, int, int]:
    """Select optimal tile sizes based on problem dimensions.

    Returns (tile_m, tile_n, tile_k, group_m)

    """
    return (256, 256, 64, 8)


# =============================================================================
# Kernel Definition
# =============================================================================

_matmul_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "matmul.cuh",
    kernel_name="matmul_kernel",
)

_persistent_matmul_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "persistent_matmul.cuh",
    kernel_name="static_persistent_matmul_kernel",
)


# =============================================================================
# =============================================================================


def _matmul_autotune_configs():
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        configs = [
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=32, num_ctas=1, occupancy=2),
        ]
    else:
        # sm100 (Blackwell)
        configs = [
            Config(TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=4, occupancy=1),
            Config(TILE_SIZE_M=512, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1),
        ]
    return configs


def _persistent_matmul_autotune_configs():
    """Autotune search space for static_persistent_matmul_kernel."""
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        configs = [
            Config(TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=2),
            Config(TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=4),
            Config(TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=2),
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=4),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1),
        ]
    elif gpu_capability[0] < 9:
        # sm80 (A100)
        configs = [
            Config(TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1),
        ]
    else:
        configs = [
            Config(TILE_SIZE_M=128, TILE_SIZE_N=512, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=4, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=2, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=128, GROUP_SIZE_M=8, num_ctas=2, occupancy=1),
            Config(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=4, occupancy=1),
            Config(TILE_SIZE_M=512, TILE_SIZE_N=256, TILE_SIZE_K=32, GROUP_SIZE_M=8, num_ctas=4, occupancy=1),
        ]
    return configs


# =============================================================================
# Kernel Launch Functions
# =============================================================================


def _get_kernel(
    dtype: torch.dtype,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    group_m: int,
    transpose_a: bool,
    transpose_b: bool,
    num_ctas: int = 1,
    occupancy: int = 1,
):
    """Get compiled kernel for specific configuration.

    Template params: T, M, N, K, TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K,
      GROUP_SIZE_M, num_tiles_k, TRANSPOSE_A, TRANSPOSE_B, num_ctas, occupancy
    Signature: const T*, const T*, T* (A, B, C pointers)
    """
    bool_to_str = lambda b: "true" if b else "false"
    num_tiles_k = (K + tile_k - 1) // tile_k

    kernel, mangled_name, _ = _matmul_kernel.get_kernel(
        dtype=dtype,
        template_params=[
            M,
            N,
            K,
            tile_m,
            tile_n,
            tile_k,
            group_m,
            num_tiles_k,
            bool_to_str(transpose_a),
            bool_to_str(transpose_b),
            num_ctas,
            occupancy,
        ],
        signature="const {T}*, const {T}*, {T}*",
    )
    return kernel


def _launch_matmul_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    tile_m: int = DEFAULT_TILE_SIZE_M,
    tile_n: int = DEFAULT_TILE_SIZE_N,
    tile_k: int = DEFAULT_TILE_SIZE_K,
    group_m: int = DEFAULT_GROUP_SIZE_M,
    num_ctas: int = 1,
    occupancy: int = 1,
):
    """Launch the matmul kernel."""
    dump_kernel_types("matmul_kernel", a, b, c)
    dtype = a.dtype

    kernel = _get_kernel(dtype, M, N, K, tile_m, tile_n, tile_k, group_m, transpose_a, transpose_b, num_ctas, occupancy)

    grid_m = ceil(M / tile_m)
    grid_n = ceil(N / tile_n)
    grid = grid_m * grid_n
    _matmul_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(a.data_ptr()),
            np.uint64(b.data_ptr()),
            np.uint64(c.data_ptr()),
        ],
    )


# =============================================================================
# Persistent-matmul launch helpers (static_persistent=True path)
# =============================================================================


def _get_persistent_kernel(
    dtype: torch.dtype,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    group_m: int,
    transpose_a: bool,
    transpose_b: bool,
    num_ctas: int,
    occupancy: int,
):
    """Compile/retrieve the static_persistent_matmul_kernel for a config.

    Template params (kernel source order):
      T, M, N, K, TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, GROUP_SIZE_M,
      TRANSPOSE_A, TRANSPOSE_B, num_ctas, occupancy
    Signature:
      const T*, const T*, T*   (M/N/K baked into the template)
    """
    bool_to_str = lambda b: "true" if b else "false"
    kernel, _, _ = _persistent_matmul_kernel.get_kernel(
        dtype=dtype,
        template_params=[
            M,
            N,
            K,
            tile_m,
            tile_n,
            tile_k,
            group_m,
            bool_to_str(transpose_a),
            bool_to_str(transpose_b),
            num_ctas,
            occupancy,
        ],
        signature="const {T}*, const {T}*, {T}*",
    )
    return kernel


def _persistent_grid(M: int, N: int, tile_m: int, tile_n: int, num_ctas: int, occupancy: int) -> int:
    """

    min(NUM_SMS // num_ctas, ceil(M/TILE_M) * ceil(N/TILE_N)) * occupancy
    """
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_tiles = ceil(M / tile_m) * ceil(N / tile_n)
    return min(num_sms // num_ctas, num_tiles) * occupancy


def _launch_persistent_matmul_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    tile_m: int = DEFAULT_TILE_SIZE_M,
    tile_n: int = DEFAULT_TILE_SIZE_N,
    tile_k: int = DEFAULT_TILE_SIZE_K,
    group_m: int = DEFAULT_GROUP_SIZE_M,
    num_ctas: int = 1,
    occupancy: int = 1,
):
    dump_kernel_types("static_persistent_matmul_kernel", a, b, c)
    kernel = _get_persistent_kernel(
        a.dtype,
        M,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        group_m,
        transpose_a,
        transpose_b,
        num_ctas,
        occupancy,
    )
    grid = _persistent_grid(M, N, tile_m, tile_n, num_ctas, occupancy)
    _persistent_matmul_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(a.data_ptr()),
            np.uint64(b.data_ptr()),
            np.uint64(c.data_ptr()),
        ],
    )


@autotune(search_space=_matmul_autotune_configs())
def tilecpp_autotune_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    autotuner: TileCppAutotuner | None = None,
):
    """Autotuned matmul that searches for optimal tile sizes."""
    key = ("tilecpp_matmul", str(a.dtype), M, N, K, transpose_a, transpose_b)
    named_args = {"M": M, "N": N, "K": K}

    def launch_fn(cfg: Config):
        group_m = cfg.kwargs.get("GROUP_SIZE_M", DEFAULT_GROUP_SIZE_M)
        num_ctas = cfg.num_ctas if cfg.num_ctas is not None else 1
        occupancy = cfg.occupancy if cfg.occupancy is not None else 1
        _launch_matmul_kernel(
            a,
            b,
            c,
            M,
            N,
            K,
            transpose_a,
            transpose_b,
            tile_m=cfg.TILE_SIZE_M,
            tile_n=cfg.TILE_SIZE_N,
            tile_k=cfg.TILE_SIZE_K,
            group_m=group_m,
            num_ctas=num_ctas,
            occupancy=occupancy,
        )

    def grid_fn(args: dict, cfg: Config) -> tuple[int, ...]:
        grid_m = ceil(args["M"] / cfg.TILE_SIZE_M)
        grid_n = ceil(args["N"] / cfg.TILE_SIZE_N)
        return (grid_m * grid_n, 1, 1)

    autotuner(
        torch.cuda.current_stream(),
        key=key,
        launch_fn=launch_fn,
        grid_fn=grid_fn,
        named_args=named_args,
    )

    return c


@autotune(search_space=_persistent_matmul_autotune_configs())
def tilecpp_autotune_persistent_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    transpose_a: bool = False,
    transpose_b: bool = False,
    autotuner: TileCppAutotuner | None = None,
):
    key = ("tilecpp_persistent_matmul", str(a.dtype), M, N, K, transpose_a, transpose_b)
    named_args = {"M": M, "N": N, "K": K}

    def launch_fn(cfg: Config):
        group_m = cfg.kwargs.get("GROUP_SIZE_M", DEFAULT_GROUP_SIZE_M)
        num_ctas = cfg.num_ctas if cfg.num_ctas is not None else 1
        occupancy = cfg.occupancy if cfg.occupancy is not None else 1
        _launch_persistent_matmul_kernel(
            a,
            b,
            c,
            M,
            N,
            K,
            transpose_a,
            transpose_b,
            tile_m=cfg.TILE_SIZE_M,
            tile_n=cfg.TILE_SIZE_N,
            tile_k=cfg.TILE_SIZE_K,
            group_m=group_m,
            num_ctas=num_ctas,
            occupancy=occupancy,
        )

    def grid_fn(args: dict, cfg: Config) -> tuple[int, ...]:
        num_ctas = cfg.num_ctas if cfg.num_ctas is not None else 1
        occupancy = cfg.occupancy if cfg.occupancy is not None else 1
        return (_persistent_grid(args["M"], args["N"], cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, num_ctas, occupancy), 1, 1)

    autotuner(
        torch.cuda.current_stream(),
        key=key,
        launch_fn=launch_fn,
        grid_fn=grid_fn,
        named_args=named_args,
    )

    return c


# =============================================================================
# Python Interface Functions
# =============================================================================


def matmul_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    kernel_configs: dict = None,
):
    """
    Simple matrix multiplication: C = A @ B

    Args:
        a: Input tensor A [M, K] or [K, M] if transpose_a
        b: Input tensor B [K, N] or [N, K] if transpose_b
        transpose_a: If True, A is stored as [K, M] and will be transposed
        transpose_b: If True, B is stored as [N, K] and will be transposed
        kernel_configs: Optional dict with TILE_SIZE_M, TILE_SIZE_N, TILE_SIZE_K, GROUP_SIZE_M

    Returns:
        Output tensor C of shape (M, N)
    """
    if transpose_a:
        K_A, M = a.shape
    else:
        M, K_A = a.shape

    if transpose_b:
        N, K_B = b.shape
    else:
        K_B, N = b.shape

    assert K_A == K_B, f"incompatible dimensions: K_A={K_A}, K_B={K_B}"
    K = K_A

    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Extract tile sizes from kernel_configs or use size-adaptive defaults
    if kernel_configs:
        tile_m = kernel_configs.get("TILE_SIZE_M", DEFAULT_TILE_SIZE_M)
        tile_n = kernel_configs.get("TILE_SIZE_N", DEFAULT_TILE_SIZE_N)
        tile_k = kernel_configs.get("TILE_SIZE_K", DEFAULT_TILE_SIZE_K)
        group_m = kernel_configs.get("GROUP_SIZE_M", DEFAULT_GROUP_SIZE_M)
    else:
        tile_m, tile_n, tile_k, group_m = _get_best_tile_sizes(M, N, K)

    _launch_matmul_kernel(
        a,
        b,
        c,
        M,
        N,
        K,
        transpose_a,
        transpose_b,
        tile_m,
        tile_n,
        tile_k,
        group_m,
    )

    return c


# =============================================================================
# Autograd Function
# =============================================================================


class Matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, transpose_a=False, transpose_b=False):
        c = matmul_fn(a, b, transpose_a=transpose_a, transpose_b=transpose_b)
        ctx.save_for_backward(a, b)
        ctx.transpose_a = transpose_a
        ctx.transpose_b = transpose_b
        return c

    @staticmethod
    def backward(ctx, dy):
        a, b = ctx.saved_tensors
        transpose_a = ctx.transpose_a
        transpose_b = ctx.transpose_b

        # Compute gradients with appropriate transposes
        # da = dy @ b^T (or handle transposes based on forward transposes)
        # db = a^T @ dy (or handle transposes based on forward transposes)
        if transpose_a:
            # Forward was: c = a^T @ b
            da = matmul_fn(b, dy, transpose_a=transpose_b, transpose_b=True)
        else:
            # Forward was: c = a @ b
            da = matmul_fn(dy, b, transpose_b=not transpose_b)

        if transpose_b:
            # Forward was: c = a @ b^T
            db = matmul_fn(dy, a, transpose_a=True, transpose_b=transpose_a)
        else:
            # Forward was: c = a @ b
            db = matmul_fn(a, dy, transpose_a=not transpose_a)

        return da, db, None, None


# =============================================================================
# =============================================================================


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a=False,
    trans_b=False,
    static_persistent=None,
    use_tma=False,
    **kwargs,
):
    """
    Matrix multiplication with optional transpose: C = A @ B

    Args:
        a: Input tensor A [M, K] or [K, M] if trans_a
        b: Input tensor B [K, N] or [N, K] if trans_b
        trans_a: If True, transpose A
        trans_b: If True, transpose B
        static_persistent: If True, launches static_persistent_matmul_kernel
            with a fixed grid of min(NUM_SMS // num_ctas, num_tiles) * occupancy
        use_tma: Ignored (API compatibility)

    Returns:
        Output tensor C = A @ B of shape (M, N)
    """
    if static_persistent is None:
        static_persistent = False

    if trans_a:
        K, M = a.shape
    else:
        M, K = a.shape
    if trans_b:
        N, KB = b.shape
    else:
        KB, N = b.shape
    assert K == KB, f"Incompatible matrices: K dimension of A is {K}, K dimension of B is {KB}"

    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Autotuning path.
    if is_autotuning_enabled():
        if static_persistent:
            tilecpp_autotune_persistent_matmul(a, b, c, M, N, K, trans_a, trans_b)
        else:
            tilecpp_autotune_matmul(a, b, c, M, N, K, trans_a, trans_b)
        return c

    kernel_configs = kwargs.get("kernel_configs", None)
    if kernel_configs:
        tile_m = kernel_configs.get("TILE_SIZE_M", DEFAULT_TILE_SIZE_M)
        tile_n = kernel_configs.get("TILE_SIZE_N", DEFAULT_TILE_SIZE_N)
        tile_k = kernel_configs.get("TILE_SIZE_K", DEFAULT_TILE_SIZE_K)
        group_m = kernel_configs.get("GROUP_SIZE_M", DEFAULT_GROUP_SIZE_M)
        num_ctas = kernel_configs.get("num_ctas", 1)
        occupancy = kernel_configs.get("occupancy", 1)
    else:
        tile_m, tile_n, tile_k, group_m = _get_best_tile_sizes(M, N, K)
        num_ctas = 1
        occupancy = 1

    if static_persistent:
        _launch_persistent_matmul_kernel(
            a,
            b,
            c,
            M,
            N,
            K,
            trans_a,
            trans_b,
            tile_m,
            tile_n,
            tile_k,
            group_m,
            num_ctas=num_ctas,
            occupancy=occupancy,
        )
    else:
        _launch_matmul_kernel(
            a,
            b,
            c,
            M,
            N,
            K,
            trans_a,
            trans_b,
            tile_m,
            tile_n,
            tile_k,
            group_m,
        )

    return c


register_impl("matmul", "tilecpp")(matmul)
