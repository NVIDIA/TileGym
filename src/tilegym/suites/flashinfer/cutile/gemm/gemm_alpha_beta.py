# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
from math import ceil
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl
from tilegym.kernel_utils import get_kernel_configs

# Module-level tune cache: (M, N, K, transpose_a_int, transpose_b_int, dtype, num_sms, device) -> (best_cfg, tuned_kernel)
_gemm_alpha_beta_tune_cache: dict = {}


def cdiv(a, b):
    """Ceiling division helper function."""
    return (a + b - 1) // b


@ct.kernel
def gemm_alpha_beta_kernel_cutile(
    a_ptr,  # Input matrix A [M, K] or [K, M] if transpose_a
    b_ptr,  # Input matrix B [K, N] or [N, K] if transpose_b
    c_ptr,  # Output/Input matrix C [M, N] - modified in place
    alpha: ct.Constant[float],  # Alpha scaling factor
    beta: ct.Constant[float],  # Beta scaling factor
    M: ct.Constant[int],  # M dimension
    N: ct.Constant[int],  # N dimension
    K: ct.Constant[int],  # K dimension
    total_tiles: ct.Constant[int],  # Total number of tiles
    num_programs: ct.Constant[int],  # Number of SMs
    num_pid_m: ct.Constant[int],  # Number of M tiles
    num_pid_n: ct.Constant[int],  # Number of N tiles
    transpose_a: ct.Constant[int],  # Whether A is transposed (0 or 1)
    transpose_b: ct.Constant[int],  # Whether B is transposed (0 or 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    EPILOGUE_SUBTILE: ct.Constant[int],
):
    """
    CuTile kernel for GEMM with alpha/beta scaling: C = alpha * A @ B + beta * C

    Features:
    - Standard GEMM with alpha and beta scaling factors
    - Supports transpose_a and transpose_b
    - Uses persistent scheduling with SM-aware grid sizing
    - Uses GROUP_SIZE_M based tile swizzling
    - Optimized with latency hints for better pipelining
    """
    pid = ct.bid(0)

    num_k_tiles = ct.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    zero_pad = ct.PaddingMode.ZERO

    # Persistent scheduling loop
    for current_pid in range(pid, total_tiles, num_programs):
        # Calculate pid_m, pid_n with GROUP_SIZE_M swizzling
        group_id = current_pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = first_pid_m + (current_pid % group_size_m_actual)
        pid_n = (current_pid % num_pid_in_group) // group_size_m_actual

        # Initialize accumulator
        acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

        # K-loop for matrix multiplication using tile indices
        for k in range(num_k_tiles):
            # Load A block based on transpose_a flag with latency hint for pipelining
            if transpose_a == 1:
                # A is [K, M], load [BLOCK_K, BLOCK_M] and transpose
                a_block_kt = ct.load(
                    a_ptr,
                    index=(k, pid_m),  # tile indices
                    shape=(BLOCK_K, BLOCK_M),
                    order=(0, 1),
                    padding_mode=zero_pad,
                    latency=3,
                )
                a_block = ct.permute(a_block_kt, (1, 0))  # [BLOCK_M, BLOCK_K]
            else:
                # A is [M, K], load [BLOCK_M, BLOCK_K]
                a_block = ct.load(
                    a_ptr,
                    index=(pid_m, k),  # tile indices
                    shape=(BLOCK_M, BLOCK_K),
                    order=(0, 1),
                    padding_mode=zero_pad,
                    latency=3,
                )

            # Load B block based on transpose_b flag with latency hint
            if transpose_b == 1:
                # B is [N, K], load [BLOCK_N, BLOCK_K] and transpose
                b_block_nt = ct.load(
                    b_ptr,
                    index=(pid_n, k),  # tile indices
                    shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                    padding_mode=zero_pad,
                    latency=3,
                )
                b_block = ct.permute(b_block_nt, (1, 0))  # [BLOCK_K, BLOCK_N]
            else:
                # B is [K, N], load [BLOCK_K, BLOCK_N]
                b_block = ct.load(
                    b_ptr,
                    index=(k, pid_n),  # tile indices
                    shape=(BLOCK_K, BLOCK_N),
                    order=(0, 1),
                    padding_mode=zero_pad,
                    latency=3,
                )

            # Matrix multiplication: A @ B
            acc = ct.mma(a_block, b_block, acc=acc)

        if EPILOGUE_SUBTILE == 1:
            # Split accumulator into two N/2 halves to reduce shared memory in epilogue
            acc0 = ct.extract(acc, index=(0, 0), shape=(BLOCK_M, BLOCK_N // 2))
            acc1 = ct.extract(acc, index=(0, 1), shape=(BLOCK_M, BLOCK_N // 2))

            c_load0 = ct.load(
                c_ptr,
                index=(pid_m, pid_n * 2),
                shape=(BLOCK_M, BLOCK_N // 2),
                order=(0, 1),
                padding_mode=zero_pad,
            )
            c_load0_f32 = ct.astype(c_load0, ct.float32)
            result0 = alpha * acc0 + beta * c_load0_f32
            c_block0 = ct.astype(result0, c_ptr.dtype)
            ct.store(
                c_ptr,
                index=(pid_m, pid_n * 2),
                tile=c_block0,
                order=(0, 1),
            )

            c_load1 = ct.load(
                c_ptr,
                index=(pid_m, pid_n * 2 + 1),
                shape=(BLOCK_M, BLOCK_N // 2),
                order=(0, 1),
                padding_mode=zero_pad,
            )
            c_load1_f32 = ct.astype(c_load1, ct.float32)
            result1 = alpha * acc1 + beta * c_load1_f32
            c_block1 = ct.astype(result1, c_ptr.dtype)
            ct.store(
                c_ptr,
                index=(pid_m, pid_n * 2 + 1),
                tile=c_block1,
                order=(0, 1),
            )
        else:
            c_load = ct.load(
                c_ptr,
                index=(pid_m, pid_n),
                shape=(BLOCK_M, BLOCK_N),
                order=(0, 1),
                padding_mode=zero_pad,
            )

            c_load_f32 = ct.astype(c_load, ct.float32)
            result = alpha * acc + beta * c_load_f32

            c_block = ct.astype(result, c_ptr.dtype)

            ct.store(
                c_ptr,
                index=(pid_m, pid_n),
                tile=c_block,
                order=(0, 1),
            )


def _gemm_alpha_beta_autotune_configs():
    """
    Iterator of autotune configurations for gemm_alpha_beta kernel.
    Returns configurations optimized for different GPU architectures.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] >= 10:
        # EPILOGUE_SUBTILE=1 only validated on SM100; disable on SM12x to avoid correctness issues
        subtile_options = [0, 1] if gpu_capability == (10, 0) else [0]
        for BM, BN, nc in [
            (64, 64, 1),
            (64, 128, 1),
            (128, 64, 1),
            (128, 128, 1),
            (256, 64, 1),
            (256, 128, 1),
            (256, 128, 2),
            (256, 256, 2),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4, 8]:
                    for subtile in subtile_options:
                        yield SimpleNamespace(
                            BLOCK_M=BM,
                            BLOCK_N=BN,
                            BLOCK_K=BK,
                            GROUP_SIZE_M=8,
                            num_ctas=nc,
                            occupancy=occupancy,
                            EPILOGUE_SUBTILE=subtile,
                        )
    elif gpu_capability == (9, 0):
        for BM, BN in [
            (128, 128),
            (128, 256),
            (64, 128),
            (128, 64),
            (256, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2, 4]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                        EPILOGUE_SUBTILE=0,
                    )
    else:
        for BM, BN in [
            (64, 64),
            (128, 64),
            (128, 128),
            (128, 256),
            (256, 128),
        ]:
            for BK in [64]:
                for occupancy in [1, 2]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        num_ctas=1,
                        occupancy=occupancy,
                        EPILOGUE_SUBTILE=0,
                    )


def _get_default_kernel_configs():
    """
    Get GPU-specific default kernel configs for non-autotune path.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability == (10, 0):
        # Blackwell SM100 – aggressive config with epilogue subtiling
        return {
            "BLOCK_M": 256,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 2,
            "occupancy": 2,
            "EPILOGUE_SUBTILE": 1,
        }
    elif gpu_capability[0] >= 10:
        # SM10x / SM12x – conservative default (autotuner finds best)
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 2,
            "EPILOGUE_SUBTILE": 0,
        }
    elif gpu_capability == (9, 0):
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
            "EPILOGUE_SUBTILE": 0,
        }
    else:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": 64,
            "GROUP_SIZE_M": 8,
            "num_ctas": 1,
            "occupancy": 1,
            "EPILOGUE_SUBTILE": 0,
        }


def _compute_grid_and_programs(M, N, BLOCK_M, BLOCK_N, num_sms, num_ctas, occupancy):
    """Helper to compute grid size and number of programs."""
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    if num_sms is not None:
        NUM_SMS = min(NUM_SMS, num_sms)

    num_pid_m = cdiv(M, BLOCK_M)
    num_pid_n = cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n
    # Ensure num_programs >= 1: cuTile requires positive step for range() in persistent kernel.
    # When num_sms is very small (e.g., 1) and num_ctas > 1, NUM_SMS // num_ctas can be 0.
    num_programs = max(1, min(NUM_SMS // num_ctas, total_tiles) * occupancy)

    return num_pid_m, num_pid_n, total_tiles, num_programs


@register_impl("flashinfer.gemm.gemm_alpha_beta", backend="cutile")
def gemm_alpha_beta(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    trans_a=False,
    trans_b=True,
    alpha=1.0,
    beta=0.0,
    num_sms=None,
    **kwargs,
):
    """
    CuTile implementation of GEMM with alpha/beta scaling.

    Computes: C = alpha * A @ B + beta * C

    Args:
        a: Input matrix A [M, K] or [K, M] if trans_a
        b: Input matrix B [K, N] or [N, K] if trans_b
        c: Input/Output matrix C [M, N] - modified in place
        trans_a: Whether A is transposed
        trans_b: Whether B is transposed
        alpha: Scaling factor for A @ B
        beta: Scaling factor for existing C
        num_sms: Number of SMs to use (for SM throttling)

    Returns:
        Output tensor C [M, N]
    """
    # Get dimensions
    if trans_a:
        K, M = a.shape
    else:
        M, K = a.shape

    if trans_b:
        N, KB = b.shape
    else:
        KB, N = b.shape

    assert K == KB, "incompatible dimensions"
    assert c.shape == (M, N), "C must have shape [M, N]"
    assert a.is_contiguous(), "A matrix must be contiguous"
    assert b.is_contiguous(), "B matrix must be contiguous"
    assert c.is_contiguous(), "C matrix must be contiguous"

    # Convert boolean to int for ct.Constant
    transpose_a_int = 1 if trans_a else 0
    transpose_b_int = 1 if trans_b else 0

    # Check if autotune is requested
    use_autotune = kwargs.get("use_autotune", True)

    # For very low SM counts (1-16), disable autotune and use fixed config
    # to avoid autotune overhead dominating runtime
    if num_sms is not None and num_sms <= 16:
        use_autotune = False

    if use_autotune:
        # Use exhaustive_search for automatic configuration selection
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

        def grid_fn(cfg):
            num_pid_m, num_pid_n, total_tiles, num_programs = _compute_grid_and_programs(
                M, N, cfg.BLOCK_M, cfg.BLOCK_N, num_sms, cfg.num_ctas, cfg.occupancy
            )
            return (num_programs, 1, 1)

        def args_fn(cfg):
            num_pid_m, num_pid_n, total_tiles, num_programs = _compute_grid_and_programs(
                M, N, cfg.BLOCK_M, cfg.BLOCK_N, num_sms, cfg.num_ctas, cfg.occupancy
            )
            return (
                a,
                b,
                c.clone(),  # Clone for tuning to avoid corrupting C
                float(alpha),
                float(beta),
                M,
                N,
                K,
                total_tiles,
                num_programs,
                num_pid_m,
                num_pid_n,
                transpose_a_int,
                transpose_b_int,
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                cfg.BLOCK_K,
                cfg.GROUP_SIZE_M,
                cfg.EPILOGUE_SUBTILE,
            )

        def launch_args_fn(cfg):
            num_pid_m, num_pid_n, total_tiles, num_programs = _compute_grid_and_programs(
                M, N, cfg.BLOCK_M, cfg.BLOCK_N, num_sms, cfg.num_ctas, cfg.occupancy
            )
            return (
                a,
                b,
                c,  # Use actual C for final launch
                float(alpha),
                float(beta),
                M,
                N,
                K,
                total_tiles,
                num_programs,
                num_pid_m,
                num_pid_n,
                transpose_a_int,
                transpose_b_int,
                cfg.BLOCK_M,
                cfg.BLOCK_N,
                cfg.BLOCK_K,
                cfg.GROUP_SIZE_M,
                cfg.EPILOGUE_SUBTILE,
            )

        stream = torch.cuda.current_stream()
        cache_key = (M, N, K, transpose_a_int, transpose_b_int, a.dtype, num_sms, str(a.device))
        if cache_key not in _gemm_alpha_beta_tune_cache:
            result = exhaustive_search(
                list(_gemm_alpha_beta_autotune_configs()),
                stream,
                grid_fn,
                gemm_alpha_beta_kernel_cutile,
                args_fn,
                lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _gemm_alpha_beta_tune_cache[cache_key] = (
                best_cfg,
                ct.kernel(
                    gemm_alpha_beta_kernel_cutile._pyfunc,
                    num_ctas=best_cfg.num_ctas,
                    occupancy=best_cfg.occupancy,
                ),
            )
        best_cfg, tuned_kernel = _gemm_alpha_beta_tune_cache[cache_key]
        ct.launch(stream, grid_fn(best_cfg), tuned_kernel, launch_args_fn(best_cfg))

        return c

    else:
        # Fallback to non-autotune path
        default_configs = _get_default_kernel_configs()
        kernel_configs = get_kernel_configs(default_configs, kwargs.get("kernel_configs"))

        BLOCK_M = kernel_configs.get("BLOCK_M")
        BLOCK_N = kernel_configs.get("BLOCK_N")
        BLOCK_K = kernel_configs.get("BLOCK_K")
        GROUP_SIZE_M = kernel_configs.get("GROUP_SIZE_M", 8)
        num_ctas = kernel_configs.get("num_ctas", 1)
        occupancy = kernel_configs.get("occupancy", 1)
        epilogue_subtile = kernel_configs.get("EPILOGUE_SUBTILE", 0)

        num_pid_m, num_pid_n, total_tiles, num_programs = _compute_grid_and_programs(
            M, N, BLOCK_M, BLOCK_N, num_sms, num_ctas, occupancy
        )

        # 1D grid for persistent scheduling
        grid = (num_programs, 1, 1)

        # Build kernel with hints

        kernel = gemm_alpha_beta_kernel_cutile

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel,
            (
                a,
                b,
                c,
                float(alpha),
                float(beta),
                M,
                N,
                K,
                total_tiles,
                num_programs,
                num_pid_m,
                num_pid_n,
                transpose_a_int,
                transpose_b_int,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                GROUP_SIZE_M,
                epilogue_subtile,
            ),
        )

        return c
