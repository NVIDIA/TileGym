# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
from math import ceil
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch

from tilegym.backend import register_impl


# CuTile implementation of BMM kernel
@ct.kernel
def ct_bmm_kernel(A, B, C, tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int]):
    """CuTile kernel for batch matrix multiplication
    A has shape (Q, M, K), B has shape (Q, K, N) and C has shape (Q, M, N)
    where Q is the batch size (number of independent matrix multiplications)
    """
    bidx = ct.bid(0)  # M dimension
    bidy = ct.bid(1)  # N dimension
    bidz = ct.bid(2)  # Q (batch) dimension

    # Calculate number of K tiles using ct.num_tiles
    num_k_tiles = ct.num_tiles(A, axis=2, shape=(1, tm, tk))

    # Initialize accumulator
    sum = ct.full((tm, tn), 0.0, dtype=ct.float32)

    zero_pad = ct.PaddingMode.ZERO

    # K-dimension loop
    for k in range(num_k_tiles):
        # Load tiles with 3D index and 3D shape
        a = ct.load(A, index=(bidz, bidx, k), shape=(1, tm, tk), padding_mode=zero_pad)
        a = ct.reshape(a, (tm, tk))  # Reshape to 2D

        b = ct.load(B, index=(bidz, k, bidy), shape=(1, tk, tn), padding_mode=zero_pad)
        b = ct.reshape(b, (tk, tn))  # Reshape to 2D

        sum = ct.mma(a, b, acc=sum)

    # Convert to output dtype and store
    result = ct.astype(sum, C.dtype)
    # Store with 3D index and 3D shape
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(C, index=(bidz, bidx, bidy), tile=result_3d)


@ct.kernel
def ct_static_persistent_bmm_kernel(
    A,
    B,
    C,
    Batch,
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int],
    TILE_M: ct.Constant[int],
    TILE_N: ct.Constant[int],
    TILE_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    TRANSPOSE_A: ct.Constant[bool],
    TRANSPOSE_B: ct.Constant[bool],
):
    """CuTile static persistent GEMM kernel: C = A @ B with static scheduling

    - Uses 2D accumulator to avoid reshape overhead
    - Proper handling of transposed inputs via order parameter
    - Static persistent scheduling with proper grid calculation
    """
    # Get program ID
    bid = ct.bid(0)

    # Calculate total number of tiles
    num_tiles_m = ct.cdiv(M, TILE_M)
    num_tiles_n = ct.cdiv(N, TILE_N)
    total_tiles = num_tiles_m * num_tiles_n * Batch

    # Static persistent scheduling loop
    num_programs = ct.num_blocks(0)
    for current_bid in range(bid, total_tiles, num_programs):
        # Inline bmm_calculate_bid logic directly in kernel
        num_bid_m = ct.cdiv(M, TILE_M)
        num_bid_n = ct.cdiv(N, TILE_N)
        bid_q = current_bid // (num_bid_m * num_bid_n)
        num_bid_in_group = GROUP_SIZE_M * num_bid_n

        current_bid_2d = current_bid % (num_bid_m * num_bid_n)
        group_id = current_bid_2d // num_bid_in_group
        first_bid_m = group_id * GROUP_SIZE_M
        group_size_m_temp = num_bid_m - first_bid_m
        group_size_m = ct.minimum(group_size_m_temp, GROUP_SIZE_M)
        bid_m = first_bid_m + (current_bid_2d % group_size_m)
        bid_n = (current_bid_2d % num_bid_in_group) // group_size_m

        # Initialize 2D accumulator (avoid 3D reshape overhead)
        accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

        zero_pad = ct.PaddingMode.ZERO

        # K-dimension loop
        num_k_tiles = ct.cdiv(K, TILE_K)
        for k_tile in range(num_k_tiles):
            # Load A tile
            if TRANSPOSE_A:
                # A is transposed: physical layout (Q, K, M), load as (TILE_M, TILE_K)
                # Use order=(0, 2, 1) to read transposed
                a_tile_3d = ct.load(
                    A,
                    index=(bid_q, k_tile, bid_m),
                    shape=(1, TILE_K, TILE_M),
                    order=(0, 1, 2),
                    padding_mode=zero_pad,
                )
                # Transpose to get (1, TILE_M, TILE_K)
                a_tile_3d = ct.permute(a_tile_3d, (0, 2, 1))
            else:
                # A is normal: physical layout (Q, M, K)
                a_tile_3d = ct.load(
                    A,
                    index=(bid_q, bid_m, k_tile),
                    shape=(1, TILE_M, TILE_K),
                    order=(0, 1, 2),
                    padding_mode=zero_pad,
                )
            # Reshape to 2D for MMA
            a_tile = ct.reshape(a_tile_3d, (TILE_M, TILE_K))

            # Load B tile
            if TRANSPOSE_B:
                # B is transposed: physical layout (Q, N, K), load and transpose
                b_tile_3d = ct.load(
                    B,
                    index=(bid_q, bid_n, k_tile),
                    shape=(1, TILE_N, TILE_K),
                    order=(0, 1, 2),
                    padding_mode=zero_pad,
                )
                # Transpose to get (1, TILE_K, TILE_N)
                b_tile_3d = ct.permute(b_tile_3d, (0, 2, 1))
            else:
                # B is normal: physical layout (Q, K, N)
                b_tile_3d = ct.load(
                    B,
                    index=(bid_q, k_tile, bid_n),
                    shape=(1, TILE_K, TILE_N),
                    order=(0, 1, 2),
                    padding_mode=zero_pad,
                )
            # Reshape to 2D for MMA
            b_tile = ct.reshape(b_tile_3d, (TILE_K, TILE_N))

            # Matrix multiplication and accumulation (2D MMA)
            accumulator = ct.mma(a_tile, b_tile, acc=accumulator)

        # Convert to output dtype
        result = ct.astype(accumulator, C.dtype)
        # Reshape to 3D for store
        result_3d = ct.reshape(result, (1, TILE_M, TILE_N))
        ct.store(C, index=(bid_q, bid_m, bid_n), tile=result_3d, order=(0, 1, 2))


def _bmm_autotune_configs():
    if torch.cuda.get_device_capability() in [(12, 0), (12, 1)]:
        # B200 (sm_120/121): Use smaller tiles with occupancy tuning, num_ctas=1
        for TILE_M in [64, 128]:
            for TILE_N in [64, 128]:
                for TILE_K in [32, 64]:
                    for occupancy in [1, 2, 4]:
                        yield SimpleNamespace(
                            TILE_M=TILE_M,
                            TILE_N=TILE_N,
                            TILE_K=TILE_K,
                            GROUP_SIZE_M=8,
                            occupancy=occupancy,
                            num_ctas=1,
                        )
    elif torch.cuda.get_device_capability() == (9, 0):
        # H100 (sm_90): Medium tiles with occupancy tuning
        for TILE_M in [64, 128, 256]:
            for TILE_N in [64, 128, 256]:
                for TILE_K in [64]:
                    for occupancy in [1, 2]:
                        yield SimpleNamespace(
                            TILE_M=TILE_M,
                            TILE_N=TILE_N,
                            TILE_K=TILE_K,
                            GROUP_SIZE_M=8,
                            occupancy=occupancy,
                            num_ctas=1,
                        )
    else:
        # Other GPUs (e.g., GB100): Larger tiles with num_ctas=2
        for TILE_M in [128, 256]:
            for TILE_N in [256]:
                for TILE_K in [64]:
                    yield SimpleNamespace(
                        TILE_M=TILE_M,
                        TILE_N=TILE_N,
                        TILE_K=TILE_K,
                        GROUP_SIZE_M=8,
                        occupancy=1,
                        num_ctas=2,
                    )


def _persistent_bmm_autotune_base(stream, a, b, output, batch_size, M, N, K, transpose_a, transpose_b):
    """
    Autotuned static persistent BMM kernel

    Args:
        stream: CUDA stream
        a: Input tensor A with shape (Q, M, K) or (Q, K, M) if transposed
        b: Input tensor B with shape (Q, K, N) or (Q, N, K) if transposed
        output: Output tensor with shape (Q, M, N)
        batch_size: Batch size (Q)
        M: M dimension
        N: N dimension
        K: K dimension
        transpose_a: Whether to transpose A
        transpose_b: Whether to transpose B

    Returns:
        output: The computed result
    """

    # args_fn: maps config parameters to kernel arguments
    def args_fn(cfg):
        return (
            a,
            b,
            output,
            batch_size,
            M,
            N,
            K,
            cfg.TILE_M,
            cfg.TILE_N,
            cfg.TILE_K,
            cfg.GROUP_SIZE_M,
            transpose_a,
            transpose_b,
        )

    # grid_fn: computes grid size based on config
    def grid_fn(cfg):
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_tiles_m = (M + cfg.TILE_M - 1) // cfg.TILE_M
        num_tiles_n = (N + cfg.TILE_N - 1) // cfg.TILE_N
        total_tiles = num_tiles_m * num_tiles_n * batch_size

        occupancy = getattr(cfg, "occupancy", 1)
        num_ctas = getattr(cfg, "num_ctas", 1)

        base_programs = NUM_SMS // num_ctas
        grid_size = min(base_programs, total_tiles) * occupancy
        return (grid_size,)

    # Call autotuner to find the best config and execute the kernel
    ct_experimental.autotune_launch(
        stream,
        grid_fn=grid_fn,
        kernel=ct_static_persistent_bmm_kernel,
        args_fn=args_fn,
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=_bmm_autotune_configs,
        compiler_time_limit_sec=30,
    )

    return output


def bmm(a, b, transpose_a=False, transpose_b=False, static_persistent=True, **kwargs):
    """
    Batch Matrix Multiplication using CuTile

    Args:
        a: Input tensor A with shape (Q, M, K) where Q is the batch size
        b: Input tensor B with shape (Q, K, N) where Q is the batch size
        transpose_a: Whether to transpose A
        transpose_b: Whether to transpose B
        static_persistent: Whether to use static persistent schedule
        **kwargs: Additional arguments, including kernel_configs if needed (TILE_M, TILE_N, TILE_K, GROUP_SIZE_M, num_ctas)

    Returns:
        Output tensor C with shape (Q, M, N) where Q is the batch size
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

    # Create output tensor
    output = torch.empty((Q_A, M, N), device=a.device, dtype=a.dtype)
    kernel_configs = kwargs.get("kernel_configs", None)

    if static_persistent:
        _persistent_bmm_autotune_base(
            torch.cuda.current_stream(), a, b, output, Q_A, M, N, K_A, transpose_a, transpose_b
        )
    else:
        assert not transpose_a, "Transpose A is not supported for BMM"
        assert not transpose_b, "Transpose B is not supported for BMM"
        # Defaults for non-persistent schedule (lighter tiles)
        default_configs = {
            "TILE_M": 128,
            "TILE_N": 128,
            "TILE_K": 32,
        }
        if kernel_configs is None:
            kernel_configs = default_configs
        else:
            kernel_configs = {**default_configs, **kernel_configs}
        TILE_M = kernel_configs.get("TILE_M")
        TILE_N = kernel_configs.get("TILE_N")
        TILE_K = kernel_configs.get("TILE_K")
        # Grid calculation
        grid = (ceil(M / TILE_M), ceil(N / TILE_N), Q_A)
        # Launch kernel
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            ct_bmm_kernel,
            (a, b, output, TILE_M, TILE_N, TILE_K),
        )

    return output


# Backend registration
register_impl("bmm", "cutile")(bmm)
