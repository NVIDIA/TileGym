# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
from math import ceil
from types import SimpleNamespace

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.kernel_utils import get_kernel_configs


def cdiv(a, b):
    """Ceiling division helper function."""
    return (a + b - 1) // b


def _is_large_m(total_m, Q):
    """Determine if average M is large enough for non-swapped configs."""
    average_m = total_m / Q
    is_large_m = average_m >= 256
    return is_large_m


@ct.kernel
def ragged_block_scaled_bmm_kernel_cutile(
    a_ptr,  # Input matrix A [total_m, K] FP8
    b_ptr,  # Input matrix B [Q, N, K] FP8
    a_scale_ptr,  # Scale for A [total_m, k_tiles] FP32
    b_scale_ptr,  # Scale for B [Q, n_tiles, k_tiles] FP32
    c_ptr,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    Q: ct.Constant[int],  # Number of batches
    max_m: ct.Constant[int],  # Max segment size
    N: ct.Constant[int],  # Output N dimension
    K: ct.Constant[int],  # K dimension
    total_m: ct.Constant[int],  # Total M (for bounds checking)
    total_tiles: ct.Constant[int],  # Total number of tiles
    num_programs: ct.Constant[int],  # Number of SMs
    num_k_tiles: ct.Constant[int],  # Number of K tiles
    num_pid_m: ct.Constant[int],  # Number of M tiles per batch
    num_pid_n: ct.Constant[int],  # Number of N tiles per batch
    tiles_per_batch: ct.Constant[int],  # num_pid_m * num_pid_n
    stride_a0: ct.Constant[int],  # Stride for A dim 0
    stride_a1: ct.Constant[int],  # Stride for A dim 1
    stride_b0: ct.Constant[int],  # Stride for B dim 0
    stride_b1: ct.Constant[int],  # Stride for B dim 1
    stride_b2: ct.Constant[int],  # Stride for B dim 2
    stride_sa0: ct.Constant[int],  # Stride for a_scale dim 0
    stride_sa1: ct.Constant[int],  # Stride for a_scale dim 1
    stride_sb0: ct.Constant[int],  # Stride for b_scale dim 0
    stride_sb1: ct.Constant[int],  # Stride for b_scale dim 1
    stride_sb2: ct.Constant[int],  # Stride for b_scale dim 2
    stride_c0: ct.Constant[int],  # Stride for C dim 0
    stride_c1: ct.Constant[int],  # Stride for C dim 1
    has_a_scale: ct.Constant[int],  # Whether a_scale is provided (0 or 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    CuTile kernel for ragged block-scaled batched matrix multiplication.

    Performs (A * a_scale) @ (B * b_scale)^T where:
    - A is flattened FP8 with segment offsets (m_indptr defines boundaries)
    - B is batched FP8 [Q, N, K]
    - a_scale and b_scale are per-block scales
    - Output C is [total_m, N]

    Uses persistent scheduling with static grid and GROUP_SIZE_M tile swizzling.
    Uses Array.slice + TMA (ct.load/ct.store) for A and C access.
    """
    pid = ct.bid(0)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Persistent scheduling loop
    for current_pid in range(pid, total_tiles, num_programs):
        # Calculate pid_q, pid_m, pid_n with GROUP_SIZE_M swizzling
        # pid_q = batch index
        pid_q = current_pid // tiles_per_batch
        pid_in_batch = current_pid % tiles_per_batch

        group_id = pid_in_batch // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = first_pid_m + (pid_in_batch % group_size_m_actual)
        pid_n = (pid_in_batch % num_pid_in_group) // group_size_m_actual

        # Load segment boundaries using ct.load with dynamic index
        m_start_tile = ct.load(m_indptr, index=(pid_q,), shape=(1,))
        m_start = m_start_tile.item()
        m_end_tile = ct.load(m_indptr, index=(pid_q + 1,), shape=(1,))
        m_end = m_end_tile.item()
        valid_m = m_end - m_start

        # Only process if this tile is within valid M range
        if pid_m * BLOCK_M < valid_m:
            # Create sliced views for A and C using Array.slice
            Ai = a_ptr.slice(axis=0, start=m_start, stop=m_end)
            Ci = c_ptr.slice(axis=0, start=m_start, stop=m_end)

            if has_a_scale == 1:
                a_scale_i = a_scale_ptr.slice(axis=0, start=m_start, stop=m_end)

            # Initialize accumulator
            acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            # N tile offset (element-level) for b_scale calculation
            n_offset = pid_n * BLOCK_N
            offs_bsn = n_offset // BLOCK_K

            # Zero accumulator for per-K MMA (reused each iteration)
            mma_zeros = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            # K-loop for matrix multiplication
            for k in range(num_k_tiles):
                k_offset = k * BLOCK_K

                # Load A block using TMA
                a_block = ct.load(
                    Ai,
                    index=(pid_m, k),
                    shape=(BLOCK_M, BLOCK_K),
                    padding_mode=ct.PaddingMode.ZERO,
                )

                # Load B block - B is [Q, N, K], we need [BLOCK_N, BLOCK_K]
                b_block_3d = ct.load(
                    b_ptr,
                    index=(pid_q, n_offset // BLOCK_N, k_offset // BLOCK_K),
                    shape=(1, BLOCK_N, BLOCK_K),
                    order=(0, 1, 2),
                    padding_mode=ct.PaddingMode.ZERO,
                )
                # Reshape to [BLOCK_N, BLOCK_K] then transpose to get [BLOCK_K, BLOCK_N]
                b_block_nk = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))
                b_block = ct.permute(b_block_nk, (1, 0))  # [BLOCK_K, BLOCK_N]

                # Matrix multiplication: A [BLOCK_M, BLOCK_K] @ B [BLOCK_K, BLOCK_N] = C [BLOCK_M, BLOCK_N]
                c_mma = ct.mma(a_block, b_block, acc=mma_zeros)

                # Load and apply scales
                if has_a_scale == 1:
                    # Load a_scale for this block using TMA
                    a_scale_block = ct.load(
                        a_scale_i,
                        index=(pid_m, k),
                        shape=(BLOCK_M, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                    )

                    # Load b_scale - scalar at [pid_q, offs_bsn, k]
                    b_scale_block = ct.load(
                        b_scale_ptr,
                        index=(pid_q, offs_bsn, k),
                        shape=(1, 1, 1),
                        order=(0, 1, 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_scale_val = ct.reshape(b_scale_block, (1, 1))

                    # Combined scale: a_scale [BLOCK_M, 1] * b_scale [1, 1] = [BLOCK_M, 1]
                    scale_combined = a_scale_block * ct.broadcast_to(b_scale_val, (BLOCK_M, 1))
                    scale_ab = ct.broadcast_to(scale_combined, (BLOCK_M, BLOCK_N))
                else:
                    # Only b_scale
                    b_scale_block = ct.load(
                        b_scale_ptr,
                        index=(pid_q, offs_bsn, k),
                        shape=(1, 1, 1),
                        order=(0, 1, 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_scale_val = ct.reshape(b_scale_block, (1, 1))
                    scale_ab = ct.broadcast_to(b_scale_val, (BLOCK_M, BLOCK_N))

                # Apply scale and accumulate
                acc = acc + c_mma * scale_ab

            # Convert to output dtype
            c_block = ct.astype(acc, c_ptr.dtype)

            # Store to output C using TMA
            ct.store(Ci, index=(pid_m, pid_n), tile=c_block)


@ct.kernel
def ragged_block_scaled_bmm_kernel_cutile_swap_ab(
    a_ptr,  # Input matrix A [total_m, K] FP8
    b_ptr,  # Input matrix B [Q, N, K] FP8
    a_scale_ptr,  # Scale for A [total_m, k_tiles] FP32
    b_scale_ptr,  # Scale for B [Q, n_tiles, k_tiles] FP32
    c_ptr,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    Q: ct.Constant[int],
    max_m: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int],
    total_m: ct.Constant[int],
    total_tiles: ct.Constant[int],
    num_programs: ct.Constant[int],
    num_k_tiles: ct.Constant[int],
    num_pid_m: ct.Constant[int],
    num_pid_n: ct.Constant[int],
    tiles_per_batch: ct.Constant[int],
    stride_a0: ct.Constant[int],
    stride_a1: ct.Constant[int],
    stride_b0: ct.Constant[int],
    stride_b1: ct.Constant[int],
    stride_b2: ct.Constant[int],
    stride_sa0: ct.Constant[int],
    stride_sa1: ct.Constant[int],
    stride_sb0: ct.Constant[int],
    stride_sb1: ct.Constant[int],
    stride_sb2: ct.Constant[int],
    stride_c0: ct.Constant[int],
    stride_c1: ct.Constant[int],
    has_a_scale: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    CuTile kernel for ragged block-scaled BMM with swap_ab optimization.
    Uses Array.slice + TMA (ct.load/ct.store) for A and C access.
    """
    pid = ct.bid(0)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Persistent scheduling loop
    for current_pid in range(pid, total_tiles, num_programs):
        pid_q = current_pid // tiles_per_batch
        pid_in_batch = current_pid % tiles_per_batch

        group_id = pid_in_batch // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_m = first_pid_m + (pid_in_batch % group_size_m_actual)
        pid_n = (pid_in_batch % num_pid_in_group) // group_size_m_actual

        m_start_tile = ct.load(m_indptr, index=(pid_q,), shape=(1,))
        m_start = m_start_tile.item()
        m_end_tile = ct.load(m_indptr, index=(pid_q + 1,), shape=(1,))
        m_end = m_end_tile.item()
        valid_m = m_end - m_start

        if pid_m * BLOCK_M < valid_m:
            # Create sliced views for A and C using Array.slice
            Ai = a_ptr.slice(axis=0, start=m_start, stop=m_end)
            Ci = c_ptr.slice(axis=0, start=m_start, stop=m_end)

            if has_a_scale == 1:
                a_scale_i = a_scale_ptr.slice(axis=0, start=m_start, stop=m_end)

            acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

            n_offset = pid_n * BLOCK_N
            offs_bsn = n_offset // BLOCK_K

            # Zero accumulator for per-K MMA (reused each iteration)
            mma_zeros = ct.full((BLOCK_N, BLOCK_M), 0.0, dtype=ct.float32)

            for k in range(num_k_tiles):
                k_offset = k * BLOCK_K

                # Load A block using TMA
                a_block = ct.load(
                    Ai,
                    index=(pid_m, k),
                    shape=(BLOCK_M, BLOCK_K),
                    padding_mode=ct.PaddingMode.ZERO,
                )

                # Load B block
                b_block_3d = ct.load(
                    b_ptr,
                    index=(pid_q, n_offset // BLOCK_N, k_offset // BLOCK_K),
                    shape=(1, BLOCK_N, BLOCK_K),
                    order=(0, 1, 2),
                    padding_mode=ct.PaddingMode.ZERO,
                )
                b_block_nk = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))

                # swap_ab: compute (B @ A^T)^T
                a_block_t = ct.permute(a_block, (1, 0))
                c_swapped = ct.mma(b_block_nk, a_block_t, acc=mma_zeros)
                c_mma = ct.permute(c_swapped, (1, 0))

                # Load and apply scales
                if has_a_scale == 1:
                    a_scale_block = ct.load(
                        a_scale_i,
                        index=(pid_m, k),
                        shape=(BLOCK_M, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                    )

                    b_scale_block = ct.load(
                        b_scale_ptr,
                        index=(pid_q, offs_bsn, k),
                        shape=(1, 1, 1),
                        order=(0, 1, 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_scale_val = ct.reshape(b_scale_block, (1, 1))
                    scale_combined = a_scale_block * ct.broadcast_to(b_scale_val, (BLOCK_M, 1))
                    scale_ab = ct.broadcast_to(scale_combined, (BLOCK_M, BLOCK_N))
                else:
                    b_scale_block = ct.load(
                        b_scale_ptr,
                        index=(pid_q, offs_bsn, k),
                        shape=(1, 1, 1),
                        order=(0, 1, 2),
                        padding_mode=ct.PaddingMode.ZERO,
                    )
                    b_scale_val = ct.reshape(b_scale_block, (1, 1))
                    scale_ab = ct.broadcast_to(b_scale_val, (BLOCK_M, BLOCK_N))

                acc = acc + c_mma * scale_ab

            c_block = ct.astype(acc, c_ptr.dtype)

            # Store to output C using TMA
            ct.store(Ci, index=(pid_m, pid_n), tile=c_block)


def _ragged_block_scaled_bmm_autotune_configs():
    """
    Iterator of autotune configurations for ragged_block_scaled_bmm kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        for BM, BN, swap_ab in [
            (128, 128, False),
            (64, 128, True),
            (32, 128, True),
        ]:
            for BK in [128]:
                for occupancy in [1, 2]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8,
                        swap_ab=swap_ab,
                        num_ctas=1,
                        occupancy=occupancy,
                    )
    elif gpu_capability == (9, 0):
        for BM, BN, swap_ab in [
            (256, 128, False),
            (128, 128, False),
            (64, 128, True),
            (32, 128, True),
            (16, 256, True),
            (32, 256, True),
            (64, 256, True),
        ]:
            for BK in [128]:
                for occupancy in [1, 2]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=8 if not swap_ab else 4,
                        swap_ab=swap_ab,
                        num_ctas=2 if BM == 256 else 1,
                        occupancy=occupancy,
                    )
    else:
        # Non-swapped configs (for large M)
        for BM, nc, occ in [
            (256, 2, 1),
            (128, 1, 1),
            (128, 2, 2),  # for small M
        ]:
            yield SimpleNamespace(
                BLOCK_M=BM,
                BLOCK_N=128,
                BLOCK_K=128,
                GROUP_SIZE_M=8,
                swap_ab=False,
                num_ctas=nc,
                occupancy=occ,
            )
        # Swapped configs (for small M)
        for GM in [2, 4]:
            for BM in [16, 32, 64]:
                yield SimpleNamespace(
                    BLOCK_M=BM,
                    BLOCK_N=256,
                    BLOCK_K=128,
                    GROUP_SIZE_M=GM,
                    swap_ab=True,
                    num_ctas=1,
                    occupancy=1,
                )


def _get_default_kernel_configs(total_m, Q, VEC_SIZE):
    """
    Get GPU-specific default kernel configs for non-autotune path.
    """
    gpu_capability = torch.cuda.get_device_capability()
    is_large_m = _is_large_m(total_m, Q)

    if gpu_capability in [(12, 0), (12, 1)]:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": VEC_SIZE,
            "GROUP_SIZE_M": 8,
            "swap_ab": False,
            "num_ctas": 1,
            "occupancy": 2,
        }
    elif gpu_capability == (9, 0):
        if is_large_m:
            return {
                "BLOCK_M": 256,
                "BLOCK_N": 128,
                "BLOCK_K": VEC_SIZE,
                "GROUP_SIZE_M": 8,
                "swap_ab": False,
                "num_ctas": 2,
                "occupancy": 1,
            }
        else:
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": VEC_SIZE,
                "GROUP_SIZE_M": 8,
                "swap_ab": False,
                "num_ctas": 1,
                "occupancy": 1,
            }
    else:
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": VEC_SIZE,
            "GROUP_SIZE_M": 8,
            "swap_ab": False,
            "num_ctas": 1,
            "occupancy": 1,
        }


@register_impl("flashinfer.gemm.ragged_block_scaled_bmm", backend="cutile")
def ragged_block_scaled_bmm(
    a,
    b,
    a_scale,
    b_scale,
    m_indptr,
    max_m,
    max_m_device=None,
    transpose_a=False,
    transpose_b=True,
    out_dtype=None,
    **kwargs,
):
    """
    CuTile implementation of ragged block-scaled BMM.
    """
    # Validate inputs
    assert transpose_a == False and transpose_b == True, "Only NT layout is supported"
    assert a.is_contiguous(), "A matrix must be contiguous"
    assert b.is_contiguous(), "B matrix must be contiguous"
    assert a_scale is None or a_scale.is_contiguous(), "A scale matrix must be contiguous"
    assert b_scale.is_contiguous(), "B scale matrix must be contiguous"
    assert m_indptr.is_contiguous(), "m_indptr must be contiguous"

    # Get dimensions
    total_m, K_A = a.shape
    Q, N, K_B = b.shape

    assert K_A == K_B, f"K dimensions must match: {K_A} != {K_B}"
    assert m_indptr.shape[0] == Q + 1, "m_indptr must have Q+1 elements"

    # Validate scale dimensions
    Q_SB, rnb, rkb = b_scale.shape
    VEC_SIZE = K_B // rkb

    if a_scale is not None:
        total_ma, rka = a_scale.shape
        assert total_ma == total_m, "a_scale total_m dimension mismatch"

    assert Q_SB == Q, "b_scale Q dimension mismatch"

    # Determine output dtype
    if out_dtype is None:
        out_dtype = torch.bfloat16

    # Allocate output
    c = torch.empty((total_m, N), device=a.device, dtype=out_dtype)

    # Get kernel configs
    default_configs = _get_default_kernel_configs(total_m, Q, VEC_SIZE)
    kernel_configs = get_kernel_configs(default_configs, kwargs.get("kernel_configs"))

    BLOCK_M = kernel_configs.get("BLOCK_M")
    BLOCK_N = kernel_configs.get("BLOCK_N")
    BLOCK_K = kernel_configs.get("BLOCK_K", VEC_SIZE)
    GROUP_SIZE_M = kernel_configs.get("GROUP_SIZE_M", 8)
    swap_ab = kernel_configs.get("swap_ab", False)
    num_ctas = kernel_configs.get("num_ctas", 1)
    occupancy = kernel_configs.get("occupancy", 1)

    # Calculate grid size for persistent scheduling
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid_m = cdiv(max_m, BLOCK_M)
    num_pid_n = cdiv(N, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * Q
    num_programs = min(NUM_SMS // num_ctas, total_tiles) * occupancy
    num_k_tiles = cdiv(K_A, BLOCK_K)

    grid = (num_programs, 1, 1)

    # Prepare strides
    stride_a0 = a.stride(0)
    stride_a1 = a.stride(1)
    stride_b0 = b.stride(0)
    stride_b1 = b.stride(1)
    stride_b2 = b.stride(2)
    stride_sa0 = a_scale.stride(0) if a_scale is not None else 0
    stride_sa1 = a_scale.stride(1) if a_scale is not None else 0
    stride_sb0 = b_scale.stride(0)
    stride_sb1 = b_scale.stride(1)
    stride_sb2 = b_scale.stride(2)
    stride_c0 = c.stride(0)
    stride_c1 = c.stride(1)
    has_a_scale = 1 if a_scale is not None else 0

    if a_scale is None:
        a_scale_ptr = torch.empty(1, device=a.device, dtype=torch.float32)
    else:
        a_scale_ptr = a_scale

    kernel_fn = ragged_block_scaled_bmm_kernel_cutile_swap_ab if swap_ab else ragged_block_scaled_bmm_kernel_cutile

    kernel = kernel_fn

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            a,
            b,
            a_scale_ptr,
            b_scale,
            c,
            m_indptr,
            Q,
            max_m,
            N,
            K_A,
            total_m,
            total_tiles,
            num_programs,
            num_k_tiles,
            num_pid_m,
            num_pid_n,
            tiles_per_batch,
            stride_a0,
            stride_a1,
            stride_b0,
            stride_b1,
            stride_b2,
            stride_sa0,
            stride_sa1,
            stride_sb0,
            stride_sb1,
            stride_sb2,
            stride_c0,
            stride_c1,
            has_a_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_SIZE_M,
        ),
    )

    return c
