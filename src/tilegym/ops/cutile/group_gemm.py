# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch

from tilegym.backend import register_impl
from tilegym.logger import get_logger

logger = get_logger(__name__)

# Type aliases for constants
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


@ct.kernel
def group_gemm_kernel(
    As,  # List of A matrices
    Bs,  # List of B matrices
    Cs,  # List of C matrices
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    num_sm: ConstInt,
    transpose_b: ConstBool,
):
    tile_idx = ct.bid(0)
    last_problem_end = 0
    group_size = len(As)
    zero_pad = ct.PaddingMode.ZERO

    for g in range(group_size):
        Ai = As[g]
        Bi = Bs[g]
        Ci = Cs[g]

        # Get dimensions dynamically for each group
        # A: (M, K), B: (K, N) or (N, K) if transpose_b, C: (M, N)
        num_m_tiles = ct.num_tiles(Ai, 0, (TILE_M, TILE_K))
        num_k_tiles = ct.num_tiles(Ai, 1, (TILE_M, TILE_K))
        if transpose_b:
            num_n_tiles = ct.num_tiles(Bi, 0, (TILE_N, TILE_K))
        else:
            num_n_tiles = ct.num_tiles(Bi, 1, (TILE_K, TILE_N))

        num_tiles = num_m_tiles * num_n_tiles

        # Process tiles for this group using persistent scheduling
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles:
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # Initialize accumulator
            acc = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)

            # K-dimension loop
            for kk in range(num_k_tiles):
                # Load A tile
                ta = ct.load(
                    Ai,
                    (tile_m_idx, kk),
                    shape=(TILE_M, TILE_K),
                    padding_mode=zero_pad,
                )

                # Load B tile
                if transpose_b:
                    # B is transposed: load from (N, K) layout
                    tb = ct.load(
                        Bi,
                        (tile_n_idx, kk),
                        shape=(TILE_N, TILE_K),
                        padding_mode=zero_pad,
                    )
                    tb = ct.transpose(tb)  # Convert to (TILE_K, TILE_N)
                else:
                    # B is normal: load from (K, N) layout
                    tb = ct.load(
                        Bi,
                        (kk, tile_n_idx),
                        shape=(TILE_K, TILE_N),
                        padding_mode=zero_pad,
                    )

                # Matrix multiplication and accumulation
                acc = ct.mma(ta, tb, acc)

            # Convert to output dtype and store
            acc = ct.astype(acc, Ci.dtype)
            ct.store(Ci, (tile_m_idx, tile_n_idx), tile=acc)

            # Move to next tile
            tile_idx += num_sm

        # Update the end position for the next group
        last_problem_end = last_problem_end + num_tiles


def _group_gemm_autotune_configs():
    """
    Iterator of autotune configurations for group GEMM kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()
    if gpu_capability in [(12, 0), (12, 1)]:
        yield SimpleNamespace(TILE_M=64, TILE_N=128, TILE_K=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, TILE_K=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, TILE_K=64, num_ctas=1, occupancy=1)
    else:
        yield SimpleNamespace(TILE_M=256, TILE_N=256, TILE_K=64, num_ctas=2, occupancy=1)


def cutile_autotune_group_gemm(stream, group_A, group_B, group_C, transpose_b, device):
    """Autotune group GEMM kernel."""
    NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (NUM_SMS // cfg.num_ctas * cfg.occupancy, 1, 1),
        kernel=group_gemm_kernel,
        args_fn=lambda cfg: (
            group_A,
            group_B,
            group_C,
            cfg.TILE_M,
            cfg.TILE_N,
            cfg.TILE_K,
            NUM_SMS // cfg.num_ctas * cfg.occupancy,
            transpose_b,
        ),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=_group_gemm_autotune_configs,
    )
    return group_C


def group_gemm(
    group_A,
    group_B,
    static_persistent=True,
    transpose_b=False,
    **kwargs,
):
    if not group_A or not group_B:
        raise ValueError("group_A and group_B must not be empty")

    if len(group_A) != len(group_B):
        raise ValueError(f"group_A and group_B must have same length, got {len(group_A)} and {len(group_B)}")

    device = group_A[0].device
    dtype = group_A[0].dtype

    # Create output tensors
    group_C = []
    for A, B in zip(group_A, group_B):
        M, K = A.shape
        N = B.shape[0] if transpose_b else B.shape[1]
        C = torch.empty((M, N), device=device, dtype=dtype)
        group_C.append(C)

    # Autotune mode
    stream = torch.cuda.current_stream()
    cutile_autotune_group_gemm(stream, group_A, group_B, group_C, transpose_b, device)
    return group_C


register_impl("group_gemm", "cutile")(group_gemm)
