# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from math import ceil
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch

from tilegym.backend import register_impl
from tilegym.logger import get_logger

logger = get_logger(__name__)

# Type aliases for constants
ConstInt = ct.Constant[int]


def swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M):
    # Get the global IDs of the current CUDA block (CTA) in a 1D grid.
    bid = ct.bid(0)
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(
    A,
    B,
    C,
    TILE_SIZE_M: ConstInt,  # Tile size along M dimension (rows of C)
    TILE_SIZE_N: ConstInt,  # Tile size along N dimension (columns of C)
    TILE_SIZE_K: ConstInt,
):  # Tile size along K dimension (inner product dimension)
    """
    cuTile kernel for performing matrix multiplication C = A @ B.

    This kernel uses a tiled approach, where each CUDA thread block (CTA)
    computes a `TILE_SIZE_M` x `TILE_SIZE_N` tile of the output matrix C. The computation
    involves iterating over the K-dimension in chunks of `TILE_SIZE_K`.

    Args:
        A: Input matrix A (M x K).
        B: Input matrix B (K x N).
        C: Output matrix C (M x N).
        TILE_SIZE_M (ConstInt): The height of the output tile computed by this block.
                       Corresponds to rows of A and C.
        TILE_SIZE_N (ConstInt): The width of the output tile computed by this block.
                       Corresponds to columns of B and C.
        TILE_SIZE_K (ConstInt): The depth of the inner loop (K-dimension) tile size.
                       Corresponds to columns of A and rows of B.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, TILE_SIZE_M, TILE_SIZE_N, GROUP_SIZE_M)

    # Calculate the total number of K-tiles that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))` extracts the K-dimension (axis 1)
    # from matrix A's shape, assuming A's shape is conceptually (M_tiles, K_tiles),
    # and then implicitly performs ceiling division by `TILE_SIZE_K` to get the number of K-tiles.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(TILE_SIZE_M, TILE_SIZE_K))

    # Initialize an accumulator for the current output tile (TILE_SIZE_M x TILE_SIZE_N).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # Convert fp32 to tf32 to use tensorcore
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype

    # K-dimension loop: Iterate over the K-dimension in chunks of 'TILE_SIZE_K'.
    # In each iteration, a `TILE_SIZE_M` x `TILE_SIZE_K` tile from A and a `TILE_SIZE_K` x `TILE_SIZE_N` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(TILE_SIZE_M, TILE_SIZE_K)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(TILE_SIZE_M, TILE_SIZE_K), padding_mode=zero_pad).astype(dtype)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(TILE_SIZE_K, TILE_SIZE_N)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(TILE_SIZE_K, TILE_SIZE_N), padding_mode=zero_pad).astype(dtype)

        # Perform Matrix Multiplication for the current tiles.
        # `ct.mma` computes the product of the two loaded tiles and accumulates the result.
        accumulator = ct.mma(a, b, accumulator)

    # Convert the final accumulated result to the desired output data type (C.dtype).
    # This might downcast from float32 to float16 if the output is float16.
    accumulator = ct.astype(accumulator, C.dtype)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)


def _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M):
    group_id = tile_id // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (tile_id % group_size_m)
    bid_n = (tile_id % num_bid_in_group) // group_size_m
    return bid_m, bid_n


@ct.kernel(num_ctas=2)
def static_persistent_matmul_kernel(
    A,
    B,
    C,
    M: int,
    N: int,
    K: int,
    TILE_SIZE_M: ct.Constant[int],
    TILE_SIZE_N: ct.Constant[int],
    TILE_SIZE_K: ct.Constant[int],
    transpose_a: ct.Constant[bool],
    transpose_b: ct.Constant[bool],
    GROUP_SIZE_M: ct.Constant[int],
):
    """CuTile static persistent matmul kernel: C = A @ B with static scheduling"""
    # Get program ID
    start_bid = ct.bid(0)

    # Calculate total number of tiles
    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    k_tiles = ct.cdiv(K, TILE_SIZE_K)
    num_tiles = num_bid_m * num_bid_n
    num_programs = ct.num_blocks(0)

    # Static persistent scheduling loop
    for tile_id in range(start_bid, num_tiles, num_programs):
        # Calculate tile coordinates using GROUP_SIZE_M grouping
        num_bid_in_group = GROUP_SIZE_M * num_bid_n
        bid_m, bid_n = _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M)

        # Initialize accumulator
        accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)

        # K-dimension loop
        for k_tile in range(k_tiles):
            # Load A tile
            if transpose_a:
                # A is transposed: load from (K, M) layout
                a = ct.load(A, index=(k_tile, bid_m), shape=(TILE_SIZE_K, TILE_SIZE_M))
                a = ct.transpose(a)  # Convert to (TILE_SIZE_M, TILE_SIZE_K)
            else:
                # A is normal: load from (M, K) layout
                a = ct.load(A, index=(bid_m, k_tile), shape=(TILE_SIZE_M, TILE_SIZE_K))

            # Load B tile
            if transpose_b:
                # B is transposed: load from (N, K) layout
                b = ct.load(B, index=(bid_n, k_tile), shape=(TILE_SIZE_N, TILE_SIZE_K))
                b = ct.transpose(b)  # Convert to (TILE_SIZE_K, TILE_SIZE_N)
            else:
                # B is normal: load from (K, N) layout
                b = ct.load(B, index=(k_tile, bid_n), shape=(TILE_SIZE_K, TILE_SIZE_N))

            # Convert fp32 to tf32 to use tensorcore
            dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
            a = ct.astype(a, dtype)
            b = ct.astype(b, dtype)

            # Matrix multiplication and accumulation
            accumulator = ct.mma(a, b, acc=accumulator)

        # Convert to output dtype and store
        result = ct.astype(accumulator, C.dtype)
        ct.store(C, index=(bid_m, bid_n), tile=result)


def _matmul_autotune_configs():
    """
    Iterator of autotune configurations for matmul kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=32, num_ctas=1, occupancy=2)
    else:
        # sm100 (Blackwell)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=128, TILE_SIZE_K=32, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=4, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=512, TILE_SIZE_N=256, TILE_SIZE_K=64, num_ctas=2, occupancy=1)


def cutile_autotune_matmul(stream, a, b, c):
    M, N = c.shape
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N),
            1,
            1,
        ),
        kernel=matmul_kernel,
        args_fn=lambda cfg: (a, b, c, cfg.TILE_SIZE_M, cfg.TILE_SIZE_N, cfg.TILE_SIZE_K),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=_matmul_autotune_configs,
    )
    return c


def _static_persistent_matmul_autotune_configs():
    """
    Iterator of autotune configurations for static persistent matmul kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        yield SimpleNamespace(TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=4)
        yield SimpleNamespace(TILE_SIZE_M=64, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=64, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=4)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1)
    else:
        # sm100 (Blackwell)
        yield SimpleNamespace(TILE_SIZE_M=128, TILE_SIZE_N=512, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=4, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=2, occupancy=1)
        yield SimpleNamespace(TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=64, GROUP_SIZE_M=8, num_ctas=1, occupancy=1)
        yield SimpleNamespace(
            TILE_SIZE_M=256, TILE_SIZE_N=256, TILE_SIZE_K=128, GROUP_SIZE_M=8, num_ctas=2, occupancy=1
        )


def cutile_autotune_static_persistent_matmul(stream, a, b, c, M, N, K, trans_a, trans_b):
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            min(NUM_SMS // cfg.num_ctas, ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N)) * cfg.occupancy,
            1,
            1,
        ),
        kernel=static_persistent_matmul_kernel,
        args_fn=lambda cfg: (
            a,
            b,
            c,
            M,
            N,
            K,
            cfg.TILE_SIZE_M,
            cfg.TILE_SIZE_N,
            cfg.TILE_SIZE_K,
            trans_a,
            trans_b,
            cfg.GROUP_SIZE_M,
        ),
        hints_fn=lambda cfg: {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        },
        search_space=_static_persistent_matmul_autotune_configs,
    )
    return c


def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a=False,
    trans_b=False,
    static_persistent=None,
    use_tma=False,
    **kwargs,
):
    if static_persistent is None:
        static_persistent = False

    # Get matrix dimensions
    if trans_a:
        K, M = a.shape
    else:
        M, K = a.shape
    if trans_b:
        N, KB = b.shape
    else:
        KB, N = b.shape
    assert K == KB, f"Incompatible matrices: K dimension of A is {K}, K dimension of B is {KB}"

    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Grid calculation
    stream = torch.cuda.current_stream()
    if static_persistent:
        cutile_autotune_static_persistent_matmul(stream, a, b, c, M, N, K, trans_a, trans_b)
    else:
        assert trans_a == False, "trans_a is not supported for cutile"
        assert trans_b == False, "trans_b is not supported for cutile"
        cutile_autotune_matmul(stream, a, b, c)
    return c


register_impl("matmul", "cutile")(matmul)
