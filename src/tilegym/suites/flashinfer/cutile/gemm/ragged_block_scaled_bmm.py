# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import is_autotune_enabled
from tilegym.backend import register_impl
from tilegym.kernel_utils import get_kernel_configs
from tilegym.logger import get_logger
from tilegym.ops.cutile.utils import cached_replace_hints

logger = get_logger(__name__)

# The tuned kernel is whichever of the per-expert / uniform / swap_ab variants
# measured faster for a given shape. Keyed on the shape scalars.
_ragged_block_scaled_bmm_tune_cache: dict = {}

# Use per-expert scheduling when the uniform schedule processes over 25% extra rows.
_PER_EXPERT_INFLATION_THRESHOLD = 1.25


def _is_large_m(total_m, Q):
    """Determine if average M is large enough for non-swapped configs."""
    average_m = total_m / Q
    is_large_m = average_m >= 256
    return is_large_m


def _compute_and_store_tile(
    a,  # Input matrix A [total_m, K] FP8
    b,  # Input matrix B [Q, N, K] FP8
    a_scale,  # Scale for A [total_m, k_tiles] FP32
    b_scale,  # Scale for B [Q, n_tiles, k_tiles] FP32
    c,  # Output matrix C [total_m, N]
    m_start,  # Segment start row (expert pid_q)
    m_end,  # Segment end row (expert pid_q)
    pid_m,  # Row tile index within the segment
    pid_n,  # Column tile index
    pid_q,  # Batch/expert index
    num_k_tiles,
    HAS_A_SCALE: ct.Constant[int],
    SWAP_AB: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
):
    """
    Compute one output tile C[pid_m, pid_n] for expert pid_q and store it.

    Shared K-loop body for all three scheduling variants (per-expert, uniform,
    swap_ab). Slices A/C/a_scale for the segment, runs the scaled FP8 MMA over
    the K dimension, and writes the result. SWAP_AB selects the (B @ A^T)^T path.
    """
    # Sliced views for A and C (and a_scale) using Array.slice
    Ai = a.slice(axis=0, start=m_start, stop=m_end)
    Ci = c.slice(axis=0, start=m_start, stop=m_end)
    if HAS_A_SCALE == 1:
        a_scale_i = a_scale.slice(axis=0, start=m_start, stop=m_end)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

    # N tile offset (element-level) for b_scale calculation
    n_offset = pid_n * BLOCK_N
    offs_bsn = n_offset // BLOCK_K

    # Zero accumulator for per-K MMA (reused each iteration)
    if SWAP_AB == 1:
        mma_zeros = ct.full((BLOCK_N, BLOCK_M), 0.0, dtype=ct.float32)
    else:
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
            b,
            index=(pid_q, n_offset // BLOCK_N, k_offset // BLOCK_K),
            shape=(1, BLOCK_N, BLOCK_K),
            order=(0, 1, 2),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_block_nk = ct.reshape(b_block_3d, (BLOCK_N, BLOCK_K))

        if SWAP_AB == 1:
            # swap_ab: compute (B @ A^T)^T
            a_block_t = ct.permute(a_block, (1, 0))
            c_swapped = ct.mma(b_block_nk, a_block_t, acc=mma_zeros)
            c_mma = ct.permute(c_swapped, (1, 0))
        else:
            # Transpose B to [BLOCK_K, BLOCK_N] then A [BLOCK_M, BLOCK_K] @ B = [BLOCK_M, BLOCK_N]
            b_block = ct.permute(b_block_nk, (1, 0))  # [BLOCK_K, BLOCK_N]
            c_mma = ct.mma(a_block, b_block, acc=mma_zeros)

        # Load and apply scales
        if HAS_A_SCALE == 1:
            a_scale_block = ct.load(
                a_scale_i,
                index=(pid_m, k),
                shape=(BLOCK_M, 1),
                padding_mode=ct.PaddingMode.ZERO,
            )
            b_scale_block = ct.load(
                b_scale,
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
                b_scale,
                index=(pid_q, offs_bsn, k),
                shape=(1, 1, 1),
                order=(0, 1, 2),
                padding_mode=ct.PaddingMode.ZERO,
            )
            b_scale_val = ct.reshape(b_scale_block, (1, 1))
            scale_ab = ct.broadcast_to(b_scale_val, (BLOCK_M, BLOCK_N))

        # Apply scale and accumulate
        acc = acc + c_mma * scale_ab

    # Convert to output dtype and store to C using TMA
    c_block = ct.astype(acc, c.dtype)
    ct.store(Ci, index=(pid_m, pid_n), tile=c_block)


@ct.kernel
def _ragged_block_scaled_bmm_kernel(
    a,  # Input matrix A [total_m, K] FP8
    b,  # Input matrix B [Q, N, K] FP8
    a_scale,  # Scale for A [total_m, k_tiles] FP32
    b_scale,  # Scale for B [Q, n_tiles, k_tiles] FP32
    c,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    q,  # Number of batches
    max_m,  # Host-side max segment size hint (kept for autotune cache key)
    max_m_device,  # 1-element int32 tensor (shape (1,)) — device-side ground truth for max(valid_m)
    n,  # Output N dimension
    HAS_A_SCALE: ct.Constant[int],  # Whether a_scale is provided (0 or 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    cuTile kernel for ragged block-scaled batched matrix multiplication.

    Performs (A * a_scale) @ (B * b_scale)^T where:
    - A is flattened FP8 with segment offsets (m_indptr defines boundaries)
    - B is batched FP8 [Q, N, K]
    - a_scale and b_scale are per-block scales
    - Output C is [total_m, N]

    Uses persistent scheduling with static grid and GROUP_SIZE_M tile swizzling.
    Uses Array.slice + TMA (ct.load/ct.store) for A and C access.

    Per-expert (grouped-GEMM) tile scheduling: the flat tile space is built from
    each expert's own valid_m via a running prefix sum over `m_indptr`
    (last_problem_end), so every expert contributes only its real tiles
    (cdiv(valid_m, BLOCK_M) * num_pid_n). This avoids the phantom out-of-range
    tiles a uniform per-expert bound would emit for less-loaded experts.
    `max_m` / `max_m_device` are unused here, retained in the signature only for
    API / autotune-cache-key compatibility.
    """
    pid = ct.bid(0)

    num_k_tiles = ct.num_tiles(a, axis=1, shape=(BLOCK_M, BLOCK_K))
    num_pid_n = ct.cdiv(n, BLOCK_N)
    num_programs = ct.num_blocks(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Per-expert tile scheduling: each expert contributes only its own real tiles
    # (cdiv(valid_m, BLOCK_M) * num_pid_n) via a running prefix sum. A uniform
    # per-expert bound would add phantom tiles for less-loaded experts and cause
    # persistent-loop wave quantization.
    tile_idx = pid
    last_problem_end = 0
    # Chain segment boundaries: expert q's end is expert q+1's start, so we do
    # Q+1 scalar loads total instead of 2*Q.
    m_start = ct.load(m_indptr, index=(0,), shape=(1,)).item()
    for pid_q in range(q):
        m_end = ct.load(m_indptr, index=(pid_q + 1,), shape=(1,)).item()
        valid_m = m_end - m_start
        num_pid_m = ct.cdiv(valid_m, BLOCK_M)
        tiles_this_expert = num_pid_m * num_pid_n

        # Process every flat tile this CTA owns within the current expert's range.
        while tile_idx >= last_problem_end and tile_idx < last_problem_end + tiles_this_expert:
            pid_in_batch = tile_idx - last_problem_end

            group_id = pid_in_batch // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

            pid_m = first_pid_m + (pid_in_batch % group_size_m_actual)
            pid_n = (pid_in_batch % num_pid_in_group) // group_size_m_actual

            _compute_and_store_tile(
                a,
                b,
                a_scale,
                b_scale,
                c,
                m_start,
                m_end,
                pid_m,
                pid_n,
                pid_q,
                num_k_tiles,
                HAS_A_SCALE,
                0,  # SWAP_AB
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
            )

            # Advance to this CTA's next flat tile (persistent stride).
            tile_idx = tile_idx + num_programs

        # Running prefix sum over experts; chain start = previous end.
        last_problem_end = last_problem_end + tiles_this_expert
        m_start = m_end


@ct.kernel
def _ragged_block_scaled_bmm_uniform_kernel(
    a,  # Input matrix A [total_m, K] FP8
    b,  # Input matrix B [Q, N, K] FP8
    a_scale,  # Scale for A [total_m, k_tiles] FP32
    b_scale,  # Scale for B [Q, n_tiles, k_tiles] FP32
    c,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    q,  # Number of batches
    max_m,  # Host-side max segment size hint (kept for autotune cache key)
    max_m_device,  # 1-element int32 tensor (shape (1,)) — device-side ground truth for max(valid_m)
    n,  # Output N dimension
    HAS_A_SCALE: ct.Constant[int],  # Whether a_scale is provided (0 or 1)
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    Uniform (per-batch) tile scheduling for the non-swap path.

    The flat tile space is sized to the per-batch bound max(valid_m); phantom
    tiles past each expert's real rows are masked by `if pid_m*BLOCK_M < valid_m`.
    For balanced routing (every expert ~= max_m) this emits the same real tiles
    as the per-expert schedule but without its prefix-sum + nested control flow,
    which the per-expert kernel pays as pure overhead. The host gates per-expert
    vs uniform on measured imbalance. Reads the bound from device-side
    `max_m_device` (defense-in-depth against a stale host `max_m` hint).
    """
    pid = ct.bid(0)

    num_k_tiles = ct.num_tiles(a, axis=1, shape=(BLOCK_M, BLOCK_K))
    max_m_runtime = ct.load(max_m_device, index=(0,), shape=(1,)).item()
    num_pid_m = ct.cdiv(max_m_runtime, BLOCK_M)
    num_pid_n = ct.cdiv(n, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * q
    num_programs = ct.num_blocks(0)
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
            _compute_and_store_tile(
                a,
                b,
                a_scale,
                b_scale,
                c,
                m_start,
                m_end,
                pid_m,
                pid_n,
                pid_q,
                num_k_tiles,
                HAS_A_SCALE,
                0,  # SWAP_AB
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
            )


@ct.kernel
def _ragged_block_scaled_bmm_swap_ab_kernel(
    a,  # Input matrix A [total_m, K] FP8
    b,  # Input matrix B [Q, N, K] FP8
    a_scale,  # Scale for A [total_m, k_tiles] FP32
    b_scale,  # Scale for B [Q, n_tiles, k_tiles] FP32
    c,  # Output matrix C [total_m, N]
    m_indptr,  # Segment offsets [Q+1], flattened 1D
    q,
    max_m,  # Host-side max segment size hint (kept for autotune cache key)
    max_m_device,  # 1-element int32 tensor (shape (1,)) — device-side ground truth for max(valid_m)
    n,
    HAS_A_SCALE: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    cuTile kernel for ragged block-scaled BMM with swap_ab optimization.
    Uses Array.slice + TMA (ct.load/ct.store) for A and C access.

    Defense-in-depth: this swap_ab path keeps uniform tiles-per-batch scheduling
    and computes the persistent-loop bound from the device-side `max_m_device`,
    not the host-side `max_m` hint, preventing silent output corruption when the
    host hint underestimates the actual per-batch max.
    """
    pid = ct.bid(0)

    num_k_tiles = ct.num_tiles(a, axis=1, shape=(BLOCK_M, BLOCK_K))
    max_m_runtime = ct.load(max_m_device, index=(0,), shape=(1,)).item()
    num_pid_m = ct.cdiv(max_m_runtime, BLOCK_M)
    num_pid_n = ct.cdiv(n, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * q
    num_programs = ct.num_blocks(0)
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
            _compute_and_store_tile(
                a,
                b,
                a_scale,
                b_scale,
                c,
                m_start,
                m_end,
                pid_m,
                pid_n,
                pid_q,
                num_k_tiles,
                HAS_A_SCALE,
                1,  # SWAP_AB
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
            )


def _ragged_block_scaled_bmm_autotune_configs():
    """
    Iterator of autotune configurations for ragged_block_scaled_bmm kernel.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # SM120/SM121 (Blackwell RTX-Pro / consumer) tuning space.
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
    elif gpu_capability[0] == 10:
        # SM100 tuning space:
        # - Keep large-M non-swapped paths (better arithmetic intensity)
        # - Add swapped small-M paths (better utilization for sparse/ragged tails)
        # - Explore occupancy=2 variants aggressively to hide ragged scheduling overhead
        for BM, BN, swap_ab, GROUP_M, num_ctas in [
            (256, 128, False, 8, 2),
            (256, 128, False, 8, 1),
            (128, 128, False, 8, 1),
            (128, 128, False, 8, 2),
            (128, 256, False, 4, 1),
            (64, 128, False, 8, 1),
            (64, 128, True, 4, 1),
            (32, 128, True, 4, 1),
            (64, 256, True, 2, 1),
            (32, 256, True, 2, 1),
            (16, 256, True, 2, 1),
        ]:
            for BK in [128]:
                for occupancy in [1, 2]:
                    yield SimpleNamespace(
                        BLOCK_M=BM,
                        BLOCK_N=BN,
                        BLOCK_K=BK,
                        GROUP_SIZE_M=GROUP_M,
                        swap_ab=swap_ab,
                        num_ctas=num_ctas,
                        occupancy=occupancy,
                    )
    elif gpu_capability == (9, 0):
        # SM90 (Hopper) tuning space.
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
                BLOCK_M=BM, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE_M=8, swap_ab=False, num_ctas=nc, occupancy=occ
            )
        # Swapped configs (for small M)
        for GM in [2, 4]:
            for BM in [16, 32, 64]:
                yield SimpleNamespace(
                    BLOCK_M=BM, BLOCK_N=256, BLOCK_K=128, GROUP_SIZE_M=GM, swap_ab=True, num_ctas=1, occupancy=1
                )


def _get_default_kernel_configs(total_m, Q, VEC_SIZE):
    """
    Get GPU-specific default kernel configs for non-autotune path.
    """
    gpu_capability = torch.cuda.get_device_capability()
    is_large_m = _is_large_m(total_m, Q)

    if gpu_capability in [(12, 0), (12, 1)]:
        # SM120/SM121 (Blackwell RTX-Pro / consumer) default.
        return {
            "BLOCK_M": 128,
            "BLOCK_N": 128,
            "BLOCK_K": VEC_SIZE,
            "GROUP_SIZE_M": 8,
            "swap_ab": False,
            "num_ctas": 1,
            "occupancy": 2,
        }
    elif gpu_capability[0] == 10:
        if is_large_m:
            # The num_ctas=2 / occupancy=2 tuning (and BLOCK_M=256 for large avg-M)
            # added in 1fb8021e regressed the block-scaled MoE BMM 19-47% on these
            # ragged fp8 shapes vs the pre-tuning default on datacenter Blackwell
            # (bit-exact B300/sm103 A/B). Restore the baseline num_ctas=1 /
            # occupancy=1 / BLOCK_M=128 config, which recovers it.
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
            # Small avg-M (avg_m < 256) sm10x (gpu_capability[0] == 10) path.
            # Restore the baseline num_ctas=1 / occupancy=1 / BLOCK_M=128 /
            # swap_ab=False config; the swapped BLOCK_M=64 / occupancy=2 tuning
            # regressed the small-M ragged fp8 shapes on datacenter Blackwell
            # vs this pre-tuning default.
            return {
                "BLOCK_M": 128,
                "BLOCK_N": 128,
                "BLOCK_K": VEC_SIZE,
                "GROUP_SIZE_M": 8,
                "swap_ab": False,
                "num_ctas": 1,
                "occupancy": 1,
            }
    elif gpu_capability == (9, 0):
        # SM90 (Hopper) default.
        if is_large_m:
            return {
                "BLOCK_M": 32,
                "BLOCK_N": 256,
                "BLOCK_K": VEC_SIZE,
                "GROUP_SIZE_M": 4,
                "swap_ab": True,
                "num_ctas": 1,
                "occupancy": 2,
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


def _ragged_block_scaled_bmm_autotune(
    stream, a, b, a_scale, b_scale, c, m_indptr, Q, max_m, max_m_device, N, K, total_m, has_a_scale
):
    """
    Autotuned launch for ragged block-scaled BMM.

    Tunes the per-expert, uniform and swap_ab kernels over their config spaces and
    launches whichever measured faster. All three compute the same product, so the
    choice only shifts fp8 accumulation grouping (results agree within tolerance).
    Which one wins is not predictable from the shape alone (it is non-monotonic in
    the per-batch M and differs per arch), so it is measured rather than guessed
    with a host-side gate.
    """
    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count

    def args_fn(cfg):
        return (
            a,
            b,
            a_scale,
            b_scale,
            c,
            m_indptr,
            Q,
            max_m,
            max_m_device,
            N,
            has_a_scale,
            cfg.BLOCK_M,
            cfg.BLOCK_N,
            cfg.BLOCK_K,
            cfg.GROUP_SIZE_M,
        )

    def grid_fn(cfg):
        num_pid_m = ct.cdiv(max_m, cfg.BLOCK_M)
        num_pid_n = ct.cdiv(N, cfg.BLOCK_N)
        total_tiles = num_pid_m * num_pid_n * Q
        num_programs = min(NUM_SMS // cfg.num_ctas, total_tiles) * cfg.occupancy
        # Never launch zero programs when there are rows to process, or the
        # output would be left silently unwritten (max_m can be 0 only for a
        # degenerate/empty batch, which correctly launches nothing).
        return (max(num_programs, 1) if total_m > 0 else num_programs, 1, 1)

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    all_configs = list(_ragged_block_scaled_bmm_autotune_configs())
    nonswap_configs = [cfg for cfg in all_configs if not cfg.swap_ab]
    swap_configs = [cfg for cfg in all_configs if cfg.swap_ab]

    cache_key = (Q, max_m, total_m, N, K, has_a_scale, a.dtype, str(a.device))
    if cache_key not in _ragged_block_scaled_bmm_tune_cache:
        best = None
        for kernel, configs in (
            (_ragged_block_scaled_bmm_kernel, nonswap_configs),
            (_ragged_block_scaled_bmm_uniform_kernel, nonswap_configs),
            (_ragged_block_scaled_bmm_swap_ab_kernel, swap_configs),
        ):
            if not configs:
                continue
            try:
                result = exhaustive_search(configs, stream, grid_fn, kernel, args_fn, hints_fn)
            except Exception as exc:
                # A whole config space can fail to build on a given arch (e.g. smem
                # limits); fall back to whichever variant did tune successfully.
                logger.debug("ragged_block_scaled_bmm autotune skipped %s: %s", kernel, exc)
                continue
            if best is None or result.best.mean_us < best[0]:
                best = (result.best.mean_us, kernel, result.best.config)
        if best is None:
            raise RuntimeError("ragged_block_scaled_bmm autotune found no working configuration")
        _, best_kernel, best_cfg = best
        _ragged_block_scaled_bmm_tune_cache[cache_key] = (
            best_cfg,
            best_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _ragged_block_scaled_bmm_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


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
    cuTile implementation of ragged block-scaled BMM.

    `max_m_device` is an optional [1]-shape int tensor with the device-side
    ground truth for max(per-batch valid_m). When provided, the kernel uses it
    for its persistent-loop bound — preventing silent corruption if the host-side `max_m` hint
    underestimates the actual per-batch max. When None, a fallback tensor is
    materialized from `max_m`.
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
    c = torch.empty((total_m, N), device=a.device, dtype=out_dtype)

    # Materialize fallback max_m_device if the caller didn't pass one. The
    # kernel always reads its grid bound from a device tensor (defense-in-depth).
    if max_m_device is None:
        max_m_device = torch.tensor([max_m], dtype=torch.int32, device=a.device)

    has_a_scale = 1 if a_scale is not None else 0
    if a_scale is None:
        a_scale = torch.empty(1, device=a.device, dtype=torch.float32)

    if is_autotune_enabled():
        # The per-expert, uniform and swap_ab kernels are all tuned and the faster
        # one wins. There is no host-side shape heuristic here on purpose: a
        # threshold on the per-batch M mis-routes MoE GEMMs, because which variant
        # is faster is not monotonic in M and differs per arch.
        _ragged_block_scaled_bmm_autotune(
            torch.cuda.current_stream(),
            a,
            b,
            a_scale,
            b_scale,
            c,
            m_indptr,
            Q,
            max_m,
            max_m_device,
            N,
            K_A,
            total_m,
            has_a_scale,
        )
        return c

    # Fixed default configs; the ratio gate selects per-expert vs uniform.
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
    num_pid_m = ct.cdiv(max_m, BLOCK_M)
    num_pid_n = ct.cdiv(N, BLOCK_N)
    tiles_per_batch = num_pid_m * num_pid_n
    total_tiles = tiles_per_batch * Q
    num_programs = min(NUM_SMS // num_ctas, total_tiles) * occupancy
    # Never launch zero programs when there are rows to process, or the output
    # would be left silently unwritten (max_m is 0 only for an empty batch).
    if total_m > 0:
        num_programs = max(num_programs, 1)

    grid = (num_programs, 1, 1)

    if swap_ab:
        kernel_fn = _ragged_block_scaled_bmm_swap_ab_kernel
    else:
        # Balanced routing -> uniform (cheaper); clearly imbalanced -> per-expert.
        use_per_expert = Q * max_m > total_m * _PER_EXPERT_INFLATION_THRESHOLD
        kernel_fn = _ragged_block_scaled_bmm_kernel if use_per_expert else _ragged_block_scaled_bmm_uniform_kernel

    hints = {}
    if num_ctas is not None:
        hints["num_ctas"] = num_ctas
    if occupancy is not None:
        hints["occupancy"] = occupancy
    kernel = cached_replace_hints(kernel_fn, **hints) if hints else kernel_fn

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            a,
            b,
            a_scale,
            b_scale,
            c,
            m_indptr,
            Q,
            max_m,
            max_m_device,
            N,
            has_a_scale,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_SIZE_M,
        ),
    )

    return c
