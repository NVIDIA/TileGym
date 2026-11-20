# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl

# Module-level tune caches:
#   _bs_gemm_scaled_tune_cache  → fast path (NT + ct.mma_scaled, 5D packed scales)
#   _bs_gemm_manual_tune_cache  → fallback (any layout + 2D scales + ct.mma)
_bs_gemm_scaled_tune_cache: dict = {}
_bs_gemm_manual_tune_cache: dict = {}


def cdiv(a, b):
    """Ceiling division helper function."""
    return (a + b - 1) // b


# ============================================================================
# Scaled MMA Kernel (fast path: NT layout + hardware ct.mma_scaled)
# ============================================================================


@ct.kernel
def bs_gemm_scaled_kernel_cutile(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,  # Raw 5D: [1, M//128, K//VEC_SIZE//4, 2, 256]
    b_scale_ptr,  # Raw 5D: [1, N//128, K//VEC_SIZE//4, 2, 256]
    M: ct.Constant[int],
    N: ct.Constant[int],
    num_k_tiles: ct.Constant[int],
    num_pid_m: ct.Constant[int],
    num_pid_n: ct.Constant[int],
    total_tiles: ct.Constant[int],
    num_programs: ct.Constant[int],
    ELEM_PER_BYTE_A: ct.Constant[int],
    ELEM_PER_BYTE_B: ct.Constant[int],
    MIXED_PREC: ct.Constant[int],
    SCALES_PER_BLOCK_K: ct.Constant[int],
    SCALE_REP_M: ct.Constant[int],  # BLOCK_M // 128
    SCALE_REP_N: ct.Constant[int],  # BLOCK_N // 128
    SCALE_REP_K: ct.Constant[int],  # BLOCK_K // VEC_SIZE // 4
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    CuTile kernel for block-scaled GEMM using ct.mma_scaled (fast path).

    Performs C = (A * A_scale) @ (B * B_scale) where:
    - A is FP8/FP4 [M, K] (packed uint8 for FP4)
    - B is FP8/FP4 [N, K] (NT layout; FP4 is packed uint8; permuted to [K, N])
    - A_scale / B_scale are raw 5D MX-swizzled scales
    - Output C is [M, N]

    In-kernel scale unswizzle: loads 5D packed scales via TMA and unswizzles
    in-register using a reshape+permute+reshape pattern. Uses ct.mma_scaled
    for hardware-accelerated scaled MMA.

    The outer `range(pid, total_tiles, num_programs)` loop makes this kernel
    persistent-friendly: with `num_programs == total_tiles` each CTA handles
    a single tile (non-persistent), with `num_programs < total_tiles` each
    CTA persistently iterates over multiple tiles (static persistent mode).
    """
    pid = ct.bid(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    zero_pad = ct.PaddingMode.ZERO

    for current_pid in range(pid, total_tiles, num_programs):
        # GROUP_SIZE_M tile swizzling for L2 locality
        group_id = current_pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (current_pid % group_size_m_actual)
        pid_n = (current_pid % num_pid_in_group) // group_size_m_actual

        acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

        for k in range(num_k_tiles):
            # ─── Load A ───
            if MIXED_PREC == 1:
                # Mixed: A is fp8, will be downcast to fp4 to match B
                # (ct.mma_scaled requires matching input dtypes).
                a_fp8 = ct.load(
                    a_ptr,
                    index=(pid_m, k),
                    shape=(BLOCK_M, BLOCK_K),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )
                # Cross-precision downcast may reduce accuracy in mixed-mode
                # workloads relative to native mixed-precision MMA paths.
                a_block = ct.astype(a_fp8, ct.float4_e2m1fn)
            elif ELEM_PER_BYTE_A == 2:
                # FP4: load packed uint8 [BLOCK_M, BLOCK_K//2] and unpack to fp4.
                a_bytes = ct.load(
                    a_ptr,
                    index=(pid_m, k),
                    shape=(BLOCK_M, BLOCK_K // 2),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )
                a_flat = ct.reshape(a_bytes, (-1,))
                a_unpacked = ct.unpack_from_bytes(a_flat, ct.float4_e2m1fn)
                a_block = ct.reshape(a_unpacked, (BLOCK_M, BLOCK_K))
            else:
                # FP8: load directly as [BLOCK_M, BLOCK_K].
                a_block = ct.load(
                    a_ptr,
                    index=(pid_m, k),
                    shape=(BLOCK_M, BLOCK_K),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )

            # ─── Load B (NT layout: stored as [N, K], load and permute to [K, N]) ───
            if ELEM_PER_BYTE_B == 2:
                # FP4: load packed uint8 [BLOCK_N, BLOCK_K//2] and unpack to fp4.
                b_bytes = ct.load(
                    b_ptr,
                    index=(pid_n, k),
                    shape=(BLOCK_N, BLOCK_K // 2),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )
                b_flat = ct.reshape(b_bytes, (-1,))
                b_unpacked = ct.unpack_from_bytes(b_flat, ct.float4_e2m1fn)
                b_nk = ct.reshape(b_unpacked, (BLOCK_N, BLOCK_K))
            else:
                # FP8: load as [BLOCK_N, BLOCK_K].
                b_nk = ct.load(
                    b_ptr,
                    index=(pid_n, k),
                    shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )
            b_block = ct.permute(b_nk, (1, 0))  # [BLOCK_K, BLOCK_N]

            # ─── In-kernel scale unswizzle ───
            a_scale_5d = ct.load(
                a_scale_ptr,
                index=(0, pid_m, k, 0, 0),
                shape=(1, SCALE_REP_M, SCALE_REP_K, 2, 256),
                padding_mode=zero_pad,
            )
            a_scale_6d = ct.reshape(a_scale_5d, (1, SCALE_REP_M, SCALE_REP_K, 32, 4, 4))
            a_scale_perm = ct.permute(a_scale_6d, (0, 1, 4, 3, 2, 5))
            a_scale_block = ct.reshape(a_scale_perm, (BLOCK_M, SCALES_PER_BLOCK_K))

            b_scale_5d = ct.load(
                b_scale_ptr,
                index=(0, pid_n, k, 0, 0),
                shape=(1, SCALE_REP_N, SCALE_REP_K, 2, 256),
                padding_mode=zero_pad,
            )
            b_scale_6d = ct.reshape(b_scale_5d, (1, SCALE_REP_N, SCALE_REP_K, 32, 4, 4))
            b_scale_perm = ct.permute(b_scale_6d, (0, 1, 4, 3, 2, 5))
            b_scale_2d = ct.reshape(b_scale_perm, (BLOCK_N, SCALES_PER_BLOCK_K))
            # ct.mma_scaled expects b_scale as (K_scale, N) to match b_block (K, N).
            b_scale_block = ct.permute(b_scale_2d, (1, 0))

            # Hardware-accelerated scaled MMA
            acc = ct.mma_scaled(a_block, a_scale_block, b_block, b_scale_block, acc)

        c_block = ct.astype(acc, c_ptr.dtype)
        ct.store(
            c_ptr,
            index=(pid_m, pid_n),
            tile=c_block,
            order=(0, 1),
        )


# ============================================================================
# Manual MMA Kernel (fallback path: any layout + 2D scale + ct.mma)
# ============================================================================


@ct.kernel
def bs_gemm_manual_kernel_cutile(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,  # 2D: [M, K // VEC_SIZE]
    b_scale_ptr,  # 2D: [N, K // VEC_SIZE]
    M: ct.Constant[int],
    N: ct.Constant[int],
    num_k_tiles: ct.Constant[int],
    num_pid_m: ct.Constant[int],
    num_pid_n: ct.Constant[int],
    total_tiles: ct.Constant[int],
    num_programs: ct.Constant[int],
    VEC_SIZE: ct.Constant[int],
    SCALES_PER_BLOCK_K: ct.Constant[int],  # BLOCK_K // VEC_SIZE
    ELEM_PER_BYTE_A: ct.Constant[int],
    ELEM_PER_BYTE_B: ct.Constant[int],
    transpose_a: ct.Constant[int],
    transpose_b: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
):
    """
    Manual ct.mma fallback for block-scaled GEMM.

    Used when ct.mma_scaled's NT-only constraint can't be met. Computes the
    GEMM as a sequence of per-VEC_SIZE sub-tile MMAs with manual scale
    broadcast on the operands (fp32 path, then accumulate).

    Supports arbitrary transpose_a / transpose_b via conditional loads, and
    both FP8 (e4m3) and FP4 (e2m1). FP4 operands are loaded as packed
    uint8 and unpacked via ct.unpack_from_bytes; the unpack pattern
    depends on the transpose flag (each byte contains two consecutive
    K-axis values from the original unpacked tensor).
    """
    pid = ct.bid(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    zero_pad = ct.PaddingMode.ZERO

    for current_pid in range(pid, total_tiles, num_programs):
        group_id = current_pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m_actual = ct.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (current_pid % group_size_m_actual)
        pid_n = (current_pid % num_pid_in_group) // group_size_m_actual

        acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)

        for k in range(num_k_tiles):
            for s in range(SCALES_PER_BLOCK_K):
                # Tile-unit index of the current VEC_SIZE-wide K sub-block; every
                # load below uses a K-dim tile extent of VEC_SIZE (or VEC_SIZE//2
                # bytes for packed FP4), so the same count addresses all of them.
                ks = k * SCALES_PER_BLOCK_K + s

                # Load per-block scales (2D layout).
                a_scale_s = ct.load(
                    a_scale_ptr,
                    index=(pid_m, ks),
                    shape=(BLOCK_M, 1),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )
                b_scale_s = ct.load(
                    b_scale_ptr,
                    index=(pid_n, ks),
                    shape=(BLOCK_N, 1),
                    order=(0, 1),
                    padding_mode=zero_pad,
                )

                # ─── Load A sub-block → (BLOCK_M, VEC_SIZE) ───
                if ELEM_PER_BYTE_A == 2:
                    # FP4: load packed uint8 and unpack along original K.
                    if transpose_a == 1:
                        # Packed storage is (K//2, M); load (VEC_SIZE//2, BLOCK_M) bytes,
                        # unpack so result[m, k] = original-K value.
                        a_bytes = ct.load(
                            a_ptr,
                            index=(ks, pid_m),
                            shape=(VEC_SIZE // 2, BLOCK_M),
                            order=(0, 1),
                            padding_mode=zero_pad,
                        )
                        a_flat = ct.reshape(a_bytes, (-1,))
                        a_unp = ct.unpack_from_bytes(a_flat, ct.float4_e2m1fn)
                        a_3d = ct.reshape(a_unp, (VEC_SIZE // 2, BLOCK_M, 2))
                        a_3d_perm = ct.permute(a_3d, (1, 0, 2))  # (BLOCK_M, VEC_SIZE//2, 2)
                        a_sub = ct.reshape(a_3d_perm, (BLOCK_M, VEC_SIZE))
                    else:
                        a_bytes = ct.load(
                            a_ptr,
                            index=(pid_m, ks),
                            shape=(BLOCK_M, VEC_SIZE // 2),
                            order=(0, 1),
                            padding_mode=zero_pad,
                        )
                        a_flat = ct.reshape(a_bytes, (-1,))
                        a_unp = ct.unpack_from_bytes(a_flat, ct.float4_e2m1fn)
                        a_sub = ct.reshape(a_unp, (BLOCK_M, VEC_SIZE))
                else:
                    # FP8: direct load.
                    if transpose_a == 1:
                        a_sub_km = ct.load(
                            a_ptr,
                            index=(ks, pid_m),
                            shape=(VEC_SIZE, BLOCK_M),
                            order=(0, 1),
                            padding_mode=zero_pad,
                        )
                        a_sub = ct.permute(a_sub_km, (1, 0))  # [BLOCK_M, VEC_SIZE]
                    else:
                        a_sub = ct.load(
                            a_ptr,
                            index=(pid_m, ks),
                            shape=(BLOCK_M, VEC_SIZE),
                            order=(0, 1),
                            padding_mode=zero_pad,
                        )

                # ─── Load B sub-block → (VEC_SIZE, BLOCK_N) ───
                if ELEM_PER_BYTE_B == 2:
                    # FP4: load packed uint8 and unpack along original K.
                    if transpose_b == 1:
                        # Packed storage is (N, K//2); load (BLOCK_N, VEC_SIZE//2) bytes,
                        # unpack so result[n, k] = original-K value, then permute to (K, N).
                        b_bytes = ct.load(
                            b_ptr,
                            index=(pid_n, ks),
                            shape=(BLOCK_N, VEC_SIZE // 2),
                            order=(0, 1),
                            padding_mode=zero_pad,
                        )
                        b_flat = ct.reshape(b_bytes, (-1,))
                        b_unp = ct.unpack_from_bytes(b_flat, ct.float4_e2m1fn)
                        b_nv = ct.reshape(b_unp, (BLOCK_N, VEC_SIZE))
                        b_sub = ct.permute(b_nv, (1, 0))  # [VEC_SIZE, BLOCK_N]
                    else:
                        # Packed storage is (K//2, N); load (VEC_SIZE//2, BLOCK_N) bytes.
                        b_bytes = ct.load(
                            b_ptr,
                            index=(ks, pid_n),
                            shape=(VEC_SIZE // 2, BLOCK_N),
                            order=(0, 1),
                            padding_mode=zero_pad,
                        )
                        b_flat = ct.reshape(b_bytes, (-1,))
                        b_unp = ct.unpack_from_bytes(b_flat, ct.float4_e2m1fn)
                        b_3d = ct.reshape(b_unp, (VEC_SIZE // 2, BLOCK_N, 2))
                        b_3d_perm = ct.permute(b_3d, (0, 2, 1))  # (VEC_SIZE//2, 2, BLOCK_N)
                        b_sub = ct.reshape(b_3d_perm, (VEC_SIZE, BLOCK_N))
                else:
                    # FP8: direct load.
                    if transpose_b == 1:
                        b_sub_nv = ct.load(
                            b_ptr,
                            index=(pid_n, ks),
                            shape=(BLOCK_N, VEC_SIZE),
                            order=(0, 1),
                            padding_mode=zero_pad,
                        )
                        b_sub = ct.permute(b_sub_nv, (1, 0))  # [VEC_SIZE, BLOCK_N]
                    else:
                        b_sub = ct.load(
                            b_ptr,
                            index=(ks, pid_n),
                            shape=(VEC_SIZE, BLOCK_N),
                            order=(0, 1),
                            padding_mode=zero_pad,
                        )

                # Upcast operands and scales to fp32 and apply scale broadcast.
                a_sub_f32 = ct.astype(a_sub, ct.float32)
                b_sub_f32 = ct.astype(b_sub, ct.float32)
                a_scale_f32 = ct.astype(a_scale_s, ct.float32)
                b_scale_f32 = ct.astype(b_scale_s, ct.float32)

                a_scale_bc = ct.broadcast_to(a_scale_f32, (BLOCK_M, VEC_SIZE))
                b_scale_t = ct.permute(b_scale_f32, (1, 0))  # [1, BLOCK_N]
                b_scale_bc = ct.broadcast_to(b_scale_t, (VEC_SIZE, BLOCK_N))

                a_scaled = a_sub_f32 * a_scale_bc
                b_scaled = b_sub_f32 * b_scale_bc

                acc = ct.mma(a_scaled, b_scaled, acc=acc)

        c_block = ct.astype(acc, c_ptr.dtype)
        ct.store(
            c_ptr,
            index=(pid_m, pid_n),
            tile=c_block,
            order=(0, 1),
        )


# ============================================================================
# Autotune configurations
# ============================================================================


def _bs_gemm_scaled_autotune_configs():
    """
    Iterator of autotune configurations for the ct.mma_scaled scaled kernel.
    Configurations tuned for Blackwell-family block-scaled GEMM.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability == (10, 7):
        # SM107-only tuning.
        for BM, BN, nc, occ, gsm, tiles_per_program in [
            (128, 128, 1, 1, 8, 1),
            (128, 128, 1, 2, 8, 1),
            (128, 256, 1, 1, 8, 1),
            (128, 256, 1, 2, 8, 1),
            (256, 128, 2, 1, 8, 1),
            (256, 256, 2, 1, 8, 1),
            (128, 128, 1, 4, 1, 1),
            (128, 256, 1, 2, 1, 1),
            (256, 256, 2, 1, 4, 1),
            (256, 256, 2, 1, 4, 2),
            (256, 256, 2, 1, 8, 2),
            (256, 256, 2, 1, 4, 4),
            (256, 256, 2, 1, 8, 4),
            (256, 256, 2, 1, 4, 8),
            (256, 256, 2, 1, 8, 8),
        ]:
            yield SimpleNamespace(
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=128,
                GROUP_SIZE_M=gsm,
                num_ctas=nc,
                occupancy=occ,
                tiles_per_program=tiles_per_program,
            )
        return

    if gpu_capability[0] >= 10:
        # Blackwell family (sm_100/103/120/121)
        # BLOCK_K kept at 128 to keep TMA alignment for block-scaled ops.
        for BM, BN, nc, occ, gsm in [
            (128, 128, 1, 1, 8),
            (128, 128, 1, 2, 8),
            (128, 256, 1, 1, 8),
            (128, 256, 1, 2, 8),
            (256, 128, 2, 1, 8),
            (256, 256, 2, 1, 8),
            (128, 128, 1, 4, 1),
            (128, 256, 1, 2, 1),
            # GROUP_SIZE_M swizzle variants of the 256x256 workhorse tile. The
            # best L2-locality group width depends on the shape: large square
            # shapes prefer a wider group, tall/thin and K-heavy shapes a
            # narrower one. Autotune picks the best per shape; gsm=8 above stays
            # the default.
            (256, 256, 2, 1, 2),
            (256, 256, 2, 1, 4),
            (256, 256, 2, 1, 16),
        ]:
            yield SimpleNamespace(
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=128,
                GROUP_SIZE_M=gsm,
                num_ctas=nc,
                occupancy=occ,
            )
    elif gpu_capability == (9, 0):
        # Hopper
        for BM, BN, BK in [(128, 256, 128), (128, 128, 128)]:
            for occupancy in [1, 2]:
                yield SimpleNamespace(
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    BLOCK_K=BK,
                    GROUP_SIZE_M=8,
                    num_ctas=1,
                    occupancy=occupancy,
                )
    else:
        for BM, BN, BK in [(128, 128, 128)]:
            for occupancy in [1, 2]:
                yield SimpleNamespace(
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    BLOCK_K=BK,
                    GROUP_SIZE_M=8,
                    num_ctas=1,
                    occupancy=occupancy,
                )


def _bs_gemm_manual_autotune_configs():
    """
    Iterator of autotune configurations for the manual ct.mma fallback kernel.
    Conservative tile sizes (fp32 register pressure is higher than mma_scaled).
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability[0] >= 10:
        for BM, BN, BK, nc, occ, gsm in [
            (128, 128, 128, 1, 1, 8),
            (128, 128, 128, 1, 2, 8),
            (64, 128, 128, 1, 2, 8),
            (128, 64, 128, 1, 2, 8),
        ]:
            yield SimpleNamespace(
                BLOCK_M=BM,
                BLOCK_N=BN,
                BLOCK_K=BK,
                GROUP_SIZE_M=gsm,
                num_ctas=nc,
                occupancy=occ,
            )
    else:
        for BM, BN, BK in [(128, 128, 128), (64, 128, 128)]:
            for occupancy in [1, 2]:
                yield SimpleNamespace(
                    BLOCK_M=BM,
                    BLOCK_N=BN,
                    BLOCK_K=BK,
                    GROUP_SIZE_M=8,
                    num_ctas=1,
                    occupancy=occupancy,
                )


# ============================================================================
# Helpers
# ============================================================================


def _resolve_dtype_params(DTYPE_A: str, DTYPE_B: str, VEC_SIZE: int):
    """
    Compute ELEM_PER_BYTE_A/B and MIXED_PREC, plus validate the combo.

    DTYPE_A / DTYPE_B are one of "e4m3" (FP8 E4M3), "e5m2" (FP8 E5M2), or
    "e2m1" (FP4). Per the OCP MX-format spec, mxfp8 has two FP8 variants
    (E4M3 and E5M2) both paired with E8M0 scales at VEC_SIZE=32.
    """
    _supported = ("e4m3", "e5m2", "e2m1")
    assert DTYPE_A in _supported, f"Unsupported DTYPE_A: {DTYPE_A!r}. Must be one of {_supported}."
    assert DTYPE_B in _supported, f"Unsupported DTYPE_B: {DTYPE_B!r}. Must be one of {_supported}."
    ELEM_PER_BYTE_A = 2 if DTYPE_A == "e2m1" else 1
    ELEM_PER_BYTE_B = 2 if DTYPE_B == "e2m1" else 1
    # MIXED_PREC = A is fp8 and B is fp4. A will be downcast to fp4 inside the
    # kernel (ct.mma_scaled requires matching input dtypes).
    MIXED_PREC = 1 if ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2 else 0
    # The reverse (A fp4, B fp8) is uncommon in practice and would require a
    # symmetric downcast path; not implemented.
    assert not (ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 1), "A=fp4 × B=fp8 is not supported by the CuTile backend"
    return ELEM_PER_BYTE_A, ELEM_PER_BYTE_B, MIXED_PREC


def _maybe_view_scale_dtype(scale: torch.Tensor, scale_dtype_hint: str) -> torch.Tensor:
    """
    Reinterpret a uint8 scale tensor as the dtype expected by ct.mma_scaled.
    - "nvfp4"-style scales (V=16, paired with FP4): float8_e4m3fn
    - MX-style scales (V=32): float8_e8m0fnu
    """
    if scale_dtype_hint == "e4m3":
        target = torch.float8_e4m3fn
    else:
        target = torch.float8_e8m0fnu
    if scale.dtype == torch.uint8:
        return scale.view(target)
    if scale.dtype != target:
        return scale.view(torch.uint8).view(target)
    return scale


def _scale_values_f32(scale: torch.Tensor, vec_size: int) -> torch.Tensor:
    """
    Decode a scale tensor to fp32 values. Raw uint8 encodings are reinterpreted
    to the fp8 scale dtype first — a bare .to(float32) on uint8 would decode the
    e8m0 exponent byte as an integer (127 -> 127.0 instead of 1.0).
    """
    if scale.dtype != torch.float32:
        scale = _maybe_view_scale_dtype(scale, "e4m3" if vec_size == 16 else "e8m0fnu")
    return scale.to(torch.float32)


def _unpack_scale_5d_to_2d(packed: torch.Tensor) -> torch.Tensor:
    """
    Unswizzle a 5D MX-swizzled scale tensor into the 2D [M_or_N, K//VEC_SIZE]
    layout consumed by the manual fallback kernel.

    The 5D layout is `[1, num_chunks, num_k_chunks, 2, 256]` (uint8) and is
    equivalent to a reshape+permute+reshape sequence:
        reshape(rep, rep_k, 32, 4, 4) -> permute(0, 3, 2, 1, 4) -> reshape(M_or_N, K//VEC_SIZE)
    The scaled kernel does the same unswizzle inside the kernel via in-register
    reshape+permute+reshape; here we do it on the host since the manual kernel
    consumes the unpacked 2D form directly.
    """
    packed = packed.squeeze(0)  # [num_chunks, num_k_chunks, 2, 256]
    packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)  # [num_chunks, num_k_chunks, 32, 4, 4]
    num_chunk, num_chunk_k = packed.shape[0], packed.shape[1]
    return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk * 128, num_chunk_k * 4).contiguous()


def _compute_num_programs(
    total_tiles: int,
    num_ctas: int,
    occupancy: int,
    persistent_mode: str,
    tiles_per_program: int = 1,
) -> int:
    """
    Compute the launch grid size (num_programs) for the persistent-loop kernel.

    - persistent_mode == 'none': num_programs =
      ceil(total_tiles / tiles_per_program).
    - persistent_mode == 'static': num_programs = min(NUM_SMS // num_ctas,
      total_tiles) * occupancy (each CTA persistently iterates over multiple
      tiles, saturating the GPU).
    """
    if tiles_per_program <= 0:
        raise ValueError("tiles_per_program must be >= 1")
    if persistent_mode is None or persistent_mode.lower() == "none":
        return cdiv(total_tiles, tiles_per_program)
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    # Ensure num_programs >= 1 — cuTile requires positive step in range().
    return max(1, min(num_sms // num_ctas, total_tiles) * occupancy)


# ============================================================================
# Host impl: bs_matmul (5D packed scales)
# ============================================================================


def _launch_scaled_path(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scale_5d: torch.Tensor,
    b_scale_5d: torch.Tensor,
    M: int,
    N: int,
    K: int,
    VEC_SIZE: int,
    DTYPE_A: str,
    DTYPE_B: str,
    persistent_mode: str,
):
    """Launch the ct.mma_scaled fast path. Caller must ensure NT layout."""
    ELEM_PER_BYTE_A, ELEM_PER_BYTE_B, MIXED_PREC = _resolve_dtype_params(DTYPE_A, DTYPE_B, VEC_SIZE)

    # Scale dtype reinterpretation — ct.mma_scaled requires f8e8m0fnu (MX) or
    # f8e4m3fn (NVFP4-style) on the inputs.
    scale_hint = "e4m3" if (VEC_SIZE == 16) else "e8m0fnu"
    a_scale_5d = _maybe_view_scale_dtype(a_scale_5d, scale_hint)
    b_scale_5d = _maybe_view_scale_dtype(b_scale_5d, scale_hint)

    def args_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        BK = cfg.BLOCK_K
        GSM = cfg.GROUP_SIZE_M
        num_pid_m = cdiv(M, BM)
        num_pid_n = cdiv(N, BN)
        num_k_tiles = cdiv(K, BK)
        total_tiles = num_pid_m * num_pid_n
        num_programs = _compute_num_programs(
            total_tiles,
            cfg.num_ctas,
            cfg.occupancy,
            persistent_mode,
            getattr(cfg, "tiles_per_program", 1),
        )
        SCALES_PER_BK = BK // VEC_SIZE
        SCALE_REP_M = BM // 128
        SCALE_REP_N = BN // 128
        SCALE_REP_K = BK // VEC_SIZE // 4
        return (
            a,
            b,
            c,
            a_scale_5d,
            b_scale_5d,
            M,
            N,
            num_k_tiles,
            num_pid_m,
            num_pid_n,
            total_tiles,
            num_programs,
            ELEM_PER_BYTE_A,
            ELEM_PER_BYTE_B,
            MIXED_PREC,
            SCALES_PER_BK,
            SCALE_REP_M,
            SCALE_REP_N,
            SCALE_REP_K,
            BM,
            BN,
            BK,
            GSM,
        )

    def grid_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        num_pid_m = cdiv(M, BM)
        num_pid_n = cdiv(N, BN)
        total_tiles = num_pid_m * num_pid_n
        num_programs = _compute_num_programs(
            total_tiles,
            cfg.num_ctas,
            cfg.occupancy,
            persistent_mode,
            getattr(cfg, "tiles_per_program", 1),
        )
        return (num_programs, 1, 1)

    stream = torch.cuda.current_stream()
    cache_key = (M, N, K, DTYPE_A, DTYPE_B, VEC_SIZE, "scaled", persistent_mode, a.dtype, str(a.device))
    if cache_key not in _bs_gemm_scaled_tune_cache:
        if is_autotune_disabled():
            best_cfg = next(iter(_bs_gemm_scaled_autotune_configs()))
        else:
            result = exhaustive_search(
                list(_bs_gemm_scaled_autotune_configs()),
                stream,
                grid_fn,
                bs_gemm_scaled_kernel_cutile,
                args_fn,
                lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
        _bs_gemm_scaled_tune_cache[cache_key] = (
            best_cfg,
            ct.kernel(
                bs_gemm_scaled_kernel_cutile._pyfunc,
                num_ctas=best_cfg.num_ctas,
                occupancy=best_cfg.occupancy,
            ),
        )
    best_cfg, tuned_kernel = _bs_gemm_scaled_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


def _launch_manual_path(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scale_2d: torch.Tensor,
    b_scale_2d: torch.Tensor,
    M: int,
    N: int,
    K: int,
    VEC_SIZE: int,
    DTYPE_A: str,
    DTYPE_B: str,
    transpose_a: bool,
    transpose_b: bool,
    persistent_mode: str,
):
    """Launch the manual ct.mma fallback (arbitrary layout, 2D scales, FP8 or FP4)."""
    transpose_a_int = 1 if transpose_a else 0
    transpose_b_int = 1 if transpose_b else 0
    ELEM_PER_BYTE_A = 2 if DTYPE_A == "e2m1" else 1
    ELEM_PER_BYTE_B = 2 if DTYPE_B == "e2m1" else 1
    # A=fp4 × B=fp8 not supported by this fallback (uncommon in practice).
    assert not (ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 1), (
        "Manual fallback does not support A=fp4 × B=fp8 mixed precision"
    )

    def args_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        BK = cfg.BLOCK_K
        GSM = cfg.GROUP_SIZE_M
        num_pid_m = cdiv(M, BM)
        num_pid_n = cdiv(N, BN)
        num_k_tiles = cdiv(K, BK)
        total_tiles = num_pid_m * num_pid_n
        num_programs = _compute_num_programs(total_tiles, cfg.num_ctas, cfg.occupancy, persistent_mode)
        SCALES_PER_BK = BK // VEC_SIZE
        return (
            a,
            b,
            c,
            a_scale_2d,
            b_scale_2d,
            M,
            N,
            num_k_tiles,
            num_pid_m,
            num_pid_n,
            total_tiles,
            num_programs,
            VEC_SIZE,
            SCALES_PER_BK,
            ELEM_PER_BYTE_A,
            ELEM_PER_BYTE_B,
            transpose_a_int,
            transpose_b_int,
            BM,
            BN,
            BK,
            GSM,
        )

    def grid_fn(cfg):
        BM = cfg.BLOCK_M
        BN = cfg.BLOCK_N
        num_pid_m = cdiv(M, BM)
        num_pid_n = cdiv(N, BN)
        total_tiles = num_pid_m * num_pid_n
        num_programs = _compute_num_programs(total_tiles, cfg.num_ctas, cfg.occupancy, persistent_mode)
        return (num_programs, 1, 1)

    stream = torch.cuda.current_stream()
    cache_key = (
        M,
        N,
        K,
        DTYPE_A,
        DTYPE_B,
        VEC_SIZE,
        transpose_a,
        transpose_b,
        "manual",
        persistent_mode,
        a.dtype,
        str(a.device),
    )
    if cache_key not in _bs_gemm_manual_tune_cache:
        if is_autotune_disabled():
            best_cfg = next(iter(_bs_gemm_manual_autotune_configs()))
        else:
            result = exhaustive_search(
                list(_bs_gemm_manual_autotune_configs()),
                stream,
                grid_fn,
                bs_gemm_manual_kernel_cutile,
                args_fn,
                lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
        _bs_gemm_manual_tune_cache[cache_key] = (
            best_cfg,
            ct.kernel(
                bs_gemm_manual_kernel_cutile._pyfunc,
                num_ctas=best_cfg.num_ctas,
                occupancy=best_cfg.occupancy,
            ),
        )
    best_cfg, tuned_kernel = _bs_gemm_manual_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


def _bs_matmul_impl_cutile(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    c_dtype: torch.dtype,
    config: dict,
):
    """
    CuTile implementation of bs_matmul (5D packed scale layout).

    Dispatches to the ct.mma_scaled fast path when the layout is NT
    (transpose_a=False, transpose_b=True), and to the manual ct.mma fallback
    for any other layout. Both paths support FP8 x FP8 and FP4 x FP4 operands.
    A=fp4 with B=fp8 is not supported.
    """
    sm_major = torch.cuda.get_device_capability()[0]
    if sm_major not in (10, 12):
        msg = (
            f"CuTile bs_matmul requires SM10x (Blackwell) or SM12x GPUs for "
            f"ct.mma_scaled; current device is SM{sm_major}x."
        )
        raise ValueError(msg)

    assert config is not None, "bs_matmul requires a config dict"

    M, N, K = config["M"], config["N"], config["K"]
    VEC_SIZE = config["VEC_SIZE"]
    transpose_a = config.get("transpose_a", False)
    transpose_b = config.get("transpose_b", True)
    DTYPE_A = config.get("DTYPE_A", "e4m3")
    DTYPE_B = config.get("DTYPE_B", "e4m3")
    persistent_mode = config.get("persistent_mode", "none")

    c = torch.empty((M, N), device=a.device, dtype=c_dtype)

    is_nt = (not transpose_a) and transpose_b
    if is_nt:
        # Fast path: NT + ct.mma_scaled with raw 5D packed scales.
        assert a_scale.ndim == 5 and b_scale.ndim == 5, (
            "CuTile fast path expects 5D packed scales [1, *//128, K//VEC_SIZE//4, 2, 256]"
        )
        _, sa1, sa2, sa3, sa4 = a_scale.shape
        _, sb1, sb2, sb3, sb4 = b_scale.shape
        assert sa3 == 2 and sa4 == 256 and sb3 == 2 and sb4 == 256, "incompatible scale inner dims"
        assert sa1 * 128 == M, "a_scale dim 1 must be M//128"
        assert sb1 * 128 == N, "b_scale dim 1 must be N//128"
        assert sa2 * 4 * VEC_SIZE == K and sb2 * 4 * VEC_SIZE == K, "incompatible scale K dim"
        _launch_scaled_path(a, b, c, a_scale, b_scale, M, N, K, VEC_SIZE, DTYPE_A, DTYPE_B, persistent_mode)
    else:
        # Fallback: arbitrary layout via manual ct.mma. Supports FP8 + FP4.
        # Unpack 5D scales to 2D for the manual kernel.
        if a_scale.ndim == 5:
            a_scale_2d = _unpack_scale_5d_to_2d(_scale_values_f32(a_scale, VEC_SIZE))[:M, : K // VEC_SIZE]
        else:
            a_scale_2d = _scale_values_f32(a_scale, VEC_SIZE)
        if b_scale.ndim == 5:
            b_scale_2d = _unpack_scale_5d_to_2d(_scale_values_f32(b_scale, VEC_SIZE))[:N, : K // VEC_SIZE]
        else:
            b_scale_2d = _scale_values_f32(b_scale, VEC_SIZE)
        _launch_manual_path(
            a,
            b,
            c,
            a_scale_2d,
            b_scale_2d,
            M,
            N,
            K,
            VEC_SIZE,
            DTYPE_A,
            DTYPE_B,
            transpose_a,
            transpose_b,
            persistent_mode,
        )

    return c


# ============================================================================
# Registrations
# ============================================================================


@register_impl("flashinfer.gemm.bs_matmul", backend="cutile")
def bs_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    c_dtype: torch.dtype,
    config: dict = None,
):
    """
    CuTile implementation of block-scaled matmul (5D packed scales).

    Routing:
    - NT layout (transpose_a=False, transpose_b=True): hardware ct.mma_scaled
      fast path, supports FP8/FP4 and persistent_mode in {'none', 'static'}.
    - Other layouts: manual ct.mma fallback, supports FP8/FP4 and
      persistent_mode in {'none', 'static'}.
    - persistent_mode == 'dynamic': not supported (raises ValueError; this is
      a library-wide capability gap that holds across backends).
    """
    if config is None:
        config = {}

    persistent_mode = config.get("persistent_mode", "none")
    if persistent_mode is not None and persistent_mode.lower() == "dynamic":
        # TODO: replace this once dynamic persistent mode is supported
        raise ValueError("Dynamic persistent mode is not supported yet.")

    return _bs_matmul_impl_cutile(a, b, a_scale, b_scale, c_dtype, config)
