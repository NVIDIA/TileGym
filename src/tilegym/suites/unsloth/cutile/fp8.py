# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.

"""
FP8 quantization CuTile kernels.

CuTile kernels:
  - weight_dequant_kernel_ct: block-wise FP8 weight dequantization
  - act_quant_kernel_ct: activation quantization to FP8
  - w8a8_block_fp8_matmul_kernel_ct: block-wise FP8 matmul (W8A8)
"""

import math

import cuda.tile as ct
import torch
from cuda.tile._numeric_semantics import RoundingMode as RMd

from tilegym.backend import register_impl

from .ct_ops import autotune_configs
from .ct_ops import next_power_of_2

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


# ---- CuTile kernel: weight dequantization ----


@ct.kernel
def weight_dequant_kernel_ct(
    x,  # (M, N) FP8
    s,  # (M_blocks, N_blocks) float32 scales
    y,  # (M, N) output
    M: ConstInt,
    N: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    """
    Block-wise FP8 weight dequantization: y = x * s[block_m, block_n]

    Each program processes one (BLOCK_SIZE, BLOCK_SIZE) tile.
    """
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    offs_m = pid_m * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
    offs_n = pid_n * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Load FP8 block
    x_tile = ct.gather(
        x,
        (offs_m[:, None], offs_n[None, :]),
        check_bounds=True,
        padding_value=0,
    )
    x_f32 = ct.astype(x_tile, ct.float32)

    # Load scale (scalar for this block)
    scale = ct.gather(s, (pid_m, pid_n))

    # Dequantize and cast to output dtype
    y_f32 = x_f32 * scale
    y_out = ct.astype(y_f32, y.dtype)
    ct.scatter(y, (offs_m[:, None], offs_n[None, :]), y_out, check_bounds=True)


# ---- CuTile kernel: activation quantization ----


@ct.kernel
def act_quant_kernel_ct(
    x,  # (total_elements,) flattened input
    y,  # (total_elements,) FP8 output
    s,  # (n_blocks,) scales output
    BLOCK_SIZE: ConstInt,
):
    """
    Activation quantization to FP8: find max abs per block, compute scale, quantize.

    Each program processes one block of BLOCK_SIZE elements.
    FP8 range: max representable value is 448.0 for e4m3fn.

    Pattern reference: flag_gems/cutile/per_token_group_quant_fp8.py
    """
    FP8_MAX = 448.0

    pid = ct.bid(0)
    offs = pid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Load block
    x_tile = ct.gather(x, offs, padding_value=0)
    x_f32 = ct.astype(x_tile, ct.float32)

    # abs(x) = max(x, -x) — standard CuTile pattern
    abs_x = ct.maximum(x_f32, ct.negative(x_f32))

    # Global max over block (no dim arg = full reduction)
    max_abs = ct.max(abs_x)

    # For all-zero blocks (e.g., LoRA dY), return 1.0 to avoid NaN
    # IEEE-compliant division required for FP8 quantization precision
    scale = ct.where(max_abs == 0.0, 1.0, ct.truediv(max_abs, FP8_MAX, rounding_mode=RMd.FULL))

    # Element-wise division — IEEE mode preserves FP8 bucket boundaries
    y_f32 = ct.truediv(x_f32, scale, rounding_mode=RMd.FULL)

    # Clamp to FP8 range and cast
    y_clamped = ct.maximum(ct.minimum(y_f32, FP8_MAX), -FP8_MAX)
    y_fp8 = ct.astype(y_clamped, y.dtype)

    ct.scatter(y, offs, y_fp8)
    # Store scale
    ct.scatter(s, (pid,), scale)


# ---- CuTile kernel: block-wise FP8 matmul ----


@ct.kernel
def w8a8_block_fp8_matmul_kernel_ct(
    A,  # (M, K) FP8
    B,  # (N, K) FP8 — note: B is (N, K) not (K, N)
    C,  # (M, N) output
    As,  # (M, K_groups) float32 activation scales
    Bs,  # (N_groups, K_groups) float32 weight scales
    M: ConstInt,
    N: ConstInt,
    K: ConstInt,
    group_n: ConstInt,
    group_k: ConstInt,
    BLOCK_SIZE_M: ConstInt,
    BLOCK_SIZE_N: ConstInt,
    BLOCK_SIZE_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
    OUTPUT_DTYPE: ConstInt,  # 0=f32, 1=f16, 2=bf16
):
    """
    Block-wise FP8 matmul: C = A @ B^T with per-block quantization scales.

    Reference: src/tilegym/ops/cutile/fp8_quantization_matmul.py
    Uses 2D gather/scatter to avoid pointer arithmetic overflow.
    """
    pid = ct.bid(0)

    # L2-cache-friendly swizzled tile assignment
    num_pid_m = ct.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = ct.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row/col offset arrays
    offs_am = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
    offs_bn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
    offs_k_base = ct.arange(BLOCK_SIZE_K, dtype=ct.int32)

    # Scale index for N dimension
    offs_bsn = offs_bn // group_n

    # Initialize accumulator
    accumulator = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)

    # K-dimension loop
    num_k_tiles = ct.cdiv(K, BLOCK_SIZE_K)
    for k_tile in range(num_k_tiles):
        k_start = k_tile * BLOCK_SIZE_K
        offs_k = offs_k_base + k_start

        # Load A block: (BLOCK_SIZE_M, BLOCK_SIZE_K) from (M, K)
        a = ct.gather(
            A,
            (offs_am[:, None], offs_k[None, :]),
            check_bounds=True,
            padding_value=ct.float8_e4m3fn(0.0),
        )

        # Load B block: (BLOCK_SIZE_K, BLOCK_SIZE_N) from (N, K)
        # B layout is (N, K), so gather with (offs_bn, offs_k) gives (N, K) slice
        # We need (K, N) for matmul, so gather as (offs_k[:, None], offs_bn via N dim)
        # Actually B is (N, K), so b[n, k] accessed as (offs_bn[col], offs_k[row])
        b = ct.gather(
            B,
            (offs_bn[None, :], offs_k[:, None]),
            check_bounds=True,
            padding_value=ct.float8_e4m3fn(0.0),
        )

        # Load quantization scales
        offs_ks = k_start // group_k

        # As: (M, K_groups) → load column offs_ks for rows offs_am
        a_s = ct.gather(As, (offs_am, offs_ks), check_bounds=True, padding_value=0.0)

        # Bs: (N_groups, K_groups) → load column offs_ks for rows offs_bsn
        b_s = ct.gather(Bs, (offs_bsn, offs_ks), check_bounds=True, padding_value=0.0)

        # MMA: a(M,K) @ b(K,N) → (M,N), then scale
        zero_acc = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)
        dot_result = ct.mma(a, b, acc=zero_acc)
        scaled_result = dot_result * (a_s[:, None] * b_s[None, :])
        accumulator = accumulator + scaled_result

    # Convert to output dtype
    if OUTPUT_DTYPE == 1:  # float16
        c = ct.astype(accumulator, ct.float16)
    elif OUTPUT_DTYPE == 2:  # bfloat16
        c = ct.astype(accumulator, ct.bfloat16)
    else:  # float32
        c = accumulator

    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
    offs_cn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
    ct.scatter(C, (offs_cm[:, None], offs_cn[None, :]), c, check_bounds=True)


# ---- TMA-optimized matmul kernel + autotune ----


def _gemm_swizzle_pid(pid, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M):
    """Swizzle linear block id into (pid_m, pid_n) for L2 cache locality."""
    num_pid_m = ct.cdiv(M, BLOCK_M)
    num_pid_n = ct.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@ct.kernel(num_ctas=1)
def w8a8_block_fp8_matmul_kernel_ct_tma(
    A,  # (M, K) FP8
    B,  # (N, K) FP8 — note: B is (N, K) not (K, N)
    C,  # (M, N) output
    As,  # (M, K_groups) float32 activation scales
    Bs,  # (N_groups, K_groups) float32 weight scales
    M: ConstInt,
    N: ConstInt,
    K: ConstInt,
    group_n: ConstInt,
    group_k: ConstInt,
    BLOCK_SIZE_M: ConstInt,
    BLOCK_SIZE_N: ConstInt,
    BLOCK_SIZE_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
    OUTPUT_DTYPE: ConstInt,  # 0=f32, 1=f16, 2=bf16
    swap_ab: ConstInt,  # 0=normal A@B^T, 1=swap (B@A^T)^T
):
    """
    TMA-optimized block-wise FP8 matmul: C = A @ B^T with per-block quantization scales.

    Uses TMA load for A, B and TMA store for C. Scale tensors use gather (too small for TMA).
    swap_ab: when 1, computes (B @ A^T)^T for potentially better MMA perf on some tile shapes.

    Requires BLOCK_SIZE_N == group_n and BLOCK_SIZE_K == group_k for correct
    scale indexing (one scale per N-tile, one per K-tile).
    """
    pid = ct.bid(0)
    pid_m, pid_n = _gemm_swizzle_pid(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # Pre-compute row indices for scale gather (outside K loop)
    offs_am = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)

    # Initialize accumulator
    accumulator = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)

    # K-dimension loop
    num_k_tiles = ct.cdiv(K, BLOCK_SIZE_K)
    for k_tile in range(num_k_tiles):
        # TMA load A: (M, K) -> (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a = ct.load(
            A,
            index=(pid_m, k_tile),
            shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(0, 1),
            latency=3,
            allow_tma=True,
        )

        # TMA load B: (N, K) -> (BLOCK_SIZE_N, BLOCK_SIZE_K)
        b = ct.load(
            B,
            index=(pid_n, k_tile),
            shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
            order=(0, 1),
            latency=3,
            allow_tma=True,
        )

        # Per-block scales via gather (TMA needs contig_dim * elem_size >= 16)
        # a_s: (BLOCK_SIZE_M,) vector — one scale per M row
        a_s = ct.gather(As, (offs_am, k_tile), check_bounds=True, padding_value=0.0, latency=4)
        # b_s: scalar — one scale per (pid_n, k_tile) block
        b_s = ct.gather(Bs, (pid_n, k_tile), check_bounds=True, padding_value=0.0, latency=4)
        ab_s = ct.mul(a_s[:, None], b_s)

        # MMA with permute for transpose
        if swap_ab:
            # Compute (B @ A^T)^T: B(N,K) @ A^T(K,M) -> (N,M), then transpose -> (M,N)
            zero_acc = ct.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=ct.float32)
            a_t = ct.permute(a, (1, 0))
            dot_result = ct.mma(b, a_t, acc=zero_acc)
            dot_result = ct.permute(dot_result, (1, 0))
        else:
            # Compute A @ B^T: A(M,K) @ B^T(K,N) -> (M,N)
            zero_acc = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)
            b_t = ct.permute(b, (1, 0))
            dot_result = ct.mma(a, b_t, acc=zero_acc)

        accumulator = ct.add(accumulator, ct.mul(dot_result, ab_s))

    # Convert to output dtype
    if OUTPUT_DTYPE == 1:  # float16
        c = ct.astype(accumulator, ct.float16)
    elif OUTPUT_DTYPE == 2:  # bfloat16
        c = ct.astype(accumulator, ct.bfloat16)
    else:  # float32
        c = accumulator

    # TMA store result
    ct.store(C, index=(pid_m, pid_n), tile=c, order=(0, 1), allow_tma=True)


# ---- Wrapper functions ----


# Output dtype encoding for kernel
_DTYPE_TO_INT = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


@register_impl("unsloth.weight_dequant", backend="cutile")
def weight_dequant_block(x: torch.Tensor, s: torch.Tensor, block_size: int = 128, dtype=torch.bfloat16) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    if not s.is_contiguous():
        s = s.contiguous()
    assert x.dim() == 2 and s.dim() == 2
    M, N = x.size()
    y = torch.empty_like(x, dtype=dtype)

    grid_m = math.ceil(M / block_size)
    grid_n = math.ceil(N / block_size)

    ct.launch(
        torch.cuda.current_stream(),
        (grid_m, grid_n, 1),
        weight_dequant_kernel_ct,
        (x, s, y, M, N, block_size),
    )
    return y


@register_impl("unsloth.act_quant", backend="cutile")
def act_quant(x: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.shape[-1] % block_size == 0
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)

    n_blocks = x.numel() // block_size
    ct.launch(
        torch.cuda.current_stream(),
        (n_blocks, 1, 1),
        act_quant_kernel_ct,
        (x.reshape(-1), y.reshape(-1), s.reshape(-1), block_size),
    )
    return y, s


@register_impl("unsloth.w8a8_block_fp8_matmul", backend="cutile")
def w8a8_block_fp8_matmul_cutile(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list[int],
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Block-wise FP8 matmul with TMA optimization and heuristic block size."""
    if block_size is None:
        block_n, block_k = 128, 128
    else:
        assert len(block_size) == 2
        block_n, block_k = block_size[0], block_size[1]

    N, K = B.shape
    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert math.ceil(A.shape[-1] / block_k) == As.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert math.ceil(N / block_n) == Bs.shape[0]
    assert math.ceil(K / block_k) == Bs.shape[1]

    M = A.numel() // A.shape[-1]
    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    output_dtype_int = _DTYPE_TO_INT.get(output_dtype, 0)

    BLOCK_SIZE_M = min(128, next_power_of_2(M)) if M < 128 else 128

    grid_m = math.ceil(M / BLOCK_SIZE_M)
    grid_n = math.ceil(N / block_n)

    ct.launch(
        torch.cuda.current_stream(),
        (grid_m * grid_n, 1, 1),
        w8a8_block_fp8_matmul_kernel_ct_tma,
        (
            A.view(M, K),
            B,
            C.view(M, N),
            As.view(M, -1),
            Bs,
            M,
            N,
            K,
            block_n,
            block_k,
            BLOCK_SIZE_M,
            block_n,  # BLOCK_SIZE_N = block_n (locked to quant block)
            block_k,  # BLOCK_SIZE_K = block_k (locked to quant block)
            8,  # GROUP_SIZE_M
            output_dtype_int,
            0,  # swap_ab = False (default, no autotune search)
        ),
    )
    return C
