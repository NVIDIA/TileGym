# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace
from typing import Optional

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl
from tilegym.logger import get_logger

from .utils import cached_replace_hints

logger = get_logger(__name__)

# Module-level tune cache: (M, N, K, block_n, block_k, output_dtype_int, use_tma_override, dtype, device)
# -> (best_cfg, tuned_kernel). use_tma_override=None tunes TMA and non-TMA as one workload.
_w8a8_tune_cache: dict = {}


# Tile-index helpers
def _gemm_calculate_pid_ct(pid, M, N, BLOCK_M, BLOCK_N, GROUP_SIZE_M):
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


# Autotuning
def _w8a8_autotune_configs(block_n_quant, block_k_quant):
    """Yield autotune configurations for the W8A8 FP8 matmul kernel.

    BLOCK_SIZE_N and BLOCK_SIZE_K must equal the quantization block sizes for
    correct scale indexing, so only BLOCK_SIZE_M, occupancy, swap_ab, and
    use_tma are searched.
    """
    for block_m in [16, 32, 64, 128]:
        for occupancy in [1, 2, 4]:
            for swap_ab in [True, False]:
                for use_tma in [False, True]:
                    yield SimpleNamespace(
                        BLOCK_SIZE_M=block_m,
                        BLOCK_SIZE_N=block_n_quant,
                        BLOCK_SIZE_K=block_k_quant,
                        GROUP_SIZE_M=16,
                        num_ctas=1,
                        occupancy=occupancy,
                        swap_ab=swap_ab,
                        use_tma=use_tma,
                    )


def _w8a8_early_config_prune(configs, M):
    """Drop configs whose BLOCK_SIZE_M exceeds the M dimension."""
    pruned = [cfg for cfg in configs if cfg.BLOCK_SIZE_M <= M]
    return pruned if pruned else configs


def _w8a8_filter_use_tma(configs, use_tma: Optional[bool]):
    """Constrain TMA only when the caller explicitly asks for one mode."""
    if use_tma is None:
        return configs
    return [cfg for cfg in configs if cfg.use_tma is use_tma]


@ct.kernel
def _w8a8_block_fp8_matmul_kernel(
    # Tensors
    A,
    B,
    C,
    As,
    Bs,
    # Dimensions
    M: ct.Constant[int],
    N: ct.Constant[int],
    K: ct.Constant[int],
    # Quantization block sizes
    GROUP_N: ct.Constant[int],
    GROUP_K: ct.Constant[int],
    # Strides
    STRIDE_AM: ct.Constant[int],
    STRIDE_AK: ct.Constant[int],
    STRIDE_BK: ct.Constant[int],
    STRIDE_BN: ct.Constant[int],
    STRIDE_CM: ct.Constant[int],
    STRIDE_CN: ct.Constant[int],
    STRIDE__AS_M: ct.Constant[int],
    STRIDE__AS_K: ct.Constant[int],
    STRIDE__BS_K: ct.Constant[int],
    STRIDE__BS_N: ct.Constant[int],
    # Tile parameters
    BLOCK_SIZE_M: ct.Constant[int],
    BLOCK_SIZE_N: ct.Constant[int],
    BLOCK_SIZE_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    OUTPUT_DTYPE: ct.Constant[int],
    SWAP_AB: ct.Constant[int],
    USE_TMA: ct.Constant[int],
):
    """Gather/scatter W8A8 block-scaled FP8 matmul.

    When swap_ab=1: compute (B @ A^T)^T * scales  (swap operand order).
    When swap_ab=0: compute (A @ B^T) * scales     (normal order).

    A: (M, K)  B: (N, K)  As: (M, K_groups)  Bs: (N_groups, K_groups)  C: (M, N)

    Requires BLOCK_SIZE_N == group_n and BLOCK_SIZE_K == group_k for correct
    scale indexing (one scale per tile).
    """
    ct.static_assert(
        BLOCK_SIZE_N == GROUP_N, f"Kernel requires BLOCK_SIZE_N == group_n, got {BLOCK_SIZE_N} vs {GROUP_N}"
    )
    ct.static_assert(
        BLOCK_SIZE_K == GROUP_K, f"Kernel requires BLOCK_SIZE_K == group_k, got {BLOCK_SIZE_K} vs {GROUP_K}"
    )

    pid = ct.bid(0)
    pid_m, pid_n = _gemm_calculate_pid_ct(pid, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    # Create index arrays for dimensions
    offs_am = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
    offs_bn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
    offs_k_base = ct.arange(BLOCK_SIZE_K, dtype=ct.int32)

    # Initialize accumulator
    accumulator = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)

    # K-dimension loop with block-wise quantization
    num_k_tiles = ct.cdiv(K, BLOCK_SIZE_K)
    for k_tile in range(num_k_tiles):
        # Calculate current K indices
        k_start = k_tile * BLOCK_SIZE_K
        offs_k = offs_k_base + k_start

        if USE_TMA:
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

            # Per-block scales via gather (latency=4 tuned for scale loads)
            a_s = ct.gather(As, (offs_am, k_tile), check_bounds=True, padding_value=0.0, latency=4)
            b_s = ct.gather(Bs, (pid_n, k_tile), check_bounds=True, padding_value=0.0, latency=4)
        else:
            # Load A block: (M, K) -> (BLOCK_SIZE_M, BLOCK_SIZE_K)
            a = ct.gather(
                A,
                (offs_am[:, None], offs_k[None, :]),
                check_bounds=True,
                padding_value=ct.float8_e4m3fn(0.0),
            )

            # Load B block: (N, K) -> (BLOCK_SIZE_N, BLOCK_SIZE_K)
            b = ct.gather(
                B,
                (offs_bn[:, None], offs_k[None, :]),
                check_bounds=True,
                padding_value=ct.float8_e4m3fn(0.0),
            )

            # As: (M, K_groups) -> (BLOCK_SIZE_M,)
            a_s = ct.gather(As, (offs_am, k_tile), check_bounds=True, padding_value=0.0)

            # Bs: (N_groups, K_groups) -> scalar
            b_s = ct.gather(Bs, (pid_n, k_tile), check_bounds=True, padding_value=0.0)
        ab_s = a_s[:, None] * b_s

        # MMA with permute for transpose
        if SWAP_AB:
            zero_acc = ct.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=ct.float32)
            a_t = ct.permute(a, (1, 0))
            dot_result = ct.mma(b, a_t, acc=zero_acc)
            dot_result = ct.permute(dot_result, (1, 0))
        else:
            zero_acc = ct.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ct.float32)
            b_t = ct.permute(b, (1, 0))
            dot_result = ct.mma(a, b_t, acc=zero_acc)

        accumulator = accumulator + dot_result * ab_s

    # Convert to output data type
    if OUTPUT_DTYPE == 0:  # torch.float32
        c = accumulator
    elif OUTPUT_DTYPE == 1:  # torch.float16
        c = ct.astype(accumulator, ct.float16)
    elif OUTPUT_DTYPE == 2:  # torch.bfloat16
        c = ct.astype(accumulator, ct.bfloat16)
    else:
        c = accumulator
    if USE_TMA:
        ct.store(C, index=(pid_m, pid_n), tile=c, order=(0, 1), allow_tma=True)
    else:
        offs_cm = pid_m * BLOCK_SIZE_M + ct.arange(BLOCK_SIZE_M, dtype=ct.int32)
        offs_cn = pid_n * BLOCK_SIZE_N + ct.arange(BLOCK_SIZE_N, dtype=ct.int32)
        ct.scatter(C, (offs_cm[:, None], offs_cn[None, :]), c, check_bounds=True)


# Autotuned launcher
def _cutile_autotune_w8a8(
    stream,
    kernel,
    A,
    B,
    C,
    As,
    Bs,
    M,
    N,
    K,
    block_n,
    block_k,
    output_dtype_int,
    use_tma: Optional[bool],
):
    """Launch W8A8 FP8 matmul kernel with exhaustive_search autotuning."""
    configs = _w8a8_filter_use_tma(
        _w8a8_early_config_prune(
            list(_w8a8_autotune_configs(block_n, block_k)),
            M,
        ),
        use_tma,
    )

    def grid_fn(cfg):
        grid_m = (M + cfg.BLOCK_SIZE_M - 1) // cfg.BLOCK_SIZE_M
        grid_n = (N + cfg.BLOCK_SIZE_N - 1) // cfg.BLOCK_SIZE_N
        return (grid_m * grid_n, 1, 1)

    def args_fn(cfg):
        return (
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            block_n,
            block_k,
            A.stride(-2),
            A.stride(-1),
            B.stride(1),
            B.stride(0),
            C.stride(-2),
            C.stride(-1),
            As.stride(-2),
            As.stride(-1),
            Bs.stride(1),
            Bs.stride(0),
            cfg.BLOCK_SIZE_M,
            cfg.BLOCK_SIZE_N,
            cfg.BLOCK_SIZE_K,
            cfg.GROUP_SIZE_M,
            output_dtype_int,
            int(cfg.swap_ab),
            int(cfg.use_tma),
        )

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    cache_key = (M, N, K, block_n, block_k, output_dtype_int, use_tma, A.dtype, str(A.device))
    if cache_key not in _w8a8_tune_cache:
        result = exhaustive_search(
            configs,
            stream,
            grid_fn,
            kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        _w8a8_tune_cache[cache_key] = (
            best_cfg,
            kernel.replace_hints(num_ctas=best_cfg.num_ctas, occupancy=best_cfg.occupancy),
        )
    best_cfg, tuned_kernel = _w8a8_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


@register_impl("w8a8_block_fp8_matmul", backend="cutile")
def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: list = None,
    kernel_configs: dict = None,
    output_dtype: torch.dtype = torch.float16,
    use_tma: Optional[bool] = None,
) -> torch.Tensor:
    """Block-scaled W8A8 FP8 matrix multiplication with autotuning.

    Computes C = dequant(A @ B^T) where A and B are FP8-quantized with
    per-block scales As and Bs.

    Args:
        A: The input tensor, e.g., activation.
        B: The input tensor, e.g., weight.
        As: The per-token-group quantization scale for `A`.
        Bs: The per-block quantization scale for `B`.
        block_size: The block size for per-block quantization. It should be 2-dim, e.g., [128, 128].
        kernel_configs: Kernel configuration parameters including block sizes and optimization levels.
        output_dtype: The dtype of the returned tensor.
        use_tma: Whether to force TMA. None tunes TMA and non-TMA as one workload.

    Returns:
        torch.Tensor: The result of matmul.
    """
    if block_size is None:
        block_size = [128, 128]

    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    # Shape validation
    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1] and A.is_contiguous()
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    M = A.numel() // A.shape[-1]

    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    N, K = B.shape
    assert (N + block_n - 1) // block_n == Bs.shape[0]
    assert (K + block_k - 1) // block_k == Bs.shape[1]

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    # Map output dtype to integer for kernel
    dtype_map = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
    output_dtype_int = dtype_map.get(output_dtype, 0)

    kernel = _w8a8_block_fp8_matmul_kernel

    if kernel_configs is not None:
        # Fixed config: skip autotuning, launch directly
        grid_m = (M + kernel_configs["BLOCK_SIZE_M"] - 1) // kernel_configs["BLOCK_SIZE_M"]
        grid_n = (N + kernel_configs["BLOCK_SIZE_N"] - 1) // kernel_configs["BLOCK_SIZE_N"]
        hints = {"num_ctas": kernel_configs.get("num_ctas", 1)}
        occupancy = kernel_configs.get("occupancy")
        if occupancy is not None:
            hints["occupancy"] = occupancy
        fixed_kernel = cached_replace_hints(kernel, **hints)
        fixed_use_tma = kernel_configs.get("use_tma", use_tma)
        if fixed_use_tma is None:
            fixed_use_tma = True
        ct.launch(
            torch.cuda.current_stream(),
            (grid_m * grid_n, 1, 1),
            fixed_kernel,
            (
                A,
                B,
                C,
                As,
                Bs,
                M,
                N,
                K,
                block_n,
                block_k,
                A.stride(-2),
                A.stride(-1),
                B.stride(1),
                B.stride(0),
                C.stride(-2),
                C.stride(-1),
                As.stride(-2),
                As.stride(-1),
                Bs.stride(1),
                Bs.stride(0),
                kernel_configs["BLOCK_SIZE_M"],
                kernel_configs["BLOCK_SIZE_N"],
                kernel_configs["BLOCK_SIZE_K"],
                kernel_configs["GROUP_SIZE_M"],
                output_dtype_int,
                int(kernel_configs.get("swap_ab", False)),
                int(fixed_use_tma),
            ),
        )
    else:
        # Autotune over BLOCK_SIZE_M, occupancy, swap_ab, and use_tma.
        _cutile_autotune_w8a8(
            torch.cuda.current_stream(),
            kernel,
            A,
            B,
            C,
            As,
            Bs,
            M,
            N,
            K,
            block_n,
            block_k,
            output_dtype_int,
            use_tma,
        )

    return C
