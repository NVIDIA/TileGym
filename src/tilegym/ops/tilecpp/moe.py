# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Mixture of Experts (MOE)
Fused MOE computation using CUDA C++ tile kernels compiled with nvcc.
"""

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel
from tilegym.ops.tilecpp.utils._cuda_utils import make_kernel_args
from tilegym.ops.tilecpp.utils._dump_types import dump_kernel_types

logger = logging.getLogger(__name__)

# =============================================================================
# Kernel Definitions
# =============================================================================

_fused_moe_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "moe.cuh",
    kernel_name="fused_moe_kernel",
)

# =============================================================================
# Kernel Launch Functions
# =============================================================================


def _get_moe_kernel(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    mul_routed_weight: bool,
    use_fp8_scales: bool,
    N: int,
    K: int,
    group_n: int,
    group_k: int,
    top_k: int,
    stride_am: int,
    stride_ak: int,
    stride_be: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    stride_asm: int,
    stride_ask: int,
    stride_bse: int,
    stride_bsk: int,
    stride_bsn: int,
    EM: int,
):
    """Get compiled kernel for specific configuration."""
    bool_to_str = lambda b: "true" if b else "false"

    # Map dtypes to C++ types
    def dtype_to_cpp(dtype):
        if dtype == torch.float8_e4m3fn:
            return "__nv_fp8_e4m3"
        elif dtype == torch.float16:
            return "__half"
        elif dtype == torch.bfloat16:
            return "__nv_bfloat16"
        else:
            return "float"

    input_type = dtype_to_cpp(input_dtype)
    output_type = dtype_to_cpp(output_dtype)

    kernel, mangled_name, _ = _fused_moe_kernel.get_kernel(
        dtype=output_dtype,  # Use output dtype for {T} placeholder
        template_params=[
            input_type,  # IN_T - explicit input type
            block_m,
            block_n,
            block_k,
            group_m,
            bool_to_str(mul_routed_weight),
            bool_to_str(use_fp8_scales),
            N,
            K,
            group_n,
            group_k,
            top_k,
            stride_am,
            stride_ak,
            stride_be,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_asm,
            stride_ask,
            stride_bse,
            stride_bsk,
            stride_bsn,
            EM,
        ],
        signature=(
            f"const {input_type}*, const {input_type}*, {{T}}*, const float*, const float*, const float*, "
            "const int*, const int*, const int*, "
            "int"  # Only num_valid_tokens remains as runtime param
        ),
    )
    return kernel


def _launch_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    N: int,
    K: int,
    EM: int,
    num_valid_tokens: int,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    use_fp8_w8a8: bool,
    block_shape: Optional[List[int]] = None,
):
    """Launch the fused MOE kernel."""
    dump_kernel_types("fused_moe_kernel", A, B, C, topk_weights)
    input_dtype = A.dtype
    output_dtype = C.dtype

    # Ensure topk_weights is float32 for the kernel
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)

    # Ensure scales are float32
    if A_scale is not None and A_scale.dtype != torch.float32:
        A_scale = A_scale.to(torch.float32)
    if B_scale is not None and B_scale.dtype != torch.float32:
        B_scale = B_scale.to(torch.float32)

    block_m = config.get("BLOCK_SIZE_M", 128)
    block_n = config.get("BLOCK_SIZE_N", 128)
    block_k = config.get("BLOCK_SIZE_K", 128)
    group_m = config.get("GROUP_SIZE_M", 32)

    # Block shape for quantization
    group_n = block_shape[0] if block_shape else block_n
    group_k = block_shape[1] if block_shape else block_k

    # Compute strides (now needed for template parameters)
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    stride_be = B.stride(0)
    stride_bk = B.stride(2)
    stride_bn = B.stride(1)
    stride_cm = C.stride(1)
    stride_cn = C.stride(2)
    stride_asm = A_scale.stride(0) if A_scale is not None and A_scale.ndim >= 2 else 0
    stride_ask = A_scale.stride(1) if A_scale is not None and A_scale.ndim >= 2 else 0
    stride_bse = B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0
    stride_bsk = B_scale.stride(2) if B_scale is not None and B_scale.ndim >= 3 else 0
    stride_bsn = B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0

    kernel = _get_moe_kernel(
        input_dtype,
        output_dtype,
        block_m,
        block_n,
        block_k,
        group_m,
        mul_routed_weight,
        use_fp8_w8a8,
        # NTTP parameters
        N,
        K,
        group_n,
        group_k,
        top_k,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        EM,
    )

    # Grid dimensions
    num_pid_m = (EM + block_m - 1) // block_m
    num_pid_n = (N + block_n - 1) // block_n
    grid = num_pid_m * num_pid_n

    _fused_moe_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(A.data_ptr()),
            np.uint64(B.data_ptr()),
            np.uint64(C.data_ptr()),
            np.uint64(A_scale.data_ptr() if A_scale is not None else 0),
            np.uint64(B_scale.data_ptr() if B_scale is not None else 0),
            np.uint64(topk_weights.data_ptr()),
            np.uint64(sorted_token_ids.data_ptr()),
            np.uint64(expert_ids.data_ptr()),
            np.uint64(num_tokens_post_padded.data_ptr()),
            np.int32(num_valid_tokens),  # Only runtime param left
        ],
    )


# =============================================================================
# Public Interface
# =============================================================================


@register_impl("invoke_fused_moe_kernel", "tilecpp")
def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool = False,
    block_shape: Optional[List[int]] = None,
) -> None:
    """
    Fused MOE kernel for computing expert outputs.

    Supports both regular and FP8 quantized computation with block-wise scaling.
    """
    if use_int8_w8a16:
        raise NotImplementedError("INT8 quantization not supported in TileCpp MOE kernel")

    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    # FP8 validation
    if use_fp8_w8a8:
        assert A_scale is not None, "A_scale required for FP8"
        assert B_scale is not None, "B_scale required for FP8"
        assert A_scale.shape[-1] == A.shape[-1] // config["BLOCK_SIZE_K"]
        assert B_scale.shape[-1] == B.shape[-1] // config["BLOCK_SIZE_K"]
        assert B_scale.shape[1] == B.shape[1] // config["BLOCK_SIZE_N"]

    N = B.shape[1]
    K = B.shape[2]
    EM = sorted_token_ids.shape[0]
    num_valid_tokens = topk_ids.numel()

    logger.debug(
        f"[tilecpp] calling fused_moe_kernel, A.shape: {A.shape}, B.shape: {B.shape}, "
        f"C.shape: {C.shape}, mul_routed_weight: {mul_routed_weight}, top_k: {top_k}, "
        f"use_fp8_w8a8: {use_fp8_w8a8}"
    )

    _launch_fused_moe_kernel(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        mul_routed_weight,
        top_k,
        config,
        use_fp8_w8a8,
        block_shape,
    )


# Commented out - using implementation from moe_align_block.py instead
# @register_impl("moe_align_block_size", "tilecpp")
def _moe_align_block_size_cpu_fallback(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Align and sort tokens for block-wise MoE computation.

    This is a CPU implementation that matches the PyTorch reference.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert assignments
        block_size: Size of blocks for matrix operations
        num_experts: Total number of experts

    Returns:
        sorted_token_ids: Tensor containing sorted and padded token indices
        expert_ids: Tensor containing corresponding expert IDs
        num_tokens_post_padded: Total number of tokens after padding
        cumsum: Cumulative sum tensor for block alignment
    """
    device = "cpu"

    # Calculate dimensions
    num_tokens, top_k = topk_ids.shape
    total_tokens = num_tokens * top_k

    # Flatten both arrays
    flat_expert_ids = topk_ids.reshape(-1).to(device)
    sorted_token_indices = torch.argsort(flat_expert_ids, stable=True)

    # Count tokens per expert before padding
    expert_token_counts = torch.bincount(flat_expert_ids, minlength=num_experts)
    expert_block_counts = (expert_token_counts - 1 + block_size) // block_size
    total_blocks = expert_block_counts.sum()
    sorted_token_ids = torch.zeros((total_blocks * block_size,), device=device) + total_tokens
    sorted_expert_ids = torch.zeros((total_blocks,), device=device)

    # Compute cumsum for block alignment (padded cumulative sum)
    padded_counts = expert_block_counts * block_size
    cumsum = torch.zeros((num_experts + 1,), dtype=torch.int32, device=device)
    cumsum[1:] = torch.cumsum(padded_counts, dim=0)

    current_block = 0
    current_token = 0
    for i in range(num_experts):
        sorted_expert_ids[current_block : current_block + expert_block_counts[i]] = i
        sorted_token_start = current_block * block_size
        sorted_token_end = sorted_token_start + expert_token_counts[i]
        sorted_token_ids[sorted_token_start:sorted_token_end] = sorted_token_indices[
            current_token : current_token + expert_token_counts[i]
        ]
        current_token += expert_token_counts[i]
        current_block += expert_block_counts[i]

    sorted_token_ids = sorted_token_ids.to(torch.int32).to(topk_ids.device)
    sorted_expert_ids = sorted_expert_ids.to(torch.int32).to(topk_ids.device)
    num_tokens_post_padded = torch.tensor(sorted_token_ids.numel()).to(torch.int32).to(topk_ids.device)
    cumsum = cumsum.to(topk_ids.device)
    return sorted_token_ids, sorted_expert_ids, num_tokens_post_padded, cumsum


# =============================================================================
# FP8 V2 Interface (FC1 and FC2 Kernels)
# =============================================================================

_fused_moe_fc1_layer_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "moe.cuh",
    kernel_name="fused_moe_fc1_layer_kernel",
)

_fused_moe_fc2_layer_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "moe.cuh",
    kernel_name="fused_moe_fc2_layer_kernel",
)


def _get_fc1_kernel(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    mul_routed_weight: bool,
):
    """Get compiled FC1 kernel for specific configuration."""
    bool_to_str = lambda b: "true" if b else "false"

    # Map dtypes to C++ types
    input_type = "__nv_fp8_e4m3"  # FP8 E4M3
    if output_dtype == torch.float16:
        output_type = "__half"
    elif output_dtype == torch.bfloat16:
        output_type = "__nv_bfloat16"
    else:
        output_type = "float"

    # Use output_dtype for the kernel dtype (it determines the {T} placeholder)
    # But we override with explicit types in template_params
    kernel, _, _ = _fused_moe_fc1_layer_kernel.get_kernel(
        dtype=output_dtype,  # Use output dtype for {T} placeholder
        template_params=[
            input_type,  # INPUT_T - explicit FP8 type
            block_m,
            block_n,
            block_k,
            group_m,
            bool_to_str(mul_routed_weight),
        ],
        signature=(
            f"const {input_type}*, const {input_type}*, const {input_type}*, {{T}}*, "
            "const float*, const float*, const float*, "
            "const {T}*, const int*, const int*, const int*, "
            "int, int, int, int, int, "
            "int, int, int, int, int, int, int, int, int, "
            "int, int, int, int, int, int, int, int, int, int, int, int"
        ),
    )
    return kernel


def _get_fc2_kernel(
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    group_m: int,
    mul_routed_weight: bool,
):
    """Get compiled FC2 kernel for specific configuration."""
    bool_to_str = lambda b: "true" if b else "false"

    input_type = "__nv_fp8_e4m3"

    kernel, _, _ = _fused_moe_fc2_layer_kernel.get_kernel(
        dtype=output_dtype,  # Use output dtype for {T} placeholder
        template_params=[
            input_type,  # INPUT_T - explicit FP8 type
            block_m,
            block_n,
            block_k,
            group_m,
            bool_to_str(mul_routed_weight),
        ],
        signature=(
            f"const {input_type}*, const {input_type}*, {{T}}*, "
            "const float*, const float*, "
            "const {T}*, const int*, const int*, const int*, "
            "int, int, int, int, int, "
            "int, int, int, int, int, int, int, "
            "int, int, int, int, int, int, int, int"
        ),
    )
    return kernel


@register_impl("invoke_fused_moe_fc1_layer_kernel", "tilecpp")
def invoke_fused_moe_fc1_layer_kernel(
    A: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B1_scale: Optional[torch.Tensor],
    B2_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cta_token_offsets: Optional[torch.Tensor],
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type,
    use_fp8_w8a8: bool,
    use_scatter_store: bool,
    block_shape: Optional[List[int]] = None,
) -> None:
    """
    Fused MOE FC1 layer kernel (V2 interface).
    Computes: output = silu(A @ B1) * (A @ B2) with FP8 quantization.
    """
    assert use_fp8_w8a8, "TileCpp FC1 kernel only supports FP8"
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    dump_kernel_types("fused_moe_fc1_layer_kernel", A, B1, B2, C)

    # Get dimensions
    M = A.shape[0]
    N = B1.shape[1]
    K = B1.shape[2]
    EM = sorted_token_ids.shape[0]
    num_valid_tokens = topk_ids.numel()

    # Get block sizes from config
    block_m = config.get("BLOCK_SIZE_M", 128)
    block_n = config.get("BLOCK_SIZE_N", 128)
    block_k = config.get("BLOCK_SIZE_K", 128)
    group_m = config.get("GROUP_SIZE_M", 32)

    # Block shape for quantization (default to block sizes)
    group_n = block_shape[0] if block_shape else block_n
    group_k = block_shape[1] if block_shape else block_k

    # Ensure scales are float32
    if A_scale is not None and A_scale.dtype != torch.float32:
        A_scale = A_scale.to(torch.float32)
    if B1_scale is not None and B1_scale.dtype != torch.float32:
        B1_scale = B1_scale.to(torch.float32)
    if B2_scale is not None and B2_scale.dtype != torch.float32:
        B2_scale = B2_scale.to(torch.float32)

    # Get kernel
    kernel = _get_fc1_kernel(
        A.dtype,
        C.dtype,
        block_m,
        block_n,
        block_k,
        group_m,
        mul_routed_weight=False,  # FC1 doesn't multiply routed weights
    )

    # Grid dimensions
    num_pid_m = (EM + block_m - 1) // block_m
    num_pid_n = (N + block_n - 1) // block_n
    grid = num_pid_m * num_pid_n

    _fused_moe_fc1_layer_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(A.data_ptr()),
            np.uint64(B1.data_ptr()),
            np.uint64(B2.data_ptr()),
            np.uint64(C.data_ptr()),
            np.uint64(A_scale.data_ptr() if A_scale is not None else 0),
            np.uint64(B1_scale.data_ptr() if B1_scale is not None else 0),
            np.uint64(B2_scale.data_ptr() if B2_scale is not None else 0),
            np.uint64(topk_weights.data_ptr()),
            np.uint64(sorted_token_ids.data_ptr()),
            np.uint64(expert_ids.data_ptr()),
            np.uint64(num_tokens_post_padded.data_ptr()),
            np.int32(M),
            np.int32(N),
            np.int32(K),
            np.int32(EM),
            np.int32(num_valid_tokens),
            np.int32(A.stride(0)),
            np.int32(A.stride(1)),
            np.int32(B1.stride(0)),
            np.int32(B1.stride(2)),  # stride_b1k
            np.int32(B1.stride(1)),  # stride_b1n
            np.int32(B2.stride(0)),
            np.int32(B2.stride(2)),  # stride_b2k
            np.int32(B2.stride(1)),  # stride_b2n
            np.int32(C.stride(0)),
            np.int32(C.stride(1)),
            np.int32(A_scale.stride(0) if A_scale is not None and A_scale.ndim >= 2 else 0),
            np.int32(A_scale.stride(1) if A_scale is not None and A_scale.ndim >= 2 else 0),
            np.int32(B1_scale.stride(0) if B1_scale is not None and B1_scale.ndim >= 2 else 0),
            np.int32(B1_scale.stride(2) if B1_scale is not None and B1_scale.ndim >= 3 else 0),
            np.int32(B1_scale.stride(1) if B1_scale is not None and B1_scale.ndim >= 2 else 0),
            np.int32(B2_scale.stride(0) if B2_scale is not None and B2_scale.ndim >= 2 else 0),
            np.int32(B2_scale.stride(2) if B2_scale is not None and B2_scale.ndim >= 3 else 0),
            np.int32(B2_scale.stride(1) if B2_scale is not None and B2_scale.ndim >= 2 else 0),
            np.int32(group_n),
            np.int32(group_k),
            np.int32(top_k),
        ],
    )


@register_impl("invoke_fused_moe_fc2_layer_kernel", "tilecpp")
def invoke_fused_moe_fc2_layer_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    cta_token_offsets: Optional[torch.Tensor],
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type,
    use_fp8_w8a8: bool,
    use_gather_load: bool,
    block_shape: Optional[List[int]] = None,
) -> None:
    """
    Fused MOE FC2 layer kernel (V2 interface).
    Computes: output = A @ B with FP8 quantization and optional weight multiplication.
    """
    assert use_fp8_w8a8, "TileCpp FC2 kernel only supports FP8"
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    dump_kernel_types("fused_moe_fc2_layer_kernel", A, B, C)

    # Get dimensions
    M = A.shape[0]
    N = B.shape[1]
    K = B.shape[2]
    EM = sorted_token_ids.shape[0]
    num_valid_tokens = topk_ids.numel()

    # Get block sizes from config
    block_m = config.get("BLOCK_SIZE_M", 128)
    block_n = config.get("BLOCK_SIZE_N", 128)
    block_k = config.get("BLOCK_SIZE_K", 128)
    group_m = config.get("GROUP_SIZE_M", 32)

    # Block shape for quantization
    group_n = block_shape[0] if block_shape else block_n
    group_k = block_shape[1] if block_shape else block_k

    # Ensure scales are float32
    if A_scale is not None and A_scale.dtype != torch.float32:
        A_scale = A_scale.to(torch.float32)
    if B_scale is not None and B_scale.dtype != torch.float32:
        B_scale = B_scale.to(torch.float32)

    # Get kernel
    kernel = _get_fc2_kernel(
        A.dtype,
        C.dtype,
        block_m,
        block_n,
        block_k,
        group_m,
        mul_routed_weight,
    )

    # Grid dimensions
    num_pid_m = (EM + block_m - 1) // block_m
    num_pid_n = (N + block_n - 1) // block_n
    grid = num_pid_m * num_pid_n

    _fused_moe_fc2_layer_kernel.launch(
        grid=grid,
        kernel=kernel,
        args=[
            np.uint64(A.data_ptr()),
            np.uint64(B.data_ptr()),
            np.uint64(C.data_ptr()),
            np.uint64(A_scale.data_ptr() if A_scale is not None else 0),
            np.uint64(B_scale.data_ptr() if B_scale is not None else 0),
            np.uint64(topk_weights.data_ptr()),
            np.uint64(sorted_token_ids.data_ptr()),
            np.uint64(expert_ids.data_ptr()),
            np.uint64(num_tokens_post_padded.data_ptr()),
            np.int32(M),
            np.int32(N),
            np.int32(K),
            np.int32(EM),
            np.int32(num_valid_tokens),
            np.int32(A.stride(0)),
            np.int32(A.stride(1)),
            np.int32(B.stride(0)),
            np.int32(B.stride(2)),  # stride_bk
            np.int32(B.stride(1)),  # stride_bn
            # C is [M, topk, hidden_size] - pass stride(1) and stride(2) for the 2D view
            np.int32(C.stride(1) if C.ndim >= 3 else C.stride(0)),  # stride_cm
            np.int32(C.stride(2) if C.ndim >= 3 else C.stride(1)),  # stride_cn
            np.int32(A_scale.stride(0) if A_scale is not None and A_scale.ndim >= 2 else 0),
            np.int32(A_scale.stride(1) if A_scale is not None and A_scale.ndim >= 2 else 0),
            np.int32(B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0),
            np.int32(B_scale.stride(2) if B_scale is not None and B_scale.ndim >= 3 else 0),
            np.int32(B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0),
            np.int32(group_n),
            np.int32(group_k),
            np.int32(top_k),
        ],
    )
