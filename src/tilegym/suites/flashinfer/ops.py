# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
FlashInfer Suite Operations Interface

This module provides operation interfaces for FlashInfer-specific operations.
These operations are automatically dispatched to the appropriate backend implementation.
"""

from typing import Optional
from typing import Tuple

import torch

from tilegym.backend import dispatch
from tilegym.backend import get_current_backend

# ============================================================================
# GEMM Operations
# ============================================================================


@dispatch(
    "flashinfer.gemm.gemm_alpha_beta",
)
def gemm_alpha_beta(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = True,
    alpha: float = 1.0,
    beta: float = 0.0,
    num_sms: int = 1,
):
    """
    FlashInfer GEMM operation: Matrix multiplication with alpha/beta scaling.

    Computes: C = alpha * A @ B + beta * C

    Args:
        a: Input matrix A [M, K] or [K, M] if trans_a=True
        b: Input matrix B [K, N] or [N, K] if trans_b=True
        c: Input/output matrix C [M, N] (modified in-place)
        trans_a: Whether to transpose A
        trans_b: Whether to transpose B
        alpha: Scaling factor for A @ B
        beta: Scaling factor for existing C

    Returns:
        torch.Tensor: Modified C tensor (C = alpha * A @ B + beta * C)
    """
    raise NotImplementedError(f"flashinfer.gemm.gemm_alpha_beta is not implemented for {get_current_backend()}")


@dispatch(
    "flashinfer.gemm.masked_bmm",
)
def masked_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    masked_m: torch.Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False,
    static_persistent: Optional[bool] = None,
):
    """
    FlashInfer operation: Masked batch matrix multiplication.

    Performs batched matrix multiplication where each batch has a different valid M dimension.

    Args:
        a: Input matrix A [Q, M, K] or [Q, K, M] if transpose_a=True
        b: Input matrix B [Q, K, N] or [Q, N, K] if transpose_b=True
        masked_m: Valid M dimensions for each batch [Q]
        transpose_a: Whether to transpose A
        transpose_b: Whether to transpose B
        static_persistent: Whether to use static persistent scheduling

    Returns:
        torch.Tensor: Output matrix C [Q, M, N]
    """
    raise NotImplementedError(f"flashinfer.gemm.masked_bmm is not implemented for {get_current_backend()}")


@dispatch(
    "flashinfer.gemm.ragged_bmm",
)
def ragged_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    m_indptr: torch.Tensor,
    max_m: int,
    max_m_device: torch.Tensor = None,
    transpose_a: bool = False,
    transpose_b: bool = True,
    out_dtype: torch.dtype = None,
):
    """
    FlashInfer operation: Ragged batch matrix multiplication.

    Performs batched matrix multiplication where batches have non-uniform M dimensions.
    Matrix A is flattened with m_indptr defining batch boundaries.

    Args:
        a: Flattened input matrix A [total_m, K] or [K, total_m] if transpose_a=True
        b: Input matrix B [Q, K, N] or [Q, N, K] if transpose_b=True
        m_indptr: Segment offsets marking batch boundaries [Q+1]
        max_m: Maximum M dimension across all batches
        max_m_device: Optional device tensor containing max_m (for CUDA graph compatibility)
        transpose_a: Whether to transpose A
        transpose_b: Whether to transpose B
        out_dtype: Optional output dtype override

    Returns:
        torch.Tensor: Flattened output matrix C [total_m, N]
    """
    raise NotImplementedError(f"flashinfer.gemm.ragged_bmm is not implemented for {get_current_backend()}")


@dispatch(
    "flashinfer.gemm.ragged_block_scaled_bmm",
)
def ragged_block_scaled_bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    max_m: int,
    max_m_device: torch.Tensor = None,
    transpose_a: bool = False,
    transpose_b: bool = True,
    out_dtype: torch.dtype = None,
):
    """
    FlashInfer operation: Ragged batch matrix multiplication with block-wise scaling.

    Performs ragged batched matrix multiplication with per-block scaling factors.

    Args:
        a: Flattened input matrix A [total_m, K]
        b: Input matrix B [Q, N, K] or [Q, K, N]
        a_scale: Scaling factors for A blocks
        b_scale: Scaling factors for B blocks
        m_indptr: Segment offsets marking batch boundaries [Q+1]
        max_m: Maximum M dimension across all batches
        max_m_device: Optional device tensor containing max_m (for CUDA graph compatibility)
        transpose_a: Whether to transpose A
        transpose_b: Whether to transpose B
        out_dtype: Optional output dtype override

    Returns:
        torch.Tensor: Flattened output matrix C [total_m, N]
    """
    raise NotImplementedError(f"flashinfer.gemm.ragged_block_scaled_bmm is not implemented for {get_current_backend()}")


# ============================================================================
# RoPE Operations
# ============================================================================


@dispatch(
    "flashinfer.rope.rope_quantize_fp8",
)
def rope_quantize_fp8(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    q_nope: Optional[torch.Tensor],
    k_nope: Optional[torch.Tensor],
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool = True,
    quantize_dtype: Optional[torch.dtype] = None,
    quant_scale_q: float = 1.0,
    quant_scale_kv: float = 1.0,
    q_rope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
    q_nope_out: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    FlashInfer operation: RoPE (Rotary Position Embedding) with FP8 quantization.

    Applies RoPE to q_rope and k_rope tensors, then quantizes all outputs to FP8.
    Supports both interleave mode (is_neox=False, GPT-J style) and non-interleave mode.

    Example (MLA with head_size=576, rope_dim=64, no_rope_dim=512):
        q_in: [num_tokens, num_qo_heads, 576]  # e.g., [N, 128, 576]
        k_in: [num_tokens, 576]                # MLA uses 2D key tensor

        flashinfer.ops.rope_quantize_fp8(
            q_rope=q_in[..., :64],              # [N, 128, 64]
            k_rope=k_in[..., :64],              # [N, 64]
            q_nope=q_in[..., 64:],              # [N, 128, 512]
            k_nope=k_in[..., 64:],              # [N, 512]
            cos_sin_cache=cos_sin_cache,        # [max_seq_len, rope_dim]
            pos_ids=pos_ids,                    # [num_tokens]
            is_neox=False,
            q_rope_out=q_out[..., :64],
            k_rope_out=k_out[..., :64],
            q_nope_out=q_out[..., 64:],
            k_nope_out=k_out[..., 64:],
        )

    Args:
        q_rope: Query tensor for RoPE [num_tokens, num_qo_heads, rope_dim]
        k_rope: Key tensor for RoPE [num_tokens, rope_dim] (MLA 2D) or [num_tokens, num_kv_heads, rope_dim] (3D)
        q_nope: Query tensor without RoPE [num_tokens, num_qo_heads, no_rope_dim] or None
        k_nope: Key tensor without RoPE [num_tokens, no_rope_dim] (MLA 2D) or None
        cos_sin_cache: Precomputed cos/sin cache [max_seq_len, rope_dim]
        pos_ids: Position IDs [num_tokens]
        is_neox: Whether to use NeoX-style RoPE (True) or interleave/GPT-J style (False)
        quantize_dtype: Output dtype for quantization (default: float8_e4m3fn)
        quant_scale_q: Quantization scale for Q tensors
        quant_scale_kv: Quantization scale for K tensors
        q_rope_out: Optional pre-allocated output for q_rope
        k_rope_out: Optional pre-allocated output for k_rope
        q_nope_out: Optional pre-allocated output for q_nope
        k_nope_out: Optional pre-allocated output for k_nope
        enable_pdl: Whether to enable PDL (Persistent Data Layout)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            (q_rope_out, k_rope_out, q_nope_out, k_nope_out)
    """
    raise NotImplementedError(f"flashinfer.rope.rope_quantize_fp8 is not implemented for {get_current_backend()}")


# ============================================================================
# Per-token group 8-bit quantization
# ============================================================================


@dispatch(
    "flashinfer.quant.per_token_group_quant_8bit",
)
def per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dst_dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token group 8-bit quantization (FP8 or INT8).

    Args:
        x: Input tensor [num_tokens, hidden_dim], contiguous.
        group_size: Group size for quantization.
        eps: Minimum value to avoid division by zero.
        dst_dtype: Output dtype (torch.float8_e4m3fn or torch.int8).
        column_major_scales: If True, scale tensor layout is column-major.
        scale_tma_aligned: If True, scale buffer is aligned for TMA (requires column_major_scales=True).
        scale_ue8m0: If True, round scale to power of 2 (UE8M0) (requires column_major_scales=True).

    Returns:
        (x_q, x_s): Quantized tensor and scale tensor.
    """
    raise NotImplementedError(
        f"flashinfer.quant.per_token_group_quant_8bit is not implemented for {get_current_backend()}"
    )


# ============================================================================
# Attention Operations
# ============================================================================


@dispatch(
    "flashinfer.attention.decode_attention_kv_paged",
)
def decode_attention_kv_paged(
    q,
    k_cache,
    v_cache,
    actual_seq_lens,
    block_tables,
    k_scale,
    v_scale,
    max_seq_len: int = -1,
    outputs: Optional[torch.Tensor] = None,
    force_split_kv: bool = False,
    force_persistent: bool = False,
):
    """
    FlashInfer operation: Decode attention with paged KV cache.

    Args:
        q: Query tensor
        k_cache: Paged key cache
        v_cache: Paged value cache
        actual_seq_lens: Actual sequence lengths
        block_tables: Block tables for paged attention
        k_scale: Key scaling factor
        v_scale: Value scaling factor
        max_seq_len: Maximum sequence length
        outputs: Optional pre-allocated output tensor
        force_split_kv: Whether to force use split KV mode
        force_persistent: Whether to force use persistent mode

    Returns:
        torch.Tensor: Attention output
    """
    raise NotImplementedError(
        f"flashinfer.attention.decode_attention_kv_paged is not implemented for {get_current_backend()}"
    )


@dispatch(
    "flashinfer.attention.decode_mla_kv_paged",
)
def decode_mla_kv_paged(
    q,
    q_rope,
    kv_cache,
    k_rope,
    actual_seq_lens,
    block_tables,
    k_scale,
    v_scale,
    max_seq_len: int = -1,
    outputs: Optional[torch.Tensor] = None,
    force_split_kv: bool = False,
    force_persistent: bool = False,
):
    """
    FlashInfer operation: Decode MLA (Multi-Latent Attention) with paged KV cache.

    Args:
        q: Query tensor
        q_rope: Query RoPE embeddings
        kv_cache: Paged KV cache
        k_rope: Key RoPE embeddings
        actual_seq_lens: Actual sequence lengths
        block_tables: Block tables for paged attention
        k_scale: Key scaling factor
        v_scale: Value scaling factor
        max_seq_len: Maximum sequence length
        outputs: Optional pre-allocated output tensor
        force_split_kv: Whether to force use split KV mode
        force_persistent: Whether to force use persistent mode

    Returns:
        torch.Tensor: MLA attention output
    """
    raise NotImplementedError(
        f"flashinfer.attention.decode_mla_kv_paged is not implemented for {get_current_backend()}"
    )


@dispatch(
    "flashinfer.attention.prefill_attention_kv_paged",
)
def prefill_attention_kv_paged(
    q,
    k_cache,
    v_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    actual_seq_offset,
    block_tables,
    k_scale,
    v_scale,
    num_batch,
    max_seq_len,
    is_causal: bool = True,
    outputs: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    use_lpt_scheduler: bool = True,
):
    """
    FlashInfer operation: Prefill attention with paged KV cache.

    Args:
        q: Query tensor
        k_cache: Paged key cache
        v_cache: Paged value cache
        actual_seq_lens_q: Actual query sequence lengths
        actual_seq_lens_kv: Actual KV sequence lengths
        actual_seq_offset: Sequence offsets
        block_tables: Block tables for paged attention
        k_scale: Key scaling factor
        v_scale: Value scaling factor
        num_batch: Number of batches
        max_seq_len: Maximum sequence length
        is_causal: Whether to apply causal masking
        outputs: Optional pre-allocated output tensor
        out_lse: Optional output log-sum-exp tensor
        use_lpt_scheduler: Whether to use LPT scheduler

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (attention output, lse output)
    """
    raise NotImplementedError(
        f"flashinfer.attention.prefill_attention_kv_paged is not implemented for {get_current_backend()}"
    )


@dispatch(
    "flashinfer.attention.prefill_attention_kv_ragged",
)
def prefill_attention_kv_ragged(
    q,
    k_cache,
    v_cache,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    actual_seq_offset,
    block_tables,
    k_scale,
    v_scale,
    num_batch,
    max_seq_len,
    is_causal: bool = True,
    outputs: Optional[torch.Tensor] = None,
    out_lse: Optional[torch.Tensor] = None,
    use_lpt_scheduler: bool = True,
):
    """
    FlashInfer operation: Prefill attention with ragged KV cache.

    Args:
        q: Query tensor
        k_cache: Ragged key cache
        v_cache: Ragged value cache
        actual_seq_lens_q: Actual query sequence lengths
        actual_seq_lens_kv: Actual KV sequence lengths
        actual_seq_offset: Sequence offsets
        block_tables: Block tables (unused for ragged)
        k_scale: Key scaling factor
        v_scale: Value scaling factor
        num_batch: Number of batches
        max_seq_len: Maximum sequence length
        is_causal: Whether to apply causal masking
        outputs: Optional pre-allocated output tensor
        out_lse: Optional output log-sum-exp tensor
        use_lpt_scheduler: Whether to use LPT scheduler

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (attention output, lse output)
    """
    raise NotImplementedError(
        f"flashinfer.attention.prefill_attention_kv_ragged is not implemented for {get_current_backend()}"
    )


__all__ = [
    # GEMM (cuTile backends available, no dot_scaled)
    "gemm_alpha_beta",
    "masked_bmm",
    "ragged_bmm",
    "ragged_block_scaled_bmm",
    # RoPE
    "rope_quantize_fp8",
    # Quantization
    "per_token_group_quant_8bit",
    # Attention
    "decode_attention_kv_paged",
    "decode_mla_kv_paged",
    "prefill_attention_kv_paged",
    "prefill_attention_kv_ragged",
]
