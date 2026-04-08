# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unsloth Suite - Dispatch interfaces for Unsloth kernels.

Source: https://github.com/unslothai/unsloth
Upstream commit: TODO (pin after migration)

"""

from typing import List
from typing import Optional
from typing import Tuple

import torch

from tilegym.backend import dispatch
from tilegym.backend import get_current_backend

# =============================================================================
# Activation: SwiGLU
# =============================================================================


@dispatch(
    "unsloth.swiglu_fg",
)
def swiglu_fg(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU forward: h = silu(e) * g = (e * sigmoid(e)) * g

    Args:
        e: Gate input, shape (batch, seq_len, hd)
        g: Up-projection input, shape (batch, seq_len, hd)

    Returns:
        Output tensor, same shape as inputs.
    """
    raise NotImplementedError(f"swiglu_fg not implemented for {get_current_backend()}")


@dispatch(
    "unsloth.swiglu_bwd",
)
def swiglu_bwd(DW: torch.Tensor, e: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SwiGLU backward (in-place). Overwrites DW, e, g with:
      DW → h = f*g,  e → df,  g → de

    Args:
        DW: Upstream gradient, shape (batch*seq_len, hd)
        e: Gate input (flattened 2D)
        g: Up-projection input (flattened 2D)

    Returns:
        Tuple (DW, e, g) — all modified in-place.
    """
    raise NotImplementedError(f"swiglu_bwd not implemented for {get_current_backend()}")


# =============================================================================
# Activation: GEGLU
# =============================================================================


@dispatch(
    "unsloth.geglu_exact_forward",
)
def geglu_exact_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    GEGLU exact forward: out = GELU_exact(gate) * up
    Uses erf: f = 0.5 * gate * (1 + erf(gate / sqrt(2)))

    Args:
        gate: Gate input, shape (batch, seq_len, hd)
        up: Up-projection input, shape (batch, seq_len, hd)

    Returns:
        Output tensor, same shape as inputs.
    """
    raise NotImplementedError(f"geglu_exact_forward not implemented for {get_current_backend()}")


@dispatch(
    "unsloth.geglu_exact_backward",
)
def geglu_exact_backward(
    DW: torch.Tensor, e: torch.Tensor, g: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GEGLU exact backward (in-place).

    Args:
        DW: Upstream gradient, shape (batch*seq_len, hd)
        e: Gate input (flattened 2D)
        g: Up-projection input (flattened 2D)

    Returns:
        Tuple (DW, e, g) — all modified in-place.
    """
    raise NotImplementedError(f"geglu_exact_backward not implemented for {get_current_backend()}")


@dispatch(
    "unsloth.geglu_approx_forward",
)
def geglu_approx_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    GEGLU approximate forward: out = GELU_approx(gate) * up
    Uses tanh approximation.

    Args:
        gate: Gate input, shape (batch, seq_len, hd)
        up: Up-projection input, shape (batch, seq_len, hd)

    Returns:
        Output tensor, same shape as inputs.
    """
    raise NotImplementedError(f"geglu_approx_forward not implemented for {get_current_backend()}")


@dispatch(
    "unsloth.geglu_approx_backward",
)
def geglu_approx_backward(
    DW: torch.Tensor, e: torch.Tensor, g: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GEGLU approximate backward (in-place).

    Args:
        DW: Upstream gradient, shape (batch*seq_len, hd)
        e: Gate input (flattened 2D)
        g: Up-projection input (flattened 2D)

    Returns:
        Tuple (DW, e, g) — all modified in-place.
    """
    raise NotImplementedError(f"geglu_approx_backward not implemented for {get_current_backend()}")


# =============================================================================
# Normalization: RMS LayerNorm
# =============================================================================


@dispatch(
    "unsloth.rms_layernorm",
)
def rms_layernorm(X: torch.Tensor, W: torch.Tensor, eps: float = 1e-6, gemma: bool = False) -> torch.Tensor:
    """
    RMS LayerNorm (autograd-capable forward + backward).

    When gemma=True, uses Gemma-style: (W + 1) * norm(X).

    Args:
        X: Input tensor, shape (*, hidden_dim)
        W: Weight tensor, shape (hidden_dim,)
        eps: Epsilon for numerical stability
        gemma: Use Gemma variant (W + 1) instead of W

    Returns:
        Normalized tensor, same shape as X.
    """
    raise NotImplementedError(f"rms_layernorm not implemented for {get_current_backend()}")


# =============================================================================
# Normalization: LayerNorm
# =============================================================================


@dispatch(
    "unsloth.layernorm",
)
def layernorm(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Full LayerNorm with bias (autograd-capable forward + backward).

    Args:
        X: Input tensor, shape (*, hidden_dim)
        W: Weight tensor, shape (hidden_dim,)
        b: Bias tensor, shape (hidden_dim,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor, same shape as X.
    """
    raise NotImplementedError(f"layernorm not implemented for {get_current_backend()}")


# =============================================================================
# Quantization: FP8
# =============================================================================


@dispatch(
    "unsloth.weight_dequant",
)
def weight_dequant(
    x: torch.Tensor,
    s: torch.Tensor,
    block_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    FP8 block-wise weight dequantization.

    Args:
        x: FP8 quantized weight, shape (M, N)
        s: Block scales, shape (ceil(M/block_size), ceil(N/block_size))
        block_size: Block quantization tile size
        dtype: Output dtype

    Returns:
        Dequantized weight tensor, shape (M, N).
    """
    raise NotImplementedError(f"weight_dequant not implemented for {get_current_backend()}")


@dispatch(
    "unsloth.act_quant",
)
def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 block-wise activation quantization.

    Args:
        x: Activation tensor (last dim divisible by block_size)
        block_size: Block size for quantization

    Returns:
        Tuple (y, s): y is FP8 quantized tensor, s is per-block scales.
    """
    raise NotImplementedError(f"act_quant not implemented for {get_current_backend()}")


@dispatch(
    "unsloth.w8a8_block_fp8_matmul",
)
def w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Block-wise FP8 matrix multiplication (W8A8).

    Args:
        A: FP8 activations, shape (..., K)
        B: FP8 weights, shape (N, K)
        As: Activation scales, shape (..., ceil(K/block_k))
        Bs: Weight scales, shape (ceil(N/block_n), ceil(K/block_k))
        block_size: [block_n, block_k], defaults to [128, 128]
        output_dtype: Output dtype

    Returns:
        Result tensor, shape (..., N).
    """
    raise NotImplementedError(f"w8a8_block_fp8_matmul not implemented for {get_current_backend()}")


# =============================================================================
# Embedding: RoPE
# =============================================================================


@dispatch(
    "unsloth.rope_embedding",
)
def rope_embedding(Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotary Position Embedding for a single tensor (autograd-capable).

    Args:
        Q: Query tensor, shape (batch, seq_len, n_heads, head_dim)
        cos: Cosine rotary table, shape (seq_len, head_dim/2)
        sin: Sine rotary table, shape (seq_len, head_dim/2)

    Returns:
        Rotated tensor, same shape as Q.
    """
    raise NotImplementedError(f"rope_embedding not implemented for {get_current_backend()}")


@dispatch(
    "unsloth.rope_embedding_qk",
)
def rope_embedding_qk(
    Q: torch.Tensor,
    K: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary Position Embedding for Q and K jointly (autograd-capable).
    Processes both Q and K in a single kernel launch for efficiency.

    Args:
        Q: Query tensor, shape (batch, n_heads_Q, seq_len, head_dim)
        K: Key tensor, shape (batch, n_heads_K, seq_len, head_dim)
        cos: Cosine rotary table, shape (seq_len, head_dim/2)
        sin: Sine rotary table, shape (seq_len, head_dim/2)
        rope_indices: Optional per-position rotation indices

    Returns:
        Tuple (Q_out, K_out), both rotated.
    """
    raise NotImplementedError(f"rope_embedding_qk not implemented for {get_current_backend()}")


# =============================================================================
# Loss: Cross-Entropy
# =============================================================================


@dispatch(
    "unsloth.cross_entropy_loss",
)
def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    n_items: Optional[int] = None,
) -> torch.Tensor:
    """
    Fast cross-entropy loss (autograd-capable).
    Automatically uses chunked computation for vocab > 65536.

    Args:
        logits: Input logits, shape (batch, seq_len, vocab_size)
        labels: Integer labels, shape (batch, seq_len). Use -100 to ignore.
        logit_softcapping: Gemma-2 style softcapping (0 = disabled)
        logit_scaling: Cohere style scaling (0 = disabled)
        n_items: Custom denominator for mean reduction (None = auto-count)

    Returns:
        Scalar mean cross-entropy loss.
    """
    raise NotImplementedError(f"cross_entropy_loss not implemented for {get_current_backend()}")


# =============================================================================
# MoE: Grouped GEMM
# =============================================================================


@dispatch(
    "unsloth.grouped_gemm",
)
def grouped_gemm(
    X: torch.Tensor,
    W: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    gather_indices: Optional[torch.Tensor] = None,
    permute_x: bool = False,
    permute_y: bool = False,
    topk_weights: Optional[torch.Tensor] = None,
    fuse_mul_post: bool = False,
    is_first_gemm: bool = True,
) -> torch.Tensor:
    """
    MoE grouped GEMM (autograd-capable forward + backward).

    Args:
        X: Input hidden states, shape (M, K)
        W: Expert weights, shape (E, N, K)
        m_sizes: Tokens per expert, shape (E,)
        topk: Number of top experts per token
        gather_indices: Token-to-expert assignment, shape (total_tokens,)
        permute_x: Whether X needs permutation
        permute_y: Whether output needs permutation
        topk_weights: Routing weights, shape (total_tokens,)
        fuse_mul_post: Fuse topk_weights multiplication into GEMM
        is_first_gemm: Whether this is the first or second grouped GEMM in MoE MLP.
            First GEMM: permute_x allowed, permute_y disallowed.
            Second GEMM: permute_y allowed, permute_x disallowed.

    Returns:
        Output tensor, shape (total_tokens, N).
    """
    raise NotImplementedError(f"grouped_gemm not implemented for {get_current_backend()}")
