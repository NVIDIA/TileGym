# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
TileGym operations interface - unified interface for all operations.

This module provides operation interfaces that automatically dispatch to the appropriate
backend implementation based on the current backend setting.
"""

from typing import Any
from typing import List
from typing import Optional

import torch

from tilegym.backend import dispatch
from tilegym.backend import get_current_backend

# ============================================================================
# NN Operations
# ============================================================================


@dispatch(
    "get_apply_rope_func",
)
def get_apply_rope_func(model: str = "llama"):
    """
    Returns a callable that applies Rotary Position Embedding (RoPE) for a given model variant.

    Args:
        model: Model name that determines the RoPE layout transformation. Supported: 'llama', 'deepseek'

    Returns:
        Callable implementing RoPE application with signature similar to `apply_rope_base`
    """
    raise NotImplementedError(f"get_apply_rope_func is not implemented for {get_current_backend()}")


@dispatch(
    "apply_rope_base",
)
def apply_rope_base(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
    use_tma: bool = False,
):
    """
    Applies Rotary Position Embedding (RoPE) to query and key tensors.

    Args:
        q: Query tensor of shape (B, H_q, S, D)
        k: Key tensor of shape (B, H_kv, S, D)
        cos: Cosine tensor of shape (1, S, D) or (B, S, D)
        sin: Sine tensor with the same shape as `cos`
        position_ids: Optional - Position IDs tensor, default None
        unsqueeze_dim: Optional - Dimension to unsqueeze, default 1
        use_tma: Whether to use TMA optimized path when available

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Query and key tensor pair with RoPE applied
    """
    raise NotImplementedError(f"apply_rope_base is not implemented for {get_current_backend()}")


@dispatch(
    "get_swiglu_module",
)
def get_swiglu_module():
    """
    Returns the SwiGLU MLP module class.

    The returned module computes: down_proj(SiLU(gate_proj(x)) * up_proj(x)).

    Returns:
        nn.Module subclass implementing the SwiGLU MLP
    """
    raise NotImplementedError(f"get_swiglu_module is not implemented for {get_current_backend()}")


@dispatch(
    "get_swiglu",
)
def get_swiglu():
    """
    Returns a functional SwiGLU implementation for elementwise SiLU*mul fusion.

    The function expects two tensors `a` and `b` of the same shape and computes
    SiLU(a) * b.

    Expected input shapes:
        a: (batch_size, seq_len, intermediate_size)
        b: (batch_size, seq_len, intermediate_size)
    """
    raise NotImplementedError(f"get_swiglu is not implemented for {get_current_backend()}")


def get_fused_swiglu_module():
    """
    Returns the fused SwiGLU module class.

    This module uses a partial fused kernel for the entire SwiGLU operation:
    output = activation(input @ W1_act^T) ⊙ (input @ W1_noact^T) @ W2^T

    This eliminates ALL PyTorch linear operations and intermediate tensor materializations,
    providing better performance than get_swiglu_module().

    Note: This doesn't need backend dispatch - the PartiallyFusedSwiGLUMLP class automatically
    dispatches to the correct backend kernel internally.

    Returns:
        PartiallyFusedSwiGLUMLP class
    """
    from tilegym.ops.fused_swiglu import PartiallyFusedSwiGLUMLP

    return PartiallyFusedSwiGLUMLP


@dispatch(
    "rms_norm",
)
def rms_norm(
    input: torch.Tensor,
    normalized_shape: Any,
    weight: torch.Tensor,
    eps: float,
    bias: Optional[torch.Tensor] = None,
    static_persistent: bool = False,
    **kwargs: Any,
):
    """
    Returns the Root-Mean-Squared Norm of input along dimension N.

    Args:
        input: Tensor of shape (M, N)
        normalized_shape: Unused
        weight: Tensor of shape (N,)
        eps: small scaler to be added to variance calculation prior to division.
        bias: Bias tensor of shape (N,), default is None
        static_persistent: Whether to use static persistent kernel, default is False
        **kwargs: Additional arguments for backend-specific configurations
    """
    raise NotImplementedError(f"rms_norm is not implemented for {get_current_backend()}")


@dispatch(
    "get_rms_norm_module",
)
def get_rms_norm_module():
    """
    Returns the RMSNorm module class.
    """
    raise NotImplementedError(f"get_rms_norm_module is not implemented for {get_current_backend()}")


@dispatch(
    "silu_and_mul",
)
def silu_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    **kwargs: Any,
):
    """
    Fused SiLU and multiply operation.

    Implements: SiLU(input[..., :H]) * input[..., H:].

    Args:
        input: Tensor with last-dimension size 2*H
        out: Optional preallocated output tensor, if specified kernel will update in-place
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        torch.Tensor with shape input[..., :H]
    """
    raise NotImplementedError(f"silu_and_mul is not implemented for {get_current_backend()}")


@dispatch(
    "dropout",
)
def dropout(
    x: torch.Tensor,
    seed: int,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    **kwargs: Any,
):
    """
    Dropout operation with stateless random generation from a given seed.

    Args:
        x: Input tensor
        seed: Integer seed used to generate dropout mask in the kernel
        p: Drop probability, default is 0.5
        training: Apply dropout if True, otherwise return input
        inplace: If True, perform the operation in-place
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Tensor of the same shape as `x` with dropout applied
    """
    raise NotImplementedError(f"dropout is not implemented for {get_current_backend()}")


@dispatch(
    "softmax",
)
def softmax(
    x: torch.Tensor,
    use_tma: bool = False,
    **kwargs: Any,
):
    """
    Performs Softmax on a 2D tensor (M, N) along the N axis.

    Args:
        x: Input tensor of shape (M, N). Softmax is computed over the last dimension (N)
        use_tma: Whether to use TMA (Tensor Memory Accelerator) implementation
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        torch.Tensor of shape (M, N) containing softmax probabilities
    """
    raise NotImplementedError(f"softmax is not implemented for {get_current_backend()}")


@dispatch(
    "fmha",
)
def fmha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: Optional[float] = None,
    is_causal: bool = True,
    **kwargs: Any,
):
    """
    Fused Multi-Head Attention (prefill) operation.

    This is the main FMHA kernel for prefill phase, corresponding to:
    - cutile: tile_fmha in tilegym.ops.cutile.attention

    Args:
        q: Query tensor of shape (B, H, S, D)
        k: Key tensor of shape (B, H, S, D)
        v: Value tensor of shape (B, H, S, D)
        scaling: Scale factor for attention scores (default: 1/sqrt(D))
        is_causal: Whether to apply causal masking
        **kwargs: Additional arguments, including:
            - has_backward (bool): Whether backward pass is needed (default: False)
            - kernel_configs (dict): Backend-specific kernel configurations

    Returns:
        Output tensor of shape (B, H, S, D)
    """
    raise NotImplementedError(f"fmha is not implemented for {get_current_backend()}")


@dispatch(
    "fmha_decode",
)
def fmha_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: Optional[float],
    kv_len_per_split: Optional[int] = None,
    **kwargs: Any,
):
    """
    Grouped Query Attention decoding for a single-token query.

    Args:
        q: Query tensor of shape (B, H_q, 1, D)
        k: Key tensor of shape (B, H_kv, S_kv, D)
        v: Value tensor of shape (B, H_kv, S_kv, D)
        sm_scale: Scale factor for attention computation
        kv_len_per_split: Optional KV length per split for parallelization
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Output tensor of shape (B, H_q, 1, D)
    """
    raise NotImplementedError(f"fmha_decode is not implemented for {get_current_backend()}")


@dispatch(
    "mla",
)
def mla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qpe: torch.Tensor,
    kpe: torch.Tensor,
    is_causal: bool,
    scaling: Optional[float] = None,
    **kwargs: Any,
):
    """
    Multi-Latent Attention (MLA) prefill operation.

    This is the main MLA kernel for prefill phase, corresponding to:
    - cutile: tile_mla in tilegym.ops.cutile.mla

    Args:
        q: Query tensor of shape (B, H, S, D)
        k: Key tensor of shape (B, H, S, D)
        v: Value tensor of shape (B, H, S, D)
        qpe: Query positional embedding of shape (B, H, S, D_PE)
        kpe: Key positional embedding of shape (B, 1, S, D_PE)
        is_causal: Whether to apply causal masking
        scaling: Scale factor for attention scores (default: 1/sqrt(D + D_PE))
        **kwargs: Additional arguments, including kernel_configs if needed

    Returns:
        Output tensor of shape (B, H, S, D)
    """
    raise NotImplementedError(f"mla is not implemented for {get_current_backend()}")


@dispatch(
    "mla_decoding",
)
def mla_decoding(
    q: torch.Tensor,
    qpe: torch.Tensor,
    kv: torch.Tensor,
    kpe: torch.Tensor,
    sm_scale: Optional[float] = None,
    transpose: bool = True,
    **kwargs: Any,
):
    """
    MLA decoding kernel for a single-step query attending to KV cache.

    Args:
        q: Query tensor of shape (B, H, D) for single-token decoding
        qpe: Query positional embedding tensor of shape (B, H, D_PE)
        kv: Key/Value cache tensor shaped as required by the backend kernel
        kpe: Key positional embedding tensor compatible with KV layout
        sm_scale: Optional softmax scaling factor
        transpose: Whether to transpose internal layouts to match kernels
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            o: (B, H, D)
            l: (B, H)
    """
    raise NotImplementedError(f"mla_decoding is not implemented for {get_current_backend()}")


@dispatch(
    "mla_decoding_split_kv",
)
def mla_decoding_split_kv(
    q: torch.Tensor,
    qpe: torch.Tensor,
    kv: torch.Tensor,
    kpe: torch.Tensor,
    sm_scale: Optional[float] = None,
    kv_len_per_split: Optional[int] = None,
    **kwargs: Any,
):
    """
    MLA decoding with split-KV parallel reduction.

    Args:
        q: Query tensor of shape (B, H, D)
        qpe: Query positional embedding tensor of shape (B, H, D_PE)
        kv: Key/Value cache tensor of shape (B, S_kv, D)
        kpe: Key positional embedding tensor of shape (B, S_kv, D_PE)
        sm_scale: Optional softmax scaling factor
        kv_len_per_split: Optional per-split KV length to control parallelism
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        Output tensor of shape (B, H, D)
    """
    raise NotImplementedError(f"mla_decoding_split_kv is not implemented for {get_current_backend()}")


@dispatch(
    "splitk_reduce",
)
def splitk_reduce(
    attn_splitk_out: torch.Tensor,
    lse_splitk_out: torch.Tensor,
    attn_out: torch.Tensor,
    S_kv: int,
    **kwargs: Any,
):
    """
    Reduce the intermediate attention results and lse results into the final output for attention decode.

    Args:
        attn_splitk_out: Intermediate attention results [B, H, NUM_KV_SPLITS, D]
        lse_splitk_out: Intermediate log-sum-exp stats [B, H, NUM_KV_SPLITS]
        attn_out: Output tensor [B, H, D]
        S_kv: KV sequence length for boundary handling
        **kwargs: Additional arguments for backend-specific configurations

    Returns:
        The finalized `attn_out` tensor [B, H, D]
    """
    raise NotImplementedError(f"splitk_reduce is not implemented for {get_current_backend()}")


# ============================================================================
# Linear Algebra Operations
# ============================================================================


@dispatch(
    "matmul",
)
def matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: Optional[bool] = None,
    trans_b: Optional[bool] = None,
    static_persistent: Optional[bool] = True,
    use_tma: Optional[bool] = True,
    **kwargs: Any,
):
    """
    Matrix multiplication operation that automatically selects implementation based on current backend

    Args:
        a: Input matrix A
        b: Input matrix B
        trans_a: Whether to transpose matrix A (None uses backend default)
        trans_b: Whether to transpose matrix B (None uses backend default)
        static_persistent: Whether to use static persistent mode (default: True)
        use_tma: Whether to use TMA (default: True)
        **kwargs: Additional arguments, including kernel_configs if needed
    Returns:
        torch.Tensor: Matrix multiplication result
    """
    raise NotImplementedError(f"Matmul is not implemented for this backend: {get_current_backend()}")


@dispatch(
    "group_gemm",
)
def group_gemm(
    group_A: List[torch.Tensor],
    group_B: List[torch.Tensor],
    static_persistent: Optional[bool] = True,
    use_tma: Optional[bool] = True,
    **kwargs: Any,
):
    """
    Group GEMM operation that automatically selects implementation based on current backend

    Performs multiple matrix multiplications in batch mode, potentially with better performance
    than running individual multiplications.

    Args:
        group_A: List of input matrices A
        group_B: List of input matrices B
        static_persistent: Whether to use static persistent mode (default: True)
        use_tma: Whether to use TMA (default: True)
        **kwargs: Additional arguments, including kernel_configs if needed with keys:
            - BLOCK_M: Tile size for M dimension
            - BLOCK_N: Tile size for N dimension
            - BLOCK_K: Tile size for K dimension
            - num_ctas: Number of CTAs per SM
    Returns:
        List[torch.Tensor]: Results of matrix multiplications
    """
    raise NotImplementedError(f"Group GEMM is not implemented for this backend: {get_current_backend()}")
