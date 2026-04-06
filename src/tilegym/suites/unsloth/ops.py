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
