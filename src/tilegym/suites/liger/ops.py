# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
liger Suite - Unified interface for Liger-Kernel compatible operations
"""

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import torch

from tilegym.backend import dispatch
from tilegym.backend import get_current_backend


@dispatch(
    "liger.jsd",
)
def jsd(
    input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Generalized Jensen-Shannon Divergence loss.

    JSD(β)(P || Q) = β * KL(P || M) + (1-β) * KL(Q || M), M = β*P + (1-β)*Q.

    Args:
        input: Student model log-probabilities with shape (BT, V)
        target: Teacher model log-probabilities with shape (BT, V)
        shift_labels: Optional token indices for per-row masking, shape (BT,)
        beta: Interpolation coefficient in [0, 1].
            beta=0 → forward KL, beta=1 → reverse KL, beta=0.5 → symmetric JSD.
            Default: 0.5
        ignore_index: Label index to ignore when shift_labels is provided. Default: -100

    Returns:
        Scalar loss tensor
    """
    raise NotImplementedError(f"jsd is not implemented for {get_current_backend()}")


@dispatch(
    "liger.fused_linear_jsd",
)
def fused_linear_jsd(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Fused linear + Jensen-Shannon Divergence loss (chunked to avoid materializing logits).

    Computes JSD between student and teacher distributions without materializing the
    full (BT, V) logit matrices.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_jsd.py

    Args:
        student_input: Student hidden states of shape (BT, H).
        student_weight: Student vocabulary weight of shape (V, H).
        teacher_input: Teacher hidden states of shape (BT, H).
        teacher_weight: Teacher vocabulary weight of shape (V, H).
        shift_labels: Optional token indices for masking, shape (BT,). Default: None
        beta: JSD interpolation coefficient in [0, 1]. Default: 0.5
        ignore_index: Label index to ignore. Default: -100
        temperature: Temperature for softmax scaling. Default: 1.0

    Returns:
        Scalar loss tensor.
    """
    raise NotImplementedError(f"fused_linear_jsd is not implemented for {get_current_backend()}")


@dispatch(
    "liger.geglu",
)
def geglu(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    GEGLU activation: c = GELU(a) * b using tanh approximation.

    Computes: c = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3))) * b

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py

    Args:
        a: Input gate tensor of shape (*, N).
        b: Input value tensor of shape (*, N).

    Returns:
        Output tensor of same shape as a and b.
    """
    raise NotImplementedError(f"geglu is not implemented for {get_current_backend()}")


@dispatch(
    "liger.layer_norm",
)
def layer_norm(
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Layer Normalization.

    Normalizes each row of X independently, then applies affine transform Y = norm(X) * W + B.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py

    Args:
        X: Input tensor of shape (*, H).
        W: Affine scale weight of shape (H,).
        B: Affine shift bias of shape (H,).
        eps: Epsilon for numerical stability. Default: 1e-5

    Returns:
        Normalized output tensor of same shape as X.
    """
    raise NotImplementedError(f"layer_norm is not implemented for {get_current_backend()}")
