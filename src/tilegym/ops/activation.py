# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import torch

from tilegym.backend import dispatch
from tilegym.backend import get_current_backend


@dispatch(
    "relu",
)
def relu(x: torch.Tensor) -> torch.Tensor:
    """
    Applies the Rectified Linear Unit function element-wise.

    ReLU(x) = max(0, x)

    Args:
        x: Input tensor

    Returns:
        Output tensor with ReLU applied
    """
    raise NotImplementedError(f"relu is not implemented for {get_current_backend()}")


@dispatch(
    "geglu",
)
def geglu(input: torch.Tensor, dim: int = -1, approximate: str = "none") -> torch.Tensor:
    """
    Applies the Gated GELU (GEGLU) activation function.

    GEGLU(x) = a ⊗ GELU(b)
    where a is the first half of the input and b is the second half,
    split along the specified dimension.

    Args:
        input: Input tensor
        dim: Dimension along which to split the input (default: -1)
        approximate: The approximation type for GELU. Can be 'none' or 'tanh'

    Returns:
        Output tensor with GEGLU applied
    """
    raise NotImplementedError(f"geglu is not implemented for {get_current_backend()}")


@dispatch(
    "gelu",
)
def gelu(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    """
    Applies the Gaussian Error Linear Unit function element-wise.

    GELU(x) = x * Φ(x)
    where Φ(x) is the Cumulative Distribution Function for Gaussian Distribution

    Args:
        x: Input tensor
        approximate: The approximation type. Can be 'none' or 'tanh'

    Returns:
        Output tensor with GELU applied
    """
    raise NotImplementedError(f"gelu is not implemented for {get_current_backend()}")
