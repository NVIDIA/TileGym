# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unsloth Suite - Kernels ported from https://github.com/unslothai/unsloth

Usage:
    from tilegym.suites import unsloth
    output = unsloth.geglu_exact_forward(gate, up)
    output = unsloth.grouped_gemm(X, W, m_sizes, topk)
"""

from tilegym.backend import is_backend_available

if is_backend_available("cutile"):
    from . import cutile as _cutile_impl
from .ops import geglu_approx_backward
from .ops import geglu_approx_forward
from .ops import geglu_exact_backward
from .ops import geglu_exact_forward
from .ops import grouped_gemm

__all__ = [
    "geglu_exact_forward",
    "geglu_exact_backward",
    "geglu_approx_forward",
    "geglu_approx_backward",
    "grouped_gemm",
]
