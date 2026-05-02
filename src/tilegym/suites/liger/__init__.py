# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
liger Suite - cutile implementations for Liger-Kernel compatible operations

Usage:
    from tilegym.suites import liger
    output = liger.jsd(input_log_prob, target_log_prob)
"""

from tilegym.backend import is_backend_available

# Import backend implementations to register them

if is_backend_available("cutile"):
    from . import cutile as _cutile_impl

# Import unified interface
from .ops import fused_linear_jsd
from .ops import geglu
from .ops import jsd
from .ops import layer_norm

__all__ = [
    "fused_linear_jsd",
    "geglu",
    "jsd",
    "layer_norm",
]
