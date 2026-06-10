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
from .ops import cross_entropy
from .ops import fused_linear_jsd
from .ops import fused_neighborhood_attention
from .ops import geglu
from .ops import group_norm
from .ops import jsd
from .ops import kl_div
from .ops import layer_norm
from .ops import llama4_rope
from .ops import multi_token_attention
from .ops import qwen2vl_mrope
from .ops import rope
from .ops import sparsemax
from .ops import tiled_mlp

__all__ = [
    "cross_entropy",
    "fused_linear_jsd",
    "fused_neighborhood_attention",
    "geglu",
    "group_norm",
    "jsd",
    "kl_div",
    "layer_norm",
    "llama4_rope",
    "multi_token_attention",
    "qwen2vl_mrope",
    "rope",
    "sparsemax",
    "tiled_mlp",
]
