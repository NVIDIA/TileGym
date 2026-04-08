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

from .ops import act_quant
from .ops import cross_entropy_loss
from .ops import geglu_approx_backward
from .ops import geglu_approx_forward
from .ops import geglu_exact_backward
from .ops import geglu_exact_forward
from .ops import grouped_gemm
from .ops import layernorm
from .ops import rms_layernorm
from .ops import rope_embedding
from .ops import rope_embedding_qk
from .ops import swiglu_bwd
from .ops import swiglu_fg
from .ops import w8a8_block_fp8_matmul
from .ops import weight_dequant

__all__ = [
    "act_quant",
    "cross_entropy_loss",
    "geglu_approx_backward",
    "geglu_approx_forward",
    "geglu_exact_backward",
    "geglu_exact_forward",
    "grouped_gemm",
    "layernorm",
    "rms_layernorm",
    "rope_embedding",
    "rope_embedding_qk",
    "swiglu_bwd",
    "swiglu_fg",
    "w8a8_block_fp8_matmul",
    "weight_dequant",
]
