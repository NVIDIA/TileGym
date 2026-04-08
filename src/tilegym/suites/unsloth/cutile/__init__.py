# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""CuTile implementations for unsloth suite."""

# Shared CuTile op helpers (sigmoid, erf, etc.)
from . import ct_ops  # noqa: F401
from .cross_entropy_loss import cross_entropy_loss as cross_entropy_loss_fn
from .fp8 import act_quant
from .fp8 import w8a8_block_fp8_matmul_cutile
from .fp8 import weight_dequant_block
from .geglu import geglu_approx_backward
from .geglu import geglu_approx_forward
from .geglu import geglu_exact_backward
from .geglu import geglu_exact_forward
from .grouped_gemm import grouped_gemm_cutile
from .layernorm import layernorm as layernorm_fn
from .rms_layernorm import rms_layernorm
from .rope_embedding import rope_embedding
from .rope_embedding import rope_embedding_qk
from .swiglu import swiglu_bwd
from .swiglu import swiglu_fg

__all__ = [
    "act_quant",
    "cross_entropy_loss_fn",
    "ct_ops",
    "geglu_approx_backward",
    "geglu_approx_forward",
    "geglu_exact_backward",
    "geglu_exact_forward",
    "grouped_gemm_cutile",
    "layernorm_fn",
    "rms_layernorm",
    "rope_embedding",
    "rope_embedding_qk",
    "swiglu_bwd",
    "swiglu_fg",
    "w8a8_block_fp8_matmul_cutile",
    "weight_dequant_block",
]
