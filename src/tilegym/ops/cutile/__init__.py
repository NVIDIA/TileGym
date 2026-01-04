# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile backend implementations for all TileGym operations"""

from tilegym.backend import is_backend_available

# Only import if cutile backend is available
if is_backend_available("cutile"):
    # Activation functions
    # Non-DL operations
    # Linear algebra operations
    # NN operations
    from . import activation
    from . import attention
    from . import dropout
    from . import flash_decode
    from . import group_gemm
    from . import matmul
    from . import mla
    from . import mla_decoding
    from . import mla_decoding_split_kv
    from . import moe
    from . import moe_align_block
    from . import rms_norm
    from . import rope
    from . import silu_and_mul
    from . import softmax
    from . import splitk_reduce
    from . import swiglu

    # Import specific functions for direct access
    from .flash_decode import fmha_decode
    from .moe import fused_moe_kernel as invoke_fused_moe_kernel
    from .moe_align_block import moe_align_block_size
    from .rms_norm import get_rms_norm_module
    from .rms_norm import rms_norm
    from .rope import apply_rope_base
    from .rope import get_apply_rope_func
    from .silu_and_mul import silu_and_mul
    from .softmax import softmax
    from .splitk_reduce import splitk_reduce
    from .swiglu import get_swiglu
    from .swiglu import get_swiglu_module

    __all__ = [
        # NN operations
        "fmha_decode",
        "flash_decode",
        "splitk_reduce",
        "invoke_fused_moe_kernel",
        "moe_align_block_size",
        "attention",
        "mla",
        "mla_decoding",
        "get_swiglu_module",
        "get_swiglu",
        "get_apply_rope_func",
        "get_rms_norm_module",
        "rms_norm",
        "silu_and_mul",
        "dropout",
        "softmax",
        "mla_decoding_split_kv",
        "moe",
        "moe_align_block",
        "rope",
        "swiglu",
        "apply_rope_base",
        # Linalg operations
        "matmul",
        "group_gemm",
    ]
else:
    __all__ = []
