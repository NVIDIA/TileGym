# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""CUDA Tile C++ backend implementations for all TileGym operations"""

from tilegym.backend import is_backend_available

# Only import if CUDA Tile C++ backend is available
if is_backend_available("tilecpp"):
    # Activation functions
    from . import activation
    from . import attention
    from . import attention_sink
    from . import attention_sink_decode
    from . import bmm
    from . import chunk_gated_delta_rule
    from . import dropout
    from . import flash_decode
    from . import gemma_attention
    from . import gemma_attention_decode
    from . import layer_norm_legacy
    from . import matmul
    from . import mla
    from . import mla_decoding
    from . import mla_decoding_split_kv
    from . import moe
    from . import moe_align_block
    from . import persistent_layer_norm
    from . import recurrent_gated_delta_rule
    from . import rms_norm
    from . import rope
    from . import silu_and_mul
    from . import softmax
    from . import splitk_reduce
    from . import swiglu

    # Import specific functions for direct access
    from .dropout import dropout
    from .flash_decode import fmha_decode
    from .mla import tilecpp_mla
    from .moe import invoke_fused_moe_kernel
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
        # Activation functions
        "activation",
        # NN operations
        "fmha_decode",
        "flash_decode",
        "splitk_reduce",
        "invoke_fused_moe_kernel",
        "moe_align_block_size",
        "attention",
        "attention_sink",
        "mla",
        "mla_decoding",
        "tilecpp_mla",
        "get_swiglu_module",
        "get_swiglu",
        "get_apply_rope_func",
        "get_rms_norm_module",
        "rms_norm",
        "silu_and_mul",
        "dropout",
        "softmax",
        "mla_decoding_split_kv",
        "layer_norm_legacy",
        "rope",
        "swiglu",
        "apply_rope_base",
        "moe",
        "moe_align_block",
        "bmm",
        "matmul",
        # Additional modules
    ]

else:
    __all__ = []
