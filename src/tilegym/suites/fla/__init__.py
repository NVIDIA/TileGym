# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""FLA linear-attention operators."""

# Import backend implementations to register them with the dispatcher.
from tilegym.backend import is_backend_available

if is_backend_available("cutile"):
    from . import cutile as _cutile_impl  # noqa: F401

# Unified dispatch interface.
from .ops import linear_attention_chunk
from .ops import linear_attention_deltanet_fused_recurrent
from .ops import linear_attention_fused_chunk
from .ops import linear_attention_gdn_chunk
from .ops import linear_attention_gdn_fused_recurrent
from .ops import linear_attention_mamba2_fused_chunk
from .ops import linear_attention_mamba2_fused_recurrent

__all__ = [
    "linear_attention_chunk",
    "linear_attention_fused_chunk",
    "linear_attention_mamba2_fused_chunk",
    "linear_attention_mamba2_fused_recurrent",
    "linear_attention_deltanet_fused_recurrent",
    "linear_attention_gdn_chunk",
    "linear_attention_gdn_fused_recurrent",
]
