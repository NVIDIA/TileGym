# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
TileGym Suites - cutile implementations for external kernel libraries

Usage:
    # Import flashinfer suite
    from tilegym.suites import flashinfer
    output = flashinfer.ops.decode_attention_kv_paged(q, k_cache, v_cache, ...)

    # Import unsloth suite
    from tilegym.suites import unsloth
    output = unsloth.geglu_exact_forward(gate, up)
"""

from typing import List


def list_available() -> List[str]:
    """List all available suites"""
    available = []
    try:
        from . import flashinfer
        from . import unsloth

        available.append("flashinfer")
        available.append("unsloth")
    except ImportError:
        pass
    return available


__all__ = ["list_available"]
