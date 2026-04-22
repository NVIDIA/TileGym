# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
TileGym FlashInfer Suite

This suite contains implementations for FlashInfer-compatible operations.

Usage:
    from tilegym.suites import flashinfer

    # Use FlashInfer ops interface (recommended)


    # Use cuTile backend (if available)
    if flashinfer.cutile is not None:
        output = flashinfer.cutile.decode_attention_kv_paged(...)

Available backends:
- ops: FlashInfer operations interface (auto backend selection)
- cutile/: cuTile backend implementations (high-performance, if available)

See USAGE.md for detailed documentation and examples.
"""

import warnings

from tilegym.backend import is_backend_available

# Import ops interface first to register dispatch functions
from . import ops

if is_backend_available("cutile"):
    try:
        from . import cutile
    except (ImportError, RuntimeError):
        cutile = None
        warnings.warn("Cutile backend import failed in flashinfer suite, cutile operations will not be available")
else:
    cutile = None


ref = None
__all__ = ["ops"]


if cutile is not None:
    __all__.append("cutile")

if ref is not None:
    __all__.append("ref")
