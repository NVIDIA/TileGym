# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""TileGym ops module - contains all operation interfaces and backend implementations"""

from tilegym.backend import is_backend_available

# Backend implementations
# Import interface modules
from . import activation
from . import attn_interface

# Make cutile optional - only import if backend is fully available
if is_backend_available("cutile"):
    try:
        from . import cutile
    except (ImportError, RuntimeError):
        import warnings

        warnings.warn("Cutile backend import failed, cutile operations will not be available")
        cutile = None  # type: ignore
else:
    cutile = None  # type: ignore

from . import moe_interface

# Re-export key interfaces
from .attn_interface import fmha_interface
from .attn_interface import get_fmha_interface
from .attn_interface import mla_decoding_interface
from .attn_interface import mla_interface
from .moe_interface import fused_moe_kernel_interface

# Import all operation interfaces from the unified ops module
from .ops import *

__all__ = [
    # Export all operations from ops module
    # Backend implementations
    # Interface modules
    "attn_interface",
    "moe_interface",
    # Re-exported submodules
    # Key interfaces
    "fmha_interface",
    "get_fmha_interface",
    "mla_interface",
    "mla_decoding_interface",
    "fused_moe_kernel_interface",
]

# Add cutile to exports only if successfully imported
if cutile is not None:
    __all__.append("cutile")
