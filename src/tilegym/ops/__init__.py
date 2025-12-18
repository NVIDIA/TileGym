# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""TileGym ops module - contains all operation interfaces and backend implementations"""

from tilegym.backend import is_backend_available

# Backend implementations
# Import interface modules
from . import activation
from . import attn_interface
from . import cutile
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
    "cutile",
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
