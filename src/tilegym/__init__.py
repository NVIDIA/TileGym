# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Import logging utilities
from .logger import get_logger
from .logger import set_env_log_level
from .logger import set_log_level
from .logger import warn_once

logger = get_logger()

# Initialize backend selector first to avoid import order issues
from .backend import get_available_backends
from .backend import get_available_backends_for_op
from .backend import get_current_backend
from .backend import get_registry_info
from .backend import is_backend_available
from .backend import print_registry_info
from .backend import set_backend

# Setup cutile integration
if is_backend_available("cutile"):
    # Apply experimental kernel tracking patch
    from .experimental import _apply_patch as _apply_experimental_patch

    _apply_experimental_patch()

# Import other modules
from . import ops  # Unified ops module

try:
    import transformers
except ImportError:
    logger.warning("transformers is not available")

__all__ = [
    "ops",  # Unified ops module
    "transformers",
    "set_backend",
    "get_current_backend",
    "get_available_backends",
    "is_backend_available",
    "get_available_backends_for_op",
    "get_registry_info",
    "print_registry_info",
    # Logging utilities
    "warn_once",
    "get_logger",
    "set_log_level",
    "set_env_log_level",
]

# Version info
__version__ = "0.1.0"

import contextlib
from enum import Enum
