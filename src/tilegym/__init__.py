# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT


def _check_torch_dependencies():
    """Verify that PyTorch is installed with helpful error message."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "\n\n[TileGym] PyTorch is required to run this library.\n"
            "Since CUDA versions vary significantly across devices, please "
            "manually install the version that matches your hardware:\n"
            "👉 https://pytorch.org/get-started/locally/\n"
            "Alternatively, try: pip install tilegym[torch]\n"
        ) from None


def _check_ct_experimental_dependency():
    """Verify that cuda-tile-experimental is installed with helpful error message."""
    try:
        import cuda.tile_experimental  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "\n\n[TileGym] cuda-tile-experimental is required but not installed.\n"
            "It is not available on PyPI and must be installed from source:\n\n"
            '  pip install "cuda-tile-experimental @ '
            'git+https://github.com/NVIDIA/cutile-python.git#subdirectory=experimental"\n\n'
            "See: https://github.com/NVIDIA/cutile-python?tab=readme-ov-file"
            "#experimental-features-optional\n"
        ) from None


# Check dependencies before any imports
_check_torch_dependencies()
_check_ct_experimental_dependency()

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
__version__ = "1.1.0"

import contextlib
from enum import Enum
