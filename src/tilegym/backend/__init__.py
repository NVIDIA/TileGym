# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Backend management for TileGym project
"""

from .dispatcher import dispatch
from .dispatcher import get_available_backends_for_op
from .dispatcher import get_registry_info
from .dispatcher import print_registry_info
from .dispatcher import register_impl
from .selector import assert_backend_available
from .selector import get_available_backends
from .selector import get_current_backend
from .selector import is_backend_available
from .selector import set_backend


def make_missing_backend_handler(package_name, available_items):
    """
    Create a unified __getattr__ function for cutile packages.

    Args:
        package_name: Name of the package (e.g., 'tilegym.nn.cutile')
        available_items: List of items that require cutile backend

    Returns:
        A __getattr__ function that provides helpful error messages
    """

    def __getattr__(name):
        """Provide helpful error messages when cutile backend is not available."""
        if not is_backend_available("cutile") and name in available_items:
            raise ImportError(
                f"'{name}' requires cutile backend. Please install cutile using the following command:\n"
                "pip install --pre --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi-local/simple cuda-tile"
            )
        raise AttributeError(f"module '{package_name}' has no attribute '{name}'")

    return __getattr__


__all__ = [
    # Backend selector
    "get_available_backends",
    "get_current_backend",
    "set_backend",
    "is_backend_available",
    "assert_backend_available",
    # Backend dispatcher
    "dispatch",
    "register_impl",
    "get_available_backends_for_op",
    "get_registry_info",
    "print_registry_info",
    # Cutile utilities
    "make_missing_backend_handler",
]
