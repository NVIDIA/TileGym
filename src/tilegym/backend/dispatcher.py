# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
TileGym Backend Dispatcher

This module provides a generic dispatch mechanism that maps function calls to their corresponding
implementations based on the currently selected backend.
"""

import functools
import os
from typing import Callable
from typing import Dict

from tilegym.logger import get_logger

from .selector import get_current_backend


def _is_fallback_disabled() -> bool:
    """Check if fallback is disabled via environment variable."""
    return os.environ.get("DISABLE_FALLBACK", "0") == "1"


logger = get_logger(__name__)

# Global registry with structure: {function_name: {backend_name: implementation}}
_REGISTRY: Dict[str, Dict[str, Callable]] = {}


def register_impl(name: str, backend: str):
    """
    Register a backend-specific implementation for a function

    Args:
        name: Function name
        backend: Backend name

    Returns:
        Decorator function
    """

    def decorator(func):
        if name not in _REGISTRY:
            _REGISTRY[name] = {}

        _REGISTRY[name][backend] = func
        logger.debug(f"[Backend Register] Registered '{backend}' implementation for '{name}'")
        return func

    return decorator


_LOGGED_WARNINGS = set()


def dispatch(name: str, fallback_backend: str = "pytorch"):
    """
    Create a dispatcher that selects the correct implementation based on current backend

    Args:
        name: Function name
        fallback_backend: Fallback backend to use if current backend has no implementation

    Returns:
        Decorator function
    """

    def decorator(default_impl):
        @functools.wraps(default_impl)
        def wrapper(*args, **kwargs):
            # Check if backend is explicitly specified in kwargs
            explicit_backend = kwargs.pop("backend", None)

            if explicit_backend is not None:
                current_backend = explicit_backend
            else:
                current_backend = get_current_backend()

            logger.debug(f"[Backend Dispatch] Function: '{name}', Current backend: '{current_backend}'")

            # Try implementation from current backend
            if name in _REGISTRY and current_backend in _REGISTRY[name]:
                logger.debug(f"[Backend Dispatch] Using '{current_backend}' implementation for '{name}'")
                return _REGISTRY[name][current_backend](*args, **kwargs)

            # Try implementation from fallback backend
            if name in _REGISTRY and fallback_backend in _REGISTRY[name]:
                # If DISABLE_FALLBACK=1, raise error instead of falling back
                if _is_fallback_disabled():
                    raise NotImplementedError(
                        f"Current backend '{current_backend}' has no implementation for '{name}'. "
                        f"Fallback to '{fallback_backend}' is disabled (DISABLE_FALLBACK=1)."
                    )
                warning_key = f"{name}_{current_backend}_{fallback_backend}"
                if warning_key not in _LOGGED_WARNINGS:
                    logger.warning(
                        f"Current backend '{current_backend}' has no implementation for '{name}', "
                        f"falling back to '{fallback_backend}' backend"
                    )
                    _LOGGED_WARNINGS.add(warning_key)
                logger.debug(f"[Backend Dispatch] Using fallback '{fallback_backend}' implementation for '{name}'")
                return _REGISTRY[name][fallback_backend](*args, **kwargs)

            # Use default implementation
            # If DISABLE_FALLBACK=1, raise error instead of using default
            if _is_fallback_disabled():
                raise NotImplementedError(
                    f"No backend implementation found for '{name}' with backend '{current_backend}'. "
                    f"Fallback to default implementation is disabled (DISABLE_FALLBACK=1)."
                )
            logger.warning(f"No backend implementation found for '{name}', using default implementation")
            logger.debug(f"[Backend Dispatch] Using default implementation for '{name}'")
            return default_impl(*args, **kwargs)

        # Register default implementation
        if name not in _REGISTRY:
            _REGISTRY[name] = {}

        _REGISTRY[name]["default"] = default_impl

        return wrapper

    return decorator


def get_available_backends_for_op(name: str) -> list:
    """
    Get list of all backends that support the specified operation

    Args:
        name: Operation name

    Returns:
        List of supported backends
    """
    if name not in _REGISTRY:
        return ["default"]

    return list(_REGISTRY[name].keys())


def get_registry_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about all registered implementations

    Returns:
        Dictionary with function names and their available backends
    """
    result = {}
    for func_name, backends in _REGISTRY.items():
        result[func_name] = {
            backend: (
                impl.__module__ + "." + impl.__name__
                if hasattr(impl, "__module__") and hasattr(impl, "__name__")
                else str(impl)
            )
            for backend, impl in backends.items()
        }
    return result


def print_registry_info():
    """
    Print detailed information about all registered implementations
    """
    print("\n=== Backend Registry Information ===")
    for func_name, backends in _REGISTRY.items():
        print(f"\n📋 Function: {func_name}")
        for backend, impl in backends.items():
            impl_info = (
                f"{impl.__module__}.{impl.__name__}"
                if hasattr(impl, "__module__") and hasattr(impl, "__name__")
                else str(impl)
            )
            print(f"  └─ {backend}: {impl_info}")
    print("=" * 40)
