# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unified Backend Selector
Used to manage backend implementations of various operations in TileGym library
"""

import os
from typing import Dict, Set
from tilegym.logger import get_logger

logger = get_logger(__name__)

try:
    import cuda.tile as ct

    CUTILE_AVAILABLE = True

except ImportError:
    import warnings

    warnings.warn("Failed to import cuda_tile_compiler, CUDA Tile backend is not available")
    CUTILE_AVAILABLE = False


def is_cutile_available():
    return CUTILE_AVAILABLE


_AVAILABLE_BACKENDS: Set[str] = set()
_CURRENT_BACKENDS: str = "cutile"


def _check_backends_availability() -> Dict[str, bool]:
    availability = {
        "cutile": is_cutile_available(),
    }
    return availability


def _initialize_available_backends():
    global _AVAILABLE_BACKENDS
    global _CURRENT_BACKENDS
    backend_availability = _check_backends_availability()

    for backend, available in backend_availability.items():
        if available:
            _AVAILABLE_BACKENDS.add(backend)


def _load_from_environment():
    """CUTILE_TUTORIALS_BACKEND=xxx"""
    global _CURRENT_BACKENDS
    backend = os.environ.get("CUTILE_TUTORIALS_BACKEND", _CURRENT_BACKENDS)
    if backend in _AVAILABLE_BACKENDS:
        _CURRENT_BACKENDS = backend
    else:
        raise ValueError(f"Unknown backend: {backend}, available backends: {_AVAILABLE_BACKENDS}")


def get_available_backends() -> Set[str]:
    return _AVAILABLE_BACKENDS


def get_current_backend() -> str:
    return _CURRENT_BACKENDS


def set_backend(backend: str) -> None:
    '''set the backend for ops'''
    global _CURRENT_BACKENDS
    if backend not in _AVAILABLE_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}, available backends: {_AVAILABLE_BACKENDS}")
    _CURRENT_BACKENDS = backend
    logger.info(f"Set backend to {backend}")


def is_backend_available(backend: str) -> bool:
    '''check if the backend is available'''
    return backend in _AVAILABLE_BACKENDS


def assert_backend_available(backend: str) -> None:
    '''assert the backend is available'''
    if not is_backend_available(backend):
        raise ValueError(
            f"Backend {backend} is not available, available backends: {_AVAILABLE_BACKENDS}"
        )


_initialize_available_backends()
_load_from_environment()
