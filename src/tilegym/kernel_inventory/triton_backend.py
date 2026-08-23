# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Lightweight detection of TileGym's active Triton compiler backend."""

from __future__ import annotations

import os


def is_triton_tileir_available() -> bool:
    """Match TileGym selector semantics without importing the backend package."""
    try:
        import triton.backends.tileir

        tileir_exists = True
    except ImportError:
        tileir_exists = False
    return tileir_exists and int(os.environ.get("ENABLE_TILE", -1)) == 1


def get_available_triton_backend() -> str:
    """Return TileGym's canonical active Triton compiler label."""
    return "nvt" if is_triton_tileir_available() else "oait"
