# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""cutile-rs backend: CUPTI-based autotuner + FFI library loader.

Backend infrastructure (the autotuner and the per-kernel cdylib loader) lives
here in ``backend/cutile_rs/``; the op wrappers and kernel crates live in
``ops/cutile_rs/``. This package re-exports the autotuner entry points.
"""

from .autotuner import TunedResult
from .autotuner import autotune_launch
from .autotuner import clear_cache
from .autotuner import get_cache_stats

__all__ = ["autotune_launch", "clear_cache", "get_cache_stats", "TunedResult"]
