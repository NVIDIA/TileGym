# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Stable output-comparison boundary for kernel-inventory runtime checks."""

from __future__ import annotations

from tilegym.kernel_inventory._correctness_compat import Correctness
from tilegym.kernel_inventory._correctness_compat import compare_outputs
from tilegym.kernel_inventory._correctness_compat import poison_output_

__all__ = [
    "Correctness",
    "compare_outputs",
    "poison_output_",
]
