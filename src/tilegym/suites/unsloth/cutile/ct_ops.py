# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CuTile helper ops and shared utilities for unsloth kernels.

Includes:
  - Host-side utilities: next_power_of_2, cdiv, calculate_settings, autotune_configs
  - Device-side helpers (for @ct.kernel): erf_ct
"""

import math
from types import SimpleNamespace

import cuda.tile as ct

# ---- Host-side utilities ----

MAX_FUSED_SIZE = 65536  # 2**16


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def cdiv(a, b):
    """Ceiling division."""
    return math.ceil(a / b)


def calculate_settings(n):
    """Calculate BLOCK_SIZE (next power of 2) for a given dimension n.

    Raises RuntimeError if n exceeds MAX_FUSED_SIZE.
    """
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch kernel since n = {n} exceeds the maximum blocksize = {MAX_FUSED_SIZE}.")
    return BLOCK_SIZE


def autotune_configs():
    """Yield standard occupancy configs for autotune_launch."""
    for occ in [1, 2, 4, 8]:
        yield SimpleNamespace(occupancy=occ)


# ---- Device-side helpers (for use inside @ct.kernel) ----


def erf_ct(x):
    """Element-wise erf(x) via Abramowitz & Stegun polynomial approximation.

    Maximum error: |ε(x)| ≤ 1.5 × 10⁻⁷
    Reference: Abramowitz & Stegun, formula 7.1.26
    Source: src/tilegym/suites/unsloth/cutile/geglu.py

    Args:
        x: CuTile float32 tensor of any shape.

    Returns:
        Approximation of erf(x) with same shape as x.
    """
    abs_x = ct.maximum(x, ct.negative(x))
    t_denom = 1.0 + 0.3275911 * abs_x
    # 1/t via rsqrt: rsqrt(t_denom^2) = 1/|t_denom|; t_denom > 0 always
    t = ct.rsqrt(t_denom * t_denom)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t
    poly = 0.254829592 * t + (-0.284496736) * t2 + 1.421413741 * t3 + (-1.453152027) * t4 + 1.061405429 * t5
    erf_abs = 1.0 - poly * ct.exp(ct.negative(abs_x * abs_x))
    # erf is odd function
    return ct.where(x < 0.0, ct.negative(erf_abs), erf_abs)
