# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from collections import OrderedDict
from dataclasses import dataclass

import cuda.tile as ct


@dataclass(frozen=True)
class _HintedKernelEntry:
    hinted: ct.kernel
    owner: ct.kernel  # keeps source kernel alive so id() can't be recycled


_HINTED_KERNEL_CACHE: "OrderedDict[tuple, _HintedKernelEntry]" = OrderedDict()
_HINTED_KERNEL_CACHE_MAX = 256


def cached_replace_hints(kernel: ct.kernel, **hints) -> ct.kernel:
    # cuTile's JIT cache lives on the hinted-kernel object; reuse it
    # across launches so per-shape cached compiles survive.
    key = (id(kernel), tuple(sorted(hints.items())))
    entry = _HINTED_KERNEL_CACHE.get(key)
    if entry is not None:
        _HINTED_KERNEL_CACHE.move_to_end(key)
        return entry.hinted
    entry = _HintedKernelEntry(hinted=kernel.replace_hints(**hints), owner=kernel)
    _HINTED_KERNEL_CACHE[key] = entry
    if len(_HINTED_KERNEL_CACHE) > _HINTED_KERNEL_CACHE_MAX:
        _HINTED_KERNEL_CACHE.popitem(last=False)
    return entry.hinted


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


def is_power_of_2(n: int):
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0
