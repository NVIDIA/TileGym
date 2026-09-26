# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import sys
from contextlib import contextmanager


def cutile_caches():
    """Include tuning choices stored in launch, hint and tile metadata caches."""
    for name, module in tuple(sys.modules.items()):
        if module is None or not name.startswith("tilegym.") or "cutile" not in name.split("."):
            continue
        owners = [module]
        owners.extend(value for value in vars(module).values() if isinstance(value, type) and value.__module__ == name)
        for owner in owners:
            for attr, value in vars(owner).items():
                if "cache" in attr.lower() and isinstance(value, dict):
                    yield owner, attr, value


def empty_cache(cache):
    result = cache.copy()
    result.clear()
    return result


@contextmanager
def isolated_cutile_tune_caches(perf_caches):
    original = {}
    for owner, attr, cache in cutile_caches():
        key = (owner, attr)
        original[key] = cache
        if key not in perf_caches:
            perf_caches[key] = empty_cache(cache)
        setattr(owner, attr, perf_caches[key])
    try:
        yield
    finally:
        for owner, attr, cache in cutile_caches():
            key = (owner, attr)
            perf_caches[key] = cache
            # Lazily imported operators have no pre-existing native scores.
            setattr(owner, attr, original[key] if key in original else empty_cache(cache))
