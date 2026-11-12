# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared launch tuning for FLA cuTile operators."""

import functools
import logging

import torch

from tilegym.autotune import is_autotune_disabled
from tilegym.suites.fla.cutile._host_utils import device_cache_key

logger = logging.getLogger(__name__)


def shape_bucket(n):
    """Power-of-two regime bucket for sequence/batch extents in tuning keys.

    Tile and launch-hint choices are valid for any extent, so equivalent
    workloads share one search instead of re-tuning every raw length.
    """
    n = max(int(n), 1)
    return 1 << (n - 1).bit_length()


def output_tile_configs(K, V):
    """Return distinct padded tiles with the baseline tile first."""
    if K < 1 or V < 1:
        raise ValueError("K and V must be positive")
    kp, vp = max(16, 1 << (K - 1).bit_length()), max(16, 1 << (V - 1).bit_length())
    candidates = [
        (min(kp, 64), min(vp, 64)),
        (min(kp, 128), min(vp, 128)),
        (min(kp, 64), min(vp, 128)),
        (min(kp, 128), min(vp, 64)),
    ]
    return list(dict.fromkeys(candidates))


def _ensure_cuda_context():
    """Attach the current device context on threads without a driver context."""
    try:
        from cuda.bindings import driver as _drv

        err, cur = _drv.cuCtxGetCurrent()
        if err == _drv.CUresult.CUDA_SUCCESS and int(cur) != 0:
            return
        dev = torch.cuda.current_device()
        err, ctx = _drv.cuDevicePrimaryCtxRetain(dev)
        if err == _drv.CUresult.CUDA_SUCCESS:
            _drv.cuCtxSetCurrent(ctx)
    except Exception:  # noqa: BLE001
        pass


@functools.cache
def _runtime():
    import cuda.tile as ct
    from cuda.tile.tune import exhaustive_search

    return ct, exhaustive_search


def _freeze(value):
    """Hashable identity for candidate fields (``hints`` dicts included)."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def launch_hints(*, occupancy=(1, 2, 4), num_worker_warps=(4, 8)):
    """Launch-hint candidates: every occupancy x num_worker_warps pair, as ``replace_hints`` kwargs.

    Occupancy-major order, so the first entry (lowest occupancy, fewest warps) is
    the one used when autotune is disabled or the search fails.
    """
    return tuple(dict(occupancy=o, num_worker_warps=w) for o in occupancy for w in num_worker_warps)


def prune_occupancy(configs, grid_fn, num_sms=None):
    """Drop candidates whose occupancy hint cannot take effect for their grid.

    An occupancy of ``o`` only lets an SM hold ``o`` CTAs at once if the grid has
    more than ``(o - 1) * num_sms`` CTAs; with fewer CTAs it can only lower the
    per-thread register budget, never raise concurrency. Candidates without a
    ``hints["occupancy"]`` (or with occupancy 1) are always kept.
    """
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    kept = []
    for cfg in configs:
        occupancy = getattr(cfg, "hints", {}).get("occupancy", 1)
        ctas = 1
        for extent in grid_fn(cfg):
            ctas *= extent
        if occupancy <= 1 or ctas > (occupancy - 1) * num_sms:
            kept.append(cfg)
    return kept


_search_cache = {}


def cached_exhaustive_search(search_space, stream, grid_fn, kernel, args_fn, hints_fn=None, *, cache_key, timeout=60):
    """``cuda.tile.tune.exhaustive_search`` with a per-device result cache.

    Same arguments as ``exhaustive_search`` plus ``cache_key``. Returns
    ``(config, kernel)``: the best candidate and ``kernel`` with its hints
    applied (``replace_hints(**hints_fn(config))``). The caller launches it with
    ``ct.launch(stream, grid_fn(config), kernel, args_fn(config))``.

    When autotune is disabled, or the search fails, returns the first candidate
    with the unmodified kernel; failures are not cached so a later call retries.
    """
    search_space = list(search_space)
    if not search_space:
        raise ValueError("At least one launch configuration is required")
    if is_autotune_disabled():
        return search_space[0], kernel
    space_id = tuple(_freeze(vars(c)) for c in search_space)
    key = (kernel, device_cache_key(), cache_key, space_id)
    entry = _search_cache.get(key)
    if entry is None:
        ct, search = _runtime()
        try:
            _ensure_cuda_context()
            with ct.compiler_timeout(timeout):
                result = search(search_space, stream, grid_fn, kernel, args_fn, hints_fn, quiet=True)
        except Exception:
            logger.warning("FLA launch tuning failed; using the first configuration for this call", exc_info=True)
            return search_space[0], kernel
        best = result.best.config
        entry = _search_cache[key] = (best, kernel.replace_hints(**hints_fn(best)) if hints_fn else kernel)
    return entry
