# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Shared launch tuning for FLA cuTile operators."""

import functools
import logging
from types import SimpleNamespace

import torch

from tilegym.autotune import is_autotune_disabled
from tilegym.suites.fla.cutile._host_utils import device_cache_key

STATE_HINT_PAIRS = ((1, 4), (2, 4), (4, 4), (1, 8), (2, 8))
OUTPUT_HINT_PAIRS = ((4, 4), (2, 4), (1, 4), (4, 8), (1, 8))

logger = logging.getLogger(__name__)
_cache = {}


def hint_configs(occupancies, warps=(4, 8)):
    return [
        SimpleNamespace(occupancy=o, num_worker_warps=w)
        for o in dict.fromkeys(occupancies)
        for w in dict.fromkeys(warps)
    ]


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


def tune_launch(kernel, stream, grid_fn, args_fn, configs, cache_key, *, timeout=60, config_key=None):
    """Cache successful searches; build lazy candidates only on a miss.

    Lazy candidates require a complete immutable config_key. Eager candidates
    derive their identity from candidate values. Failures use the first
    candidate's grid/arguments without caching a fallback.
    """
    lazy = callable(configs)
    if lazy and config_key is None:
        raise ValueError("Lazy launch configurations require config_key")
    if not lazy and not configs:
        raise ValueError("At least one launch configuration is required")
    ct, search = _runtime()
    disabled = is_autotune_disabled()
    if not disabled:
        if config_key is None:
            config_key = tuple(tuple(sorted(vars(c).items())) for c in configs)
        key = (kernel, device_cache_key(), cache_key, config_key)
        chosen = _cache.get(key)
        if chosen is not None:
            tuned, cfg = chosen
            ct.launch(stream, grid_fn(cfg), tuned, args_fn(cfg))
            return
    candidates = configs() if lazy else configs
    if not candidates:
        raise ValueError("At least one launch configuration is required")
    if disabled:
        ct.launch(stream, grid_fn(candidates[0]), kernel, args_fn(candidates[0]))
        return
    try:
        _ensure_cuda_context()
        with ct.compiler_timeout(timeout):
            result = search(
                candidates,
                stream,
                grid_fn,
                kernel,
                args_fn,
                lambda c: {"occupancy": c.occupancy, "num_worker_warps": c.num_worker_warps},
            )
        best = result.best.config
        tuned = kernel.replace_hints(occupancy=best.occupancy, num_worker_warps=best.num_worker_warps)
    except Exception:
        logger.warning("FLA launch tuning failed; using the base kernel for this call", exc_info=True)
        ct.launch(stream, grid_fn(candidates[0]), kernel, args_fn(candidates[0]))
        return
    _cache[key] = (tuned, best)
    ct.launch(stream, grid_fn(best), tuned, args_fn(best))
