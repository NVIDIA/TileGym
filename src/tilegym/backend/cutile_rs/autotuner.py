# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""
CUPTI-based autotuner for cutile-rs FFI kernels.

INSTALLATION (one-time, for a fresh ocean / tilegym checkout):
    1. Copy this file to:
         {TILEGYM_PATH}/src/tilegym/backend/cutile_rs/autotuner.py
    2. Ensure {TILEGYM_PATH}/src/tilegym/backend/cutile_rs/__init__.py
       exposes the public API (see installable/cutile_rs_backend_init.py).

WHY CUPTI (not torch.cuda.Event):
    cutile-rs JIT-compiles MLIR → cubin on first call per (kernel, generics)
    combo. That JIT can take 50-500 ms — torch.cuda.Event measures HOST-to-host
    time which includes JIT + ctypes FFI + Python marshalling, producing
    phantom 1.5–2.5x perf gaps that don't exist at the GPU kernel level.
    CUPTI (via torch.profiler) measures pure GPU kernel execution time, which
    is the ONLY apples-to-apples comparison vs cuTile-py / NVT.

    Verified empirically: layer_norm 2D showed 1.5x with cuda.Event but 0.96x
    with CUPTI (rs actually faster).

API SHAPE:
    result = autotune_launch(
        kernel_fn=lambda cfg: _run_ffi(... cfg.BM, cfg.BN ...),
        configs=[SimpleNamespace(BM=128, BN=128), SimpleNamespace(BM=256, BN=128)],
        key=(M, N, dtype),       # cache key — first call autotunes, rest are O(1)
        kernel_name="my_kernel", # for logging
    )
    return result.output

LAMBDA RULE (Rule 16-autotuner — non-negotiable):
    Inside `kernel_fn(cfg)` ALL output tensor allocations MUST use torch.empty.
    NEVER `.clone()`, `torch.zeros`, `torch.ones`, `.expand().contiguous()` —
    these launch GPU kernels (DtoD memcpy / fill) that CUPTI counts in the
    benchmark and create phantom gaps. layer_norm `.clone()` once added 4.8μs
    DtoD on a 7μs kernel → CUPTI reported 1.8x gap when kernel was at parity.
"""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Optional
from typing import Sequence

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile

# ─── Logger — fallback to stdlib if tilegym.logger is unavailable ──────────
#
# If you install this in tilegym, swap the next 3 lines for:
#     from tilegym.logger import get_logger
#     logger = get_logger(__name__)
try:
    from tilegym.logger import get_logger  # type: ignore

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# ─── Cache ──────────────────────────────────────────────────────────────────

_cache_lock = threading.RLock()
_cache: dict[tuple[str, Any], "_CacheEntry"] = {}


@dataclass
class TunedResult:
    """Result of a single `autotune_launch` call."""

    best_config: Any
    output: torch.Tensor
    tuning_record: Sequence[tuple[Any, float]]  # (config, median_ms)
    cache_hit: bool


@dataclass
class _CacheEntry:
    best_config: Any
    tuning_record: list[tuple[Any, float]] = field(default_factory=list)


# ─── CUPTI timing ───────────────────────────────────────────────────────────


def _bench_config_cupti(
    kernel_fn: Callable,
    cfg: Any,
    warmup: int = 2,
    rep: int = 10,
    kernel_filter: Optional[str] = None,
) -> tuple[Optional[torch.Tensor], float]:
    """Benchmark one config using CUPTI via torch.profiler.

    Returns (output_tensor, median_ms). On config-specific error returns
    (None, +inf). Re-raises environment / setup errors.
    """
    stream = torch.cuda.current_stream()
    kernel_re = re.compile(kernel_filter) if kernel_filter else None

    # Warmup — also triggers JIT compilation so it's excluded from timing.
    # Distinguish config-specific errors (illegal mem, OOM) from environment
    # errors (missing libcupti, FileNotFound, etc.) — the latter must propagate.
    try:
        for _ in range(warmup):
            out = kernel_fn(cfg)
        stream.synchronize()
    except (OSError, ImportError, FileNotFoundError, PermissionError):
        raise
    except RuntimeError as e:
        err_str = str(e)
        if "illegal" in err_str or "out of memory" in err_str or "error code" in err_str:
            logger.warning(f"  config {cfg}: warmup failed (config-specific): {e}")
            return None, float("inf")
        raise
    except Exception:
        raise

    stream.synchronize()
    times_us: list[float] = []
    for i in range(rep):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            out = kernel_fn(cfg)
            stream.synchronize()
        # ensure profiler fully flushes before next rep — cutile-rs DeviceContext
        # races with profiler if you don't sync between reps.
        stream.synchronize()

        total_us = sum(
            evt.self_device_time_total
            for evt in prof.key_averages()
            if evt.self_device_time_total > 0 and (kernel_re is None or kernel_re.search(evt.key))
        )

        if i == 0 and total_us == 0:
            raise RuntimeError(
                f"CUPTI returned 0 device time for config {cfg}. Check that libcupti is installed and accessible."
            )

        times_us.append(total_us)

    times_ms = sorted(t / 1000.0 for t in times_us)
    median_ms = times_ms[len(times_ms) // 2]
    return out, median_ms


# ─── Public API ─────────────────────────────────────────────────────────────


def autotune_launch(
    kernel_fn: Callable[[Any], torch.Tensor],
    configs: list[Any],
    key: Any,
    kernel_name: str = "unknown",
    warmup: int = 2,
    rep: int = 10,
    kernel_filter: Optional[str] = None,
) -> TunedResult:
    """Autotune a cutile-rs kernel using CUPTI timing.

    Args:
        kernel_fn      : Callable(cfg) -> output_tensor. MUST use torch.empty
                         for output allocation (Rule 16-autotuner). The lambda
                         is called once per config during autotune, then once
                         more with the winning config.
        configs        : List of config objects (typically `SimpleNamespace`s
                         with tile params: BM, BN, BK, LATENCY, etc.).
        key            : Hashable cache key, e.g. (M, N, K, dtype).
        kernel_name    : For logging and cache namespacing.
        warmup         : Warmup iterations per config (triggers JIT, excluded).
        rep            : CUPTI-timed iterations per config.
        kernel_filter  : Regex to select GPU kernel(s) for timing. If None,
                         sums all GPU kernels per invocation. Useful when a
                         wrapper calls auxiliary kernels you don't want timed.

    Returns:
        TunedResult with best_config, output, tuning_record, cache_hit.
    """
    if warmup < 0:
        raise ValueError(f"autotune_launch: warmup must be >= 0, got {warmup}")
    if rep < 1:
        raise ValueError(f"autotune_launch: rep must be >= 1, got {rep}")
    cache_key = (kernel_name, key)

    with _cache_lock:
        entry = _cache.get(cache_key)

    if entry is not None:
        out = kernel_fn(entry.best_config)
        return TunedResult(
            best_config=entry.best_config,
            output=out,
            tuning_record=entry.tuning_record,
            cache_hit=True,
        )

    # Cache miss — benchmark all configs with CUPTI.
    logger.info(f"cutile-rs autotune [{kernel_name}] key={key}: benchmarking {len(configs)} configs (CUPTI) ...")
    t0 = time.time()

    tuning_record: list[tuple[Any, float]] = []
    best_cfg = None
    best_ms = float("inf")
    best_out = None

    for i, cfg in enumerate(configs):
        out, ms = _bench_config_cupti(
            kernel_fn,
            cfg,
            warmup=warmup,
            rep=rep,
            kernel_filter=kernel_filter,
        )
        tuning_record.append((cfg, ms))
        if ms < best_ms:
            best_ms = ms
            best_cfg = cfg
            best_out = out
        logger.info(
            f"  [{kernel_name}] config {i + 1}/{len(configs)}: {cfg} → {ms:.4f} ms {'← best' if ms == best_ms else ''}"
        )

    if best_cfg is None:
        raise RuntimeError(f"cutile-rs autotune [{kernel_name}]: all {len(configs)} configs failed")

    elapsed = time.time() - t0
    logger.info(
        f"cutile-rs autotune [{kernel_name}] key={key}: best={best_cfg} ({best_ms:.4f} ms) total={elapsed:.1f}s"
    )

    with _cache_lock:
        _cache[cache_key] = _CacheEntry(
            best_config=best_cfg,
            tuning_record=tuning_record,
        )

    # Flush all CUPTI/profiler state before returning to caller — caller may
    # wrap us in another profiler, and nested profilers cause an abort.
    torch.cuda.synchronize()
    gc.collect()

    # Final run with the winning config (so the caller can use the output).
    final_out = kernel_fn(best_cfg)
    return TunedResult(
        best_config=best_cfg,
        output=final_out,
        tuning_record=tuning_record,
        cache_hit=False,
    )


def clear_cache(kernel_name: Optional[str] = None, key: Any = None) -> None:
    """Clear the per-process autotune cache.

    Useful in tests that want to force re-autotuning, or when you change the
    config search space and want to invalidate stale winners.
    """
    with _cache_lock:
        if kernel_name is None and key is None:
            _cache.clear()
            return
        to_delete = [k for k in _cache if (kernel_name is None or k[0] == kernel_name) and (key is None or k[1] == key)]
        for k in to_delete:
            del _cache[k]


def get_cache_stats() -> dict:
    """Return cache contents for debugging."""
    with _cache_lock:
        return {
            str(k): {
                "best_config": str(v.best_config),
                "best_ms": min(ms for _, ms in v.tuning_record) if v.tuning_record else None,
                "n_tested": len(v.tuning_record),
            }
            for k, v in _cache.items()
        }
