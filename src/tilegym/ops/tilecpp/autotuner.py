# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CUDA Tile C++ Autotuner

Provides autotuning infrastructure for CUDA Tile C++ kernels.

The autotuner is controlled by the TILECPP_AUTOTUNE environment variable:
- TILECPP_AUTOTUNE=0 (default): Use default configurations
- TILECPP_AUTOTUNE=1: Enable autotuning to find optimal configurations
"""

from __future__ import annotations

import functools
import logging
import os
import random
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Sequence

import torch

logger = logging.getLogger(__name__)


class Config:
    """One kernel variant: meta-params in kwargs (e.g., TILE_SIZE_M)."""

    def __init__(self, *, num_ctas=None, occupancy=None, opt_level=3, **kwargs):
        self.kwargs = dict(kwargs)
        self.num_ctas = num_ctas
        self.occupancy = occupancy
        self.opt_level = opt_level

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(f"Attribute {name} not found in {self.kwargs}")

    def __str__(self):
        res = []
        for k, v in self.kwargs.items():
            res.append(f"{k}={v}")
        res.append(f"num_ctas={self.num_ctas}")
        res.append(f"occupancy={self.occupancy}")
        res.append(f"opt_level={self.opt_level}")
        return f"Config({', '.join(res)})"


class SearchSpace:
    """Collection of configurations with optional predicate filtering."""

    def __init__(self, configs: list[Config], predicate_fn: Callable | None = None):
        if len(configs) < 1:
            raise ValueError("At least one configuration is required for autotuning")
        self.kwargs_keys = set(configs[0].kwargs.keys())
        for config in configs[1:]:
            if set(config.kwargs.keys()) != self.kwargs_keys:
                raise ValueError("All configurations must have the same set of keyword arguments")
        self.configs = configs
        self.predicate_fn = predicate_fn

    def __iter__(self):
        return iter(self.configs)

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, index):
        return self.configs[index]

    def filter(self, named_args: dict[str, Any], cfg: Config) -> bool:
        if self.predicate_fn is None:
            return True
        result = self.predicate_fn(named_args, cfg)
        if not isinstance(result, bool):
            raise TypeError(
                f"Predicate function {self.predicate_fn.__name__} must return "
                f"a boolean value, but returned {type(result).__name__} instead."
            )
        return result


def is_verbose_autotune() -> bool:
    """Check if verbose autotuning output is enabled via TILECPP_VERBOSE_AUTOTUNE."""
    return os.environ.get("TILECPP_VERBOSE_AUTOTUNE", "0") != "0"


def _normalize_search_space(space: SearchSpace | Sequence[Config]) -> SearchSpace:
    """Convert a sequence of Configs to a SearchSpace if needed."""
    if isinstance(space, SearchSpace):
        return space
    if isinstance(space, Sequence) and all(isinstance(c, Config) for c in space):
        return SearchSpace(list(space))
    raise TypeError("search_space must be a SearchSpace, or a sequence of Configs")


@dataclass
class TunedResult:
    """Result of autotuning containing the best configuration."""

    tuned_params: dict[str, Any]
    grid: tuple[int, ...]
    num_ctas: int
    occupancy: int
    opt_level: int

    def __getattr__(self, name):
        if name in self.tuned_params:
            return self.tuned_params[name]
        raise AttributeError(f"Attribute {name} not found in {self.tuned_params}")


def _time_ms(
    run_once: Callable,
    *,
    stream,
    warmup_ms: float = 25.0,
    rep_ms: float = 100.0,
) -> float:
    """Measure execution time in milliseconds using per-invocation CUDA events.

    1. Pilot run to estimate per-call cost.
    2. Derive warmup/repeat counts from time budgets.
    3. Per-invocation event pairs so each run is timed independently.
    4. Returns the **median** of a 10%-trimmed distribution for stability.
    """
    stream.synchronize()

    # Pilot: estimate per-call cost
    run_once()
    stream.synchronize()

    pilot_start = torch.cuda.Event(enable_timing=True)
    pilot_end = torch.cuda.Event(enable_timing=True)
    pilot_start.record(stream)
    for _ in range(5):
        run_once()
    pilot_end.record(stream)
    pilot_end.synchronize()
    estimate_ms = pilot_start.elapsed_time(pilot_end) / 5

    n_warmup = max(1, int(warmup_ms / max(estimate_ms, 1e-3)))
    n_repeat = max(10, int(rep_ms / max(estimate_ms, 1e-3)))

    # Warmup — stabilises GPU clocks, caches, and TLBs
    for _ in range(n_warmup):
        run_once()
    stream.synchronize()

    # Benchmark with per-invocation events
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        starts[i].record(stream)
        run_once()
        ends[i].record(stream)
    ends[-1].synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))

    # Trim fastest and slowest 10%, take median of the rest
    lo = len(times) // 10
    hi = len(times) - lo
    trimmed = times[lo:hi] if hi > lo else times
    return trimmed[len(trimmed) // 2]


def _default_key(
    kernel_name: str,
    dtype: torch.dtype,
    M: int,
    N: int,
    K: int,
    transpose_a: bool,
    transpose_b: bool,
) -> tuple:
    """Generate a cache key based on problem configuration."""
    return (kernel_name, str(dtype), M, N, K, transpose_a, transpose_b)


class TileCppAutotuner:
    """
    Autotuner for CUDA Tile C++ kernels.

    The CUDA Tile C++ autotuner leverages the TileCppKernel caching system - each unique
    set of template parameters produces a cached compiled kernel.

    The autotuner searches over block size configurations and measures performance
    to find the optimal configuration for each problem size.
    """

    def __init__(self, search_space: SearchSpace | Sequence[Config]):
        self._search_space = _normalize_search_space(search_space)
        self._cache: dict[tuple, tuple[int, tuple[int, ...]]] = {}

    def clear_cache(self, key=None):
        """Clear the autotuning cache."""
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __call__(
        self,
        stream,
        key: tuple,
        launch_fn: Callable[[Config], None],
        grid_fn: Callable[[dict[str, Any], Config], tuple[int, ...]],
        named_args: dict[str, Any] = {},
        *,
        max_iter: int = 60,
        seed: int | None = None,
        force_retune: bool = False,
    ) -> TunedResult:
        """
        Run the autotuned kernel and return its result.

        Args:
            stream:
                CUDA stream to use for all kernel launches during tuning and
                for the final run.
            key:
                Cache key for this problem configuration.
            launch_fn:
                Callable that takes a single positional :class:`Config` and
                launches the kernel.
            grid_fn:
                Callable that takes the named arguments dict and a single
                positional :class:`Config` object and returns a tuple of grid
                dimensions.
            named_args:
                Named arguments dict to pass to grid_fn and for filter predicate.
            max_iter:
                Maximum number of (valid) configurations to sample from the
                search space.
            seed:
                Optional seed for the random number generator used when
                sampling configurations. If ``None``, the global random number
                generator state is used.
            force_retune:
                If ``True``, ignore any cached best config for this key and
                re-run the search. The new best config is then written back
                to the cache.

        Returns:
            TunedResult with the best configuration.
        """
        verbose = is_verbose_autotune()

        if not force_retune and key in self._cache:
            best_idx, best_grid = self._cache[key]
            best_cfg = self._search_space[best_idx]
            if verbose:
                logger.info(f"[TileCpp Autotuner] Cache hit for {key}: {best_cfg}")
        else:
            if verbose:
                logger.info(f"[TileCpp Autotuner] Starting autotuning for {key} with {len(self._search_space)} configs")
            rng = random.Random(seed)
            indices = rng.sample(range(len(self._search_space)), len(self._search_space))

            # Phase 1: Pre-compile all configurations to warm up the compile cache
            # This ensures JIT compilation overhead doesn't affect timing
            if verbose:
                logger.info(f"[TileCpp Autotuner] Pre-compiling up to {max_iter} configurations...")
            valid_configs = []
            successes = 0
            for cfg_idx in indices:
                if successes >= max_iter:
                    break
                cfg = self._search_space[cfg_idx]

                # Apply filter predicate if defined
                if not self._search_space.filter(named_args, cfg):
                    if verbose:
                        logger.debug(f"[TileCpp Autotuner] Config {cfg} filtered out by predicate function")
                    continue

                grid = grid_fn(named_args, cfg)
                try:
                    # Run once to trigger JIT compilation
                    launch_fn(cfg)
                    valid_configs.append((cfg_idx, cfg, grid))
                    successes += 1
                except Exception as e:
                    if verbose:
                        logger.info(f"[TileCpp Autotuner] Config {cfg} failed during pre-compile: {e}")
                    continue

            if not valid_configs:
                raise ValueError("No valid config found")

            # Synchronize to ensure all compilations are complete
            stream.synchronize()
            if verbose:
                logger.info(f"[TileCpp Autotuner] Pre-compilation done. Timing {len(valid_configs)} valid configs...")

            # Phase 2: Time each pre-compiled configuration
            best_time_ms, best_idx, best_grid = float("inf"), None, None

            for cfg_idx, cfg, grid in valid_configs:
                try:

                    def run_once(c=cfg):  # Capture cfg in closure
                        launch_fn(c)

                    time_ms = _time_ms(run_once, stream=stream)

                    if time_ms < best_time_ms:
                        best_time_ms = time_ms
                        best_idx, best_grid = cfg_idx, grid
                        if verbose:
                            logger.info(f"[TileCpp Autotuner] New best: {cfg} -> {best_time_ms:.3f} ms")
                    else:
                        if verbose:
                            logger.info(f"[TileCpp Autotuner] Tried: {cfg} -> {time_ms:.3f} ms")

                except Exception as e:
                    if verbose:
                        logger.info(f"[TileCpp Autotuner] Config {cfg} failed during timing: {e}")
                    continue

            if best_idx is None:
                raise ValueError("No valid config found after timing")

            best_cfg = self._search_space[best_idx]
            if verbose:
                logger.info(f"[TileCpp Autotuner] Tuning complete. Best: {best_cfg} -> {best_time_ms:.3f} ms")
            self._cache[key] = (best_idx, best_grid)

        best_cfg = self._search_space[best_idx]

        # Launch with the best configuration
        launch_fn(best_cfg)

        return TunedResult(
            best_cfg.kwargs,
            best_grid,
            num_ctas=best_cfg.num_ctas,
            occupancy=best_cfg.occupancy,
            opt_level=best_cfg.opt_level,
        )


def autotune(search_space):
    """
    Decorator to add autotuning capability to a CUDA Tile C++ kernel wrapper.

    Usage:
        @autotune(search_space=_matmul_autotune_configs())
        def tilecpp_autotune_matmul(a, b, c, autotuner: TileCppAutotuner | None = None):
            ...
    """

    def decorator(func):
        tuner = TileCppAutotuner(search_space)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.setdefault("autotuner", tuner)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def is_autotuning_enabled() -> bool:
    """Check if autotuning is enabled via TILECPP_AUTOTUNE environment variable."""
    return os.environ.get("TILECPP_AUTOTUNE", "0") != "0"
