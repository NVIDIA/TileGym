# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import functools
import inspect
import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Sequence

import cuda.tile as ct
import torch
from cuda.tile._cext import default_tile_context
from cuda.tile._exception import TileCompilerExecutionError
from cuda.tile._exception import TileCompilerTimeoutError
from cuda.tile._execution import TileDispatcher

logger = logging.getLogger(__name__)


class Config:
    """One kernel variant: meta-params in kwargs (e.g., TILE)."""

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
    def __init__(self, configs: list[Config], predicate_fn: Callable | None = None):
        if len(configs) < 1:
            raise ValueError(
                "At least one configurations in the search space are required for autotuning"
            )
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


def _shape_dtype_stride(arg: Any) -> tuple[tuple[int, ...], str, tuple[int, ...] | None]:
    shape = tuple(arg.shape)
    dtype = arg.dtype
    stride = None
    if hasattr(arg, "stride"):  # PyTorch, etc.
        s = arg.stride() if callable(arg.stride) else arg.stride
        stride = tuple(int(x) for x in s)
    elif hasattr(arg, "strides"):  # NumPy, etc. (bytes)
        itemsize = getattr(arg, "itemsize", 1)
        stride = tuple(int(b // itemsize) for b in arg.strides)

    return shape, dtype, stride


def _default_key(kernel: TileDispatcher, args: tuple[Any, ...]):
    """Default cache key for autotune.
    The key(for now) is a tuple of:
    - kernel function name
    - tuple of (shape, dtype, stride) for each argument in the runtime argument (tensor),
    - or its type name for each argument in the runtime argument (other types).
    """
    tinfo = []
    for arg in args:
        if hasattr(arg, "shape") and hasattr(arg, "dtype"):
            shape, dtype, stride = _shape_dtype_stride(arg)
            tinfo.append((shape, dtype, stride))
        else:
            tinfo.append(type(arg).__name__)
    return (kernel._pyfunc.__name__, tuple(tinfo))


def _time_ms(run_once, *, get_args, stream, warmup=2, rep=10):
    stream.synchronize()
    for _ in range(warmup):
        run_once(get_args())

    args_per_run = [get_args() for _ in range(rep)]
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for i in range(rep):
        run_once(args_per_run[i])
    end.record(stream)
    end.synchronize()

    ms = start.elapsed_time(end)
    return ms / max(1, rep)


@dataclass
class TunedResult:
    # The tuned parameters
    tuned_params: dict[str, Any]
    # The grid to be used for launching the kernel
    grid: tuple[int, ...]
    # The updated tile dispatcher to be used for launching the kernel
    kernel: TileDispatcher
    num_ctas: int
    occupancy: int
    opt_level: int

    def __getattr__(self, name):
        if name in self.tuned_params:
            return self.tuned_params[name]
        raise AttributeError(f"Attribute {name} not found in {self.tuned_params}")


def _make_trial_args(
    args_fn: Callable[[Config], tuple[Any, ...]],
    cfg: Config,
    kernel: TileDispatcher,
    transforms: dict[str, Callable[[Any], Any]],
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Make trial runtime arguments applying the transforms."""
    args = args_fn(cfg)

    trial_named_args = {}
    trial_args = []
    kernel_sig = inspect.signature(kernel._pyfunc)
    for kernel_key, arg in zip(kernel_sig.parameters.keys(), args, strict=True):
        if kernel_key in transforms:
            trial_named_args[kernel_key] = transforms[kernel_key](arg)
        else:
            trial_named_args[kernel_key] = arg
        trial_args.append(trial_named_args[kernel_key])
    return trial_named_args, tuple(trial_args)


def _normalize_search_space(space: SearchSpace | Sequence[Config]) -> SearchSpace:
    if isinstance(space, SearchSpace):
        return space

    # Allow sequence of Configs
    if isinstance(space, Sequence) and all(isinstance(c, Config) for c in space):
        return SearchSpace(list(space))

    raise TypeError("search_space must be a SearchSpace, or a sequence of Configs")


@contextmanager
def compiler_timeout(timeout_sec: int):
    old_timeout = default_tile_context.config.compiler_timeout_sec
    default_tile_context.config.compiler_timeout_sec = timeout_sec
    try:
        yield
    finally:
        default_tile_context.config.compiler_timeout_sec = old_timeout


class Autotuner:
    def __init__(self, search_space: SearchSpace | Sequence[Config]):
        self._search_space = _normalize_search_space(search_space)
        self._cache = {}

    def clear_cache(self, key=None):
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def __call__(
        self,
        stream,
        grid_fn,
        kernel,
        args_fn: Callable[[Config], tuple[Any, ...]],
        transforms: dict[str, Callable] = {},
        *,
        key_fn=_default_key,
        max_iter: int = 60,
        compiler_time_limit_sec: int = 10,
        seed: int | None = None,
        force_retune: bool = False,
    ) -> TunedResult:
        """
        Run the autotuned kernel and return its result.

        It performs the following steps:
        1) picks a configuration from the search space or reuses the cached
           best configuration for the given key (unless ``force_retune=True``),
        2) launches the kernel with the best configuration,
        3) returns the tuned result.

        Args:
            stream:
                CUDA stream to use for all kernel launches during tuning and
                for the final run.
            grid_fn:
                Callable that takes the named arguments and a single
                positional :class:`Config` object and returns a tuple of grid
                dimensions.
            kernel:
                The kernel to autotune.
            args_fn:
                Callable that takes a single positional :class:`Config` and
                returns a tuple of runtime arguments for ``kernel``.
            transforms:
                Optional transform or sequence of transforms applied to the
                runtime arguments before each kernel launch. Use this to
                perform lightweight pre-/post-processing without changing
                the search space.
            key_fn:
                Optional function that maps the named arguments to a hashable
                cache key. When omitted, a default key derived from argument
                shapes/dtypes is used. The key is used to look up and store
                the best config in the autotuner cache.
            max_iter:
                Maximum number of (valid) configurations to sample from the
                search space.
            compiler_time_limit_sec:
                The compilation time limit for each kernel.
            seed:
                Optional seed for the random number generator used when
                sampling configurations. If ``None``, the global random number
                generator state is used.
            force_retune:
                If ``True``, ignore any cached best config for this key and
                re-run the search. The new best config is then written back
                to the cache.
        """
        key = key_fn(kernel, args_fn(self._search_space.configs[0]))
        if not force_retune and key in self._cache:
            best_idx, best_grid, best_kernel = self._cache[key]
            logger.debug(f"Using cached config for key {key}: {self._search_space[best_idx]}")
        else:
            rng = random.Random(seed)
            indices = rng.sample(range(len(self._search_space)), len(self._search_space))
            best_time_ms, best_idx, best_kernel = float("inf"), None, None
            successes = 0
            for cfg_idx in indices:
                if successes >= max_iter:
                    break
                cfg = self._search_space[cfg_idx]
                trial_named_args, trial_args = _make_trial_args(args_fn, cfg, kernel, transforms)
                if not self._search_space.filter(trial_named_args, cfg):
                    logger.debug(f"Config {cfg} filtered out by predicate function")
                    continue

                grid = grid_fn(trial_named_args, cfg)
                updated_kernel = ct.kernel(
                    kernel._pyfunc,
                    num_ctas=cfg.num_ctas,
                    occupancy=cfg.occupancy,
                    opt_level=cfg.opt_level,
                )

                def run_once(args):
                    ct.launch(stream, grid, updated_kernel, args)

                try:
                    with compiler_timeout(compiler_time_limit_sec):
                        time_ms = _time_ms(
                            run_once,
                            get_args=lambda: _make_trial_args(args_fn, cfg, kernel, transforms)[
                                1
                            ],  # noqa
                            stream=stream,
                        )
                except TileCompilerTimeoutError as e:
                    logger.debug(f"{cfg} compilation timeout: {e}")
                    continue
                except TileCompilerExecutionError as e:
                    logger.debug(f"{cfg} compilation error: {e}")
                    continue

                if time_ms < best_time_ms:
                    best_time_ms = time_ms
                    best_idx, best_grid, best_kernel = cfg_idx, grid, updated_kernel
                    logger.debug(
                        f"Iteration {successes} updated best config to {cfg}: {best_time_ms} ms"
                    )
                successes += 1

            # Save the best config and kernel.
            if best_idx is None:
                raise ValueError("No valid config found")
            self._cache[key] = (best_idx, best_grid, best_kernel)

        best_cfg = self._search_space[best_idx]

        # Use the original runtime arguments to run the kernel with the best config
        best_packed_args = args_fn(best_cfg)
        ct.launch(stream, best_grid, best_kernel, best_packed_args)

        # Return the tuned result
        return TunedResult(
            best_cfg.kwargs,
            best_grid,
            best_kernel,
            num_ctas=best_cfg.num_ctas,
            occupancy=best_cfg.occupancy,
            opt_level=best_cfg.opt_level,
        )


def autotune(search_space):
    def decorator(func):
        tuner = Autotuner(search_space)  # single, device-agnostic instance

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Inject the tuner into the function arguments
            kwargs.setdefault("autotuner", tuner)
            return func(*args, **kwargs)

        return wrapper

    return decorator
