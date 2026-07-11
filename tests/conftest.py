# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import pathlib

import pytest

try:
    import torch
except ImportError:
    raise ImportError(
        "\n\n[TileGym] PyTorch is required to run this library.\n"
        "Since CUDA versions vary significantly across devices, please "
        "manually install the version that matches your hardware:\n"
        "👉 https://pytorch.org/get-started/locally/\n"
        "Alternatively, try: pip install tilegym[torch]\n"
    ) from None


def _apply_cutile_launch_timeout_override():
    """Raise cuda.tile's autotune launch-timeout budget when requested via env.

    cuda-tile's exhaustive_search() benchmarks each candidate config in a
    watchdog subprocess whose wall-clock budget starts at
    _MAX_DYNAMIC_LAUNCH_TIMEOUT_SEC (hardcoded 5.0s upstream, no env knob).
    On heavily loaded CI nodes (many pytest-xdist workers sharing one GPU),
    subprocess CUDA-context init + JIT compile + launch can exceed 5s for
    perfectly valid configs, so every config dies with TileLaunchTimeoutError
    and exhaustive_search raises "No valid config found in search space".

    CUTILE_MAX_LAUNCH_TIMEOUT_SEC=<float> patches the module constant before
    any autotune runs. Runs in every xdist worker (conftest is imported per
    worker). No-op when the env var is unset or cuda.tile is unavailable.
    """
    timeout_override = os.getenv("CUTILE_MAX_LAUNCH_TIMEOUT_SEC")
    if not timeout_override:
        return
    try:
        from cuda.tile.tune import _tune as _cutile_tune
    except ImportError:
        return
    if hasattr(_cutile_tune, "_MAX_DYNAMIC_LAUNCH_TIMEOUT_SEC"):
        _cutile_tune._MAX_DYNAMIC_LAUNCH_TIMEOUT_SEC = float(timeout_override)


def pytest_configure(config):
    """Register custom markers"""
    if config.getoption("--run-full"):
        os.environ["RUN_FULL_TEST"] = "1"

    _apply_cutile_launch_timeout_override()

    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "slow: indicate whether the test is in slow CI pipeline")
    config.addinivalue_line("markers", "serial: indicate whether the test is in single thread pipeline")
    config.addinivalue_line("markers", "fast: indicate whether the test is in fast CI pipeline")


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--arch",
            type=str,
            default=f"sm{torch.cuda.get_device_capability('cuda')[0]}{torch.cuda.get_device_capability('cuda')[1]}",
            help="GPU Backend Type",
        )
        parser.addoption("--quick-run", action="store_true", default=False, help="Quick Run")
        parser.addoption(
            "--print-record",
            action="store_true",
            default=False,
            help="Print record_property content in tests",
        )
        parser.addoption(
            "--run-full",
            action="store_true",
            default=False,
            help="Run all tests",
        )
    except ValueError:
        # if it is added by parent directory, skip it
        pass


@pytest.fixture
def arch(request):
    return request.config.getoption("--arch")


@pytest.fixture
def quick_run(request):
    return request.config.getoption("--quick-run")


@pytest.fixture
def framework(request):
    return request.config.getoption("--framework")


def _has_object_repr(val):
    """Helper function to recursively check if any value in nested structure has object representation"""
    if isinstance(val, (int, float, str, bool, type(None))):
        return False

    if isinstance(val, (list, tuple, set)):
        return any(_has_object_repr(item) for item in val)

    if isinstance(val, dict):
        return any(_has_object_repr(v) for v in val.values())

    val_repr = repr(val)
    return val_repr.startswith("<") and " at 0x" in val_repr


def pytest_make_parametrize_id(config, val, argname):
    # Replace "-" with "_" to avoid conflicts with pytest's automatic parameter naming
    if isinstance(val, (int, float, str, bool, type(None))):
        return str(val).replace("-", "_")

    if _has_object_repr(val):
        return None

    return repr(val).replace("-", "_")
