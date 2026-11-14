# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import pathlib

import pytest

from tests.autotune_cache import isolated_cutile_tune_caches

_graph_cutile_autotune_caches = {}


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


def pytest_configure(config):
    """Register custom markers"""
    if config.getoption("--run-full"):
        os.environ["RUN_FULL_TEST"] = "1"

    config.addinivalue_line("markers", "interpreter: indicate whether interpreter supports the test")
    config.addinivalue_line("markers", "slow: indicate whether the test is in slow CI pipeline")
    config.addinivalue_line("markers", "serial: indicate whether the test is in single thread pipeline")
    config.addinivalue_line("markers", "fast: indicate whether the test is in fast CI pipeline")
    config.addinivalue_line("markers", "disable_cudagraph: time and autotune this test without CUDA Graph capture")


def _detect_arch_default() -> str:
    try:
        major, minor = torch.cuda.get_device_capability("cuda")
    except RuntimeError:
        return "sm100"
    return f"sm{major}{minor}"


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--arch",
            type=str,
            default=_detect_arch_default(),
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
        parser.addoption(
            "--first-config-only",
            action="store_true",
            default=False,
            help="Only run the first configuration of each parametrized test, once per backend (for CI speed)",
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


def _drop_inapplicable_cudagraph_markers(items):
    """disable_cudagraph(backends=[...]) applies only to items whose backend parameter is listed."""
    for item in items:
        marker = item.get_closest_marker("disable_cudagraph")
        backends = marker.kwargs.get("backends") if marker else None
        if backends is None:
            continue
        callspec = getattr(item, "callspec", None)
        if callspec is not None and callspec.params.get("backend") in backends:
            continue
        item.own_markers = [mark for mark in item.own_markers if mark.name != "disable_cudagraph"]


def pytest_collection_modifyitems(config, items):
    """Filter test items based on various parameters"""
    _drop_inapplicable_cudagraph_markers(items)

    def try_get_option(config, option_name):
        try:
            return config.getoption(option_name)
        except ValueError:
            return False

    # Apply first-config-only filter for ALL parametrized tests
    first_config_only = try_get_option(config, "--first-config-only")
    if first_config_only:
        filtered_items = []
        # Dict to track first candidate (possibly skipped) and whether we selected a runnable item
        first_candidate = {}
        selected = set()

        def _has_marker(it, name):
            return it.get_closest_marker(name) is not None or any(True for _ in it.iter_markers(name=name))

        def _group_key(it):
            # One configuration per test method *and* backend, so every backend a test
            # is parametrized over keeps a configuration, not just the first one.
            params = it.callspec.params if getattr(it, "callspec", None) is not None else {}
            # Both parameter names are in use across the suites.
            backend = params.get("backend") or params.get("framework")
            return it.nodeid.split("[")[0], None if backend is None else str(backend)

        for item in items:
            # For ALL parametrized tests, we want to group by test class + test method (+ backend)
            # e.g., test_rms_norm.py::Test_RMSNorm::test_op or test_softmax.py::Test_Softmax::test_perf
            # Non-parametrized tests (no "[" in nodeid) pass through directly
            if "[" not in item.nodeid:
                filtered_items.append(item)
                continue

            key = _group_key(item)

            skiplike = any(_has_marker(item, m) for m in ("skip", "skipif", "xfail"))

            # If we haven't recorded any candidate for this group, store the first one we see (could be skipped)
            if key not in first_candidate:
                first_candidate[key] = item

            # If this item is not skip, and we haven't selected for this group yet, choose it immediately
            if (not skiplike) and (key not in selected):
                selected.add(key)
                filtered_items.append(item)

        # For groups where we never found a runnable item, fallback to first candidate (even if skipped)
        for key, cand in first_candidate.items():
            if key not in selected:
                filtered_items.append(cand)

        items[:] = filtered_items


@pytest.fixture(autouse=True)
def _eager_timing_without_cuda_graph(request, monkeypatch):
    if request.node.get_closest_marker("disable_cudagraph"):
        from tests import common

        monkeypatch.setattr(common, "USE_CUDA_GRAPH", False)


@pytest.fixture(autouse=True)
def _graph_cutile_autotune_in_perf_tests(request, monkeypatch):
    if "test_perf" not in request.node.name or request.node.get_closest_marker("disable_cudagraph"):
        yield
        return
    try:
        from cuda.tile.tune import _tune
    except ImportError:
        yield
        return
    from tilegym.benchmark import autotune_cutile_cuda_graph

    with isolated_cutile_tune_caches(_graph_cutile_autotune_caches):
        monkeypatch.setattr(_tune, "_benchmark", autotune_cutile_cuda_graph)
        yield


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
