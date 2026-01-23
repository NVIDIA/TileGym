# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import os
import pathlib

import pytest
import torch


def pytest_configure(config):
    """Register custom markers"""
    if config.getoption("--run-full"):
        os.environ["RUN_FULL_TEST"] = "1"

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
