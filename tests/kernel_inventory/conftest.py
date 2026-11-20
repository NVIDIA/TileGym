# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import sys
import types
from pathlib import Path

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    # Collect these tests under a placeholder tilegym package so their kernel_inventory imports
    # skip src/tilegym/__init__.py, which loads the op backends. Tests import the real tilegym at run time.
    if "tilegym" in sys.modules:
        return (yield)
    placeholder = types.ModuleType("tilegym")
    placeholder.__path__ = [str(Path(__file__).resolve().parents[2] / "src/tilegym")]
    sys.modules["tilegym"] = placeholder
    try:
        return (yield)
    finally:
        if sys.modules.get("tilegym") is placeholder:
            del sys.modules["tilegym"]


def pytest_addoption(parser):
    parser.addoption(
        "--kernel-submodule",
        action="append",
        default=None,
        help=(
            "Limit Workload/Definition/Solution runtime checks to directories "
            "under src/tilegym that contain kernel_workloads, kernel_definitions, and kernel_solutions. "
            "Repeat the option or pass a comma-separated list to select several submodules."
        ),
    )


def _requested_submodules(config) -> list[str]:
    values = config.getoption("--kernel-submodule") or []
    submodules = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in submodules:
                submodules.append(item)
    return submodules


def pytest_generate_tests(metafunc):
    if {"workload_record", "definition_path", "solution_path"}.issubset(metafunc.fixturenames):
        from tests.kernel_inventory.kernel_runtime_utils import _kernel_submodule_root
        from tests.kernel_inventory.kernel_runtime_utils import workload_runtime_cases_for_submodule

        submodules = _requested_submodules(metafunc.config)
        if not submodules:
            cases = workload_runtime_cases_for_submodule(None)
        else:
            # Different spellings (``fla`` vs ``src/tilegym/suites/fla``) can name
            # the same directory; select each directory once.
            roots = []
            for submodule in submodules:
                root = _kernel_submodule_root(submodule).resolve()
                if root not in roots:
                    roots.append(root)
            cases = []
            for root in roots:
                cases.extend(workload_runtime_cases_for_submodule(root))
        metafunc.parametrize(("workload_record", "definition_path", "solution_path"), cases)
