# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest


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
        from tests.kernel_inventory.kernel_runtime_utils import workload_runtime_cases_for_submodule

        submodules = _requested_submodules(metafunc.config)
        if not submodules:
            cases = workload_runtime_cases_for_submodule(None)
        else:
            cases = []
            for submodule in submodules:
                cases.extend(workload_runtime_cases_for_submodule(submodule))
        metafunc.parametrize(("workload_record", "definition_path", "solution_path"), cases)
