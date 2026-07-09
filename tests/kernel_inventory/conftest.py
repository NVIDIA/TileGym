# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--kernel-submodule",
        action="store",
        default=None,
        help=(
            "Limit kernel Definition/Solution runtime checks to one directory "
            "under src/tilegym that contains kernel_definitions and kernel_solutions."
        ),
    )


def pytest_generate_tests(metafunc):
    if {"definition_path", "solution_path"}.issubset(metafunc.fixturenames):
        from tests.kernel_inventory.kernel_runtime_utils import definition_solution_cases_for_submodule

        cases = definition_solution_cases_for_submodule(metafunc.config.getoption("--kernel-submodule"))
        metafunc.parametrize(("definition_path", "solution_path"), cases)
