# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--transformer-submodule",
        action="store",
        default=None,
        help=(
            "Limit transformer kernel Definition/Solution runtime checks to one "
            "src/tilegym/transformers/<module> directory or module name."
        ),
    )


def pytest_generate_tests(metafunc):
    if {"definition_path", "solution_path"}.issubset(metafunc.fixturenames):
        from tests.transformers.kernel_runtime_utils import definition_solution_cases_for_submodule

        cases = definition_solution_cases_for_submodule(metafunc.config.getoption("--transformer-submodule"))
        metafunc.parametrize(("definition_path", "solution_path"), cases)
