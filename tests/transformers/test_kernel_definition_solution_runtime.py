# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest

from tests.transformers.kernel_runtime_utils import _skip_if_solution_does_not_target_current_hardware
from tests.transformers.kernel_runtime_utils import run_definition_solution_runtime


def test_kernel_definition_solution_runtime(definition_path, solution_path):
    run_definition_solution_runtime(definition_path, solution_path)


class _FakeCuda:
    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "NVIDIA GB300"


class _FakeTorch:
    cuda = _FakeCuda()


def test_runtime_skips_solutions_that_do_not_target_current_hardware():
    solution = {
        "spec": {
            "target_hardware": ["NVIDIA_B200"],
        }
    }
    with pytest.raises(pytest.skip.Exception, match="current hardware is NVIDIA_GB300"):
        _skip_if_solution_does_not_target_current_hardware(solution, _FakeTorch)


def test_runtime_accepts_solutions_that_target_current_hardware():
    solution = {
        "spec": {
            "target_hardware": ["NVIDIA_B200", "NVIDIA_GB300"],
        }
    }
    _skip_if_solution_does_not_target_current_hardware(solution, _FakeTorch)
