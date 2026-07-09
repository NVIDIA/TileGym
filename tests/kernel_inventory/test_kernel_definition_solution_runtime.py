# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools
import sys
import types

import pytest

from tests.kernel_inventory.kernel_runtime_utils import _as_outputs
from tests.kernel_inventory.kernel_runtime_utils import _assert_matching_entry_signatures
from tests.kernel_inventory.kernel_runtime_utils import _assert_return_contract
from tests.kernel_inventory.kernel_runtime_utils import _boolean_branch_assignments
from tests.kernel_inventory.kernel_runtime_utils import _call_entry_strictly
from tests.kernel_inventory.kernel_runtime_utils import _current_compute_capability_label
from tests.kernel_inventory.kernel_runtime_utils import _isolated_solution_modules
from tests.kernel_inventory.kernel_runtime_utils import _satisfy_boolean_constraints
from tests.kernel_inventory.kernel_runtime_utils import _skip_if_solution_does_not_target_current_compute_capability
from tests.kernel_inventory.kernel_runtime_utils import run_definition_solution_runtime


def test_kernel_definition_solution_runtime(definition_path, solution_path):
    run_definition_solution_runtime(definition_path, solution_path)


class _FakeCuda:
    @staticmethod
    def get_device_name(index):
        assert index == 0
        return "NVIDIA GB300"

    @staticmethod
    def get_device_capability(index):
        assert index == 0
        return 10, 3


class _FakeTorch:
    cuda = _FakeCuda()


def test_runtime_derives_target_label_from_cuda_compute_capability():
    assert _current_compute_capability_label(_FakeTorch) == "SM103"


def test_runtime_skips_solutions_that_do_not_target_current_compute_capability():
    solution = {
        "spec": {
            "target_hardware": ["SM100"],
        }
    }
    with pytest.raises(pytest.skip.Exception, match="current compute capability is SM103"):
        _skip_if_solution_does_not_target_current_compute_capability(solution, _FakeTorch)


def test_runtime_accepts_solutions_that_target_current_compute_capability():
    solution = {
        "spec": {
            "target_hardware": ["SM100", "SM103"],
        }
    }
    _skip_if_solution_does_not_target_current_compute_capability(solution, _FakeTorch)


def test_runtime_rejects_mismatched_reference_and_solution_signatures():
    def reference(q, scale=None):
        return q

    def solution(q, initial_state=None, scale=None):
        return q

    definition = {"name": "test_definition"}
    schema = {"spec": {"entry_point": "test.py::solution"}}
    with pytest.raises(AssertionError, match="does not match Solution entry point"):
        _assert_matching_entry_signatures(definition, reference, schema, solution)


def test_runtime_rejects_reference_and_solution_default_mismatch():
    def reference(q, scale):
        return q

    def solution(q, scale=None):
        return q

    definition = {"name": "test_definition"}
    schema = {"spec": {"entry_point": "test.py::solution"}}
    with pytest.raises(AssertionError, match="does not match Solution entry point"):
        _assert_matching_entry_signatures(definition, reference, schema, solution)


def test_runtime_calls_required_parameters_positionally_and_optional_parameters_by_keyword():
    calls = []

    def entry(required, optional=None):
        return required, optional

    @functools.wraps(entry)
    def recorded_entry(*args, **kwargs):
        calls.append((args, kwargs))
        return entry(*args, **kwargs)

    assert _call_entry_strictly(recorded_entry, {"required": 1, "optional": 2}, "test entry") == (1, 2)
    assert calls == [((1,), {"optional": 2})]


def test_runtime_preserves_none_outputs_in_return_arity():
    output = object()
    assert _as_outputs((output, None)) == (output, None)


def test_runtime_uses_boolean_constraints_to_select_a_schema_case():
    definition = {
        "name": "constrained_boolean_case",
        "inputs": {
            "layout": {"shape": None, "dtype": "bool"},
            "emit": {"shape": None, "dtype": "bool"},
        },
        "constraints": ["layout is True", "emit is True"],
    }
    inputs = {"layout": False, "emit": False}

    _satisfy_boolean_constraints(definition, inputs, {})

    assert inputs == {"layout": True, "emit": True}


def test_runtime_enumerates_every_unconstrained_boolean_branch():
    definition = {
        "inputs": {
            "layout": {"shape": None, "dtype": "bool"},
            "emit": {"shape": None, "dtype": "bool"},
        },
        "constraints": [],
    }

    assert _boolean_branch_assignments(definition) == [
        {"layout": False, "emit": False},
        {"layout": False, "emit": True},
        {"layout": True, "emit": False},
        {"layout": True, "emit": True},
    ]


def test_runtime_enumerates_boolean_branches_allowed_by_constraints():
    definition = {
        "name": "constrained_boolean_branches",
        "inputs": {
            "layout": {"shape": None, "dtype": "bool"},
            "emit": {"shape": None, "dtype": "bool"},
        },
        "constraints": ["layout is True"],
    }

    assert _boolean_branch_assignments(definition) == [
        {"layout": True, "emit": False},
        {"layout": True, "emit": True},
    ]


def test_runtime_ignores_unrelated_constraints_when_enumerating_booleans():
    definition = {
        "name": "unrelated_constraints",
        "inputs": {
            "layout": {"shape": None, "dtype": "bool"},
            "emit": {"shape": None, "dtype": "bool"},
        },
        "constraints": [
            "H % 2 == 0",
            "layout selects a documented implementation branch",
        ],
    }

    assert _boolean_branch_assignments(definition, {"H": 2}) == [
        {"layout": False, "emit": False},
        {"layout": False, "emit": True},
        {"layout": True, "emit": False},
        {"layout": True, "emit": True},
    ]


def test_runtime_compares_named_nested_returns_and_matching_none_values():
    torch = pytest.importorskip("torch")
    axes = {"B": 1, "ONE": 1, "H": 2, "K": 4, "V": 3}
    output_specs = {
        "output": {"shape": ["B", "H", "V"], "dtype": "bfloat16"},
        "final_state": {"shape": ["B", "H", "K", "V"], "dtype": "float32"},
        "z_state": {"shape": ["B", "ONE", "H", "K"], "dtype": "bfloat16"},
    }
    output = torch.randn(1, 2, 3, dtype=torch.bfloat16)
    z_state = torch.randn(1, 1, 2, 4, dtype=torch.bfloat16)

    _assert_return_contract(
        (output.clone(), (None, z_state.clone())),
        (output, (None, z_state)),
        ("output", ("final_state", "z_state")),
        output_specs,
        axes,
        torch,
        "test definition",
    )


def test_solution_module_isolation_restores_existing_packages():
    original_tilegym = sys.modules.get("tilegym")
    original_flashinfer = sys.modules.get("flashinfer")
    tilegym_sentinel = types.ModuleType("tilegym")
    flashinfer_sentinel = types.ModuleType("flashinfer")
    sys.modules["tilegym"] = tilegym_sentinel
    sys.modules["flashinfer"] = flashinfer_sentinel
    try:
        with _isolated_solution_modules():
            assert sys.modules["tilegym"] is not tilegym_sentinel
            assert sys.modules["flashinfer"] is not flashinfer_sentinel
            sys.modules["tilegym.generated"] = types.ModuleType("tilegym.generated")
        assert sys.modules["tilegym"] is tilegym_sentinel
        assert sys.modules["flashinfer"] is flashinfer_sentinel
        assert "tilegym.generated" not in sys.modules
    finally:
        if original_tilegym is None:
            sys.modules.pop("tilegym", None)
        else:
            sys.modules["tilegym"] = original_tilegym
        if original_flashinfer is None:
            sys.modules.pop("flashinfer", None)
        else:
            sys.modules["flashinfer"] = original_flashinfer
