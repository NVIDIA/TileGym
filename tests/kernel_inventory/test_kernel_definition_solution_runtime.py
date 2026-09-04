# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import builtins
import functools
import importlib.util
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

from tests.kernel_inventory.kernel_runtime_utils import _as_outputs
from tests.kernel_inventory.kernel_runtime_utils import _assert_matching_entry_signatures
from tests.kernel_inventory.kernel_runtime_utils import _assert_matching_entry_signatures_static
from tests.kernel_inventory.kernel_runtime_utils import _assert_return_contract
from tests.kernel_inventory.kernel_runtime_utils import _assert_triton_autotune_argument_ownership
from tests.kernel_inventory.kernel_runtime_utils import _call_entry_strictly
from tests.kernel_inventory.kernel_runtime_utils import _current_compute_capability_label
from tests.kernel_inventory.kernel_runtime_utils import _current_triton_backend
from tests.kernel_inventory.kernel_runtime_utils import _isolated_solution_modules
from tests.kernel_inventory.kernel_runtime_utils import _launch_raw_solution
from tests.kernel_inventory.kernel_runtime_utils import _load_reference
from tests.kernel_inventory.kernel_runtime_utils import _require_solution_runtime_dependencies
from tests.kernel_inventory.kernel_runtime_utils import _run_definition_solution_workload_runtime
from tests.kernel_inventory.kernel_runtime_utils import _run_runtime_branch
from tests.kernel_inventory.kernel_runtime_utils import _runtime_case_id
from tests.kernel_inventory.kernel_runtime_utils import _skip_if_solution_does_not_target_current_compute_capability
from tests.kernel_inventory.kernel_runtime_utils import _skip_if_solution_does_not_target_current_triton_backend
from tests.kernel_inventory.kernel_runtime_utils import _unwrap_triton_heuristics
from tests.kernel_inventory.kernel_runtime_utils import installed_reference_modules
from tests.kernel_inventory.kernel_runtime_utils import materialize_workload_inputs
from tests.kernel_inventory.kernel_runtime_utils import run_definition_solution_workload_runtime
from tilegym.kernel_inventory.layout import definition_solution_paths_for_workload
from tilegym.kernel_inventory.schema import Workload
from tilegym.kernel_inventory.workloads import WorkloadRecord


def _workload_record(tmp_path, *, inputs=None, axes=None):
    return WorkloadRecord(
        path=tmp_path / "workload.jsonl",
        line_number=1,
        workload=Workload.model_validate(
            {
                "uuid": "c8ab8964-8624-44f7-bd4f-bd1ace75c959",
                "axes": axes or {},
                "inputs": inputs or {},
                "tolerance": {
                    "max_atol": 0.02,
                    "max_rtol": 0.02,
                    "required_matched_ratio": 1.0,
                    "max_error_cap": None,
                    "allow_negative_inf": False,
                },
                "eval_mode": "full",
            }
        ),
    )


def test_kernel_definition_solution_runtime(workload_record, definition_path, solution_path):
    run_definition_solution_workload_runtime(workload_record, definition_path, solution_path)


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


class _AvailableFakeCuda(_FakeCuda):
    @staticmethod
    def is_available():
        return True


class _AvailableFakeTorch:
    cuda = _AvailableFakeCuda()

    @staticmethod
    def device(name):
        assert name == "cuda"
        return name

    @staticmethod
    def manual_seed(seed):
        assert seed == 2026


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


def test_raw_runtime_unwraps_only_fully_explicit_triton_heuristics():
    from triton.runtime.autotuner import Heuristics

    inner = object()
    wrapped = Heuristics(inner, ["x"], {"HAS_SCALE": lambda args: args["x"] is not None})
    assert _unwrap_triton_heuristics(wrapped, {"HAS_SCALE"}) is inner

    with pytest.raises(pytest.fail.Exception, match="partially override"):
        _unwrap_triton_heuristics(wrapped, set())


def test_raw_runtime_requires_compiler_scoped_bindings_to_match_active_autotune_configs():
    import triton
    from triton.runtime.autotuner import Autotuner

    autotuner = object.__new__(Autotuner)
    autotuner.configs = [triton.Config({"BK": 32}), triton.Config({"BK": 64})]
    _assert_triton_autotune_argument_ownership(autotuner, {}, {"BK"})
    _assert_triton_autotune_argument_ownership(types.SimpleNamespace(fn=autotuner), {}, {"BK"})
    with pytest.raises(pytest.fail.Exception, match="also supplies"):
        _assert_triton_autotune_argument_ownership(autotuner, {"BK": 32}, {"BK"})

    autotuner.configs = [triton.Config({}), triton.Config({"BK": 64})]
    with pytest.raises(pytest.fail.Exception, match="not every active autotune config supplies"):
        _assert_triton_autotune_argument_ownership(autotuner, {}, {"BK"})
    with pytest.raises(pytest.fail.Exception, match="also supplies"):
        _assert_triton_autotune_argument_ownership(autotuner, {"BK": 32}, {"BK"})

    autotuner.configs = [triton.Config({}), triton.Config({})]
    _assert_triton_autotune_argument_ownership(autotuner, {"BK": 32}, {"BK"})

    with pytest.raises(pytest.fail.Exception, match="require an active triton.autotune"):
        _assert_triton_autotune_argument_ownership(object(), {}, {"BK"})


def test_runtime_accepts_solutions_that_target_current_compute_capability():
    solution = {
        "spec": {
            "target_hardware": ["SM100", "SM103"],
        }
    }
    _skip_if_solution_does_not_target_current_compute_capability(solution, _FakeTorch)


def test_runtime_detects_triton_compiler_with_kernel_inventory_detector(monkeypatch):
    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils.get_available_triton_backend", lambda: "nvt")
    assert _current_triton_backend() == "nvt"


def test_kernel_inventory_triton_backend_detection_is_backend_isolated(monkeypatch):
    detector_path = Path(__file__).resolve().parents[2] / "src/tilegym/kernel_inventory/triton_backend.py"
    spec = importlib.util.spec_from_file_location("_tilegym_inventory_triton_backend_test", detector_path)
    assert spec is not None and spec.loader is not None
    detector = importlib.util.module_from_spec(spec)

    backend_modules_before = {
        name for name in sys.modules if name == "tilegym.backend" or name.startswith("tilegym.backend.")
    }
    spec.loader.exec_module(detector)

    tileir_available = True
    original_import = builtins.__import__

    def isolated_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "triton.backends.tileir":
            if not tileir_available:
                raise ImportError("synthetic missing TileIR backend")
            return types.ModuleType(name)
        if name == "tilegym.backend" or name.startswith("tilegym.backend."):
            pytest.fail(f"kernel inventory detector imported backend module {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", isolated_import)
    monkeypatch.setenv("ENABLE_TILE", "1")
    assert detector.is_triton_tileir_available() is True
    assert detector.get_available_triton_backend() == "nvt"

    monkeypatch.setenv("ENABLE_TILE", "0")
    assert detector.is_triton_tileir_available() is False
    assert detector.get_available_triton_backend() == "oait"

    tileir_available = False
    monkeypatch.setenv("ENABLE_TILE", "1")
    assert detector.is_triton_tileir_available() is False
    assert detector.get_available_triton_backend() == "oait"
    backend_modules_after = {
        name for name in sys.modules if name == "tilegym.backend" or name.startswith("tilegym.backend.")
    }
    assert backend_modules_after == backend_modules_before


def test_runtime_skips_declared_triton_compiler_mismatch(monkeypatch):
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._current_triton_backend",
        lambda: "oait",
    )
    solution = {"spec": {"language": "triton", "target_triton_backends": ["nvt"]}}
    with pytest.raises(pytest.skip.Exception, match="current Triton compiler backend is oait"):
        _skip_if_solution_does_not_target_current_triton_backend(solution)


def test_runtime_accepts_declared_matching_triton_compiler(monkeypatch):
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._current_triton_backend",
        lambda: "nvt",
    )
    solution = {"spec": {"language": "triton", "target_triton_backends": ["nvt"]}}
    _skip_if_solution_does_not_target_current_triton_backend(solution)


def test_runtime_without_declared_triton_compiler_target_does_not_probe_backend(monkeypatch):
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._current_triton_backend",
        lambda: pytest.fail("backend must not be probed without an explicit compiler target"),
    )
    _skip_if_solution_does_not_target_current_triton_backend({"spec": {"language": "triton"}})


def test_runtime_accepts_optional_solution_signature_extensions():
    def reference(q, scale=None):
        return q

    def solution(q, initial_state=None, scale=None):
        return q

    definition = {"name": "test_definition", "inputs": {"q": {}, "scale": {}}}
    schema = {"spec": {"entry_point": "test.py::solution"}}
    _assert_matching_entry_signatures(definition, reference, schema, solution)


def test_runtime_preserves_exact_legacy_signature_with_noncanonical_input_order():
    def reference(q, scale=None):
        return q

    def solution(q, scale=None):
        return q

    definition = {"name": "legacy_definition", "inputs": {"scale": {}, "q": {}}}
    schema = {"spec": {"entry_point": "test.py::solution"}}
    _assert_matching_entry_signatures(definition, reference, schema, solution)


def test_static_runtime_accepts_exact_kwargs_only_unsupported_stub(tmp_path):
    source = tmp_path / "solution.py"
    source.write_text("def solution(**kwargs):\n    raise NotImplementedError\n", encoding="utf-8")
    definition = {
        "name": "unsupported",
        "inputs": {},
        "reference": "def run(**kwargs):\n    raise NotImplementedError\n",
    }
    solution = {"spec": {"entry_point": "solution.py::solution"}}

    _assert_matching_entry_signatures_static(definition, solution, tmp_path / "definition.json", tmp_path)


def test_static_runtime_accepts_reordered_optional_solution_extensions(tmp_path):
    source = tmp_path / "solution.py"
    source.write_text(
        "def solution(q, initial_state=None, reverse=False, normalize=False):\n    return q\n",
        encoding="utf-8",
    )
    definition = {
        "name": "public",
        "inputs": {"q": {}, "normalize": {}, "reverse": {}},
        "reference": "def run(q, normalize=False, reverse=False):\n    return q\n",
    }
    solution = {"spec": {"entry_point": "solution.py::solution"}}

    _assert_matching_entry_signatures_static(definition, solution, tmp_path / "definition.json", tmp_path)


def test_runtime_rejects_required_solution_signature_extensions():
    def reference(q, scale=None):
        return q

    def solution(q, initial_state, scale=None):
        return q

    definition = {"name": "test_definition", "inputs": {"q": {}, "scale": {}}}
    schema = {"spec": {"entry_point": "test.py::solution"}}
    with pytest.raises(AssertionError, match="does not match Solution entry point"):
        _assert_matching_entry_signatures(definition, reference, schema, solution)


def test_runtime_rejects_reference_and_solution_default_mismatch():
    def reference(q, scale):
        return q

    def solution(q, scale=None):
        return q

    definition = {"name": "test_definition", "inputs": {"q": {}, "scale": {}}}
    schema = {"spec": {"entry_point": "test.py::solution"}}
    with pytest.raises(AssertionError, match="does not match Solution entry point"):
        _assert_matching_entry_signatures(definition, reference, schema, solution)


def test_static_signature_mismatch_precedes_backend_dependency_gate(tmp_path, monkeypatch):
    workload_record = _workload_record(tmp_path)
    definition_path = tmp_path / "definition.json"
    solution_path = tmp_path / "solution.json"
    definition_path.write_text(
        json.dumps({"name": "wrapper", "reference": "def run(value):\n    return value\n"}),
        encoding="utf-8",
    )
    solution_path.write_text(
        json.dumps({"spec": {"language": "triton", "entry_point": "solution.py::entry"}}),
        encoding="utf-8",
    )
    dependency_gate_called = False
    compiler_gate_called = False
    static_signature_checked = False

    def static_signature_check(*_args):
        nonlocal static_signature_checked
        static_signature_checked = True
        raise AssertionError("does not match Solution entry point")

    def dependency_gate(_solution):
        nonlocal dependency_gate_called
        dependency_gate_called = True

    def compiler_gate(_solution):
        nonlocal compiler_gate_called
        compiler_gate_called = True

    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._require_solution_runtime_dependencies",
        dependency_gate,
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._skip_if_solution_does_not_target_current_triton_backend",
        compiler_gate,
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._assert_matching_entry_signatures_static",
        static_signature_check,
    )
    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils.validate_definition", lambda *_: None)
    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils.validate_solution", lambda *_, **__: None)
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.validate_solution_entry_point",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.validate_workload_against_definition",
        lambda *_, **__: None,
    )
    with pytest.raises(AssertionError, match="does not match Solution entry point"):
        _run_definition_solution_workload_runtime(workload_record, definition_path, solution_path)
    assert static_signature_checked
    assert not dependency_gate_called
    assert not compiler_gate_called


def test_launch_contract_validation_precedes_triton_compiler_gate(tmp_path, monkeypatch):
    workload_record = _workload_record(tmp_path)
    definition_path = tmp_path / "definition.json"
    solution_path = tmp_path / "solution.json"
    definition_path.write_text(
        json.dumps({"name": "leaf", "reference": "def run(value):\n    return value\n"}),
        encoding="utf-8",
    )
    solution_path.write_text(
        json.dumps(
            {
                "spec": {
                    "language": "triton",
                    "entry_point": "solution.py::entry",
                    "target_triton_backends": ["nvt"],
                },
                "launch": {"grid": [1], "arguments": []},
            }
        ),
        encoding="utf-8",
    )
    compiler_gate_called = False

    def invalid_launch(*_args, **_kwargs):
        raise ValueError("Solution launch contract invalid")

    def compiler_gate(_solution):
        nonlocal compiler_gate_called
        compiler_gate_called = True

    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils.validate_definition", lambda *_: None)
    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils.validate_solution", invalid_launch)
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._skip_if_solution_does_not_target_current_triton_backend",
        compiler_gate,
    )
    with pytest.raises(ValueError, match="launch contract invalid"):
        _run_definition_solution_workload_runtime(workload_record, definition_path, solution_path)
    assert not compiler_gate_called


def test_triton_compiler_mismatch_skips_before_solution_import(tmp_path, monkeypatch):
    workload_record = _workload_record(tmp_path)
    definition_path = tmp_path / "definition.json"
    solution_path = tmp_path / "solution.json"
    definition_path.write_text(
        json.dumps({"name": "leaf", "reference": "def run(value):\n    return value\n"}),
        encoding="utf-8",
    )
    solution_path.write_text(
        json.dumps(
            {
                "spec": {
                    "language": "triton",
                    "entry_point": "solution.py::entry",
                    "target_triton_backends": ["nvt"],
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils.validate_definition", lambda *_: None)
    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils.validate_solution", lambda *_, **__: None)
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.validate_solution_entry_point",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._assert_matching_entry_signatures_static",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.validate_workload_against_definition",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._require_solution_runtime_dependencies",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._current_triton_backend",
        lambda: "oait",
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._load_solution_entry",
        lambda *_: pytest.fail("mismatched compiler target must skip before importing the Solution"),
    )
    with pytest.raises(pytest.skip.Exception, match="current Triton compiler backend is oait"):
        _run_definition_solution_workload_runtime(workload_record, definition_path, solution_path)


def test_hardware_gate_precedes_solution_import_but_matching_hardware_executes(tmp_path, monkeypatch):
    workload_record = _workload_record(tmp_path, inputs={"value": {"type": "scalar", "value": 4}})
    definition_path = tmp_path / "definition.json"
    solution_path = tmp_path / "solution.json"
    module_name = "tilegym.hardware_gate_test.poison"
    source_path = tmp_path / "src/tilegym/hardware_gate_test/poison.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        'raise RuntimeError("SM-incompatible Solution module was imported")\ndef entry(value):\n    return value\n',
        encoding="utf-8",
    )
    definition_path.write_text(
        json.dumps(
            {
                "name": "wrapper",
                "reference": "def run(value):\n    return value\n",
                "axes": {},
                "inputs": {"value": {}},
                "outputs": {},
            }
        ),
        encoding="utf-8",
    )
    solution = {
        "spec": {
            "language": "triton",
            "entry_point": "src/tilegym/hardware_gate_test/poison.py::entry",
            "target_hardware": ["SM100"],
        }
    }
    solution_path.write_text(json.dumps(solution), encoding="utf-8")
    order = []

    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils.REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.validate_definition",
        lambda *_: order.append("definition"),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.validate_solution",
        lambda *_, **__: order.append("solution"),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.validate_solution_entry_point",
        lambda *_, **__: order.append("entry-point"),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.validate_workload_against_definition",
        lambda *_, **__: order.append("workload"),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._assert_matching_entry_signatures_static",
        lambda *_: order.append("static-signature"),
    )
    monkeypatch.setattr(
        pytest,
        "importorskip",
        lambda name: order.append(f"import:{name}") or _AvailableFakeTorch,
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._require_solution_runtime_dependencies",
        lambda *_: order.append("dependency"),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._skip_if_solution_does_not_target_current_triton_backend",
        lambda *_: order.append("compiler"),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.installed_reference_modules",
        lambda *_: nullcontext(),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._load_reference",
        lambda *_: order.append("reference") or (lambda value: value),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._assert_matching_entry_signatures",
        lambda *_: order.append("dynamic-signature"),
    )
    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.materialize_workload_inputs",
        lambda *_, **__: ({}, {"value": 4}),
    )

    def run_branch(_definition, _reference, _solution, solution_fn, *_, **__):
        order.append("execute")
        assert solution_fn(4) == 5

    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils._run_runtime_branch",
        run_branch,
    )
    with pytest.raises(pytest.skip.Exception, match="current compute capability is SM103"):
        _run_definition_solution_workload_runtime(workload_record, definition_path, solution_path)
    assert order == [
        "definition",
        "solution",
        "entry-point",
        "workload",
        "static-signature",
        "import:torch",
        "dependency",
        "compiler",
    ]
    assert module_name not in sys.modules

    source_path.write_text("def entry(value):\n    return value + 1\n", encoding="utf-8")
    solution["spec"]["target_hardware"] = ["SM103"]
    solution_path.write_text(json.dumps(solution), encoding="utf-8")

    order.clear()
    try:
        _run_definition_solution_workload_runtime(workload_record, definition_path, solution_path)
    finally:
        sys.modules.pop(module_name, None)
    assert order == [
        "definition",
        "solution",
        "entry-point",
        "workload",
        "static-signature",
        "import:torch",
        "dependency",
        "compiler",
        "reference",
        "dynamic-signature",
        "execute",
    ]


def test_static_signature_validation_rejects_actual_ast_mismatch(tmp_path):
    source = tmp_path / "solution.py"
    source.write_text("def entry(value, extra):\n    return value\n", encoding="utf-8")
    definition = {"name": "wrapper", "inputs": {"value": {}}, "reference": "def run(value):\n    return value\n"}
    solution = {"spec": {"entry_point": "solution.py::entry"}}
    with pytest.raises(AssertionError, match="does not match Solution entry point"):
        _assert_matching_entry_signatures_static(definition, solution, tmp_path / "definition.json", tmp_path)


def test_runtime_dependency_gate_is_language_specific(monkeypatch):
    requested = []
    monkeypatch.setattr(pytest, "importorskip", lambda name: requested.append(name))
    _require_solution_runtime_dependencies({"spec": {"language": "triton"}})
    assert requested == ["triton"]
    requested.clear()
    _require_solution_runtime_dependencies({"spec": {"language": "cuda-tile"}})
    assert requested == ["cuda.tile"]


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


def test_runtime_passes_explicit_none_to_every_optional_parameter_kind():
    calls = []

    def entry(required, optional_posonly=None, /, optional_positional_or_keyword=None, *, optional_keyword=None):
        return required, optional_posonly, optional_positional_or_keyword, optional_keyword

    @functools.wraps(entry)
    def recorded_entry(*args, **kwargs):
        calls.append((args, kwargs))
        return entry(*args, **kwargs)

    inputs = {
        "required": object(),
        "optional_posonly": None,
        "optional_positional_or_keyword": None,
        "optional_keyword": None,
    }
    assert _call_entry_strictly(recorded_entry, inputs, "test entry") == (
        inputs["required"],
        None,
        None,
        None,
    )
    assert calls == [
        (
            (inputs["required"], None),
            {"optional_positional_or_keyword": None, "optional_keyword": None},
        )
    ]


def test_runtime_can_omit_optional_positional_only_and_bind_later_keyword_parameter():
    def entry(required, optional_posonly=None, /, optional_keyword=None):
        return required, optional_posonly, optional_keyword

    assert _call_entry_strictly(entry, {"required": 1, "optional_keyword": 2}, "test entry") == (1, None, 2)


def test_runtime_preserves_none_outputs_in_return_arity():
    output = object()
    assert _as_outputs((output, None)) == (output, None)


def test_runtime_case_id_contains_definition_coordinate_and_solution_backend(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    definition = inventory / "kernel_definitions/op/cutile/op.json"
    solution = inventory / "kernel_solutions/op/triton/op.json"
    definition.parent.mkdir(parents=True)
    solution.parent.mkdir(parents=True)
    definition.write_text("{}\n", encoding="utf-8")
    solution.write_text('{"spec": {"language": "triton"}}\n', encoding="utf-8")
    record = _workload_record(tmp_path)
    case_id = _runtime_case_id(record, definition, solution)
    assert case_id == (
        "src/tilegym/suites/example::definition::op/cutile/op::line=1::"
        "uuid=c8ab8964-8624-44f7-bd4f-bd1ace75c959::backend=triton"
    )


def test_public_and_wrapper_workloads_plan_four_adjacent_runtime_cases(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    definitions = [
        inventory / "kernel_definitions/op/op.json",
        inventory / "kernel_definitions/op/cutile/op.json",
        inventory / "kernel_definitions/op/triton/op.json",
    ]
    solutions = [
        inventory / "kernel_solutions/op/cutile/op.json",
        inventory / "kernel_solutions/op/triton/op.json",
    ]
    workload_paths = [
        inventory / "kernel_workloads/op/op/workload.jsonl",
        inventory / "kernel_workloads/op/cutile/op/workload.jsonl",
        inventory / "kernel_workloads/op/triton/op/workload.jsonl",
    ]
    for path in definitions:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    for backend, path in zip(("cutile", "triton"), solutions, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"spec": {"language": backend}}), encoding="utf-8")
    for path in workload_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    records = [
        WorkloadRecord(path=path, line_number=1, workload=_workload_record(tmp_path).workload)
        for path in workload_paths
    ]
    planned = [
        (record, definition, solution)
        for record in records
        for definition, solution in definition_solution_paths_for_workload(record.path)
    ]
    ids = {_runtime_case_id(*case) for case in planned}

    assert len(planned) == len(ids) == 4
    assert [(definition.parent.name, solution.parent.name) for _, definition, solution in planned] == [
        ("op", "triton"),
        ("op", "cutile"),
        ("cutile", "cutile"),
        ("triton", "triton"),
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


def test_raw_triton_launch_uses_definition_outputs_and_bound_arguments(monkeypatch):
    torch = pytest.importorskip("torch")
    definition = {
        "name": "raw_copy",
        "axes": {"N": {"type": "var"}},
        "inputs": {"input": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float32"}},
    }
    solution = {
        "spec": {"language": "triton", "entry_point": "unused.py::copy_kernel"},
        "launch": {
            "grid": [1],
            "arguments": [
                {"parameter": "input", "kind": "input", "name": "input"},
                {"parameter": "output", "kind": "output", "name": "output", "initialize": "zeros"},
                {"parameter": "n", "kind": "axis", "name": "N"},
            ],
        },
    }
    launches = []

    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.inspect_raw_entry_signature",
        lambda *_: types.SimpleNamespace(parameters=()),
    )

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                launches.append((grid, kwargs["n"]))
                kwargs["output"].copy_(kwargs["input"])

            return launch

    input_tensor = torch.arange(4, dtype=torch.float32)
    output = _launch_raw_solution(
        definition,
        solution,
        FakeKernel(),
        {"input": input_tensor},
        {"N": 4},
        torch,
    )

    torch.testing.assert_close(output, input_tensor)
    assert launches == [((1,), 4)]


def test_raw_triton_launch_can_seed_mutated_output_from_definition_input(monkeypatch):
    torch = pytest.importorskip("torch")
    definition = {
        "name": "raw_inplace_update",
        "axes": {"N": {"type": "var"}},
        "inputs": {"state_seed": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"state": {"shape": ["N"], "dtype": "float32"}},
    }
    solution = {
        "spec": {"language": "triton", "entry_point": "unused.py::update_kernel"},
        "launch": {
            "grid": [1],
            "arguments": [
                {
                    "parameter": "state",
                    "kind": "output",
                    "name": "state",
                    "initialize_from": "state_seed",
                }
            ],
        },
    }

    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.inspect_raw_entry_signature",
        lambda *_: types.SimpleNamespace(parameters=()),
    )

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (1,)

            def launch(**kwargs):
                kwargs["state"].add_(1)

            return launch

    seed = torch.arange(4, dtype=torch.float32)
    output = _launch_raw_solution(
        definition,
        solution,
        FakeKernel(),
        {"state_seed": seed},
        {"N": 4},
        torch,
    )

    torch.testing.assert_close(output, seed + 1)
    torch.testing.assert_close(seed, torch.arange(4, dtype=torch.float32))


def test_raw_triton_launch_preserves_shaped_seeded_output_through_flatten_view(monkeypatch):
    torch = pytest.importorskip("torch")
    definition = {
        "name": "raw_flattened_inplace_update",
        "axes": {"M": {"type": "var"}, "N": {"type": "var"}},
        "inputs": {"state_seed": {"shape": ["M", "N"], "dtype": "float32"}},
        "outputs": {"state": {"shape": ["M", "N"], "dtype": "float32"}},
    }
    solution = {
        "spec": {"language": "triton", "entry_point": "unused.py::update_kernel"},
        "launch": {
            "grid": [1],
            "arguments": [
                {
                    "parameter": "state_seed",
                    "kind": "input",
                    "name": "state_seed",
                    "view": "flatten",
                },
                {
                    "parameter": "state",
                    "kind": "output",
                    "name": "state",
                    "initialize_from": "state_seed",
                    "view": "flatten",
                },
            ],
        },
    }

    monkeypatch.setattr(
        "tests.kernel_inventory.kernel_runtime_utils.inspect_raw_entry_signature",
        lambda *_: types.SimpleNamespace(parameters=()),
    )

    class FakeKernel:
        def __getitem__(self, grid):
            assert grid == (1,)

            def launch(**kwargs):
                assert kwargs["state_seed"].shape == (6,)
                assert kwargs["state"].shape == (6,)
                kwargs["state"].copy_(kwargs["state_seed"])
                kwargs["state"].add_(1)

            return launch

    seed = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    output = _launch_raw_solution(
        definition,
        solution,
        FakeKernel(),
        {"state_seed": seed},
        {"M": 2, "N": 3},
        torch,
    )

    assert output.shape == (2, 3)
    torch.testing.assert_close(output, seed + 1)
    torch.testing.assert_close(seed, torch.arange(6, dtype=torch.float32).reshape(2, 3))


def test_runtime_rejects_undeclared_solution_input_mutation(tmp_path):
    torch = pytest.importorskip("torch")
    definition = {
        "name": "mutation_guard",
        "axes": {"N": {"type": "const", "value": 3}},
        "inputs": {"input": {"shape": ["N"], "dtype": "float32"}},
        "outputs": {"output": {"shape": ["N"], "dtype": "float32"}},
        "reference": "def run(input):\n    output = input + 1\n    return output\n",
    }
    reference = _load_reference(definition, tmp_path / "definition.json")

    def mutating_solution(input):
        input.add_(1)
        return input

    with pytest.raises(AssertionError, match="Solution unexpectedly mutated input input"):
        _run_runtime_branch(
            definition,
            reference,
            {"spec": {"entry_point": "synthetic.py::mutating_solution"}},
            mutating_solution,
            {"input": torch.arange(3, dtype=torch.float32)},
            {"N": 3},
            torch,
            (),
        )


def test_runtime_return_comparison_uses_explicit_workload_tolerance():
    torch = pytest.importorskip("torch")
    actual = torch.tensor([1.01], dtype=torch.float32)
    expected = torch.tensor([1.0], dtype=torch.float32)
    output_specs = {"output": {"shape": ["N"], "dtype": "float32"}}

    _assert_return_contract(
        actual,
        expected,
        "output",
        output_specs,
        {"N": 1},
        torch,
        "loose-row",
        rtol=0.0,
        atol=0.02,
    )
    with pytest.raises(AssertionError, match="strict-row output output mismatch"):
        _assert_return_contract(
            actual,
            expected,
            "output",
            output_specs,
            {"N": 1},
            torch,
            "strict-row",
            rtol=0.0,
            atol=0.001,
        )


def test_cpu_composed_wrapper_runtime_canary_executes_each_explicit_boolean_row_once(tmp_path):
    torch = pytest.importorskip("torch")
    leaf_scale = {
        "name": "leaf_scale",
        "reference": "def run(input, negate):\n    return -input if negate else input\n",
    }
    leaf_update = {
        "name": "leaf_update",
        "reference": "def run(value, state):\n    state.add_(1)\n    return value + state\n",
    }
    wrapper = {
        "name": "wrapper",
        "include": ["leaf_scale", "leaf_update"],
        "axes": {"N": {"type": "const", "value": 3}},
        "inputs": {
            "input": {"shape": ["N"], "dtype": "float32"},
            "state": {"shape": ["N"], "dtype": "float32", "inplace_output": True},
            "negate": {"shape": None, "dtype": "bool"},
        },
        "outputs": {"output": {"shape": ["N"], "dtype": "float32"}},
        "reference": (
            "import leaf_scale\n"
            "import leaf_update\n\n"
            "def run(input, state, negate):\n"
            "    value = leaf_scale.run(input, negate)\n"
            "    output = leaf_update.run(value, state)\n"
            "    return output\n"
        ),
    }
    wrapper_path = tmp_path / "wrapper.json"
    for name, definition in (("leaf_scale", leaf_scale), ("leaf_update", leaf_update)):
        (tmp_path / f"{name}.json").write_text(json.dumps(definition), encoding="utf-8")

    executed = []

    def solution(input, state, negate):
        executed.append(negate)
        value = -input if negate else input
        state.add_(1)
        return value + state

    with installed_reference_modules(wrapper, wrapper_path):
        reference = _load_reference(wrapper, wrapper_path)
        solution_schema = {"spec": {"entry_point": "synthetic.py::solution"}}
        _assert_matching_entry_signatures(wrapper, reference, solution_schema, solution)
        records = [
            _workload_record(
                tmp_path,
                inputs={
                    "input": {"type": "random"},
                    "state": {"type": "random"},
                    "negate": {"type": "scalar", "value": negate},
                },
            )
            for negate in (False, True)
        ]
        for record in records:
            torch.manual_seed(2026)
            axes, inputs = materialize_workload_inputs(
                record,
                wrapper,
                torch=torch,
                device=torch.device("cpu"),
            )
            _run_runtime_branch(
                wrapper,
                reference,
                solution_schema,
                solution,
                inputs,
                axes,
                torch,
                ("state",),
                tolerance=record.workload.tolerance,
                case_label=record.source_label,
            )

    assert executed == [False, True]
