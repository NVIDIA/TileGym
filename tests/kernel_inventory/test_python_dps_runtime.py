# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from tests.kernel_inventory.kernel_runtime_utils import _assert_matching_entry_signatures
from tests.kernel_inventory.kernel_runtime_utils import _assert_matching_entry_signatures_static
from tests.kernel_inventory.kernel_runtime_utils import _load_reference
from tests.kernel_inventory.kernel_runtime_utils import _run_runtime_branch
from tests.kernel_inventory.kernel_runtime_utils import run_definition_solution_workload_runtime
from tilegym.kernel_inventory import WorkloadRecord
from tilegym.kernel_inventory.correctness import poison_output_
from tilegym.kernel_inventory.schema import Workload
from tilegym.kernel_inventory.schema import resolve_definition_axes

torch = pytest.importorskip("torch")

_DEFINITION: dict[str, Any] = {
    "name": "dps_op",
    "axes": {
        "M": {"type": "var"},
        "N": {"type": "const", "value": 8},
        "HALF": {"type": "expr", "expression": "N // 2"},
    },
    "inputs": {
        "x": {"shape": ["M", "N"], "dtype": "float32"},
        "scale": {"shape": None, "dtype": "float32"},
    },
    "outputs": {
        "head": {"shape": ["M", "HALF"], "dtype": "float32"},
        "argmax": {"shape": ["M"], "dtype": "int64"},
    },
    "reference": (
        "import torch\n\n"
        "def run(x, scale):\n"
        "    head = x[:, : x.shape[1] // 2] * scale\n"
        "    argmax = x.argmax(dim=1)\n"
        "    return head, argmax\n"
    ),
}
_SOLUTION = {"spec": {"entry_point": "dps_solution.py::run", "destination_passing_style": True}}


def _run(
    solution_fn: Any,
    m: int = 3,
    *,
    definition: dict[str, Any] = _DEFINITION,
    solution: dict[str, Any] = _SOLUTION,
    skip_correctness: bool = False,
) -> None:
    axes = resolve_definition_axes(definition, {"M": m})
    _run_runtime_branch(
        definition,
        _load_reference(definition, Path("dps_op.json")),
        solution,
        solution_fn,
        {"x": torch.randn(m, 8), "scale": 2.0},
        axes,
        torch,
        (),
        case_label="row",
        skip_correctness=skip_correctness,
    )


def test_correct_dps_solution_passes_with_expression_axis_outputs():
    seen_shapes = []

    def run(x, scale, head, argmax):
        seen_shapes.append((tuple(head.shape), tuple(argmax.shape)))
        assert torch.isnan(head).all()
        assert (argmax == torch.iinfo(torch.int64).min).all()
        head.copy_(x[:, :4] * scale)
        argmax.copy_(x.argmax(dim=1))

    _run(run, m=5)

    assert seen_shapes == [((5, 4), (5,))]


def test_dps_solution_with_wrong_output_is_rejected():
    def run(x, scale, head, argmax):
        head.copy_(x[:, :4] * scale + 1)
        argmax.copy_(x.argmax(dim=1))

    with pytest.raises(AssertionError, match="outputs rejected: head"):
        _run(run)


def test_dps_solution_leaving_output_unwritten_is_rejected():
    def run(x, scale, head, argmax):
        head.copy_(x[:, :4] * scale)

    with pytest.raises(AssertionError, match="outputs rejected: argmax"):
        _run(run)


def test_dps_solution_must_return_none():
    def run(x, scale, head, argmax):
        head.copy_(x[:, :4] * scale)
        argmax.copy_(x.argmax(dim=1))
        return head, argmax

    with pytest.raises(AssertionError, match="must return None"):
        _run(run)


def test_benchmark_only_dps_skips_numerical_comparison():
    def run(x, scale, head, argmax):
        head.copy_(x[:, :4] * scale + 1)
        argmax.zero_()

    _run(run, skip_correctness=True)


def test_benchmark_only_dps_rejects_unwritten_output():
    def run(x, scale, head, argmax):
        head.copy_(x[:, :4] * scale)

    with pytest.raises(AssertionError, match="left output argmax unwritten"):
        _run(run, skip_correctness=True)


def test_benchmark_only_dps_rejects_partially_written_floating_output():
    def run(x, scale, head, argmax):
        head[0].copy_(x[0, :4] * scale)
        argmax.copy_(x.argmax(dim=1))

    with pytest.raises(AssertionError, match=r"left output head unwritten \(12 of 16 elements\)"):
        _run(run, m=4, skip_correctness=True)


def test_benchmark_only_dps_accepts_integer_values_equal_to_the_poison():
    def run(x, scale, head, argmax):
        head.copy_(x[:, :4] * scale)
        argmax.copy_(x.argmax(dim=1))
        argmax[0] = torch.iinfo(torch.int64).min

    _run(run, m=4, skip_correctness=True)


def test_benchmark_only_dps_must_return_none():
    def run(x, scale, head, argmax):
        head.copy_(x[:, :4] * scale)
        argmax.copy_(x.argmax(dim=1))
        return head

    with pytest.raises(AssertionError, match="must return None"):
        _run(run, skip_correctness=True)


_LITERAL_DIM_DEFINITION = {
    **_DEFINITION,
    "outputs": {
        "head": {"shape": ["M", "4"], "dtype": "float32"},
        "argmax": {"shape": ["M"], "dtype": "int64"},
    },
}


def test_dps_outputs_with_integer_literal_dimensions():
    seen_shapes = []

    def run(x, scale, head, argmax):
        seen_shapes.append(tuple(head.shape))
        head.copy_(x[:, :4] * scale)
        argmax.copy_(x.argmax(dim=1))

    _run(run, m=5, definition=_LITERAL_DIM_DEFINITION)

    assert seen_shapes == [(5, 4)]


def test_return_value_outputs_with_integer_literal_dimensions():
    def run(x, scale):
        return x[:, :4] * scale, x.argmax(dim=1)

    _run(
        run,
        definition=_LITERAL_DIM_DEFINITION,
        solution={"spec": {"entry_point": "dps_solution.py::run", "destination_passing_style": False}},
    )


def _run_workload_runtime(tmp_path: Path, monkeypatch: Any, definition: dict[str, Any], solution_fn: Any) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for kernel inventory runtime checks")
    definition_path = tmp_path / "dps_op.json"
    source = f"# Source: https://github.com/NVIDIA/TileGym/blob/{'0' * 40}/src/tilegym/ops/dps_op.py#L1-L4\n"
    definition = {**definition, "op_type": "elementwise", "reference": source + definition["reference"]}
    definition_path.write_text(json.dumps(definition), encoding="utf-8")
    solution_path = tmp_path / "dps_solution.json"
    solution_path.write_text(json.dumps({"spec": {**_SOLUTION["spec"], "language": "python"}}), encoding="utf-8")
    record = WorkloadRecord(
        path=tmp_path / "workload.jsonl",
        line_number=1,
        workload=Workload.model_validate(
            {
                "uuid": "5b0c1f8e-3f0d-4a51-9d7e-2f6f3c1a9e42",
                "axes": {"M": 5},
                "inputs": {"x": {"type": "random"}, "scale": {"type": "scalar", "value": 2.0}},
                "tolerance": {"max_atol": 1e-5, "max_rtol": 1e-5, "required_matched_ratio": 1.0},
            }
        ),
    )
    for name in ("validate_solution", "validate_solution_entry_point", "_assert_matching_entry_signatures_static"):
        monkeypatch.setattr(f"tests.kernel_inventory.kernel_runtime_utils.{name}", lambda *_, **__: None)
    monkeypatch.setattr("tests.kernel_inventory.kernel_runtime_utils._load_solution_entry", lambda *_: solution_fn)

    run_definition_solution_workload_runtime(record, definition_path, solution_path)


def _correct_dps_run(x, scale, head, argmax):
    head.copy_(x[:, :4] * scale)
    argmax.copy_(x.argmax(dim=1))


def test_workload_runtime_allocates_dps_outputs_with_integer_literal_dimensions(tmp_path, monkeypatch):
    seen_shapes = []

    def run(x, scale, head, argmax):
        seen_shapes.append(tuple(head.shape))
        _correct_dps_run(x, scale, head, argmax)

    _run_workload_runtime(tmp_path, monkeypatch, _LITERAL_DIM_DEFINITION, run)

    assert seen_shapes == [(5, 4)]


def test_workload_runtime_executes_reference_once_for_custom_comparator(tmp_path, monkeypatch):
    effects = types.SimpleNamespace(executions=0, checks=0)
    monkeypatch.setitem(sys.modules, "_reference_side_effects", effects)
    definition = {
        **_DEFINITION,
        "custom_correctness_entrypoint": "check",
        "reference": (
            "import _reference_side_effects as effects\n"
            "effects.executions += 1\n" + _DEFINITION["reference"] + "\n\ndef check(ref_outputs, user_outputs):\n"
            "    effects.checks += 1\n"
            "    return {name: {'passed': True} for name in ref_outputs}\n"
        ),
    }

    _run_workload_runtime(tmp_path, monkeypatch, definition, _correct_dps_run)

    assert (effects.executions, effects.checks) == (1, 1)


def test_dps_signature_requires_output_parameters(tmp_path):
    reference = _load_reference(_DEFINITION, Path("dps_op.json"))

    def run_ok(x, scale, head, argmax):
        pass

    def run_missing(x, scale, head):
        pass

    def run_reordered(x, scale, argmax, head):
        pass

    def run_returning(x, scale):
        pass

    _assert_matching_entry_signatures(_DEFINITION, reference, _SOLUTION, run_ok)
    for bad in (run_missing, run_reordered, run_returning):
        with pytest.raises(AssertionError, match="destination-passing"):
            _assert_matching_entry_signatures(_DEFINITION, reference, _SOLUTION, bad)

    source = tmp_path / "dps_solution.py"
    source.write_text("def run(x, scale, head, argmax):\n    pass\n", encoding="utf-8")
    _assert_matching_entry_signatures_static(_DEFINITION, _SOLUTION, Path("dps_op.json"), repo_root=tmp_path)
    source.write_text("def run(x, scale, head):\n    pass\n", encoding="utf-8")
    with pytest.raises(AssertionError, match="destination-passing"):
        _assert_matching_entry_signatures_static(_DEFINITION, _SOLUTION, Path("dps_op.json"), repo_root=tmp_path)


def test_non_dps_launchless_solution_keeps_return_value_contract():
    reference = _load_reference(_DEFINITION, Path("dps_op.json"))
    solution = {"spec": {"entry_point": "dps_solution.py::run", "destination_passing_style": False}}

    def run(x, scale):
        return x[:, :4] * scale, x.argmax(dim=1)

    _assert_matching_entry_signatures(_DEFINITION, reference, solution, run)
    _run_runtime_branch(
        _DEFINITION,
        reference,
        solution,
        run,
        {"x": torch.randn(3, 8), "scale": 2.0},
        resolve_definition_axes(_DEFINITION, {"M": 3}),
        torch,
        (),
    )


def test_optional_inplace_input_passed_as_none_is_skipped():
    definition = {
        "name": "optional_state",
        "axes": {"N": {"type": "const", "value": 4}},
        "inputs": {
            "x": {"shape": ["N"], "dtype": "float32"},
            "state": {"shape": ["N"], "dtype": "float32", "inplace_output": True},
        },
        "outputs": {"y": {"shape": ["N"], "dtype": "float32"}},
        "reference": (
            "def run(x, state=None):\n    if state is not None:\n        state.add_(x)\n    y = x * 2\n    return y\n"
        ),
    }

    def run(x, state=None):
        if state is not None:
            state.add_(x)
        return x * 2

    _run_runtime_branch(
        definition,
        _load_reference(definition, Path("optional_state.json")),
        {"spec": {"entry_point": "optional_state.py::run", "destination_passing_style": False}},
        run,
        {"x": torch.randn(4), "state": None},
        {"N": 4},
        torch,
        ("state",),
    )


@pytest.mark.parametrize(
    ("dtype", "check"),
    [
        (torch.float32, lambda t: torch.isnan(t).all()),
        (torch.bfloat16, lambda t: torch.isnan(t).all()),
        (torch.int32, lambda t: (t == torch.iinfo(torch.int32).min).all()),
        (torch.uint8, lambda t: (t == 255).all()),
        (torch.bool, lambda t: t.all()),
    ],
)
def test_poison_output_matches_harness_convention(dtype, check):
    assert check(poison_output_(torch.empty(3, 5, dtype=dtype)))
