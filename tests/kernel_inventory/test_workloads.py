# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tilegym.kernel_inventory.layout import inventory_coordinate
from tilegym.kernel_inventory.layout import iter_inventory_workload_paths
from tilegym.kernel_inventory.layout import mirrored_definition_path
from tilegym.kernel_inventory.workloads import KernelWorkloadError
from tilegym.kernel_inventory.workloads import load_workload_jsonl
from tilegym.kernel_inventory.workloads import materialize_workload_inputs
from tilegym.kernel_inventory.workloads import validate_workload_against_definition
from tilegym.kernel_inventory.workloads import validate_workload_catalog

REPO_ROOT = Path(__file__).resolve().parents[2]
_TOLERANCE = {
    "max_atol": 0.02,
    "max_rtol": 0.02,
    "required_matched_ratio": 1.0,
    "max_error_cap": None,
    "allow_negative_inf": False,
}


def _payload(uuid: str = "276d6a74-bc55-4a5b-9d46-cb82b7f85e92", **updates: Any) -> dict[str, Any]:
    payload = {
        "uuid": uuid,
        "axes": {"N": 8},
        "inputs": {
            "x": {"type": "random"},
            "scale": {"type": "scalar", "value": 1.0},
            "enabled": {"type": "scalar", "value": False},
            "bias": {"type": "null"},
        },
        "tolerance": copy.deepcopy(_TOLERANCE),
        "eval_mode": "full",
    }
    payload.update(updates)
    return payload


def _definition() -> dict[str, Any]:
    return {
        "name": "op",
        "axes": {"N": {"type": "var"}, "K": {"type": "const", "value": 4}},
        "inputs": {
            "x": {"shape": ["N", "K"], "dtype": "float32"},
            "scale": {"shape": None, "dtype": "float32"},
            "enabled": {"shape": None, "dtype": "bool"},
            "bias": {"shape": ["K"], "dtype": "float32"},
        },
        "outputs": {"output": {"shape": ["N", "K"], "dtype": "float32"}},
        "reference": "def run(x, scale, enabled, bias=None):\n    return x\n",
        "constraints": ["N >= K", "enabled is False"],
    }


def _write_jsonl(path: Path, *payloads: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(payload, separators=(",", ":")) + "\n" for payload in payloads), encoding="utf-8"
    )
    return path


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_load_workload_jsonl_returns_path_and_line_records(tmp_path):
    path = _write_jsonl(
        tmp_path / "op.jsonl",
        _payload(),
        _payload("496432e2-102a-4541-b3f5-72c10f7e1f87", axes={"N": 16}),
    )

    records = load_workload_jsonl(path)

    assert [record.line_number for record in records] == [1, 2]
    assert [record.path for record in records] == [path, path]
    assert [record.workload.axes for record in records] == [{"N": 8}, {"N": 16}]


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("", "must not be empty"),
        ("\n", "blank lines"),
        (json.dumps(_payload()) + "\n\n", "blank lines"),
        ("# comment\n", "comments"),
        ("{bad json}\n", "invalid JSON"),
        ("[]\n", "expected one JSON object"),
        (json.dumps(_payload()) + json.dumps(_payload()) + "\n", "invalid JSON"),
    ],
)
def test_load_workload_jsonl_rejects_noncanonical_physical_lines(tmp_path, content, match):
    path = tmp_path / "invalid.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(KernelWorkloadError, match=match):
        load_workload_jsonl(path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update(private_extension=True), "unsupported Workload fields"),
        (lambda payload: payload.pop("tolerance"), "missing explicit Workload fields"),
        (lambda payload: payload["tolerance"].pop("max_error_cap"), "tolerance fields mismatch"),
        (lambda payload: payload["tolerance"].update(max_atol=0.01), "migration policy"),
        (lambda payload: payload["tolerance"].update(required_matched_ratio=0.97), "migration policy"),
        (
            lambda payload: payload.update(eval_mode="dynamic"),
            "eval_mode must be 'full', 'correctness_only', or 'benchmark_only'",
        ),
        (lambda payload: payload.update(uuid="not-a-uuid"), "UUIDv4"),
        (lambda payload: payload.update(uuid="00000000-0000-1000-8000-000000000000"), "UUIDv4"),
        (
            lambda payload: payload.update(inputs={"scale": {"type": "scalar", "value": float("inf")}}),
            "numbers must be finite",
        ),
    ],
)
def test_checked_in_workload_policy_is_stricter_than_schema(tmp_path, mutate, match):
    payload = _payload()
    mutate(payload)
    path = _write_jsonl(tmp_path / "invalid.jsonl", payload)

    with pytest.raises(KernelWorkloadError, match=match):
        load_workload_jsonl(path)


@pytest.mark.parametrize("eval_mode", ["benchmark_only", "correctness_only"])
def test_checked_in_workload_policy_allows_non_full_eval_modes(tmp_path, eval_mode):
    payload = _payload()
    payload["eval_mode"] = eval_mode
    path = _write_jsonl(tmp_path / f"{eval_mode}.jsonl", payload)

    records = load_workload_jsonl(path)

    assert len(records) == 1
    assert records[0].workload.eval_mode == eval_mode


def test_checked_in_correctness_only_policy_still_requires_uuid4(tmp_path):
    payload = _payload(uuid="00000000-0000-1000-8000-000000000000", eval_mode="correctness_only")
    path = _write_jsonl(tmp_path / "correctness_only.jsonl", payload)

    with pytest.raises(KernelWorkloadError, match="UUIDv4"):
        load_workload_jsonl(path)


def test_materialize_resolves_definition_expression_axes(tmp_path):
    torch = pytest.importorskip("torch")
    definition = _definition()
    definition["axes"]["NK"] = {"type": "expr", "expression": "N * K // 2"}
    definition["inputs"]["x"]["shape"] = ["NK"]
    definition["constraints"] = ["NK == N * 2"]
    path = _write_jsonl(tmp_path / "expr.jsonl", _payload())
    record = load_workload_jsonl(path)[0]

    validate_workload_against_definition(record, definition, tmp_path / "definition.json")
    axes, inputs = materialize_workload_inputs(record, definition, torch=torch, device="cpu")

    assert axes["NK"] == 16
    assert tuple(inputs["x"].shape) == (16,)


def test_expression_axis_constraint_violation_is_rejected(tmp_path):
    definition = _definition()
    definition["axes"]["NK"] = {"type": "expr", "expression": "N * K"}
    definition["constraints"] = ["NK <= 16"]
    path = _write_jsonl(tmp_path / "expr.jsonl", _payload())
    record = load_workload_jsonl(path)[0]

    with pytest.raises(KernelWorkloadError, match="NK <= 16"):
        validate_workload_against_definition(record, definition, tmp_path / "definition.json")


def test_checked_in_workload_policy_allows_relaxed_matched_ratio(tmp_path):
    payload = _payload()
    payload["tolerance"]["required_matched_ratio"] = 0.98
    path = _write_jsonl(tmp_path / "relaxed.jsonl", payload)

    records = load_workload_jsonl(path)

    assert len(records) == 1
    assert records[0].workload.tolerance.required_matched_ratio == 0.98


@pytest.mark.parametrize("ratio", [True, False])
def test_checked_in_workload_policy_rejects_boolean_matched_ratio(tmp_path, ratio):
    payload = _payload()
    payload["tolerance"]["required_matched_ratio"] = ratio
    path = _write_jsonl(tmp_path / "boolean.jsonl", payload)

    with pytest.raises(KernelWorkloadError, match="booleans and non-numeric values are rejected"):
        load_workload_jsonl(path)


def test_load_workload_jsonl_accepts_custom_inputs_kwargs_and_benchmark_cudagraph(tmp_path):
    payload = _payload(custom_inputs_kwargs={"block": 4, "mode": "dense"}, benchmark_cudagraph=True)

    record = load_workload_jsonl(_write_jsonl(tmp_path / "fields.jsonl", payload))[0]

    assert record.workload.custom_inputs_kwargs == {"block": 4, "mode": "dense"}
    assert record.workload.benchmark_cudagraph is True


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"benchmark_cudagraph": "true"}, "Workload schema invalid"),
        ({"benchmark_cudagraph": 1}, "Workload schema invalid"),
        ({"custom_inputs_kwargs": [1]}, "Workload schema invalid"),
        ({"custom_inputs_kwargs": {"scale": float("inf")}}, "numbers must be finite"),
        ({"custom_inputs_kwargs": {"blob": "x" * (256 * 1024)}}, "Workload schema invalid"),
    ],
)
def test_load_workload_jsonl_rejects_invalid_custom_inputs_kwargs_and_benchmark_cudagraph(tmp_path, updates, match):
    path = _write_jsonl(tmp_path / "invalid.jsonl", _payload(**updates))

    with pytest.raises(KernelWorkloadError, match=match):
        load_workload_jsonl(path)


def _dynamic_axis_definition() -> dict[str, Any]:
    definition = _definition()
    definition["axes"]["NMAX"] = {"type": "const", "value": 64}
    definition["constraints"] = ["N >= K", "N == 8", "enabled is False"]
    return definition


def test_correctness_only_dynamic_axes_validate_against_definition(tmp_path):
    payload = _payload(axes={"N": {"min": "K", "max": "NMAX", "multiple_of": 4}}, eval_mode="correctness_only")
    record = load_workload_jsonl(_write_jsonl(tmp_path / "dynamic.jsonl", payload))[0]

    validate_workload_against_definition(record, _dynamic_axis_definition(), tmp_path / "definition.json")


@pytest.mark.parametrize(
    ("axis", "match"),
    [
        ({"min": "M", "max": 8}, "'M', which is not a const Definition axis"),
        ({"min": 4, "max": "N"}, "'N', which is not a const Definition axis"),
        ({"min": 16, "max": "K"}, "min must be <= max"),
        ({"min": 5, "max": 7, "multiple_of": 8}, "no values divisible by multiple_of=8"),
        ({"min": 0, "max": 8}, "bounds must be positive|Workload schema invalid"),
    ],
)
def test_correctness_only_dynamic_axes_reject_unresolvable_ranges(tmp_path, axis, match):
    payload = _payload(axes={"N": axis}, eval_mode="correctness_only")

    with pytest.raises(KernelWorkloadError, match=match):
        record = load_workload_jsonl(_write_jsonl(tmp_path / "dynamic.jsonl", payload))[0]
        validate_workload_against_definition(record, _dynamic_axis_definition(), tmp_path / "definition.json")


def test_correctness_only_dynamic_axes_reject_safetensors_inputs(tmp_path):
    payload = _payload(axes={"N": {"min": 4, "max": 8}}, eval_mode="correctness_only")
    payload["inputs"]["x"] = {"type": "safetensors", "path": "x.safetensors", "tensor_key": "x"}
    record = load_workload_jsonl(_write_jsonl(tmp_path / "dynamic.jsonl", payload))[0]

    with pytest.raises(KernelWorkloadError, match="cannot be combined with safetensors inputs"):
        validate_workload_against_definition(record, _dynamic_axis_definition(), tmp_path / "definition.json")


def test_materialize_rejects_dynamic_axes_explicitly(tmp_path):
    torch = pytest.importorskip("torch")
    payload = _payload(axes={"N": {"min": 4, "max": 8}}, eval_mode="correctness_only")
    record = load_workload_jsonl(_write_jsonl(tmp_path / "dynamic.jsonl", payload))[0]

    with pytest.raises(KernelWorkloadError, match="does not yet support dynamic axes"):
        materialize_workload_inputs(record, _dynamic_axis_definition(), torch=torch, device="cpu")


def _expression_axis_definition(constraints: list[str]) -> dict[str, Any]:
    definition = _definition()
    definition["axes"]["HALF"] = {"type": "expr", "expression": "N // 2"}
    definition["axes"]["KK"] = {"type": "expr", "expression": "K * 2"}
    definition["constraints"] = constraints
    return definition


def test_unresolvable_expression_axis_is_reported_instead_of_skipping_constraints(tmp_path):
    definition = _expression_axis_definition(["GROUPS >= 1"])
    definition["axes"]["GROUPS"] = {"type": "expr", "expression": "K // (N - 8)"}
    record = load_workload_jsonl(_write_jsonl(tmp_path / "op.jsonl", _payload()))[0]

    with pytest.raises(KernelWorkloadError, match="Definition 'op' cannot resolve expression axis 'GROUPS'"):
        validate_workload_against_definition(record, definition, tmp_path / "op.json")


@pytest.mark.parametrize(
    ("constraints", "match"),
    [
        (["KK == 8", "HALF >= 100"], None),
        (["KK == 6"], "violates Definition constraint: KK == 6"),
    ],
)
def test_dynamic_axes_skip_only_dependent_expression_axes(tmp_path, constraints, match):
    payload = _payload(axes={"N": {"min": 4, "max": 8}}, eval_mode="correctness_only")
    record = load_workload_jsonl(_write_jsonl(tmp_path / "dynamic.jsonl", payload))[0]
    definition = _expression_axis_definition(constraints)

    if match is None:
        validate_workload_against_definition(record, definition, tmp_path / "op.json")
    else:
        with pytest.raises(KernelWorkloadError, match=match):
            validate_workload_against_definition(record, definition, tmp_path / "op.json")


def test_validate_workload_against_definition_accepts_exact_axes_inputs_and_constraint(tmp_path):
    record = load_workload_jsonl(_write_jsonl(tmp_path / "op.jsonl", _payload()))[0]

    validate_workload_against_definition(record, _definition(), tmp_path / "op.json")


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"axes": {}}, "axes mismatch"),
        ({"axes": {"N": 8, "K": 4}}, "unknown_or_constant"),
        ({"inputs": {"x": {"type": "random"}}}, "inputs mismatch"),
        (
            {
                "inputs": {
                    "x": {"type": "scalar", "value": 1.0},
                    "scale": {"type": "scalar", "value": 1.0},
                    "enabled": {"type": "scalar", "value": False},
                    "bias": {"type": "null"},
                }
            },
            "scalar descriptor for tensor",
        ),
        (
            {
                "inputs": {
                    "x": {"type": "random"},
                    "scale": {"type": "random"},
                    "enabled": {"type": "scalar", "value": False},
                    "bias": {"type": "null"},
                }
            },
            "tensor descriptor for scalar",
        ),
        (
            {
                "inputs": {
                    "x": {"type": "random"},
                    "scale": {"type": "scalar", "value": True},
                    "enabled": {"type": "scalar", "value": False},
                    "bias": {"type": "null"},
                }
            },
            "incompatible with scalar dtype",
        ),
        (
            {
                "inputs": {
                    "x": {"type": "random"},
                    "scale": {"type": "scalar", "value": 1.0},
                    "enabled": {"type": "scalar", "value": True},
                    "bias": {"type": "null"},
                }
            },
            "violates Definition constraint",
        ),
    ],
)
def test_validate_workload_against_definition_rejects_contract_mismatch(tmp_path, updates, match):
    record = load_workload_jsonl(_write_jsonl(tmp_path / "op.jsonl", _payload(**updates)))[0]

    with pytest.raises(KernelWorkloadError, match=match):
        validate_workload_against_definition(record, _definition(), tmp_path / "op.json")


def test_null_descriptor_requires_explicit_none_reference_default(tmp_path):
    definition = _definition()
    definition["reference"] = "def run(x, scale, enabled, bias):\n    return x\n"
    record = load_workload_jsonl(_write_jsonl(tmp_path / "op.jsonl", _payload()))[0]

    with pytest.raises(KernelWorkloadError, match="explicit None default"):
        validate_workload_against_definition(record, definition, tmp_path / "op.json")


def test_validate_workload_against_definition_checks_safetensors_asset_path(tmp_path):
    payload = _payload(
        axes={"N": 2},
        inputs={"x": {"type": "safetensors", "path": "input.safetensors", "tensor_key": "x"}},
    )
    record = load_workload_jsonl(_write_jsonl(tmp_path / "op.jsonl", payload))[0]
    definition = {
        "axes": {"N": {"type": "var"}, "K": {"type": "const", "value": 4}},
        "inputs": {"x": {"shape": ["N", "K"], "dtype": "float32"}},
    }

    with pytest.raises(KernelWorkloadError, match="safetensors file does not exist"):
        validate_workload_against_definition(record, definition, tmp_path / "op.json")

    asset = tmp_path / "input.safetensors"
    asset.touch()
    validate_workload_against_definition(record, definition, tmp_path / "op.json")


def test_materialize_workload_inputs_uses_definition_shape_dtype_and_literals(tmp_path):
    torch = pytest.importorskip("torch")
    record = load_workload_jsonl(_write_jsonl(tmp_path / "op.jsonl", _payload()))[0]

    torch.manual_seed(2026)
    axes, inputs = materialize_workload_inputs(record, _definition(), torch=torch, device=torch.device("cpu"))

    assert axes == {"K": 4, "N": 8}
    assert inputs["x"].shape == (8, 4)
    assert inputs["x"].dtype == torch.float32
    assert inputs["scale"] == 1.0
    assert inputs["enabled"] is False
    assert inputs["bias"] is None


def test_materialize_workload_inputs_loads_safetensors_relative_to_workload(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    asset = tmp_path / "input.safetensors"
    safetensors.save_file({"x": tensor}, asset)
    payload = _payload(
        axes={"N": 2},
        inputs={"x": {"type": "safetensors", "path": "input.safetensors", "tensor_key": "x"}},
    )
    record = load_workload_jsonl(_write_jsonl(tmp_path / "op.jsonl", payload))[0]
    definition = {
        "axes": {"N": {"type": "var"}, "K": {"type": "const", "value": 4}},
        "inputs": {"x": {"shape": ["N", "K"], "dtype": "float32"}},
    }

    _, inputs = materialize_workload_inputs(record, definition, torch=torch, device=torch.device("cpu"))

    assert torch.equal(inputs["x"], tensor)
    assert inputs["x"].is_contiguous()


def test_custom_correctness_kwargs_require_definition_entrypoint(tmp_path):
    record = load_workload_jsonl(
        _write_jsonl(tmp_path / "op.jsonl", _payload(custom_correctness_kwargs={"threshold": 1}))
    )[0]
    definition = _definition()

    with pytest.raises(KernelWorkloadError, match="custom_correctness_kwargs require"):
        validate_workload_against_definition(record, definition, tmp_path / "op.json")

    definition["custom_correctness_entrypoint"] = "check"
    validate_workload_against_definition(record, definition, tmp_path / "op.json")


def test_custom_inputs_kwargs_require_custom_inputs(tmp_path):
    definition = _definition()
    definition["custom_inputs_entrypoint"] = "make_inputs"
    rows = _write_jsonl(
        tmp_path / "op.jsonl",
        _payload(custom_inputs_kwargs={"block": 4}),
        _payload(
            uuid="5f0b8f1e-2c55-4c43-9a8e-6a0f3b1d2e47",
            inputs={name: {"type": "custom"} for name in definition["inputs"]},
            custom_inputs_kwargs={"block": 4},
        ),
    )
    plain_row, custom_row = load_workload_jsonl(rows)

    with pytest.raises(KernelWorkloadError, match="custom_inputs_kwargs require custom inputs"):
        validate_workload_against_definition(plain_row, definition, tmp_path / "op.json")

    validate_workload_against_definition(custom_row, definition, tmp_path / "op.json")


def test_materialize_random_input_with_integer_literal_dimension(tmp_path):
    torch = pytest.importorskip("torch")
    record = load_workload_jsonl(
        _write_jsonl(tmp_path / "op.jsonl", _payload(axes={"N": 2}, inputs={"x": {"type": "random"}}))
    )[0]
    definition = {"axes": {"N": {"type": "var"}}, "inputs": {"x": {"shape": ["N", "4"], "dtype": "float32"}}}

    _, inputs = materialize_workload_inputs(record, definition, torch=torch, device=torch.device("cpu"))

    assert inputs["x"].shape == (2, 4)


def test_materialize_safetensors_input_with_integer_literal_dimension(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    safetensors.save_file({"x": tensor}, tmp_path / "input.safetensors")
    payload = _payload(
        axes={"N": 2},
        inputs={"x": {"type": "safetensors", "path": "input.safetensors", "tensor_key": "x"}},
    )
    record = load_workload_jsonl(_write_jsonl(tmp_path / "op.jsonl", payload))[0]
    definition = {"axes": {"N": {"type": "var"}}, "inputs": {"x": {"shape": ["N", "4"], "dtype": "float32"}}}

    _, inputs = materialize_workload_inputs(record, definition, torch=torch, device=torch.device("cpu"))

    assert torch.equal(inputs["x"], tensor)


def test_materialize_workload_inputs_rejects_non_basename_safetensors_path(tmp_path):
    torch = pytest.importorskip("torch")
    payload = _payload(
        axes={"N": 2},
        inputs={"x": {"type": "safetensors", "path": "../escape.safetensors", "tensor_key": "x"}},
    )
    record = load_workload_jsonl(_write_jsonl(tmp_path / "owner/op.jsonl", payload))[0]
    definition = {
        "axes": {"N": {"type": "var"}, "K": {"type": "const", "value": 4}},
        "inputs": {"x": {"shape": ["N", "K"], "dtype": "float32"}},
    }

    with pytest.raises(KernelWorkloadError, match="must be a .safetensors basename"):
        materialize_workload_inputs(record, definition, torch=torch, device=torch.device("cpu"))


def test_validate_workload_catalog_checks_completeness_and_uuid_uniqueness(tmp_path):
    inventory = tmp_path / "src/tilegym/transformers/example"
    definition = _definition()
    _write_json(inventory / "kernel_definitions/op.json", definition)
    _write_json(inventory / "kernel_solutions/op.json", {"name": "op"})
    workload = _write_jsonl(inventory / "kernel_workloads/op/workload.jsonl", _payload())

    validate_workload_catalog(tmp_path, require_complete=True)

    _write_json(inventory / "kernel_definitions/second.json", definition | {"name": "second"})
    _write_json(inventory / "kernel_solutions/second.json", {"name": "second"})
    with pytest.raises(KernelWorkloadError, match="missing mirrored Workloads"):
        validate_workload_catalog(tmp_path, require_complete=True)

    _write_jsonl(inventory / "kernel_workloads/second/workload.jsonl", _payload())
    with pytest.raises(KernelWorkloadError, match="duplicate Workload UUID"):
        validate_workload_catalog(tmp_path, require_complete=True)

    workload.unlink()
    validate_workload_catalog(tmp_path, require_complete=False)


def test_validate_workload_catalog_rejects_stale_workload(tmp_path):
    inventory = tmp_path / "src/tilegym/transformers/example"
    (inventory / "kernel_definitions").mkdir(parents=True)
    (inventory / "kernel_solutions").mkdir()
    _write_jsonl(inventory / "kernel_workloads/stale/workload.jsonl", _payload())

    with pytest.raises(KernelWorkloadError, match="Stale Workload"):
        validate_workload_catalog(tmp_path, require_complete=False)


def test_validate_workload_catalog_rejects_orphan_safetensors_asset(tmp_path):
    inventory = tmp_path / "src/tilegym/transformers/example"
    definition = _definition()
    _write_json(inventory / "kernel_definitions/op.json", definition)
    _write_json(inventory / "kernel_solutions/op.json", {"name": "op"})
    workload = _write_jsonl(inventory / "kernel_workloads/op/workload.jsonl", _payload())
    del workload
    (inventory / "kernel_workloads/op/orphan.safetensors").touch()

    with pytest.raises(KernelWorkloadError, match="Orphan safetensors"):
        validate_workload_catalog(tmp_path, require_complete=True)


def test_checked_in_workload_catalog_is_complete_and_valid():
    validate_workload_catalog(REPO_ROOT, require_complete=True)
