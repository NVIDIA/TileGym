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
        (lambda payload: payload.update(eval_mode="correctness_only"), "eval_mode must be 'full'"),
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
