# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tilegym.kernel_inventory import _workload_schema_compat as compat
from tilegym.kernel_inventory.schema import USING_CANONICAL_WORKLOAD_SCHEMA
from tilegym.kernel_inventory.schema import workload_model_dump
from tilegym.kernel_inventory.schema import workload_model_validate

_BASE_WORKLOAD: dict[str, Any] = {
    "uuid": "52bedf2e-bdbc-4618-9f38-a592a4de71a9",
    "axes": {"B": 2},
    "inputs": {"x": {"type": "random"}},
    "tolerance": {
        "max_atol": 0.02,
        "max_rtol": 0.02,
        "required_matched_ratio": 1.0,
        "max_error_cap": None,
        "allow_negative_inf": False,
    },
    "eval_mode": "full",
}


def _workload(**updates: Any) -> dict[str, Any]:
    workload = copy.deepcopy(_BASE_WORKLOAD)
    workload.update(updates)
    return workload


_ACCEPTED_WORKLOADS = [
    pytest.param(_BASE_WORKLOAD, id="random-explicit-policy"),
    pytest.param(
        _workload(
            inputs={
                "integer": {"type": "scalar", "value": 1},
                "floating": {"type": "scalar", "value": 0.5},
                "boolean": {"type": "scalar", "value": True},
                "optional": {"type": "null"},
                "mapping": {"type": "string", "value": ""},
            }
        ),
        id="literal-inputs",
    ),
    pytest.param(
        _workload(inputs={"x": {"type": "safetensors", "path": "inputs.safetensors", "tensor_key": "x"}}),
        id="replicated-safetensors",
    ),
    pytest.param(
        _workload(
            inputs={
                "x": {
                    "type": "safetensors",
                    "shards": [
                        {"path": "rank0.safetensors", "tensor_key": "x"},
                        {"path": "rank1.safetensors", "tensor_key": "x"},
                    ],
                }
            }
        ),
        id="sharded-safetensors",
    ),
    pytest.param(_workload(inputs={"x": {"type": "custom"}}), id="custom"),
    pytest.param(_workload(weight=3.5), id="positive-weight"),
    pytest.param(
        {
            "uuid": "52bedf2e-bdbc-4618-9f38-a592a4de71aa",
            "axes": {"B": 2},
            "inputs": {"x": {"type": "random"}},
        },
        id="schema-defaults",
    ),
    pytest.param(
        _workload(
            axes={"B": {"min": 1, "max": 8, "multiple_of": 2, "sampling_strategy": "linear"}},
            inputs={"x": {"type": "custom"}},
            eval_mode="correctness_only",
        ),
        id="dynamic-correctness-only",
    ),
    pytest.param(
        _workload(
            axes={"B": {"min": 1, "max": 8}},
            inputs={"x": {"type": "safetensors", "path": "x.safetensors", "tensor_key": "x"}},
            eval_mode="correctness_only",
        ),
        id="schema-allows-dynamic-safetensors",
    ),
    pytest.param(
        _workload(inputs={"x": {"type": "scalar", "value": float("inf")}}),
        id="schema-allows-nonfinite-scalar",
    ),
]


_REJECTED_WORKLOADS = [
    pytest.param(_workload(inputs={"x": {"type": "custom"}, "y": {"type": "random"}}), id="mixed-custom"),
    pytest.param(_workload(inputs={"x": {"type": "safetensors", "path": "x.safetensors"}}), id="missing-key"),
    pytest.param(
        _workload(
            inputs={
                "x": {
                    "type": "safetensors",
                    "path": "x.safetensors",
                    "tensor_key": "x",
                    "shards": [{"path": "rank0.safetensors", "tensor_key": "x"}],
                }
            }
        ),
        id="replicated-and-sharded",
    ),
    pytest.param(_workload(inputs={"x": {"type": "safetensors", "shards": []}}), id="empty-shards"),
    pytest.param(_workload(axes={"B": -1}), id="negative-axis"),
    pytest.param(_workload(uuid=""), id="empty-uuid"),
    pytest.param(_workload(eval_mode="invalid"), id="invalid-eval-mode"),
    pytest.param(_workload(axes={"B": {"min": 1, "max": 8}}), id="dynamic-full"),
    pytest.param(_workload(weight=0.0), id="zero-weight"),
    pytest.param(_workload(weight=float("inf")), id="infinite-weight"),
    pytest.param(_workload(custom_correctness_kwargs={"threshold": float("nan")}), id="nonfinite-kwargs"),
]


@pytest.mark.parametrize("payload", _ACCEPTED_WORKLOADS)
def test_public_compat_workload_accepts_documented_contract(payload):
    workload = compat.Workload.model_validate(payload)
    assert workload.model_dump(mode="json")["uuid"] == payload["uuid"]


@pytest.mark.parametrize("payload", _REJECTED_WORKLOADS)
def test_public_compat_workload_rejects_invalid_contract(payload):
    with pytest.raises(ValidationError):
        compat.Workload.model_validate(payload)


def test_schema_adapter_uses_fallback_when_canonical_dependency_is_absent():
    if USING_CANONICAL_WORKLOAD_SCHEMA:
        pytest.skip("internal environment has the canonical schema package")

    workload = workload_model_validate(_BASE_WORKLOAD)

    assert isinstance(workload, compat.Workload)
    assert workload_model_dump(workload)["eval_mode"] == "full"
