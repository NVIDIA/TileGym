# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
from typing import Any

import pytest
from pydantic import ValidationError

from tilegym.kernel_inventory import _definition_schema_compat as compat
from tilegym.kernel_inventory.generation import TileGymDefinition
from tilegym.kernel_inventory.generation import axis_const
from tilegym.kernel_inventory.generation import axis_expr
from tilegym.kernel_inventory.generation import axis_var
from tilegym.kernel_inventory.generation import definition_to_tilegym_json
from tilegym.kernel_inventory.generation import tensor
from tilegym.kernel_inventory.schema import TILEGYM_DEFINITION_EXTENSION_FIELDS
from tilegym.kernel_inventory.schema import definition_model_validate
from tilegym.kernel_inventory.schema import resolve_definition_axes

_REFERENCE = (
    "import torch\n\n"
    "def run(x, scale):\n"
    "    out = x[:, :HALF] * scale\n"
    "    return out\n\n"
    "def make_inputs(axes_and_scalars, device, seed):\n"
    "    return {}\n\n"
    "def check(ref_outputs, user_outputs, *, threshold=0.5):\n"
    "    return {name: {'passed': True} for name in ref_outputs}\n"
)

_BASE_DEFINITION: dict[str, Any] = {
    "name": "expr_op",
    "op_type": "elementwise",
    "axes": {
        "M": {"type": "var"},
        "N": {"type": "const", "value": 8},
        "HALF": {"type": "expr", "expression": "N // 2"},
        "MH": {"type": "expr", "expression": "M * HALF + 1"},
    },
    "inputs": {
        "x": {"shape": ["M", "N"], "dtype": "float32"},
        "scale": {"shape": None, "dtype": "float32"},
    },
    "outputs": {"out": {"shape": ["M", "HALF"], "dtype": "float32"}},
    "reference": _REFERENCE,
    "constraints": ["M >= 1", "HALF * 2 == N", "MH > M"],
    "custom_inputs_entrypoint": "make_inputs",
    "custom_correctness_entrypoint": "check",
}


def _definition(**updates: Any) -> dict[str, Any]:
    definition = copy.deepcopy(_BASE_DEFINITION)
    for key, value in updates.items():
        if value is None:
            definition.pop(key, None)
        else:
            definition[key] = value
    return definition


_ACCEPTED_DEFINITIONS = [
    pytest.param(_BASE_DEFINITION, id="expr-constraints-entrypoints"),
    pytest.param(_definition(custom_inputs_entrypoint=None, custom_correctness_entrypoint=None), id="no-entrypoints"),
    pytest.param(_definition(constraints=None), id="no-constraints"),
    pytest.param(
        _definition(axes={**_BASE_DEFINITION["axes"], "UNUSED": {"type": "expr", "expression": "M - 1"}}),
        id="expr-not-used-by-tensors",
    ),
    *(
        pytest.param(_definition(outputs={"out": {"shape": ["M", dim], "dtype": "float32"}}), id=f"literal-dim-{dim}")
        for dim in ("4", "0", "007")
    ),
    pytest.param(
        _definition(inputs={**_BASE_DEFINITION["inputs"], "x": {"shape": ["M", "8"], "dtype": "float32"}}),
        id="literal-input-dim",
    ),
]

_REJECTED_DEFINITIONS = [
    pytest.param(_definition(custom_correctness_entrypoint="missing"), id="correctness-entrypoint-missing"),
    pytest.param(_definition(custom_inputs_entrypoint="missing"), id="inputs-entrypoint-missing"),
    pytest.param(_definition(custom_correctness_entrypoint="not-an-identifier"), id="correctness-entrypoint-name"),
    pytest.param(_definition(custom_inputs_entrypoint=""), id="inputs-entrypoint-empty"),
    pytest.param(_definition(custom_correctness_entrypoint="make_inputs"), id="correctness-entrypoint-arity"),
    pytest.param(
        _definition(axes={**_BASE_DEFINITION["axes"], "HALF": {"type": "expr", "expression": ""}}),
        id="empty-expression",
    ),
    pytest.param(
        _definition(axes={**_BASE_DEFINITION["axes"], "HALF": {"type": "expr"}}),
        id="missing-expression",
    ),
    pytest.param(
        _definition(outputs={"out": {"shape": ["M", "UNKNOWN"], "dtype": "float32"}}),
        id="undefined-axis-reference",
    ),
    *(
        pytest.param(
            _definition(outputs={"out": {"shape": ["M", dim], "dtype": "float32"}}), id=f"non-literal-dim-{dim}"
        )
        for dim in ("-4", "+4", "4.0", " 4", "1e3")
    ),
    pytest.param(_definition(constraints=["M >="]), id="constraint-syntax"),
]


@pytest.mark.parametrize("payload", _ACCEPTED_DEFINITIONS)
def test_public_compat_definition_accepts_extended_fields(payload):
    definition = compat.Definition.model_validate(payload)

    assert definition.model_dump(mode="json")["custom_correctness_entrypoint"] == payload.get(
        "custom_correctness_entrypoint"
    )


@pytest.mark.parametrize("payload", _REJECTED_DEFINITIONS)
def test_public_compat_definition_rejects_invalid_extended_fields(payload):
    with pytest.raises(ValidationError):
        compat.Definition.model_validate(payload)


def test_public_compat_definition_resolves_expression_axes():
    definition = compat.Definition.model_validate(_BASE_DEFINITION)

    assert list(definition.expr_axes) == ["HALF", "MH"]
    assert definition.get_resolved_axes_values({"M": 3}) == {"N": 8, "M": 3, "HALF": 4, "MH": 13}
    assert definition.get_input_shapes({"M": 3}) == [(3, 8), None]
    assert definition.get_output_shapes({"M": 3}) == [(3, 4)]
    assert definition.get_axes_values([(3, 8), None]) == {"M": 3}


@pytest.mark.parametrize(
    ("expression", "variables", "expected"),
    [
        ("N // 2", {"N": 9}, 4),
        ("(M + 1) * -2 % 7", {"M": 3}, 6),
        ("2 ** K", {"K": 5}, 32),
        ("M / 2", {"M": 6}, 3),
        ("(K + 63) // 64 * 64", {"K": 100}, 128),
        pytest.param("2 ** 64", {}, 2**64, id="pow-65-bits"),
        pytest.param("(2 ** 63) ** 64", {}, 2**4032, id="pow-4033-bits"),
    ],
)
def test_public_compat_shape_expression_evaluation(expression, variables, expected):
    assert compat.resolve_shape_expression(expression, variables) == expected


@pytest.mark.parametrize(
    ("expression", "variables", "error"),
    [
        ("M / 4", {"M": 6}, TypeError),
        ("M // 0", {"M": 6}, ValueError),
        ("UNKNOWN + 1", {}, ValueError),
        ("max(M, 1)", {"M": 1}, TypeError),
        ("M & 1", {"M": 1}, TypeError),
        ("2 ** 65", {}, ValueError),
        pytest.param("(2 ** 64) ** 64", {}, ValueError, id="pow-4097-bits"),
        pytest.param("(((9 ** 64) ** 64) ** 64) ** 64", {}, ValueError, id="pow-nested-9"),
    ],
)
def test_public_compat_shape_expression_rejections(expression, variables, error):
    with pytest.raises(error):
        compat.resolve_shape_expression(expression, variables)


def test_resolve_definition_axes_binds_payload_expressions_in_order():
    assert resolve_definition_axes(_BASE_DEFINITION, {"M": 2}) == {"N": 8, "M": 2, "HALF": 4, "MH": 9}


def test_payload_and_model_axis_resolution_agree_and_name_the_failing_axis():
    model = compat.Definition.model_validate(_BASE_DEFINITION)
    assert resolve_definition_axes(_BASE_DEFINITION, {"M": 3}) == model.get_resolved_axes_values({"M": 3})

    broken = _definition(axes={**_BASE_DEFINITION["axes"], "HALF": {"type": "expr", "expression": "N // (M - 3)"}})
    broken_model = compat.Definition.model_validate(broken)
    for resolve in (
        lambda: resolve_definition_axes(broken, {"M": 3}),
        lambda: broken_model.get_resolved_axes_values({"M": 3}),
    ):
        with pytest.raises(ValueError, match="expression axis 'HALF' = 'N // \\(M - 3\\)': Division by zero"):
            resolve()


def test_resolve_axes_values_leaves_out_expressions_on_unresolved_axes():
    resolved = compat.resolve_axes_values(
        {"N": 8},
        {},
        {"HALF": "N // 2", "MH": "M * HALF + 1", "MH1": "MH + 1", "N2": "N * 2"},
        unresolved_axes={"M"},
    )

    assert resolved == {"N": 8, "HALF": 4, "N2": 16}


def test_custom_inputs_entrypoint_without_seed_warns():
    reference = _REFERENCE.replace("def make_inputs(axes_and_scalars, device, seed)", "def make_inputs(axes, device)")

    with pytest.warns(UserWarning, match="seed"):
        compat.Definition.model_validate(_definition(reference=reference))


def test_tilegym_definition_round_trips_extended_fields():
    definition = TileGymDefinition.model_validate(_definition(tags=["status:verified"]))
    data = definition_to_tilegym_json(definition)

    assert data["axes"]["HALF"] == {"type": "expr", "expression": "N // 2"}
    assert data["custom_inputs_entrypoint"] == "make_inputs"
    assert data["custom_correctness_entrypoint"] == "check"
    assert data["constraints"] == _BASE_DEFINITION["constraints"]


def test_generation_helpers_build_expression_axes():
    definition = TileGymDefinition(
        name="helper_op",
        op_type="elementwise",
        axes={"M": axis_var(), "N": axis_const(8), "HALF": axis_expr("N // 2")},
        inputs={"x": tensor(["M", "N"], "float32"), "scale": tensor(None, "float32")},
        outputs={"out": tensor(["M", "HALF"], "float32")},
        reference=_REFERENCE,
    )

    assert definition.get_output_shapes({"M": 2}) == [(2, 4)]


def test_tilegym_definition_accepts_integer_literal_dimensions():
    payload = _definition(outputs={"out": {"shape": ["M", "4"], "dtype": "float32"}})

    assert TileGymDefinition.model_validate(payload).get_output_shapes({"M": 3}) == [(3, 4)]
    with pytest.raises(ValidationError, match='undefined axis "-4"'):
        TileGymDefinition.model_validate(_definition(outputs={"out": {"shape": ["M", "-4"], "dtype": "float32"}}))


def test_definition_validation_ignores_tilegym_extension_fields():
    payload = _definition(tags=["status:verified"], include=["leaf_op"])

    definition = definition_model_validate(payload)

    assert definition.custom_correctness_entrypoint == "check"
    assert "include" in TILEGYM_DEFINITION_EXTENSION_FIELDS
