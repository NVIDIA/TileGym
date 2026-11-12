# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Stable schema boundary for kernel-inventory Workloads and Definitions.

Import Workload and Definition symbols from this module rather than from the underlying compatibility modules.
"""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from tilegym.kernel_inventory._definition_schema_compat import resolve_axes_values

USING_CANONICAL_WORKLOAD_SCHEMA = False

if not USING_CANONICAL_WORKLOAD_SCHEMA:
    from tilegym.kernel_inventory._definition_schema_compat import AxisConst
    from tilegym.kernel_inventory._definition_schema_compat import AxisExpr
    from tilegym.kernel_inventory._definition_schema_compat import AxisVar
    from tilegym.kernel_inventory._definition_schema_compat import Definition
    from tilegym.kernel_inventory._definition_schema_compat import TensorSpec
    from tilegym.kernel_inventory._definition_schema_compat import resolve_shape_expression
    from tilegym.kernel_inventory._workload_schema_compat import CustomInput
    from tilegym.kernel_inventory._workload_schema_compat import EvalMode
    from tilegym.kernel_inventory._workload_schema_compat import InputSpec
    from tilegym.kernel_inventory._workload_schema_compat import NullInput
    from tilegym.kernel_inventory._workload_schema_compat import RandomInput
    from tilegym.kernel_inventory._workload_schema_compat import SafetensorsInput
    from tilegym.kernel_inventory._workload_schema_compat import SafetensorsShard
    from tilegym.kernel_inventory._workload_schema_compat import ScalarInput
    from tilegym.kernel_inventory._workload_schema_compat import StringInput
    from tilegym.kernel_inventory._workload_schema_compat import ToleranceSpec
    from tilegym.kernel_inventory._workload_schema_compat import Workload


def workload_model_validate(payload: Any) -> Workload:
    """Validate one Workload payload."""
    return Workload.model_validate(payload)


def workload_model_dump(workload: Workload) -> dict[str, Any]:
    """Return normalized JSON-compatible Workload data."""
    return workload.model_dump(mode="json")


# TileGym-only Definition fields that the Definition model does not declare.
TILEGYM_DEFINITION_EXTENSION_FIELDS = frozenset(
    name for name in ("include", "tags") if name not in Definition.model_fields
)


def definition_model_validate(payload: dict[str, Any]) -> Definition:
    """Validate one Definition payload, ignoring TileGym-only fields."""
    return Definition.model_validate(
        {key: value for key, value in payload.items() if key not in TILEGYM_DEFINITION_EXTENSION_FIELDS}
    )


def resolve_definition_axes(
    definition: dict[str, Any],
    var_axes_values: dict[str, Any],
    *,
    unresolved_axes: Collection[str] = (),
) -> dict[str, int]:
    """Bind const and var axes of a Definition payload, then evaluate expression axes in order.

    Expression axes that depend on ``unresolved_axes`` are left out.
    """
    axes = definition.get("axes", {})
    return resolve_axes_values(
        {name: spec["value"] for name, spec in axes.items() if spec.get("type") == "const"},
        var_axes_values,
        {name: spec["expression"] for name, spec in axes.items() if spec.get("type") == "expr"},
        resolve_shape_expression,
        unresolved_axes=unresolved_axes,
    )


def resolve_tensor_shape(shape: Sequence[str] | None, axes: Mapping[str, int]) -> tuple[int, ...]:
    """Resolve TensorSpec dimensions: all-digit tokens are integer literals, others name axes."""
    if shape is None:
        return ()
    return tuple(int(dimension) if dimension.isdigit() else axes[dimension] for dimension in shape)


__all__ = [
    "TILEGYM_DEFINITION_EXTENSION_FIELDS",
    "USING_CANONICAL_WORKLOAD_SCHEMA",
    "AxisConst",
    "AxisExpr",
    "AxisVar",
    "CustomInput",
    "Definition",
    "EvalMode",
    "InputSpec",
    "NullInput",
    "RandomInput",
    "SafetensorsInput",
    "SafetensorsShard",
    "ScalarInput",
    "StringInput",
    "TensorSpec",
    "ToleranceSpec",
    "Workload",
    "definition_model_validate",
    "resolve_definition_axes",
    "resolve_shape_expression",
    "resolve_tensor_shape",
    "workload_model_dump",
    "workload_model_validate",
]
