# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Definition models with expression axes and custom input/correctness entrypoints.

FlashInfer-Bench 0.1.2 Definitions lack expression axes, integer-literal shape
dimensions and the custom input/correctness entrypoints. This module adds them
on top of the FIB models.
"""

from __future__ import annotations

import ast
import operator
import warnings
from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Mapping
from functools import cached_property
from typing import Annotated
from typing import Any
from typing import Literal
from typing import Optional
from typing import TypeAlias

from flashinfer_bench.data import AxisConst
from flashinfer_bench.data import AxisVar
from flashinfer_bench.data import Definition as _FibDefinition
from flashinfer_bench.data import TensorSpec
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

_MAX_EXPR_LEN = 1024
_MAX_AST_DEPTH = 64
_MAX_POW_EXPONENT = 64
_MAX_POW_RESULT_BITS = 4096


class AxisExpr(BaseModel):
    """Axis whose value is an arithmetic expression over other axes."""

    type: Literal["expr"] = "expr"
    expression: str = Field(min_length=1)
    description: Optional[str] = Field(default=None)


AxisSpec: TypeAlias = Annotated[AxisConst | AxisVar | AxisExpr, Field(discriminator="type")]


def _checked_div(left: Any, right: Any) -> Any:
    if right == 0:
        raise ValueError("Division by zero in shape expression")
    return operator.truediv(left, right)


def _checked_floordiv(left: Any, right: Any) -> Any:
    if right == 0:
        raise ValueError("Division by zero in shape expression")
    return operator.floordiv(left, right)


def _checked_mod(left: Any, right: Any) -> Any:
    if right == 0:
        raise ValueError("Modulo by zero in shape expression")
    return operator.mod(left, right)


def _checked_pow(base: Any, exponent: Any) -> Any:
    if isinstance(exponent, int) or (isinstance(exponent, float) and exponent.is_integer()):
        if int(exponent) > _MAX_POW_EXPONENT:
            raise ValueError(f"Exponent {exponent} exceeds maximum {_MAX_POW_EXPONENT}")
        if isinstance(base, int) and abs(base).bit_length() * int(exponent) > _MAX_POW_RESULT_BITS:
            raise ValueError(f"Power result would exceed {_MAX_POW_RESULT_BITS} bits")
    return operator.pow(base, exponent)


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: _checked_div,
    ast.FloorDiv: _checked_floordiv,
    ast.Mod: _checked_mod,
    ast.Pow: _checked_pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def resolve_shape_expression(expr: str, variables: dict[str, int]) -> int:
    """Evaluate an ``AxisExpr`` expression: numbers, names, ``+ - * / // % **``, parentheses, unary +/-."""
    if not isinstance(expr, str):
        raise TypeError(f"Expression must be a str, got {type(expr).__name__}")
    if len(expr) > _MAX_EXPR_LEN:
        raise ValueError(f"Expression too long ({len(expr)} chars, max {_MAX_EXPR_LEN})")

    def evaluate(node: ast.AST, depth: int = 0) -> Any:
        if depth > _MAX_AST_DEPTH:
            raise ValueError(f"Expression nesting too deep (max depth {_MAX_AST_DEPTH})")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise TypeError(f"Unsupported constant type: {type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Shape expression references undefined axis '{node.id}'")
            value = variables[node.id]
            if not isinstance(value, int):
                raise TypeError(f"Variable '{node.id}' must be int, got {type(value).__name__}")
            return value
        if isinstance(node, ast.BinOp):
            function = _BINARY_OPERATORS.get(type(node.op))
            if function is None:
                raise TypeError(f"Unsupported operator: {type(node.op).__name__}")
            return function(evaluate(node.left, depth + 1), evaluate(node.right, depth + 1))
        if isinstance(node, ast.UnaryOp):
            function = _UNARY_OPERATORS.get(type(node.op))
            if function is None:
                raise TypeError(f"Unsupported unary operator: {type(node.op).__name__}")
            return function(evaluate(node.operand, depth + 1))
        raise TypeError(f"Unsupported expression node: {type(node).__name__}")

    try:
        tree = ast.parse(expr, mode="eval")
    except RecursionError as exc:
        raise ValueError("Expression nesting too deep to parse") from exc
    value = evaluate(tree.body)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, int):
        raise TypeError(f"Expression '{expr}' must evaluate to an integer, got {type(value).__name__}")
    return value


def _expression_names(expression: str) -> set[str]:
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError):
        return set()
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def resolve_axes_values(
    const_axes: Mapping[str, int],
    var_axes_values: Mapping[str, int],
    expressions: Mapping[str, str],
    evaluate: Callable[[str, dict[str, int]], int] = resolve_shape_expression,
    *,
    unresolved_axes: Collection[str] = (),
) -> dict[str, int]:
    """Bind const and var axes, then evaluate expression axes in declaration order.

    Expression axes that depend on ``unresolved_axes`` (directly or through another
    expression axis) are left out; any other evaluation failure names the axis.
    """
    resolved = dict(const_axes)
    resolved.update(var_axes_values)
    unresolved = set(unresolved_axes)
    for name, expression in expressions.items():
        if unresolved and unresolved & _expression_names(expression):
            unresolved.add(name)
            continue
        try:
            resolved[name] = evaluate(expression, resolved)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"expression axis {name!r} = {expression!r}: {exc}") from exc
    return resolved


def _top_level_function(reference: str, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(reference, mode="exec")
    return next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name), None)


class Definition(_FibDefinition):
    """FIB Definition plus expression axes and custom input/correctness entrypoints."""

    axes: dict[Annotated[str, Field(min_length=1)], AxisSpec]
    custom_inputs_entrypoint: Optional[str] = Field(default=None, min_length=1)
    custom_correctness_entrypoint: Optional[str] = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _verify_custom_inputs_entrypoint(self) -> "Definition":
        name = self.custom_inputs_entrypoint
        if name is None:
            return self
        if not name.isidentifier():
            raise ValueError(f"custom_inputs_entrypoint must be a valid Python identifier, got: {name!r}")
        function = _top_level_function(self.reference, name)
        if function is None:
            raise ValueError(
                f"custom_inputs_entrypoint '{name}' is not defined as a top-level function in the reference code"
            )
        arguments = function.args
        parameters = [a.arg for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)]
        if "seed" not in parameters:
            warnings.warn(
                f"custom_inputs_entrypoint '{name}' does not declare a 'seed' parameter. Update the signature to "
                "fn(axes_and_scalars, device, seed) for reproducible custom inputs.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @model_validator(mode="after")
    def _verify_custom_correctness_entrypoint(self) -> "Definition":
        name = self.custom_correctness_entrypoint
        if name is None:
            return self
        if not name.isidentifier():
            raise ValueError(f"custom_correctness_entrypoint must be a valid Python identifier, got: {name!r}")
        function = _top_level_function(self.reference, name)
        if function is None:
            raise ValueError(
                f"custom_correctness_entrypoint '{name}' is not defined as a top-level function in the reference code"
            )
        parameters = [a.arg for a in (*function.args.posonlyargs, *function.args.args)]
        if len(parameters) != 2:
            raise ValueError(
                f"custom_correctness_entrypoint '{name}' must take exactly two positional parameters "
                f"(ref_outputs, user_outputs), plus optional keyword-only parameters; "
                f"got {len(parameters)}: {parameters}"
            )
        return self

    @model_validator(mode="after")
    def _validate_tensor_axis_references(self) -> "Definition":
        for tensor_name, tensor_spec in {**self.inputs, **self.outputs}.items():
            for axis_name in tensor_spec.shape or ():
                if not axis_name.isdigit() and axis_name not in self.axes:
                    tensor_type = "input" if tensor_name in self.inputs else "output"
                    raise ValueError(
                        f'{tensor_type.capitalize()} "{tensor_name}" references undefined axis "{axis_name}"'
                    )
        return self

    @cached_property
    def expr_axes(self) -> dict[str, AxisExpr]:
        """Return expression axes in declaration order."""
        return {name: axis for name, axis in self.axes.items() if isinstance(axis, AxisExpr)}

    def get_resolved_axes_values(self, var_axes_values: dict[str, int]) -> dict[str, int]:
        """Bind const and var axes, then evaluate expression axes in declaration order."""
        return resolve_axes_values(
            self.const_axes, var_axes_values, {name: axis.expression for name, axis in self.expr_axes.items()}
        )

    def _get_shapes(
        self,
        tensors: Any,
        var_axes_values: Optional[dict[str, int]] = None,
    ) -> list[Optional[tuple[int, ...]]]:
        resolved = self.get_resolved_axes_values(var_axes_values or {})
        shapes: list[Optional[tuple[int, ...]]] = []
        for spec in tensors:
            if spec.shape is None:
                shapes.append(None)
                continue
            try:
                shapes.append(tuple(int(name) if name.isdigit() else resolved[name] for name in spec.shape))
            except KeyError as exc:
                raise ValueError(f"Missing value for variable axis '{exc.args[0]}'") from exc
        return shapes


__all__ = [
    "AxisConst",
    "AxisExpr",
    "AxisSpec",
    "AxisVar",
    "Definition",
    "TensorSpec",
    "resolve_axes_values",
    "resolve_shape_expression",
]
