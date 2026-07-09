# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""AST-derived mapping from reference return values to Definition outputs."""

from __future__ import annotations

import ast
import copy
from collections.abc import Sequence

CAPTURE_RETURN_NAME = "__tilegym_capture_reference_return__"


class ReturnContractError(ValueError):
    """Raised when reference return expressions cannot map to Definition outputs."""


def instrument_reference_returns(
    reference: str,
    output_names: Sequence[str],
    *,
    allow_no_return: bool = False,
) -> ast.Module:
    """Wrap each executed ``run`` return with its parallel output-name tree."""
    module = ast.parse(reference, mode="exec")
    run_nodes = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    if len(run_nodes) != 1:
        raise ReturnContractError("Definition.reference must define exactly one global run function")

    transformer = _ReturnTransformer(tuple(output_names))
    run = run_nodes[0]
    run.body = [transformer.visit(statement) for statement in run.body]
    if not transformer.descriptors:
        if allow_no_return:
            return ast.fix_missing_locations(module)
        raise ReturnContractError("Definition.reference run must return its declared outputs")

    referenced_names = {
        node.value
        for descriptor in transformer.descriptors
        for node in ast.walk(descriptor)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    missing_names = sorted(set(output_names) - referenced_names)
    if missing_names:
        raise ReturnContractError(f"Definition.outputs are not referenced by run return expressions: {missing_names}")
    return ast.fix_missing_locations(module)


class _ReturnTransformer(ast.NodeTransformer):
    def __init__(self, output_names: tuple[str, ...]):
        self.output_names = output_names
        self.descriptors: list[ast.expr] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Do not treat returns from helpers nested inside ``run`` as public outputs."""
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return node

    def visit_Lambda(self, node: ast.Lambda):
        return node

    def visit_Return(self, node: ast.Return):
        value = node.value if node.value is not None else ast.Constant(value=None)
        descriptor = _top_level_descriptor(value, self.output_names)
        self.descriptors.append(copy.deepcopy(descriptor))
        node.value = ast.Call(
            func=ast.Name(id=CAPTURE_RETURN_NAME, ctx=ast.Load()),
            args=[value, descriptor],
            keywords=[],
        )
        return node


def _top_level_descriptor(value: ast.expr, output_names: tuple[str, ...]) -> ast.expr:
    if isinstance(value, ast.Tuple):
        if len(value.elts) == len(output_names):
            try:
                explicit = [_value_descriptor(element, output_names, None) for element in value.elts]
            except ReturnContractError:
                explicit = []
            explicit_names = {
                node.value
                for descriptor in explicit
                for node in ast.walk(descriptor)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            if explicit_names == set(output_names):
                return ast.Tuple(elts=explicit, ctx=ast.Load())
            return ast.Tuple(
                elts=[
                    _positional_descriptor(element, output_name)
                    for element, output_name in zip(value.elts, output_names, strict=True)
                ],
                ctx=ast.Load(),
            )
        return ast.Tuple(
            elts=[_value_descriptor(element, output_names, None) for element in value.elts],
            ctx=ast.Load(),
        )

    fallback_name = output_names[0] if len(output_names) == 1 else None
    return _value_descriptor(value, output_names, fallback_name)


def _positional_descriptor(value: ast.expr, output_name: str) -> ast.expr:
    if isinstance(value, ast.Constant) and value.value is None:
        return ast.Constant(value=None)
    if isinstance(value, ast.IfExp):
        return ast.IfExp(
            test=copy.deepcopy(value.test),
            body=_positional_descriptor(value.body, output_name),
            orelse=_positional_descriptor(value.orelse, output_name),
        )
    if isinstance(value, ast.Tuple):
        raise ReturnContractError("Nested reference return tuples require SSA variables named after Definition outputs")
    return ast.Constant(value=output_name)


def _value_descriptor(
    value: ast.expr,
    output_names: tuple[str, ...],
    fallback_name: str | None,
) -> ast.expr:
    if isinstance(value, ast.Constant) and value.value is None:
        return ast.Constant(value=None)
    if isinstance(value, ast.IfExp):
        return ast.IfExp(
            test=copy.deepcopy(value.test),
            body=_value_descriptor(value.body, output_names, fallback_name),
            orelse=_value_descriptor(value.orelse, output_names, fallback_name),
        )
    if isinstance(value, ast.Tuple):
        return ast.Tuple(
            elts=[_value_descriptor(element, output_names, None) for element in value.elts],
            ctx=ast.Load(),
        )

    root_name = _root_name(value)
    if root_name in output_names:
        return ast.Constant(value=root_name)
    if fallback_name is not None:
        return ast.Constant(value=fallback_name)
    raise ReturnContractError(
        "Reference return tensor must use an SSA variable named after a Definition output; "
        f"could not map expression {ast.unparse(value)!r}"
    )


def _root_name(value: ast.expr) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return _root_name(value.value)
    if isinstance(value, ast.Call):
        return _root_name(value.func)
    if isinstance(value, ast.Subscript):
        return _root_name(value.value)
    return None
