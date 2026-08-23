# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Validation and runtime loading for composed Definition references."""

from __future__ import annotations

import ast
import builtins
import json
import keyword
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Iterator

from tilegym.kernel_inventory.layout import inventory_coordinate


class DefinitionCompositionError(ValueError):
    """Raised when a hierarchical Definition composition is invalid."""


_ALLOWED_MODULES = {"collections", "functools", "itertools", "math", "operator", "torch"}
_ALLOWED_BUILTINS = {
    "abs",
    "all",
    "any",
    "bool",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "round",
    "sum",
    "tuple",
    "zip",
}
_ALLOWED_BUILTINS.update(
    name
    for name in ("AssertionError", "NotImplementedError", "RuntimeError", "TypeError", "ValueError")
    if hasattr(builtins, name)
)
_RESERVED_INCLUDE_NAMES = _ALLOWED_MODULES | _ALLOWED_BUILTINS | {"__builtins__", "run"}


def validate_definition_composition(definition: dict[str, Any], definition_path: str | Path) -> None:
    """Validate one wrapper Definition and its recursively included leaves."""
    path = Path(definition_path)
    includes = _include_names(definition)
    coordinate = _classified_coordinate(path)
    if coordinate is not None:
        if coordinate.level in {"public", "leaf"}:
            if "include" in definition:
                raise DefinitionCompositionError(
                    f"{path}: hierarchical {coordinate.level} Definitions must not declare include"
                )
            return
        if coordinate.level == "wrapper":
            unsupported = "runtime:unsupported" in definition.get("tags", [])
            if not includes and not unsupported:
                raise DefinitionCompositionError(
                    f"{path}: executable hierarchical wrapper Definitions must include at least one leaf Definition"
                )
            _validate_composition_tree(path, definition, ())
            return
    if not includes:
        return
    _validate_composition_tree(path, definition, ())


def included_definition_paths(definition: dict[str, Any], definition_path: str | Path) -> list[Path]:
    """Resolve direct include names relative to a wrapper Definition."""
    path = Path(definition_path)
    resolved = []
    for name in _include_names(definition):
        child = path.parent / f"{name}.json"
        if not child.is_file():
            raise DefinitionCompositionError(f"{path}: included Definition does not exist: {name}")
        child_definition = _load_definition(child)
        if child_definition.get("name") != name:
            raise DefinitionCompositionError(
                f"{child}: included Definition name {child_definition.get('name')!r} does not match {name!r}"
            )
        parent_coordinate = _classified_coordinate(path)
        child_coordinate = _classified_coordinate(child)
        if parent_coordinate is not None and parent_coordinate.level == "wrapper":
            if child_coordinate is None or child_coordinate.level != "leaf":
                raise DefinitionCompositionError(f"{path}: include {name!r} must resolve to a leaf Definition")
            if (
                child_coordinate.operation != parent_coordinate.operation
                or child_coordinate.backend != parent_coordinate.backend
                or child.parent != path.parent
            ):
                raise DefinitionCompositionError(
                    f"{path}: include {name!r} must resolve in the wrapper's operation/backend directory"
                )
        resolved.append(child)
    return resolved


@contextmanager
def installed_reference_modules(definition: dict[str, Any], definition_path: str | Path) -> Iterator[None]:
    """Install included Definition references as temporary Python modules."""
    path = Path(definition_path)
    ordered_definitions: list[tuple[str, Path, dict[str, Any]]] = []
    _collect_reference_modules(definition, path, ordered_definitions, (), {})
    saved = {name: sys.modules.get(name) for name, _path, _definition in ordered_definitions}
    try:
        for name, child_path, child in ordered_definitions:
            module = types.ModuleType(name)
            module.__file__ = str(child_path)
            sys.modules[name] = module
            exec(compile(child["reference"], str(child_path), "exec"), module.__dict__)
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _collect_reference_modules(
    definition: dict[str, Any],
    definition_path: Path,
    ordered: list[tuple[str, Path, dict[str, Any]]],
    stack: tuple[Path, ...],
    names: dict[str, Path],
) -> None:
    for child_path in included_definition_paths(definition, definition_path):
        if child_path in stack:
            cycle = " -> ".join(path.name for path in (*stack, child_path))
            raise DefinitionCompositionError(f"Definition include cycle: {cycle}")
        child = _load_definition(child_path)
        _collect_reference_modules(child, child_path, ordered, (*stack, child_path), names)
        name = child["name"]
        previous = names.get(name)
        if previous is not None and previous != child_path:
            raise DefinitionCompositionError(
                f"Included Definition module name {name!r} resolves to both {previous} and {child_path}"
            )
        if previous is None:
            names[name] = child_path
            ordered.append((name, child_path, child))


def _validate_composition_tree(path: Path, definition: dict[str, Any], stack: tuple[Path, ...]) -> None:
    if path in stack:
        cycle = " -> ".join(entry.name for entry in (*stack, path))
        raise DefinitionCompositionError(f"Definition include cycle: {cycle}")
    includes = _include_names(definition)
    tree = ast.parse(definition["reference"], filename=str(path))
    run = _global_run(tree, path)
    _validate_flat_module(tree, run, includes, path)
    _validate_run_header(run, includes, path)
    imported_roots, imported_names = _import_bindings(tree, includes)
    _validate_supported_control_flow(run, path)

    child_runs = {}
    for child_path in included_definition_paths(definition, path):
        child = _load_definition(child_path)
        child_tree = ast.parse(child["reference"], filename=str(child_path))
        child_runs[child["name"]] = _global_run(child_tree, child_path)
        child_coordinate = _classified_coordinate(child_path)
        if child_coordinate is not None and child_coordinate.level == "leaf":
            if "include" in child:
                raise DefinitionCompositionError(
                    f"{child_path}: hierarchical leaf Definitions must not declare include"
                )
        else:
            _validate_composition_tree(child_path, child, (*stack, path))

    calls_by_include = {name: [] for name in includes}
    _validate_protected_names(run, includes, imported_roots | imported_names, path)
    local_names = _run_local_names(run)
    for node in _walk_run_body(run):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and node is not run:
            raise DefinitionCompositionError(f"{path}: wrapper reference run must not define nested functions")
        if not isinstance(node, ast.Call):
            continue
        included_name = _included_call_name(node)
        if included_name is not None:
            if included_name not in child_runs:
                raise DefinitionCompositionError(f"{path}: reference calls non-included module {included_name}.run")
            _validate_call(child_runs[included_name], node, path, included_name)
            calls_by_include[included_name].append(node)
            continue
        _validate_allowed_call(node, includes, imported_roots, imported_names, local_names, path)

    unused = sorted(name for name, calls in calls_by_include.items() if not calls)
    if unused:
        raise DefinitionCompositionError(f"{path}: included Definitions are not called by reference run: {unused}")


def _validate_flat_module(
    tree: ast.Module, run: ast.FunctionDef | ast.AsyncFunctionDef, includes: list[str], path: Path
) -> None:
    imported_includes = set()
    bound_imports: dict[str, str] = {}

    def record_binding(bound: str, source: str) -> None:
        if bound == "run":
            raise DefinitionCompositionError(
                f"{path}: wrapper reference import {source!r} shadows the required global run function"
            )
        if bound in _ALLOWED_BUILTINS:
            raise DefinitionCompositionError(
                f"{path}: wrapper reference import {source!r} shadows protected builtin {bound!r}"
            )
        previous = bound_imports.get(bound)
        if previous is not None:
            raise DefinitionCompositionError(
                f"{path}: wrapper reference imports {previous!r} and {source!r} as the same name {bound!r}"
            )
        bound_imports[bound] = source

    for node in tree.body:
        if node is run:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in includes:
                    if alias.asname is not None:
                        raise DefinitionCompositionError(
                            f"{path}: included Definitions must use their canonical module name"
                        )
                    imported_includes.add(alias.name)
                    record_binding(alias.name, alias.name)
                elif alias.name.split(".", 1)[0] not in _ALLOWED_MODULES:
                    raise DefinitionCompositionError(
                        f"{path}: wrapper reference imports unsupported module {alias.name!r}"
                    )
                else:
                    bound = alias.asname or alias.name.split(".", 1)[0]
                    if bound in includes:
                        raise DefinitionCompositionError(
                            f"{path}: wrapper reference import {alias.name!r} shadows included Definition {bound!r}"
                        )
                    record_binding(bound, alias.name)
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".", 1)[0] not in _ALLOWED_MODULES:
                raise DefinitionCompositionError(
                    f"{path}: wrapper reference imports unsupported module {node.module!r}"
                )
            for alias in node.names:
                if alias.name == "*":
                    raise DefinitionCompositionError(f"{path}: wrapper reference must not use wildcard imports")
                bound = alias.asname or alias.name
                if bound in includes:
                    raise DefinitionCompositionError(
                        f"{path}: wrapper reference import {node.module}.{alias.name} "
                        f"shadows included Definition {bound!r}"
                    )
                record_binding(bound, f"{node.module}.{alias.name}")
            continue
        raise DefinitionCompositionError(
            f"{path}: wrapper reference must contain only imports and one top-level run function"
        )
    missing = sorted(set(includes) - imported_includes)
    if missing:
        raise DefinitionCompositionError(f"{path}: included Definitions must be imported by module name: {missing}")

    for node in ast.walk(run):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise DefinitionCompositionError(f"{path}: wrapper reference run must not contain imports")


def _validate_allowed_call(
    node: ast.Call,
    includes: list[str],
    imported_roots: set[str],
    imported_names: set[str],
    local_names: set[str],
    path: Path,
) -> None:
    if isinstance(node.func, ast.Name):
        if node.func.id not in _ALLOWED_BUILTINS | imported_names:
            raise DefinitionCompositionError(f"{path}: wrapper reference calls unsupported function {node.func.id!r}")
        return
    if not isinstance(node.func, ast.Attribute):
        raise DefinitionCompositionError(f"{path}: wrapper reference contains an unsupported dynamic call")
    root = node.func.value
    while isinstance(root, ast.Attribute):
        root = root.value
    if isinstance(root, ast.Name) and root.id in includes:
        raise DefinitionCompositionError(f"{path}: included Definition calls must target <name>.run exactly")
    if isinstance(root, ast.Name) and root.id not in imported_roots | imported_names | local_names:
        raise DefinitionCompositionError(f"{path}: wrapper reference calls an unsupported attribute root {root.id!r}")
    if not isinstance(root, (ast.Name, ast.Call, ast.Subscript)):
        raise DefinitionCompositionError(f"{path}: wrapper reference contains an unsupported dynamic call")


def _included_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Attribute) and node.func.attr == "run" and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    return None


def _validate_supported_control_flow(
    run: ast.FunctionDef | ast.AsyncFunctionDef,
    path: Path,
) -> None:
    """Validate only the structural control-flow subset used by wrapper references."""
    if isinstance(run, ast.AsyncFunctionDef):
        raise DefinitionCompositionError(f"{path}: wrapper reference uses unsupported async run control flow")

    supported_statements = (
        ast.AnnAssign,
        ast.Assign,
        ast.AugAssign,
        ast.Expr,
        ast.If,
        ast.Pass,
        ast.Raise,
        ast.Return,
        ast.With,
    )
    for node in ast.walk(run):
        if node is not run and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            raise DefinitionCompositionError(
                f"{path}: wrapper reference run must not define nested functions or classes"
            )
        if node is not run and isinstance(node, ast.stmt) and not isinstance(node, supported_statements):
            raise DefinitionCompositionError(
                f"{path}: wrapper reference uses unsupported {type(node).__name__} control flow"
            )
        if isinstance(node, (ast.NamedExpr, ast.Yield, ast.YieldFrom)):
            raise DefinitionCompositionError(
                f"{path}: wrapper reference uses unsupported {type(node).__name__} control flow"
            )
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            raise DefinitionCompositionError(
                f"{path}: wrapper reference uses unsupported {type(node).__name__} loop expression"
            )


def _validate_call(run: ast.FunctionDef | ast.AsyncFunctionDef, call: ast.Call, path: Path, name: str) -> None:
    args = run.args
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise DefinitionCompositionError(f"{path}: {name}.run calls must not use *args expansion")
    if any(keyword.arg is None for keyword in call.keywords):
        raise DefinitionCompositionError(f"{path}: {name}.run calls must not use **kwargs expansion")
    positional = [*args.posonlyargs, *args.args]
    parameters = {parameter.arg for parameter in (*positional, *args.kwonlyargs)}
    if len(call.args) > len(positional) and args.vararg is None:
        raise DefinitionCompositionError(f"{path}: {name}.run receives too many positional arguments")
    keyword_list = [keyword.arg for keyword in call.keywords]
    duplicate_keywords = sorted({keyword for keyword in keyword_list if keyword_list.count(keyword) > 1})
    if duplicate_keywords:
        raise DefinitionCompositionError(
            f"{path}: {name}.run receives duplicate keyword arguments: {duplicate_keywords}"
        )
    keyword_names = set(keyword_list)
    posonly_names = {parameter.arg for parameter in args.posonlyargs}
    posonly_keywords = keyword_names & posonly_names
    if posonly_keywords and args.kwarg is None:
        raise DefinitionCompositionError(
            f"{path}: {name}.run receives positional-only parameters by keyword: {sorted(posonly_keywords)}"
        )
    binding_keyword_names = keyword_names - posonly_keywords
    positionally_bound = {parameter.arg for parameter in positional[: len(call.args)]}
    double_bound = sorted(positionally_bound & binding_keyword_names)
    if double_bound:
        raise DefinitionCompositionError(f"{path}: {name}.run receives arguments more than once: {double_bound}")
    unknown = sorted(binding_keyword_names - parameters) if args.kwarg is None else []
    if unknown:
        raise DefinitionCompositionError(f"{path}: {name}.run receives unknown keyword arguments: {unknown}")
    positional_required = len(positional) - len(args.defaults)
    required = {parameter.arg for parameter in positional[:positional_required]}
    required.update(parameter.arg for parameter, default in zip(args.kwonlyargs, args.kw_defaults) if default is None)
    supplied = {parameter.arg for parameter in positional[: len(call.args)]} | binding_keyword_names
    missing = sorted(required - supplied)
    if missing:
        raise DefinitionCompositionError(f"{path}: {name}.run is missing required arguments: {missing}")


def _include_names(definition: dict[str, Any]) -> list[str]:
    raw = definition.get("include", [])
    if not isinstance(raw, list):
        raise DefinitionCompositionError("Definition.include must be a list")
    if not all(isinstance(name, str) and name and name.isidentifier() and not keyword.iskeyword(name) for name in raw):
        raise DefinitionCompositionError("Definition.include entries must be nonempty Python module identifiers")
    if len(set(raw)) != len(raw):
        raise DefinitionCompositionError("Definition.include must not contain duplicates")
    reserved = sorted(set(raw) & _RESERVED_INCLUDE_NAMES)
    if reserved:
        raise DefinitionCompositionError(f"Definition.include entries use reserved module names: {reserved}")
    return raw


def _classified_coordinate(path: Path) -> Any | None:
    try:
        return inventory_coordinate(path)
    except ValueError as exc:
        if any(parent.name in {"kernel_definitions", "kernel_solutions"} for parent in (path.parent, *path.parents)):
            raise DefinitionCompositionError(str(exc)) from exc
        return None


def _walk_run_body(run: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    for statement in run.body:
        yield from ast.walk(statement)


def _validate_run_header(
    run: ast.FunctionDef | ast.AsyncFunctionDef,
    includes: list[str],
    path: Path,
) -> None:
    """Reject include references evaluated while defining, rather than calling, run."""
    header_nodes: list[ast.AST] = [*run.decorator_list, *run.args.defaults]
    header_nodes.extend(default for default in run.args.kw_defaults if default is not None)
    header_nodes.extend(
        argument.annotation
        for argument in (*run.args.posonlyargs, *run.args.args, *run.args.kwonlyargs)
        if argument.annotation is not None
    )
    if run.args.vararg is not None and run.args.vararg.annotation is not None:
        header_nodes.append(run.args.vararg.annotation)
    if run.args.kwarg is not None and run.args.kwarg.annotation is not None:
        header_nodes.append(run.args.kwarg.annotation)
    if run.returns is not None:
        header_nodes.append(run.returns)
    header_nodes.extend(getattr(run, "type_params", ()))
    for header in header_nodes:
        for node in ast.walk(header):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in includes:
                raise DefinitionCompositionError(
                    f"{path}: included Definition module {node.id!r} may only be referenced inside run body"
                )


def _import_bindings(tree: ast.Module, includes: list[str]) -> tuple[set[str], set[str]]:
    roots: set[str] = set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if alias.name not in includes:
                    roots.add(bound)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return roots, names


def _run_local_names(run: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = {argument.arg for argument in (*run.args.posonlyargs, *run.args.args, *run.args.kwonlyargs)}
    if run.args.vararg is not None:
        names.add(run.args.vararg.arg)
    if run.args.kwarg is not None:
        names.add(run.args.kwarg.arg)
    for node in ast.walk(run):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _validate_protected_names(
    run: ast.FunctionDef | ast.AsyncFunctionDef,
    includes: list[str],
    imported_names: set[str],
    path: Path,
) -> None:
    """Prevent import shadowing and include-module alias escape paths."""
    protected = set(includes) | imported_names
    arguments = {argument.arg for argument in (*run.args.posonlyargs, *run.args.args, *run.args.kwonlyargs)}
    if run.args.vararg is not None:
        arguments.add(run.args.vararg.arg)
    if run.args.kwarg is not None:
        arguments.add(run.args.kwarg.arg)
    shadowed = sorted(arguments & protected)
    for node in ast.walk(run):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and node.id in protected:
            shadowed.append(node.id)
    if shadowed:
        raise DefinitionCompositionError(
            f"{path}: wrapper reference run must not shadow imported or included module names: {sorted(set(shadowed))}"
        )

    parents = {child: parent for parent in ast.walk(run) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(run):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load) or node.id not in includes:
            continue
        attribute = parents.get(node)
        call = parents.get(attribute)
        if (
            not isinstance(attribute, ast.Attribute)
            or attribute.value is not node
            or attribute.attr != "run"
            or not isinstance(call, ast.Call)
            or call.func is not attribute
        ):
            raise DefinitionCompositionError(
                f"{path}: included Definition module {node.id!r} may only be referenced as {node.id}.run(...)"
            )


def _global_run(tree: ast.Module, path: Path) -> ast.FunctionDef | ast.AsyncFunctionDef:
    runs = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "run"
    ]
    if len(runs) != 1:
        raise DefinitionCompositionError(f"{path}: Definition.reference must define exactly one global run")
    return runs[0]


def _load_definition(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DefinitionCompositionError(f"Unable to load included Definition {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DefinitionCompositionError(f"Included Definition must be a JSON object: {path}")
    return data
