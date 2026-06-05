# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Helpers for FlashInfer-style transformer kernel inventory metadata.

The metadata files live under transformer model submodules:

- ``kernel_definitions/*.json`` for FlashInfer Trace Definition objects.
- ``kernel_solutions/*.json`` for Solution objects.
- ``kernels/*.py`` for reusable kernel implementations referenced by Solution
  source paths.

This module intentionally avoids a flashinfer-bench runtime dependency. It
validates the fields needed for in-repo inventory and can materialize
path-only sources into content-bearing Solution objects for external export.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any
from typing import Iterator

DEFINITION_SCHEMA_URL = (
    "https://github.com/flashinfer-ai/flashinfer-bench/blob/main/docs/flashinfer-trace/definition.mdx"
)
SOLUTION_SCHEMA_URL = "https://github.com/flashinfer-ai/flashinfer-bench/blob/main/docs/flashinfer-trace/solution.mdx"

_VALID_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3fn",
    "float8_e5m2",
    "float4_e2m1",
    "int64",
    "int32",
    "int16",
    "int8",
    "bool",
}


class KernelInventoryError(ValueError):
    """Raised when transformer kernel inventory metadata is invalid."""


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""
    with Path(path).open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise KernelInventoryError(f"{path}: expected a JSON object")
    return data


def iter_kernel_definition_paths(root: str | Path) -> Iterator[Path]:
    """Yield all transformer kernel Definition JSON files under ``root``."""
    yield from sorted(Path(root).glob("src/tilegym/transformers/*/kernel_definitions/*.json"))


def iter_kernel_solution_paths(root: str | Path) -> Iterator[Path]:
    """Yield all transformer kernel Solution JSON files under ``root``."""
    yield from sorted(Path(root).glob("src/tilegym/transformers/*/kernel_solutions/*.json"))


def iter_kernel_python_paths(root: str | Path) -> Iterator[Path]:
    """Yield all dedicated transformer kernel Python modules under ``root``."""
    yield from sorted(Path(root).glob("src/tilegym/transformers/*/kernels/*.py"))


def validate_definition(definition: dict[str, Any]) -> None:
    """Validate required FlashInfer Definition fields used by Ocean."""
    _require_mapping(definition, "Definition")
    _require_keys(definition, ["name", "op_type", "axes", "inputs", "outputs", "reference"], "Definition")
    _require_str(definition, "name", "Definition")
    _require_str(definition, "op_type", "Definition")
    _validate_optional_string_array(definition, "tags", "Definition")
    _validate_optional_string_array(definition, "constraints", "Definition")
    _validate_axes(definition["axes"])
    _validate_tensor_specs(definition["inputs"], "Definition.inputs")
    _validate_tensor_specs(definition["outputs"], "Definition.outputs")
    _require_str(definition, "reference", "Definition")
    _validate_reference_run(definition["reference"])


def validate_solution(solution: dict[str, Any], repo_root: str | Path | None = None) -> None:
    """Validate required Solution fields and optional source path existence."""
    _require_mapping(solution, "Solution")
    _require_keys(solution, ["name", "definition", "author", "spec", "sources"], "Solution")
    _require_str(solution, "name", "Solution")
    _require_str(solution, "definition", "Solution")
    _require_str(solution, "author", "Solution")
    _validate_solution_spec(solution["spec"])
    source_paths = normalize_solution_source_paths(solution)
    if repo_root is not None:
        root = Path(repo_root).resolve()
        for source_path in source_paths:
            path = _resolve_repo_relative_path(root, source_path, "Solution source path")
            if not path.is_file():
                raise KernelInventoryError(f"Solution source path does not exist: {source_path}")


def validate_solution_entry_point(solution: dict[str, Any], repo_root: str | Path) -> None:
    """Validate that a Python Solution entry point file exists and defines the symbol."""
    validate_solution(solution, repo_root)
    entry_point = solution["spec"]["entry_point"]
    file_path, symbol = entry_point.split("::", 1)
    if not file_path.endswith(".py"):
        return
    root = Path(repo_root).resolve()
    source_path = _resolve_repo_relative_path(root, file_path, "Solution entry point path")
    if not source_path.is_file():
        raise KernelInventoryError(f"Solution entry point path does not exist: {file_path}")
    module_ast = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    symbols = {
        node.name for node in module_ast.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    if symbol not in symbols:
        raise KernelInventoryError(f"Solution entry point symbol not found: {entry_point}")


def normalize_solution_source_paths(solution: dict[str, Any]) -> list[str]:
    """Return source paths from either path-list or file-object Solution sources.

    Ocean transformer inventory stores source references by path and does not
    require embedded ``content``. This accepts both ``{"path": [...]}`` and the
    file-object array shape used by current flashinfer-bench examples.
    """
    sources = solution.get("sources")
    paths: list[str] = []
    if isinstance(sources, dict):
        raw_paths = sources.get("path")
        if isinstance(raw_paths, str):
            paths.append(raw_paths)
        elif isinstance(raw_paths, list) and all(isinstance(path, str) for path in raw_paths):
            paths.extend(raw_paths)
        else:
            raise KernelInventoryError("Solution.sources.path must be a string or list of strings")
    elif isinstance(sources, list):
        for index, source in enumerate(sources):
            if not isinstance(source, dict):
                raise KernelInventoryError(f"Solution.sources[{index}] must be an object")
            path = source.get("path")
            if not isinstance(path, str):
                raise KernelInventoryError(f"Solution.sources[{index}].path must be a string")
            paths.append(path)
    else:
        raise KernelInventoryError("Solution.sources must be an object or array")
    if not paths:
        raise KernelInventoryError("Solution.sources must reference at least one source path")
    return paths


def materialize_solution_sources(solution: dict[str, Any], repo_root: str | Path) -> dict[str, Any]:
    """Return a copy of ``solution`` with ``sources`` expanded to path/content files."""
    validate_solution(solution, repo_root)
    materialized = copy.deepcopy(solution)
    root = Path(repo_root).resolve()
    materialized["sources"] = [
        {
            "path": source_path,
            "content": _resolve_repo_relative_path(root, source_path, "Solution source path").read_text(
                encoding="utf-8"
            ),
        }
        for source_path in normalize_solution_source_paths(solution)
    ]
    return materialized


def _validate_reference_run(reference: str) -> None:
    try:
        module_ast = ast.parse(reference, filename="<Definition.reference>")
    except SyntaxError as exc:
        raise KernelInventoryError("Definition.reference must be valid Python") from exc

    run_nodes = [
        node
        for node in module_ast.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "run"
    ]
    if len(run_nodes) != 1:
        raise KernelInventoryError("Definition.reference must contain exactly one global run function")


def _resolve_repo_relative_path(root: Path, path: str, label: str) -> Path:
    raw_path = Path(path)
    if not path or raw_path.is_absolute() or ".." in raw_path.parts:
        raise KernelInventoryError(f"{label} must be a repo-relative path: {path}")

    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise KernelInventoryError(f"{label} must stay inside repo root: {path}") from exc
    return resolved


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise KernelInventoryError(f"{label} must be an object")


def _require_keys(data: dict[str, Any], keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KernelInventoryError(f"{label} missing required fields: {', '.join(missing)}")


def _require_str(data: dict[str, Any], key: str, label: str) -> None:
    if not isinstance(data.get(key), str) or not data[key]:
        raise KernelInventoryError(f"{label}.{key} must be a non-empty string")


def _validate_optional_string_array(data: dict[str, Any], key: str, label: str) -> None:
    if key not in data or data[key] is None:
        return
    if not isinstance(data[key], list) or not all(isinstance(item, str) for item in data[key]):
        raise KernelInventoryError(f"{label}.{key} must be an array of strings")


def _validate_axes(axes: Any) -> None:
    _require_mapping(axes, "Definition.axes")
    if not axes:
        raise KernelInventoryError("Definition.axes must not be empty")
    for name, axis in axes.items():
        if not isinstance(name, str) or not name:
            raise KernelInventoryError("Definition.axes keys must be non-empty strings")
        _require_mapping(axis, f"Definition.axes.{name}")
        axis_type = axis.get("type")
        if axis_type == "const":
            if not isinstance(axis.get("value"), int):
                raise KernelInventoryError(f"Definition.axes.{name}.value must be an integer")
        elif axis_type != "var":
            raise KernelInventoryError(f"Definition.axes.{name}.type must be 'const' or 'var'")
        if "description" in axis and not isinstance(axis["description"], str):
            raise KernelInventoryError(f"Definition.axes.{name}.description must be a string")


def _validate_tensor_specs(tensors: Any, label: str) -> None:
    _require_mapping(tensors, label)
    if not tensors:
        raise KernelInventoryError(f"{label} must not be empty")
    for name, spec in tensors.items():
        if not isinstance(name, str) or not name:
            raise KernelInventoryError(f"{label} keys must be non-empty strings")
        _require_mapping(spec, f"{label}.{name}")
        _require_keys(spec, ["shape", "dtype"], f"{label}.{name}")
        shape = spec["shape"]
        if shape is not None and (
            not isinstance(shape, list) or not all(isinstance(axis_name, str) for axis_name in shape)
        ):
            raise KernelInventoryError(f"{label}.{name}.shape must be null or an array of strings")
        dtype = spec["dtype"]
        if dtype not in _VALID_DTYPES:
            raise KernelInventoryError(f"{label}.{name}.dtype is unsupported: {dtype}")


def _validate_solution_spec(spec: Any) -> None:
    _require_mapping(spec, "Solution.spec")
    _require_keys(spec, ["language", "target_hardware", "entry_point"], "Solution.spec")
    _require_str(spec, "language", "Solution.spec")
    _require_str(spec, "entry_point", "Solution.spec")
    if "::" not in spec["entry_point"]:
        raise KernelInventoryError("Solution.spec.entry_point must use file_path::function_name")
    if not isinstance(spec["target_hardware"], list) or not all(
        isinstance(item, str) for item in spec["target_hardware"]
    ):
        raise KernelInventoryError("Solution.spec.target_hardware must be an array of strings")
    if "destination_passing_style" in spec and not isinstance(spec["destination_passing_style"], bool):
        raise KernelInventoryError("Solution.spec.destination_passing_style must be a bool")
    if "dependencies" in spec and (
        not isinstance(spec["dependencies"], list) or not all(isinstance(item, str) for item in spec["dependencies"])
    ):
        raise KernelInventoryError("Solution.spec.dependencies must be an array of strings")
