# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Helpers for FlashInfer-style transformer kernel inventory metadata.

The metadata files live under transformer model submodules:

- ``kernel_definitions/*.json`` for FlashInfer Trace Definition objects.
- ``kernel_solutions/*.json`` for Solution objects.
- ``kernels/*.py`` for reusable kernel implementations referenced by Solution
  source paths.

This module intentionally avoids importing flashinfer-bench at module import
time. Schema validation uses FlashInfer-Bench lazily when called, and Ocean
keeps a few repo-local checks around source permalinks and path-only sources.
"""

from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any
from typing import Iterator

from tilegym.transformers.inventory_common import SourceContractError
from tilegym.transformers.inventory_common import resolve_repo_relative_path
from tilegym.transformers.inventory_common import validate_reference_source_contract

DEFINITION_SCHEMA_URL = (
    "https://github.com/flashinfer-ai/flashinfer-bench/blob/main/docs/flashinfer-trace/definition.mdx"
)
SOLUTION_SCHEMA_URL = "https://github.com/flashinfer-ai/flashinfer-bench/blob/main/docs/flashinfer-trace/solution.mdx"


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
    """Validate a transformer kernel Definition with FIB and Ocean checks."""
    _require_mapping(definition, "Definition")
    try:
        from tilegym.transformers.inventory_generation import validate_ocean_definition_model

        validate_ocean_definition_model(definition)
    except ImportError as exc:
        raise KernelInventoryError(
            "flashinfer-bench is required to validate transformer kernel Definitions. "
            "Install the tilegym-hf-bench dev environment with `uv sync --extra dev` "
            "from modeling/transformers."
        ) from exc
    except Exception as exc:
        raise KernelInventoryError(f"Definition schema invalid: {exc}") from exc

    _validate_reference_run(definition["reference"], list(definition["inputs"]))


def validate_solution(solution: dict[str, Any], repo_root: str | Path | None = None) -> None:
    """Validate a transformer kernel Solution with FIB and Ocean checks."""
    _require_mapping(solution, "Solution")
    source_paths = normalize_solution_source_paths(solution)
    if repo_root is not None:
        root = Path(repo_root).resolve()
        for source_path in source_paths:
            path = _resolve_inventory_path(root, source_path, "Solution source path")
            if not path.is_file():
                raise KernelInventoryError(f"Solution source path does not exist: {source_path}")
        spec = solution.get("spec")
        if isinstance(spec, dict) and isinstance(spec.get("entry_point"), str) and "::" in spec["entry_point"]:
            entry_file = spec["entry_point"].split("::", 1)[0]
            _resolve_inventory_path(root, entry_file, "Solution entry point path")

    try:
        from tilegym.transformers.inventory_generation import validate_ocean_solution_model

        validate_ocean_solution_model(solution, repo_root)
    except ImportError as exc:
        raise KernelInventoryError(
            "flashinfer-bench is required to validate transformer kernel Solutions. "
            "Install the tilegym-hf-bench dev environment with `uv sync --extra dev` "
            "from modeling/transformers."
        ) from exc
    except Exception as exc:
        raise KernelInventoryError(f"Solution schema invalid: {exc}") from exc


def validate_solution_entry_point(solution: dict[str, Any], repo_root: str | Path) -> None:
    """Validate that a Python Solution entry point file exists and defines the symbol."""
    validate_solution(solution, repo_root)
    entry_point = solution["spec"]["entry_point"]
    file_path, symbol = entry_point.split("::", 1)
    if not file_path.endswith(".py"):
        return
    root = Path(repo_root).resolve()
    source_path = _resolve_inventory_path(root, file_path, "Solution entry point path")
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
            "content": _resolve_inventory_path(root, source_path, "Solution source path").read_text(encoding="utf-8"),
        }
        for source_path in normalize_solution_source_paths(solution)
    ]
    return materialized


def _validate_reference_run(reference: str, input_names: list[str]) -> None:
    try:
        validate_reference_source_contract(reference)
    except SourceContractError as exc:
        raise KernelInventoryError(str(exc)) from exc
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
    run_args = run_nodes[0].args
    if run_args.vararg is not None or run_args.kwarg is not None:
        raise KernelInventoryError("Definition.reference run function must not use *args or **kwargs")
    arg_names = [arg.arg for arg in (*run_args.posonlyargs, *run_args.args, *run_args.kwonlyargs)]
    if arg_names != input_names:
        raise KernelInventoryError(
            f"Definition.reference run arguments {arg_names} must match Definition.inputs {input_names}"
        )


def _resolve_inventory_path(root: Path, path: str, label: str) -> Path:
    try:
        return resolve_repo_relative_path(root, path, label)
    except ValueError as exc:
        raise KernelInventoryError(str(exc)) from exc


def _require_mapping(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise KernelInventoryError(f"{label} must be an object")
