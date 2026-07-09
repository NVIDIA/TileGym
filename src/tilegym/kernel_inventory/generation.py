# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""FIB-shaped helpers for TileGym kernel inventory generation.

This module is intentionally imported lazily by ``kernel_inventory`` so normal
TileGym runtime imports do not require FlashInfer-Bench. FIB validates the data
schema; TileGym owns execution of the checked-in metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from typing import Sequence

from flashinfer_bench.data import AxisConst
from flashinfer_bench.data import AxisVar
from flashinfer_bench.data import Definition
from flashinfer_bench.data import Solution
from flashinfer_bench.data import SupportedLanguages
from flashinfer_bench.data import TensorSpec
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from tilegym.kernel_inventory.source_contract import resolve_repo_relative_path as _resolve_repo_relative_path
from tilegym.kernel_inventory.source_contract import validate_reference_source_contract

TILEGYM_CUTILE_LANGUAGE = "cuda-tile"
_FIB_LANGUAGE_BY_TILEGYM_LANGUAGE = {
    TILEGYM_CUTILE_LANGUAGE: SupportedLanguages.PYTHON.value,
}


class TileGymSourceFile(BaseModel):
    """TileGym source reference that may omit content in checked-in JSON."""

    path: str
    content: str | None = None

    @model_validator(mode="after")
    def _validate_source_path(self) -> "TileGymSourceFile":
        raw_path = Path(self.path)
        if not self.path or raw_path.is_absolute() or ".." in raw_path.parts:
            raise ValueError(f"Invalid source path: {self.path}")
        return self


class TileGymBuildSpec(BaseModel):
    """Build spec compatible with TileGym's cuTile language label."""

    language: str
    target_hardware: list[str] = Field(min_length=1)
    entry_point: str
    dependencies: list[str] = Field(default_factory=list)
    destination_passing_style: bool = True
    binding: str | None = None

    @model_validator(mode="after")
    def _validate_spec(self) -> "TileGymBuildSpec":
        supported_languages = {language.value for language in SupportedLanguages}
        supported_languages.add(TILEGYM_CUTILE_LANGUAGE)
        if self.language not in supported_languages:
            raise ValueError(f"Unsupported Solution.spec.language: {self.language}")
        if self.entry_point.count("::") != 1:
            raise ValueError(
                f'Invalid entry point format: {self.entry_point}. Expected "<file_path>::<function_name>".'
            )
        invalid_targets = [label for label in self.target_hardware if re.fullmatch(r"SM\d{2,}", label) is None]
        if invalid_targets:
            raise ValueError(
                f"Solution.spec.target_hardware entries must use the SM<major><minor> format: {invalid_targets}"
            )
        return self

    def fib_language(self) -> str:
        """Return the closest FIB language for schema validation."""
        return _FIB_LANGUAGE_BY_TILEGYM_LANGUAGE.get(self.language, self.language)


class TileGymSolution(BaseModel):
    """TileGym Solution schema with path-only source support."""

    name: str
    definition: str
    author: str
    spec: TileGymBuildSpec
    sources: list[TileGymSourceFile] = Field(min_length=1)
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_sources(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "sources" not in data:
            return data

        normalized = dict(data)
        sources = normalized["sources"]
        if isinstance(sources, dict):
            raw_paths = sources.get("path")
            if isinstance(raw_paths, str):
                paths = [raw_paths]
            elif isinstance(raw_paths, list):
                paths = raw_paths
            else:
                return data
            normalized["sources"] = [{"path": path} for path in paths]
        return normalized

    @model_validator(mode="after")
    def _validate_source_path_entry_point(self) -> "TileGymSolution":
        seen_paths: set[str] = set()
        for source in self.sources:
            if source.path in seen_paths:
                raise ValueError(f"Duplicate source path '{source.path}'")
            seen_paths.add(source.path)

        entry_file = self.spec.entry_point.split("::", 1)[0]
        if entry_file not in seen_paths:
            raise ValueError(f"Entry source file '{entry_file}' not found in sources")
        return self

    def to_fib_solution(
        self,
        repo_root: str | Path | None = None,
        *,
        allow_placeholder_content: bool = False,
    ) -> Solution:
        """Materialize path-only sources and validate through FIB Solution."""
        sources = []
        for source in self.sources:
            content = source.content
            if content is None:
                if repo_root is None:
                    if not allow_placeholder_content:
                        raise ValueError("repo_root is required to materialize path-only Solution.sources")
                    content = "# Placeholder content for schema-only validation.\n"
                else:
                    content = _resolve_repo_relative_path(
                        Path(repo_root).resolve(),
                        source.path,
                        "Solution source path",
                    ).read_text(encoding="utf-8")
            sources.append({"path": source.path, "content": content})

        fib_data = self.model_dump(mode="json", exclude_none=True)
        fib_data["spec"]["language"] = self.spec.fib_language()
        fib_data["sources"] = sources
        return Solution.model_validate(fib_data)

    def to_tilegym_dict(self, *, path_only_sources: bool = True) -> dict[str, Any]:
        """Return stable TileGym JSON data."""
        data = self.model_dump(mode="json", exclude_none=True)
        if path_only_sources:
            data["sources"] = {"path": [source.path for source in self.sources]}
        return _drop_none_except_shape(data)


def axis_var(description: str | None = None) -> AxisVar:
    """Create a FIB variable axis."""
    return AxisVar(description=description)


def axis_const(value: int, description: str | None = None) -> AxisConst:
    """Create a FIB constant axis."""
    return AxisConst(value=value, description=description)


def tensor(shape: Sequence[str] | None, dtype: str, description: str | None = None) -> TensorSpec:
    """Create a FIB tensor specification."""
    return TensorSpec(shape=list(shape) if shape is not None else None, dtype=dtype, description=description)


def make_definition(
    *,
    name: str,
    op_type: str,
    axes: dict[str, Any],
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference: str,
    tags: Sequence[str] | None = None,
    description: str | None = None,
    constraints: Sequence[str] | None = None,
) -> Definition:
    """Create a Definition after enforcing TileGym's source permalink contract."""
    validate_reference_source_contract(reference)
    return Definition(
        name=name,
        op_type=op_type,
        axes=axes,
        inputs=inputs,
        outputs=outputs,
        reference=reference,
        tags=list(tags or []),
        description=description,
        constraints=list(constraints or []),
    )


def make_solution(
    *,
    name: str,
    definition: str,
    author: str,
    spec: dict[str, Any] | TileGymBuildSpec,
    sources: dict[str, Any] | Sequence[str | dict[str, Any] | TileGymSourceFile],
    repo_root: str | Path | None = None,
    description: str | None = None,
) -> TileGymSolution:
    """Create a TileGym Solution and validate it through FIB Solution."""
    normalized_sources: Any
    if isinstance(sources, dict):
        normalized_sources = sources
    elif isinstance(sources, str):
        normalized_sources = [{"path": sources}]
    else:
        normalized_sources = [{"path": source} if isinstance(source, str) else source for source in sources]

    solution = TileGymSolution.model_validate(
        {
            "name": name,
            "definition": definition,
            "author": author,
            "spec": spec.model_dump(mode="json") if isinstance(spec, TileGymBuildSpec) else spec,
            "sources": normalized_sources,
            "description": description,
        }
    )
    solution.to_fib_solution(repo_root, allow_placeholder_content=repo_root is None)
    return solution


def source_path(path: str) -> TileGymSourceFile:
    """Create a path-only TileGym source reference."""
    return TileGymSourceFile(path=path)


def validate_tilegym_definition_model(definition: dict[str, Any]) -> Definition:
    """Validate a TileGym Definition with FIB Definition."""
    model = Definition.model_validate(definition)
    validate_reference_source_contract(model.reference)
    return model


def validate_tilegym_solution_model(solution: dict[str, Any], repo_root: str | Path | None = None) -> TileGymSolution:
    """Validate a TileGym Solution with TileGym and FIB schema checks."""
    model = TileGymSolution.model_validate(solution)
    model.to_fib_solution(repo_root, allow_placeholder_content=repo_root is None)
    return model


def materialize_solution_for_fib(solution: dict[str, Any], repo_root: str | Path) -> Solution:
    """Return a FIB Solution with source contents materialized from TileGym JSON."""
    return TileGymSolution.model_validate(solution).to_fib_solution(repo_root)


def definition_to_tilegym_json(definition: Definition) -> dict[str, Any]:
    """Return stable TileGym JSON data for a Definition."""
    return _drop_none_except_shape(definition.model_dump(mode="json"))


def solution_to_tilegym_json(solution: TileGymSolution, *, path_only_sources: bool = True) -> dict[str, Any]:
    """Return stable TileGym JSON data for a Solution."""
    return solution.to_tilegym_dict(path_only_sources=path_only_sources)


def write_definition_json(definition: Definition, path: str | Path) -> None:
    """Write a Definition JSON file using TileGym's stable formatting."""
    _write_stable_json(definition_to_tilegym_json(definition), path)


def write_solution_json(solution: TileGymSolution, path: str | Path, *, path_only_sources: bool = True) -> None:
    """Write a Solution JSON file using TileGym's stable formatting."""
    _write_stable_json(solution_to_tilegym_json(solution, path_only_sources=path_only_sources), path)


def _drop_none_except_shape(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for child_key, child_value in value.items():
            cleaned_value = _drop_none_except_shape(child_value, key=child_key)
            if cleaned_value is None and child_key != "shape":
                continue
            cleaned[child_key] = cleaned_value
        return cleaned
    if isinstance(value, list):
        return [_drop_none_except_shape(item) for item in value]
    if key == "shape" and value is None:
        return None
    return value


def _write_stable_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
