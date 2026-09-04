# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Path-derived identities and pairings for TileGym kernel inventories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from typing import Literal

InventoryLevel = Literal["legacy", "public", "wrapper", "leaf"]
InventoryKind = Literal["definition", "solution", "workload"]


@dataclass(frozen=True)
class InventoryCoordinate:
    """Canonical, path-derived coordinate for one inventory metadata file."""

    inventory_root: Path
    kind: InventoryKind
    path: Path
    relative_path: Path
    operation: str | None
    backend: str | None
    local_name: str
    level: InventoryLevel

    @property
    def canonical_id(self) -> str:
        """Return a repository-stable identity without relying on local names."""
        parts = self.inventory_root.parts
        try:
            source_index = parts.index("src")
        except ValueError:
            root = self.inventory_root.as_posix()
        else:
            root = Path(*parts[source_index:]).as_posix()
        relative = self.relative_path.with_suffix("").as_posix()
        return f"{root}::{self.kind}::{relative}"


def inventory_coordinate(path: str | Path) -> InventoryCoordinate:
    """Classify an inventory path using only its checked-in hierarchy."""
    path = Path(path)
    anchor = _inventory_anchor(path)
    kind_by_anchor: dict[str, InventoryKind] = {
        "kernel_definitions": "definition",
        "kernel_solutions": "solution",
        "kernel_workloads": "workload",
    }
    kind = kind_by_anchor[anchor.name]
    if kind == "workload":
        if path.name != "workload.jsonl":
            raise ValueError(f"Expected workload.jsonl for workload inventory path: {path}")
        storage_relative = path.relative_to(anchor)
        semantic_parts = storage_relative.parent.parts
        if not semantic_parts:
            raise ValueError(f"Workload must live in a semantic coordinate directory: {path}")
        relative = Path(*semantic_parts).with_suffix(".jsonl")
        parts = semantic_parts
        local_name = semantic_parts[-1]
    else:
        if path.suffix != ".json":
            raise ValueError(f"Expected .json for {kind} inventory path: {path}")
        relative = path.relative_to(anchor)
        parts = relative.parts
        local_name = path.stem
    if len(parts) == 1:
        operation = None
        backend = None
        level: InventoryLevel = "legacy"
    elif kind == "solution" and len(parts) == 2 and parts[0] in {"triton", "cutile", "cutile_rs"}:
        operation = None
        backend = parts[0]
        level = "legacy"
    elif len(parts) == 2:
        if local_name != parts[0]:
            raise ValueError(f"Hierarchical public inventory file must be named after its operation directory: {path}")
        operation = parts[0]
        backend = None
        level = "public"
    elif len(parts) == 3:
        operation, backend = parts[:2]
        level = "wrapper" if local_name == operation else "leaf"
    else:
        raise ValueError(f"Unsupported kernel inventory hierarchy: {path}")
    return InventoryCoordinate(
        inventory_root=anchor.parent,
        kind=kind,
        path=path,
        relative_path=relative,
        operation=operation,
        backend=backend,
        local_name=local_name,
        level=level,
    )


def iter_inventory_json_paths(root: str | Path, directory_name: str) -> Iterator[Path]:
    """Yield JSON files recursively below every matching inventory directory."""
    if directory_name not in {"kernel_definitions", "kernel_solutions"}:
        raise ValueError(f"Expected kernel_definitions or kernel_solutions, got {directory_name!r}")
    root = Path(root)
    paths = set()
    for directory in root.glob(f"src/tilegym/**/{directory_name}"):
        if not directory.is_dir() or directory.name != directory_name:
            continue
        sibling_name = "kernel_solutions" if directory_name == "kernel_definitions" else "kernel_definitions"
        if not (directory.parent / sibling_name).is_dir():
            continue
        paths.update(directory.rglob("*.json"))
    yield from sorted(paths)


def iter_inventory_workload_paths(root: str | Path) -> Iterator[Path]:
    """Yield Workload JSONL files from active Definition/Solution inventories."""
    root = Path(root)
    paths = set()
    for directory in root.glob("src/tilegym/**/kernel_workloads"):
        if not directory.is_dir() or directory.name != "kernel_workloads":
            continue
        if not (directory.parent / "kernel_definitions").is_dir():
            continue
        if not (directory.parent / "kernel_solutions").is_dir():
            continue
        for path in directory.rglob("*.jsonl"):
            inventory_coordinate(path)
            paths.add(path)
    yield from sorted(paths)


def solution_paths_for_definition(definition_path: str | Path) -> Iterator[Path]:
    """Resolve legacy or hierarchical Solution pairings for one Definition.

    Public Definitions pair with every registered wrapper entrance. Wrapper
    Definitions pair only with their same-backend entrance, and leaf
    Definitions pair only with the exact mirrored leaf Solution.
    """
    definition = Path(definition_path)
    coordinate = inventory_coordinate(definition)
    if coordinate.kind != "definition":
        raise ValueError(f"Expected a Definition path: {definition}")
    solution_root = coordinate.inventory_root / "kernel_solutions"

    if coordinate.level == "legacy":
        transformer_solution = solution_root / definition.name
        if transformer_solution.is_file():
            yield transformer_solution
        for backend in ("triton", "cutile", "cutile_rs"):
            suite_solution = solution_root / backend / definition.name
            if suite_solution.is_file():
                yield suite_solution
        return

    assert coordinate.operation is not None
    operation = coordinate.operation
    if coordinate.level == "public":
        operation_solution_root = solution_root / operation
        candidates = [
            path
            for path in operation_solution_root.glob(f"*/{operation}.json")
            if inventory_coordinate(path).level == "wrapper"
        ]
        yield from sorted(candidates, key=_wrapper_solution_sort_key)
        return

    assert coordinate.backend is not None
    if coordinate.level == "wrapper":
        solution = solution_root / operation / coordinate.backend / f"{operation}.json"
        if solution.is_file():
            yield solution
        return

    solution = solution_root / operation / coordinate.backend / definition.name
    if solution.is_file():
        yield solution


def mirrored_workload_path(definition_path: str | Path) -> Path:
    """Return the adjacent Workload path for one Definition."""
    definition = Path(definition_path)
    coordinate = inventory_coordinate(definition)
    if coordinate.kind != "definition":
        raise ValueError(f"Expected a Definition path: {definition}")
    semantic_directory = coordinate.relative_path.with_suffix("")
    return coordinate.inventory_root / "kernel_workloads" / semantic_directory / "workload.jsonl"


def mirrored_definition_path(inventory_path: str | Path) -> Path:
    """Return the exact mirrored Definition for a Solution or Workload."""
    path = Path(inventory_path)
    coordinate = inventory_coordinate(path)
    if coordinate.kind not in {"solution", "workload"}:
        raise ValueError(f"Expected a Solution or Workload path: {path}")
    return coordinate.inventory_root / "kernel_definitions" / coordinate.relative_path.with_suffix(".json")


def definition_solution_paths_for_workload(workload_path: str | Path) -> Iterator[tuple[Path, Path]]:
    """Yield the adjacent Definition and its accepted Solutions for a Workload."""
    workload = Path(workload_path)
    coordinate = inventory_coordinate(workload)
    if coordinate.kind != "workload":
        raise ValueError(f"Expected a Workload path: {workload}")
    definition = mirrored_definition_path(workload)
    if not definition.is_file():
        raise ValueError(f"Missing mirrored Definition for Workload {workload}: {definition}")
    for solution in solution_paths_for_definition(definition):
        yield definition, solution


def validate_hierarchical_operation_topology(
    inventory_root: str | Path,
    operation: str,
    backends: tuple[str, ...] | list[str],
) -> None:
    """Require the three-level topology and accepted local pairings.

    ``inventory_root`` is the package or suite directory containing sibling
    ``kernel_definitions`` and ``kernel_solutions`` directories. Backends must
    come from the caller's authoritative registration inventory.
    """
    root = Path(inventory_root)
    definition_root = root / "kernel_definitions"
    solution_root = root / "kernel_solutions"
    public = definition_root / operation / f"{operation}.json"
    if not public.is_file():
        raise ValueError(f"Missing hierarchical public Definition: {public}")
    if inventory_coordinate(public).level != "public":
        raise ValueError(f"Invalid hierarchical public Definition path: {public}")

    wrapper_solutions = []
    wrapper_definitions = []
    for backend in backends:
        wrapper_definition = definition_root / operation / backend / f"{operation}.json"
        wrapper_solution = solution_root / operation / backend / f"{operation}.json"
        if not wrapper_definition.is_file():
            raise ValueError(f"Missing {backend} wrapper Definition: {wrapper_definition}")
        if not wrapper_solution.is_file():
            raise ValueError(f"Missing {backend} wrapper Solution: {wrapper_solution}")
        wrapper_definitions.append(wrapper_definition)
        wrapper_solutions.append(wrapper_solution)

        backend_definition_dir = wrapper_definition.parent
        backend_solution_dir = wrapper_solution.parent
        leaf_definitions = {path.name for path in backend_definition_dir.glob("*.json") if path != wrapper_definition}
        leaf_solutions = {path.name for path in backend_solution_dir.glob("*.json") if path != wrapper_solution}
        if leaf_definitions != leaf_solutions:
            raise ValueError(
                f"{operation}/{backend} leaf Definition/Solution mismatch: "
                f"definitions_only={sorted(leaf_definitions - leaf_solutions)}, "
                f"solutions_only={sorted(leaf_solutions - leaf_definitions)}"
            )

    expected_public = set(wrapper_solutions)
    actual_public = set(solution_paths_for_definition(public))
    if actual_public != expected_public:
        raise ValueError(
            f"Incomplete public entrance pairing for {public}: "
            f"expected={sorted(expected_public)}, actual={sorted(actual_public)}"
        )
    for definition, solution in zip(wrapper_definitions, wrapper_solutions, strict=True):
        actual = set(solution_paths_for_definition(definition))
        if actual != {solution}:
            raise ValueError(
                f"Invalid backend-local wrapper pairing for {definition}: "
                f"expected={[solution]}, actual={sorted(actual)}"
            )


def _inventory_anchor(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if parent.name in {"kernel_definitions", "kernel_solutions", "kernel_workloads"}:
            return parent
    raise ValueError(f"Path does not live below kernel_definitions, kernel_solutions, or kernel_workloads: {path}")


def _wrapper_solution_sort_key(path: Path) -> tuple[int, str]:
    backend = path.parent.name
    preferred_order = {"triton": 0, "cutile": 1, "cutile_rs": 2}
    return preferred_order.get(backend, len(preferred_order)), backend
