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


@dataclass(frozen=True)
class InventoryCoordinate:
    """Canonical, path-derived coordinate for one inventory JSON file."""

    inventory_root: Path
    kind: Literal["definition", "solution"]
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
    kind: Literal["definition", "solution"] = "definition" if anchor.name == "kernel_definitions" else "solution"
    relative = path.relative_to(anchor)
    parts = relative.parts
    if len(parts) == 1:
        operation = None
        backend = None
        level: InventoryLevel = "legacy"
    elif kind == "solution" and len(parts) == 2 and parts[0] in {"triton", "cutile", "cutile_rs"}:
        operation = None
        backend = parts[0]
        level = "legacy"
    elif len(parts) == 2:
        if path.stem != parts[0]:
            raise ValueError(f"Hierarchical public Definition must be named after its operation directory: {path}")
        operation = parts[0]
        backend = None
        level = "public"
    elif len(parts) == 3:
        operation, backend = parts[:2]
        level = "wrapper" if Path(parts[2]).stem == operation else "leaf"
    else:
        raise ValueError(f"Unsupported kernel inventory hierarchy: {path}")
    return InventoryCoordinate(
        inventory_root=anchor.parent,
        kind=kind,
        path=path,
        relative_path=relative,
        operation=operation,
        backend=backend,
        local_name=path.stem,
        level=level,
    )


def iter_inventory_json_paths(root: str | Path, directory_name: str) -> Iterator[Path]:
    """Yield JSON files recursively below every matching inventory directory."""
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


def solution_paths_for_definition(definition_path: str | Path) -> Iterator[Path]:
    """Resolve legacy or hierarchical Solution pairings for one Definition.

    Hierarchical public and wrapper Definitions implement the literal acceptance
    matrix: all three wrapper-level semantic contracts are checked against both
    backend wrapper Solutions. Leaf Definitions pair only with the mirrored
    backend leaf Solution.
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
    if coordinate.level in {"public", "wrapper"}:
        for backend in ("triton", "cutile"):
            solution = solution_root / operation / backend / f"{operation}.json"
            if solution.is_file():
                yield solution
        return

    assert coordinate.backend is not None
    solution = solution_root / operation / coordinate.backend / definition.name
    if solution.is_file():
        yield solution


def mirrored_definition_path(solution_path: str | Path) -> Path:
    """Return the exact mirrored Definition for a hierarchical leaf Solution."""
    solution = Path(solution_path)
    coordinate = inventory_coordinate(solution)
    if coordinate.kind != "solution":
        raise ValueError(f"Expected a Solution path: {solution}")
    return coordinate.inventory_root / "kernel_definitions" / coordinate.relative_path


def validate_hierarchical_operation_topology(
    inventory_root: str | Path,
    operation: str,
    backends: tuple[str, ...] | list[str],
) -> None:
    """Require the literal three-level topology and wrapper acceptance matrix.

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

    expected_matrix = set(wrapper_solutions)
    for definition in (public, *wrapper_definitions):
        actual = set(solution_paths_for_definition(definition))
        if actual != expected_matrix:
            raise ValueError(
                f"Incomplete wrapper acceptance matrix for {definition}: "
                f"expected={sorted(expected_matrix)}, actual={sorted(actual)}"
            )


def _inventory_anchor(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if parent.name in {"kernel_definitions", "kernel_solutions"}:
            return parent
    raise ValueError(f"Path does not live below kernel_definitions or kernel_solutions: {path}")
