# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_original_tilegym = sys.modules.get("tilegym")
if _original_tilegym is None:
    tilegym_pkg = types.ModuleType("tilegym")
    tilegym_pkg.__path__ = [str(REPO_ROOT / "src/tilegym")]
    sys.modules["tilegym"] = tilegym_pkg
try:
    from tilegym.kernel_inventory.layout import inventory_coordinate
    from tilegym.kernel_inventory.layout import iter_inventory_json_paths
    from tilegym.kernel_inventory.layout import solution_paths_for_definition
    from tilegym.kernel_inventory.layout import validate_hierarchical_operation_topology
finally:
    if _original_tilegym is None:
        sys.modules.pop("tilegym", None)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_hierarchical_coordinates_are_path_scoped(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    public = _touch(inventory / "kernel_definitions/op/op.json")
    wrapper = _touch(inventory / "kernel_definitions/op/cutile/op.json")
    leaf = _touch(inventory / "kernel_definitions/op/cutile/leaf.json")
    (inventory / "kernel_solutions").mkdir()

    public_coordinate = inventory_coordinate(public)
    wrapper_coordinate = inventory_coordinate(wrapper)
    leaf_coordinate = inventory_coordinate(leaf)

    assert public_coordinate.level == "public"
    assert wrapper_coordinate.level == "wrapper"
    assert leaf_coordinate.level == "leaf"
    assert len({public_coordinate.canonical_id, wrapper_coordinate.canonical_id, leaf_coordinate.canonical_id}) == 3


def test_wrapper_acceptance_matrix_and_leaf_pairing(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    public = _touch(inventory / "kernel_definitions/op/op.json")
    cutile_wrapper = _touch(inventory / "kernel_definitions/op/cutile/op.json")
    triton_wrapper = _touch(inventory / "kernel_definitions/op/triton/op.json")
    cutile_leaf = _touch(inventory / "kernel_definitions/op/cutile/leaf.json")
    cutile_solution = _touch(inventory / "kernel_solutions/op/cutile/op.json")
    triton_solution = _touch(inventory / "kernel_solutions/op/triton/op.json")
    leaf_solution = _touch(inventory / "kernel_solutions/op/cutile/leaf.json")

    expected_wrappers = [triton_solution, cutile_solution]
    assert list(solution_paths_for_definition(public)) == expected_wrappers
    assert list(solution_paths_for_definition(cutile_wrapper)) == expected_wrappers
    assert list(solution_paths_for_definition(triton_wrapper)) == expected_wrappers
    assert list(solution_paths_for_definition(cutile_leaf)) == [leaf_solution]


def test_leaf_semantic_variants_can_share_one_raw_entry_point_without_collapsing(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    _touch(inventory / "kernel_definitions/op/op.json")
    _touch(inventory / "kernel_definitions/op/cutile/op.json")
    _touch(inventory / "kernel_solutions/op/cutile/op.json")
    variants = (("raw_kernel__beta_bf16", "bfloat16"), ("raw_kernel__beta_fp32", "float32"))
    definition_paths = []
    solution_paths = []
    for name, dtype in variants:
        definition_path = inventory / f"kernel_definitions/op/cutile/{name}.json"
        definition_path.parent.mkdir(parents=True, exist_ok=True)
        definition_path.write_text(
            json.dumps({"name": name, "inputs": {"beta": {"shape": ["N"], "dtype": dtype}}}) + "\n",
            encoding="utf-8",
        )
        definition_paths.append(definition_path)
        solution_path = inventory / f"kernel_solutions/op/cutile/{name}.json"
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        solution_path.write_text(
            json.dumps({"definition": name, "spec": {"entry_point": "src/example.py::raw_kernel"}}) + "\n",
            encoding="utf-8",
        )
        solution_paths.append(solution_path)

    discovered = set(iter_inventory_json_paths(tmp_path, "kernel_definitions"))
    assert set(definition_paths) <= discovered
    assert len({inventory_coordinate(path).canonical_id for path in definition_paths}) == 2
    assert {json.loads(path.read_text(encoding="utf-8"))["inputs"]["beta"]["dtype"] for path in definition_paths} == {
        "bfloat16",
        "float32",
    }
    assert [list(solution_paths_for_definition(path)) for path in definition_paths] == [
        [solution_paths[0]],
        [solution_paths[1]],
    ]
    assert {json.loads(path.read_text(encoding="utf-8"))["spec"]["entry_point"] for path in solution_paths} == {
        "src/example.py::raw_kernel"
    }
    validate_hierarchical_operation_topology(inventory, "op", ["cutile"])


def test_recursive_discovery_ignores_archived_directories(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    definition = _touch(inventory / "kernel_definitions/op/op.json")
    _touch(inventory / "_kernel_definitions/archived.json")
    (inventory / "kernel_solutions").mkdir(parents=True)

    assert list(iter_inventory_json_paths(tmp_path, "kernel_definitions")) == [definition]


def test_legacy_cutile_rs_solution_coordinate_and_pairing(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    definition = _touch(inventory / "kernel_definitions/op.json")
    solution = _touch(inventory / "kernel_solutions/cutile_rs/op.json")

    coordinate = inventory_coordinate(solution)
    assert coordinate.level == "legacy"
    assert coordinate.backend == "cutile_rs"
    assert list(solution_paths_for_definition(definition)) == [solution]


def test_hierarchical_operation_topology_requires_literal_matrix_and_leaf_pairs(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    public = _touch(inventory / "kernel_definitions/op/op.json")
    del public
    for backend in ("triton", "cutile"):
        _touch(inventory / f"kernel_definitions/op/{backend}/op.json")
        _touch(inventory / f"kernel_solutions/op/{backend}/op.json")
        _touch(inventory / f"kernel_definitions/op/{backend}/leaf.json")
        _touch(inventory / f"kernel_solutions/op/{backend}/leaf.json")

    validate_hierarchical_operation_topology(inventory, "op", ["triton", "cutile"])

    (inventory / "kernel_solutions/op/cutile/leaf.json").unlink()
    with pytest.raises(ValueError, match="leaf Definition/Solution mismatch"):
        validate_hierarchical_operation_topology(inventory, "op", ["triton", "cutile"])


def test_hierarchical_operation_topology_rejects_missing_wrapper_matrix_member(tmp_path):
    inventory = tmp_path / "src/tilegym/suites/example"
    _touch(inventory / "kernel_definitions/op/op.json")
    for backend in ("triton", "cutile"):
        _touch(inventory / f"kernel_definitions/op/{backend}/op.json")
    _touch(inventory / "kernel_solutions/op/triton/op.json")

    with pytest.raises(ValueError, match="Missing cutile wrapper Solution"):
        validate_hierarchical_operation_topology(inventory, "op", ["triton", "cutile"])


def test_hierarchical_public_path_must_match_operation_directory(tmp_path):
    path = _touch(tmp_path / "src/tilegym/suites/example/kernel_definitions/op/wrong.json")
    with pytest.raises(ValueError, match="named after its operation"):
        inventory_coordinate(path)
