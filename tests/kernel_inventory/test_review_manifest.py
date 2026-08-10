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
    from tilegym.kernel_inventory.review import ReviewManifestError
    from tilegym.kernel_inventory.review import load_review_manifest
    from tilegym.kernel_inventory.review import make_leaf_review_record
    from tilegym.kernel_inventory.review import validate_leaf_review_manifest
finally:
    if _original_tilegym is None:
        sys.modules.pop("tilegym", None)


def _inventory(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "src/tilegym/suites/example"
    public = root / "kernel_definitions/op/op.json"
    leaf = root / "kernel_definitions/op/cutile/leaf.json"
    solution = root / "kernel_solutions/op/cutile/leaf.json"
    for path, value in (
        (public, {"name": "op"}),
        (leaf, {"name": "leaf"}),
        (solution, {"name": "leaf_cutile"}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return public, leaf, solution


def _manifest(leaf: Path, solution: Path, verdict: str = "PASS") -> dict:
    canonical_id, record = make_leaf_review_record(
        leaf,
        solution,
        author="author-agent",
        reviewer="review-agent",
        verdict=verdict,
        reviewed_commit="a" * 40,
    )
    return {"version": 1, "reviews": {canonical_id: record}}


def test_review_manifest_requires_current_independent_pass_for_every_leaf(tmp_path):
    public, leaf, solution = _inventory(tmp_path)
    manifest = _manifest(leaf, solution)
    validate_leaf_review_manifest(manifest, [public, leaf])


def test_review_manifest_rejects_need_refine_at_acceptance(tmp_path):
    _public, leaf, solution = _inventory(tmp_path)
    manifest = _manifest(leaf, solution, verdict="NEED REFINE")
    validate_leaf_review_manifest(manifest, [leaf], require_pass=False)
    with pytest.raises(ReviewManifestError, match="expected PASS"):
        validate_leaf_review_manifest(manifest, [leaf])


def test_review_manifest_rejects_self_approval(tmp_path):
    _public, leaf, solution = _inventory(tmp_path)
    manifest = _manifest(leaf, solution)
    record = next(iter(manifest["reviews"].values()))
    record["reviewer"] = "AUTHOR-AGENT"
    with pytest.raises(ReviewManifestError, match="self-approve"):
        validate_leaf_review_manifest(manifest, [leaf])


def test_review_manifest_rejects_stale_definition_or_solution(tmp_path):
    _public, leaf, solution = _inventory(tmp_path)
    manifest = _manifest(leaf, solution)
    leaf.write_text('{"name": "changed"}\n', encoding="utf-8")
    with pytest.raises(ReviewManifestError, match="Definition changed"):
        validate_leaf_review_manifest(manifest, [leaf])

    manifest = _manifest(leaf, solution)
    solution.write_text('{"name": "changed"}\n', encoding="utf-8")
    with pytest.raises(ReviewManifestError, match="Solution changed"):
        validate_leaf_review_manifest(manifest, [leaf])


def test_review_manifest_requires_exact_leaf_coverage(tmp_path):
    _public, leaf, solution = _inventory(tmp_path)
    manifest = _manifest(leaf, solution)
    manifest["reviews"] = {}
    with pytest.raises(ReviewManifestError, match="missing leaf reviews"):
        validate_leaf_review_manifest(manifest, [leaf])

    manifest = _manifest(leaf, solution)
    manifest["reviews"]["unknown"] = next(iter(manifest["reviews"].values())).copy()
    with pytest.raises(ReviewManifestError, match="unknown or non-leaf"):
        validate_leaf_review_manifest(manifest, [leaf])


def test_review_manifest_tracks_same_entry_point_semantic_variants_independently(tmp_path):
    root = tmp_path / "src/tilegym/suites/example"
    leaves = []
    reviews = {}
    for name in ("raw_kernel__beta_bf16", "raw_kernel__beta_fp32"):
        leaf = root / f"kernel_definitions/op/cutile/{name}.json"
        solution = root / f"kernel_solutions/op/cutile/{name}.json"
        leaf.parent.mkdir(parents=True, exist_ok=True)
        solution.parent.mkdir(parents=True, exist_ok=True)
        leaf.write_text(json.dumps({"name": name}) + "\n", encoding="utf-8")
        solution.write_text(
            json.dumps(
                {
                    "name": f"{name}_cutile",
                    "definition": name,
                    "spec": {"entry_point": "src/example.py::raw_kernel"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        canonical_id, record = make_leaf_review_record(
            leaf,
            solution,
            author="author-agent",
            reviewer="review-agent",
            verdict="PASS",
            reviewed_commit="a" * 40,
        )
        leaves.append(leaf)
        reviews[canonical_id] = record

    assert len(reviews) == 2
    validate_leaf_review_manifest({"version": 1, "reviews": reviews}, leaves)


def test_load_review_manifest_requires_json_object(tmp_path):
    path = tmp_path / "reviews.json"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ReviewManifestError, match="JSON object"):
        load_review_manifest(path)
