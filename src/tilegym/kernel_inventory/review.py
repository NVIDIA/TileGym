# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Independent-review records for hierarchical leaf kernel inventories."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from typing import Iterable

from tilegym.kernel_inventory.layout import inventory_coordinate
from tilegym.kernel_inventory.layout import mirrored_definition_path
from tilegym.kernel_inventory.layout import solution_paths_for_definition

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_VERDICTS = {"PASS", "NEED REFINE"}


class ReviewManifestError(ValueError):
    """Raised when a leaf review manifest is incomplete, stale, or invalid."""


def file_sha256(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for a checked-in inventory file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_leaf_review_record(
    definition_path: str | Path,
    solution_path: str | Path,
    *,
    author: str,
    reviewer: str,
    verdict: str,
    reviewed_commit: str,
) -> tuple[str, dict[str, str]]:
    """Build one digest-bound review record keyed by canonical leaf identity."""
    definition_path = Path(definition_path)
    solution_path = Path(solution_path)
    coordinate = inventory_coordinate(definition_path)
    if coordinate.kind != "definition" or coordinate.level != "leaf":
        raise ReviewManifestError(f"Review records require a leaf Definition: {definition_path}")
    if mirrored_definition_path(solution_path) != definition_path:
        raise ReviewManifestError(
            f"Review Solution does not mirror leaf Definition: {solution_path} != {definition_path}"
        )
    record = {
        "author": author,
        "reviewer": reviewer,
        "verdict": verdict,
        "reviewed_commit": reviewed_commit,
        "definition_sha256": file_sha256(definition_path),
        "solution_sha256": file_sha256(solution_path),
    }
    _validate_record(coordinate.canonical_id, record, require_pass=False)
    return coordinate.canonical_id, record


def load_review_manifest(path: str | Path) -> dict[str, Any]:
    """Load a review manifest JSON object."""
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ReviewManifestError(f"{path}: review manifest must be a JSON object")
    return manifest


def validate_leaf_review_manifest(
    manifest: dict[str, Any],
    definition_paths: Iterable[str | Path],
    *,
    require_pass: bool = True,
) -> None:
    """Require one current, independent review for every supplied leaf.

    Content digests make a previous verdict stale as soon as either side of the
    reviewed Definition/Solution pair changes. ``reviewed_commit`` records the
    source snapshot inspected by the reviewer without requiring Git at runtime.
    """
    if not isinstance(manifest, dict):
        raise ReviewManifestError("Review manifest must be a mapping")
    if manifest.get("version") != 1:
        raise ReviewManifestError("Review manifest version must be 1")
    reviews = manifest.get("reviews")
    if not isinstance(reviews, dict):
        raise ReviewManifestError("Review manifest reviews must be a mapping")

    leaves: dict[str, Path] = {}
    for raw_path in definition_paths:
        path = Path(raw_path)
        coordinate = inventory_coordinate(path)
        if coordinate.kind == "definition" and coordinate.level == "leaf":
            leaves[coordinate.canonical_id] = path

    missing = sorted(set(leaves) - set(reviews))
    extra = sorted(set(reviews) - set(leaves))
    if missing:
        raise ReviewManifestError(f"Review manifest is missing leaf reviews: {missing}")
    if extra:
        raise ReviewManifestError(f"Review manifest contains unknown or non-leaf reviews: {extra}")

    for canonical_id, definition_path in leaves.items():
        record = reviews[canonical_id]
        _validate_record(canonical_id, record, require_pass=require_pass)
        solutions = list(solution_paths_for_definition(definition_path))
        if len(solutions) != 1:
            raise ReviewManifestError(
                f"{canonical_id}: leaf must have exactly one mirrored Solution, found {len(solutions)}"
            )
        solution_path = solutions[0]
        if record["definition_sha256"] != file_sha256(definition_path):
            raise ReviewManifestError(f"{canonical_id}: Definition changed after review")
        if record["solution_sha256"] != file_sha256(solution_path):
            raise ReviewManifestError(f"{canonical_id}: Solution changed after review")


def _validate_record(canonical_id: str, record: Any, *, require_pass: bool) -> None:
    if not isinstance(record, dict):
        raise ReviewManifestError(f"{canonical_id}: review record must be a mapping")
    expected = {
        "author",
        "reviewer",
        "verdict",
        "reviewed_commit",
        "definition_sha256",
        "solution_sha256",
    }
    unknown = sorted(set(record) - expected)
    missing = sorted(expected - set(record))
    if missing or unknown:
        raise ReviewManifestError(f"{canonical_id}: review fields mismatch; missing={missing}, unknown={unknown}")
    author = record["author"]
    reviewer = record["reviewer"]
    if not isinstance(author, str) or not author.strip():
        raise ReviewManifestError(f"{canonical_id}: author must be a nonempty string")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ReviewManifestError(f"{canonical_id}: reviewer must be a nonempty string")
    if author.strip().casefold() == reviewer.strip().casefold():
        raise ReviewManifestError(f"{canonical_id}: leaf author cannot self-approve")
    verdict = record["verdict"]
    if verdict not in _VERDICTS:
        raise ReviewManifestError(f"{canonical_id}: verdict must be one of {sorted(_VERDICTS)}")
    if require_pass and verdict != "PASS":
        raise ReviewManifestError(f"{canonical_id}: review verdict is {verdict}, expected PASS")
    if not isinstance(record["reviewed_commit"], str) or not _COMMIT_RE.fullmatch(record["reviewed_commit"]):
        raise ReviewManifestError(f"{canonical_id}: reviewed_commit must be a full lowercase Git SHA")
    for field in ("definition_sha256", "solution_sha256"):
        if not isinstance(record[field], str) or not _DIGEST_RE.fullmatch(record[field]):
            raise ReviewManifestError(f"{canonical_id}: {field} must be a lowercase SHA-256 digest")
