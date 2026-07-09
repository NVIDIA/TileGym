# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Shared dependency-light checks for kernel inventory metadata."""

from __future__ import annotations

import re
from pathlib import Path

SOURCE_PREFIX = "# Source: "
LINE_ANCHOR_RE = re.compile(r"#L\d+(?:-L\d+)?$")
BLOB_COMMIT_RE = re.compile(r"/blob/[0-9a-f]{40}/")


class SourceContractError(ValueError):
    """Raised when a Definition.reference source contract is invalid."""


def is_precise_source_permalink(url: str) -> bool:
    """Return whether ``url`` is a precise GitHub or Hugging Face source permalink."""
    if not LINE_ANCHOR_RE.search(url):
        return False
    if url.startswith(("https://github.com/", "https://huggingface.co/")):
        return bool(BLOB_COMMIT_RE.search(url))
    return False


def leading_reference_source_urls(reference: str) -> list[str]:
    """Return leading ``# Source:`` URLs after validating their required position."""
    lines = reference.splitlines()
    if not lines:
        raise SourceContractError("Definition.reference must not be empty")
    if not lines[0].startswith(SOURCE_PREFIX):
        raise SourceContractError("Definition.reference must begin with '# Source:'")

    source_urls = []
    for line in lines:
        if not line.startswith(SOURCE_PREFIX):
            break
        source_urls.append(line.removeprefix(SOURCE_PREFIX).strip())

    if not source_urls:
        raise SourceContractError("Definition.reference must include at least one '# Source:' comment")
    return source_urls


def validate_reference_source_contract(reference: str) -> None:
    """Require leading precise GitHub/Hugging Face source permalinks."""
    for url in leading_reference_source_urls(reference):
        if not is_precise_source_permalink(url):
            raise SourceContractError(
                "Definition.reference source must be a GitHub or Hugging Face "
                f"/blob/<40-hex-commit>/ permalink with line anchors: {url}"
            )


def resolve_repo_relative_path(root: Path, path: str, label: str) -> Path:
    """Resolve a repo-relative path while rejecting absolute or escaping paths."""
    raw_path = Path(path)
    if not path or raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError(f"{label} must be a repo-relative path: {path}")

    resolved = (root / raw_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside repo root: {path}") from exc
    return resolved
