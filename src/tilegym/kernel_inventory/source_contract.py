# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Shared dependency-light checks for kernel inventory metadata."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from pathlib import PurePosixPath

SOURCE_PREFIX = "# Source: "
GITHUB_LINE_ANCHOR_RE = re.compile(r"#L\d+(?:-L\d+)?$")
BLOB_COMMIT_RE = re.compile(r"/blob/[0-9a-f]{40}/")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class SourceRepository(str, Enum):
    """Repositories supported by the TileGym source permalink generator."""

    TILEGYM_GITHUB = "tilegym-github"


_REPOSITORY_BASE_URLS = {
    SourceRepository.TILEGYM_GITHUB: "https://github.com/NVIDIA/TileGym",
}


class SourceContractError(ValueError):
    """Raised when a Definition.reference source contract is invalid."""


def is_precise_source_permalink(url: str) -> bool:
    """Return whether ``url`` is a precise supported source permalink."""
    if url.startswith(("https://github.com/", "https://huggingface.co/")):
        return bool(GITHUB_LINE_ANCHOR_RE.search(url) and BLOB_COMMIT_RE.search(url))
    return False


def make_pinned_source_permalink(
    *,
    repo_kind: SourceRepository | str,
    commit: str,
    path: str,
    start_line: int,
    end_line: int | None = None,
) -> str:
    """Construct a precise permalink into a supported TileGym source repository."""
    try:
        repository = SourceRepository(repo_kind)
    except ValueError as exc:
        supported = ", ".join(repo.value for repo in SourceRepository)
        raise ValueError(f"Unsupported source repository '{repo_kind}'; expected one of: {supported}") from exc

    if COMMIT_RE.fullmatch(commit) is None:
        raise ValueError("Source commit must be a lowercase 40-hex commit SHA")

    source_path = PurePosixPath(path)
    if not path or source_path.is_absolute() or ".." in source_path.parts or "\\" in path or "#" in path or "?" in path:
        raise ValueError(f"Source path must be a repository-relative POSIX path: {path}")

    if not isinstance(start_line, int) or isinstance(start_line, bool) or start_line < 1:
        raise ValueError("Source start_line must be a positive integer")
    if end_line is not None and (not isinstance(end_line, int) or isinstance(end_line, bool) or end_line < start_line):
        raise ValueError("Source end_line must be an integer greater than or equal to start_line")

    blob_segment = "blob"
    anchor = f"#L{start_line}" if end_line is None else f"#L{start_line}-L{end_line}"
    return f"{_REPOSITORY_BASE_URLS[repository]}/{blob_segment}/{commit}/{source_path.as_posix()}{anchor}"


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
    """Require leading precise supported source permalinks."""
    for url in leading_reference_source_urls(reference):
        if not is_precise_source_permalink(url):
            raise SourceContractError(
                "Definition.reference source must be a GitHub or Hugging Face permalink "
                f"pinned to a 40-hex commit with line anchors: {url}"
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
