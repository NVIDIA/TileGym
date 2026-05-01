#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Script to check and add SPDX license headers to source files.

Usage:
    python check_spdx_headers.py --action check
    python check_spdx_headers.py --action write
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

# Year constants for copyright validation
MIN_COPYRIGHT_YEAR = 2025  # TileGym project inception year
CURRENT_YEAR = datetime.now().year

# SPDX header content — uses the current year for newly added headers
SPDX_COPYRIGHT = (
    f"SPDX-FileCopyrightText: Copyright (c) {CURRENT_YEAR} NVIDIA CORPORATION & AFFILIATES. All rights reserved."
)
# Default SPDX license identifier line for the main repo (MIT).
SPDX_LICENSE = "SPDX-License-Identifier: MIT"
# SPDX license identifier line used for skill files (under ``.agents/skills/``
# and the ``.claude/skills`` symlink). These files are dual-licensed under
# CC-BY-4.0 (documentation) AND Apache-2.0 (source code) per the NVIDIA
# Skills Publishing Onboarding guide and the OSRB-approved CC-BY-4.0-Apache2
# Dual License pattern.
SPDX_LICENSE_SKILLS = "SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0"

# Regex pattern to validate SPDX copyright lines with any valid year or year range
SPDX_COPYRIGHT_PATTERN = re.compile(
    r"SPDX-FileCopyrightText: Copyright \(c\) (\d{4})(?:-(\d{4}))? NVIDIA CORPORATION & AFFILIATES\. All rights reserved\."
)

# License identifiers accepted by the SPDX check.
#
# Public / exportable code (default): MIT only — matches the repo-wide license
# for everything that is not a dual-licensed agent skill.
#
# Skill content (under ``.agents/skills/``): the dual-licensed combination
# ``CC-BY-4.0 AND Apache-2.0`` only. We deliberately do not accept MIT here
# so that the gate catches any skill file that was authored before the
# relicensing or imported from elsewhere with a stale header.
ALLOWED_LICENSES_DEFAULT: Tuple[str, ...] = ("MIT",)
ALLOWED_LICENSES_SKILLS: Tuple[str, ...] = ("CC-BY-4.0 AND Apache-2.0",)

# Directory names (anywhere under root) to skip entirely.
#
# ``.agents`` and ``.claude`` are skipped from the default walker because
# they are dual-licensed and therefore cannot use the default MIT header.
# Skill files under those directories are processed separately via
# :func:`iter_skill_files` and :func:`iter_skill_content_files`, both of
# which target ``.agents/skills/`` (the canonical path; ``.claude/skills``
# is a symlink to ``../.agents/skills`` for agent-tool compatibility).
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "venv",
    "env",
    ".egg-info",
    "dist",
    "build",
    ".agents",
    ".claude",
}

# Specific path prefixes (relative to root) to skip, even if the leaf directory
# name is not globally ignored above.
SKIP_PATH_PREFIXES: set = set()

# File extensions that should NOT be forced to carry SPDX headers.
# JSON, for example, does not have a standard comment syntax, so we
# avoid inserting header lines that would break parsers.
SKIP_EXTENSIONS = {
    ".json",
}


# Comment styles for different file types
COMMENT_STYLES: Dict[str, Tuple[str, str, str]] = {
    # Extension: (prefix, middle, suffix)
    # For single-line comments: prefix is the comment marker, middle/suffix are empty
    # For multi-line comments: prefix is opening, middle is for middle lines, suffix is closing
    # Python, Shell, YAML, Makefile, etc.
    ".py": ("#", "#", ""),
    ".sh": ("#", "#", ""),
    ".yml": ("#", "#", ""),
    ".yaml": ("#", "#", ""),
    ".cmake": ("#", "#", ""),
    "Dockerfile": ("#", "#", ""),
    # C, C++, Java, JavaScript, TypeScript, etc.
    ".c": ("//", "//", ""),
    ".cc": ("//", "//", ""),
    ".cpp": ("//", "//", ""),
    ".cxx": ("//", "//", ""),
    ".h": ("//", "//", ""),
    ".hpp": ("//", "//", ""),
    ".hxx": ("//", "//", ""),
    ".cu": ("//", "//", ""),
    ".cuh": ("//", "//", ""),
    ".java": ("//", "//", ""),
    ".js": ("//", "//", ""),
    ".jsx": ("//", "//", ""),
    ".ts": ("//", "//", ""),
    ".tsx": ("//", "//", ""),
    ".go": ("//", "//", ""),
    ".rs": ("//", "//", ""),
    ".swift": ("//", "//", ""),
    # CSS, HTML
    ".css": ("/*", "*", "*/"),
    ".scss": ("/*", "*", "*/"),
    ".html": ("<!--", "*", "-->"),
    # Markdown
    ".md": ("<!---", "", "--->"),
    # TOML, INI
    ".toml": ("#", "#", ""),
    ".ini": ("#", "#", ""),
    # Julia
    ".jl": ("#", "#", ""),
}


def _is_under_skip_prefix(rel_path: str) -> bool:
    for prefix in SKIP_PATH_PREFIXES:
        if rel_path == prefix or rel_path.startswith(prefix + os.sep):
            return True
    return False


def should_skip_file(file_path: Path, root_dir: Path) -> bool:
    """Check if a file should be skipped."""
    rel_path = os.path.relpath(str(file_path), str(root_dir))
    parts = rel_path.split(os.sep)

    # Skip any file under a disallowed path prefix
    if _is_under_skip_prefix(rel_path):
        return True

    # Skip if under any ignored directory (by name, anywhere under root)
    for p in parts[:-1]:
        if p in SKIP_DIRS:
            return True

    # Skip .git directory specifically (but not .github)
    if ".git" in file_path.parts:
        return True

    # Skip exact files
    exact_match_patterns = [".gitkeep", ".gitignore", "LICENSE"]
    if file_path.name in exact_match_patterns:
        return True

    # Skip by file extension
    if file_path.suffix.lower() in SKIP_EXTENSIONS:
        return True

    skip_extensions = [
        ".pyc",
        ".pyo",
        ".so",
        ".dylib",
        ".dll",
        ".class",
        ".o",
        ".obj",
        ".exe",
        ".bin",
        ".log",
        ".pem",
        ".sample",
        ".TAG",
    ]
    if file_path.suffix.lower() in skip_extensions:
        return True

    # Skip files without extensions that aren't Dockerfile
    if not file_path.suffix and file_path.name != "Dockerfile":
        return True

    return False


# License field to insert into SKILL.md (and other frontmatter .md) files
# under ``.agents/skills/``. These files are dual-licensed; the YAML
# ``license:`` field carries the same SPDX expression as the in-file SPDX
# comment used for non-frontmatter files.
SKILL_LICENSE_LINE = "license: CC-BY-4.0 AND Apache-2.0"

# Regex matching any ``license:`` line at the start of a YAML line, regardless
# of value. Used to detect (and replace) stale or pre-relicensing entries.
SKILL_LICENSE_LINE_PATTERN = re.compile(r"^\s*license\s*:.*$")


def iter_skill_files(root_dir: Path) -> Iterator[Path]:
    """Yield .md files with YAML frontmatter under .agents/skills/.

    This includes SKILL.md files and any other .md files that start with
    ``---`` frontmatter (e.g. sub-skill definitions).  All yielded files are
    checked/written using the frontmatter ``license:`` field approach.

    SKILL.md files *without* frontmatter are intentionally skipped here so
    that :func:`iter_skill_content_files` can give them a standard SPDX
    comment header instead.

    Note: ``.claude/skills`` is a symlink to ``../.agents/skills`` for
    backward compatibility with agents that hard-code the ``.claude/`` path.
    Walking the canonical ``.agents/skills/`` path avoids double-processing
    the same files via the symlink.
    """
    skills_dir = root_dir / ".agents" / "skills"
    if not skills_dir.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(skills_dir):
        for name in sorted(filenames):
            if not name.endswith(".md"):
                continue
            path = Path(dirpath) / name
            # Only yield .md files that actually have YAML frontmatter.
            if _has_yaml_frontmatter(path):
                yield path


def has_skill_license(content: str, expected_line: str = SKILL_LICENSE_LINE) -> bool:
    """Check whether a SKILL.md frontmatter has the expected ``license:`` line.

    Returns True only when an exact-match (modulo surrounding whitespace)
    ``license:`` entry equal to ``expected_line`` is present in the YAML
    frontmatter.  Stale entries (e.g. ``license: MIT. Complete terms in
    LICENSE.``) intentionally return False so that :func:`add_skill_license`
    will replace them with the current value.
    """
    lines = content.split("\n")
    if not lines or lines[0].strip() != "---":
        return False
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            for fm_line in lines[1:i]:
                if fm_line.strip() == expected_line.strip():
                    return True
            return False
    return False


def add_skill_license(file_path: Path, license_line: str = SKILL_LICENSE_LINE) -> bool:
    """Ensure ``license_line`` is present in the YAML frontmatter.

    If a ``license:`` field is missing, insert ``license_line`` just before
    the closing ``---``.  If a ``license:`` field is present but its value
    does not match ``license_line`` (e.g. a stale ``license: MIT. ...``),
    replace it in place. Returns True iff the file was modified.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if has_skill_license(content, expected_line=license_line):
            return False

        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return False

        # Locate the frontmatter range (lines[1:end_idx] is the body).
        end_idx = None
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = i
                break
        if end_idx is None:
            return False

        # Replace any existing ``license:`` line, or insert before closing ---.
        replaced = False
        new_lines = list(lines)
        for j in range(1, end_idx):
            if SKILL_LICENSE_LINE_PATTERN.match(new_lines[j]):
                new_lines[j] = license_line
                replaced = True
                break
        if not replaced:
            new_lines.insert(end_idx, license_line)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines))
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def _has_yaml_frontmatter(path: Path) -> bool:
    """Return True if *path* begins with a YAML frontmatter delimiter ``---``."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.readline().strip() == "---"
    except Exception:
        return False


def iter_skill_content_files(root_dir: Path) -> Iterator[Path]:
    """Yield .py and non-frontmatter .md files under .agents/skills/ for SPDX headers.

    Files with YAML frontmatter (.md starting with ``---``) are handled by
    :func:`iter_skill_files` using the frontmatter ``license:`` approach.
    Everything else that has a recognised comment style gets a standard SPDX
    comment header (currently dual-licensed CC-BY-4.0 AND Apache-2.0).
    """
    skills_dir = root_dir / ".agents" / "skills"
    if not skills_dir.is_dir():
        return
    for dirpath, _dirnames, filenames in os.walk(skills_dir):
        for name in sorted(filenames):
            path = Path(dirpath) / name
            # Must have a known comment style
            if get_comment_style(path) is None:
                continue
            # .md files with frontmatter are handled by iter_skill_files
            if path.suffix == ".md" and _has_yaml_frontmatter(path):
                continue
            yield path


def get_comment_style(file_path: Path) -> Optional[Tuple[str, str, str]]:
    """Get the comment style for a given file."""
    # Check for Dockerfile specifically
    if file_path.name == "Dockerfile":
        return COMMENT_STYLES.get("Dockerfile")

    # Check by extension
    return COMMENT_STYLES.get(file_path.suffix)


def create_header(
    prefix: str,
    middle: str,
    suffix: str,
    spdx_license: str = SPDX_LICENSE,
) -> List[str]:
    """Create the SPDX header lines based on comment style.

    ``spdx_license`` is the full ``SPDX-License-Identifier: ...`` line to
    embed (defaults to the repo-wide MIT identifier; pass
    :data:`SPDX_LICENSE_SKILLS` for dual-licensed skill files).
    """
    lines = []

    if middle:
        # Multi-line comment style (e.g., CSS, HTML)
        lines.append((f"{prefix} {SPDX_COPYRIGHT} {suffix}").rstrip() + "\n")
        lines.append(f"{middle}\n")
        lines.append((f"{prefix} {spdx_license} {suffix}").rstrip() + "\n")
    else:
        # Single-line comment style (e.g., Python, Shell, Markdown)
        if prefix == "<!---":
            # Special case for Markdown
            lines.append((f"{prefix} {SPDX_COPYRIGHT} {suffix}").rstrip() + "\n")
            lines.append("\n")
            lines.append((f"{prefix} {spdx_license} {suffix}").rstrip() + "\n")
        else:
            # Standard single-line comments
            lines.append((f"{prefix} {SPDX_COPYRIGHT}").rstrip() + "\n")
            lines.append(f"{prefix}\n")
            lines.append((f"{prefix} {spdx_license}").rstrip() + "\n")

    lines.append("\n")
    return lines


def has_spdx_header(content: str, allowed_licenses: Optional[Tuple[str, ...]] = None) -> bool:
    """Check if content already has SPDX headers.

    Validates that:
    - An SPDX copyright line exists in the first 100 lines (files may
      have headers after a module docstring / banner comment)
    - The copyright year (or year range) is between MIN_COPYRIGHT_YEAR and CURRENT_YEAR
    - An SPDX license identifier line with one of ``allowed_licenses`` exists
      in the first 100 lines

    Args:
        content: File content to check.
        allowed_licenses: Tuple of SPDX license identifiers accepted as valid.
            Defaults to :data:`ALLOWED_LICENSES_DEFAULT` (MIT only).
    """
    if allowed_licenses is None:
        allowed_licenses = ALLOWED_LICENSES_DEFAULT
    head_lines = content.split("\n")[:100]
    head_text = "\n".join(head_lines)
    if not any(f"SPDX-License-Identifier: {lic}" in head_text for lic in allowed_licenses):
        return False
    match = SPDX_COPYRIGHT_PATTERN.search(head_text)
    if not match:
        return False
    start_year = int(match.group(1))
    end_year = int(match.group(2)) if match.group(2) else start_year
    return (
        MIN_COPYRIGHT_YEAR <= start_year <= CURRENT_YEAR
        and MIN_COPYRIGHT_YEAR <= end_year <= CURRENT_YEAR
        and start_year <= end_year
    )


def _strip_partial_spdx_lines(content: str) -> str:
    """Remove any existing partial SPDX header lines to avoid duplication.

    If a file already contains one of the two SPDX lines (but not both),
    ``has_spdx_header`` will return False and a full header will be prepended.
    Without stripping the existing partial line first, that line would appear
    twice in the resulting file.
    """
    cleaned = []
    for line in content.split("\n"):
        stripped = line.strip()
        # Remove lines that are (only) an SPDX tag, possibly wrapped in a
        # comment.  We match generously so that any comment style is covered.
        if "SPDX-FileCopyrightText:" in stripped or "SPDX-License-Identifier:" in stripped:
            continue
        # Also drop blank comment separator lines that sit between the two
        # SPDX lines (e.g. a bare "#" or "//").
        cleaned.append(line)
    return "\n".join(cleaned)


def add_header_to_file(
    file_path: Path,
    comment_style: Tuple[str, str, str],
    allowed_licenses: Optional[Tuple[str, ...]] = None,
    spdx_license: str = SPDX_LICENSE,
) -> bool:
    """Add (or fix) SPDX header on a file.

    ``allowed_licenses`` controls which existing SPDX license identifiers are
    considered valid and therefore left alone. ``spdx_license`` is the full
    ``SPDX-License-Identifier: ...`` line embedded into newly written
    headers (defaults to repo-wide MIT; callers should pass
    :data:`SPDX_LICENSE_SKILLS` for skill files).

    Stale headers are rewritten in place: ``has_spdx_header`` returns False
    when the existing license is not in ``allowed_licenses``,
    :func:`_strip_partial_spdx_lines` then removes the stale lines, and a
    fresh header (using ``spdx_license``) is prepended.
    """
    try:
        # Read existing content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if header already exists
        if has_spdx_header(content, allowed_licenses=allowed_licenses):
            return False

        # Remove any partial SPDX lines so they won't be duplicated.
        content = _strip_partial_spdx_lines(content)

        # Create header
        header_lines = create_header(*comment_style, spdx_license=spdx_license)

        # Handle shebang lines (keep them at the top)
        lines = content.split("\n")
        if lines and lines[0].startswith("#!"):
            # Keep shebang, add header after it
            shebang = lines[0] + "\n"
            rest = "\n".join(lines[1:])
            new_content = shebang + "".join(header_lines) + rest
        else:
            # Add header at the beginning
            new_content = "".join(header_lines) + content

        # Write back
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return False


def check_file(file_path: Path, allowed_licenses: Optional[Tuple[str, ...]] = None) -> bool:
    """Check if a file has a valid SPDX header.

    ``allowed_licenses`` is forwarded to :func:`has_spdx_header` and defaults
    to :data:`ALLOWED_LICENSES_DEFAULT` (MIT only).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return has_spdx_header(content, allowed_licenses=allowed_licenses)
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return True  # Skip files we can't read


def find_files(root_dir: Path) -> List[Path]:
    """Find all files that should have SPDX headers."""
    return list(iter_files(root_dir))


def iter_files(root_dir: Path) -> Iterator[Path]:
    """Yield files under root_dir that should carry SPDX headers (prunes skip dirs/prefixes)."""
    root_dir = root_dir.resolve()

    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)

        # If the current directory is under a skip prefix, do not descend further.
        if _is_under_skip_prefix(rel_dir):
            dirnames[:] = []
            continue

        # In-place prune of directories we don't want to descend into
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for name in filenames:
            path = Path(dirpath) / name
            if should_skip_file(path, root_dir):
                continue

            comment_style = get_comment_style(path)
            if comment_style is None:
                continue

            yield path


def action_write(root_dir: Path) -> int:
    """Add SPDX headers to files that are missing them."""
    files = find_files(root_dir)
    modified_count = 0

    for file_path in files:
        comment_style = get_comment_style(file_path)
        if comment_style is None:
            continue

        if add_header_to_file(file_path, comment_style):
            print(f"Added header to: {file_path.relative_to(root_dir)}")
            modified_count += 1

    # Handle SKILL.md (and other frontmatter .md) files under .agents/skills/.
    # These carry the dual-license expression in the YAML ``license:`` field.
    for skill_md in iter_skill_files(root_dir):
        if add_skill_license(skill_md, license_line=SKILL_LICENSE_LINE):
            print(f"Added/updated license in frontmatter: {skill_md.relative_to(root_dir)}")
            modified_count += 1

    # Handle .py and non-frontmatter .md files under .agents/skills/.
    # These are dual-licensed under CC-BY-4.0 AND Apache-2.0.
    for content_file in iter_skill_content_files(root_dir):
        comment_style = get_comment_style(content_file)
        if comment_style is None:
            continue
        if add_header_to_file(
            content_file,
            comment_style,
            allowed_licenses=ALLOWED_LICENSES_SKILLS,
            spdx_license=SPDX_LICENSE_SKILLS,
        ):
            print(f"Added/updated header on: {content_file.relative_to(root_dir)}")
            modified_count += 1

    print(f"\nModified {modified_count} file(s)")
    return 0


def action_check(root_dir: Path) -> int:
    """Check that all files have SPDX headers."""
    files = find_files(root_dir)
    missing_headers = []

    for file_path in files:
        if not check_file(file_path):
            missing_headers.append(file_path)

    # Check SKILL.md (and other frontmatter .md) files under .agents/skills/.
    for skill_md in iter_skill_files(root_dir):
        try:
            with open(skill_md, "r", encoding="utf-8") as f:
                content = f.read()
            if not has_skill_license(content, expected_line=SKILL_LICENSE_LINE):
                missing_headers.append(skill_md)
        except Exception as e:
            print(f"Error reading {skill_md}: {e}", file=sys.stderr)

    # Check .py and non-frontmatter .md files under .agents/skills/. These
    # must carry the dual-license SPDX expression.
    for content_file in iter_skill_content_files(root_dir):
        if not check_file(content_file, allowed_licenses=ALLOWED_LICENSES_SKILLS):
            missing_headers.append(content_file)

    if missing_headers:
        print("❌ The following files are missing SPDX headers:\n")
        for file_path in missing_headers:
            print(f"  {file_path.relative_to(root_dir)}")
        print(f"\n{len(missing_headers)} file(s) missing headers")
        print("\nRun with --action write to add headers automatically")
        return 1
    else:
        print("✅ All files have SPDX headers")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Check and add SPDX license headers to source files")
    parser.add_argument(
        "--action",
        choices=["check", "write"],
        required=True,
        help="Action to perform: check (verify headers exist) or write (add missing headers)",
    )
    parser.add_argument(
        "--root", type=Path, default=None, help="Root directory to search (defaults to repository root)"
    )

    args = parser.parse_args()

    # Determine root directory
    if args.root:
        root_dir = args.root.resolve()
    else:
        # Find repository root: this script lives at .github/scripts/, so
        # the repo root is two levels above the script's directory.
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent.parent

    if not root_dir.exists():
        print(f"Error: Root directory does not exist: {root_dir}", file=sys.stderr)
        return 1

    print(f"Searching in: {root_dir}\n")

    if args.action == "check":
        return action_check(root_dir)
    elif args.action == "write":
        return action_write(root_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
