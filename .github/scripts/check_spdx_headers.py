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
import sys
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

# SPDX header content
SPDX_COPYRIGHT = "SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
SPDX_LICENSE = "SPDX-License-Identifier: MIT"


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
    ".mk": ("#", "#", ""),
    # Markdown
    ".md": ("<!---", "", "--->"),
    # C/C++/CUDA (using C++ style comments)
    ".c": ("//", "//", ""),
    ".h": ("//", "//", ""),
    ".cpp": ("//", "//", ""),
    ".hpp": ("//", "//", ""),
    ".cu": ("//", "//", ""),
    ".cuh": ("//", "//", ""),
    # JavaScript/TypeScript
    ".js": ("//", "//", ""),
    ".ts": ("//", "//", ""),
    ".jsx": ("//", "//", ""),
    ".tsx": ("//", "//", ""),
    # CSS
    ".css": ("/*", " *", " */"),
    # HTML/XML
    ".html": ("<!--", "", "-->"),
    ".xml": ("<!--", "", "-->"),
    # Dockerfile
    "Dockerfile": ("#", "#", ""),
    # TOML, INI
    ".toml": ("#", "#", ""),
    ".ini": ("#", "#", ""),
}


def should_skip_file(file_path: Path) -> bool:
    """Check if a file should be skipped."""
    path_str = str(file_path)

    # Skip .git directory specifically (but not .github)
    if ".git" in file_path.parts:
        return True

    # Skip files by exact name match
    exact_match_patterns = ["LICENSE", "ATTRIBUTIONS.md", "CLA.md", ".gitignore", ".cursorignore"]
    if file_path.name in exact_match_patterns:
        return True

    # Skip directories
    dir_patterns = ["__pycache__", ".pytest_cache", "node_modules", "venv", "env", ".egg-info", "dist", "build"]
    for pattern in dir_patterns:
        if pattern in file_path.parts:
            return True

    # Skip by file extension
    skip_extensions = [
        ".pyc",
        ".pyo",
        ".so",
        ".o",
        ".a",
        ".lib",
        ".dll",
        ".dylib",
        ".idx",
        ".pack",
        ".rev",
        ".sample",
        ".TAG",
    ]
    if file_path.suffix in skip_extensions:
        return True

    # Skip files without extensions that aren't Dockerfile
    if not file_path.suffix and file_path.name != "Dockerfile":
        return True

    return False


def get_comment_style(file_path: Path) -> Optional[Tuple[str, str, str]]:
    """Get the comment style for a given file."""
    # Check for Dockerfile specifically
    if file_path.name == "Dockerfile":
        return COMMENT_STYLES.get("Dockerfile")

    # Check by extension
    return COMMENT_STYLES.get(file_path.suffix)


def create_header(prefix: str, middle: str, suffix: str) -> List[str]:
    """Create the SPDX header lines based on comment style."""
    lines = []

    if middle:
        # Multi-line comment style (e.g., CSS, HTML)
        lines.append(f"{prefix} {SPDX_COPYRIGHT} {suffix}\n")
        lines.append(f"{middle}\n")
        lines.append(f"{prefix} {SPDX_LICENSE} {suffix}\n")
    else:
        # Single-line comment style (e.g., Python, Shell, Markdown)
        if prefix == "<!---":
            # Special case for Markdown
            lines.append(f"{prefix} {SPDX_COPYRIGHT} {suffix}\n")
            lines.append("\n")
            lines.append(f"{prefix} {SPDX_LICENSE} {suffix}\n")
        else:
            # Standard single-line comments
            lines.append(f"{prefix} {SPDX_COPYRIGHT}\n")
            lines.append(f"{prefix}\n")
            lines.append(f"{prefix} {SPDX_LICENSE}\n")

    lines.append("\n")
    return lines


def has_spdx_header(content: str) -> bool:
    """Check if content already has SPDX headers."""
    # Check for both required strings within the first 10 lines
    first_lines = "\n".join(content.split("\n")[:10])
    return SPDX_COPYRIGHT in first_lines and SPDX_LICENSE in first_lines


def add_header_to_file(file_path: Path, comment_style: Tuple[str, str, str]) -> bool:
    """Add SPDX header to a file if missing."""
    try:
        # Read existing content
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if header already exists
        if has_spdx_header(content):
            return False

        # Create header
        header_lines = create_header(*comment_style)

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


def check_file(file_path: Path) -> bool:
    """Check if a file has the SPDX header."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return has_spdx_header(content)
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return True  # Skip files we can't read


def find_files(root_dir: Path) -> List[Path]:
    """Find all files that should have SPDX headers."""
    files = []

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        if should_skip_file(path):
            continue

        comment_style = get_comment_style(path)
        if comment_style is None:
            continue

        files.append(path)

    return files


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

    print(f"\nModified {modified_count} file(s)")
    return 0


def action_check(root_dir: Path) -> int:
    """Check that all files have SPDX headers."""
    files = find_files(root_dir)
    missing_headers = []

    for file_path in files:
        if not check_file(file_path):
            missing_headers.append(file_path)

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
        # Find repository root (look for .git directory)
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
