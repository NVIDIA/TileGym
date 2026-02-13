#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Get version information for TileGym wheels.

This script determines the appropriate version based on git context:
- Tagged commit (v1.0.0-rc1) → 1.0.0rc1
- PR branch → 1.0.0.dev0+pr123.abc123
- Main/nightly → 1.0.0.dev20260211+abc123

Usage:
    python get_version.py [--context {auto|pr|nightly|release}] [--base-version VERSION]

Examples:
    # Auto-detect context from git
    python get_version.py

    # Force PR context with PR number from env
    PR_NUMBER=123 python get_version.py --context pr

    # Use custom base version
    python get_version.py --base-version 2.0.0
"""

import argparse
import os
import re
import subprocess
from datetime import datetime


def run_git_command(cmd):
    """Run a git command and return output."""
    try:
        result = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return result
    except subprocess.CalledProcessError:
        return None


def get_base_version_from_setup():
    """Extract base version from setup.py."""
    try:
        with open("setup.py", "r") as f:
            content = f.read()
            match = re.search(r'version="([^"]+)"', content)
            if match:
                version = match.group(1)
                # Strip any existing suffix to get clean base version
                clean = re.sub(r"\.(dev|rc|alpha|beta)\d+.*$", "", version)
                clean = re.sub(r"\+.*$", "", clean)
                return clean
    except FileNotFoundError:
        pass
    return "1.0.0"  # Default fallback


def get_git_sha(short=True):
    """Get current git commit SHA."""
    length = 7 if short else 40
    sha = run_git_command(["git", "rev-parse", f"--short={length}", "HEAD"])
    return sha or "unknown"


def get_git_tag():
    """Get git tag if current commit is tagged."""
    tag = run_git_command(["git", "describe", "--exact-match", "--tags", "HEAD"])
    if tag:
        # Convert v1.0.0-rc1 → 1.0.0rc1 (PyPI format)
        version = tag.lstrip("v").replace("-", "")
        return version
    return None


def detect_context():
    """Auto-detect the build context (release, pr, nightly)."""
    # Check for git tag first
    if get_git_tag():
        return "release"

    # Check for PR number in environment
    if os.environ.get("PR_NUMBER"):
        return "pr"

    # Check if on pull-request branch
    branch = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if branch and branch.startswith("pull-request/"):
        return "pr"

    # Default to nightly
    return "nightly"


def generate_version(base_version, context="auto", pr_number=None):
    """
    Generate version string based on context.

    Args:
        base_version: Base version like "1.0.0"
        context: One of 'auto', 'release', 'pr', 'nightly'
        pr_number: PR number (optional, read from env if not provided)

    Returns:
        Tuple of (version_string, context_used)
    """
    if context == "auto":
        context = detect_context()

    sha = get_git_sha(short=True)

    if context == "release":
        # For tagged releases, use the tag version
        tag_version = get_git_tag()
        if tag_version:
            return tag_version, "release"
        else:
            print("Warning: release context specified but no git tag found")
            return base_version, "release"

    elif context == "pr":
        # PR builds: 1.0.0.dev0+pr123.abc123
        pr_num = pr_number or os.environ.get("PR_NUMBER", "")
        if not pr_num:
            # Try to extract from branch name
            branch = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            if branch:
                match = re.search(r"pull-request/(\d+)", branch)
                if match:
                    pr_num = match.group(1)

        if pr_num:
            version = f"{base_version}.dev0+pr{pr_num}.{sha}"
        else:
            version = f"{base_version}.dev0+{sha}"
        return version, "pr"

    elif context == "nightly":
        # Nightly builds: 1.0.0.dev20260211+abc123
        date = datetime.now().strftime("%Y%m%d")
        version = f"{base_version}.dev{date}+{sha}"
        return version, "nightly"

    else:
        raise ValueError(f"Invalid context: {context}")


def main():
    parser = argparse.ArgumentParser(description="Get version information for TileGym wheels")
    parser.add_argument(
        "--context",
        choices=["auto", "release", "pr", "nightly"],
        default="auto",
        help="Build context (default: auto-detect)",
    )
    parser.add_argument("--base-version", help="Base version to use (default: read from setup.py)")
    parser.add_argument("--pr-number", help="PR number for PR builds")
    parser.add_argument(
        "--output-format", choices=["text", "github"], default="text", help="Output format (text or github actions)"
    )

    args = parser.parse_args()

    # Get base version
    base_version = args.base_version or get_base_version_from_setup()

    # Generate version
    version, context_used = generate_version(base_version, context=args.context, pr_number=args.pr_number)

    # Output
    if args.output_format == "github":
        # Write to GITHUB_OUTPUT for GitHub Actions
        output_file = os.environ.get("GITHUB_OUTPUT", "/dev/stdout")
        with open(output_file, "a") as f:
            f.write(f"version={version}\n")
            f.write(f"context={context_used}\n")
            f.write(f"base_version={base_version}\n")

    # Always print to stdout for visibility
    print(f"Version: {version}")
    print(f"Context: {context_used}")
    print(f"Base version: {base_version}")

    return 0


if __name__ == "__main__":
    exit(main())
