#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Check if a Docker image with a specific tag exists in GHCR.

Used to skip rebuilds in nightly workflows if the image for
a commit SHA already exists.
"""

import os
import subprocess
import sys

from utils import write_github_output


def check_image_exists(registry_image: str, tag: str, token: str) -> bool:
    """
    Check if 'latest' tag points to the image with the given SHA tag.

    This ensures we only skip builds when the current commit SHA has been
    successfully tested (promoted to 'latest').

    Args:
        registry_image: Full registry path (e.g., ghcr.io/nvidia/tilegym)
        tag: SHA tag to check (e.g., commit SHA)
        token: GitHub token for authentication

    Returns:
        True if 'latest' points to the same image as the SHA tag, False otherwise
    """
    latest_image = f"{registry_image}:latest"
    sha_image = f"{registry_image}:{tag}"

    try:
        # Login to GHCR
        print(f"Checking if 'latest' points to SHA: {tag}", file=sys.stderr)

        login_process = subprocess.run(
            ["docker", "login", "ghcr.io", "-u", os.environ.get("GITHUB_ACTOR", ""), "--password-stdin"],
            input=token.encode(),
            capture_output=True,
            check=True,
        )

        # Get digest of 'latest' tag
        latest_inspect = subprocess.run(
            ["docker", "manifest", "inspect", latest_image], capture_output=True, check=False
        )

        if latest_inspect.returncode != 0:
            print("'latest' tag does not exist", file=sys.stderr)
            return False

        # Get digest of SHA tag
        sha_inspect = subprocess.run(["docker", "manifest", "inspect", sha_image], capture_output=True, check=False)

        if sha_inspect.returncode != 0:
            print(f"SHA tag '{tag}' does not exist", file=sys.stderr)
            return False

        # Compare digests (both outputs are JSON with a 'config' field containing the digest)
        import json

        latest_digest = json.loads(latest_inspect.stdout)["config"]["digest"]
        sha_digest = json.loads(sha_inspect.stdout)["config"]["digest"]

        return latest_digest == sha_digest

    except subprocess.CalledProcessError as e:
        print(f"Error checking image: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return False


def get_inputs():
    """Get and validate inputs from environment variables."""
    registry_image = os.environ.get("REGISTRY_IMAGE")
    tag = os.environ.get("IMAGE_TAG")
    token = os.environ.get("GITHUB_TOKEN")
    is_pr = os.environ.get("IS_PR", "true").lower() == "true"

    if not all([registry_image, tag, token]):
        print("Error: Missing required environment variables", file=sys.stderr)
        print(f"  REGISTRY_IMAGE: {registry_image}", file=sys.stderr)
        print(f"  IMAGE_TAG: {tag}", file=sys.stderr)
        print(f"  GITHUB_TOKEN: {'set' if token else 'not set'}", file=sys.stderr)
        sys.exit(1)

    return registry_image, tag, token, is_pr


def should_skip_build(registry_image: str, tag: str, token: str, is_pr: bool) -> bool:
    """Determine if build should be skipped based on context."""
    if is_pr:
        print("PR context detected, skipping image existence check", file=sys.stderr)
        return False

    print("Nightly/main context detected, checking if 'latest' points to current SHA", file=sys.stderr)
    latest_matches = check_image_exists(registry_image, tag, token)

    if latest_matches:
        print(f"✅ 'latest' already points to SHA {tag} (tests passed previously)", file=sys.stderr)
        return True
    else:
        print(f"🔨 'latest' does not point to SHA {tag} (build and test needed)", file=sys.stderr)
        return False


def write_output(skipped: bool):
    """Write result to GitHub Actions output."""
    write_github_output("skipped", str(skipped).lower())
    print(f"Build will be {'skipped' if skipped else 'executed'}", file=sys.stderr)


def main():
    registry_image, tag, token, is_pr = get_inputs()
    skipped = should_skip_build(registry_image, tag, token, is_pr)
    write_output(skipped)


if __name__ == "__main__":
    main()
