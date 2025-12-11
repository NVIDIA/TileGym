#!/usr/bin/env python3
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
    Check if an image with the given tag exists in the registry.

    Args:
        registry_image: Full registry path (e.g., ghcr.io/nvidia/tilegym-transformers)
        tag: Image tag to check (e.g., commit SHA)
        token: GitHub token for authentication

    Returns:
        True if image exists, False otherwise
    """
    full_image = f"{registry_image}:{tag}"

    try:
        # Login to GHCR
        print(f"Checking if image exists: {full_image}", file=sys.stderr)

        login_process = subprocess.run(
            ["docker", "login", "ghcr.io", "-u", os.environ.get("GITHUB_ACTOR", ""), "--password-stdin"],
            input=token.encode(),
            capture_output=True,
            check=True,
        )

        # Check if manifest exists
        inspect_process = subprocess.run(
            ["docker", "manifest", "inspect", full_image], capture_output=True, check=False
        )

        return inspect_process.returncode == 0

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

    print("Nightly/main context detected, checking if image exists", file=sys.stderr)
    exists = check_image_exists(registry_image, tag, token)

    if exists:
        print(f"✅ Image {registry_image}:{tag} already exists", file=sys.stderr)
        return True
    else:
        print(f"🔨 Image {registry_image}:{tag} does not exist", file=sys.stderr)
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
