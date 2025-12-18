#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Clean up stale Docker images from GHCR.

Removes:
1. Images for closed PRs (pr-* tags)
2. Untracked images without pr-* or latest tags (older than threshold)
"""

import os
import sys
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

import requests
from utils import get_github_api_headers
from utils import get_github_token


def get_open_pr_numbers(owner: str, repo: str, token: str) -> set:
    """Fetch all open PR numbers from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
    headers = get_github_api_headers(token)

    try:
        response = requests.get(url, headers=headers, params={"state": "open", "per_page": 100})
        response.raise_for_status()
        prs = response.json()
        return {pr["number"] for pr in prs}
    except Exception as e:
        print(f"Error fetching open PRs: {e}", file=sys.stderr)
        return set()


def get_package_versions(owner: str, package_name: str, token: str, is_org: bool) -> List[Dict[str, Any]]:
    """Fetch all versions of a package from GHCR."""
    endpoint = "orgs" if is_org else "users"
    url = f"https://api.github.com/users/{owner}/packages/container/{package_name}/versions"

    if is_org:
        url = f"https://api.github.com/{endpoint}/{owner}/packages/container/{package_name}/versions"

    headers = get_github_api_headers(token)

    try:
        response = requests.get(url, headers=headers, params={"per_page": 100})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching package versions: {e}", file=sys.stderr)
        return []


def delete_package_version(owner: str, package_name: str, version_id: int, token: str, is_org: bool) -> bool:
    """Delete a specific package version."""
    endpoint = "orgs" if is_org else "users"
    url = f"https://api.github.com/{endpoint}/{owner}/packages/container/{package_name}/versions/{version_id}"
    headers = get_github_api_headers(token)

    try:
        response = requests.delete(url, headers=headers)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error deleting version {version_id}: {e}", file=sys.stderr)
        return False


def should_delete_closed_pr_image(tags: List[str], open_pr_numbers: set) -> tuple[bool, str]:
    """Check if image is for a closed PR and should be deleted."""
    for tag in tags:
        if tag.startswith("pr-"):
            try:
                pr_number = int(tag[3:])
                if pr_number not in open_pr_numbers:
                    return True, f"Closed PR #{pr_number}"
            except ValueError:
                pass
    return False, ""


def should_delete_untracked_image(tags: List[str], created_at: str, days_threshold: int) -> tuple[bool, str]:
    """Check if image is untracked and old enough to delete."""
    if not tags:
        return False, ""

    # Check if untracked (no pr-*, latest, or -verified tags)
    has_pr_tag = any(tag.startswith("pr-") for tag in tags)
    has_latest_tag = "latest" in tags
    has_verified_tag = any(tag.endswith("-verified") for tag in tags)

    if has_pr_tag or has_latest_tag or has_verified_tag:
        return False, ""

    # Check age
    created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    age = datetime.now(created.tzinfo) - created
    age_days = age.days

    if age_days > days_threshold:
        tag_str = ", ".join(tags[:3])  # Show first 3 tags
        return True, f"Untracked (age: {age_days} days, tags: {tag_str})"

    return False, ""


def cleanup_images(owner: str, repo: str, package_name: str, token: str, untracked_days: int = 7):
    """Main cleanup logic."""
    print(f"Starting cleanup for package: {package_name}", file=sys.stderr)

    # Detect if owner is org or user (try org first)
    is_org = True  # Assume org, fallback to user if needed

    # Get open PRs
    open_prs = get_open_pr_numbers(owner, repo, token)
    print(f"Found {len(open_prs)} open PRs", file=sys.stderr)

    # Get package versions
    versions = get_package_versions(owner, package_name, token, is_org)
    if not versions and is_org:
        # Try as user
        is_org = False
        versions = get_package_versions(owner, package_name, token, is_org)

    print(f"Found {len(versions)} image versions", file=sys.stderr)

    deleted_count = 0

    for version in versions:
        version_id = version["id"]
        tags = version.get("metadata", {}).get("container", {}).get("tags", [])
        created_at = version.get("created_at", "")

        # Check for closed PR images
        should_delete, reason = should_delete_closed_pr_image(tags, open_prs)

        # Check for untracked images if not already marked for deletion
        if not should_delete:
            should_delete, reason = should_delete_untracked_image(tags, created_at, untracked_days)

        if should_delete:
            print(f"Deleting version {version_id}: {reason}", file=sys.stderr)
            if delete_package_version(owner, package_name, version_id, token, is_org):
                deleted_count += 1
                print("✅ Deleted successfully", file=sys.stderr)
            else:
                print("❌ Failed to delete", file=sys.stderr)

    print(f"Cleanup complete! Deleted {deleted_count} image(s)", file=sys.stderr)


def main():
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER")
    repo = os.environ.get("GITHUB_REPOSITORY", "").split("/")[-1]
    package_name = os.environ.get("PACKAGE_NAME")
    untracked_days = int(os.environ.get("UNTRACKED_DAYS_THRESHOLD", "7"))

    if not all([owner, repo, package_name]):
        print("Error: Missing required environment variables", file=sys.stderr)
        print(f"  GITHUB_REPOSITORY_OWNER: {owner}", file=sys.stderr)
        print(f"  GITHUB_REPOSITORY: {repo}", file=sys.stderr)
        print(f"  PACKAGE_NAME: {package_name}", file=sys.stderr)
        sys.exit(1)

    token = get_github_token()
    cleanup_images(owner, repo, package_name, token, untracked_days)


if __name__ == "__main__":
    main()
