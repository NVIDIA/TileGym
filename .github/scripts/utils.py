#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Shared utilities for GitHub Actions scripts.
"""

import os
import sys


def get_github_token() -> str:
    """
    Get GitHub token from environment.

    Returns:
        GitHub token string

    Exits:
        If GITHUB_TOKEN is not set
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    return token


def write_github_output(key: str, value: str):
    """
    Write a single key-value pair to GitHub Actions output.

    Args:
        key: Output variable name
        value: Output variable value
    """
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"{key}={value}\n")


def get_github_api_headers(token: str) -> dict:
    """
    Get standard GitHub API headers with authentication.

    Args:
        token: GitHub token for authentication

    Returns:
        Dictionary of headers for GitHub API requests
    """
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
