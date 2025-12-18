#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Parse CI configuration from PR body.

Extracts YAML config block from PR description and outputs
GitHub Actions environment variables.
"""

import os
import re
import sys

import yaml
from utils import write_github_output


def get_default_config():
    """Return default CI configuration."""
    return {"build": True, "test": ["ops", "benchmark"]}


def extract_yaml_from_pr_body(pr_body):
    """Extract YAML config block from PR body text."""
    if not pr_body:
        return None

    pattern = r"```yaml\s*\nconfig:(.*?)\n```"
    match = re.search(pattern, pr_body, re.DOTALL)
    return match.group(1) if match else None


def parse_config_yaml(yaml_text):
    """Parse YAML config text into dictionary."""
    try:
        config_text = "config:" + yaml_text
        parsed = yaml.safe_load(config_text)
        return parsed.get("config", {})
    except yaml.YAMLError as e:
        print(f"Warning: Failed to parse config YAML: {e}", file=sys.stderr)
        return None


def resolve_config(pr_body):
    """Resolve final config from PR body or defaults."""
    config = get_default_config()

    if not pr_body or not pr_body.strip():
        print("Not in PR context or empty PR body, using defaults (run everything)", file=sys.stderr)
        return config

    yaml_text = extract_yaml_from_pr_body(pr_body)
    if not yaml_text:
        print("No config found in PR body, using defaults", file=sys.stderr)
        return config

    parsed_config = parse_config_yaml(yaml_text)
    if parsed_config:
        config.update(parsed_config)
        print(f"Parsed config from PR: {parsed_config}", file=sys.stderr)

    return config


def write_github_outputs(config):
    """Write config to GitHub Actions outputs."""
    # Normalize test list
    test_list = config.get("test", [])
    if not isinstance(test_list, list):
        test_list = []

    # Calculate output values
    outputs = {
        "build": str(config["build"]).lower(),
        "run_ops": str("ops" in test_list).lower(),
        "run_benchmark": str("benchmark" in test_list).lower(),
    }

    # Write outputs
    for key, value in outputs.items():
        write_github_output(key, value)

    # Log final config
    print(
        f"Config: build={outputs['build']}, run_ops={outputs['run_ops']}, run_benchmark={outputs['run_benchmark']}",
        file=sys.stderr,
    )


def main():
    pr_body = os.environ.get("PR_BODY", "")
    config = resolve_config(pr_body)
    write_github_outputs(config)


if __name__ == "__main__":
    main()
