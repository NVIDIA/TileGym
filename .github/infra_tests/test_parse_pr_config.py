# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unit tests for parse_pr_config.py"""

import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))

import parse_pr_config


class TestParsePRConfig:
    """Tests for parse_pr_config.py"""

    def test_get_default_config(self):
        """Test default configuration."""
        config = parse_pr_config.get_default_config()
        assert config["build"] is True
        assert config["test"] == ["ops", "benchmark"]

    def test_extract_yaml_from_pr_body(self):
        """Test YAML extraction from PR body."""
        pr_body = """Some text
```yaml
config:
  build: false
  test: ["ops"]
```
More text"""
        yaml_text = parse_pr_config.extract_yaml_from_pr_body(pr_body)
        assert yaml_text is not None
        assert "build: false" in yaml_text

    def test_extract_yaml_no_config(self):
        """Test extraction when no config present."""
        pr_body = "Just some text without yaml"
        yaml_text = parse_pr_config.extract_yaml_from_pr_body(pr_body)
        assert yaml_text is None

    def test_parse_config_yaml(self):
        """Test YAML parsing."""
        yaml_text = """
  build: false
  test: ["benchmark"]
"""
        config = parse_pr_config.parse_config_yaml(yaml_text)
        assert config is not None
        assert config["build"] is False
        assert config["test"] == ["benchmark"]

    def test_resolve_config_with_pr_body(self):
        """Test config resolution with valid PR body."""
        pr_body = """```yaml
config:
  build: false
  test: ["ops"]
```"""
        config = parse_pr_config.resolve_config(pr_body)
        assert config["build"] is False
        assert config["test"] == ["ops"]

    def test_resolve_config_empty_pr_body(self):
        """Test config resolution with empty PR body."""
        config = parse_pr_config.resolve_config("")
        assert config["build"] is True  # defaults
        assert config["test"] == ["ops", "benchmark"]
