# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unit tests for utils.py"""

import os
import sys
import tempfile
from unittest.mock import patch

import pytest

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))

import utils


class TestUtils:
    """Tests for utils.py"""

    def test_get_github_token_success(self):
        """Test successful token retrieval."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            token = utils.get_github_token()
            assert token == "test_token"

    def test_get_github_token_missing(self):
        """Test missing token raises SystemExit."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                utils.get_github_token()

    def test_write_github_output(self):
        """Test writing GitHub output."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
            output_file = f.name

        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": output_file}):
                utils.write_github_output("test_key", "test_value")

            with open(output_file) as f:
                content = f.read()
                assert content == "test_key=test_value\n"
        finally:
            os.unlink(output_file)

    def test_write_github_output_no_env(self):
        """Test write_github_output when GITHUB_OUTPUT not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Should not raise, just return silently
            utils.write_github_output("key", "value")

    def test_get_github_api_headers(self):
        """Test GitHub API headers generation."""
        headers = utils.get_github_api_headers("my_token")
        assert headers == {"Authorization": "token my_token", "Accept": "application/vnd.github.v3+json"}
