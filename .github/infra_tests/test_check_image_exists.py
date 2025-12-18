# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unit tests for check_image_exists.py"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))

import check_image_exists


class TestCheckImageExists:
    """Tests for check_image_exists.py"""

    def test_get_inputs_success(self):
        """Test successful input retrieval."""
        env = {
            "REGISTRY_IMAGE": "ghcr.io/test/image",
            "IMAGE_TAG": "abc123",
            "GITHUB_TOKEN": "token123",
            "IS_PR": "false",
        }
        with patch.dict(os.environ, env):
            registry, tag, token, is_pr = check_image_exists.get_inputs()
            assert registry == "ghcr.io/test/image"
            assert tag == "abc123"
            assert token == "token123"
            assert is_pr is False

    def test_get_inputs_missing_env_vars(self):
        """Test that missing env vars causes exit."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit):
                check_image_exists.get_inputs()

    def test_should_skip_build_pr_context(self):
        """Test that PR context skips the check."""
        skipped = check_image_exists.should_skip_build("ghcr.io/test/image", "tag", "token", is_pr=True)
        assert skipped is False

    @patch("check_image_exists.check_image_exists")
    def test_should_skip_build_image_exists(self, mock_check):
        """Test that existing images are skipped."""
        mock_check.return_value = True
        skipped = check_image_exists.should_skip_build("ghcr.io/test/image", "tag", "token", is_pr=False)
        assert skipped is True

    @patch("check_image_exists.check_image_exists")
    def test_should_skip_build_image_not_exists(self, mock_check):
        """Test that non-existing images are not skipped."""
        mock_check.return_value = False
        skipped = check_image_exists.should_skip_build("ghcr.io/test/image", "tag", "token", is_pr=False)
        assert skipped is False

    @patch("check_image_exists.subprocess.run")
    def test_check_image_exists_latest_matches_sha(self, mock_run):
        """Test that check returns True when 'latest' points to current SHA."""
        # Mock login
        login_result = MagicMock(returncode=0)
        # Mock latest inspect
        latest_result = MagicMock(returncode=0, stdout=json.dumps({"config": {"digest": "sha256:abc123"}}).encode())
        # Mock SHA inspect
        sha_result = MagicMock(returncode=0, stdout=json.dumps({"config": {"digest": "sha256:abc123"}}).encode())
        mock_run.side_effect = [login_result, latest_result, sha_result]

        result = check_image_exists.check_image_exists("ghcr.io/test/image", "sha123", "token")
        assert result is True

    @patch("check_image_exists.subprocess.run")
    def test_check_image_exists_latest_different_sha(self, mock_run):
        """Test that check returns False when 'latest' points to different SHA."""
        # Mock login
        login_result = MagicMock(returncode=0)
        # Mock latest inspect (different digest)
        latest_result = MagicMock(returncode=0, stdout=json.dumps({"config": {"digest": "sha256:abc123"}}).encode())
        # Mock SHA inspect (different digest)
        sha_result = MagicMock(returncode=0, stdout=json.dumps({"config": {"digest": "sha256:def456"}}).encode())
        mock_run.side_effect = [login_result, latest_result, sha_result]

        result = check_image_exists.check_image_exists("ghcr.io/test/image", "sha456", "token")
        assert result is False

    @patch("check_image_exists.subprocess.run")
    def test_check_image_exists_no_latest(self, mock_run):
        """Test that check returns False when 'latest' tag doesn't exist."""
        # Mock login
        login_result = MagicMock(returncode=0)
        # Mock latest inspect (not found)
        latest_result = MagicMock(returncode=1)
        mock_run.side_effect = [login_result, latest_result]

        result = check_image_exists.check_image_exists("ghcr.io/test/image", "sha123", "token")
        assert result is False

    @patch("check_image_exists.subprocess.run")
    def test_check_image_exists_no_sha(self, mock_run):
        """Test that check returns False when SHA tag doesn't exist."""
        # Mock login
        login_result = MagicMock(returncode=0)
        # Mock latest inspect (exists)
        latest_result = MagicMock(returncode=0, stdout=json.dumps({"config": {"digest": "sha256:abc123"}}).encode())
        # Mock SHA inspect (not found)
        sha_result = MagicMock(returncode=1)
        mock_run.side_effect = [login_result, latest_result, sha_result]

        result = check_image_exists.check_image_exists("ghcr.io/test/image", "sha123", "token")
        assert result is False

    def test_write_output(self):
        """Test output file writing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            output_file = f.name

        try:
            with patch.dict(os.environ, {"GITHUB_OUTPUT": output_file}):
                check_image_exists.write_output(True)

            with open(output_file) as f:
                content = f.read()
            assert "skipped=true" in content
        finally:
            os.unlink(output_file)
