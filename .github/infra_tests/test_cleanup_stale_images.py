# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Unit tests for cleanup_stale_images.py"""

import os
import sys
from datetime import datetime
from datetime import timedelta
from unittest.mock import MagicMock
from unittest.mock import patch

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))

import cleanup_stale_images


class TestCleanupStaleImages:
    """Tests for cleanup_stale_images.py"""

    def test_should_delete_closed_pr_image(self):
        """Test detection of closed PR images."""
        open_prs = {1, 2, 3}

        # Closed PR
        should_delete, reason = cleanup_stale_images.should_delete_closed_pr_image(["pr-4"], open_prs)
        assert should_delete is True
        assert "Closed PR" in reason

        # Open PR
        should_delete, reason = cleanup_stale_images.should_delete_closed_pr_image(["pr-2"], open_prs)
        assert should_delete is False

    def test_should_delete_untracked_image_with_latest(self):
        """Test that images with 'latest' tag are not deleted."""
        old_date = (datetime.now() - timedelta(days=10)).isoformat() + "Z"
        should_delete, _ = cleanup_stale_images.should_delete_untracked_image(["latest", "abc123"], old_date, 7)
        assert should_delete is False

    def test_should_delete_untracked_image_with_pr_tag(self):
        """Test that images with pr-* tags are not deleted."""
        old_date = (datetime.now() - timedelta(days=10)).isoformat() + "Z"
        should_delete, _ = cleanup_stale_images.should_delete_untracked_image(["pr-1", "abc123"], old_date, 7)
        assert should_delete is False

    def test_should_delete_untracked_image_with_verified_tag(self):
        """Test that images with -verified tags are not deleted."""
        old_date = (datetime.now() - timedelta(days=10)).isoformat() + "Z"
        should_delete, _ = cleanup_stale_images.should_delete_untracked_image(["abc123-verified"], old_date, 7)
        assert should_delete is False

    def test_should_delete_untracked_image_old_enough(self):
        """Test that old untracked images are marked for deletion."""
        old_date = (datetime.now() - timedelta(days=10)).isoformat() + "Z"
        should_delete, reason = cleanup_stale_images.should_delete_untracked_image(["abc123", "def456"], old_date, 7)
        assert should_delete is True
        assert "Untracked" in reason

    def test_should_delete_untracked_image_too_recent(self):
        """Test that recent untracked images are not deleted."""
        recent_date = (datetime.now() - timedelta(days=3)).isoformat() + "Z"
        should_delete, _ = cleanup_stale_images.should_delete_untracked_image(["abc123"], recent_date, 7)
        assert should_delete is False

    @patch("cleanup_stale_images.requests.get")
    def test_get_open_pr_numbers(self, mock_get):
        """Test fetching open PR numbers."""
        mock_response = MagicMock()
        mock_response.json.return_value = [{"number": 1}, {"number": 2}, {"number": 3}]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        pr_numbers = cleanup_stale_images.get_open_pr_numbers("owner", "repo", "token")
        assert pr_numbers == {1, 2, 3}
