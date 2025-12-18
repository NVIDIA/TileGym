# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Tests for check_spdx_headers.py script."""

import sys
import tempfile
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from check_spdx_headers import create_header
from check_spdx_headers import get_comment_style
from check_spdx_headers import has_spdx_header
from check_spdx_headers import should_skip_file


def test_has_spdx_header():
    """Test SPDX header detection."""
    # Content with header
    content_with_header = """# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import sys
"""
    assert has_spdx_header(content_with_header)

    # Content without header
    content_without_header = """import sys
print("hello")
"""
    assert not has_spdx_header(content_without_header)


def test_create_header_python():
    """Test header creation for Python files."""
    header = create_header("#", "#", "")
    assert len(header) == 4  # 3 lines + blank line
    assert "SPDX-FileCopyrightText" in header[0]
    assert "SPDX-License-Identifier" in header[2]


def test_create_header_markdown():
    """Test header creation for Markdown files."""
    header = create_header("<!---", "", "--->")
    assert len(header) == 4  # 3 lines + blank line
    assert "<!---" in header[0]
    assert "--->" in header[0]


def test_get_comment_style():
    """Test comment style detection."""
    # Python file
    py_file = Path("test.py")
    assert get_comment_style(py_file) == ("#", "#", "")

    # Markdown file
    md_file = Path("README.md")
    assert get_comment_style(md_file) == ("<!---", "", "--->")

    # Shell script
    sh_file = Path("script.sh")
    assert get_comment_style(sh_file) == ("#", "#", "")

    # Dockerfile
    dockerfile = Path("Dockerfile")
    assert get_comment_style(dockerfile) == ("#", "#", "")

    # Unknown extension
    unknown_file = Path("file.xyz")
    assert get_comment_style(unknown_file) is None


def test_should_skip_file():
    """Test file skipping logic."""
    # Should skip compiled files
    assert should_skip_file(Path("test.pyc"))
    assert should_skip_file(Path("__pycache__/test.py"))

    # Should skip .git directory files
    assert should_skip_file(Path(".git/config"))

    # Should NOT skip .github directory files
    assert not should_skip_file(Path(".github/workflows/ci.yml"))
    assert not should_skip_file(Path(".github/scripts/utils.py"))

    # Should not skip source files
    assert not should_skip_file(Path("test.py"))
    assert not should_skip_file(Path("README.md"))


def test_integration_write_and_check():
    """Integration test: write and check headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.py"

        # Create a file without header
        test_file.write_text("import sys\nprint('hello')\n")

        # Import the functions
        from check_spdx_headers import add_header_to_file
        from check_spdx_headers import check_file

        # Check that it's missing header
        assert not check_file(test_file)

        # Add header
        comment_style = get_comment_style(test_file)
        assert add_header_to_file(test_file, comment_style)

        # Check that it now has header
        assert check_file(test_file)

        # Verify content
        content = test_file.read_text()
        assert "SPDX-FileCopyrightText" in content
        assert "SPDX-License-Identifier: MIT" in content
        assert "import sys" in content  # Original content preserved


def test_shebang_preservation():
    """Test that shebang lines are preserved at the top."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.sh"

        # Create a file with shebang
        test_file.write_text("#!/bin/bash\necho 'hello'\n")

        from check_spdx_headers import add_header_to_file

        # Add header
        comment_style = ("#", "#", "")
        add_header_to_file(test_file, comment_style)

        # Verify shebang is still first line
        content = test_file.read_text()
        lines = content.split("\n")
        assert lines[0] == "#!/bin/bash"
        assert "SPDX-FileCopyrightText" in lines[1]


if __name__ == "__main__":
    # Run basic tests
    test_has_spdx_header()
    test_create_header_python()
    test_create_header_markdown()
    test_get_comment_style()
    test_should_skip_file()
    test_integration_write_and_check()
    test_shebang_preservation()
    print("✅ All tests passed!")
