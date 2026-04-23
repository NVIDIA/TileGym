# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pathlib

import setuptools

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")


def parse_requirements(filename: str) -> list[str]:
    """Parse a requirements.txt file into a list of dependency strings."""
    lines = (HERE / filename).read_text(encoding="utf-8").splitlines()
    reqs = []
    for line in lines:
        line = line.strip()
        # skip empty lines and full-line comments
        if not line or line.startswith("#"):
            continue
        # strip inline comments (e.g. "pkg  # comment")
        if " #" in line:
            line = line[: line.index(" #")].strip()
        if line:
            reqs.append(line)
    return reqs


setuptools.setup(
    name="tilegym",
    version="1.2.0",
    author="NVIDIA Corporation",
    description="TileGym",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/TileGym",
    project_urls={
        "Homepage": "https://github.com/NVIDIA/TileGym",
        "Repository": "https://github.com/NVIDIA/TileGym",
        "Bug Tracker": "https://github.com/NVIDIA/TileGym/issues",
    },
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=parse_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest",
            "ruff==0.14.9",
        ],
        "torch": [
            "torch>=2.9.1",
        ],
        "tileiras": [
            "cuda-tile[tileiras]",
        ],
    },
)
