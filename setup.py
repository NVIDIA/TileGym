# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pathlib

import setuptools

README = (pathlib.Path(__file__).parent / "README.md").read_text()


setuptools.setup(
    name="tilegym",
    version="1.0.0",
    author="NVIDIA Corporation",
    description="TileGym",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/TileGym",
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
    install_requires=[
        # Note: torch and triton should be pre-installed in your environment
        "transformers==4.56.2",
        "tokenizers==0.22.0",
        # 'accelerate', # Use `pip install accelerate --no-deps` to avoid reinstall torch
        "huggingface_hub",
        "matplotlib",
        "pandas",
        "pytest",
        "numpy",
        "cuda-tile",
        "cuda-tile-experimental @ git+https://github.com/NVIDIA/cutile-python.git#subdirectory=experimental",
        "filelock>=3.20.3",  # CVE fix: GHSA-w853-jp5j-5j7f, GHSA-qmgc-5h2g-mvrw
        "pillow>=12.1.1",  # CVE fix: GHSA-cfh3-3jmp-rvhc
        # 'nvidia-ml-py', # optional
    ],
    extras_require={
        "dev": [
            "ruff==0.14.9",
        ],
        "torch": [
            "torch>=2.9.1",
        ],
    },
)
