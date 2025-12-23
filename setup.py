# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pathlib

import setuptools

README = (pathlib.Path(__file__).parent / "README.md").read_text()


setuptools.setup(
    name="tilegym",
    version="0.0.0",
    author="NVIDIA Corporation",
    description="TileGym",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
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
        # 'nvidia-ml-py', # optional
    ],
    extras_require={
        "dev": [
            "ruff==0.14.9",
        ],
    },
)
