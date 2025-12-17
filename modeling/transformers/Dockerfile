# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# TileGym Transformers Inference Container
# This is a simplified Dockerfile for running TileGym-accelerated transformer inference
#
# Build from tilegym repository root:
#   docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.1.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.9.1

ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python-is-python3 \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 13.0 support (after CUDA toolkit)
RUN pip install --no-cache-dir --pre "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu130

# Install cuda-tile and accelerate (before copying source code)
RUN pip install --no-cache-dir cuda-tile && \
    pip install --no-cache-dir --no-deps accelerate

# Set working directory
WORKDIR /workspace/tilegym

# Copy TileGym source code (do this LATE so code changes don't invalidate above layers)
COPY . /workspace/tilegym/

# Verify the repository structure (fail fast if built from wrong directory)
RUN test -f setup.py || (echo "ERROR: setup.py not found! Please build from tilegym repository root." && exit 1)

# Install TileGym (only this layer rebuilds on code changes)
RUN pip install --no-cache-dir -e .

# Set up model cache directory
ENV HF_HOME=/workspace/.cache/huggingface \
    TILEGYM_MODEL_CACHE_DIR=/workspace/.cache

# Create necessary directories
RUN mkdir -p /workspace/.cache/huggingface/hub \
    /logs \
    /workspace/tilegym/modeling/transformers

# Set working directory to transformers modeling
WORKDIR /workspace/tilegym/modeling/transformers

# Expose port for potential web services
EXPOSE 8000

# Default command
CMD ["/bin/bash"]
