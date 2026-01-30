# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# TileGym Transformers Inference Container
# Multi-stage build supporting both wheel (CI) and source (development) installs
#
# Usage:
#   CI (wheel):     docker build --target wheel -f modeling/transformers/Dockerfile .
#   Dev (source):   docker build --target source -f modeling/transformers/Dockerfile .
#   Default:        docker build -f modeling/transformers/Dockerfile .  (uses source)

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.1.0-devel-ubuntu22.04

####################
# Stage 1: Base dependencies
####################
FROM ${BASE_IMAGE} AS base

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

# Install PyTorch with CUDA 13.0 support
RUN pip install --no-cache-dir --pre "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu130

# Install cuda-tile and accelerate
RUN pip install --no-cache-dir cuda-tile && \
    pip install --no-cache-dir --no-deps accelerate

WORKDIR /workspace/tilegym

####################
# Stage 2: Wheel install (for CI)
####################
FROM base AS wheel

# Copy and install pre-built wheel
COPY wheel/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -f /tmp/*.whl

# Copy modeling scripts for inference
# Tests are mounted at runtime in CI
COPY modeling/ /workspace/tilegym/modeling/
COPY README.md LICENSE /workspace/tilegym/

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

####################
# Stage 3: Source install (for open source developers)
# This is the default when no --target is specified
####################
FROM base AS source

# Copy entire source tree
COPY . /workspace/tilegym/

# Verify repository structure
RUN test -f setup.py || (echo "ERROR: setup.py not found! Please build from tilegym repository root." && exit 1)

# Install TileGym in editable mode for development
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
