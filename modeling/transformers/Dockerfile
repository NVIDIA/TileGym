# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# TileGym Transformers Inference Container
# This is a simplified Dockerfile for running TileGym-accelerated transformer inference
#
# Build from tilegym repository root:
#   docker build -t tilegym-transformers -f modeling/transformers/Dockerfile .

ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu22.04
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

# Install PyTorch with CUDA 13.0 support
RUN pip install --pre "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu130

# Set working directory
WORKDIR /workspace/tilegym

# Copy TileGym source code
COPY . /workspace/tilegym/

# Verify the repository structure (fail fast if built from wrong directory)
RUN test -f setup.py || (echo "ERROR: setup.py not found! Please build from tilegym repository root." && exit 1)

# Install CUDA Toolkit 13.1
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda-repo-ubuntu2204-13-1-local_13.1.0-590.44.01-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-13-1-local_13.1.0-590.44.01-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-13-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda-toolkit-13-1

# Set CUDA 13.1 as default
ENV CUDA_HOME=/usr/local/cuda-13.1
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install cuda-tile
RUN pip install cuda-tile

# Install TileGym
RUN pip install -e .

# Install accelerate used in models
RUN pip install accelerate --no-deps

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
