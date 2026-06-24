# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Use the CUDA 13.3 / Ubuntu 24.04 base image, which ships a newer default
# toolchain (GCC 13) and Python 3.12 for building the user-space environment.
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.3.0-devel-ubuntu24.04

FROM ${BASE_IMAGE} AS base

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip && \
    curl -fsSL https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/nvidia.pub | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/ /" > \
        /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nsight-systems-cli && \
    python -m pip install --no-cache-dir --break-system-packages uv && \
    apt-get purge -y --auto-remove curl gnupg && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip

FROM base AS source

ENV HF_HOME=/workspace/.cache/huggingface \
    TILEGYM_MODEL_CACHE_DIR=/workspace/.cache

COPY . /workspace/tilegym/

WORKDIR /workspace/tilegym/modeling/transformers

RUN uv sync --locked --extra dev && \
    mkdir -p /workspace/.cache/huggingface/hub /logs
