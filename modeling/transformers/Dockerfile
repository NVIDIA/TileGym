# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Keep the runtime image at CUDA 13.2 for host-driver compatibility.
# The uv environment installs CUDA 13.3 compiler/user-space wheels from pyproject.toml.
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04

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
    curl -fsSL https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/nvidia.pub | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/ /" > \
        /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nsight-systems-cli && \
    python -m pip install --no-cache-dir uv && \
    apt-get purge -y --auto-remove curl gnupg && \
    rm -rf /var/lib/apt/lists/* /root/.cache/pip

FROM base AS wheel

ENV HF_HOME=/workspace/.cache/huggingface \
    TILEGYM_MODEL_CACHE_DIR=/workspace/.cache

WORKDIR /workspace/tilegym-hf-bench

COPY modeling/transformers/ /workspace/tilegym-hf-bench/

RUN uv sync --locked && \
    mkdir -p /workspace/.cache/huggingface/hub /logs

FROM wheel AS source

COPY . /workspace/tilegym/

RUN uv pip uninstall --python /workspace/tilegym-hf-bench/.venv/bin/python tilegym && \
    uv pip install --python /workspace/tilegym-hf-bench/.venv/bin/python --no-deps -e /workspace/tilegym
