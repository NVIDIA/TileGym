# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""TileGym Hugging Face inference benchmark utilities."""

import os
from pathlib import Path


def _setup_hf_home_early() -> None:
    """Resolve HF_HOME from TILEGYM_MODEL_CACHE_DIR before any transformers import.

    huggingface_hub freezes HF_HOME/HF_HUB_CACHE at import time, so setting the
    environment later (e.g. inside hf_shim.load_model_with_cache) silently leaves
    downloads flowing to the default location (~/.cache/huggingface or, with
    XDG_CACHE_HOME set, $XDG_CACHE_HOME/huggingface). This package initializer runs
    before tilegym_hf_bench submodules import transformers, keeping the cache in
    the configured place. Explicitly user-set HF_HOME always wins.
    """
    if "HF_HOME" in os.environ:
        return
    cache_base = os.environ.get("TILEGYM_MODEL_CACHE_DIR")
    if cache_base:
        os.environ["HF_HOME"] = str(Path(cache_base) / "huggingface")


_setup_hf_home_early()
