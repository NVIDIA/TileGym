# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def check_and_setup_model_cache(model_id):
    """Set Hugging Face cache env vars and return the cache base directory."""
    cache_base = os.environ.get("TILEGYM_MODEL_CACHE_DIR", os.path.expanduser("~/.cache"))
    hf_home = os.path.join(cache_base, "huggingface")
    hf_hub = os.path.join(hf_home, "hub")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HF_HUB_CACHE", hf_hub)
    Path(hf_hub).mkdir(parents=True, exist_ok=True)
    return str(cache_base)


def _check_cache_complete(model_id):
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    hf_hub = os.path.join(hf_home, "hub")
    model_cache_path = Path(hf_hub) / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = model_cache_path / "snapshots"
    return snapshots_dir.exists() and any(snapshots_dir.iterdir())


def _load_with_fallback(model_id, loader_class, resource_name, **kwargs):
    """Load a local path or HF Hub resource, preferring complete local cache."""
    model_path = Path(model_id)
    if model_path.exists() and model_path.is_dir():
        print(f"Loading {resource_name} from local path: {model_id}")
        try:
            result = loader_class.from_pretrained(model_id, **kwargs)
            print(f"Successfully loaded {resource_name} from local path")
            return result
        except Exception as e:
            print(f"Error loading {resource_name} from local path: {e}")
            raise

    check_and_setup_model_cache(model_id)
    print(f"Loading {resource_name} {model_id}...")

    if _check_cache_complete(model_id):
        print("Found cached files, attempting to use local cache")
        try:
            kwargs_local = kwargs.copy()
            kwargs_local["local_files_only"] = True
            result = loader_class.from_pretrained(model_id, **kwargs_local)
            print(f"Successfully loaded {resource_name} from local cache")
            return result
        except Exception:
            print("Failed to load from local cache, will download from HuggingFace")

    try:
        result = loader_class.from_pretrained(model_id, **kwargs)
        print(f"Successfully loaded {resource_name}")
        return result
    except Exception as e:
        print(f"Error loading {resource_name}: {e}")
        raise


def load_model_with_cache(model_id, **kwargs):
    return _load_with_fallback(model_id, AutoModelForCausalLM, "model", **kwargs)


def _fix_tokenizer_decoder_if_needed(tokenizer, model_id):
    """Fix ByteLevel BPE tokenizer decoders lost by native-format conversion."""
    if not hasattr(tokenizer, "_tokenizer"):
        return

    tokenizer_json_path = Path(model_id) / "tokenizer.json"
    if not tokenizer_json_path.exists():
        return

    with open(tokenizer_json_path) as f:
        decoder_config = json.load(f).get("decoder", {})

    if decoder_config.get("type") != "ByteLevel":
        return

    import tokenizers as hf_tokenizers

    fast_tok = hf_tokenizers.Tokenizer.from_file(str(tokenizer_json_path))
    tokenizer._tokenizer.decoder = fast_tok.decoder
    print("Fixed tokenizer decoder: replaced ByteFallback with ByteLevel (from tokenizer.json)")


def load_tokenizer_with_cache(model_id, **kwargs):
    tokenizer = _load_with_fallback(model_id, AutoTokenizer, "tokenizer", **kwargs)
    _fix_tokenizer_decoder_if_needed(tokenizer, model_id)
    return tokenizer
