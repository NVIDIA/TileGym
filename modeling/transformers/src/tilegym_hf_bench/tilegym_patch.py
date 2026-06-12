# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from tilegym.transformers import apply_tilegym_kernel_to_deepseek_v2
from tilegym.transformers import apply_tilegym_kernel_to_gemma3
from tilegym.transformers import apply_tilegym_kernel_to_gpt_oss
from tilegym.transformers import apply_tilegym_kernel_to_llama
from tilegym.transformers import apply_tilegym_kernel_to_mistral
from tilegym.transformers import apply_tilegym_kernel_to_olmo3
from tilegym.transformers import apply_tilegym_kernel_to_olmoe
from tilegym.transformers import apply_tilegym_kernel_to_phi3
from tilegym.transformers import apply_tilegym_kernel_to_qwen2
from tilegym.transformers import apply_tilegym_kernel_to_qwen3


def apply_tilegym_patch(model_id, use_attn=False, use_cutile=False):
    model_name = model_id.lower()
    if "llama" in model_name:
        apply_tilegym_kernel_to_llama(rope=True, swiglu=True, rms_norm=True, attn=use_attn, use_cutile=use_cutile)
    elif "deepseek" in model_name:
        apply_tilegym_kernel_to_deepseek_v2(
            rope=True, rms_norm=True, swiglu=True, attn=use_attn, moe=True, use_cutile=use_cutile
        )
    elif "gpt-oss" in model_name:
        apply_tilegym_kernel_to_gpt_oss(rope=True, rms_norm=True, swiglu=False, attn=use_attn, use_cutile=use_cutile)
    elif "mistral" in model_name:
        apply_tilegym_kernel_to_mistral(rope=True, rms_norm=True, swiglu=True, attn=use_attn, use_cutile=use_cutile)
    elif "qwen3.5" in model_name or "qwen3_5" in model_name:
        apply_tilegym_kernel_to_qwen3(
            rope=True, rms_norm=True, swiglu=True, attn=use_attn, gated_delta_rule=True, use_cutile=use_cutile
        )
    elif "qwen" in model_name:
        apply_tilegym_kernel_to_qwen2(rope=True, rms_norm=True, swiglu=True, attn=use_attn, use_cutile=use_cutile)
    elif "gemma-3" in model_name or "gemma3" in model_name:
        apply_tilegym_kernel_to_gemma3(rope=True, rms_norm=True, mlp=True, attn=use_attn, use_cutile=use_cutile)
    elif "gemma" in model_name:
        print(f"Warning: Gemma variant {model_id} is not supported in tilegym patch. No optimizations will be applied.")
    elif "phi-3" in model_name or "phi3" in model_name:
        apply_tilegym_kernel_to_phi3(rope=True, rms_norm=True, swiglu=True, attn=use_attn, use_cutile=use_cutile)
    elif "olmoe" in model_name:
        apply_tilegym_kernel_to_olmoe(rope=True, rms_norm=True, attn=use_attn, moe=True, use_cutile=use_cutile)
    elif "olmo-3" in model_name or "olmo3" in model_name:
        apply_tilegym_kernel_to_olmo3(rope=True, rms_norm=True, swiglu=True, attn=use_attn, use_cutile=use_cutile)
    else:
        print(f"Warning: Model {model_id} is not supported in tilegym patch. No optimizations will be applied.")
