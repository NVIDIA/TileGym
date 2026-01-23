# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import inspect

from transformers import PreTrainedModel

from tilegym import set_backend
from tilegym.logger import get_logger
from tilegym.ops import get_apply_rope_func
from tilegym.ops import get_fmha_interface
from tilegym.ops import get_fused_swiglu_module
from tilegym.ops import get_rms_norm_module
from tilegym.ops import get_swiglu_module
from tilegym.transformers.deepseek2.modeling_deepseek import DeepseekV2MoETileGym
from tilegym.transformers.deepseek2.modeling_deepseek import tilegym_deepseek_v2_forward

logger = get_logger(__name__)


def apply_tilegym_kernel_to_llama(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    attn: bool = True,
    model: PreTrainedModel = None,
    use_cutile: bool = False,
) -> None:
    """
    Apply TileGym kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        rope (bool): Whether to apply TileGym's rotary position embedding. Default is True.
        rms_norm (bool): Whether to apply TileGym's RMSNorm. Default is True.
        swiglu (bool): Whether to apply TileGym's SwiGLU MLP. Default is True.
        attn (bool): Whether to apply TileGym's attention. Default is True.
        model (PreTrainedModel): The model instance to apply TileGym kernels to, if the model has already been
        loaded. Default is None.
        use_cutile (bool): Whether to apply using cutile. Default is False.
    """
    logger.info("--------------------------------")
    logger.info("apply_tilegym_kernel_to_llama")
    logger.info("--------------------------------")
    from transformers.models.llama import modeling_llama

    if use_cutile:
        set_backend("cutile")

    if rope:
        modeling_llama.apply_rotary_pos_emb = get_apply_rope_func(model="llama")
    if rms_norm:
        modeling_llama.LlamaRMSNorm = get_rms_norm_module()
    if swiglu:
        modeling_llama.LlamaMLP = get_swiglu_module()
    if attn:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["sdpa"] = get_fmha_interface()


def apply_tilegym_kernel_to_deepseek_v2(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    attn: bool = True,
    moe: bool = True,
    model: PreTrainedModel = None,
    use_cutile: bool = False,
) -> None:
    """
    Apply TileGym kernels to replace original implementation in HuggingFace DeepSeek V2 models

    Args:
        rope (bool): Whether to apply TileGym's rotary position embedding. Default is True.
        rms_norm (bool): Whether to apply TileGym's RMSNorm. Default is True.
        swiglu (bool): Whether to apply TileGym's SwiGLU MLP for shared experts. Default is True.
        attn (bool): Whether to apply TileGym's Multi-head Latent Attention. Default is True.
        moe (bool): Whether to apply TileGym's fused MoE. Default is True.
        model (PreTrainedModel): The model instance to apply kernels to, if the model has already been
        loaded. Default is None.
        use_cutile (bool): Whether to use cutile backend. Default is False.
    """
    logger.info("--------------------------------")
    logger.info("apply_tilegym_kernel_to_deepseek_v2")
    logger.info("--------------------------------")
    from transformers.models.deepseek_v2 import modeling_deepseek_v2 as modeling_deepseek

    if use_cutile:
        set_backend("cutile")

    if rope:
        modeling_deepseek.apply_rotary_emb = get_apply_rope_func(model="deepseek")

    if rms_norm:
        modeling_deepseek.DeepseekV2RMSNorm = get_rms_norm_module()

    if swiglu:
        # Replace DeepseekV2MLP with TileGym's FUSED SwiGLU implementation
        # This eliminates ALL PyTorch linear operations by fusing gate+up+down projections.
        # This is critical for shared experts which run on every token.
        modeling_deepseek.DeepseekV2MLP = get_fused_swiglu_module()

    if attn:
        # Replace attention forward with TileGym implementation
        modeling_deepseek.DeepseekV2Attention.forward = tilegym_deepseek_v2_forward
    if moe:
        modeling_deepseek.DeepseekV2MoE = DeepseekV2MoETileGym


def apply_tilegym_kernel_to_qwen2(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    attn: bool = True,
    model: PreTrainedModel = None,
    use_cutile: bool = False,
) -> None:
    """
    Apply TileGym kernels to replace original implementation in HuggingFace Qwen2 models

    Args:
        rope (bool): Whether to apply TileGym's rotary position embedding. Default is True.
        rms_norm (bool): Whether to apply TileGym's RMSNorm. Default is True.
        swiglu (bool): Whether to apply TileGym's SwiGLU MLP. Default is True.
        attn (bool): Whether to apply TileGym's attention. Default is True.
        model (PreTrainedModel): The model instance to apply TileGym kernels to, if the model has already been
        loaded. Default is None.
        use_cutile (bool): Whether to apply using cutile. Default is False.
    """
    logger.info("--------------------------------")
    logger.info("apply_tilegym_kernel_to_qwen2")
    logger.info("--------------------------------")
    from transformers.models.qwen2 import modeling_qwen2

    if use_cutile:
        set_backend("cutile")

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = get_apply_rope_func(model="qwen2")
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm = get_rms_norm_module()
    if swiglu:
        modeling_qwen2.Qwen2MLP = get_swiglu_module()
    if attn:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["sdpa"] = get_fmha_interface()


MODEL_TYPE_TO_APPLY_TILEGYM_FN = {
    "llama": apply_tilegym_kernel_to_llama,
    "deepseek_v2": apply_tilegym_kernel_to_deepseek_v2,
    "qwen2": apply_tilegym_kernel_to_qwen2,
}


def _apply_tilegym_kernel(model_type: str, **kwargs) -> None:
    """
    Applies TileGym kernels based on the specified model type. The custom
    kernels for the specified model type will be applied with the provided
    keyword arguments, otherwise the default configuration will be used.

    ** Note: This must be called before model initialization.

    Args:
        - model_type: the model types as defined in transformers/models/auto/modeling_auto.py
          and specified in the model's config.json
        - kwargs: keyword arguments that are passed to the corresponding apply_TileGym_kernel_to_* function.
    """
    if not model_type:
        logger.info("Model type was not provided. No TileGym kernels will be applied.")
        return

    if model_type not in MODEL_TYPE_TO_APPLY_TILEGYM_FN.keys():
        logger.info(f"There are currently no TileGym kernels supported for model type: {model_type}.")
        return

    apply_fn = MODEL_TYPE_TO_APPLY_TILEGYM_FN[model_type]
    apply_fn_signature = inspect.signature(apply_fn)

    # Filter out the keyword arguments that are not supported by the apply function
    applicable_kwargs = {key: value for key, value in kwargs.items() if key in apply_fn_signature.parameters}

    logger.info(f"Applying TileGym kernels for model type: {model_type} with kwargs: {applicable_kwargs}")

    # Assume this is invoked pre-model initialization, so we only need to patch transformers code
    apply_fn(**applicable_kwargs)
