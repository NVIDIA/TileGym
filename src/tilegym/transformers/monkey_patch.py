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
from tilegym.transformers.phi3.modeling_phi3 import Phi3MLPTileGym
from tilegym.transformers.phi3.modeling_phi3 import get_fmha_phi3_interface

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


def apply_tilegym_kernel_to_gpt_oss(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = False,  # Set to False by default since GPT-OSS has custom expert implementation
    attn: bool = True,
    use_cutile: bool = False,
) -> None:
    """
    Apply TileGym kernels to replace original implementation in HuggingFace GPT-OSS models.

    NOTE: GPT-OSS is supported in transformers >= 4.55.0
    NOTE: SwiGLU patching is disabled by default for GPT-OSS as it uses a custom expert
          implementation with clamping and MXFP4 quantization.

    Args:
        rope (bool): Whether to apply TileGym's rotary position embedding. Default is True.
        rms_norm (bool): Whether to apply TileGym's RMSNorm. Default is True.
        swiglu (bool): Whether to apply TileGym's SwiGLU MLP. Default is False.
            Note: GPT-OSS uses a custom expert implementation, so SwiGLU patching is disabled by default.
        attn (bool): Whether to apply TileGym's attention. Default is True.
        use_cutile (bool): Whether to use cutile backend. Default is False.
    """
    import transformers
    from packaging import version

    if version.parse(transformers.__version__) < version.parse("4.55.0"):
        logger.warning("GPT-OSS support requires transformers >= 4.55.0")
        return

    logger.info("--------------------------------")
    logger.info("apply_tilegym_kernel_to_gpt_oss")
    logger.info("--------------------------------")

    from transformers.models.gpt_oss import modeling_gpt_oss

    if use_cutile:
        set_backend("cutile")

    if rope:
        modeling_gpt_oss.apply_rotary_pos_emb = get_apply_rope_func(model="gpt-oss")

    if rms_norm:
        modeling_gpt_oss.GptOssRMSNorm = get_rms_norm_module()

    if attn:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from tilegym.ops import get_attention_sink_interface

        # GPT-OSS uses attention sinks, so we use the attention sink interface
        modeling_gpt_oss.eager_attention_forward = get_attention_sink_interface()


def apply_tilegym_kernel_to_gemma3(
    rope: bool = True,
    rms_norm: bool = True,
    mlp: bool = True,
    attn: bool = True,
    model: PreTrainedModel = None,
    use_cutile: bool = False,
) -> None:
    """
    Apply TileGym kernels to replace original implementation in HuggingFace Gemma3 models

    Args:
        rope (bool): Whether to apply TileGym's rotary position embedding. Default is True.
        rms_norm (bool): Whether to apply TileGym's RMSNorm. Default is True.
        mlp (bool): Whether to apply TileGym's MLP (GEGLU). Default is True.
        attn (bool): Whether to apply TileGym's attention. Default is True.
        model (PreTrainedModel): The model instance to apply TileGym kernels to, if the model has already been
        loaded. Default is None.
        use_cutile (bool): Whether to apply using cutile. Default is False.
    """
    logger.info("--------------------------------")
    logger.info("apply_tilegym_kernel_to_gemma3")
    logger.info("--------------------------------")
    from transformers.models.gemma3 import modeling_gemma3 as modeling_gemma

    if use_cutile:
        set_backend("cutile")

    if rope:
        modeling_gemma.apply_rotary_pos_emb = get_apply_rope_func(model="gemma3")
    if rms_norm:
        modeling_gemma.Gemma3RMSNorm = get_rms_norm_module(model="gemma3")
    if mlp:
        # Use PartiallyFusedGEGLUMLP for Gemma3 which uses GELU activation
        from tilegym.ops.fused_mlp import PartiallyFusedGEGLUMLP

        modeling_gemma.Gemma3MLP = PartiallyFusedGEGLUMLP
        logger.info("✅ Replaced Gemma3MLP with PartiallyFusedGEGLUMLP (GEGLU fusion)")
    if attn:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from tilegym.ops import get_fmha_gemma3_interface

        gemma3_interface = get_fmha_gemma3_interface()

        ALL_ATTENTION_FUNCTIONS["eager"] = gemma3_interface
        ALL_ATTENTION_FUNCTIONS["sdpa"] = gemma3_interface

        if hasattr(modeling_gemma, "eager_attention_forward"):
            modeling_gemma.eager_attention_forward = gemma3_interface
            logger.info("✅ Replaced Gemma3 attention with TileGym FMHA (soft cap + sliding window)")
            logger.info("   Registered to: eager, sdpa, and replaced eager_attention_forward")
        else:
            logger.info("✅ Replaced Gemma3 attention with TileGym FMHA (soft cap + sliding window)")
            logger.info("   Registered to: eager, sdpa")


def apply_tilegym_kernel_to_mistral(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    attn: bool = True,
    model: PreTrainedModel = None,
    use_cutile: bool = False,
) -> None:
    """
    Apply TileGym kernels to replace original implementation in HuggingFace Mistral models.

    Mistral uses the same core architecture as Llama (GQA attention with RoPE, SwiGLU MLP,
    RMSNorm) with the addition of sliding window attention.

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
    logger.info("apply_tilegym_kernel_to_mistral")
    logger.info("--------------------------------")
    from transformers.models.mistral import modeling_mistral

    if use_cutile:
        set_backend("cutile")

    if rope:
        modeling_mistral.apply_rotary_pos_emb = get_apply_rope_func(model="llama")
    if rms_norm:
        modeling_mistral.MistralRMSNorm = get_rms_norm_module()
    if swiglu:
        modeling_mistral.MistralMLP = get_swiglu_module()
    if attn:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["sdpa"] = get_fmha_interface()


def apply_tilegym_kernel_to_phi3(
    rope: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    attn: bool = True,
    model: PreTrainedModel = None,
    use_cutile: bool = False,
) -> None:
    """
    Apply TileGym kernels to replace original implementation in HuggingFace Phi-3 models.

    Phi-3 uses fused gate_up_proj for MLP, so we keep checkpoint compatibility with a
    TileGym-aware Phi3MLP replacement that preserves original parameter names.

    Args:
        rope (bool): Whether to apply TileGym's rotary position embedding. Default is True.
        rms_norm (bool): Whether to apply TileGym's RMSNorm. Default is True.
        swiglu (bool): Whether to apply TileGym's SwiGLU path for Phi3MLP. Default is True.
        attn (bool): Whether to apply TileGym's attention. Default is True.
        model (PreTrainedModel): The model instance to apply TileGym kernels to, if the model has already been
        loaded. Default is None.
        use_cutile (bool): Whether to apply using cutile. Default is False.
    """
    logger.info("--------------------------------")
    logger.info("apply_tilegym_kernel_to_phi3")
    logger.info("--------------------------------")
    from transformers.models.phi3 import modeling_phi3

    if use_cutile:
        set_backend("cutile")

    if rope:
        # Phi-3 rotary semantics match LLaMA-style apply_rotary_pos_emb.
        modeling_phi3.apply_rotary_pos_emb = get_apply_rope_func(model="llama")
    if rms_norm:
        modeling_phi3.Phi3RMSNorm = get_rms_norm_module()
    if swiglu:
        modeling_phi3.Phi3MLP = Phi3MLPTileGym
    if attn:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS["sdpa"] = get_fmha_phi3_interface()


MODEL_TYPE_TO_APPLY_TILEGYM_FN = {
    "llama": apply_tilegym_kernel_to_llama,
    "deepseek_v2": apply_tilegym_kernel_to_deepseek_v2,
    "gpt_oss": apply_tilegym_kernel_to_gpt_oss,
    "mistral": apply_tilegym_kernel_to_mistral,
    "qwen2": apply_tilegym_kernel_to_qwen2,
    "gemma3": apply_tilegym_kernel_to_gemma3,
    "phi3": apply_tilegym_kernel_to_phi3,
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
