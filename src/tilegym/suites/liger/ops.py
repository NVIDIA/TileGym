# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
liger Suite - Unified interface for Liger-Kernel compatible operations
"""

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import torch

from tilegym.backend import dispatch
from tilegym.backend import get_current_backend


@dispatch(
    "liger.jsd",
)
def jsd(
    input: torch.Tensor,
    target: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Generalized Jensen-Shannon Divergence loss.

    JSD(β)(P || Q) = β * KL(P || M) + (1-β) * KL(Q || M), M = β*P + (1-β)*Q.

    Args:
        input: Student model log-probabilities with shape (BT, V)
        target: Teacher model log-probabilities with shape (BT, V)
        shift_labels: Optional token indices for per-row masking, shape (BT,)
        beta: Interpolation coefficient in [0, 1].
            beta=0 → forward KL, beta=1 → reverse KL, beta=0.5 → symmetric JSD.
            Default: 0.5
        ignore_index: Label index to ignore when shift_labels is provided. Default: -100

    Returns:
        Scalar loss tensor
    """
    raise NotImplementedError(f"jsd is not implemented for {get_current_backend()}")


@dispatch(
    "liger.poly_norm",
)
def poly_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    PolyNorm normalization.

    Computes: y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
    where norm(u) = u / sqrt(mean(u²) + ε)

    Reference:
    1. https://github.com/BryceZhuo/PolyCom/
    2. https://arxiv.org/pdf/2411.03884

    Args:
        input: Input tensor of shape (*, H)
        weight: Weight tensor of shape (3,) for [w0, w1, w2]
        bias: Scalar bias tensor of shape (1,)
        eps: Epsilon for numerical stability. Default: 1e-6

    Returns:
        Output tensor of same shape as input
    """
    raise NotImplementedError(f"poly_norm is not implemented for {get_current_backend()}")


@dispatch(
    "liger.fused_neighborhood_attention",
)
def fused_neighborhood_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kernel_size: int = 7,
    dilation: int = 1,
    scale: float = None,
) -> torch.Tensor:
    raise NotImplementedError(f"fused_neighborhood_attention not implemented for {get_current_backend()}")


@dispatch(
    "liger.cross_entropy",
)
def cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    weight: Optional[torch.Tensor] = None,
    lse_square_scale: float = 0.0,
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
) -> torch.Tensor:
    """
    Fused cross-entropy loss with in-kernel gradient computation.

    Computes cross entropy loss and pre-computes the gradient of the input
    in a single kernel pass (Liger-style fused forward+backward).

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py

    Args:
        input: Logit tensor of shape (BT, V) where BT = batch*seq_len, V = vocab size.
            Must require grad for gradient computation to occur.
        target: Target class indices of shape (BT,).
        ignore_index: Class index to ignore when computing loss and gradient. Default: -100
        label_smoothing: Amount of label smoothing in [0, 1). Default: 0.0
        reduction: Reduction mode: "mean" | "sum" | "none". Default: "mean"
        weight: Optional per-class weight tensor of shape (V,). Default: None
        lse_square_scale: Z-loss scale: adds lse_square_scale * logsumexp^2 to loss. Default: 0.0
        softcap: If set, caps logits to (-softcap, +softcap) via softcap*tanh(x/softcap). Default: None
        return_z_loss: Return z_loss as second element of 4-tuple. Default: False
        return_token_accuracy: Return token accuracy as third element of 4-tuple. Default: False
        return_predicted_tokens: Return predicted token indices as fourth element of 4-tuple. Default: False

    Returns:
        Scalar loss tensor when all RETURN_* flags are False (default, backward-compatible).
        4-tuple (loss, z_loss, token_accuracy, predicted_tokens) when any RETURN_* flag is True.
    """
    raise NotImplementedError(f"cross_entropy is not implemented for {get_current_backend()}")


@dispatch(
    "liger.fused_add_rms_norm",
)
def fused_add_rms_norm(
    X: torch.Tensor,
    R: torch.Tensor,
    W: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused residual addition + RMS normalization.

    Computes the following sequence (common in transformer decoder layers):
      1. hidden_states = residual + hidden_states
      2. residual = hidden_states  (updated residual)
      3. hidden_states = rmsnorm(hidden_states)

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_add_rms_norm.py

    Args:
        X: Hidden states tensor of shape (*, H).
        R: Residual tensor of same shape as X.
        W: RMSNorm weight tensor of shape (H,).
        eps: Epsilon for numerical stability. Default: 1e-6
        offset: Constant offset added to W before scaling (e.g. 1.0 for Gemma). Default: 0.0
        casting_mode: Casting mode for RMSNorm computation:
            "llama" - only rstd computed in float32 (default)
            "gemma" - everything cast to float32
            "none"  - no casting; compute in original dtype

    Returns:
        Tuple (Y, S):
            Y: Normalized output of same shape as X.
            S: Updated residual (X + R) of same shape as X.
    """
    raise NotImplementedError(f"fused_add_rms_norm is not implemented for {get_current_backend()}")


@dispatch(
    "liger.fused_linear_cross_entropy",
)
def fused_linear_cross_entropy(
    input: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ce_weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    lse_square_scale: float = 0.0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    softcap: Optional[float] = None,
    return_z_loss: bool = False,
    accum_dtype: Optional[torch.dtype] = None,
    use_token_scaling: bool = False,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
) -> torch.Tensor:
    """
    Fused linear + cross-entropy loss (chunked to avoid materializing logits).

    Computes loss = cross_entropy(input @ weight.T + bias, target) without
    materializing the full (BT, V) logit matrix.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py

    Args:
        input: Hidden states of shape (BT, H).
        weight: Vocabulary weight of shape (V, H).
        target: Target class indices of shape (BT,).
        bias: Optional bias of shape (V,). Default: None
        ce_weight: Optional per-class weight of shape (V,) for weighted CE. Default: None
        ignore_index: Class index to ignore. Default: -100
        lse_square_scale: Coefficient of the z-loss regularizer (lse^2). Default: 0.0
        label_smoothing: Label smoothing factor in [0, 1). Default: 0.0
        reduction: Reduction mode: "mean" | "sum" | "none". Default: "mean"
        softcap: Optional logit soft-capping value (tanh cap). Default: None
        return_z_loss: Also return the z-loss term. Default: False
        accum_dtype: Optional accumulation dtype for grad_weight/grad_bias. Default: None
        use_token_scaling: Scale each token's loss by its predicted probability. Default: False
        return_token_accuracy: Also return per-batch token accuracy. Default: False
        return_predicted_tokens: Also return the argmax predicted token ids. Default: False

    Returns:
        Scalar loss tensor by default (or per-token losses when reduction="none").
        When any of return_z_loss / return_token_accuracy / return_predicted_tokens is set,
        returns a tuple (loss, *extras) with the requested extras appended in that order.
    """
    raise NotImplementedError(f"fused_linear_cross_entropy is not implemented for {get_current_backend()}")


@dispatch(
    "liger.fused_linear_jsd",
)
def fused_linear_jsd(
    student_input: torch.Tensor,
    student_weight: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_weight: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    beta: float = 0.5,
    ignore_index: int = -100,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Fused linear + Jensen-Shannon Divergence loss (chunked to avoid materializing logits).

    Computes JSD between student and teacher distributions without materializing the
    full (BT, V) logit matrices.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_jsd.py

    Args:
        student_input: Student hidden states of shape (BT, H).
        student_weight: Student vocabulary weight of shape (V, H).
        teacher_input: Teacher hidden states of shape (BT, H).
        teacher_weight: Teacher vocabulary weight of shape (V, H).
        shift_labels: Optional token indices for masking, shape (BT,). Default: None
        beta: JSD interpolation coefficient in [0, 1]. Default: 0.5
        ignore_index: Label index to ignore. Default: -100
        temperature: Temperature for softmax scaling. Default: 1.0

    Returns:
        Scalar loss tensor.
    """
    raise NotImplementedError(f"fused_linear_jsd is not implemented for {get_current_backend()}")


@dispatch(
    "liger.geglu",
)
def geglu(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """
    GEGLU activation: c = GELU(a) * b using tanh approximation.

    Computes: c = 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3))) * b

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/geglu.py

    Args:
        a: Input gate tensor of shape (*, N).
        b: Input value tensor of shape (*, N).

    Returns:
        Output tensor of same shape as a and b.
    """
    raise NotImplementedError(f"geglu is not implemented for {get_current_backend()}")


@dispatch(
    "liger.group_norm",
)
def group_norm(
    X: torch.Tensor,
    num_channels: int,
    num_groups: int,
    W: torch.Tensor,
    B: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Group Normalization.

    Divides channels into groups and normalizes within each group.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/group_norm.py

    Args:
        X: Input tensor of shape (batch_size, num_channels, *spatial).
        num_channels: Total number of channels.
        num_groups: Number of groups to divide channels into.
        W: Affine scale weight of shape (num_channels,).
        B: Affine shift bias of shape (num_channels,).
        eps: Epsilon for numerical stability. Default: 1e-5

    Returns:
        Normalized output tensor of same shape as X.
    """
    raise NotImplementedError(f"group_norm is not implemented for {get_current_backend()}")


@dispatch(
    "liger.dyt",
)
def dyt(
    x: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dynamic Tanh (DyT) activation: y = tanh(alpha * x) * gamma + beta

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/dyt.py

    Args:
        x: Input tensor of shape (*, N)
        alpha: Scalar learnable parameter, shape (1,)
        gamma: Per-channel scale, shape (N,)
        beta: Optional per-channel bias, shape (N,). If None, bias is omitted.

    Returns:
        Output tensor of same shape as x
    """
    raise NotImplementedError(f"dyt is not implemented for {get_current_backend()}")


@dispatch(
    "liger.kl_div",
)
def kl_div(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "batchmean",
    log_target: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    KL Divergence loss: KL(y_true || y_pred).

    Expects y_pred as log-probabilities. y_true can be probabilities (default)
    or log-probabilities (when log_target=True).

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/kl_div.py

    Args:
        y_pred: Log-probability predictions of shape (BT, V).
        y_true: Target values of shape (BT, V). Probabilities when log_target=False,
            log-probabilities when log_target=True.
        reduction: Reduction mode: "none" | "sum" | "mean" | "batchmean". Default: "batchmean"
        log_target: If True, y_true is treated as log-probabilities. Default: False
        eps: Small value for numerical stability (clamping y_true). Default: 1e-10

    Returns:
        Loss tensor. Shape (BT, V) when reduction="none", scalar otherwise.
    """
    raise NotImplementedError(f"kl_div is not implemented for {get_current_backend()}")


@dispatch(
    "liger.layer_norm",
)
def layer_norm(
    X: torch.Tensor,
    W: torch.Tensor,
    B: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Layer Normalization.

    Normalizes each row of X independently, then applies affine transform Y = norm(X) * W + B.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/layer_norm.py

    Args:
        X: Input tensor of shape (*, H).
        W: Affine scale weight of shape (H,).
        B: Affine shift bias of shape (H,).
        eps: Epsilon for numerical stability. Default: 1e-5

    Returns:
        Normalized output tensor of same shape as X.
    """
    raise NotImplementedError(f"layer_norm is not implemented for {get_current_backend()}")


@dispatch(
    "liger.llama4_rope",
)
def llama4_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    BLOCK_SIZE: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Llama4-style Rotary Position Embedding (RoPE) applied in-place to q and k.

    Performs complex multiplication: (q_r + i*q_i) * (f_r + i*f_i).

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/llama4_rope.py

    Args:
        q: Query tensor of shape (batch_size, seq_len, n_q_heads, head_dim).
        k: Key tensor of shape (batch_size, seq_len, n_k_heads, head_dim).
        freqs_cis: Frequency tensor of shape (seq_len, head_dim//2) complex,
            or (seq_len, head_dim//2, 2) real, or (seq_len, head_dim) real.
        BLOCK_SIZE: Tile size for kernel (auto-selected if None). Default: None

    Returns:
        Tuple (q, k) with rotary embeddings applied in-place.
    """
    raise NotImplementedError(f"llama4_rope is not implemented for {get_current_backend()}")


@dispatch(
    "liger.grpo_loss",
)
def grpo_loss(
    logits: torch.Tensor,
    old_logp: Optional[torch.Tensor],
    ref_logp: Optional[torch.Tensor],
    completion_ids: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: Optional[torch.Tensor] = None,
    temperature: float = 0.9,
    beta: float = 0.0,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    inplace: bool = True,
    loss_type: str = "grpo",
    max_completion_length: Optional[int] = None,
    reduce: bool = False,
    importance_sampling_level: str = "token",
    sapo_temperature_pos: float = 1.0,
    sapo_temperature_neg: float = 1.05,
    vllm_is_ratio: Optional[torch.Tensor] = None,
    delta: Optional[float] = None,
    use_bias_correction_kl: bool = False,
    num_items_in_batch: Optional[torch.Tensor] = None,
    phi_seq: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    GRPO / DAPO / BNPO / DR-GRPO / CISPO / SAPO / LUSPO / VESPO policy optimization loss.

    Computes per-token policy gradient loss with optional KL penalty.
    Logits shape: (B, L+1, N) where L is sequence length and N is vocab size.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/grpo_loss.py

    Args:
        logits: Logits of shape (B, L+1, N). The extra token is for the prefix position.
        old_logp: Old log-probs of shape (B, L), or None to use current logits.
        ref_logp: Reference log-probs of shape (B, L). Required when beta != 0.
        completion_ids: Token ids of shape (B, L) (int64).
        advantages: Per-sample advantages of shape (B,).
        completion_mask: Optional binary mask of shape (B, L). 0 = skip token.
        temperature: Softmax temperature. Default: 0.9
        beta: KL penalty coefficient. Default: 0.0
        eps_low: Lower clip epsilon for PPO. Default: 0.2
        eps_high: Upper clip epsilon for PPO. Default: 0.2
        inplace: Write gradient directly into logits buffer. Default: True
        loss_type: Algorithm variant: "grpo"|"dapo"|"bnpo"|"dr_grpo"|"cispo"|"sapo"|"luspo"|"vespo".
        max_completion_length: Maximum completion length for dr_grpo/luspo reduction. Default: None
        reduce: If True, return a reduced scalar loss instead of per-token tensors. Default: False
        importance_sampling_level: "token" or "sequence" (GSPO) IS correction level. Default: "token"
            Sequence-level is implemented for the CuTile backend; the Triton backend raises for it.
        sapo_temperature_pos: SAPO sigmoid temperature for positive advantages. Default: 1.0
        sapo_temperature_neg: SAPO sigmoid temperature for negative advantages. Default: 1.05
        vllm_is_ratio: Optional importance sampling ratio tensor.
        delta: Dual-sided clipping upper bound from INTELLECT-2 (clamps coef_1). Default: None
        use_bias_correction_kl: Enable DeepSeek-V3.2 IS-corrected KL: kl *= coef_1. Default: False
        num_items_in_batch: Optional global token count for DAPO/CISPO/VESPO normalization. Default: None
        phi_seq: Per-sequence gamma weight of shape (B,) or (B, 1); required for loss_type="vespo". Default: None

    Returns:
        If reduce=False:
            Tuple (loss, kl, is_clipped):
                loss: Per-token loss of shape (B, L), float32.
                kl: Per-token KL divergence of shape (B, L) or None when beta=0.
                is_clipped: Per-token clip indicator of shape (B, L), float32.
        If reduce=True:
            Tuple (loss, kl, clip_ratio):
                loss: Reduced scalar loss.
                kl: Mean KL divergence scalar or None when beta=0.
                clip_ratio: Fraction of clipped tokens, scalar.
    """
    raise NotImplementedError(f"grpo_loss is not implemented for {get_current_backend()}")


@dispatch(
    "liger.rms_norm",
)
def rms_norm(
    X: torch.Tensor,
    W: Optional[torch.Tensor],
    eps: float,
    offset: float = 0.0,
    casting_mode: str = "llama",
    in_place: bool = True,
    row_mode: Optional[bool] = None,
) -> torch.Tensor:
    """
    RMS Normalization: Y = X / RMS(X) * (W + offset).

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py

    Args:
        X: Input tensor of shape (*, H).
        W: Affine scale weight of shape (H,), or None for no affine transform.
        eps: Epsilon for numerical stability.
        offset: Constant added to W (e.g., 1.0 for Gemma). Default: 0.0
        casting_mode: "llama" | "gemma" | "none" controls internal precision. Default: "llama"
        in_place: Reuse dY buffer for dX in backward to save memory. Default: True
        row_mode: Force row kernel if True. Default: None (auto)

    Returns:
        Normalized output tensor of same shape as X.
    """
    raise NotImplementedError(f"rms_norm is not implemented for {get_current_backend()}")


@dispatch(
    "liger.qwen2vl_mrope",
)
def qwen2vl_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Qwen2VL Multimodal Rotary Positional Embedding (M-RoPE).

    Applies rotary embeddings to q and k using temporal / height / width sections.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py

    Args:
        q: Query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        k: Key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
        cos: Cosine tensor of shape (3, bsz, seq_len, head_dim).
        sin: Sine tensor of shape (3, bsz, seq_len, head_dim).
        mrope_section: List [t_section, h_section] with the number of head-dim
            positions allocated to temporal and height embeddings.

    Returns:
        Tuple (q, k) with M-RoPE applied in-place.
    """
    raise NotImplementedError(f"qwen2vl_mrope is not implemented for {get_current_backend()}")


@dispatch(
    "liger.rope",
)
def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary Positional Embedding (RoPE) — HuggingFace Llama/Mistral variant.

    Half-split layout: left half = real, right half = imaginary.
      forward:  new_r = r*cos - i*sin,  new_i = i*cos + r*sin
      backward: new_r = r*cos + i*sin,  new_i = i*cos - r*sin

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rope.py

    Args:
        q: Query tensor of shape (bsz, n_q_heads, seq_len, head_dim).
        k: Key tensor of shape (bsz, n_kv_heads, seq_len, head_dim).
        cos: Cosine tensor of shape (1_or_bsz, seq_len, head_dim).
        sin: Sine tensor of shape (1_or_bsz, seq_len, head_dim).

    Returns:
        Tuple (q, k) with RoPE applied.
    """
    raise NotImplementedError(f"rope is not implemented for {get_current_backend()}")


@dispatch(
    "liger.softmax",
)
def softmax(
    input: torch.Tensor,
) -> torch.Tensor:
    """
    Softmax applied on the last dimension.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/softmax.py

    Args:
        input: Input tensor of any shape (*, n_cols).

    Returns:
        Softmax output of same shape as input.
    """
    raise NotImplementedError(f"softmax is not implemented for {get_current_backend()}")


@dispatch(
    "liger.sparsemax",
)
def sparsemax(
    input: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Sparsemax: projects input onto the probability simplex (sparse alternative to softmax).

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/sparsemax.py

    Args:
        input: Input tensor of any shape.
        dim: Dimension along which sparsemax is computed. Default: -1

    Returns:
        Sparsemax output of same shape as input.
    """
    raise NotImplementedError(f"sparsemax is not implemented for {get_current_backend()}")


@dispatch(
    "liger.swiglu",
)
def swiglu(
    a: torch.Tensor,
    b: torch.Tensor,
    gate_multiplier: float = 1.0,
    down_multiplier: float = 1.0,
) -> torch.Tensor:
    """
    SwiGLU activation: c = silu(a * gate_multiplier) * b * down_multiplier
    where silu(x) = x * sigmoid(x).

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py

    Args:
        a: Gate tensor of shape (*, N).
        b: Value tensor of shape (*, N).
        gate_multiplier: Scalar multiplier applied to ``a`` before SiLU. Default 1.0.
        down_multiplier: Scalar multiplier applied to the output ``silu(a*gm)*b``. Default 1.0.

    Returns:
        Output tensor of same shape as a and b.
    """
    raise NotImplementedError(f"swiglu is not implemented for {get_current_backend()}")


@dispatch(
    "liger.tiled_mlp",
)
def tiled_mlp(
    fn: Callable,
    mlp_module: torch.nn.Module,
    x: torch.Tensor,
    num_shards: Optional[int] = None,
    compute_params: Optional[List] = None,
) -> torch.Tensor:
    """
    Tiled MLP computation for memory-efficient long-sequence processing.

    Shards the input along the sequence dimension, applies fn on each shard,
    and concatenates the results. Backward re-computes forward per shard.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/tiled_mlp.py

    Args:
        fn: Function to apply on each shard: fn(mlp_module, x_shard) -> output_shard.
        mlp_module: The MLP nn.Module object.
        x: Input tensor of shape (*, seq_len, hidden_size).
        num_shards: Number of shards. If None, auto-computed as ceil(seq_len/hidden_size).
        compute_params: Optional list of parameters for ZeRO optimization. Default: None

    Returns:
        Output tensor of same shape as x.
    """
    raise NotImplementedError(f"tiled_mlp is not implemented for {get_current_backend()}")


@dispatch(
    "liger.tvd",
)
def tvd(
    p: torch.Tensor,
    q: torch.Tensor,
    shift_labels: Optional[torch.Tensor] = None,
    reduction: str = "batchmean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Total Variation Distance loss: TVD(P || Q) = 0.5 * |P - Q|.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/tvd.py

    Args:
        p: First distribution of shape (BT, V). Can be float16/bfloat16/float32.
        q: Second distribution of shape (BT, V).
        shift_labels: Optional token labels of shape (BT,) for ignore_index masking. Default: None
        reduction: Reduction mode: "none" | "sum" | "mean" | "batchmean". Default: "batchmean"
        ignore_index: Label value to ignore when shift_labels is provided. Default: -100

    Returns:
        Loss tensor. Shape (BT, V) when reduction="none", scalar otherwise.
    """
    raise NotImplementedError(f"tvd is not implemented for {get_current_backend()}")


@dispatch(
    "liger.multi_token_attention",
)
def multi_token_attention(
    scores: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    sparse: bool = False,
) -> torch.Tensor:
    """
    Multi-Token Attention: causal masking + softmax + conv2d + causal masking.

    Applies a causal lower-triangular mask, softmax attention, a learnable 2D
    convolution over the attention matrix, and a final causal zero-mask.

    Reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/multi_token_attention.py

    Args:
        scores: Attention score tensor of shape (*, L, L).
        weight: Conv2d weight of shape (out_channels, in_channels/groups, kH, kW).
        bias: Optional conv2d bias of shape (out_channels,). Default: None
        stride: Conv2d stride. Default: 1
        padding: Conv2d padding. Default: 0
        dilation: Conv2d dilation. Default: 1
        groups: Conv2d groups. Default: 1
        sparse: Use sparsemax instead of softmax. Default: False

    Returns:
        Output tensor of same shape as scores.
    """
    raise NotImplementedError(f"multi_token_attention is not implemented for {get_current_backend()}")
