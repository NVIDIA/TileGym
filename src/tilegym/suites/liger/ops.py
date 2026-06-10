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
