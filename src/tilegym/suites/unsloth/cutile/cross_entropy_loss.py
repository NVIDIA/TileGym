# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.

"""
Cross-Entropy Loss CuTile kernels.

Includes autograd Function with forward + backward.
CuTile kernels:
  - _cross_entropy_forward_ct: single-chunk forward (vocab <= 65536)
  - _chunked_cross_entropy_forward_ct: multi-chunk forward (vocab > 65536)
  - _cross_entropy_backward_ct: backward pass (handles both variants)
"""

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .ct_ops import MAX_FUSED_SIZE
from .ct_ops import calculate_settings

_CHUNKED_FWD_BLOCK_SIZE = 4096  # block size for chunked forward path (large vocab)

ConstInt = ct.Constant[int]
ConstFloat = ct.Constant[float]


# ---- CuTile kernel: single-chunk cross-entropy forward ----


@ct.kernel
def _cross_entropy_forward_ct(
    logits,  # (n_rows, vocab_size) — 2D
    loss,  # (n_rows,)
    logsumexp_out,  # (n_rows,)
    labels,  # (n_rows,)
    VOCAB_SIZE: ConstInt,
    BLOCK_SIZE: ConstInt,
    DO_SOFTCAPPING: ConstInt,
    SOFTCAP: ConstFloat,
    DO_LOGIT_SCALING: ConstInt,
    LOGIT_SCALE: ConstFloat,
):
    """
    Single-chunk cross-entropy forward for vocab_size <= 65536.

    CE_i = logsumexp(logits) - logit_at_label
    Uses 2D gather on (n_rows, vocab_size) to avoid int32 overflow.
    """
    row_idx = ct.bid(0)
    col_offsets = ct.arange(BLOCK_SIZE, dtype=ct.int32)

    # Load label (scalar)
    label_idx = ct.gather(labels, (row_idx,), padding_value=-100).item()

    # Load logits row via 1D gather (flat load_ptr_tko IR, avoids
    # tensor_view/partition_view abstraction that causes predicate explosion)
    # Bounds-check elimination when BLOCK_SIZE == VOCAB_SIZE (power-of-2 vocab)
    no_padding = BLOCK_SIZE == VOCAB_SIZE
    logits_row = ct.gather(logits, (row_idx, col_offsets), check_bounds=not no_padding, padding_value=0)
    logits_row = ct.astype(logits_row, ct.float32)
    if BLOCK_SIZE > VOCAB_SIZE:
        logits_row = ct.where(col_offsets < VOCAB_SIZE, logits_row, -math.inf)

    # Apply logit scaling (Cohere): t * x
    if DO_LOGIT_SCALING:
        logits_row = LOGIT_SCALE * logits_row

    # Apply logit softcapping (Gemma 2): t * tanh(x / t)
    if DO_SOFTCAPPING:
        logits_row = SOFTCAP * ct.tanh(logits_row / SOFTCAP)

    # Numerically stable logsumexp: c + log(sum(exp(x - c)))
    c = ct.max(logits_row, 0)
    lse = c + ct.log(ct.sum(ct.exp(logits_row - c), 0))

    # Compute loss = logsumexp - x_label
    # Direct O(1) gather from original logits + re-apply transforms (not O(BLOCK_SIZE) scan)
    loss_val = lse * 0.0  # Same type as lse (scalar tile), zero-initialized
    if label_idx != -100:
        # Scalar gather: (row_idx, label_idx) → scalar result, matching loss_val type
        x_raw = ct.gather(logits, (row_idx, label_idx), padding_value=0)
        x = ct.astype(x_raw, ct.float32)
        if DO_LOGIT_SCALING:
            x = LOGIT_SCALE * x
        if DO_SOFTCAPPING:
            x = SOFTCAP * ct.tanh(x / SOFTCAP)
        loss_val = lse - x

    ct.scatter(logsumexp_out, (row_idx,), lse)
    ct.scatter(loss, (row_idx,), loss_val)


# ---- CuTile kernel: chunked cross-entropy forward (vocab > 65536) ----


@ct.kernel
def _chunked_cross_entropy_forward_ct(
    logits,  # (n_rows, vocab_size) — 2D
    loss,  # (n_rows,)
    logsumexp_out,  # (n_rows, n_chunks) — 2D
    labels,  # (n_rows,)
    VOCAB_SIZE: ConstInt,
    N_CHUNKS: ConstInt,
    BLOCK_SIZE: ConstInt,
    DO_SOFTCAPPING: ConstInt,
    SOFTCAP: ConstFloat,
    DO_LOGIT_SCALING: ConstInt,
    LOGIT_SCALE: ConstFloat,
):
    """
    Chunked cross-entropy forward for vocab_size > 65536.

    Each block processes one chunk of BLOCK_SIZE vocab elements.
    Chunk 0 also computes loss = -x_label (host adds logsumexp later).
    """
    row_idx = ct.bid(0)
    chunk_idx = ct.bid(1)

    # Column offsets for this chunk
    col_offsets = chunk_idx * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    label_idx = ct.gather(labels, (row_idx,), padding_value=-100).item()

    # Load logits chunk
    logits_chunk = ct.gather(logits, (row_idx, col_offsets), check_bounds=True, padding_value=-math.inf)
    logits_chunk = ct.astype(logits_chunk, ct.float32)

    # Apply logit scaling
    if DO_LOGIT_SCALING:
        logits_chunk = LOGIT_SCALE * logits_chunk

    # Apply logit softcapping
    if DO_SOFTCAPPING:
        logits_chunk = SOFTCAP * ct.tanh(logits_chunk / SOFTCAP)

    # Per-chunk logsumexp
    c = ct.max(logits_chunk, 0)
    lse = c + ct.log(ct.sum(ct.exp(logits_chunk - c), 0))

    # Chunk 0: compute the -x_label part of the loss
    if chunk_idx == 0:
        # Default: 0 loss for ignored labels. Use scalar to match
        # ct.scatter(loss, (row_idx,), ...) which expects scalar shape ().
        loss_val = 0.0
        if label_idx != -100:
            # Scalar gather from the FULL logits row using Python int index
            # (same pattern as single-chunk kernel — avoids tile/scalar type mismatch)
            x_raw = ct.gather(logits, (row_idx, label_idx), padding_value=0)
            x = ct.astype(x_raw, ct.float32)
            if DO_LOGIT_SCALING:
                x = LOGIT_SCALE * x
            if DO_SOFTCAPPING:
                x = SOFTCAP * ct.tanh(x / SOFTCAP)
            # Store just the -x part; host adds logsumexp(all chunks) later
            loss_val = -1.0 * x
        ct.scatter(loss, (row_idx,), loss_val)

    # Store per-chunk logsumexp
    ct.scatter(logsumexp_out, (row_idx, chunk_idx), lse)


# ---- CuTile kernel: cross-entropy backward ----


@ct.kernel
def _cross_entropy_backward_ct(
    logits,  # (n_rows, vocab_size) — 2D, read-only: original logits from forward
    grad_logits,  # (n_rows, vocab_size) — 2D, write-only: output gradient buffer
    dloss,  # (n_rows,) — upstream gradient
    logsumexp_in,  # (n_rows,) — saved from forward
    labels,  # (n_rows,)
    VOCAB_SIZE: ConstInt,
    BLOCK_SIZE: ConstInt,
    DO_SOFTCAPPING: ConstInt,
    SOFTCAP: ConstFloat,
    DO_LOGIT_SCALING: ConstInt,
    LOGIT_SCALE: ConstFloat,
):
    """
    Cross-entropy backward: compute dlogits into a separate output buffer.

    Reads from logits (saved from forward) and writes gradients to grad_logits,
    avoiding in-place modification of saved tensors (PyTorch autograd version tracking).

    dC/dx = dloss * (softmax(x) - 1_{x==label})
    With chain rule through softcapping and scaling.
    """
    row_idx = ct.bid(0)
    block_idx = ct.bid(1)
    col_offsets = block_idx * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)

    label_idx = ct.gather(labels, (row_idx,), padding_value=-100).item()

    # Load upstream gradient (scalar)
    dloss_val = 0.0
    if label_idx != -100:
        dloss_val = ct.gather(dloss, (row_idx,), padding_value=0).item()

    # Load logits chunk via gather (read-only access to saved logits)
    x = ct.gather(logits, (row_idx, col_offsets), check_bounds=True, padding_value=0)
    x = ct.astype(x, ct.float32)

    # Apply logit scaling
    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    # Apply softcapping and save partial for gradient
    # Pre-define partial before branch
    partial = x
    if DO_SOFTCAPPING:
        # d/dx [t * tanh(x/t)] = 1 - tanh^2(x/t)
        partial = ct.tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    # Load saved logsumexp
    lse = ct.gather(logsumexp_in, (row_idx,), padding_value=0).item()

    # Compute softmax: exp(x - logsumexp)
    y = ct.exp(x - lse)

    # Subtract 1 at label position: softmax - 1_{label}
    y = ct.where(col_offsets == label_idx, y - 1.0, y)

    # Chain rule through logit scaling
    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    # Chain rule through softcapping: * (1 - tanh^2)
    if DO_SOFTCAPPING:
        y = y * (1.0 - partial * partial)

    # Store gradient to separate output buffer, masked to valid vocab positions
    result = ct.astype(dloss_val * y, grad_logits.dtype)
    ct.scatter(grad_logits, (row_idx, col_offsets), result, check_bounds=True)


# ---- Autograd Function ----


class _Fast_CrossEntropyLoss_CT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0, logit_scaling=0):
        n_rows, vocab_size = logits.shape
        device = logits.device
        labels = labels.to(device)

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device=device)

        DO_SOFTCAPPING = bool(logit_softcapping != 0)
        DO_LOGIT_SCALING = bool(logit_scaling != 0)

        stream = torch.cuda.current_stream()

        if n_chunks == 1:
            # Small vocabs <= 65536 (Llama, Mistral)
            BLOCK_SIZE = calculate_settings(vocab_size)
            logsumexp = torch.empty(n_rows, dtype=torch.float32, device=device)

            ct.launch(
                stream,
                (n_rows, 1, 1),
                _cross_entropy_forward_ct,
                (
                    logits,
                    losses,
                    logsumexp,
                    labels,
                    vocab_size,
                    BLOCK_SIZE,
                    int(DO_SOFTCAPPING),
                    float(logit_softcapping),
                    int(DO_LOGIT_SCALING),
                    float(logit_scaling),
                ),
            )
        else:
            # Large vocabs > 65536 (Gemma 256K)
            # Use smaller block size for better performance on large vocabs
            BLOCK_SIZE = _CHUNKED_FWD_BLOCK_SIZE
            div_c, mod_c = divmod(vocab_size, BLOCK_SIZE)
            n_chunks = div_c + (mod_c != 0)
            logsumexp = torch.empty((n_rows, n_chunks), dtype=torch.float32, device=device)

            ct.launch(
                stream,
                (n_rows, n_chunks, 1),
                _chunked_cross_entropy_forward_ct,
                (
                    logits,
                    losses,
                    logsumexp,
                    labels,
                    vocab_size,
                    n_chunks,
                    BLOCK_SIZE,
                    int(DO_SOFTCAPPING),
                    float(logit_softcapping),
                    int(DO_LOGIT_SCALING),
                    float(logit_scaling),
                ),
            )
            # Reduce per-chunk logsumexp and add to losses
            logsumexp = torch.logsumexp(logsumexp, dim=1)
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.DO_SOFTCAPPING = DO_SOFTCAPPING
        ctx.logit_softcapping = logit_softcapping
        ctx.DO_LOGIT_SCALING = DO_LOGIT_SCALING
        ctx.logit_scaling = logit_scaling
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        # Ensure dlosses is contiguous for gather access
        dlosses = dlosses.contiguous()

        # Allocate separate output buffer to avoid in-place modification of
        # saved tensors (violates PyTorch autograd version tracking)
        grad_logits = torch.empty_like(logits)

        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            (n_rows, n_blocks, 1),
            _cross_entropy_backward_ct,
            (
                logits,
                grad_logits,
                dlosses,
                logsumexp,
                labels,
                vocab_size,
                BLOCK_SIZE,
                int(ctx.DO_SOFTCAPPING),
                float(ctx.logit_softcapping),
                int(ctx.DO_LOGIT_SCALING),
                float(ctx.logit_scaling),
            ),
        )
        return grad_logits, None, None, None


# ---- Registered dispatch implementation ----


@register_impl("unsloth.cross_entropy_loss", backend="cutile")
def cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0, n_items=None):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert labels.shape == (batch, seq_len)

    device = logits.device
    loss = _Fast_CrossEntropyLoss_CT.apply(
        logits.view(batch * seq_len, d),
        labels.view(-1),
        logit_softcapping,
        logit_scaling,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    if torch.is_tensor(n_items):
        n_items = n_items.to(device)
    return loss.sum() / n_items
