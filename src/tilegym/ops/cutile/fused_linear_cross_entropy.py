# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Forward-only chunked fused Linear + Cross-Entropy for cuTile experiments."""

import cuda.tile as ct
import torch
import torch.nn.functional as F
from torch import Tensor

ConstInt = ct.Constant[int]

_ALIGN = 8
_SM_COUNT = 0


def _get_sm_count() -> int:
    global _SM_COUNT
    if _SM_COUNT == 0:
        _SM_COUNT = torch.cuda.get_device_properties("cuda").multi_processor_count
    return _SM_COUNT


@ct.kernel(occupancy=1)
def _ce_online_kernel(
    logits,
    loss_out,
    target_logits,
    n_rows: ConstInt,
    vocab_size: ConstInt,
    tile_v: ConstInt,
):
    """2-pass online softmax over vocab tiles; writes loss and softmax probs in-place."""
    pid = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    num_chunks = ct.cdiv(vocab_size, tile_v)
    col_base = ct.arange(tile_v, dtype=ct.int32)

    for row in range(pid, n_rows, num_blocks):
        row_max = ct.full((1,), -1e30, dtype=ct.float32)
        sum_exp = ct.full((1,), 0.0, dtype=ct.float32)

        for chunk_idx in range(num_chunks):
            cols = ct.add(ct.full((tile_v,), chunk_idx * tile_v, dtype=ct.int32), col_base)
            chunk = ct.gather(logits, (row, cols), check_bounds=True, padding_value=-1e30)
            chunk_f32 = ct.astype(chunk, ct.float32)

            chunk_max = ct.max(chunk_f32, 0, keepdims=True)
            new_max = ct.maximum(row_max, chunk_max)
            sum_exp = ct.mul(sum_exp, ct.exp(ct.sub(row_max, new_max)))
            exp_chunk = ct.exp(ct.sub(chunk_f32, new_max))
            sum_exp = ct.add(sum_exp, ct.sum(exp_chunk, 0, keepdims=True))
            row_max = new_max

        lse = ct.add(row_max, ct.log(sum_exp))
        tgt_logit = ct.load(target_logits, index=(row,), shape=(1,), padding_mode=ct.PaddingMode.ZERO)
        tgt_logit = ct.astype(tgt_logit, ct.float32)
        loss = ct.sub(ct.reshape(lse, (1,)), tgt_logit)
        ct.store(loss_out, index=(row,), tile=loss, allow_tma=False)

        inv_sum = ct.truediv(ct.full((1,), 1.0, dtype=ct.float32), sum_exp)

        for chunk_idx in range(num_chunks):
            cols = ct.add(ct.full((tile_v,), chunk_idx * tile_v, dtype=ct.int32), col_base)
            chunk = ct.gather(logits, (row, cols), check_bounds=True, padding_value=-1e30)
            chunk_f32 = ct.astype(chunk, ct.float32)
            probs = ct.mul(ct.exp(ct.sub(chunk_f32, row_max)), inv_sum)
            ct.scatter(logits, (row, cols), ct.astype(probs, logits.dtype), check_bounds=True)


def _ce_cutile(logits_chunk: Tensor, target_chunk: Tensor, loss_chunk: Tensor, ignore_index: int) -> None:
    """Compute CE loss in-place for one (chunk_size, vocab) block."""
    n_rows, _vocab_size = logits_chunk.shape
    valid = target_chunk != ignore_index
    safe_target = target_chunk.clamp(min=0)
    rows = torch.arange(n_rows, device=logits_chunk.device)

    # Gather target logits once in PyTorch so the kernel can compute loss directly.
    target_logits = logits_chunk[rows, safe_target].float()
    target_logits[~valid] = 0.0

    tile_v = 4096
    grid = (min(_get_sm_count() * 4, n_rows),)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        _ce_online_kernel,
        (logits_chunk, loss_chunk, target_logits, n_rows, logits_chunk.shape[1], tile_v),
    )

    if not valid.all():
        loss_chunk[~valid] = 0.0


def _chunked_fwd_loss(
    x: Tensor,
    weight: Tensor,
    target: Tensor,
    chunk_size: int,
    ignore_index: int,
) -> Tensor:
    bt = x.shape[0]
    vocab_size = weight.shape[0]
    num_chunks = (bt + chunk_size - 1) // chunk_size

    loss = torch.empty(bt, device=x.device, dtype=torch.float32)
    # Reuse one logits buffer per BT chunk to avoid materializing full [BT, V].
    logits_buf = torch.empty((chunk_size, vocab_size), device=x.device, dtype=x.dtype)

    for i in range(num_chunks):
        start, end = i * chunk_size, min((i + 1) * chunk_size, bt)
        clen = end - start

        x_chunk = x[start:end]
        target_chunk = target[start:end]
        loss_chunk = loss[start:end]
        logits_chunk = logits_buf[:clen]

        # GEMM 1: logits = x @ W^T for this chunk.
        torch.mm(x_chunk, weight.mT, out=logits_chunk)
        _ce_cutile(logits_chunk, target_chunk, loss_chunk, ignore_index)

    return loss


def fused_linear_cross_entropy_forward_only(
    hidden_states: Tensor,
    weight: Tensor,
    target: Tensor,
    bias: Tensor | None = None,
    ignore_index: int = -100,
    chunk_size: int = 4096,
    reduction: str = "mean",
) -> Tensor:
    """Forward-only chunked fused linear + cross entropy.

    Notes:
    - Forward-only experimental API (not backend-registered in TileGym dispatch).
    - Main tradeoff: often higher latency than dense PyTorch CE, but much lower
      peak memory on large BT because full logits [BT, V] are not materialized.
    """
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    if hidden_states.ndim == 3:
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        target = target.reshape(-1)

    if bias is not None:
        logits = F.linear(hidden_states, weight, bias)
        return F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)

    bt = hidden_states.shape[0]

    # Pad BT for TensorCore-friendly GEMM alignment.
    pad = (-bt) % _ALIGN
    if pad:
        x_flat = F.pad(hidden_states, (0, 0, 0, pad))
        target_flat = F.pad(target.reshape(-1), (0, pad), value=ignore_index)
    else:
        x_flat = hidden_states
        target_flat = target.reshape(-1)

    loss = _chunked_fwd_loss(x_flat, weight, target_flat, chunk_size, ignore_index)

    if pad:
        loss = loss[:bt]

    if reduction == "sum":
        return loss.sum()

    n_valid = (target_flat[:bt] != ignore_index).sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=hidden_states.device, dtype=torch.float32)
    return loss.sum() / n_valid.float()
