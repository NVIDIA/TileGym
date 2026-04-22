# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.

"""
Rotary Position Embedding (RoPE) CuTile kernels.

Includes two autograd Functions:
  - _Fast_RoPE_Embedding_CT: single-tensor RoPE (Q only), registered as
    ``unsloth.rope_embedding``. Serves the general-purpose ``rope_embedding(Q,
    cos, sin)`` API for scenarios where only Q needs rotation (e.g. inference
    Q-projection, or architectures where K does not use RoPE).
  - _Fast_RoPE_Embedding_QK_CT: joint Q+K RoPE in one kernel launch,
    registered as ``unsloth.rope_embedding_qk``. This is the path used by
    Unsloth's ``fast_rope_embedding`` integration.

CuTile kernels:
  - _rope_embedding_ct: per-head grouped RoPE for (n_rows, n_heads, 2, half_dim)
  - _rope_embedding_QK_ct: joint Q+K RoPE for (batch, heads, seq, 2, half_dim)

Performance notes:
  - Autotune over occupancy=[1,2,4,8] via cuda.tile.tune.exhaustive_search.
  - Split-buffer pattern: kernel reads from Q_in and writes to Q_out.
    This allows autotune to re-run the kernel without corrupting input data
    (no clone needed). For backward (ct.launch, no autotune), Q_in=Q_out
    (same buffer, inplace).
  - Lane masking: all gather/scatter use ``mask = col_offsets < half_head_dim``
    to prevent out-of-range writes when TILE_HD > half_head_dim (non-power-of-2
    dims). When TILE_HD == half_head_dim the mask is all-True (no overhead).
  - Native bf16 computation: RoPE rotation stays in bf16 (cos/sin ∈ [-1,1]),
"""

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl

from .ct_ops import autotune_configs
from .ct_ops import calculate_settings

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

# Module-level tune caches: (n_rows, n_heads, seq_len, head_dim, TILE_HD, dtype, device) -> tuned_kernel
_rope_embedding_single_tune_cache: dict = {}
_rope_embedding_qk_tune_cache: dict = {}


# ---- CuTile kernel: joint Q+K RoPE (1D gather/scatter) ----


@ct.kernel
def _rope_embedding_QK_ct(
    Q_in,  # flattened: (total_Q_elements,) — read-only input
    Q_out,  # flattened: (total_Q_elements,) — write output
    K_in,  # flattened: (total_K_elements,) — read-only input
    K_out,  # flattened: (total_K_elements,) — write output
    cos,  # flattened: (seq_len * cos_row_stride,) — 1D gather access
    sin,  # same shape as cos, flattened
    rope_embedding_indices,  # (batch * seq_len,) int32, or dummy (1,)
    seqlen: ConstInt,
    head_dim: ConstInt,
    n_heads_Q: ConstInt,
    n_heads_K: ConstInt,
    cos_row_stride: ConstInt,
    BACKWARD_PASS: ConstInt,
    HAS_ROPE_INDICES: ConstInt,
    TILE_HD: ConstInt,
    NO_PADDING: ConstInt,
):
    """Joint Q+K RoPE using gather/scatter with split input/output buffers.

    Occupancy is injected at runtime by autotune_launch via hints_fn.
    All gather/scatter use ``mask = col_offsets < half_head_dim`` to prevent
    out-of-range lanes from corrupting adjacent heads when TILE_HD >
    half_head_dim (non-power-of-2 dims like head_dim=96,160).  When TILE_HD ==
    half_head_dim the mask is all-True and adds no overhead.
    Split-buffer: reads from Q_in/K_in, writes to Q_out/K_out. For backward
    (inplace, no autotune), Q_in=Q_out and K_in=K_out.
    """
    bid_row = ct.bid(0)  # 0 .. batch * seq_len - 1
    head_position = ct.bid(1)  # 0 .. n_heads_Q - 1

    batch_idx = bid_row // seqlen
    seq_idx = bid_row % seqlen
    half_head_dim = head_dim // 2
    col_offsets = ct.arange(TILE_HD, dtype=ct.int32)
    # Mask out-of-range lanes when TILE_HD > half_head_dim (non-power-of-2 dims
    # like head_dim=96,160). Prevents writing into adjacent heads' slots.
    # When TILE_HD == half_head_dim, mask is all-True (no functional change).
    mask = col_offsets < half_head_dim

    # Determine rotation position
    rot_position = seq_idx
    if HAS_ROPE_INDICES:
        rot_position = ct.gather(rope_embedding_indices, (bid_row,), padding_value=0).item()

    # Load cos/sin via 1D gather (matching single-tensor kernel pattern)
    cs_base = rot_position * cos_row_stride
    cos_row = ct.gather(cos, cs_base + col_offsets, mask=mask, check_bounds=False, padding_value=0)
    sin_row = ct.gather(sin, cs_base + col_offsets, mask=mask, check_bounds=False, padding_value=0)

    if BACKWARD_PASS:
        sin_row = -sin_row

    # Cast to cos/sin dtype only when needed (e.g., float32 cos with bf16 Q for Gemma).
    # When dtypes match (common bf16 case), skip cast to avoid extra instructions.
    # Matches single-tensor kernel (_rope_embedding_ct) behavior.

    # Q strides (contiguous layout: batch, n_heads_Q, seq_len, head_dim)
    q_base = batch_idx * n_heads_Q * seqlen * head_dim + head_position * seqlen * head_dim + seq_idx * head_dim
    q0 = ct.gather(Q_in, q_base + col_offsets, mask=mask, check_bounds=False, padding_value=0)
    q1 = ct.gather(Q_in, q_base + half_head_dim + col_offsets, mask=mask, check_bounds=False, padding_value=0)

    if Q_in.dtype != cos.dtype:
        q0 = ct.astype(q0, cos.dtype)
        q1 = ct.astype(q1, cos.dtype)

    new_q0 = q0 * cos_row - q1 * sin_row
    new_q1 = q1 * cos_row + q0 * sin_row

    if Q_in.dtype != cos.dtype:
        new_q0 = ct.astype(new_q0, Q_in.dtype)
        new_q1 = ct.astype(new_q1, Q_in.dtype)

    ct.scatter(Q_out, q_base + col_offsets, new_q0, mask=mask, check_bounds=False)
    ct.scatter(Q_out, q_base + half_head_dim + col_offsets, new_q1, mask=mask, check_bounds=False)

    # ---- Process K (only if this head exists in K) ----
    if head_position < n_heads_K:
        k_base = batch_idx * n_heads_K * seqlen * head_dim + head_position * seqlen * head_dim + seq_idx * head_dim
        k0 = ct.gather(K_in, k_base + col_offsets, mask=mask, check_bounds=False, padding_value=0)
        k1 = ct.gather(K_in, k_base + half_head_dim + col_offsets, mask=mask, check_bounds=False, padding_value=0)

        if K_in.dtype != cos.dtype:
            k0 = ct.astype(k0, cos.dtype)
            k1 = ct.astype(k1, cos.dtype)

        new_k0 = k0 * cos_row - k1 * sin_row
        new_k1 = k1 * cos_row + k0 * sin_row

        if K_in.dtype != cos.dtype:
            new_k0 = ct.astype(new_k0, K_in.dtype)
            new_k1 = ct.astype(new_k1, K_in.dtype)

        ct.scatter(K_out, k_base + col_offsets, new_k0, mask=mask, check_bounds=False)
        ct.scatter(K_out, k_base + half_head_dim + col_offsets, new_k1, mask=mask, check_bounds=False)


# ---- CuTile kernel: single-tensor RoPE (1D gather/scatter) ----


@ct.kernel
def _rope_embedding_ct(
    Q_in,  # flattened: (n_rows * n_heads * head_dim,) — read-only input
    Q_out,  # flattened: same shape — write output
    cos,  # flattened: (seq_len * cos_row_stride,) — gather-based access
    sin,  # same shape as cos, flattened
    seqlen: ConstInt,
    n_heads: ConstInt,
    head_dim: ConstInt,
    cos_row_stride: ConstInt,
    BACKWARD_PASS: ConstInt,
    TILE_HD: ConstInt,
    NO_PADDING: ConstInt,
):
    """Single-tensor RoPE using gather/scatter with split input/output buffers.

    One block per (row, head) — no ROPE_GROUP_SIZE loop. This maximizes
    parallelism and matches the QK kernel's 1-head-per-block pattern.

    Occupancy is injected at runtime by autotune_launch via hints_fn.
    All gather/scatter use ``mask = col_offsets < half_head_dim`` (see QK kernel
    docstring for rationale).
    Split-buffer: reads from Q_in, writes to Q_out. For backward (inplace, no
    autotune), Q_in=Q_out (same buffer).
    """
    row_position = ct.bid(0)  # 0 .. n_rows - 1
    head_idx = ct.bid(1)  # 0 .. n_heads - 1

    rot_idx = row_position % seqlen
    col_offsets = ct.arange(TILE_HD, dtype=ct.int32)
    half_head_dim = head_dim // 2
    # Mask out-of-range lanes (see QK kernel comment for rationale).
    mask = col_offsets < half_head_dim

    cs_base = rot_idx * cos_row_stride
    cos_row = ct.gather(cos, cs_base + col_offsets, mask=mask, check_bounds=False, padding_value=0)
    sin_row = ct.gather(sin, cs_base + col_offsets, mask=mask, check_bounds=False, padding_value=0)

    if BACKWARD_PASS:
        sin_row = -sin_row

    row_base = row_position * n_heads * head_dim
    offs_q0 = row_base + head_idx * head_dim + col_offsets
    offs_q1 = row_base + head_idx * head_dim + half_head_dim + col_offsets

    q0 = ct.gather(Q_in, offs_q0, mask=mask, check_bounds=False, padding_value=0)
    q1 = ct.gather(Q_in, offs_q1, mask=mask, check_bounds=False, padding_value=0)

    # Cast to cos/sin dtype only when needed (e.g., float32 cos with bf16 Q for Gemma).
    # When dtypes match (common bf16 case), skip cast to avoid extra instructions.
    if Q_in.dtype != cos.dtype:
        q0 = ct.astype(q0, cos.dtype)
        q1 = ct.astype(q1, cos.dtype)

    new_q0 = q0 * cos_row - q1 * sin_row
    new_q1 = q1 * cos_row + q0 * sin_row

    if Q_in.dtype != cos.dtype:
        new_q0 = ct.astype(new_q0, Q_in.dtype)
        new_q1 = ct.astype(new_q1, Q_in.dtype)

    ct.scatter(Q_out, offs_q0, new_q0, mask=mask, check_bounds=False)
    ct.scatter(Q_out, offs_q1, new_q1, mask=mask, check_bounds=False)


# ---- Autograd Function: single-tensor RoPE ----


class _Fast_RoPE_Embedding_CT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        cos, sin = cos.squeeze(), sin.squeeze()
        batch, seq_len, n_heads, head_dim = Q.shape
        half_head_dim = head_dim // 2

        assert seq_len <= cos.shape[0], f"seq_len ({seq_len}) > cos.shape[0] ({cos.shape[0]})"

        n_rows = batch * seq_len
        # Q is 4D contiguous from typical callers; reshape to 2D is a view.
        # Non-contiguous Q would be copied by reshape (4D→2D flat requires
        # contiguous strides for a view). No separate .contiguous() needed.
        Q_flat = Q.reshape(n_rows, n_heads * head_dim)

        # Ensure cos/sin are 2D contiguous
        if cos.dim() == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        cos = cos.contiguous()
        sin = sin.contiguous()
        cos_row_stride = cos.stride(0)

        TILE_HD = calculate_settings(half_head_dim)
        no_padding = int(TILE_HD == half_head_dim)

        Q_flat_1d = Q_flat.reshape(-1)
        # Split-buffer: autotune_launch re-runs the kernel across configs, so
        # in-place writes would corrupt input on retries. We allocate a separate
        # output buffer, doubling peak memory.
        # No public API exists to check for cached autotune results and switch
        # to ct.launch (in-place) after the first run.
        Q_result = torch.empty_like(Q_flat_1d)
        cos_flat = cos.reshape(-1)
        sin_flat = sin.reshape(-1)

        # Grid: one block per (row, head) — no ROPE_GROUP_SIZE loop.
        # Autotune over occupancy=[1,2,4,8] (matching layernorm.py pattern).
        # Split-buffer: Q_flat_1d is read-only, Q_result is write-only.
        stream = torch.cuda.current_stream()
        single_cache_key = (n_rows, n_heads, seq_len, head_dim, TILE_HD, cos_row_stride, Q.dtype, str(Q.device))
        if single_cache_key not in _rope_embedding_single_tune_cache:
            result = exhaustive_search(
                list(autotune_configs()),
                stream,
                lambda cfg: (n_rows, n_heads, 1),
                _rope_embedding_ct,
                lambda cfg: (
                    Q_flat_1d,
                    Q_result,
                    cos_flat,
                    sin_flat,
                    seq_len,
                    n_heads,
                    head_dim,
                    cos_row_stride,
                    0,
                    TILE_HD,
                    no_padding,
                ),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _rope_embedding_single_tune_cache[single_cache_key] = ct.kernel(
                _rope_embedding_ct._pyfunc,
                occupancy=best_cfg.occupancy,
            )
        tuned_kernel = _rope_embedding_single_tune_cache[single_cache_key]
        ct.launch(
            stream,
            (n_rows, n_heads, 1),
            tuned_kernel,
            (
                Q_flat_1d,
                Q_result,
                cos_flat,
                sin_flat,
                seq_len,
                n_heads,
                head_dim,
                cos_row_stride,
                0,
                TILE_HD,
                no_padding,
            ),
        )

        ctx.TILE_HD = TILE_HD
        ctx.no_padding = no_padding
        ctx.cos = cos
        ctx.sin = sin
        ctx.cos_row_stride = cos_row_stride
        return Q_result.view(batch, seq_len, n_heads, head_dim)

    @staticmethod
    def backward(ctx, dY):
        batch, seq_len, n_heads, head_dim = dY.shape
        half_head_dim = head_dim // 2
        n_rows = batch * seq_len
        dY_flat = dY.reshape(n_rows, n_heads * head_dim)

        # Backward: inplace (Q_in=Q_out), no autotune needed
        # Grid: one block per (row, head) — matching forward.
        dY_1d = dY_flat.reshape(-1)
        grid = (n_rows, n_heads, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _rope_embedding_ct,
            (
                dY_1d,  # Q_in = Q_out (same buffer, inplace)
                dY_1d,
                ctx.cos.reshape(-1),
                ctx.sin.reshape(-1),
                seq_len,
                n_heads,
                head_dim,
                ctx.cos_row_stride,
                1,  # BACKWARD_PASS = True
                ctx.TILE_HD,
                ctx.no_padding,
            ),
        )
        return dY_flat.view(batch, seq_len, n_heads, head_dim), None, None


# ---- Autograd Function: joint Q+K RoPE ----


class _Fast_RoPE_Embedding_QK_CT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, cos, sin, rope_indices):
        has_indices = rope_indices is not None
        cos, sin = cos.squeeze(), sin.squeeze()

        batch, n_heads_Q, seq_len, head_dim = Q.shape
        _, n_heads_K, _, _ = K.shape
        half_head_dim = head_dim // 2

        assert seq_len <= cos.shape[0], f"seq_len ({seq_len}) > cos.shape[0] ({cos.shape[0]})"

        # Q/K are 4D contiguous from typical callers; reshape(-1) is a view.
        # Non-contiguous inputs are copied (a flat 1D view of non-contiguous
        # 4D data is impossible). No separate .contiguous() needed.
        Q_flat = Q.reshape(-1)
        K_flat = K.reshape(-1)

        if has_indices:
            rope_ptr = rope_indices.reshape(-1).to(dtype=torch.int32, device=Q.device)
        else:
            rope_ptr = cos.new_empty(1, dtype=torch.int32)

        # Ensure cos/sin are 2D contiguous
        if cos.dim() == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        cos = cos.contiguous()
        sin = sin.contiguous()
        cos_row_stride = cos.stride(0)

        TILE_HD = calculate_settings(half_head_dim)
        no_padding = int(TILE_HD == half_head_dim)

        n_rows = batch * seq_len
        has_indices_int = int(has_indices)

        # Split-buffer: autotune_launch re-runs the kernel across configs, so
        # in-place writes would corrupt input on retries. Separate output
        # buffers double peak memory — a known CuTile autotune tradeoff.
        Q_result = torch.empty_like(Q_flat)
        K_result = torch.empty_like(K_flat)

        # Flatten cos/sin for 1D gather in kernel
        cos_flat = cos.reshape(-1)
        sin_flat = sin.reshape(-1)

        stream = torch.cuda.current_stream()
        qk_cache_key = (
            n_rows,
            n_heads_Q,
            n_heads_K,
            seq_len,
            head_dim,
            TILE_HD,
            has_indices_int,
            Q.dtype,
            str(Q.device),
        )
        if qk_cache_key not in _rope_embedding_qk_tune_cache:
            result = exhaustive_search(
                list(autotune_configs()),
                stream,
                lambda cfg: (n_rows, n_heads_Q, 1),
                _rope_embedding_QK_ct,
                lambda cfg: (
                    Q_flat,
                    Q_result,
                    K_flat,
                    K_result,
                    cos_flat,
                    sin_flat,
                    rope_ptr,
                    seq_len,
                    head_dim,
                    n_heads_Q,
                    n_heads_K,
                    cos_row_stride,
                    0,
                    has_indices_int,
                    TILE_HD,
                    no_padding,
                ),
                lambda cfg: {"occupancy": cfg.occupancy},
            )
            best_cfg = result.best.config
            _rope_embedding_qk_tune_cache[qk_cache_key] = ct.kernel(
                _rope_embedding_QK_ct._pyfunc,
                occupancy=best_cfg.occupancy,
            )
        tuned_qk_kernel = _rope_embedding_qk_tune_cache[qk_cache_key]
        ct.launch(
            stream,
            (n_rows, n_heads_Q, 1),
            tuned_qk_kernel,
            (
                Q_flat,
                Q_result,
                K_flat,
                K_result,
                cos_flat,
                sin_flat,
                rope_ptr,
                seq_len,
                head_dim,
                n_heads_Q,
                n_heads_K,
                cos_row_stride,
                0,  # BACKWARD_PASS = False
                has_indices_int,
                TILE_HD,
                no_padding,
            ),
        )

        ctx.TILE_HD = TILE_HD
        ctx.no_padding = no_padding
        ctx.has_indices = has_indices
        ctx.cos = cos
        ctx.sin = sin
        ctx.cos_row_stride = cos_row_stride
        ctx.rope_indices = rope_ptr if has_indices else None
        ctx.seq_len = seq_len
        ctx.n_heads_Q = n_heads_Q
        ctx.n_heads_K = n_heads_K

        return (
            Q_result.view(batch, n_heads_Q, seq_len, head_dim),
            K_result.view(batch, n_heads_K, seq_len, head_dim),
        )

    @staticmethod
    def backward(ctx, dQ, dK):
        batch, _, _, head_dim = dQ.shape
        half_head_dim = head_dim // 2

        rope_ptr = ctx.rope_indices if ctx.has_indices else ctx.cos.new_empty(1, dtype=torch.int32)

        # Inplace backward: Q_in=Q_out, K_in=K_out (same buffer)
        dQ_out = dQ.clone() if not dQ.is_contiguous() else dQ
        dK_out = dK.clone() if not dK.is_contiguous() else dK

        dQ_flat = dQ_out.reshape(-1)
        dK_flat = dK_out.reshape(-1)

        grid = (batch * ctx.seq_len, ctx.n_heads_Q, 1)
        ct.launch(
            torch.cuda.current_stream(),
            grid,
            _rope_embedding_QK_ct,
            (
                dQ_flat,  # Q_in = Q_out (same buffer, inplace)
                dQ_flat,
                dK_flat,  # K_in = K_out (same buffer, inplace)
                dK_flat,
                ctx.cos.reshape(-1),
                ctx.sin.reshape(-1),
                rope_ptr,
                ctx.seq_len,
                head_dim,
                ctx.n_heads_Q,
                ctx.n_heads_K,
                ctx.cos_row_stride,
                1,  # BACKWARD_PASS = True
                int(ctx.has_indices),
                ctx.TILE_HD,
                ctx.no_padding,
            ),
        )

        return dQ_out, dK_out, None, None, None


# ---- Registered dispatch implementations ----


@register_impl("unsloth.rope_embedding", backend="cutile")
def rope_embedding(Q, cos, sin):
    return _Fast_RoPE_Embedding_CT.apply(Q, cos, sin)


@register_impl("unsloth.rope_embedding_qk", backend="cutile")
def rope_embedding_qk(Q, K, cos, sin, rope_indices=None):
    return _Fast_RoPE_Embedding_QK_CT.apply(Q, K, cos, sin, rope_indices)
