# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Rotary Positional Embedding (RoPE) kernel (CuTile backend).

HuggingFace Llama/Mistral variant -- half-split layout:
  left half = real part, right half = imaginary part.
  forward:  new_r = r*cos - i*sin,   new_i = i*cos + r*sin
  backward: new_r = r*cos + i*sin,   new_i = i*cos - r*sin  (sin_sign=-1.0)

Two kernel variants selected at runtime via the ALIGNED flag:

  ALIGNED case (power-of-2 head_dim AND all head counts exactly match tile sizes):
    _rope_4d_ct -- operates on Q/K in original (bsz, n_heads, seq_len, head_dim) layout;
                   uses block indices 0/1 in the last dim for real/imag halves.
                   COS/SIN passed in (cos_bs, seq_len, head_dim) form -- no reshape needed.
                   Avoids expensive host-side transpose+contiguous+reshape copies.

  Non-ALIGNED case:
    _rope_5d_ct -- operates on Q/K in (bsz, seq_len, n_heads, 2, head_dim_half) layout;
                   uses padding_mode=PAD_ZERO for safety on non-power-of-2 shapes.
                   COS/SIN in (cos_bs, seq_len, 1, head_dim_half) form (via _prepare_cos_sin).

Grid: (bsz * seq_len,) -- one block per token.

PERF NOTES:
- ALIGNED 4D path eliminates ~0.035 ms of host-side tensor manipulation per call
  (transpose+contiguous on Q and K, plus _prepare_cos_sin reshape).
- COS/SIN TMA: for ALIGNED, load directly from (cos_bs, seq_len, head_dim) -- block index 0
  in the last dim grabs elements [0, TILE_HD) = the cosine/sine values for this token.
- Q/K TMA: block index 0/1 in last dim selects real/imag halves of head_dim.
"""

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel
def _rope_4d_ct(
    Q,  # (bsz, n_q_heads, seq_len, head_dim) -- original layout, head_dim = 2*TILE_HD
    K,  # (bsz, n_k_heads, seq_len, head_dim)
    COS,  # (cos_bs, seq_len, head_dim) -- first TILE_HD elements are the cos values
    SIN,  # (cos_bs, seq_len, head_dim)
    cos_bs: ConstInt,
    seq_len: ConstInt,
    sin_sign: ct.Constant[float],
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """Fast path for ALIGNED shapes: no host-side transpose or reshape needed."""
    cos_bs = COS.shape[0]

    pid = ct.bid(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    # Load first TILE_HD elements of cos/sin: these are the rotation values for this token.
    # Index (cos_batch_idx, seq_idx, 0) loads COS[cos_batch_idx, seq_idx, 0:TILE_HD].
    # COS/SIN are kept in their original (typically fp32) dtype -- no on-chip cast here,
    # the multiply below implicitly promotes to fp32; cast back to Q/K.dtype at store time.
    cos_row = ct.load(COS, index=(cos_batch_idx, seq_idx, 0), shape=(1, 1, TILE_HD)).reshape((1, TILE_HD))
    sin_row = ct.load(SIN, index=(cos_batch_idx, seq_idx, 0), shape=(1, 1, TILE_HD)).reshape((1, TILE_HD)) * sin_sign

    # Q in (bsz, n_q_heads, seq_len, head_dim): index (b, h, s, 0) = real half,
    # index (b, h, s, 1) = imag half (block 1 starts at element TILE_HD = head_dim_half).
    q_r = ct.load(Q, index=(batch_idx, 0, seq_idx, 0), shape=(1, TILE_QH, 1, TILE_HD)).reshape((TILE_QH, TILE_HD))
    q_i = ct.load(Q, index=(batch_idx, 0, seq_idx, 1), shape=(1, TILE_QH, 1, TILE_HD)).reshape((TILE_QH, TILE_HD))
    new_q_r = q_r * cos_row - q_i * sin_row
    new_q_i = q_i * cos_row + q_r * sin_row
    ct.store(Q, index=(batch_idx, 0, seq_idx, 0), tile=new_q_r.reshape((1, TILE_QH, 1, TILE_HD)).astype(Q.dtype))
    ct.store(Q, index=(batch_idx, 0, seq_idx, 1), tile=new_q_i.reshape((1, TILE_QH, 1, TILE_HD)).astype(Q.dtype))

    # K in (bsz, n_k_heads, seq_len, head_dim)
    k_r = ct.load(K, index=(batch_idx, 0, seq_idx, 0), shape=(1, TILE_KH, 1, TILE_HD)).reshape((TILE_KH, TILE_HD))
    k_i = ct.load(K, index=(batch_idx, 0, seq_idx, 1), shape=(1, TILE_KH, 1, TILE_HD)).reshape((TILE_KH, TILE_HD))
    new_k_r = k_r * cos_row - k_i * sin_row
    new_k_i = k_i * cos_row + k_r * sin_row
    ct.store(K, index=(batch_idx, 0, seq_idx, 0), tile=new_k_r.reshape((1, TILE_KH, 1, TILE_HD)).astype(K.dtype))
    ct.store(K, index=(batch_idx, 0, seq_idx, 1), tile=new_k_i.reshape((1, TILE_KH, 1, TILE_HD)).astype(K.dtype))


@ct.kernel
def _rope_5d_ct(
    Q,  # (bsz, seq_len, n_q_heads, 2, head_dim_half) -- 5D layout for non-aligned case
    K,  # (bsz, seq_len, n_k_heads, 2, head_dim_half)
    COS,  # (cos_bs, seq_len, 1, head_dim_half)
    SIN,  # (cos_bs, seq_len, 1, head_dim_half)
    cos_bs: ConstInt,
    seq_len: ConstInt,
    sin_sign: ct.Constant[float],
    TILE_QH: ConstInt,
    TILE_KH: ConstInt,
    TILE_HD: ConstInt,
):
    """Fallback path for non-ALIGNED shapes: uses PAD_ZERO for non-power-of-2 dims."""
    cos_bs = COS.shape[0]

    pid = ct.bid(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len
    cos_batch_idx = 0 if cos_bs == 1 else batch_idx

    cos_row = ct.astype(
        ct.load(COS, index=(cos_batch_idx, seq_idx, 0, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO).reshape(
            (1, TILE_HD)
        ),
        ct.float32,
    )
    sin_row = (
        ct.astype(
            ct.load(SIN, index=(cos_batch_idx, seq_idx, 0, 0), shape=(1, 1, 1, TILE_HD), padding_mode=PAD_ZERO).reshape(
                (1, TILE_HD)
            ),
            ct.float32,
        )
        * sin_sign
    )

    q_r = ct.astype(
        ct.load(
            Q, index=(batch_idx, seq_idx, 0, 0, 0), shape=(1, 1, TILE_QH, 1, TILE_HD), padding_mode=PAD_ZERO
        ).reshape((TILE_QH, TILE_HD)),
        ct.float32,
    )
    q_i = ct.astype(
        ct.load(
            Q, index=(batch_idx, seq_idx, 0, 1, 0), shape=(1, 1, TILE_QH, 1, TILE_HD), padding_mode=PAD_ZERO
        ).reshape((TILE_QH, TILE_HD)),
        ct.float32,
    )
    new_q_r = q_r * cos_row - q_i * sin_row
    new_q_i = q_i * cos_row + q_r * sin_row
    ct.store(
        Q,
        index=(batch_idx, seq_idx, 0, 0, 0),
        tile=new_q_r.reshape((1, 1, TILE_QH, 1, TILE_HD)).astype(Q.dtype),
    )
    ct.store(
        Q,
        index=(batch_idx, seq_idx, 0, 1, 0),
        tile=new_q_i.reshape((1, 1, TILE_QH, 1, TILE_HD)).astype(Q.dtype),
    )

    k_r = ct.astype(
        ct.load(
            K, index=(batch_idx, seq_idx, 0, 0, 0), shape=(1, 1, TILE_KH, 1, TILE_HD), padding_mode=PAD_ZERO
        ).reshape((TILE_KH, TILE_HD)),
        ct.float32,
    )
    k_i = ct.astype(
        ct.load(
            K, index=(batch_idx, seq_idx, 0, 1, 0), shape=(1, 1, TILE_KH, 1, TILE_HD), padding_mode=PAD_ZERO
        ).reshape((TILE_KH, TILE_HD)),
        ct.float32,
    )
    new_k_r = k_r * cos_row - k_i * sin_row
    new_k_i = k_i * cos_row + k_r * sin_row
    ct.store(
        K,
        index=(batch_idx, seq_idx, 0, 0, 0),
        tile=new_k_r.reshape((1, 1, TILE_KH, 1, TILE_HD)).astype(K.dtype),
    )
    ct.store(
        K,
        index=(batch_idx, seq_idx, 0, 1, 0),
        tile=new_k_i.reshape((1, 1, TILE_KH, 1, TILE_HD)).astype(K.dtype),
    )


def _prepare_cos_sin(cos, sin, seq_len, head_dim_half):
    """Slice cos/sin to head_dim_half and reshape to (cos_bs, seq_len, 1, head_dim_half).

    Used only for the non-ALIGNED (5D) path. Keeps cos/sin in their original dtype;
    the float32 cast is done on-chip inside the kernels.
    """
    cos_bs = cos.shape[0]
    cos_4d = cos[..., :head_dim_half].contiguous().reshape(cos_bs, seq_len, 1, head_dim_half)
    sin_4d = sin[..., :head_dim_half].contiguous().reshape(cos_bs, seq_len, 1, head_dim_half)
    return cos_4d, sin_4d, cos_bs


class RopeCuTileFunction(torch.autograd.Function):
    """CuTile autograd wrapper for RoPE.

    ALIGNED case (power-of-2 head_dim, all head counts exactly match tile sizes):
      Uses _rope_4d_ct on Q/K without transpose or reshape. COS/SIN passed in 3D form.
      This is the common case for LLMs (head_dim=64/128/256, n_heads=power-of-2).

    Non-ALIGNED case: falls back to _rope_5d_ct with transpose+reshape+contiguous.
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        # q: (bsz, n_q_heads, seq_len, head_dim)
        # k: (bsz, n_k_heads, seq_len, head_dim)
        bsz, n_q_heads, seq_len, head_dim = q.shape
        n_k_heads = k.shape[1]
        head_dim_half = head_dim // 2
        original_dtype = q.dtype

        TILE_HD = next_power_of_2(head_dim_half)
        TILE_QH = next_power_of_2(n_q_heads)
        TILE_KH = next_power_of_2(n_k_heads)
        # Require contiguous inputs for the 4D path: non-contiguous q/k (e.g. from
        # a transpose of seq-first storage) would force an unavoidable copy inside
        # the ALIGNED branch that doesn't happen in Liger's seq-first design.
        # Falling back to the 5D path lets transpose+contiguous be a no-op.
        ALIGNED = (
            (TILE_HD == head_dim_half)
            and (TILE_QH == n_q_heads)
            and (TILE_KH == n_k_heads)
            and q.is_contiguous()
            and k.is_contiguous()
        )

        n_row = bsz * seq_len
        grid = (n_row,)

        if ALIGNED:
            # Fast path: 4D kernel -- no transpose, no reshape, no _prepare_cos_sin
            cos_3d = cos.contiguous()  # (cos_bs, seq_len, head_dim)
            sin_3d = sin.contiguous()
            cos_bs = cos_3d.shape[0]
            q_in = q.contiguous()
            k_in = k.contiguous()
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _rope_4d_ct,
                (
                    q_in,
                    k_in,
                    cos_3d,
                    sin_3d,
                    int(cos_bs),
                    int(seq_len),
                    float(1.0),
                    int(TILE_QH),
                    int(TILE_KH),
                    int(TILE_HD),
                ),
            )
            ctx.save_for_backward(cos_3d, sin_3d)
            ctx.cos_4d = None
        else:
            # Slow path: 5D kernel with transpose+contiguous
            q_t = q.transpose(1, 2).contiguous()
            k_t = k.transpose(1, 2).contiguous()
            cos_4d, sin_4d, cos_bs = _prepare_cos_sin(cos, sin, seq_len, head_dim_half)
            q_5d = q_t.view(bsz, seq_len, n_q_heads, 2, head_dim_half)
            k_5d = k_t.view(bsz, seq_len, n_k_heads, 2, head_dim_half)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _rope_5d_ct,
                (
                    q_5d,
                    k_5d,
                    cos_4d,
                    sin_4d,
                    int(cos_bs),
                    int(seq_len),
                    float(-1.0 if False else 1.0),
                    int(TILE_QH),
                    int(TILE_KH),
                    int(TILE_HD),
                ),
            )
            q_in = q_t.view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2).to(original_dtype)
            k_in = k_t.view(bsz, seq_len, n_k_heads, head_dim).transpose(1, 2).to(original_dtype)
            cos_bs = cos_4d.shape[0]
            ctx.save_for_backward(cos_4d, sin_4d)
            ctx.cos_4d = True

        ctx.bsz = bsz
        ctx.seq_len = seq_len
        ctx.n_q_heads = n_q_heads
        ctx.n_k_heads = n_k_heads
        ctx.head_dim = head_dim
        ctx.cos_bs = cos_bs
        ctx.original_dtype = original_dtype
        ctx.ALIGNED = ALIGNED
        ctx.TILE_QH = TILE_QH
        ctx.TILE_KH = TILE_KH
        ctx.TILE_HD = TILE_HD

        return q_in, k_in

    @staticmethod
    def backward(ctx, dq, dk):
        ALIGNED = ctx.ALIGNED
        bsz = ctx.bsz
        seq_len = ctx.seq_len
        n_q_heads = ctx.n_q_heads
        n_k_heads = ctx.n_k_heads
        head_dim = ctx.head_dim
        cos_bs = ctx.cos_bs
        head_dim_half = head_dim // 2
        TILE_QH = ctx.TILE_QH
        TILE_KH = ctx.TILE_KH
        TILE_HD = ctx.TILE_HD

        n_row = bsz * seq_len
        grid = (n_row,)

        if ALIGNED:
            cos_3d, sin_3d = ctx.saved_tensors
            dq_in = dq.contiguous()
            dk_in = dk.contiguous()
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _rope_4d_ct,
                (
                    dq_in,
                    dk_in,
                    cos_3d,
                    sin_3d,
                    int(cos_bs),
                    int(seq_len),
                    float(-1.0),
                    int(TILE_QH),
                    int(TILE_KH),
                    int(TILE_HD),
                ),
            )
            return dq_in.contiguous(), dk_in.contiguous(), None, None
        else:
            cos_4d, sin_4d = ctx.saved_tensors
            dq_t = dq.transpose(1, 2).contiguous()
            dk_t = dk.transpose(1, 2).contiguous()
            dq_5d = dq_t.view(bsz, seq_len, n_q_heads, 2, head_dim_half)
            dk_5d = dk_t.view(bsz, seq_len, n_k_heads, 2, head_dim_half)
            ct.launch(
                torch.cuda.current_stream(),
                grid,
                _rope_5d_ct,
                (
                    dq_5d,
                    dk_5d,
                    cos_4d,
                    sin_4d,
                    int(cos_bs),
                    int(seq_len),
                    float(-1.0),
                    int(TILE_QH),
                    int(TILE_KH),
                    int(TILE_HD),
                ),
            )
            dq_out = dq_t.view(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2).to(ctx.original_dtype)
            dk_out = dk_t.view(bsz, seq_len, n_k_heads, head_dim).transpose(1, 2).to(ctx.original_dtype)
            # Return views directly (no copy needed); autograd handles non-contiguous grads.
            return dq_out, dk_out, None, None


@register_impl("liger.rope", backend="cutile")
def rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    **kwargs,
) -> tuple:
    return RopeCuTileFunction.apply(q, k, cos, sin)
