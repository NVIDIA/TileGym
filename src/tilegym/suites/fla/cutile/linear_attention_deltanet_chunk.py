# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import functools
from types import SimpleNamespace

import cuda.tile as ct
import torch
import torch.nn.functional as F

from tilegym.suites.fla.cutile._host_utils import _segmented_arange
from tilegym.suites.fla.cutile._host_utils import input_guard
from tilegym.suites.fla.cutile._tuning import OUTPUT_HINT_PAIRS
from tilegym.suites.fla.cutile._tuning import hint_configs
from tilegym.suites.fla.cutile._tuning import output_tile_configs
from tilegym.suites.fla.cutile._tuning import tune_launch

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

RCP_LN2 = 1.4426950216  # 1/ln(2)


def _autotuned_launch(kernel, cache_key, grid, args, occ_choices=(1, 2, 4), timeout=30):
    choices = tuple(occ_choices)
    tune_launch(
        kernel,
        torch.cuda.current_stream(),
        lambda c: grid,
        lambda c: args,
        lambda: hint_configs(choices),
        cache_key,
        timeout=timeout,
        config_key=("hints", choices, (4, 8)),
    )


_fwd_o_tile_configs = output_tile_configs


def _autotuned_launch_o(kernel, cache_key, tile_choices, grid_fn, args_fn, timeout=60):
    choices = tuple(tile_choices)
    tune_launch(
        kernel,
        torch.cuda.current_stream(),
        lambda c: grid_fn(c.BK, c.BV),
        lambda c: args_fn(c.BK, c.BV),
        lambda: [
            SimpleNamespace(occupancy=o, num_worker_warps=w, BK=bk, BV=bv)
            for bk, bv in choices
            for o, w in OUTPUT_HINT_PAIRS
        ],
        cache_key,
        timeout=timeout,
        config_key=(("BK", "BV"), choices, OUTPUT_HINT_PAIRS),
    )


def _check_pytorch_version(version_s: str = "2.4") -> bool:
    from packaging import version

    return version.parse(torch.__version__) >= version.parse(version_s)


_device = "cuda"

if _check_pytorch_version("2.4"):
    autocast_custom_fwd = functools.partial(torch.amp.custom_fwd, device_type=_device)
    autocast_custom_bwd = functools.partial(torch.amp.custom_bwd, device_type=_device)
else:
    autocast_custom_fwd = torch.cuda.amp.custom_fwd
    autocast_custom_bwd = torch.cuda.amp.custom_bwd


def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return torch.diff(cu_seqlens)


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    src = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    chunk_counts = (prepare_lens(src) + (chunk_size - 1)).div(chunk_size, rounding_mode="floor")
    seg_id, intra_chunk_idx = _segmented_arange(chunk_counts)
    return torch.stack([seg_id, intra_chunk_idx], 1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    return F.pad(_cdiv(prepare_lens(cu_seqlens), chunk_size), (1, 0), value=0).cumsum(-1)


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _cdiv(a, b):
    if isinstance(a, torch.Tensor):
        return (a + b - 1).div(b, rounding_mode="floor")
    return (a + b - 1) // b


def _ck(a, b):
    """In-kernel ceil-div over compile-time constants (static loop bounds)."""
    return (a + b - 1) // b


# ===========================================================================
# fla.modules.l2norm
# ===========================================================================


@ct.kernel
def l2norm_fwd_kernel(
    x,  # [T, D]
    y,  # [T, D]
    rstd,  # [T]
    eps: float,
    D: ConstInt,
    BD: ConstInt,
    BT: ConstInt,
):
    i_t = ct.bid(0)
    cols = ct.arange(BD, dtype=ct.int32)
    rows = ct.arange(BT, dtype=ct.int32)
    col_mask = cols < D
    # Gather a [BT, BD] tile of rows starting at i_t*BT. ct.gather/scatter index
    # a multi-dim Array with a per-axis index tuple (Arrays have no .reshape),
    # so pass (row, col) indices broadcast to [BT, BD] rather than flat indices.
    row_idx = i_t * BT + rows
    idx2d = (row_idx[:, None], cols[None, :])
    mask = (row_idx[:, None] < x.shape[0]) & col_mask[None, :]
    b_x = ct.astype(ct.gather(x, idx2d, mask=mask, padding_value=0.0), ct.float32)
    b_rstd = 1.0 / ct.sqrt(ct.sum(b_x * b_x, axis=1) + eps)
    b_y = b_x * b_rstd[:, None]
    ct.scatter(y, idx2d, ct.astype(b_y, y.dtype), mask=mask)
    rmask = row_idx < x.shape[0]
    ct.scatter(rstd, row_idx, ct.astype(b_rstd, rstd.dtype), mask=rmask)


@ct.kernel
def l2norm_bwd_kernel(
    y,  # [T, D]
    rstd,  # [T]
    dy,  # [T, D]
    dx,  # [T, D]
    eps: float,
    D: ConstInt,
    BD: ConstInt,
    BT: ConstInt,
):
    i_t = ct.bid(0)
    cols = ct.arange(BD, dtype=ct.int32)
    rows = ct.arange(BT, dtype=ct.int32)
    col_mask = cols < D
    row_idx = i_t * BT + rows
    idx2d = (row_idx[:, None], cols[None, :])
    mask = (row_idx[:, None] < y.shape[0]) & col_mask[None, :]
    rmask = row_idx < y.shape[0]
    b_y = ct.astype(ct.gather(y, idx2d, mask=mask, padding_value=0.0), ct.float32)
    b_rstd = ct.astype(ct.gather(rstd, row_idx, mask=rmask, padding_value=0.0), ct.float32)
    b_dy = ct.astype(ct.gather(dy, idx2d, mask=mask, padding_value=0.0), ct.float32)
    b_dx = b_dy * b_rstd[:, None] - ct.sum(b_dy * b_y, axis=1)[:, None] * b_y * b_rstd[:, None]
    ct.scatter(dx, idx2d, ct.astype(b_dx, dx.dtype), mask=mask)


def l2norm_fwd(x: torch.Tensor, eps: float = 1e-6, output_dtype: torch.dtype | None = None):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if output_dtype is None:
        y = torch.empty_like(x)
    else:
        y = torch.empty_like(x, dtype=output_dtype)
    T, D = x.shape[0], x.shape[-1]
    BD = _next_power_of_2(D)
    rstd = torch.empty((T,), dtype=torch.float32, device=x.device)
    BT = 8
    grid = (_cdiv(T, BT), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        l2norm_fwd_kernel,
        (x, y, rstd, float(eps), D, BD, BT),
    )
    return y.reshape(x_shape_og), rstd.reshape(x_shape_og[:-1])


def l2norm_bwd(y: torch.Tensor, rstd: torch.Tensor, dy: torch.Tensor, eps: float = 1e-6):
    y_shape_og = y.shape
    y = y.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    dx = torch.empty_like(y)
    T, D = y.shape[0], y.shape[-1]
    BD = _next_power_of_2(D)
    rstd = rstd.reshape(-1)
    BT = 8
    grid = (_cdiv(T, BT), 1, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        l2norm_bwd_kernel,
        (y, rstd, dy, dx, float(eps), D, BD, BT),
    )
    return dx.reshape(y_shape_og)


# ===========================================================================
# fla.ops.common.chunk_scaled_dot_kkt
# ===========================================================================


@ct.kernel
def chunk_scaled_dot_kkt_fwd_kernel(
    k,  # [BT_total, H, K]
    g,  # [BT_total, HV]  (dummy when not USE_G)
    beta,  # [BT_total, HV]
    A,  # [BT_total, HV, BT]
    cu_seqlens,  # [num_seq+1] int  (dummy when not IS_VARLEN)
    chunk_indices,  # [NT, 2] int   (dummy when not IS_VARLEN)
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    HVH: ConstInt,  # HV // H
    USE_G: ConstBool,
    IS_VARLEN: ConstBool,
):
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_kvh = i_h // HVH

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
        i_tt = i_t
    seqlen = eos - bos

    k_seg = k.slice(axis=0, start=bos, stop=eos)
    A_seg = A.slice(axis=0, start=bos, stop=eos)
    beta_seg = beta.slice(axis=0, start=bos, stop=eos)

    o_t = i_tt * BT + ct.arange(BT, dtype=ct.int32)
    m_t = o_t < seqlen

    b_b = ct.astype(
        ct.load(beta_seg, index=(i_tt, i_h), shape=(BT, 1), padding_mode=ct.PaddingMode.ZERO).reshape((BT,)), ct.float32
    )

    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(_ck(K, BK)):
        b_k = ct.load(k_seg, index=(i_tt, i_kvh, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_A = ct.mma(b_k, ct.transpose(b_k), b_A)

    if USE_G:
        g_seg = g.slice(axis=0, start=bos, stop=eos)
        b_g = ct.astype(
            ct.load(g_seg, index=(i_tt, i_h), shape=(BT, 1), padding_mode=ct.PaddingMode.ZERO).reshape((BT,)),
            ct.float32,
        )
        b_g_diff = b_g[:, None] - b_g[None, :]
        b_A = b_A * ct.exp2(b_g_diff)
    b_A = b_A * b_b[:, None]

    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = ct.where(m_A, b_A, ct.zeros((BT, BT), dtype=ct.float32))
    ct.store(A_seg, index=(i_tt, i_h, 0), tile=ct.astype(b_A, A.dtype).reshape((BT, 1, BT)))


def chunk_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
    chunk_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute beta * K * K^T of shape `[B, T, HV, BT]`."""
    B, T, H, K, HV = *k.shape, beta.shape[2]
    BT = chunk_size
    BK = min(max(_next_power_of_2(K), 16), 64)
    USE_G = g is not None
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    A = torch.empty(B, T, HV, BT, device=k.device, dtype=output_dtype)

    BT_total = B * T
    k3 = k.reshape(BT_total, H, K)
    beta2 = beta.reshape(BT_total, HV)
    A3 = A.reshape(BT_total, HV, BT)
    g2 = g.reshape(BT_total, HV) if USE_G else torch.empty((1, 1), device=k.device, dtype=k.dtype)
    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = torch.empty((1, 2), device=k.device, dtype=torch.int32)
        cs = torch.empty((1,), device=k.device, dtype=torch.int32)

    grid = (NT, B * HV, 1)
    args = (k3, g2, beta2, A3, cs, ci, T, H, HV, K, BT, BK, HV // H, USE_G, IS_VARLEN)
    cache_key = (
        "chunk_scaled_dot_kkt_fwd_kernel",
        B,
        T,
        H,
        HV,
        K,
        BT,
        BK,
        USE_G,
        IS_VARLEN,
        str(k.dtype),
        str(k.device),
    )
    _autotuned_launch(chunk_scaled_dot_kkt_fwd_kernel, cache_key, grid, args)
    return A


@ct.kernel
def solve_tril_kernel(
    A,  # [BT_total, H, BT]
    Ai,  # [BT_total, H, BT]  (output, pre-zeroed)
    cu_seqlens,  # [num_seq+1] int (dummy when not IS_VARLEN)
    chunk_indices,  # [NT, 2] int   (dummy when not IS_VARLEN)
    T,
    H: ConstInt,
    BT: ConstInt,
    NB: ConstInt,  # BT // 16
    IS_VARLEN: ConstBool,
):
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
        i_tt = i_t

    A_seg = A.slice(axis=0, start=bos, stop=eos)
    Ai_seg = Ai.slice(axis=0, start=bos, stop=eos)

    b_full = ct.astype(
        ct.load(A_seg, index=(i_tt, i_h, 0), shape=(BT, 1, BT), padding_mode=ct.PaddingMode.ZERO).reshape((BT, BT)),
        ct.float32,
    )

    mm = ct.float32
    if NB == 1:
        Ainv = _invert_16_skip1(_get_block16(b_full, 0, 0))
    elif NB == 2:
        Ainv = _invert_32(b_full, 0, 0, mm)
    else:  # NB == 4 (BT=64, the DeltaNet fwd hot path)
        # Two 32x32 diagonal blocks of the 64x64 tile (tile-space 16x16 quads).
        b_Ai00 = _invert_32(b_full, 0, 0, mm)  # rows/cols 0:32 (32-tile (0,0))
        b_Ai11 = _invert_32(b_full, 1, 1, mm)  # rows/cols 32:64 (32-tile (1,1))
        # Off-diagonal 32x32 sub-block A10 = b_full[32:64, 0:32].
        b_A10 = _get_block32(b_full, 1, 0)
        # Merge: Ai10 = -Ai11 @ A10 @ Ai00 (two 32x32 fp32/ieee matmuls). Keep
        # the fp32 result so it cats with the fp32 diag blocks.
        b_Ai10 = ct.astype(
            -ct.matmul(
                ct.astype(ct.matmul(ct.astype(b_Ai11, mm), ct.astype(b_A10, mm)), mm),
                ct.astype(b_Ai00, mm),
            ),
            ct.float32,
        )
        z32 = ct.zeros((32, 32), dtype=ct.float32)
        Ainv = ct.cat((ct.cat((b_Ai00, z32), 1), ct.cat((b_Ai10, b_Ai11), 1)), 0)

    ct.store(Ai_seg, index=(i_tt, i_h, 0), tile=ct.astype(Ainv, Ai.dtype).reshape((BT, 1, BT)))


def _get_block16(mat, ib, jb):
    """Extract the (ib, jb) 16x16 sub-tile from a [BT, BT] tile (tile-space index)."""
    return ct.extract(mat, index=(ib, jb), shape=(16, 16))


def _get_block32(mat, ib, jb):
    """Extract the (ib, jb) 32x32 sub-tile from a [BT, BT] tile (tile-space index)."""
    return ct.extract(mat, index=(ib, jb), shape=(32, 32))


def _invert_16_skip1(b_blk):
    """Inverse of (I + strict_lower(b_blk)) for a 16x16 fp32 tile.

    Column-reduction forward substitution (mirrors GDN
    ``_solve_kkt_16_block_full_skip1``): ``b_Ai`` starts as
    ``-strict_lower(b_blk)`` (the upper triangle of ``b_blk`` is structurally
    zero in this solve, so an unconditional negate is exact) and each row
    i = 1..15 is corrected by the already-solved rows above it. Each row uses ct.sum reductions over the resident 16-wide tile.
    """
    o = ct.arange(16, dtype=ct.int32)
    m_I = o[:, None] == o[None, :]
    b_Ai = -b_blk
    for i in range(1, 16):
        is_row = o == i
        b_a = ct.sum(ct.where(is_row[:, None], b_Ai, ct.zeros((16, 16), dtype=ct.float32)), axis=0)
        b_a = ct.sum(b_a[:, None] * b_Ai, axis=0)
        b_Ai = b_Ai + ct.where(is_row[:, None], b_a[None, :], ct.zeros((16, 16), dtype=ct.float32))
    return b_Ai + ct.astype(m_I, ct.float32)


def _invert_32(b_full, ib, jb, dtype):
    """Inverse of the 32x32 diagonal block at tile-space (ib, jb) of ``b_full``.

    Splits the 32x32 block into 16x16 quadrants; inverts the two 16x16 diagonal
    sub-blocks and recovers the off-diagonal via Ai_10 = -Ai_11 @ A_10 @ Ai_00
    (16x16 fp32/ieee matmuls). Returns a fp32 [32, 32] tile.
    """
    b00 = _get_block16(b_full, 2 * ib, 2 * jb)
    b10 = _get_block16(b_full, 2 * ib + 1, 2 * jb)
    b11 = _get_block16(b_full, 2 * ib + 1, 2 * jb + 1)
    b_Ai00 = _invert_16_skip1(b00)
    b_Ai11 = _invert_16_skip1(b11)
    # Keep the fp32 result so it cats with the fp32 blocks.
    b_Ai10 = ct.astype(
        -ct.matmul(
            ct.astype(ct.matmul(ct.astype(b_Ai11, dtype), ct.astype(b10, dtype)), dtype),
            ct.astype(b_Ai00, dtype),
        ),
        ct.float32,
    )
    z16 = ct.zeros((16, 16), dtype=ct.float32)
    return ct.cat((ct.cat((b_Ai00, z16), 1), ct.cat((b_Ai10, b_Ai11), 1)), 0)


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """Compute the inverse of the matrix I + A (A strictly lower triangular)."""
    assert A.shape[-1] in [16, 32, 64]
    output_dtype = A.dtype if output_dtype is None else output_dtype
    B, T, H, BT = A.shape
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = len(chunk_indices) if cu_seqlens is not None else _cdiv(T, BT)

    Ai = torch.zeros_like(A, dtype=output_dtype)

    BT_total = B * T
    A3 = A.reshape(BT_total, H, BT)
    Ai3 = Ai.reshape(BT_total, H, BT)
    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = torch.empty((1, 2), device=A.device, dtype=torch.int32)
        cs = torch.empty((1,), device=A.device, dtype=torch.int32)

    grid = (NT, B * H, 1)
    args = (A3, Ai3, cs, ci, T, H, BT, BT // 16, IS_VARLEN)
    cache_key = ("solve_tril_kernel", B, T, H, BT, IS_VARLEN, str(A.dtype), str(A.device))
    _autotuned_launch(solve_tril_kernel, cache_key, grid, args)
    return Ai


# ===========================================================================
# fla.ops.delta_rule.wy_fast
# ===========================================================================


@ct.kernel
def recompute_w_u_fwd_kernel(
    k,  # [BT_total, H, K]
    v,  # [BT_total, H, V]
    beta,  # [BT_total, H]
    w,  # [BT_total, H, K]
    u,  # [BT_total, H, V]
    A,  # [BT_total, H, BT]
    cu_seqlens,  # dummy when not IS_VARLEN
    chunk_indices,  # dummy when not IS_VARLEN
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    IS_VARLEN: ConstBool,
):
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
        i_tt = i_t

    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    w_seg = w.slice(axis=0, start=bos, stop=eos)
    u_seg = u.slice(axis=0, start=bos, stop=eos)
    A_seg = A.slice(axis=0, start=bos, stop=eos)
    beta_seg = beta.slice(axis=0, start=bos, stop=eos)

    b_b = ct.astype(
        ct.load(beta_seg, index=(i_tt, i_h), shape=(BT, 1), padding_mode=ct.PaddingMode.ZERO).reshape((BT,)), ct.float32
    )
    b_A = ct.load(A_seg, index=(i_tt, i_h, 0), shape=(BT, 1, BT), padding_mode=ct.PaddingMode.ZERO).reshape((BT, BT))

    for i_v in range(_ck(V, BV)):
        b_v = ct.load(v_seg, index=(i_tt, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )
        b_vb = ct.astype(ct.astype(b_v, ct.float32) * b_b[:, None], b_v.dtype)
        acc = ct.zeros((BT, BV), dtype=ct.float32)
        b_u = ct.mma(ct.astype(b_A, b_vb.dtype), b_vb, acc)
        ct.store(u_seg, index=(i_tt, i_h, i_v), tile=ct.astype(b_u, u.dtype).reshape((BT, 1, BV)))

    for i_k in range(_ck(K, BK)):
        b_k = ct.load(k_seg, index=(i_tt, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_kb = ct.astype(ct.astype(b_k, ct.float32) * b_b[:, None], b_k.dtype)
        acc = ct.zeros((BT, BK), dtype=ct.float32)
        b_w = ct.mma(ct.astype(b_A, b_kb.dtype), b_kb, acc)
        ct.store(w_seg, index=(i_tt, i_h, i_k), tile=ct.astype(b_w, w.dtype).reshape((BT, 1, BK)))


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = 64
    BK = min(max(_next_power_of_2(K), 16), 64)
    BV = min(max(_next_power_of_2(V), 16), 64)
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    u = torch.empty_like(v)
    w = torch.empty_like(k)

    BT_total = B * T
    k3 = k.reshape(BT_total, H, K)
    v3 = v.reshape(BT_total, H, V)
    beta2 = beta.reshape(BT_total, H)
    w3 = w.reshape(BT_total, H, K)
    u3 = u.reshape(BT_total, H, V)
    A3 = A.reshape(BT_total, H, BT)
    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = torch.empty((1, 2), device=k.device, dtype=torch.int32)
        cs = torch.empty((1,), device=k.device, dtype=torch.int32)

    grid = (NT, B * H, 1)
    args = (k3, v3, beta2, w3, u3, A3, cs, ci, T, H, K, V, BT, BK, BV, IS_VARLEN)
    cache_key = ("recompute_w_u_fwd_kernel", B, T, H, K, V, BT, BK, BV, IS_VARLEN, str(k.dtype), str(k.device))
    _autotuned_launch(recompute_w_u_fwd_kernel, cache_key, grid, args)
    return w, u


@ct.kernel
def prepare_wy_repr_bwd_kernel(
    k,  # [BT_total, H, K]
    v,  # [BT_total, H, V]
    beta,  # [BT_total, H]
    A,  # [BT_total, H, BT]
    dw,  # [BT_total, H, K]
    du,  # [BT_total, H, V]
    dk,  # [BT_total, H, K]  (output)
    dv,  # [BT_total, H, V]  (output)
    dbeta,  # [BT_total, H]  (output)
    cu_seqlens,
    chunk_indices,
    T,
    H: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    IS_VARLEN: ConstBool,
):
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // H
    i_h = i_bh % H

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
        i_tt = i_t
    seqlen = eos - bos

    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    A_seg = A.slice(axis=0, start=bos, stop=eos)
    dw_seg = dw.slice(axis=0, start=bos, stop=eos)
    du_seg = du.slice(axis=0, start=bos, stop=eos)
    dk_seg = dk.slice(axis=0, start=bos, stop=eos)
    dv_seg = dv.slice(axis=0, start=bos, stop=eos)
    beta_seg = beta.slice(axis=0, start=bos, stop=eos)
    dbeta_seg = dbeta.slice(axis=0, start=bos, stop=eos)

    storage_dtype = k.dtype
    b_b = ct.astype(
        ct.load(beta_seg, index=(i_tt, i_h), shape=(BT, 1), padding_mode=ct.PaddingMode.ZERO).reshape((BT,)), ct.float32
    )
    b_A = ct.transpose(
        ct.load(A_seg, index=(i_tt, i_h, 0), shape=(BT, 1, BT), padding_mode=ct.PaddingMode.ZERO).reshape((BT, BT))
    )

    b_dbeta = ct.zeros((BT,), dtype=ct.float32)
    b_dA = ct.zeros((BT, BT), dtype=ct.float32)

    for i_v in ct.static_iter(range(_ck(V, BV))):
        b_v = ct.load(v_seg, index=(i_tt, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )
        b_du = ct.load(du_seg, index=(i_tt, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )
        b_vf = ct.astype(b_v, ct.float32)
        b_v_beta = ct.astype(b_vf * b_b[:, None], b_v.dtype)
        b_dA = ct.mma(b_du, ct.transpose(b_v_beta), b_dA)
        acc_dv = ct.zeros((BT, BV), dtype=ct.float32)
        b_dv_beta = ct.mma(ct.astype(b_A, b_du.dtype), b_du, acc_dv)
        b_dv = b_dv_beta * b_b[:, None]
        b_dbeta = b_dbeta + ct.sum(b_dv_beta * b_vf, axis=1)
        ct.store(dv_seg, index=(i_tt, i_h, i_v), tile=ct.astype(b_dv, dv.dtype).reshape((BT, 1, BV)))

    for i_k in ct.static_iter(range(_ck(K, BK))):
        b_k = ct.load(k_seg, index=(i_tt, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_dw = ct.load(dw_seg, index=(i_tt, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_kf = ct.astype(b_k, ct.float32)
        b_k_beta = ct.astype(b_kf * b_b[:, None], b_k.dtype)
        b_dA = ct.mma(b_dw, ct.transpose(b_k_beta), b_dA)
        acc_dk = ct.zeros((BT, BK), dtype=ct.float32)
        b_dk_beta = ct.mma(ct.astype(b_A, b_dw.dtype), b_dw, acc_dk)
        b_dk = b_dk_beta * b_b[:, None]
        b_dbeta = b_dbeta + ct.sum(b_dk_beta * b_kf, axis=1)
        ct.store(dk_seg, index=(i_tt, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((BT, 1, BK)))

    o_t = ct.arange(BT, dtype=ct.int32)
    tri = o_t[:, None] > o_t[None, :]
    b_dA = ct.where(tri, b_dA, ct.zeros((BT, BT), dtype=ct.float32))
    acc_tt = ct.zeros((BT, BT), dtype=ct.float32)
    b_dA = ct.mma(ct.astype(b_dA, storage_dtype), ct.astype(b_A, storage_dtype), acc_tt)
    acc_tt = ct.zeros((BT, BT), dtype=ct.float32)
    b_dA = ct.mma(ct.astype(b_A, storage_dtype), ct.astype(b_dA, storage_dtype), acc_tt)
    b_dA = ct.astype(ct.where(tri, -b_dA, ct.zeros((BT, BT), dtype=ct.float32)), storage_dtype)

    for i_k in ct.static_iter(range(_ck(K, BK))):
        b_k = ct.load(k_seg, index=(i_tt, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_dk_prev = ct.astype(
            ct.load(dk_seg, index=(i_tt, i_h, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BK)
            ),
            ct.float32,
        )
        b_k_beta = ct.astype(ct.astype(b_k, ct.float32) * b_b[:, None], b_k.dtype)
        acc_dkb = ct.zeros((BT, BK), dtype=ct.float32)
        b_dk_beta = ct.mma(b_dA, b_k, acc_dkb)
        b_dbeta = b_dbeta + ct.sum(b_dk_beta * ct.astype(b_k, ct.float32), axis=1)
        # b_dk += trans(b_dA) @ b_k_beta; mma(trans(b_dA), b_k_beta) -> [BT, BK]
        acc_dk2 = ct.zeros((BT, BK), dtype=ct.float32)
        b_dk = b_dk_prev + ct.mma(ct.transpose(b_dA), b_k_beta, acc_dk2)
        b_dk = b_dk + b_dk_beta * b_b[:, None]
        ct.store(dk_seg, index=(i_tt, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((BT, 1, BK)))

    ct.store(dbeta_seg, index=(i_tt, i_h), tile=ct.astype(b_dbeta, dbeta.dtype).reshape((BT, 1)))


def prepare_wy_repr_bwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    dw: torch.Tensor,
    du: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, H, K, V = *k.shape, v.shape[-1]
    BT = A.shape[-1]
    BK = min(max(_next_power_of_2(K), 16), 64)
    BV = min(max(_next_power_of_2(V), 16), 64)
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    dbeta = torch.empty_like(beta)

    BT_total = B * T
    k3 = k.reshape(BT_total, H, K)
    v3 = v.reshape(BT_total, H, V)
    beta2 = beta.reshape(BT_total, H)
    A3 = A.reshape(BT_total, H, BT)
    dw3 = dw.reshape(BT_total, H, K)
    du3 = du.reshape(BT_total, H, V)
    dk3 = dk.reshape(BT_total, H, K)
    dv3 = dv.reshape(BT_total, H, V)
    db2 = dbeta.reshape(BT_total, H)
    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = torch.empty((1, 2), device=k.device, dtype=torch.int32)
        cs = torch.empty((1,), device=k.device, dtype=torch.int32)

    grid = (NT, B * H, 1)
    args = (k3, v3, beta2, A3, dw3, du3, dk3, dv3, db2, cs, ci, T, H, K, V, BT, BK, BV, IS_VARLEN)
    cache_key = (
        "prepare_wy_repr_bwd_kernel",
        B,
        T,
        H,
        K,
        V,
        BT,
        BK,
        BV,
        IS_VARLEN,
        str(k.dtype),
        str(k.device),
    )
    _autotuned_launch(prepare_wy_repr_bwd_kernel, cache_key, grid, args)
    return dk, dv, dbeta


def prepare_wy_repr_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, cu_seqlens=cu_seqlens, chunk_size=64, output_dtype=torch.float32, chunk_indices=chunk_indices
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    return w, u, A


@ct.kernel
def chunk_gated_delta_rule_fwd_kernel_h(
    k,  # [BT_total, H, K]
    v,  # [BT_total, HV, V]   (= u)
    w,  # [BT_total, HV, K]
    v_new,  # [BT_total, HV, V] (output)
    h,  # [B*NT, HV, K, V]  (output)
    h0,  # [N, HV, K, V]     (dummy when not USE_INITIAL_STATE)
    ht,  # [N, HV, K, V]     (dummy when not STORE_FINAL_STATE)
    cu_seqlens,
    chunk_offsets,  # [N] int  (dummy when not IS_VARLEN)
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BV: ConstInt,
    BK: ConstInt,  # next_pow2(K) -- full-width state/K tile (general over K)
    HVH: ConstInt,  # HV // H
    NT_FIXED: ConstInt,  # NT for the fixed-length case (static loop upper bound)
    USE_INITIAL_STATE: ConstBool,
    STORE_FINAL_STATE: ConstBool,
    SAVE_NEW_VALUE: ConstBool,
    IS_VARLEN: ConstBool,
):
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // HV
    i_h = i_nh % HV
    i_kvh = i_h // HVH

    if IS_VARLEN:
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
        boh = ct.load(chunk_offsets, index=(i_n,), shape=()).astype(ct.int32)
        seqlen = eos - bos
    else:
        bos = i_n * T
        eos = bos + T
        seqlen = T
        boh = i_n * NT_FIXED
    NT = _ck_rt(seqlen, BT)

    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    w_seg = w.slice(axis=0, start=bos, stop=eos)
    vnew_seg = v_new.slice(axis=0, start=bos, stop=eos)

    # Running KV state carried as a SINGLE full-width (BK, BV) tile (BK =
    # next_pow2(K)), general for ANY K (no 64-wide sub-block unroll / K<=256 cap).
    # h0 is [N, HV, BK, V]; rows [K:BK] are zero-padded on load and stay 0.
    b_h1 = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_INITIAL_STATE:
        b_h1 = ct.astype(
            ct.load(h0, index=(i_n, i_h, 0, i_v), shape=(1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BK, BV)
            ),
            ct.float32,
        )

    # Static loop upper bound NT_FIXED; runtime `if i_t < NT` skips the varlen tail.
    for i_t in range(NT_FIXED):
        if i_t < NT:
            # store current state h[i_t] (single full-width [BK, BV] tile; the
            # block-indexed store masks rows [K:BK] for non-pow2 K)
            ct.store(h, index=(boh + i_t, i_h, 0, i_v), tile=ct.astype(b_h1, h.dtype).reshape((1, 1, BK, BV)))

            # v_new = v - w @ h        (single full-width w[BT,BK] @ h[BK,BV] MMA;
            # padded w columns / h rows [K:BK] are zero -> contribute 0)
            acc = ct.zeros((BT, BV), dtype=ct.float32)
            b_w1 = ct.load(w_seg, index=(i_t, i_h, 0), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BK)
            )
            b_wh = ct.mma(b_w1, ct.astype(b_h1, b_w1.dtype), acc)

            b_v = ct.load(v_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BV)
            )
            b_v = ct.astype(b_v, ct.float32) - b_wh
            b_v_store = ct.astype(b_v, k.dtype)
            if SAVE_NEW_VALUE:
                ct.store(vnew_seg, index=(i_t, i_h, i_v), tile=ct.astype(b_v_store, v_new.dtype).reshape((BT, 1, BV)))
            b_k1 = ct.load(k_seg, index=(i_t, i_kvh, 0), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BK)
            )
            acc_h1 = ct.zeros((BK, BV), dtype=ct.float32)
            b_h1 = b_h1 + ct.mma(ct.transpose(b_k1), b_v_store, acc_h1)

    if STORE_FINAL_STATE:
        ct.store(ht, index=(i_n, i_h, 0, i_v), tile=ct.astype(b_h1, ht.dtype).reshape((1, 1, BK, BV)))


def _ck_rt(a, b):
    """Runtime ceil-div used inside kernels (a is a runtime tile/scalar)."""
    return (a + b - 1) // b


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    assert not state_v_first, "cuTile port supports STATE_V_FIRST=False (DeltaNet path)."
    assert g is None and gk is None, "cuTile port supports the ungated DeltaNet path."
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, _cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    BV = min(max(_next_power_of_2(V), 16), 64)
    # Full-width state/K tile: general for ANY K (no 64-wide sub-block unroll).
    BK = _next_power_of_2(K)
    # Pad K and V dimensions to tile-aligned sizes so the kernel's fixed-size
    # (BK, BV) loads/stores never exceed buffer bounds (ct.store has no
    # padding_mode, unlike ct.load).
    K_pad = BK  # e.g. K=32 -> 32, K=48 -> 64, K=96 -> 128, K=192 -> 256, K=320 -> 512
    V_pad = BV * _cdiv(V, BV)  # e.g. V=48, BV=64 -> 64; V=64, BV=64 -> 64

    h = k.new_empty(B * NT, HV, K_pad, V_pad)
    final_state = k.new_zeros(N, HV, K_pad, V_pad, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    BT_total = B * T
    k3 = k.reshape(BT_total, H, K)
    u3 = u.reshape(BT_total, HV, V)
    w3 = w.reshape(BT_total, HV, K)
    USE_INIT = initial_state is not None
    if USE_INIT:
        # Pad initial_state from [N, HV, K, V] to [N, HV, K_pad, V_pad]
        h0 = k.new_zeros(N, HV, K_pad, V_pad, dtype=torch.float32)
        h0[:, :, :K, :V] = initial_state
    else:
        h0 = torch.empty((1, 1, K_pad, BV), device=k.device, dtype=k.dtype)
    ht = final_state if output_final_state else torch.empty((1, 1, K_pad, BV), device=k.device, dtype=torch.float32)
    vnew3 = v_new.reshape(BT_total, HV, V) if save_new_value else torch.empty((1, 1, 1), device=k.device, dtype=k.dtype)
    if IS_VARLEN:
        co = chunk_offsets.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        co = torch.empty((1,), device=k.device, dtype=torch.int32)
        cs = torch.empty((1,), device=k.device, dtype=torch.int32)

    grid = (_cdiv(V, BV), N * HV, 1)
    args = (
        k3,
        u3,
        w3,
        vnew3,
        h,
        h0,
        ht,
        cs,
        co,
        T,
        H,
        HV,
        K,
        V,
        BT,
        BV,
        BK,
        HV // H,
        NT,
        USE_INIT,
        bool(output_final_state),
        bool(save_new_value),
        IS_VARLEN,
    )
    cache_key = (
        "chunk_gated_delta_rule_fwd_kernel_h",
        N,
        T,
        H,
        HV,
        K,
        V,
        BT,
        BV,
        BK,
        USE_INIT,
        bool(output_final_state),
        bool(save_new_value),
        IS_VARLEN,
        str(k.dtype),
        str(k.device),
    )
    _autotuned_launch(chunk_gated_delta_rule_fwd_kernel_h, cache_key, grid, args)
    # Slice padded buffers back to the logical [K, V] shape for downstream consumers.
    h = h[:, :, :K, :V].contiguous()
    if final_state is not None:
        final_state = final_state[:, :, :K, :V].contiguous()
    return h, v_new, final_state


@ct.kernel
def chunk_gated_delta_rule_bwd_kernel_dhu(
    q,  # [BT_total, H, K]
    k,  # [BT_total, H, K]
    w,  # [BT_total, HV, K]
    dht,  # [N, HV, K, V] (dummy when not USE_FINAL_STATE_GRADIENT)
    dh0,  # [N, HV, K, V] (dummy when not USE_INITIAL_STATE)
    do,  # [BT_total, HV, V]
    dh,  # [B*NT, HV, K, V] (output)
    dv,  # [BT_total, HV, V]
    dv2,  # [BT_total, HV, V] (output)
    cu_seqlens,
    chunk_offsets,
    scale: float,
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BV: ConstInt,
    BK: ConstInt,  # next_pow2(K) -- full-width state/K tile (general over K)
    HVH: ConstInt,
    NT_FIXED: ConstInt,
    USE_INITIAL_STATE: ConstBool,
    USE_FINAL_STATE_GRADIENT: ConstBool,
    IS_VARLEN: ConstBool,
):
    i_v = ct.bid(0)
    i_nh = ct.bid(1)
    i_n = i_nh // HV
    i_h = i_nh % HV
    i_kvh = i_h // HVH

    if IS_VARLEN:
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
        boh = ct.load(chunk_offsets, index=(i_n,), shape=()).astype(ct.int32)
        seqlen = eos - bos
    else:
        bos = i_n * T
        eos = bos + T
        seqlen = T
        boh = i_n * NT_FIXED
    NT = _ck_rt(seqlen, BT)

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    w_seg = w.slice(axis=0, start=bos, stop=eos)
    do_seg = do.slice(axis=0, start=bos, stop=eos)
    dv_seg = dv.slice(axis=0, start=bos, stop=eos)
    dv2_seg = dv2.slice(axis=0, start=bos, stop=eos)

    # State gradient carried as a SINGLE full-width (BK, BV) tile (BK =
    # next_pow2(K)), general for ANY K (no 64-wide sub-block unroll / K<=256 cap).
    # dht is [N, HV, BK, V]; rows [K:BK] are zero-padded on load and stay 0.
    b_dh1 = ct.zeros((BK, BV), dtype=ct.float32)
    if USE_FINAL_STATE_GRADIENT:
        b_dh1 = b_dh1 + ct.astype(
            ct.load(dht, index=(i_n, i_h, 0, i_v), shape=(1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BK, BV)
            ),
            ct.float32,
        )

    for j in range(NT_FIXED):
        i_t = NT_FIXED - 1 - j
        if i_t < NT:
            ct.store(dh, index=(boh + i_t, i_h, 0, i_v), tile=ct.astype(b_dh1, dh.dtype).reshape((1, 1, BK, BV)))

            # b_dv = k @ dh           (single full-width k[BT,BK] @ dh[BK,BV] MMA;
            # padded k columns / dh rows [K:BK] are zero -> contribute 0)
            acc = ct.zeros((BT, BV), dtype=ct.float32)
            b_k1 = ct.load(k_seg, index=(i_t, i_kvh, 0), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BK)
            )
            b_dv = ct.mma(b_k1, ct.astype(b_dh1, b_k1.dtype), acc)

            b_do = ct.load(do_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BV)
            )
            b_dv = b_dv + ct.astype(
                ct.load(dv_seg, index=(i_t, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                    (BT, BV)
                ),
                ct.float32,
            )
            b_dv_store = ct.astype(b_dv, k.dtype)
            ct.store(dv2_seg, index=(i_t, i_h, i_v), tile=ct.astype(b_dv_store, dv2.dtype).reshape((BT, 1, BV)))
            b_q1 = ct.load(q_seg, index=(i_t, i_kvh, 0), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BK)
            )
            b_w1 = ct.load(w_seg, index=(i_t, i_h, 0), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BK)
            )
            b_qdo = ct.mma(ct.transpose(b_q1), b_do, ct.zeros((BK, BV), dtype=ct.float32))
            b_wdv = ct.mma(ct.transpose(b_w1), b_dv_store, ct.zeros((BK, BV), dtype=ct.float32))
            b_dh1 = b_dh1 + b_qdo * scale - b_wdv

    if USE_INITIAL_STATE:
        ct.store(dh0, index=(i_n, i_h, 0, i_v), tile=ct.astype(b_dh1, dh0.dtype).reshape((1, 1, BK, BV)))


def chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert not state_v_first and g is None and gk is None
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, _cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT)

    BV = min(max(_next_power_of_2(V), 16), 128)
    # Full-width state/K tile: general for ANY K (same as fwd kernel).
    BK = _next_power_of_2(K)
    # Pad K and V dimensions to tile-aligned sizes (same as fwd kernel).
    K_pad = BK
    V_pad = BV * _cdiv(V, BV)

    dh = q.new_empty(B * NT, HV, K_pad, V_pad)
    dv2 = torch.empty_like(dv)

    BT_total = B * T
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    w3 = w.reshape(BT_total, HV, K)
    do3 = do.reshape(BT_total, HV, V)
    dv3 = dv.reshape(BT_total, HV, V)
    dv2_3 = dv2.reshape(BT_total, HV, V)
    USE_INIT = h0 is not None
    USE_DHT = dht is not None
    if USE_INIT:
        dh0_padded = q.new_zeros(N, HV, K_pad, V_pad, dtype=torch.float32)
    else:
        dh0_padded = torch.empty((1, 1, K_pad, BV), device=q.device, dtype=torch.float32)
    if USE_DHT:
        # Pad dht from [N, HV, K, V] to [N, HV, K_pad, V_pad]
        dht_padded = q.new_zeros(N, HV, K_pad, V_pad, dtype=dht.dtype)
        dht_padded[:, :, :K, :V] = dht
    else:
        dht_padded = torch.empty((1, 1, K_pad, BV), device=q.device, dtype=q.dtype)
    if IS_VARLEN:
        co = chunk_offsets.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        co = torch.empty((1,), device=q.device, dtype=torch.int32)
        cs = torch.empty((1,), device=q.device, dtype=torch.int32)

    grid = (_cdiv(V, BV), N * HV, 1)
    args = (
        q3,
        k3,
        w3,
        dht_padded,
        dh0_padded,
        do3,
        dh,
        dv3,
        dv2_3,
        cs,
        co,
        float(scale),
        T,
        H,
        HV,
        K,
        V,
        BT,
        BV,
        BK,
        HV // H,
        NT,
        USE_INIT,
        USE_DHT,
        IS_VARLEN,
    )
    cache_key = (
        "chunk_gated_delta_rule_bwd_kernel_dhu",
        B,
        T,
        H,
        HV,
        K,
        V,
        BT,
        BV,
        BK,
        USE_INIT,
        USE_DHT,
        IS_VARLEN,
        str(q.dtype),
        str(q.device),
    )
    _autotuned_launch(chunk_gated_delta_rule_bwd_kernel_dhu, cache_key, grid, args)
    # Slice padded buffers back to logical [K, V] shape.
    dh = dh[:, :, :K, :V].contiguous()
    dh0 = dh0_padded[:, :, :K, :V].contiguous() if USE_INIT else None
    return dh, dh0, dv2


# ===========================================================================
# fla.ops.common.chunk_o
# ===========================================================================
#
# DeltaNet path uses g=g_gamma=None and STATE_V_FIRST=False, so the gated
# branches collapse. h is [B*NT, HV, K, V].


@ct.kernel
def chunk_fwd_kernel_o(
    q,  # [BT_total, H, K]
    k,  # [BT_total, H, K]
    v,  # [BT_total, HV, V]
    h,  # [B*NT, HV, K, V]
    o,  # [BT_total, HV, V] (output)
    cu_seqlens,
    chunk_indices,
    scale: float,
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    HVH: ConstInt,
    NT_FIXED: ConstInt,
    IS_VARLEN: ConstBool,
):
    i_v = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_kvh = i_h // HVH

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
        seqlen = eos - bos
        NT = _ck_rt(seqlen, BT)
        # Chunks are stored consecutively across sequences.
        h_row = i_t
    else:
        bos = i_b * T
        eos = bos + T
        seqlen = T
        NT = _ck_rt(seqlen, BT)
        i_tt = i_t
        h_row = i_b * NT_FIXED + i_t

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    o_seg = o.slice(axis=0, start=bos, stop=eos)

    b_o = ct.zeros((BT, BV), dtype=ct.float32)
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(_ck(K, BK)):
        b_q = ct.load(
            q_seg, index=(i_tt, i_kvh, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO, latency=3
        ).reshape((BT, BK))
        b_k = ct.load(
            k_seg, index=(i_tt, i_kvh, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO, latency=3
        ).reshape((BT, BK))
        # b_h block: [BK, BV] from h[h_row, i_h, i_k*BK:(i_k+1)*BK, i_v*BV:...]
        b_h = ct.load(
            h, index=(h_row, i_h, i_k, i_v), shape=(1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO, latency=3
        ).reshape((BK, BV))
        acc_o = b_o
        b_o = ct.mma(b_q, b_h, acc_o)
        acc_A = b_A
        b_A = ct.mma(b_q, ct.transpose(b_k), acc_A)

    o_t = i_tt * BT + ct.arange(BT, dtype=ct.int32)
    m_t = o_t < seqlen
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = ct.where(m_A, b_A, ct.zeros((BT, BT), dtype=ct.float32))

    b_v = ct.load(v_seg, index=(i_tt, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape((BT, BV))
    acc_av = ct.zeros((BT, BV), dtype=ct.float32)
    b_av = ct.mma(ct.astype(b_A, b_v.dtype), b_v, acc_av)
    b_o = b_o * scale + b_av * scale
    ct.store(o_seg, index=(i_tt, i_h, i_v), tile=ct.astype(b_o, o.dtype).reshape((BT, 1, BV)))


def chunk_fwd_o(
    q,
    k,
    v,
    h,
    g=None,
    g_gamma=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    assert not state_v_first and g is None and g_gamma is None
    B, T, H, K, V, HV = *q.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    BT_total = B * T
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    v3 = v.reshape(BT_total, HV, V)
    o3 = o.reshape(BT_total, HV, V)
    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = torch.empty((1, 2), device=q.device, dtype=torch.int32)
        cs = torch.empty((1,), device=q.device, dtype=torch.int32)

    NT_fixed = _cdiv(T, BT) if cu_seqlens is None else NT

    # The first candidate is the padded baseline; the remaining candidates
    # cover square and asymmetric tiles.
    tile_choices = _fwd_o_tile_configs(K, V)

    def _grid(bk, bv):
        return (_cdiv(V, bv), NT, B * HV)

    def _args(bk, bv):
        return (q3, k3, v3, h, o3, cs, ci, float(scale), T, H, HV, K, V, BT, bk, bv, HV // H, NT_fixed, IS_VARLEN)

    cache_key = ("chunk_fwd_kernel_o", B, NT, H, HV, K, V, BT, IS_VARLEN, str(q.dtype), str(q.device))
    _autotuned_launch_o(chunk_fwd_kernel_o, cache_key, tile_choices, _grid, _args)
    return o


@ct.kernel
def chunk_bwd_kernel_dv_local(
    q,  # [BT_total, H, K]
    k,  # [BT_total, H, K]
    do,  # [BT_total, HV, V]
    dv,  # [BT_total, HV, V] (output)
    cu_seqlens,
    chunk_indices,
    scale: float,
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    HVH: ConstInt,
    IS_VARLEN: ConstBool,
):
    i_t = ct.bid(0)
    i_bh = ct.bid(1)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_kvh = i_h // HVH

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
    else:
        bos = i_b * T
        eos = bos + T
        i_tt = i_t
    seqlen = eos - bos

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    do_seg = do.slice(axis=0, start=bos, stop=eos)
    dv_seg = dv.slice(axis=0, start=bos, stop=eos)

    # recompute path (A=None, g=None): b_A = sum_k (k @ q^T) * scale
    b_A = ct.zeros((BT, BT), dtype=ct.float32)
    for i_k in range(_ck(K, BK)):
        b_q = ct.load(q_seg, index=(i_tt, i_kvh, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        b_k = ct.load(k_seg, index=(i_tt, i_kvh, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BK)
        )
        acc = b_A
        b_A = ct.mma(b_k, ct.transpose(b_q), acc)
    b_A = b_A * scale

    o_t = i_tt * BT + ct.arange(BT, dtype=ct.int32)
    m_t = o_t < seqlen
    m_A = (o_t[:, None] <= o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_A = ct.astype(ct.where(m_A, b_A, ct.zeros((BT, BT), dtype=ct.float32)), do.dtype)

    for i_v in range(_ck(V, BV)):
        b_do = ct.load(do_seg, index=(i_tt, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )
        acc = ct.zeros((BT, BV), dtype=ct.float32)
        b_dv = ct.mma(b_A, b_do, acc)
        ct.store(dv_seg, index=(i_tt, i_h, i_v), tile=ct.astype(b_dv, dv.dtype).reshape((BT, 1, BV)))


def chunk_bwd_dv_local(
    q,
    k,
    do,
    g=None,
    g_gamma=None,
    A=None,
    scale=None,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    assert A is None and g is None and g_gamma is None, "cuTile port supports the recompute DeltaNet path."
    B, T, H, K, V, HV = *k.shape, do.shape[-1], do.shape[2]
    BT = chunk_size
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    BK = min(max(_next_power_of_2(K), 16), 64)
    BV = min(max(_next_power_of_2(V), 16), 64)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    dv = torch.empty_like(do)
    BT_total = B * T
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    do3 = do.reshape(BT_total, HV, V)
    dv3 = dv.reshape(BT_total, HV, V)
    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = torch.empty((1, 2), device=q.device, dtype=torch.int32)
        cs = torch.empty((1,), device=q.device, dtype=torch.int32)

    grid = (NT, B * HV, 1)
    args = (q3, k3, do3, dv3, cs, ci, float(scale), T, H, HV, K, V, BT, BK, BV, HV // H, IS_VARLEN)
    cache_key = (
        "chunk_bwd_kernel_dv_local",
        B,
        T,
        H,
        HV,
        K,
        V,
        BT,
        BK,
        BV,
        IS_VARLEN,
        str(q.dtype),
        str(q.device),
    )
    _autotuned_launch(chunk_bwd_kernel_dv_local, cache_key, grid, args)
    return dv


@ct.kernel
def chunk_bwd_kernel_dqkwg(
    q,  # [BT_total, H, K]
    k,  # [BT_total, H, K]
    v,  # [BT_total, HV, V]
    h,  # [B*NT, HV, K, V]
    do,  # [BT_total, HV, V]
    dh,  # [B*NT, HV, K, V]
    dq,  # [BT_total, HV, K] (output)
    dk,  # [BT_total, HV, K] (output)
    dw,  # [BT_total, HV, K] (output)
    dv,  # [BT_total, HV, V]
    cu_seqlens,
    chunk_indices,
    scale: float,
    T,
    H: ConstInt,
    HV: ConstInt,
    K: ConstInt,
    V: ConstInt,
    BT: ConstInt,
    BK: ConstInt,
    BV: ConstInt,
    HVH: ConstInt,
    NT_FIXED: ConstInt,
    USE_DW: ConstBool,
    IS_VARLEN: ConstBool,
):
    i_k = ct.bid(0)
    i_t = ct.bid(1)
    i_bh = ct.bid(2)
    i_b = i_bh // HV
    i_h = i_bh % HV
    i_kvh = i_h // HVH

    if IS_VARLEN:
        i_n = ct.load(chunk_indices, index=(i_t, 0), shape=()).astype(ct.int32)
        i_tt = ct.load(chunk_indices, index=(i_t, 1), shape=()).astype(ct.int32)
        bos = ct.load(cu_seqlens, index=(i_n,), shape=()).astype(ct.int32)
        eos = ct.load(cu_seqlens, index=(i_n + 1,), shape=()).astype(ct.int32)
        seqlen = eos - bos
        h_row = i_t
    else:
        bos = i_b * T
        eos = bos + T
        seqlen = T
        i_tt = i_t
        h_row = i_b * NT_FIXED + i_t

    q_seg = q.slice(axis=0, start=bos, stop=eos)
    k_seg = k.slice(axis=0, start=bos, stop=eos)
    v_seg = v.slice(axis=0, start=bos, stop=eos)
    do_seg = do.slice(axis=0, start=bos, stop=eos)
    dv_seg = dv.slice(axis=0, start=bos, stop=eos)
    dq_seg = dq.slice(axis=0, start=bos, stop=eos)
    dk_seg = dk.slice(axis=0, start=bos, stop=eos)
    dw_seg = dw.slice(axis=0, start=bos, stop=eos)

    b_dq = ct.zeros((BT, BK), dtype=ct.float32)
    b_dk = ct.zeros((BT, BK), dtype=ct.float32)
    b_ds = ct.zeros((BT, BT), dtype=ct.float32)
    b_dw = ct.zeros((BT, BK), dtype=ct.float32)

    for i_v in ct.static_iter(range(_ck(V, BV))):
        b_v = ct.load(v_seg, index=(i_tt, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )
        b_do = ct.load(do_seg, index=(i_tt, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
            (BT, BV)
        )
        # h/dh blocks [BV, BK] (non state_v_first: stored [K, V], load (i_k, i_v) -> [BK, BV], transpose)
        b_h = ct.transpose(
            ct.load(h, index=(h_row, i_h, i_k, i_v), shape=(1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BK, BV)
            )
        )
        b_dh = ct.transpose(
            ct.load(dh, index=(h_row, i_h, i_k, i_v), shape=(1, 1, BK, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BK, BV)
            )
        )
        acc_ds = b_ds
        b_ds = ct.mma(b_do, ct.transpose(b_v), acc_ds)
        acc_dq = b_dq
        b_dq = ct.mma(b_do, ct.astype(b_h, b_do.dtype), acc_dq)
        acc_dk = b_dk
        b_dk = ct.mma(b_v, ct.astype(b_dh, b_v.dtype), acc_dk)
        if USE_DW:
            b_dv = ct.load(dv_seg, index=(i_tt, i_h, i_v), shape=(BT, 1, BV), padding_mode=ct.PaddingMode.ZERO).reshape(
                (BT, BV)
            )
            acc_dw = b_dw
            b_dw = ct.mma(ct.astype(b_dv, b_v.dtype), ct.astype(b_h, b_v.dtype), acc_dw)

    if USE_DW:
        ct.store(dw_seg, index=(i_tt, i_h, i_k), tile=ct.astype(-b_dw, dw.dtype).reshape((BT, 1, BK)))

    b_q = ct.load(q_seg, index=(i_tt, i_kvh, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
        (BT, BK)
    )
    b_k = ct.load(k_seg, index=(i_tt, i_kvh, i_k), shape=(BT, 1, BK), padding_mode=ct.PaddingMode.ZERO).reshape(
        (BT, BK)
    )

    o_t = i_tt * BT + ct.arange(BT, dtype=ct.int32)
    m_t = o_t < seqlen
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :])

    # g=None / g_gamma=None branch:
    #   b_ds = tril_incl(b_ds)
    #   b_dq += b_ds @ b_k          ;  b_dq *= scale
    #   b_dk += (b_ds^T @ b_q) * scale   (only the new term is scaled)
    b_ds = ct.astype(ct.where(m_A, b_ds, ct.zeros((BT, BT), dtype=ct.float32)), b_k.dtype)
    acc_dq = b_dq
    b_dq = ct.mma(b_ds, b_k, acc_dq)
    acc_dk = ct.zeros((BT, BK), dtype=ct.float32)
    b_dk = b_dk + ct.mma(ct.transpose(b_ds), b_q, acc_dk) * scale
    b_dq = b_dq * scale

    ct.store(dq_seg, index=(i_tt, i_h, i_k), tile=ct.astype(b_dq, dq.dtype).reshape((BT, 1, BK)))
    ct.store(dk_seg, index=(i_tt, i_h, i_k), tile=ct.astype(b_dk, dk.dtype).reshape((BT, 1, BK)))


def chunk_bwd_dqkwg(
    q,
    k,
    v,
    do,
    h,
    dh,
    w=None,
    g=None,
    g_gamma=None,
    dv=None,
    scale=None,
    state_v_first=False,
    cu_seqlens=None,
    chunk_size=64,
    chunk_indices=None,
):
    assert not state_v_first and g is None and g_gamma is None
    B, T, H, K, V, HV = *k.shape, v.shape[-1], v.shape[2]
    BT = chunk_size
    IS_VARLEN = cu_seqlens is not None
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = _cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = min(max(_next_power_of_2(K), 16), 64)
    BV = min(max(_next_power_of_2(V), 16), 64)
    NK = _ck(K, BK)

    dq = q.new_empty(B, T, HV, K)
    dk = k.new_empty(B, T, HV, K)
    dw = torch.empty_like(w) if w is not None else None
    USE_DW = (w is not None) and (dv is not None)

    BT_total = B * T
    q3 = q.reshape(BT_total, H, K)
    k3 = k.reshape(BT_total, H, K)
    v3 = v.reshape(BT_total, HV, V)
    do3 = do.reshape(BT_total, HV, V)
    dq3 = dq.reshape(BT_total, HV, K)
    dk3 = dk.reshape(BT_total, HV, K)
    dw3 = dw.reshape(BT_total, HV, K) if dw is not None else torch.empty((1, 1, 1), device=q.device, dtype=q.dtype)
    dv3 = dv.reshape(BT_total, HV, V) if dv is not None else torch.empty((1, 1, 1), device=q.device, dtype=q.dtype)
    if IS_VARLEN:
        ci = chunk_indices.to(torch.int32)
        cs = cu_seqlens.to(torch.int32)
    else:
        ci = torch.empty((1, 2), device=q.device, dtype=torch.int32)
        cs = torch.empty((1,), device=q.device, dtype=torch.int32)

    NT_fixed = _cdiv(T, BT) if cu_seqlens is None else NT
    grid = (NK, NT, B * HV)
    args = (
        q3,
        k3,
        v3,
        h,
        do3,
        dh,
        dq3,
        dk3,
        dw3,
        dv3,
        cs,
        ci,
        float(scale),
        T,
        H,
        HV,
        K,
        V,
        BT,
        BK,
        BV,
        HV // H,
        NT_fixed,
        USE_DW,
        IS_VARLEN,
    )
    cache_key = (
        "chunk_bwd_kernel_dqkwg",
        B,
        T,
        H,
        HV,
        K,
        V,
        BT,
        BK,
        BV,
        USE_DW,
        IS_VARLEN,
        str(q.dtype),
        str(q.device),
    )
    _autotuned_launch(chunk_bwd_kernel_dqkwg, cache_key, grid, args)

    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    return dq, dk, dw, None


# ===========================================================================
# fla.ops.delta_rule.chunk (entry)
# ===========================================================================


def chunk_delta_rule_fwd(
    q,
    k,
    v,
    beta,
    scale,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    chunk_indices=None,
):
    # obtain WY representation. u is actually the new v.
    w, u, A = prepare_wy_repr_fwd(k=k, v=v, beta=beta, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    return o, A, final_state


def chunk_delta_rule_bwd(
    q,
    k,
    v,
    beta,
    A,
    scale,
    initial_state,
    do,
    dht,
    cu_seqlens=None,
    chunk_indices=None,
):
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dv = chunk_bwd_dv_local(q=q, k=k, do=do, g=None, scale=scale, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=None,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dq, dk, dw, _ = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        h=h,
        w=w,
        dv=dv,
        do=do,
        dh=dh,
        g=None,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk2, dv, db = prepare_wy_repr_bwd(
        k=k, v=v, beta=beta, A=A, dw=dw, du=dv, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    dk.add_(dk2)
    return dq, dk, dv, db, dh0


class ChunkDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, 64, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
        )
        o, A, final_state = chunk_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens, chunk_indices)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return ct_astype_torch(o, q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, q_rstd, k, k_rstd, v, beta, A, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors

        dq, dk, dv, db, dh0 = chunk_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return (
            ct_astype_torch(dq, q.dtype),
            ct_astype_torch(dk, k.dtype),
            ct_astype_torch(dv, v.dtype),
            ct_astype_torch(db, beta.dtype),
            None,
            dh0,
            None,
            None,
            None,
            None,
        )


def ct_astype_torch(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Host-side tensor dtype cast (torch). cuTile `ct.astype` only applies to
    in-kernel tiles; on the host we use the regular torch cast."""
    return x if x.dtype == dtype else x.to(dtype)


@torch.compiler.disable
def chunk_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    **kwargs,
):
    r"""Chunked DeltaNet. Behavior identical to
    ``fla.ops.delta_rule.chunk.chunk_delta_rule``.

    Args:
        q, k: queries/keys of shape ``[B, T, H, K]``.
        v: values of shape ``[B, T, H, V]``.
        beta: betas of shape ``[B, T, H]``.
        scale: score scale, defaults to ``1 / sqrt(K)``.
        initial_state: recurrent state ``[N, H, K, V]`` for N sequences.
        output_final_state: whether to return the final state.
        use_qk_l2norm_in_kernel: whether to l2-normalize q/k within the kernel.
        cu_seqlens: cumulative sequence lengths ``[N+1]`` for varlen.
        cu_seqlens_cpu: CPU copy of ``cu_seqlens`` to avoid device sync.

    Returns:
        ``(o, final_state)`` mirroring fla's shapes/semantics.
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "ChunkDeltaRuleFunction does not support float32. Please use bfloat16."
    assert len(beta.shape) == 3, "beta must be of shape (batch size, num of head, seq len)."

    if "head_first" in kwargs:
        raise DeprecationWarning(
            "head_first has been removed. Inputs must be in `[B, T, H, ...]` format.",
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDeltaRuleFunction.apply(
        q,
        k,
        v,
        beta,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        cu_seqlens_cpu,
    )
    return o, final_state
