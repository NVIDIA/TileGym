# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from math import ceil
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch

from tilegym.backend import register_impl
from tilegym.logger import get_logger

logger = get_logger(__name__)

# Type aliases for constants
ConstInt = ct.Constant[int]
LOG2E = 1.4426950408889634


def _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M):
    group_id = tile_id // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = ct.minimum(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (tile_id % group_size_m)
    bid_n = (tile_id % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def _sigmoid(x):
    return 1.0 / (1.0 + ct.exp(-x))


@ct.kernel
def mhc_split_gemm_rms_kernel(
    X,
    W,
    Y_acc,
    R_acc,
    M: int,
    N: int,
    K: int,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    TILE_SIZE_K: ConstInt,
    SPLIT_K: ConstInt,
    GROUP_SIZE_M: ConstInt,
):
    """Split-K fused GEMM + RMS compute kernel for mHC.

    Key optimization: All blocks compute RMS to avoid wasting registers.
    Each block computes partial RMS for its K-tile range, which are later
    summed in the finalize kernel.
    """
    tile_id = ct.bid(0)
    bid_k = ct.bid(1)
    zero_pad = ct.PaddingMode.ZERO

    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    bid_m, bid_n = _compute_bid(tile_id, num_bid_in_group, num_bid_m, GROUP_SIZE_M)
    k_tiles = ct.cdiv(K, TILE_SIZE_K)
    k_tiles_per_split = ct.cdiv(k_tiles, SPLIT_K)
    k_tile_start = bid_k * k_tiles_per_split
    k_tile_end = ct.minimum(k_tile_start + k_tiles_per_split, k_tiles)

    rms_acc = ct.full((TILE_SIZE_M,), 0.0, dtype=ct.float32)
    accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
    mma_dtype = ct.tfloat32 if (X.dtype == ct.float32 or W.dtype == ct.float32) else X.dtype

    for k_tile in range(k_tile_start, k_tile_end):
        a = ct.load(
            X,
            index=(bid_m, k_tile),
            shape=(TILE_SIZE_M, TILE_SIZE_K),
            padding_mode=zero_pad,
            allow_tma=True,
        )
        b = ct.load(
            W,
            index=(k_tile, bid_n),
            shape=(TILE_SIZE_K, TILE_SIZE_N),
            padding_mode=zero_pad,
            allow_tma=True,
        )

        a_mma = ct.astype(a, mma_dtype)
        b_mma = ct.astype(b, mma_dtype)
        accumulator = ct.mma(a_mma, b_mma, acc=accumulator)

        a_fp32 = ct.astype(a, ct.float32)
        rms_acc = rms_acc + ct.sum(a_fp32 * a_fp32, axis=1, keepdims=False)

    bid_m_k = bid_m + bid_k * num_bid_m
    ct.store(Y_acc, index=(bid_m_k, bid_n), tile=accumulator)

    # Store RMS partial results - will be summed across bid_n in finalize kernel
    # Using bid_n as additional dimension for partial sums
    ct.store(R_acc, index=(bid_m_k, bid_n), tile=ct.reshape(rms_acc, (TILE_SIZE_M, 1)))


@ct.kernel
def mhc_finalize_scale_bias_sigmoid_kernel(
    Y_acc,
    R_acc,
    Y,
    R,
    n: int,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    Bias,
    M: int,
    N: int,
    K: int,
    TILE_SIZE_M: ConstInt,
    TILE_SIZE_N: ConstInt,
    SPLIT_K: ConstInt,
):
    """Finalize split-K + fused scale/bias/sigmoid kernel for mHC."""
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    num_bid_m = ct.cdiv(M, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)

    y_accum = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
    r_accum = ct.full((TILE_SIZE_M, 1), 0.0, dtype=ct.float32)

    # Sum across split_k dimension
    for split_idx in range(SPLIT_K):
        bid_m_k = bid_m + split_idx * num_bid_m
        y_tile = ct.load(
            Y_acc,
            index=(bid_m_k, bid_n),
            shape=(TILE_SIZE_M, TILE_SIZE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        y_accum = y_accum + y_tile

        # RMS is independent of bid_n; each bid_n block stores the same partial RMS.
        # Loading the current bid_n avoids over-counting when num_bid_n > 1.
        r_tile = ct.load(
            R_acc,
            index=(bid_m_k, bid_n),
            shape=(TILE_SIZE_M, 1),
            padding_mode=ct.PaddingMode.ZERO,
        )
        r_tile = ct.astype(r_tile, ct.float32)
        r_accum = r_accum + r_tile

    denom = ct.full((TILE_SIZE_M, 1), K * 1.0, dtype=ct.float32)
    mean = ct.truediv(r_accum, denom)
    rstd = ct.rsqrt(mean)
    ones = ct.full((TILE_SIZE_M, 1), 1.0, dtype=ct.float32)
    r = ct.truediv(ones, rstd)
    if bid_n == 0:
        r_out = ct.astype(r, R.dtype)
        ct.store(R, index=(bid_m, 0), tile=r_out)

    offsets = ct.arange(TILE_SIZE_N, dtype=ct.int32)
    col_ids = bid_n * TILE_SIZE_N + offsets
    bias = ct.load(Bias, index=(bid_n,), shape=(TILE_SIZE_N,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.reshape(bias, (1, TILE_SIZE_N))

    one = ct.full((TILE_SIZE_N,), 1.0, dtype=ct.float32)
    zero = ct.full((TILE_SIZE_N,), 0.0, dtype=ct.float32)
    mask_pre = ct.where(ct.less(col_ids, n), one, zero)
    mask_post = ct.where(ct.less(col_ids, 2 * n), one, zero)
    mask_post = mask_post - mask_pre
    mask_res = one - mask_pre - mask_post

    scale = alpha_pre * mask_pre + alpha_post * mask_post + alpha_res * mask_res
    scale = ct.reshape(scale, (1, TILE_SIZE_N))

    linear = ct.truediv(y_accum * scale, r) + ct.astype(bias, ct.float32)
    sigmoid_linear = _sigmoid(linear)
    two_sigmoid = sigmoid_linear * 2.0

    mask_pre = ct.reshape(mask_pre, (1, TILE_SIZE_N))
    mask_post = ct.reshape(mask_post, (1, TILE_SIZE_N))
    mask_res = ct.reshape(mask_res, (1, TILE_SIZE_N))

    out = linear * mask_res + sigmoid_linear * mask_pre + two_sigmoid * mask_post
    out = ct.astype(out, Y.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=out)


def _mhc_split_gemm_rms_autotune_configs():
    tile_ms = (64, 128)
    tile_ks = (64, 128)
    split_ks = (1, 2, 4, 8, 16)
    group_size_ms = (8, 16)
    tile_n = 32
    for tile_m in tile_ms:
        for tile_k in tile_ks:
            for split_k in split_ks:
                for group_size_m in group_size_ms:
                    yield SimpleNamespace(
                        TILE_SIZE_M=tile_m,
                        TILE_SIZE_N=tile_n,
                        TILE_SIZE_K=tile_k,
                        SPLIT_K=split_k,
                        GROUP_SIZE_M=group_size_m,
                    )


def cutile_autotune_mhc_split_gemm_rms(stream, x, w, M, N, K, cfg=None):
    if cfg is not None:
        if isinstance(cfg, dict):
            cfg = SimpleNamespace(**cfg)
        if not hasattr(cfg, "TILE_SIZE_M") and hasattr(cfg, "m"):
            cfg.TILE_SIZE_M = cfg.m
        if not hasattr(cfg, "TILE_SIZE_N") and hasattr(cfg, "n"):
            cfg.TILE_SIZE_N = cfg.n
        if not hasattr(cfg, "TILE_SIZE_K") and hasattr(cfg, "k"):
            cfg.TILE_SIZE_K = cfg.k
        if not hasattr(cfg, "SPLIT_K") and hasattr(cfg, "split_k"):
            cfg.SPLIT_K = cfg.split_k
        if not hasattr(cfg, "GROUP_SIZE_M") and hasattr(cfg, "group_size_m"):
            cfg.GROUP_SIZE_M = cfg.group_size_m

        num_bid_n = ceil(N / cfg.TILE_SIZE_N)
        y_acc = torch.empty((M * cfg.SPLIT_K, N), device=x.device, dtype=torch.float32)
        # R_acc now stores partial RMS for all N blocks
        r_acc = torch.empty((M * cfg.SPLIT_K, num_bid_n), device=x.device, dtype=torch.float32)
        grid = (
            ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N),
            cfg.SPLIT_K,
            1,
        )
        ct.launch(
            stream,
            grid,
            mhc_split_gemm_rms_kernel,
            (
                x,
                w,
                y_acc,
                r_acc,
                M,
                N,
                K,
                cfg.TILE_SIZE_M,
                cfg.TILE_SIZE_N,
                cfg.TILE_SIZE_K,
                cfg.SPLIT_K,
                cfg.GROUP_SIZE_M,
            ),
        )
        return y_acc, r_acc, cfg

    configs = list(_mhc_split_gemm_rms_autotune_configs())
    max_split_k = max(cfg.SPLIT_K for cfg in configs)
    # Need max num_bid_n across all configs
    max_num_bid_n = max(ceil(N / cfg.TILE_SIZE_N) for cfg in configs)
    y_acc = torch.empty((M * max_split_k, N), device=x.device, dtype=torch.float32)
    r_acc = torch.empty((M * max_split_k, max_num_bid_n), device=x.device, dtype=torch.float32)
    tuned = ct_experimental.autotune_launch(
        stream,
        grid_fn=lambda cfg: (
            ceil(M / cfg.TILE_SIZE_M) * ceil(N / cfg.TILE_SIZE_N),
            cfg.SPLIT_K,
            1,
        ),
        kernel=mhc_split_gemm_rms_kernel,
        args_fn=lambda cfg: (
            x,
            w,
            y_acc,
            r_acc,
            M,
            N,
            K,
            cfg.TILE_SIZE_M,
            cfg.TILE_SIZE_N,
            cfg.TILE_SIZE_K,
            cfg.SPLIT_K,
            cfg.GROUP_SIZE_M,
        ),
        search_space=configs,
    )
    best_cfg = tuned.tuned_config
    return y_acc, r_acc, best_cfg


def mhc_split_gemm_rms(x: torch.Tensor, w: torch.Tensor, **kwargs):
    M, K = x.shape
    KB, N = w.shape
    assert K == KB, f"Incompatible matrices: K dimension of X is {K}, K dimension of W is {KB}"

    cfg = kwargs.pop("cfg", None)
    kwargs.pop("w_nt", None)
    w = w.contiguous()

    stream = torch.cuda.current_stream()
    return cutile_autotune_mhc_split_gemm_rms(stream, x, w, M, N, K, cfg=cfg)


def mhc_finalize_scale_bias_sigmoid(
    y_acc: torch.Tensor,
    r_acc: torch.Tensor,
    n: int,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    M: int,
    K: int,
    **kwargs,
):
    cfg = kwargs.pop("cfg", None)
    split_k = kwargs.pop("split_k", None)
    tile_m = kwargs.pop("tile_m", None)
    tile_n = kwargs.pop("tile_n", None)
    if cfg is not None:
        tile_m = cfg.TILE_SIZE_M
        tile_n = cfg.TILE_SIZE_N
        split_k = cfg.SPLIT_K

    y_acc = y_acc.contiguous()
    r_acc = r_acc.contiguous()
    bias = bias.contiguous()
    N = y_acc.shape[1]

    y = torch.empty((M, N), device=y_acc.device, dtype=bias.dtype)
    r = torch.empty((M, 1), device=y_acc.device, dtype=torch.float32)

    grid = (ceil(M / tile_m), ceil(N / tile_n), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        mhc_finalize_scale_bias_sigmoid_kernel,
        (
            y_acc,
            r_acc,
            y,
            r,
            n,
            float(alpha_pre),
            float(alpha_post),
            float(alpha_res),
            bias,
            M,
            N,
            K,
            tile_m,
            tile_n,
            split_k,
        ),
    )
    return y, r


@register_impl("mhc_gemm_rms_scale", backend="cutile")
def mhc_gemm_rms_scale(
    x: torch.Tensor,
    w: torch.Tensor,
    n: int,
    alpha_pre: float,
    alpha_post: float,
    alpha_res: float,
    bias: torch.Tensor,
    **kwargs,
):
    cfg = kwargs.pop("cfg", None)
    kwargs.pop("w_nt", None)
    w = w.contiguous()

    M, K = x.shape
    _, N = w.shape
    y_acc, r_acc, cfg = cutile_autotune_mhc_split_gemm_rms(
        torch.cuda.current_stream(),
        x,
        w,
        M,
        N,
        K,
        cfg=cfg,
    )
    return mhc_finalize_scale_bias_sigmoid(
        y_acc,
        r_acc,
        n,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        M,
        K,
        cfg=cfg,
    )


@ct.kernel
def mhc_apply_residual_kernel(
    X,
    F_out,
    Y_post,
    Y_res,
    Out,
    C: int,
    n: ct.Constant[int],
    TILE_SIZE_C: ConstInt,
):
    """Apply H_res and H_post to residual stream (in-place on Out)."""
    # Shapes:
    # - X: [B, n, C] view of residual stream
    # - F_out: [B, C]
    # - Y_post: [B, n]
    # - Y_res: [B, n, n]
    # - Out: [B, n, C]
    row = ct.bid(0)
    c_tile = ct.bid(1)
    compute_dtype = (
        ct.float32 if (X.dtype == ct.float32 or F_out.dtype == ct.float32 or Y_post.dtype == ct.float32) else X.dtype
    )

    f_tile = ct.load(
        F_out,
        index=(row, c_tile),
        shape=(1, TILE_SIZE_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    f_tile = ct.astype(f_tile, compute_dtype)

    h_post = ct.load(
        Y_post,
        index=(row, 0),
        shape=(1, n),
        padding_mode=ct.PaddingMode.ZERO,
    )
    h_post = ct.reshape(h_post, (n, 1))
    h_post = ct.astype(h_post, compute_dtype)

    h_res = ct.load(
        Y_res,
        index=(row, 0, 0),
        shape=(1, n, n),
        padding_mode=ct.PaddingMode.ZERO,
    )
    h_res = ct.reshape(h_res, (n, n))
    h_res = ct.astype(h_res, compute_dtype)

    acc = ct.full((n, TILE_SIZE_C), 0.0, dtype=compute_dtype)
    for j in range(n):
        x_row = ct.load(
            X,
            index=(row, j, c_tile),
            shape=(1, 1, TILE_SIZE_C),
            padding_mode=ct.PaddingMode.ZERO,
        )
        x_row = ct.reshape(x_row, (1, TILE_SIZE_C))
        x_row = ct.astype(x_row, compute_dtype)
        h_col = ct.extract(h_res, (0, j), shape=(n, 1))
        x_row = ct.broadcast_to(x_row, (n, TILE_SIZE_C))
        h_col = ct.broadcast_to(h_col, (n, TILE_SIZE_C))
        prod = h_col * x_row
        acc = acc + prod
    h_post = ct.broadcast_to(h_post, (n, TILE_SIZE_C))
    f_tile = ct.broadcast_to(f_tile, (n, TILE_SIZE_C))
    x_post = h_post * f_tile
    out_tile = acc + x_post
    out_tile = ct.astype(out_tile, Out.dtype)
    out_tile = ct.reshape(out_tile, (1, n, TILE_SIZE_C))
    ct.store(Out, index=(row, 0, c_tile), tile=out_tile)


@register_impl("mhc_apply_residual", backend="cutile")
def mhc_apply_residual(
    x: torch.Tensor,
    f_out: torch.Tensor,
    y: torch.Tensor,
    n: int,
    **kwargs,
):
    x = x.contiguous()
    f_out = f_out.contiguous()
    y = y.contiguous()
    B, nC = x.shape
    C = f_out.shape[1]
    # Use view for [B, n, C] without changing external layout.
    x_view = x.view(B, n, C)
    y_post = y.narrow(1, n, n)
    y_res = y.narrow(1, 2 * n, n * n).view(B, n, n)
    out = torch.empty_like(x)
    out_view = out.view(B, n, C)

    TILE_SIZE_C = 1024
    grid = (B, C // TILE_SIZE_C, 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        mhc_apply_residual_kernel,
        (
            x_view,
            f_out,
            y_post,
            y_res,
            out_view,
            C,
            n,
            TILE_SIZE_C,
        ),
    )
    return out


@ct.kernel
def mhc_sinkhorn_kernel(
    Y,
    m: ct.Constant[int],
    n: ct.Constant[int],
):
    """Sinkhorn-Knopp normalization for residual block (in-place on Y)."""
    row = ct.bid(0)
    total = n * n
    mat = ct.load(Y, index=(row, 0), shape=(m, total))
    mat = ct.reshape(mat, (m, n, n))
    mat = ct.astype(mat, ct.float32)
    mat = ct.exp2(mat * LOG2E)

    for _ in range(20):
        row_sum = ct.sum(mat, axis=1, keepdims=True)
        mat = ct.truediv(mat, row_sum)
        col_sum = ct.sum(mat, axis=0, keepdims=True)
        mat = ct.truediv(mat, col_sum)

    mat = ct.reshape(mat, (m, total))
    mat = ct.astype(mat, Y.dtype)
    ct.store(Y, index=(row, 0), tile=mat)


@register_impl("mhc_sinkhorn", backend="cutile")
def mhc_sinkhorn(
    y: torch.Tensor,
    n: int, tileM: int = 8,
    **kwargs,
):
    y = y.contiguous()
    M, _ = y.shape
    y_view = y.narrow(1, 2 * n, n * n)
    grid = (ct.cdiv(M, tileM),)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        mhc_sinkhorn_kernel,
        (
            y_view,
            n,
        ),
    )
    return y
