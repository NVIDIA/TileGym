# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
CuTile implementation of Flash Multi-Head Attention variant with support for:
- Multiple layouts (bnsd, nsbd)
- Causal masking
- Window attention (sliding window)
- Random masking
- Multiple bias types (vector, matrix, alibi)
- Variable length sequences (q_lens, kv_lens)
- GQA (grouped query attention) support
"""

import math
from types import SimpleNamespace
from typing import Optional

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import is_autotune_enabled
from tilegym.backend import register_impl
from tilegym.logger import get_logger

from .utils import cached_replace_hints

# Module-level tune cache: (B, H, S_qo, S_kv, TILE_D, OUT_TILE_D, query_group_size, is_causal, window_size, BIAS_TYPE, USE_RANDOM_MASK, USE_Q_LENS, USE_KV_LENS, layout, dtype, device) -> (best_cfg, tuned_kernel)
_fmha_variant_tune_cache: dict = {}

logger = get_logger(__name__)

INV_LOG_2 = 1.0 / math.log(2)

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _fmha_variant_autotune_configs():
    """Iterator of autotune configurations for FMHA variant kernel."""
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # Blackwell: more conservative configs due to memory constraints
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=128, TILE_N=64, num_ctas=1, occupancy=2)
    elif gpu_capability[0] < 9:
        # Pre-Hopper: smaller tiles
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=128, TILE_N=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=64, TILE_N=128, num_ctas=1, occupancy=2)
    else:
        # Hopper (SM90): Larger tiles work well, matching Triton configs
        # Priority order: larger tiles first for better compute efficiency
        yield SimpleNamespace(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=128, TILE_N=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=64, TILE_N=128, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=4)


@ct.kernel
def _fmha_variant_kernel_bnsd(
    Q,
    K,
    V,
    Out,
    Bias,  # Can be vector bias [B,H,1,S_kv], matrix bias [B,H,S_qo,S_kv], or alibi [H]
    Random_mask,  # Random mask tensor [B,H,S_qo,S_kv] or None
    Q_lens,  # Variable query lengths [B] or None
    KV_lens,  # Variable kv lengths [B] or None
    qk_scale: float,
    S_qo: int,  # Max query sequence length
    S_kv: int,  # Max key/value sequence length
    TILE_D: ConstInt,
    OUT_TILE_D: ConstInt,
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
    EVEN_M: ConstBool,
    EVEN_N: ConstBool,
    USE_Q_LENS: ConstBool,
    USE_KV_LENS: ConstBool,
    WINDOW_SIZE: ConstInt,
    BIAS_TYPE: ConstInt,  # 0=none, 1=vector, 2=matrix, 3=alibi
    USE_RANDOM_MASK: ConstBool,
):
    """
    CuTile kernel for FMHA variant with bnsd layout.
    Q, K, V have shape [batch, heads, seq, head_dim]
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)  # M-dimension block (query tile index)
    bid_y = ct.bid(1)  # batch * heads
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE if QUERY_GROUP_SIZE > 0 else head_idx

    # Adjust qk_scale for exp2
    qk_scale_log2 = qk_scale * INV_LOG_2

    # Get actual sequence lengths
    if USE_Q_LENS:
        q_len_val = ct.load(Q_lens, index=(batch_idx,), shape=(1,)).reshape(())
    else:
        q_len_val = S_qo

    if USE_KV_LENS:
        kv_len_val = ct.load(KV_lens, index=(batch_idx,), shape=(1,)).reshape(())
    else:
        kv_len_val = S_kv

    # Calculate prefix_kvlen (offset for causal masking with variable lengths)
    prefix_kvlen = kv_len_val - q_len_val

    # Early exit if this block is beyond actual query length
    start_m = bid_x * TILE_M
    if start_m >= q_len_val:
        # Zero output for this block
        acc_zero = ct.full((1, 1, TILE_M, OUT_TILE_D), 0.0, dtype=Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc_zero)
        return

    # Initialize offsets for masking calculations
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]  # [TILE_M, 1]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators
    NEG_INF = -1e9
    m_i = ct.full((TILE_M, 1), NEG_INF, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, OUT_TILE_D), 0.0, dtype=ct.float32)

    # Load query tile - bnsd layout: [batch, heads, seq, dim]
    # index=(batch_tile, head_tile, seq_tile, dim_tile), shape=(1, 1, TILE_M, TILE_D)
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D))
    q = q.reshape((TILE_M, TILE_D))

    # Load alibi scale if needed
    if BIAS_TYPE == 3:  # alibi
        alibi_scale = ct.load(Bias, index=(head_idx,), shape=(1,)).reshape(())
        alibi_scale = alibi_scale.astype(ct.float32)

    # Compute loop bounds with window attention support
    if WINDOW_SIZE > 0:
        kv_start = max(0, (prefix_kvlen + start_m - WINDOW_SIZE) // TILE_N * TILE_N)
        kv_end = min(kv_len_val, prefix_kvlen + (bid_x + 1) * TILE_M + WINDOW_SIZE)
    elif CAUSAL:
        # Causal: KV positions past this query tile's last causal key are fully
        # masked, so the loop is bounded at the diagonal rather than all of S_kv.
        kv_start = 0
        kv_end = min(kv_len_val, prefix_kvlen + (bid_x + 1) * TILE_M)
    else:
        kv_start = 0
        kv_end = kv_len_val

    # Calculate number of KV tiles to process
    num_tiles = ct.cdiv(kv_end - kv_start, TILE_N)
    start_tile = kv_start // TILE_N

    # Single loop over all KV tiles
    for tile_idx in range(num_tiles):
        kv_tile = start_tile + tile_idx  # Actual KV tile index
        kv_pos = kv_start + tile_idx * TILE_N  # Starting KV position for this tile

        # Load K tile and transpose - bnsd layout: [batch, heads, seq, dim]
        # With order=(0,1,3,2): index[2] maps to dim, index[3] maps to seq
        # So we put 0 for dim tile index, and kv_tile for seq tile index
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, kv_tile),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))

        # Compute QK = Q @ K^T
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # Apply bias
        if BIAS_TYPE == 1:  # vector bias [B, H, 1, S_kv]
            bias_tile = ct.load(Bias, index=(batch_idx, head_idx, 0, kv_tile), shape=(1, 1, 1, TILE_N))
            bias_tile = bias_tile.reshape((1, TILE_N)).astype(ct.float32)
            qk = qk * qk_scale_log2 + bias_tile * INV_LOG_2
        elif BIAS_TYPE == 2:  # matrix bias [B, H, S_qo, S_kv]
            bias_tile = ct.load(Bias, index=(batch_idx, head_idx, bid_x, kv_tile), shape=(1, 1, TILE_M, TILE_N))
            bias_tile = bias_tile.reshape((TILE_M, TILE_N)).astype(ct.float32)
            qk = qk * qk_scale_log2 + bias_tile * INV_LOG_2
        elif BIAS_TYPE == 3:  # alibi bias
            offs_n_full = kv_pos + offs_n_tile  # [1, TILE_N]
            # Compute |q_pos - k_pos| using max(x, -x) since ct.abs doesn't exist
            diff = (offs_m + prefix_kvlen).astype(ct.float32) - offs_n_full.astype(ct.float32)
            neg_dist = -ct.maximum(diff, -diff)
            qk = qk * qk_scale_log2 + alibi_scale * neg_dist * INV_LOG_2

        # Apply random mask
        if USE_RANDOM_MASK:
            rmask = ct.load(Random_mask, index=(batch_idx, head_idx, bid_x, kv_tile), shape=(1, 1, TILE_M, TILE_N))
            rmask = rmask.reshape((TILE_M, TILE_N))
            qk = ct.where(rmask != 0, ct.full((TILE_M, TILE_N), NEG_INF, dtype=ct.float32), qk)

        # Tiles whose last key is at or before this query tile's first causal
        # position are entirely visible, so the causal mask is skipped for them.
        if CAUSAL:
            if kv_pos + TILE_N > start_m + prefix_kvlen + 1:
                offs_n_full = kv_pos + offs_n_tile
                causal_mask = (offs_m + prefix_kvlen) >= offs_n_full
                qk = ct.where(causal_mask, qk, ct.full((TILE_M, TILE_N), NEG_INF, dtype=ct.float32))

        # Apply window mask
        if WINDOW_SIZE > 0:
            offs_n_full = kv_pos + offs_n_tile
            qk_offset = offs_n_full - prefix_kvlen - offs_m
            window_mask = (qk_offset >= -WINDOW_SIZE) & (qk_offset <= WINDOW_SIZE)
            qk = ct.where(window_mask, qk, ct.full((TILE_M, TILE_N), NEG_INF, dtype=ct.float32))

        # Apply KV length mask for partial tiles
        if USE_KV_LENS or not EVEN_N:
            offs_n_full = kv_pos + offs_n_tile
            kv_mask = offs_n_full < kv_len_val
            qk = ct.where(kv_mask, qk, ct.full((TILE_M, TILE_N), NEG_INF, dtype=ct.float32))

        # Online softmax update
        if BIAS_TYPE == 0:  # no bias - scale after max for better precision
            m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
            qk = qk * qk_scale_log2 - m_ij
        else:  # bias already applied with scaling
            m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True))
            qk = qk - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V and compute P @ V
        # V has shape [batch, heads, seq, dim], no transpose needed
        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, kv_tile, 0),
            shape=(1, 1, TILE_N, OUT_TILE_D),
            latency=4,
        )
        v = v.reshape((TILE_N, OUT_TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    # Final normalization and store
    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, OUT_TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


@ct.kernel
def _fmha_variant_kernel_nsbd(
    Q,
    K,
    V,
    Out,
    Bias,
    Random_mask,
    Q_lens,
    KV_lens,
    qk_scale: float,
    S_qo: int,
    S_kv: int,
    TILE_D: ConstInt,
    OUT_TILE_D: ConstInt,
    H: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
    EVEN_M: ConstBool,
    EVEN_N: ConstBool,
    USE_Q_LENS: ConstBool,
    USE_KV_LENS: ConstBool,
    WINDOW_SIZE: ConstInt,
    BIAS_TYPE: ConstInt,
    USE_RANDOM_MASK: ConstBool,
):
    """
    CuTile kernel for FMHA variant with nsbd layout.
    Q, K, V have shape [num_heads, seq_len, batch, head_dim]
    """
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE if QUERY_GROUP_SIZE > 0 else head_idx

    qk_scale_log2 = qk_scale * INV_LOG_2

    if USE_Q_LENS:
        q_len_val = ct.load(Q_lens, index=(batch_idx,), shape=(1,)).reshape(())
    else:
        q_len_val = S_qo

    if USE_KV_LENS:
        kv_len_val = ct.load(KV_lens, index=(batch_idx,), shape=(1,)).reshape(())
    else:
        kv_len_val = S_kv

    prefix_kvlen = kv_len_val - q_len_val
    start_m = bid_x * TILE_M

    if start_m >= q_len_val:
        acc_zero = ct.full((1, TILE_M, 1, OUT_TILE_D), 0.0, dtype=Out.dtype)
        ct.store(Out, index=(head_idx, bid_x, batch_idx, 0), tile=acc_zero)
        return

    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m = offs_m[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]

    NEG_INF = -1e9
    m_i = ct.full((TILE_M, 1), NEG_INF, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, OUT_TILE_D), 0.0, dtype=ct.float32)

    # Load Q with nsbd layout: [heads, seq, batch, dim]
    q = ct.load(Q, index=(head_idx, bid_x, batch_idx, 0), shape=(1, TILE_M, 1, TILE_D))
    q = q.reshape((TILE_M, TILE_D))

    if BIAS_TYPE == 3:
        alibi_scale = ct.load(Bias, index=(head_idx,), shape=(1,)).reshape(())
        alibi_scale = alibi_scale.astype(ct.float32)

    if WINDOW_SIZE > 0:
        kv_start = max(0, (prefix_kvlen + start_m - WINDOW_SIZE) // TILE_N * TILE_N)
        kv_end = min(kv_len_val, prefix_kvlen + (bid_x + 1) * TILE_M + WINDOW_SIZE)
    elif CAUSAL:
        # Causal: KV positions past this query tile's last causal key are fully
        # masked, so the loop is bounded at the diagonal rather than all of S_kv.
        kv_start = 0
        kv_end = min(kv_len_val, prefix_kvlen + (bid_x + 1) * TILE_M)
    else:
        kv_start = 0
        kv_end = kv_len_val

    num_tiles = ct.cdiv(kv_end - kv_start, TILE_N)
    start_tile = kv_start // TILE_N

    for tile_idx in range(num_tiles):
        kv_tile = start_tile + tile_idx
        kv_pos = kv_start + tile_idx * TILE_N

        # Load K with nsbd layout [heads, seq, batch, dim] and transpose
        # With order=(0,3,2,1): reorders dimensions for transpose
        k = ct.load(
            K,
            index=(off_kv_h, kv_tile, batch_idx, 0),
            shape=(1, TILE_N, 1, TILE_D),
            order=(0, 3, 2, 1),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))

        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)

        # Bias is always in bnsd format [B, H, S_qo, S_kv]
        if BIAS_TYPE == 1:
            bias_tile = ct.load(Bias, index=(batch_idx, head_idx, 0, kv_tile), shape=(1, 1, 1, TILE_N))
            bias_tile = bias_tile.reshape((1, TILE_N)).astype(ct.float32)
            qk = qk * qk_scale_log2 + bias_tile * INV_LOG_2
        elif BIAS_TYPE == 2:
            bias_tile = ct.load(Bias, index=(batch_idx, head_idx, bid_x, kv_tile), shape=(1, 1, TILE_M, TILE_N))
            bias_tile = bias_tile.reshape((TILE_M, TILE_N)).astype(ct.float32)
            qk = qk * qk_scale_log2 + bias_tile * INV_LOG_2
        elif BIAS_TYPE == 3:
            offs_n_full = kv_pos + offs_n_tile
            diff = (offs_m + prefix_kvlen).astype(ct.float32) - offs_n_full.astype(ct.float32)
            neg_dist = -ct.maximum(diff, -diff)
            qk = qk * qk_scale_log2 + alibi_scale * neg_dist * INV_LOG_2

        if USE_RANDOM_MASK:
            rmask = ct.load(Random_mask, index=(batch_idx, head_idx, bid_x, kv_tile), shape=(1, 1, TILE_M, TILE_N))
            rmask = rmask.reshape((TILE_M, TILE_N))
            qk = ct.where(rmask != 0, ct.full((TILE_M, TILE_N), NEG_INF, dtype=ct.float32), qk)

        if CAUSAL:
            if kv_pos + TILE_N > start_m + prefix_kvlen + 1:
                offs_n_full = kv_pos + offs_n_tile
                causal_mask = (offs_m + prefix_kvlen) >= offs_n_full
                qk = ct.where(causal_mask, qk, ct.full((TILE_M, TILE_N), NEG_INF, dtype=ct.float32))

        if WINDOW_SIZE > 0:
            offs_n_full = kv_pos + offs_n_tile
            qk_offset = offs_n_full - prefix_kvlen - offs_m
            window_mask = (qk_offset >= -WINDOW_SIZE) & (qk_offset <= WINDOW_SIZE)
            qk = ct.where(window_mask, qk, ct.full((TILE_M, TILE_N), NEG_INF, dtype=ct.float32))

        if USE_KV_LENS or not EVEN_N:
            offs_n_full = kv_pos + offs_n_tile
            kv_mask = offs_n_full < kv_len_val
            qk = ct.where(kv_mask, qk, ct.full((TILE_M, TILE_N), NEG_INF, dtype=ct.float32))

        if BIAS_TYPE == 0:
            m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2)
            qk = qk * qk_scale_log2 - m_ij
        else:
            m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True))
            qk = qk - m_ij

        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha

        # Load V with nsbd layout [heads, seq, batch, dim]
        v = ct.load(
            V,
            index=(off_kv_h, kv_tile, batch_idx, 0),
            shape=(1, TILE_N, 1, OUT_TILE_D),
            latency=4,
        )
        v = v.reshape((TILE_N, OUT_TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, TILE_M, 1, OUT_TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(head_idx, bid_x, batch_idx, 0), tile=acc)


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _fmha_variant_autotune(
    stream,
    q,
    k,
    v,
    o,
    bias,
    random_mask,
    q_lens,
    kv_lens,
    scaling,
    S_qo,
    S_kv,
    TILE_D,
    OUT_TILE_D,
    H,
    query_group_size,
    is_causal,
    window_size,
    BIAS_TYPE,
    USE_RANDOM_MASK,
    USE_Q_LENS,
    USE_KV_LENS,
    layout,
):
    """Autotuned launch for FMHA variant kernel."""
    B = q.shape[0] if layout == "bnsd" else q.shape[2]

    # Select kernel based on layout
    if layout == "bnsd":
        kernel = _fmha_variant_kernel_bnsd
    else:
        kernel = _fmha_variant_kernel_nsbd

    def args_fn(cfg):
        TILE_M = cfg.TILE_M
        TILE_N = cfg.TILE_N
        EVEN_M = (S_qo % TILE_M) == 0
        EVEN_N = (S_kv % TILE_N) == 0

        return (
            q,
            k,
            v,
            o,
            bias,
            random_mask,
            q_lens,
            kv_lens,
            scaling,
            S_qo,
            S_kv,
            TILE_D,
            OUT_TILE_D,
            H,
            TILE_M,
            TILE_N,
            query_group_size,
            is_causal,
            EVEN_M,
            EVEN_N,
            USE_Q_LENS,
            USE_KV_LENS,
            window_size,
            BIAS_TYPE,
            USE_RANDOM_MASK,
        )

    def grid_fn(cfg):
        TILE_M = cfg.TILE_M
        num_m_blocks = math.ceil(S_qo / TILE_M)
        return (num_m_blocks, B * H, 1)

    def hints_fn(cfg):
        return {
            "num_ctas": cfg.num_ctas,
            "occupancy": cfg.occupancy,
        }

    cache_key = (
        B,
        H,
        S_qo,
        S_kv,
        TILE_D,
        OUT_TILE_D,
        query_group_size,
        is_causal,
        window_size,
        BIAS_TYPE,
        USE_RANDOM_MASK,
        USE_Q_LENS,
        USE_KV_LENS,
        layout,
        q.dtype,
        str(q.device),
    )
    if cache_key not in _fmha_variant_tune_cache:
        result = exhaustive_search(
            list(_fmha_variant_autotune_configs()),
            stream,
            grid_fn,
            kernel,
            args_fn,
            hints_fn,
        )
        best_cfg = result.best.config
        _fmha_variant_tune_cache[cache_key] = (
            best_cfg,
            kernel.replace_hints(num_ctas=best_cfg.num_ctas, occupancy=best_cfg.occupancy),
        )
    best_cfg, tuned_kernel = _fmha_variant_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))


def _tile_fmha_variant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scaling: Optional[float] = None,
    is_causal: bool = True,
    q_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
    bias_type: Optional[str] = None,
    bias: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    seed: int = 0,
    window_size: int = 0,
    random_mask: Optional[torch.Tensor] = None,
    layout: str = "bnsd",
    **kwargs,
) -> torch.Tensor:
    """
    CuTile implementation of FMHA variant.
    """
    if dropout > 0:
        raise NotImplementedError("CuTile FMHA variant does not support dropout")

    assert layout in ["bnsd", "nsbd"], f"Invalid layout: {layout}"

    # Get dimensions based on layout
    batch_dim = layout.find("b")
    head_dim = layout.find("n")
    seq_dim = layout.find("s")

    B = q.shape[batch_dim]
    H = q.shape[head_dim]
    S_qo = q.shape[seq_dim]
    head_size = q.shape[3]
    out_head_size = v.shape[3]

    num_head_kv = k.shape[head_dim]
    S_kv = k.shape[seq_dim]

    if scaling is None:
        scaling = 1.0 / math.sqrt(head_size)

    # Query group size for GQA
    if H == num_head_kv:
        query_group_size = 1
    else:
        assert H % num_head_kv == 0
        query_group_size = H // num_head_kv

    # Tile dimensions
    TILE_D = _next_power_of_2(head_size)
    OUT_TILE_D = _next_power_of_2(out_head_size)

    # Bias type encoding
    if bias_type is None or bias_type == "none":
        BIAS_TYPE = 0
    elif bias_type == "vector":
        BIAS_TYPE = 1
    elif bias_type == "matrix":
        BIAS_TYPE = 2
    elif bias_type == "alibi":
        BIAS_TYPE = 3
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")

    # Ensure tensors are contiguous
    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v

    # Create output tensor
    o = torch.empty_like(q)
    if out_head_size != head_size:
        if layout == "bnsd":
            o = torch.empty(B, H, S_qo, out_head_size, dtype=q.dtype, device=q.device)
        else:
            o = torch.empty(H, S_qo, B, out_head_size, dtype=q.dtype, device=q.device)

    # Create dummy tensors for optional inputs
    if bias is None:
        bias = torch.empty(0, dtype=q.dtype, device=q.device)
    if random_mask is None:
        random_mask = torch.empty(0, dtype=torch.int8, device=q.device)
    if q_lens is None:
        q_lens = torch.empty(0, dtype=torch.int32, device=q.device)
    if kv_lens is None:
        kv_lens = torch.empty(0, dtype=torch.int32, device=q.device)

    USE_Q_LENS = q_lens.numel() > 0
    USE_KV_LENS = kv_lens.numel() > 0
    USE_RANDOM_MASK = random_mask.numel() > 0

    # Check if autotune is enabled
    enable_autotune = is_autotune_enabled()

    if enable_autotune:
        _fmha_variant_autotune(
            torch.cuda.current_stream(),
            q,
            k,
            v,
            o,
            bias,
            random_mask,
            q_lens,
            kv_lens,
            scaling,
            S_qo,
            S_kv,
            TILE_D,
            OUT_TILE_D,
            H,
            query_group_size,
            is_causal,
            window_size,
            BIAS_TYPE,
            USE_RANDOM_MASK,
            USE_Q_LENS,
            USE_KV_LENS,
            layout,
        )
    else:
        # Use default tile sizes - optimized for H100
        gpu_capability = torch.cuda.get_device_capability()
        if gpu_capability[0] >= 9:
            TILE_M = 128
            TILE_N = 128
            occupancy = 2
        else:
            TILE_M = 64
            TILE_N = 64
            occupancy = 2

        kernel_configs = kwargs.get("kernel_configs", None)
        if kernel_configs:
            TILE_M = kernel_configs.get("TILE_M", TILE_M)
            TILE_N = kernel_configs.get("TILE_N", TILE_N)
            occupancy = kernel_configs.get("occupancy", occupancy)

        EVEN_M = (S_qo % TILE_M) == 0
        EVEN_N = (S_kv % TILE_N) == 0

        grid = (math.ceil(S_qo / TILE_M), B * H, 1)

        if layout == "bnsd":
            base_kernel = _fmha_variant_kernel_bnsd
        else:
            base_kernel = _fmha_variant_kernel_nsbd
        kernel = cached_replace_hints(base_kernel, occupancy=occupancy)

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel,
            (
                q,
                k,
                v,
                o,
                bias,
                random_mask,
                q_lens,
                kv_lens,
                scaling,
                S_qo,
                S_kv,
                TILE_D,
                OUT_TILE_D,
                H,
                TILE_M,
                TILE_N,
                query_group_size,
                is_causal,
                EVEN_M,
                EVEN_N,
                USE_Q_LENS,
                USE_KV_LENS,
                window_size,
                BIAS_TYPE,
                USE_RANDOM_MASK,
            ),
        )

    return o


@register_impl("fmha_variant", backend="cutile")
def fmha_variant_cutile(
    q,
    k,
    v,
    scaling=None,
    is_causal=True,
    q_lens=None,
    kv_lens=None,
    bias_type=None,
    bias=None,
    dropout=0.0,
    seed=torch.random.initial_seed(),
    window_size=0,
    random_mask=None,
    layout="bnsd",
    **kwargs,
):
    """CuTile implementation entry point for fmha_variant."""
    return _tile_fmha_variant(
        q=q,
        k=k,
        v=v,
        scaling=scaling,
        is_causal=is_causal,
        q_lens=q_lens,
        kv_lens=kv_lens,
        bias_type=bias_type,
        bias=bias,
        dropout=dropout,
        seed=seed,
        window_size=window_size,
        random_mask=random_mask,
        layout=layout,
        **kwargs,
    )
