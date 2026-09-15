# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl
from tilegym.logger import get_logger

from .utils import cached_replace_hints

# Module-level tune cache: (batch_size, num_heads, S_qo, S_kv, hidden_size, query_group_size, is_causal, dtype, device) -> (best_cfg, tuned_kernel)
_fmha_varlen_tune_cache: dict = {}

logger = get_logger(__name__)


INV_LOG_2 = 1.0 / math.log(2)

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


def _fmha_varlen_autotune_configs():
    """
    Iterator of autotune configurations for FMHA varlen kernel.
    Expanded search space for better performance across different problem sizes.
    """
    gpu_capability = torch.cuda.get_device_capability()

    if gpu_capability in [(12, 0), (12, 1)]:
        # sm120, sm121
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
    elif gpu_capability[0] < 9:
        # GPU capability < 9.0
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=128, TILE_N=64, num_ctas=1, occupancy=2)
    else:
        # sm100 (Blackwell) - expanded search space for better performance
        yield SimpleNamespace(TILE_M=256, TILE_N=128, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=128, TILE_N=64, num_ctas=1, occupancy=2)
        yield SimpleNamespace(TILE_M=64, TILE_N=64, num_ctas=1, occupancy=4)
        yield SimpleNamespace(TILE_M=64, TILE_N=128, num_ctas=1, occupancy=2)
        # Additional configs for better coverage
        yield SimpleNamespace(TILE_M=256, TILE_N=64, num_ctas=1, occupancy=1)
        yield SimpleNamespace(TILE_M=128, TILE_N=128, num_ctas=1, occupancy=1)


@ct.kernel
def _fmha_varlen_kernel(
    Q,
    K,
    V,
    Q_lens,  # Tensor of query lengths per batch item
    KV_lens,  # Tensor of key-value lengths per batch item
    Out,
    qk_scale: float,
    TILE_D: ConstInt,  # TILE_D = hidden_size
    H: ConstInt,
    S_QO: ConstInt,  # Max query sequence length
    S_KV: ConstInt,  # Max key-value sequence length
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    QUERY_GROUP_SIZE: ConstInt,
    CAUSAL: ConstBool,
    Q_LEN_MASK: ConstBool,
    KV_LEN_MASK: ConstBool,
):
    """
    cuTile kernel for Fused Multi-Head Attention (FMHA) with variable-length sequences.
    Supports:
    - Variable query lengths (q_lens) and key-value lengths (kv_lens) per batch item
    - Grouped Query Attention (GQA) with QUERY_GROUP_SIZE
    - Causal masking with prefix KV length handling
    """
    # Map block IDs to batch and head indices
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H

    if QUERY_GROUP_SIZE > 0:
        off_kv_h = head_idx // QUERY_GROUP_SIZE
    else:
        off_kv_h = head_idx

    # Adjust qk_scale for exp2
    qk_scale = qk_scale * INV_LOG_2

    # Load variable lengths per batch item
    if Q_LEN_MASK:
        q_len_val = ct.load(Q_lens, index=(batch_idx,), shape=(1,)).reshape(())
    else:
        q_len_val = S_QO

    if KV_LEN_MASK:
        kv_len_val = ct.load(KV_lens, index=(batch_idx,), shape=(1,)).reshape(())
    else:
        kv_len_val = S_KV

    # Compute prefix_kvlen for causal offset
    prefix_kvlen = kv_len_val - q_len_val

    # Early exit if this block is beyond the query length
    if bid_x * TILE_M >= q_len_val:
        # Fill output with zeros for this block
        acc = ct.full((1, 1, TILE_M, TILE_D), 0.0, dtype=Out.dtype)
        ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)
        return

    # Initialize offsets for current query tile (M-dimension) with prefix offset
    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)  # [TILE_M]
    offs_m_with_prefix = offs_m + prefix_kvlen
    offs_m_2d = offs_m_with_prefix[:, None]  # [TILE_M, 1]

    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)  # [TILE_N]
    offs_n_tile_2d = offs_n_tile[None, :]  # [1, TILE_N]

    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    # Load query tile for this batch, head, and M-chunk
    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)).reshape(
        (TILE_M, TILE_D)
    )  # [TILE_M, TILE_D]

    # Determine loop bounds based on causal mode
    if CAUSAL:
        # For causal attention, compute stage boundaries
        # stage_1_end: where we don't need causal masking (full blocks)
        stage_1_end = (prefix_kvlen + bid_x * TILE_M) // TILE_N * TILE_N
        # q_seq_end_cur_block: end of current query block in KV space
        q_seq_end_cur_block = prefix_kvlen + (bid_x + 1) * TILE_M

        # Determine hi based on KV_LEN_MASK
        if KV_LEN_MASK:
            hi = min(kv_len_val, q_seq_end_cur_block)
        else:
            hi = q_seq_end_cur_block

        # Mask start for causal masking
        mask_start = stage_1_end // TILE_N
        Tc = ct.cdiv(hi, TILE_N)
    else:
        # Non-causal: iterate over all KV positions
        hi = kv_len_val
        Tc = ct.cdiv(kv_len_val, TILE_N)
        mask_start = kv_len_val // TILE_N  # Only mask at the end for boundary

    # Loop over K, V blocks (N-dimension chunks)
    for j in range(0, Tc):
        k = ct.load(
            K,
            index=(batch_idx, off_kv_h, 0, j),
            shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))  # [TILE_D, TILE_N]
        qk = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k, qk)  # [TILE_M, TILE_N]

        if j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile_2d
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)

            # KV length mask (out of bound)
            if KV_LEN_MASK:
                mask = mask & (offs_n < kv_len_val)

            # Causal mask
            if CAUSAL:
                mask = mask & (offs_m_2d >= offs_n)  # [TILE_M, TILE_N]

            mask = ct.where(mask, 0.0, -math.inf)  # [TILE_M, TILE_N]
            qk = qk + mask

        # Moving qk_scale multiplication after reduce_max is to improve performance.
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [TILE_M, TILE_N]

        # Attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_M, TILE_N]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [TILE_M, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_M, 1]

        # Update m_i and l_i
        l_i = l_i * alpha + l_ij  # [TILE_M, 1]
        # Scale acc
        acc = acc * alpha  # [TILE_M, TILE_D]

        v = ct.load(
            V,
            index=(batch_idx, off_kv_h, j, 0),
            shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))  # [TILE_N, TILE_D]
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)  # [TILE_M, TILE_D]
        m_i = m_ij  # [TILE_M, 1]

    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


def _select_tile_size_for_workload(S_qo, S_kv, gpu_capability):
    """
    Select optimal tile size based on workload characteristics.
    Small S_qo with large S_kv needs a larger TILE_M
    to reduce thread block count and improve GPU utilization.
    """
    if gpu_capability[0] >= 10:
        # Blackwell (sm100)
        if S_qo <= 1024 and S_kv >= 4096:
            # Small query, large KV: use larger TILE_M to reduce parallelism overhead
            return 64, 64, 4  # Smaller tiles, higher occupancy for small workloads
        elif S_qo >= 4096:
            # Large query: use larger tiles for better memory efficiency
            return 128, 128, 2
        else:
            # Default for medium workloads
            return 128, 128, 2
    else:
        # Older GPUs
        return 64, 64, 2


def _cutile_autotune_fmha_varlen(
    stream,
    q,
    k,
    v,
    q_lens,
    kv_lens,
    o,
    sm_scale,
    hidden_size,
    num_heads,
    S_qo,
    S_kv,
    query_group_size,
    is_causal,
    Q_LEN_MASK,
    KV_LEN_MASK,
):
    batch_size = q.shape[0]
    cache_key = (batch_size, num_heads, S_qo, S_kv, hidden_size, query_group_size, is_causal, q.dtype, str(q.device))
    if cache_key not in _fmha_varlen_tune_cache:
        result = exhaustive_search(
            list(_fmha_varlen_autotune_configs()),
            stream,
            lambda cfg: (math.ceil(S_qo / cfg.TILE_M), batch_size * num_heads, 1),
            _fmha_varlen_kernel,
            lambda cfg: (
                q,
                k,
                v,
                q_lens,
                kv_lens,
                o,
                sm_scale,
                hidden_size,
                num_heads,
                S_qo,
                S_kv,
                cfg.TILE_M,
                cfg.TILE_N,
                query_group_size,
                is_causal,
                Q_LEN_MASK,
                KV_LEN_MASK,
            ),
            lambda cfg: {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy},
        )
        best_cfg = result.best.config
        _fmha_varlen_tune_cache[cache_key] = (
            best_cfg,
            _fmha_varlen_kernel.replace_hints(num_ctas=best_cfg.num_ctas, occupancy=best_cfg.occupancy),
        )
    best_cfg, tuned_kernel = _fmha_varlen_tune_cache[cache_key]
    ct.launch(
        stream,
        (math.ceil(S_qo / best_cfg.TILE_M), batch_size * num_heads, 1),
        tuned_kernel,
        (
            q,
            k,
            v,
            q_lens,
            kv_lens,
            o,
            sm_scale,
            hidden_size,
            num_heads,
            S_qo,
            S_kv,
            best_cfg.TILE_M,
            best_cfg.TILE_N,
            query_group_size,
            is_causal,
            Q_LEN_MASK,
            KV_LEN_MASK,
        ),
    )
    return o


def _tile_prefill_fmha_varlen(q, k, v, sm_scale, is_causal=True, q_lens=None, kv_lens=None, kernel_configs=None):
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    batch_size, num_heads, S_qo, hidden_size = q.shape
    _, num_head_kv, S_kv, _ = k.shape

    if num_heads == num_head_kv:
        query_group_size = 0
    else:
        assert num_heads % num_head_kv == 0
        query_group_size = num_heads // num_head_kv

    q = q.contiguous() if not q.is_contiguous() else q
    k = k.contiguous() if not k.is_contiguous() else k
    v = v.contiguous() if not v.is_contiguous() else v
    o = torch.empty_like(q)

    Q_LEN_MASK = q_lens is not None
    KV_LEN_MASK = kv_lens is not None

    # Create dummy tensors if q_lens/kv_lens are None (they won't be accessed due to mask flags)
    if q_lens is None:
        q_lens = torch.empty(batch_size, dtype=torch.int32, device=q.device)
    if kv_lens is None:
        kv_lens = torch.empty(batch_size, dtype=torch.int32, device=q.device)

    if is_autotune_disabled():
        if kernel_configs is None:
            # Dynamic tile selection based on workload
            gpu_capability = torch.cuda.get_device_capability()
            TILE_M, TILE_N, occupancy = _select_tile_size_for_workload(S_qo, S_kv, gpu_capability)
            num_ctas = None
        else:
            TILE_M = kernel_configs.get("TILE_M", 64)
            TILE_N = kernel_configs.get("TILE_N", 64)
            num_ctas = kernel_configs.get("num_ctas", None)
            occupancy = kernel_configs.get("occupancy", 2)

        grid = (math.ceil(S_qo / TILE_M), batch_size * num_heads, 1)

        hints = {}
        if num_ctas is not None:
            hints["num_ctas"] = num_ctas
        if occupancy is not None:
            hints["occupancy"] = occupancy
        kernel = cached_replace_hints(_fmha_varlen_kernel, **hints) if hints else _fmha_varlen_kernel

        ct.launch(
            torch.cuda.current_stream(),
            grid,
            kernel,
            (
                q,
                k,
                v,
                q_lens,
                kv_lens,
                o,
                sm_scale,
                hidden_size,
                num_heads,
                S_qo,
                S_kv,
                TILE_M,
                TILE_N,
                query_group_size,
                is_causal,
                Q_LEN_MASK,
                KV_LEN_MASK,
            ),
        )
        return o
    return _cutile_autotune_fmha_varlen(
        torch.cuda.current_stream(),
        q,
        k,
        v,
        q_lens,
        kv_lens,
        o,
        sm_scale,
        hidden_size,
        num_heads,
        S_qo,
        S_kv,
        query_group_size,
        is_causal,
        Q_LEN_MASK,
        KV_LEN_MASK,
    )


@register_impl("fmha_varlen", backend="cutile")
def tile_fmha_varlen(
    q,
    k,
    v,
    scaling=None,
    is_causal=True,
    q_lens=None,
    kv_lens=None,
    **kwargs,
):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1))
    kernel_configs = kwargs.get("kernel_configs", None)
    o = _tile_prefill_fmha_varlen(q, k, v, scaling, is_causal, q_lens, kv_lens, kernel_configs)
    return o
