# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import os
from types import SimpleNamespace

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl
from tilegym.experimental import experimental_kernel
from tilegym.logger import get_logger

from ..utils import next_power_of_2

logger = get_logger(__name__)

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)

# Candidate tile sizes for autotuning (must be powers of 2).
_SPARSE_MLA_TILE_HS = [1, 2, 4, 8, 16, 32, 64]
_SPARSE_MLA_TILE_NS = [16, 32, 64, 128]


def _should_disable_autotune():
    """Check if autotuning should be disabled (for testing).

    Set DISABLE_AUTOTUNE=1 to skip autotuning and use the first config.
    This is useful for CI testing where autotuning can cause timeouts.
    """
    return os.environ.get("DISABLE_AUTOTUNE", "0") == "1"


def _is_power_of_2(x):
    return x > 0 and (x & (x - 1)) == 0


def _validate_sparse_mla_config(cfg, topk, H, query_group_size, *, explicit=False):
    """Validate a (TILE_H, TILE_N) config. Raises ValueError/AssertionError if invalid.

    Args:
        cfg: SimpleNamespace with TILE_H, TILE_N attributes.
        topk, H, query_group_size: problem-size parameters.
        explicit: True when cfg came from caller-provided kernel_configs.
            Enables additional checks (required keys, types).
    """
    # Completeness checks (explicit configs only)
    if explicit:
        if not hasattr(cfg, "TILE_H"):
            raise ValueError("kernel_configs missing required key 'TILE_H'")
        if not hasattr(cfg, "TILE_N"):
            raise ValueError("kernel_configs missing required key 'TILE_N'")
        if not isinstance(cfg.TILE_H, int):
            raise TypeError(f"TILE_H must be int, got {type(cfg.TILE_H).__name__}")
        if not isinstance(cfg.TILE_N, int):
            raise TypeError(f"TILE_N must be int, got {type(cfg.TILE_N).__name__}")

    TILE_H = cfg.TILE_H
    TILE_N = cfg.TILE_N

    # Power-of-2 checks
    assert _is_power_of_2(TILE_H), f"TILE_H ({TILE_H}) must be a power of 2"
    assert _is_power_of_2(TILE_N), f"TILE_N ({TILE_N}) must be a power of 2"

    # GQA / head-group alignment
    if query_group_size == 0:
        assert TILE_H == 1, f"TILE_H must be 1 when query_group_size == 0 (1:1 head mapping), got {TILE_H}"
    else:
        assert TILE_H <= query_group_size, f"TILE_H ({TILE_H}) must not exceed query_group_size ({query_group_size})"
        assert query_group_size % TILE_H == 0, (
            f"query_group_size ({query_group_size}) must be divisible by TILE_H ({TILE_H}) "
            f"so that tile boundaries align with group boundaries"
        )

    # Divisibility
    assert H % TILE_H == 0, f"H ({H}) must be divisible by TILE_H ({TILE_H})"
    assert topk % TILE_N == 0, f"topk ({topk}) must be divisible by TILE_N ({TILE_N})"


def _sparse_mla_autotune_configs(topk, H, query_group_size):
    """Yield valid (TILE_H, TILE_N) configs for the search space."""
    for tile_h in _SPARSE_MLA_TILE_HS:
        # query_group_size == 0 means 1:1 head mapping (H_kv == H) — no sharing
        if query_group_size == 0:
            if tile_h != 1:
                continue
        else:
            if tile_h > query_group_size:
                continue
            # Tile boundaries must align with group boundaries
            if query_group_size % tile_h != 0:
                continue
        if H % tile_h != 0:
            continue
        for tile_n in _SPARSE_MLA_TILE_NS:
            if topk % tile_n == 0:
                yield SimpleNamespace(TILE_H=tile_h, TILE_N=tile_n)


@experimental_kernel
@ct.kernel
def sparse_mla_fwd_kernel(
    Q,  # [B, H, S, D]
    K,  # [B, H_kv, S_kv, D]
    V,  # [B, H_kv, S_kv, D]
    Indices,  # [B, S, H_kv, topk]
    QPE,  # [B, H, S, D_PE]
    KPE,  # [B, 1, S_kv, D_PE]
    Out,  # [B, H, S, D]
    qk_scale: float,
    TILE_D: ConstInt,  # D (head dim for nope, also V dim)
    TILE_KPE: ConstInt,  # D_PE (head dim for rope)
    H: ConstInt,  # total query heads
    TILE_N: ConstInt,  # index block size (e.g. 64)
    NI: int,  # number of index blocks = topk / TILE_N
    QUERY_GROUP_SIZE: ConstInt,  # GQA group size, 0 for 1:1
    TILE_H: ConstInt,  # heads per block
    NUM_H_BLOCKS: ConstInt,  # H // TILE_H (for bid_y decode)
):
    bid_x = ct.bid(0)  # query position s_i
    bid_y = ct.bid(1)  # batch_idx * NUM_H_BLOCKS + h_block_idx

    batch_idx = bid_y // NUM_H_BLOCKS
    h_block_idx = bid_y % NUM_H_BLOCKS
    h_start = h_block_idx * TILE_H

    # GQA head mapping: all TILE_H heads map to the same KV head
    if QUERY_GROUP_SIZE > 0:
        off_kv_h = h_start // QUERY_GROUP_SIZE
    else:
        off_kv_h = h_start  # 1:1 mapping (TILE_H must be 1 here)

    s_i = bid_x

    qk_scale = qk_scale * INV_LOG_2

    # Initialize online softmax accumulators
    m_i = ct.full((TILE_H,), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_H,), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_H, TILE_D), 0.0, dtype=ct.float32)

    # Load Q tile: TILE_H consecutive heads at position s_i
    q = ct.load(Q, index=(batch_idx, h_block_idx, s_i, 0), shape=(1, TILE_H, 1, TILE_D))
    q = ct.reshape(q, (TILE_H, TILE_D))  # [TILE_H, D]

    # Load QPE tile: TILE_H consecutive heads at position s_i
    qpe = ct.load(QPE, index=(batch_idx, h_block_idx, s_i, 0), shape=(1, TILE_H, 1, TILE_KPE))
    qpe = ct.reshape(qpe, (TILE_H, TILE_KPE))  # [TILE_H, D_PE]

    # Main loop over index blocks
    for i_i in range(0, NI):
        # Load TILE_N indices (regular contiguous load)
        # ct.load uses tile-space indexing: index is multiplied by shape,
        # so i_i (tile index) maps to elements [i_i*TILE_N : (i_i+1)*TILE_N]
        indices_tile = ct.load(
            Indices,
            index=(batch_idx, s_i, off_kv_h, i_i),
            shape=(1, 1, 1, TILE_N),
        )
        indices_tile = ct.reshape(indices_tile, (TILE_N,))  # [TILE_N]

        # Build broadcast index tiles
        s_idx = ct.expand_dims(indices_tile, 1)  # [TILE_N, 1]
        d_idx = ct.expand_dims(ct.arange(TILE_D, dtype=ct.int32), 0)  # [1, TILE_D]
        dpe_idx = ct.expand_dims(ct.arange(TILE_KPE, dtype=ct.int32), 0)  # [1, TILE_KPE]

        # Gather K at indexed positions (shared across TILE_H heads)
        gathered_k = ct.gather(K, (batch_idx, off_kv_h, s_idx, d_idx))  # [TILE_N, TILE_D]

        # Gather V at same indices (shared across TILE_H heads)
        gathered_v = ct.gather(V, (batch_idx, off_kv_h, s_idx, d_idx))  # [TILE_N, TILE_D]

        # Gather KPE at same indices (shared across TILE_H heads)
        gathered_kpe = ct.gather(KPE, (batch_idx, 0, s_idx, dpe_idx))  # [TILE_N, TILE_KPE]

        # Compute QK scores
        # Nope: [TILE_H, D] × [D, TILE_N] → [TILE_H, TILE_N]
        gathered_k_t = ct.permute(gathered_k, (1, 0))  # [D, TILE_N]
        qk = ct.full((TILE_H, TILE_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, gathered_k_t, qk)  # [TILE_H, TILE_N]

        # Rope: [TILE_H, D_PE] × [D_PE, TILE_N] → [TILE_H, TILE_N]
        gathered_kpe_t = ct.permute(gathered_kpe, (1, 0))  # [D_PE, TILE_N]
        qk = ct.mma(qpe, gathered_kpe_t, qk)  # [TILE_H, TILE_N]

        # Apply causal mask (shared across TILE_H heads, broadcasts)
        valid_mask = indices_tile <= s_i  # [TILE_N] bool
        valid_mask_2d = ct.expand_dims(valid_mask, 0)  # [1, TILE_N]
        qk = ct.where(valid_mask_2d, qk, -1.0e6)  # broadcasts to [TILE_H, TILE_N]

        # Online softmax update
        m_ij = ct.maximum(m_i, ct.max(qk, axis=-1) * qk_scale)  # [TILE_H]
        qk = qk * qk_scale - m_ij[:, None]  # [TILE_H, TILE_N]

        p = ct.exp2(qk, flush_to_zero=True)  # [TILE_H, TILE_N]
        # Re-apply mask: zero out masked positions so they contribute nothing
        # to l_i or acc. This is critical when an entire index block has no
        # causal-valid entries (all indices > s_i), which cannot happen in
        # dense attention but is common in sparse attention with shuffled indices.
        p = ct.where(valid_mask_2d, p, 0.0)  # broadcasts to [TILE_H, TILE_N]
        l_ij = ct.sum(p, axis=-1)  # [TILE_H]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [TILE_H]

        l_i = l_i * alpha + l_ij  # [TILE_H]
        acc = acc * alpha[:, None]  # [TILE_H, TILE_D]

        # Accumulate output: P @ V
        # [TILE_H, TILE_N] × [TILE_N, TILE_D] → [TILE_H, TILE_D]
        p_cast = ct.astype(p, Q.dtype)
        acc = ct.mma(p_cast, gathered_v, acc)  # [TILE_H, TILE_D]
        m_i = m_ij

    # Final rescaling
    acc = ct.truediv(acc, l_i[:, None], flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = ct.reshape(acc, (1, TILE_H, 1, TILE_D))
    acc = ct.astype(acc, Out.dtype)
    ct.store(Out, index=(batch_idx, h_block_idx, s_i, 0), tile=acc)


def _launch_sparse_mla_fwd(
    stream, q, k, v, indices, qpe, kpe, o, sm_scale, D, D_PE, H, S, topk, query_group_size, kernel_configs=None
):
    """Launch sparse MLA forward kernel.

    Three mutually exclusive config selection modes (no fallback between paths):
      Path 1: Explicit kernel_configs — validated and launched directly.
      Path 2: DISABLE_AUTOTUNE=1 — first valid config from search space.
      Path 3: Autotune — ct_experimental.autotune_launch over search space.
    """
    B = q.shape[0]

    def _launch_with_cfg(cfg):
        """Launch kernel with a validated config."""
        NI = topk // cfg.TILE_N
        NUM_H_BLOCKS = H // cfg.TILE_H
        grid = (S, B * NUM_H_BLOCKS, 1)
        ct.launch(
            stream,
            grid,
            sparse_mla_fwd_kernel,
            (
                q,
                k,
                v,
                indices,
                qpe,
                kpe,
                o,
                sm_scale,
                D,
                D_PE,
                H,
                cfg.TILE_N,
                NI,
                query_group_size,
                cfg.TILE_H,
                NUM_H_BLOCKS,
            ),
        )

    # Path 1: Explicit kernel_configs — highest precedence.
    if kernel_configs is not None:
        cfg = SimpleNamespace(**kernel_configs)
        _validate_sparse_mla_config(cfg, topk, H, query_group_size, explicit=True)
        _launch_with_cfg(cfg)

    # Path 2: DISABLE_AUTOTUNE=1 — use first valid config from search space.
    elif _should_disable_autotune():
        configs = list(_sparse_mla_autotune_configs(topk, H, query_group_size))
        assert len(configs) > 0, (
            f"No valid (TILE_H, TILE_N) config for topk={topk}, H={H}, query_group_size={query_group_size}"
        )
        cfg = configs[0]
        _validate_sparse_mla_config(cfg, topk, H, query_group_size)
        _launch_with_cfg(cfg)

    # Path 3: Autotune — search over all valid (TILE_H, TILE_N) pairs.
    else:
        ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (S, B * (H // cfg.TILE_H), 1),
            kernel=sparse_mla_fwd_kernel,
            args_fn=lambda cfg: (
                q,
                k,
                v,
                indices,
                qpe,
                kpe,
                o,
                sm_scale,
                D,
                D_PE,
                H,
                cfg.TILE_N,
                topk // cfg.TILE_N,
                query_group_size,
                cfg.TILE_H,
                H // cfg.TILE_H,
            ),
            search_space=lambda: _sparse_mla_autotune_configs(topk, H, query_group_size),
            max_iter=20,
        )
    return o


class _sparse_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, indices, qpe, kpe, sm_scale, is_causal, kernel_configs):
        assert is_causal, "CuTile sparse_mla only supports is_causal=True"

        B, H, S, D = q.shape
        assert k.shape == v.shape
        num_head_kv = k.shape[1]
        S_kv = k.shape[2]
        D_PE = qpe.shape[3]

        _, _, idx_kv_group, topk = indices.shape
        assert idx_kv_group == num_head_kv
        assert indices.dtype == torch.int32
        assert topk <= S_kv, f"topk ({topk}) must not exceed S_kv ({S_kv})"

        # GQA mapping — identical to mla.py
        if H == num_head_kv:
            query_group_size = 0
        else:
            assert H % num_head_kv == 0
            query_group_size = int(H / num_head_kv)

        o = torch.empty_like(q)

        _launch_sparse_mla_fwd(
            torch.cuda.current_stream(),
            q,
            k,
            v,
            indices,
            qpe,
            kpe,
            o,
            sm_scale,
            D,
            D_PE,
            H,
            S,
            topk,
            query_group_size,
            kernel_configs,
        )
        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("Backward pass is not implemented for CuTile sparse_mla")


def tile_sparse_mla(q, k, v, indices, qpe, kpe, is_causal=True, scaling=None, **kwargs):
    if scaling is None:
        scaling = 1.0 / math.sqrt(q.size(-1) + qpe.size(-1))

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    indices = indices.contiguous()
    qpe = qpe.contiguous()
    kpe = kpe.contiguous()

    kernel_configs = kwargs.get("kernel_configs")

    return _sparse_attention.apply(q, k, v, indices, qpe, kpe, scaling, is_causal, kernel_configs)


register_impl("sparse_mla", "cutile")(tile_sparse_mla)
