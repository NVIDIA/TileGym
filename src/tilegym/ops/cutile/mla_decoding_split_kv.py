# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd

from tilegym.backend import register_impl

from .mla_tile_config import plan_split_kv
from .mla_tile_config import select_tile_h
from .splitk_reduce import splitk_reduce
from .utils import next_power_of_2

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

INV_LOG_2 = 1.0 / math.log(2)


@ct.kernel(occupancy=2)
def _naive_absorb_mla_transpose_kernel(
    Q,
    QPE,
    K,
    V,
    KPE,
    Att_Out,  # Output intermediate attention results for reduction
    LSE_Out,  # Output intermediate e_max and e_sum for reduction [B, NUM_HEADS, NUM_KV_SPLITS]
    sm_scale: float,
    B: int,
    NUM_HEADS: int,
    S_kv: int,
    NUM_KV_SPLITS: ConstInt,
    KV_LEN_PER_SPLIT: ConstInt,
    TILE_D: ConstInt,
    TILE_H: ConstInt,
    TILE_N: ConstInt,
    TILE_KPE: ConstInt,
    EVEN_N: ConstBool,
):
    pid_x = ct.bid(0)
    batch_idx = ct.bid(1)
    tile_idx = ct.bid(2)  # Split dimension
    qk_scale = sm_scale * INV_LOG_2

    # Initialize accumulation variables
    m_prev = ct.full((TILE_H,), -math.inf, dtype=ct.float32)
    l_prev = ct.full((TILE_N, TILE_H), 1.0, dtype=ct.float32)
    acc = ct.zeros((TILE_D, TILE_H), dtype=ct.float32)

    # Load q with latency hints for pipeline optimization
    zero_pad = ct.PaddingMode.ZERO
    q = ct.load(
        Q,
        index=(batch_idx, 0, pid_x),
        shape=(1, TILE_D, TILE_H),
        order=(0, 2, 1),
        allow_tma=True,
        padding_mode=zero_pad,
        latency=2,
    )
    q = ct.reshape(q, (TILE_D, TILE_H))
    qpe = ct.load(
        QPE,
        index=(batch_idx, 0, pid_x),
        shape=(1, TILE_KPE, TILE_H),
        order=(0, 2, 1),
        allow_tma=True,
        padding_mode=zero_pad,
        latency=2,
    )
    qpe = ct.reshape(qpe, (TILE_KPE, TILE_H))

    # Calculate split range (split-specific logic)
    split_kv_start = KV_LEN_PER_SPLIT * tile_idx
    split_kv_end = split_kv_start + KV_LEN_PER_SPLIT
    if split_kv_end > S_kv:
        split_kv_end = S_kv

    # Loop over key-value pairs and update accumulator
    cnt = split_kv_start // TILE_N
    mask_start = S_kv // TILE_N * TILE_N
    offs_n = ct.arange(TILE_N, dtype=ct.int32)
    for curr_n in range(split_kv_start, split_kv_end, TILE_N):
        # Load key and compute Q@K^T with latency hints
        k = ct.load(
            K,
            index=(batch_idx, cnt, 0),
            shape=(1, TILE_N, TILE_D),
            allow_tma=True,
            padding_mode=zero_pad,
            latency=2,
        )
        k = ct.reshape(k, (TILE_N, TILE_D))
        qk = ct.full((TILE_N, TILE_H), 0.0, dtype=ct.float32)
        qk = ct.mma(k, q, qk)

        # Load key position encoding and compute QPE@KPE^T with latency hints
        kpe = ct.load(
            KPE,
            index=(batch_idx, cnt, 0),
            shape=(1, TILE_N, TILE_KPE),
            allow_tma=True,
            padding_mode=zero_pad,
            latency=2,
        )
        kpe = ct.reshape(kpe, (TILE_N, TILE_KPE))
        qk = ct.mma(kpe, qpe, qk)

        # Apply mask if needed
        if not EVEN_N and curr_n >= mask_start:
            mask = (curr_n + offs_n[:, None]) < S_kv
            qk = ct.where(mask, qk, -1.0e6)

        # Apply scaling and compute attention scores
        m_ij = ct.maximum(m_prev, ct.max(qk, 0) * qk_scale)
        qk = qk * qk_scale - m_ij[None, :]

        # Compute attention weights and update running statistics
        p = ct.exp2(qk)
        alpha = ct.exp2(m_prev - m_ij)
        l_prev = l_prev * alpha[None, :] + p
        acc = acc * alpha[None, :]

        # Load value and compute attention @ value with latency hints
        v = ct.load(
            V,
            index=(cnt, batch_idx, 0),
            shape=(TILE_N, 1, TILE_D),
            order=(1, 0, 2),
            allow_tma=True,
            padding_mode=zero_pad,
            latency=2,
        )
        v = ct.reshape(v, (TILE_N, TILE_D))
        v = ct.transpose(v, axis0=1, axis1=0)
        p = ct.astype(p, Q.dtype)
        acc = ct.mma(v, p, acc)
        m_prev = m_ij
        cnt += 1

    # Finalize attention computation
    l_prev = ct.sum(l_prev, 0)  # [TILE_N, TILE_H] -> [TILE_H]
    acc = ct.truediv(acc, (l_prev[None, :]), flush_to_zero=True, rounding_mode=RMd.APPROX)  # [TILE_D, TILE_H]
    l_prev = m_prev + ct.log2(l_prev)

    # Store results (adapted for split-kv format) with latency hints
    acc = ct.astype(acc, Att_Out.dtype)
    acc = ct.transpose(acc, axis0=1, axis1=0)
    acc = ct.reshape(acc, (1, TILE_H, 1, TILE_D))
    ct.store(
        Att_Out,
        index=(batch_idx, pid_x, tile_idx, 0),
        tile=acc,
        allow_tma=True,
        latency=2,
    )

    # Store log sum exp for this tile with latency hint
    idx_head = pid_x * TILE_H + ct.arange(TILE_H, dtype=ct.int32)
    ct.scatter(
        LSE_Out,
        (batch_idx, idx_head, tile_idx),
        l_prev,
        check_bounds=True,
        latency=2,
    )


class _MlaDecodingSplitKvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, QPE, KV, KPE, sm_scale, kv_len_per_split=None, kernel_configs=None):
        """
        MLA Decoding with Split-KV forward pass

        Args:
            Q: Query tensor [B, NUM_HEADS, TILE_D]
            QPE: Query positional embedding [B, NUM_HEADS, TILE_KPE]
            KV: Key-Value tensor [B, S_kv, TILE_D]
            KPE: Key positional embedding [B, S_kv, TILE_KPE]
            sm_scale: Softmax scale factor
            kv_len_per_split: kv_len_per_split
            kernel_configs: optional overrides, currently ``TILE_H``

        Returns:
            O: Output tensor [B, S_qo, TILE_D]
        """
        # Get dimensions
        B, NUM_HEADS, TILE_D = Q.shape
        _, S_kv, _ = KV.shape
        TILE_KPE = KPE.shape[2]

        assert TILE_D == next_power_of_2(TILE_D)
        assert TILE_KPE == next_power_of_2(TILE_KPE)

        TILE_N = 128

        NUM_SMS = torch.cuda.get_device_properties(Q.device).multi_processor_count
        CC_MAJOR = torch.cuda.get_device_capability(Q.device)[0]

        # A TILE_H override has to reach the planner, not just the launch: the
        # split length is sized for a specific tile.
        override_tile_h = None if kernel_configs is None else kernel_configs.get("TILE_H")

        if kv_len_per_split is None:
            # Tile and split length are chosen together: the grid is
            # ceil(H / TILE_H) * B * NUM_KV_SPLITS.
            tile_h, kv_len_per_split = plan_split_kv(
                NUM_HEADS, S_kv, B, TILE_N, NUM_SMS, CC_MAJOR, force_tile_h=override_tile_h
            )
        else:
            tile_h = select_tile_h(
                NUM_HEADS,
                S_kv,
                B,
                (S_kv + kv_len_per_split - 1) // kv_len_per_split,
                NUM_SMS,
                CC_MAJOR,
            )

        assert kv_len_per_split == next_power_of_2(kv_len_per_split)
        assert kv_len_per_split >= TILE_N
        NUM_KV_SPLITS = (S_kv + kv_len_per_split - 1) // kv_len_per_split

        # TILE_N is deliberately not overridable: it is baked into the
        # kv_len_per_split floor and the assert above, so an override would sail
        # past a stale check and read into the next split's KV range.
        default_configs = {"TILE_H": tile_h}
        if kernel_configs is None:
            kernel_configs = default_configs
        else:
            kernel_configs = {**default_configs, **kernel_configs}
        TILE_H = kernel_configs.get("TILE_H")

        # Allocate intermediate results
        device = Q.device
        Att_Out = torch.empty(
            (B, NUM_HEADS, NUM_KV_SPLITS, TILE_D),
            device=device,
            dtype=Q.dtype,
        )
        LSE_Out = torch.empty(
            (B, NUM_HEADS, NUM_KV_SPLITS),
            device=device,
            dtype=torch.float32,
        )

        # Prepare final outputs
        O = torch.empty_like(Q)

        # Launch split kernel
        grid_split = (
            (NUM_HEADS + TILE_H - 1) // TILE_H,
            B,
            NUM_KV_SPLITS,
        )
        ct.launch(
            torch.cuda.current_stream(),
            grid_split,
            _naive_absorb_mla_transpose_kernel,
            (
                Q,
                QPE,
                KV,
                KV,
                KPE,
                Att_Out,
                LSE_Out,
                sm_scale,
                B,
                NUM_HEADS,
                S_kv,
                NUM_KV_SPLITS,
                kv_len_per_split,
                TILE_D,
                TILE_H,
                TILE_N,
                TILE_KPE,
                (S_kv % TILE_N == 0),
            ),
        )

        # Reduce kernel splitk results
        splitk_reduce(Att_Out, LSE_Out, O, S_kv)

        return O

    @staticmethod
    def backward(ctx, do, dl):
        raise NotImplementedError("MLA Decoding Split-KV backward is not implemented yet")


@register_impl("mla_decoding_split_kv", backend="cutile")
def mla_decoding_split_kv(q, qpe, kv, kpe, sm_scale=None, kv_len_per_split=None, **kwargs):
    """
    MLA Decoding with Split-KV interface

    Args:
        q: Query tensor [batch_size, seq_len, head_dim]
        qpe: Query positional embedding [batch_size, seq_len, kpe_dim]
        kv: Key-Value tensor [batch_size, kv_seq_len, head_dim]
        kpe: Key positional embedding [batch_size, kv_seq_len, kpe_dim]
        sm_scale: Softmax scale (defaults to 1/sqrt(head_dim + kpe_dim))
        **kwargs: Additional arguments for backend-specific configurations,
            including kernel_configs if needed (TILE_H)
        kv_len_per_split: kv_len_per_split

    Returns:
        o: Output tensor [batch_size, seq_len, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / (math.sqrt(q.size(-1) + qpe.size(-1)))

    o = _MlaDecodingSplitKvFunction.apply(
        q, qpe, kv, kpe, sm_scale, kv_len_per_split, kwargs.get("kernel_configs", None)
    )
    return o
