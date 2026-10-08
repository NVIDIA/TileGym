# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Skinny TN GEMM (decode GEMV) cuTile kernel for SM120: C[M,N] = A[M,K] @ B[N,K]^T.

SM120 (RTX PRO Blackwell) variant of the decode-time skinny projection kernel.
Multi-path cuTile implementation for M <= 64: plain tensor-core GEMM, staggered
K-loop GEMM, deterministic split-K (fp32 partials + fixed-order reduce), and
M=1 CUDA-core GEMV paths, selected by a per-shape tuned table with a generic
fallback. All kernels use single-CTA launches (no cluster features), matching SM120's
capabilities.

Validated on RTX PRO 6000: 26/26 workload rows correct, geomean 1.08x over the
cuBLAS baseline.
"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
_ZP = ct.PaddingMode.ZERO


@ct.kernel
def _gemv_auto(a, b, c, tn: ConstInt, tk: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn,), 0.0, dtype=ct.float32)
    num_k_tiles = ct.cdiv(a.shape[1], tk)
    for k in range(num_k_tiles):
        a_tile = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=zero).astype(ct.float32)
        b_tile = ct.load(b, index=(bid_n, k), shape=(tn, tk), padding_mode=zero).astype(ct.float32)
        accumulator = accumulator + ct.sum(b_tile * a_tile, axis=1)
    ct.store(c, index=(0, bid_n), tile=ct.reshape(accumulator, (1, tn)).astype(c.dtype))


@ct.kernel
def _gemm_auto(a, b, c, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    num_k_tiles = ct.cdiv(a.shape[1], tk)
    for k in range(num_k_tiles):
        a_tile = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=zero)
        b_tile = ct.load(b, index=(k, bid_n), shape=(tk, tn), order=(1, 0), padding_mode=zero)
        accumulator = ct.mma(a_tile, b_tile, accumulator)
    ct.store(c, index=(0, bid_n), tile=accumulator.astype(c.dtype))


@ct.kernel(occupancy=2)
def _gemv_kfirst(a, b, c, tn: ConstInt, tk: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn,), 0.0, dtype=ct.float32)
    num_k_tiles = ct.num_tiles(b, axis=1, shape=(tn, tk))
    for k in range(num_k_tiles):
        a_tile = ct.load(a, index=(k, 0), shape=(tk, 1), order=(1, 0), padding_mode=zero).astype(ct.float32)
        b_tile = ct.load(b, index=(k, bid_n), shape=(tk, tn), order=(1, 0), padding_mode=zero).astype(ct.float32)
        accumulator = accumulator + ct.sum(b_tile * a_tile, axis=0)
    ct.store(c, index=(0, bid_n), tile=ct.reshape(accumulator, (1, tn)).astype(c.dtype))


@ct.kernel
def _gemm_bnatural(a, b, c, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn, tm), 0.0, dtype=ct.float32)
    num_k_tiles = ct.num_tiles(b, axis=1, shape=(tn, tk))
    for k in range(num_k_tiles):
        b_tile = ct.load(b, index=(bid_n, k), shape=(tn, tk), padding_mode=zero, latency=10)
        a_tile = ct.load(a, index=(k, 0), shape=(tk, tm), order=(1, 0), padding_mode=zero, latency=2)
        accumulator = ct.mma(b_tile, a_tile, accumulator)
    ct.store(c, index=(bid_n, 0), tile=accumulator.astype(c.dtype), order=(1, 0))


@ct.kernel
def _gemm_bnatural_stagger(a, b, c, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn, tm), 0.0, dtype=ct.float32)
    num_k_tiles = ct.num_tiles(b, axis=1, shape=(tn, tk))
    offset = (bid_n * 11) % num_k_tiles
    for i in range(num_k_tiles):
        k = (i + offset) % num_k_tiles
        a_tile = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=zero, latency=1)
        b_tile = ct.load(b, index=(bid_n, k), shape=(tn, tk), padding_mode=zero, latency=10)
        accumulator = ct.mma(b_tile, ct.transpose(a_tile), accumulator)
    ct.store(c, index=(bid_n, 0), tile=accumulator.astype(c.dtype), order=(1, 0))


@ct.kernel
def _gemm_split_auto(a, b, partials, tm: ConstInt, tn: ConstInt, tk: ConstInt, split_k: ConstInt):
    bid_n = ct.bid(0)
    split_id = ct.bid(1)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    num_k_tiles = ct.cdiv(a.shape[1], tk)
    for k in range(split_id, num_k_tiles, split_k):
        a_tile = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=zero)
        b_tile = ct.load(b, index=(k, bid_n), shape=(tk, tn), order=(1, 0), padding_mode=zero)
        accumulator = ct.mma(a_tile, b_tile, accumulator)
    ct.store(partials, index=(split_id, bid_n), tile=accumulator)


@ct.kernel
def _reduce_flat(partials, c, split_k: ConstInt, tm: ConstInt, rn: ConstInt):
    bid_n = ct.bid(0)
    partial = ct.load(partials, index=(0, bid_n), shape=(split_k * tm, rn), padding_mode=ct.PaddingMode.ZERO)
    total = ct.sum(ct.reshape(partial, (split_k, tm, rn)), axis=0)
    ct.store(c, index=(0, bid_n), tile=total.astype(c.dtype))


@ct.kernel(occupancy=2)
def _gemv_split_fused(a, b, c, partials, counters, tn: ConstInt, tk: ConstInt, split_k: ConstInt):
    bid_n = ct.bid(0)
    split_id = ct.bid(1)
    zero = ct.PaddingMode.ZERO
    num_k_tiles = ct.cdiv(b.shape[1], tk)
    accumulator = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(split_id, num_k_tiles, split_k):
        a_tile = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=zero, latency=2).astype(ct.float32)
        b_tile = ct.load(b, index=(bid_n, k), shape=(tn, tk), padding_mode=zero, latency=10).astype(ct.float32)
        accumulator = accumulator + ct.sum(b_tile * a_tile, axis=1)
    ct.store(partials, index=(split_id, bid_n), tile=ct.reshape(accumulator, (1, tn)))
    if ct.atomic_add(counters, bid_n, 1, memory_order=ct.MemoryOrder.ACQ_REL) == split_k - 1:
        partial = ct.load(partials, index=(0, bid_n), shape=(32, tn), padding_mode=zero, latency=10)
        total = ct.sum(partial, axis=0)
        ct.store(c, index=(0, bid_n), tile=ct.reshape(total.astype(c.dtype), (1, tn)))
        ct.atomic_xchg(counters, bid_n, 0, memory_order=ct.MemoryOrder.RELEASE)


@ct.kernel(num_ctas=1, occupancy=4)
def _gemv_direct(a, b, c, tn: ConstInt, tk: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn,), 0.0, dtype=ct.float32)
    num_k_tiles = ct.cdiv(a.shape[1], tk)
    for k in range(num_k_tiles):
        a_tile = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=zero, latency=2).astype(ct.float32)
        b_tile = ct.load(b, index=(bid_n, k), shape=(tn, tk), padding_mode=zero, latency=10).astype(ct.float32)
        accumulator = accumulator + ct.sum(b_tile * a_tile, axis=1)
    output = ct.reshape(accumulator, (1, tn)).astype(c.dtype)
    ct.store(c, index=(0, bid_n), tile=output, latency=1)


@ct.kernel(num_ctas=1, occupancy=2)
def _gemv_occ2(a, b, c, tn: ConstInt, tk: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn,), 0.0, dtype=ct.float32)
    num_k_tiles = ct.cdiv(a.shape[1], tk)
    for k in range(num_k_tiles):
        a_tile = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=zero, latency=2).astype(ct.float32)
        b_tile = ct.load(b, index=(bid_n, k), shape=(tn, tk), padding_mode=zero, latency=10).astype(ct.float32)
        accumulator = accumulator + ct.sum(b_tile * a_tile, axis=1)
    ct.store(c, index=(0, bid_n), tile=ct.reshape(accumulator, (1, tn)).astype(c.dtype))


@ct.kernel
def _gemv_stagger(a, b, c, tn: ConstInt, tk: ConstInt, mul: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn,), 0.0, dtype=ct.float32)
    num_k_tiles = ct.cdiv(a.shape[1], tk)
    offset = (bid_n * mul) % num_k_tiles
    for kk in range(num_k_tiles):
        k = kk + offset
        if k >= num_k_tiles:
            k = k - num_k_tiles
        a_tile = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=zero).astype(ct.float32)
        b_tile = ct.load(b, index=(bid_n, k), shape=(tn, tk), padding_mode=zero).astype(ct.float32)
        accumulator = accumulator + ct.sum(b_tile * a_tile, axis=1)
    ct.store(c, index=(0, bid_n), tile=ct.reshape(accumulator, (1, tn)).astype(c.dtype))


@ct.kernel(occupancy=2)
def _gemv_kstagger(a, b, c, tn: ConstInt, tk: ConstInt, mul: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn,), 0.0, dtype=ct.float32)
    num_k_tiles = ct.num_tiles(b, axis=1, shape=(tn, tk))
    offset = (bid_n * mul) % num_k_tiles
    for kk in range(num_k_tiles):
        k = kk + offset
        if k >= num_k_tiles:
            k = k - num_k_tiles
        a_tile = ct.load(a, index=(k, 0), shape=(tk, 1), order=(1, 0), padding_mode=zero).astype(ct.float32)
        b_tile = ct.load(b, index=(k, bid_n), shape=(tk, tn), order=(1, 0), padding_mode=zero).astype(ct.float32)
        accumulator = accumulator + ct.sum(b_tile * a_tile, axis=0)
    ct.store(c, index=(0, bid_n), tile=ct.reshape(accumulator, (1, tn)).astype(c.dtype))


@ct.kernel
def _gemm_bnatural_stagger_p(a, b, c, tm: ConstInt, tn: ConstInt, tk: ConstInt, mul: ConstInt):
    bid_n = ct.bid(0)
    zero = ct.PaddingMode.ZERO
    accumulator = ct.full((tn, tm), 0.0, dtype=ct.float32)
    num_k_tiles = ct.cdiv(a.shape[1], tk)
    offset = (bid_n * mul) % num_k_tiles
    for kk in range(num_k_tiles):
        k = kk + offset
        if k >= num_k_tiles:
            k = k - num_k_tiles
        b_tile = ct.load(b, index=(bid_n, k), shape=(tn, tk), padding_mode=zero, latency=10)
        a_tile = ct.load(a, index=(k, 0), shape=(tk, tm), order=(1, 0), padding_mode=zero, latency=2)
        accumulator = ct.mma(b_tile, a_tile, accumulator)
    ct.store(c, index=(bid_n, 0), tile=accumulator.astype(c.dtype), order=(1, 0))


# Per-shape tuned launch table.  entry = (tag, tm, tn, tk, split_k[, rn])
# Tags:  0 _gemv_auto   1 _gemv_kfirst   2 _gemv_direct   3 _gemm_auto
#        4 _gemm_bnatural   5 _gemm_bnatural_stagger   6 _gemv_stagger
#        7 _gemv_kstagger   8 _gemm_bnatural_stagger_p   10 _gemv_occ2
#       13 _gemm_split_auto + _reduce_flat   15 _gemv_split_fused
_CFG = {
    (1, 48, 5120): (10, 1, 1, 2048, 1),
    (1, 1024, 4096): (2, 1, 8, 512, 1),
    (1, 1024, 5120): (15, 1, 64, 256, 20),
    (1, 4096, 1024): (6, 1, 32, 256, 1, 3),
    (1, 5120, 6144): (6, 1, 32, 256, 1, 11),
    (1, 5120, 17408): (7, 1, 32, 256, 1, 11),
    (1, 6144, 5120): (5, 16, 64, 256, 1),
    (1, 10240, 5120): (2, 1, 32, 128, 1),
    (1, 12288, 5120): (7, 1, 64, 128, 1, 11),
    (1, 17408, 5120): (1, 1, 64, 128, 1),
    (1, 248320, 5120): (0, 1, 64, 128, 1),
    (8, 5120, 17408): (8, 16, 32, 256, 1, 11),
    (8, 10240, 5120): (3, 16, 64, 128, 1),
    (8, 17408, 5120): (3, 16, 64, 128, 1),
    (8, 248320, 5120): (3, 16, 64, 256, 1),
    (32, 48, 5120): (13, 32, 32, 64, 16, 16),
    (32, 1024, 4096): (3, 32, 32, 256, 1),
    (32, 1024, 5120): (13, 32, 32, 64, 4, 32),
    (32, 4096, 1024): (5, 32, 32, 256, 1),
    (32, 5120, 6144): (8, 32, 64, 128, 1, 11),
    (32, 5120, 17408): (8, 32, 64, 256, 1, 3),
    (32, 6144, 5120): (8, 32, 64, 256, 1, 37),
    (32, 10240, 5120): (8, 32, 64, 256, 1, 3),
    (32, 12288, 5120): (5, 32, 128, 256, 1),
    (32, 17408, 5120): (5, 32, 128, 256, 1),
    (32, 248320, 5120): (4, 32, 64, 256, 1),
}


def _pick(m, n, kd):
    """Generic fallback for shapes outside the tuned table."""
    if m == 1:
        return (0, 1, 64, 128, 1) if n >= 2048 else (2, 1, 32, 256, 1)
    tm = 16
    while tm < m:
        tm *= 2
    return (3, tm, 64, 128, 1)


def skinny_gemm_tn_sm120_into(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
    """Compute C = A @ B^T into preallocated C (destination-passing style)."""
    m, kd = a.shape[0], a.shape[1]
    n = b.shape[0]
    e = _CFG.get((m, n, kd)) or _pick(m, n, kd)
    tag, tm, tn, tk = e[0], e[1], e[2], e[3]
    s = torch.cuda.current_stream(a.device)
    gn = (n + tn - 1) // tn

    if tag == 5:
        ct.launch(s, (gn, 1, 1), _gemm_bnatural_stagger, (a, b, c, tm, tn, tk))
    elif tag == 3:
        ct.launch(s, (gn, 1, 1), _gemm_auto, (a, b, c, tm, tn, tk))
    elif tag == 1:
        ct.launch(s, (gn, 1, 1), _gemv_kfirst, (a, b, c, tn, tk))
    elif tag == 2:
        ct.launch(s, (gn, 1, 1), _gemv_direct, (a, b, c, tn, tk))
    elif tag == 10:
        ct.launch(s, (gn, 1, 1), _gemv_occ2, (a, b, c, tn, tk))
    elif tag == 15:
        sk = e[4]
        # The split-K partials and counters are mutable scratch: allocate them per
        # call on the input's device. Caching them across invocations would share
        # them between concurrent same-shape calls on different streams/devices.
        partials = torch.empty((sk, gn * tn), dtype=torch.float32, device=a.device)
        counters = torch.zeros((gn,), dtype=torch.int32, device=a.device)
        ct.launch(s, (gn, sk, 1), _gemv_split_fused, (a, b, c, partials, counters, tn, tk, sk))
    elif tag == 6:
        ct.launch(s, (gn, 1, 1), _gemv_stagger, (a, b, c, tn, tk, e[5]))
    elif tag == 7:
        ct.launch(s, (gn, 1, 1), _gemv_kstagger, (a, b, c, tn, tk, e[5]))
    elif tag == 8:
        ct.launch(s, (gn, 1, 1), _gemm_bnatural_stagger_p, (a, b, c, tm, tn, tk, e[5]))
    elif tag == 0:
        ct.launch(s, (gn, 1, 1), _gemv_auto, (a, b, c, tn, tk))
    elif tag == 4:
        ct.launch(s, (gn, 1, 1), _gemm_bnatural, (a, b, c, tm, tn, tk))
    else:
        # split-K: fp32 partials + a fixed-order second-pass reduce, so the
        # summation order is identical on every run (bit-reproducible).
        sk = e[4]
        # Same per-call scratch rationale as the fused path above.
        partials = torch.empty((sk * m, n), dtype=torch.float32, device=a.device)
        ct.launch(s, (gn, sk, 1), _gemm_split_auto, (a, b, partials, m, tn, tk, sk))
        rn = e[5]
        ct.launch(s, ((n + rn - 1) // rn, 1, 1), _reduce_flat, (partials, c, sk, m, rn))


def skinny_gemm_tn_sm120_cutile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute C[M,N] = A[M,K] @ B[N,K]^T (nn.Linear TN weight layout, bias-free).

    Args:
        a: Activations, (M, K), bfloat16, contiguous. M <= 64 (decode-skinny).
        b: Weight in nn.Linear layout, (N, K), bfloat16, contiguous.

    Returns:
        C: (M, N) bfloat16.
    """
    assert a.is_cuda and b.is_cuda, "skinny_gemm_tn_sm120 requires CUDA tensors"
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "skinny_gemm_tn_sm120 is bf16-only"
    M, K = a.shape
    N, Kb = b.shape
    assert K == Kb, f"K mismatch: {K} vs {Kb}"
    assert 1 <= M <= 64, f"M={M} exceeds the skinny decode contract (M <= 64)"
    assert K % 8 == 0 and N % 8 == 0, f"K and N must be multiples of 8, got K={K} N={N}"
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    skinny_gemm_tn_sm120_into(a, b, c)
    return c
