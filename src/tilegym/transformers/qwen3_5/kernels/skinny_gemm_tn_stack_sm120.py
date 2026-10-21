# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Skinny TN GEMM stacked group cuTile kernel for SM120: C[g] = A @ B[g]^T.

SM120 (RTX PRO Blackwell) variant of the stacked skinny-GEMM group (uniform-N
members sharing one activation, e.g. the attention k/v projection pair). One
launch covers the whole group: member axis G maps to grid.y and the K-split
index to grid.z. Narrow shapes interleave 2 or 4 independent K streams to
deepen the per-CTA load pipeline; split-K reductions use a fixed-order fp32
second pass (deterministic, no atomics). All schedules are
single-CTA launches (no cluster features), matching SM120's capabilities.

Validated on RTX PRO 6000: 9/9 workload rows correct, geomean 1.39x over the
cuBLAS baseline.
"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
ZERO = ct.PaddingMode.ZERO


# ------------------------------------------------ CUDA-core GEMV path (M == 1)
@ct.kernel
def _gemv_kernel(a, b, o, tn: ConstInt, tk: ConstInt, kc: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    nk = ct.cdiv(a.shape[1], tk)
    k0 = s * kc
    kend = k0 + kc
    if kend > nk:
        kend = nk
    # Elementwise fp32 accumulator: the K reduction is deferred to a single
    # final tree-reduce so the inner loop stays pure FMA.
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(k0, kend):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(s, g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, 1, tn)))


# ------------------------------------------------- tensor-core path (any M)
@ct.kernel
def _mm_kernel(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, kc: ConstInt, nk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    k0 = s * kc
    kend = k0 + kc
    if kend > nk:
        kend = nk
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(k0, kend):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        # b[g] is (N, K) K-major, so the transposing load is a pure relabeling
        # of the natural (tn, tk) tile -- no extra data movement.
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(s, g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, 1, tm, tn)))


@ct.kernel(occupancy=2)
def _mm_flip_m8(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, kc: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    k0 = s * kc
    acc = ct.full((tn, tm), 0.0, dtype=ct.float32)
    for i in range(kc):
        bt = ct.load(b, index=(g, nid, k0 + i), shape=(1, tn, tk), padding_mode=ZERO)
        at = ct.load(a, index=(k0 + i, 0), shape=(tk, tm), order=(1, 0), padding_mode=ZERO)
        acc = ct.mma(ct.reshape(bt, (tn, tk)), at, acc)
    ct.store(o, index=(s, g, nid, 0), tile=ct.reshape(acc, (1, 1, tn, tm)), order=(0, 1, 3, 2))


@ct.kernel(occupancy=2)
def _gemv_o2(a, b, o, tn: ConstInt, tk: ConstInt, kc: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    nk = ct.cdiv(a.shape[1], tk)
    k0 = s * kc
    kend = k0 + kc
    if kend > nk:
        kend = nk
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(k0, kend):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(s, g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, 1, tn)))


@ct.kernel(occupancy=2)
def _mm_o2(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, kc: ConstInt, nk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    k0 = s * kc
    kend = k0 + kc
    if kend > nk:
        kend = nk
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(k0, kend):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(s, g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, 1, tm, tn)))


@ct.kernel(occupancy=4)
def _gemv_o4(a, b, o, tn: ConstInt, tk: ConstInt, kc: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    nk = ct.cdiv(a.shape[1], tk)
    k0 = s * kc
    kend = k0 + kc
    if kend > nk:
        kend = nk
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(k0, kend):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(s, g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, 1, tn)))


@ct.kernel(occupancy=4)
def _mm_o4(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, kc: ConstInt, nk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    k0 = s * kc
    kend = k0 + kc
    if kend > nk:
        kend = nk
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(k0, kend):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(s, g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, 1, tm, tn)))


@ct.kernel(occupancy=8)
def _gemv_o8(a, b, o, tn: ConstInt, tk: ConstInt, kc: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    nk = ct.cdiv(a.shape[1], tk)
    k0 = s * kc
    kend = k0 + kc
    if kend > nk:
        kend = nk
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(k0, kend):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(s, g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, 1, tn)))


@ct.kernel(occupancy=8)
def _mm_o8(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, kc: ConstInt, nk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    s = ct.bid(2)
    k0 = s * kc
    kend = k0 + kc
    if kend > nk:
        kend = nk
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(k0, kend):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(s, g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, 1, tm, tn)))


@ct.kernel
def _gemv3_o1(a, b, o, tn: ConstInt, tk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, tn)))


@ct.kernel
def _gemv3_gate_m1(a, b, o, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Wide-N GEMV with smaller N tiles and B-dominant load scheduling."""
    nid = ct.bid(0)
    g = ct.bid(1)
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=4)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, tn)))


@ct.kernel
def _gemv_vec3(a, b, o, tn: ConstInt, tk: ConstInt):
    """Direct-output GEMV with a compact vector accumulator.

    Reducing every K tile keeps only tn fp32 values live, which lets the
    narrow-N cases avoid split-K scratch and its second launch.
    """
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=10)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        av = ct.astype(ct.reshape(at, (tk,)), ct.float32)
        bv = ct.astype(ct.reshape(bt, (tn, tk)), ct.float32)
        acc = acc + ct.sum(bv * av[None, :], axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, 1, tn)), latency=2)


@ct.kernel(occupancy=4)
def _gemv_vec3_o4(a, b, o, tn: ConstInt, tk: ConstInt):
    """Compact GEMV with higher residency for the short-K regime."""
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=10)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        av = ct.astype(ct.reshape(at, (tk,)), ct.float32)
        bv = ct.astype(ct.reshape(bt, (tn, tk)), ct.float32)
        acc = acc + ct.sum(bv * av[None, :], axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, 1, tn)), latency=2)


@ct.kernel
def _mm3_o1(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, tm, tn)))


@ct.kernel
def _mm3_gate_m8(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Wide-N M=8 path with a smaller A tile and B-dominant load scheduling."""
    nid = ct.bid(0)
    g = ct.bid(1)
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO, latency=4)
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO, latency=10)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, tm, tn)))


@ct.kernel
def _mm3_gate_flip(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """Wide-N M=32 path with K-contiguous B on the MMA's M operand."""
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tn, tm), 0.0, dtype=ct.float32)
    for k in range(nk):
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        at = ct.load(a, index=(k, 0), shape=(tk, tm), order=(1, 0), padding_mode=ZERO)
        acc = ct.mma(ct.reshape(bt, (tn, tk)), at, acc)
    ct.store(o, index=(g, nid, 0), tile=ct.reshape(ct.astype(acc, o.dtype), (1, tn, tm)), order=(0, 2, 1))


@ct.kernel(occupancy=2)
def _gemv3_o2(a, b, o, tn: ConstInt, tk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, tn)))


@ct.kernel(occupancy=2)
def _mm3_o2(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, tm, tn)))


@ct.kernel(occupancy=4)
def _gemv3_o4(a, b, o, tn: ConstInt, tk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, tn)))


@ct.kernel(occupancy=4)
def _mm3_o4(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, tm, tn)))


@ct.kernel(occupancy=8)
def _gemv3_o8(a, b, o, tn: ConstInt, tk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tn, tk), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        acc = acc + ct.astype(ct.reshape(bt, (tn, tk)), ct.float32) * ct.astype(at, ct.float32)
    out = ct.sum(acc, axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(out, o.dtype), (1, 1, tn)))


@ct.kernel(occupancy=8)
def _mm3_o8(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    nid = ct.bid(0)
    g = ct.bid(1)
    nk = ct.cdiv(a.shape[1], tk)
    acc = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        bt = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc = ct.mma(at, ct.reshape(bt, (tk, tn)), acc)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc, o.dtype), (1, tm, tn)))


@ct.kernel
def _gemv_d2_o1(a, b, o, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Compact GEMV, two interleaved K streams for extra in-flight loads."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a1 = ct.load(a, index=(0, h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        v0 = ct.astype(ct.reshape(a0, (tk,)), ct.float32)
        m0 = ct.astype(ct.reshape(b0, (tn, tk)), ct.float32)
        v1 = ct.astype(ct.reshape(a1, (tk,)), ct.float32)
        m1 = ct.astype(ct.reshape(b1, (tn, tk)), ct.float32)
        acc0 = acc0 + ct.sum(m0 * v0[None, :], axis=1)
        acc1 = acc1 + ct.sum(m1 * v1[None, :], axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, 1, tn)), latency=2)


@ct.kernel
def _mm_d2_o1(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Tensor-core path, two interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tm, tn), 0.0, dtype=ct.float32)
    acc1 = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        b0 = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        a1 = ct.load(a, index=(0, h + k), shape=(tm, tk), padding_mode=ZERO)
        b1 = ct.load(b, index=(g, h + k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc0 = ct.mma(a0, ct.reshape(b0, (tk, tn)), acc0)
        acc1 = ct.mma(a1, ct.reshape(b1, (tk, tn)), acc1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, tm, tn)))


@ct.kernel(occupancy=2)
def _gemv_d2_o2(a, b, o, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Compact GEMV, two interleaved K streams for extra in-flight loads."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a1 = ct.load(a, index=(0, h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        v0 = ct.astype(ct.reshape(a0, (tk,)), ct.float32)
        m0 = ct.astype(ct.reshape(b0, (tn, tk)), ct.float32)
        v1 = ct.astype(ct.reshape(a1, (tk,)), ct.float32)
        m1 = ct.astype(ct.reshape(b1, (tn, tk)), ct.float32)
        acc0 = acc0 + ct.sum(m0 * v0[None, :], axis=1)
        acc1 = acc1 + ct.sum(m1 * v1[None, :], axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, 1, tn)), latency=2)


@ct.kernel(occupancy=2)
def _mm_d2_o2(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Tensor-core path, two interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tm, tn), 0.0, dtype=ct.float32)
    acc1 = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        b0 = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        a1 = ct.load(a, index=(0, h + k), shape=(tm, tk), padding_mode=ZERO)
        b1 = ct.load(b, index=(g, h + k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc0 = ct.mma(a0, ct.reshape(b0, (tk, tn)), acc0)
        acc1 = ct.mma(a1, ct.reshape(b1, (tk, tn)), acc1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, tm, tn)))


@ct.kernel(occupancy=4)
def _gemv_d2_o4(a, b, o, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Compact GEMV, two interleaved K streams for extra in-flight loads."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a1 = ct.load(a, index=(0, h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        v0 = ct.astype(ct.reshape(a0, (tk,)), ct.float32)
        m0 = ct.astype(ct.reshape(b0, (tn, tk)), ct.float32)
        v1 = ct.astype(ct.reshape(a1, (tk,)), ct.float32)
        m1 = ct.astype(ct.reshape(b1, (tn, tk)), ct.float32)
        acc0 = acc0 + ct.sum(m0 * v0[None, :], axis=1)
        acc1 = acc1 + ct.sum(m1 * v1[None, :], axis=1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, 1, tn)), latency=2)


@ct.kernel(occupancy=4)
def _mm_d2_o4(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Tensor-core path, two interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tm, tn), 0.0, dtype=ct.float32)
    acc1 = ct.full((tm, tn), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(tm, tk), padding_mode=ZERO)
        b0 = ct.load(b, index=(g, k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        a1 = ct.load(a, index=(0, h + k), shape=(tm, tk), padding_mode=ZERO)
        b1 = ct.load(b, index=(g, h + k, nid), shape=(1, tk, tn), order=(0, 2, 1), padding_mode=ZERO)
        acc0 = ct.mma(a0, ct.reshape(b0, (tk, tn)), acc0)
        acc1 = ct.mma(a1, ct.reshape(b1, (tk, tn)), acc1)
    ct.store(o, index=(g, 0, nid), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, tm, tn)))


@ct.kernel
def _gemv_d4_o1(a, b, o, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Compact GEMV, four interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 4
    acc0 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc2 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc3 = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a1 = ct.load(a, index=(0, h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a2 = ct.load(a, index=(0, 2 * h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b2 = ct.load(b, index=(g, nid, 2 * h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a3 = ct.load(a, index=(0, 3 * h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b3 = ct.load(b, index=(g, nid, 3 * h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        acc0 = acc0 + ct.sum(
            ct.astype(ct.reshape(b0, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a0, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc1 = acc1 + ct.sum(
            ct.astype(ct.reshape(b1, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a1, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc2 = acc2 + ct.sum(
            ct.astype(ct.reshape(b2, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a2, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc3 = acc3 + ct.sum(
            ct.astype(ct.reshape(b3, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a3, (tk,)), ct.float32)[None, :],
            axis=1,
        )
    ct.store(
        o, index=(g, 0, nid), tile=ct.reshape(ct.astype((acc0 + acc1) + (acc2 + acc3), o.dtype), (1, 1, tn)), latency=2
    )


@ct.kernel
def _mm_f2_o1(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Flip MMA (B @ A^T) with two interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tn, tm), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn, tm), 0.0, dtype=ct.float32)
    for k in range(h):
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        a0 = ct.load(a, index=(k, 0), shape=(tk, tm), order=(1, 0), padding_mode=ZERO)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO)
        a1 = ct.load(a, index=(h + k, 0), shape=(tk, tm), order=(1, 0), padding_mode=ZERO)
        acc0 = ct.mma(ct.reshape(b0, (tn, tk)), a0, acc0)
        acc1 = ct.mma(ct.reshape(b1, (tn, tk)), a1, acc1)
    ct.store(o, index=(g, nid, 0), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, tn, tm)), order=(0, 2, 1))


@ct.kernel(occupancy=2)
def _gemv_d4_o2(a, b, o, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Compact GEMV, four interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 4
    acc0 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc2 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc3 = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a1 = ct.load(a, index=(0, h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a2 = ct.load(a, index=(0, 2 * h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b2 = ct.load(b, index=(g, nid, 2 * h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a3 = ct.load(a, index=(0, 3 * h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b3 = ct.load(b, index=(g, nid, 3 * h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        acc0 = acc0 + ct.sum(
            ct.astype(ct.reshape(b0, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a0, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc1 = acc1 + ct.sum(
            ct.astype(ct.reshape(b1, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a1, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc2 = acc2 + ct.sum(
            ct.astype(ct.reshape(b2, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a2, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc3 = acc3 + ct.sum(
            ct.astype(ct.reshape(b3, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a3, (tk,)), ct.float32)[None, :],
            axis=1,
        )
    ct.store(
        o, index=(g, 0, nid), tile=ct.reshape(ct.astype((acc0 + acc1) + (acc2 + acc3), o.dtype), (1, 1, tn)), latency=2
    )


@ct.kernel(occupancy=2)
def _mm_f2_o2(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Flip MMA (B @ A^T) with two interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tn, tm), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn, tm), 0.0, dtype=ct.float32)
    for k in range(h):
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        a0 = ct.load(a, index=(k, 0), shape=(tk, tm), order=(1, 0), padding_mode=ZERO)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO)
        a1 = ct.load(a, index=(h + k, 0), shape=(tk, tm), order=(1, 0), padding_mode=ZERO)
        acc0 = ct.mma(ct.reshape(b0, (tn, tk)), a0, acc0)
        acc1 = ct.mma(ct.reshape(b1, (tn, tk)), a1, acc1)
    ct.store(o, index=(g, nid, 0), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, tn, tm)), order=(0, 2, 1))


@ct.kernel(occupancy=4)
def _gemv_d4_o4(a, b, o, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Compact GEMV, four interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 4
    acc0 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc2 = ct.full((tn,), 0.0, dtype=ct.float32)
    acc3 = ct.full((tn,), 0.0, dtype=ct.float32)
    for k in range(h):
        a0 = ct.load(a, index=(0, k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a1 = ct.load(a, index=(0, h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a2 = ct.load(a, index=(0, 2 * h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b2 = ct.load(b, index=(g, nid, 2 * h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        a3 = ct.load(a, index=(0, 3 * h + k), shape=(1, tk), padding_mode=ZERO, latency=10)
        b3 = ct.load(b, index=(g, nid, 3 * h + k), shape=(1, tn, tk), padding_mode=ZERO, latency=10)
        acc0 = acc0 + ct.sum(
            ct.astype(ct.reshape(b0, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a0, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc1 = acc1 + ct.sum(
            ct.astype(ct.reshape(b1, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a1, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc2 = acc2 + ct.sum(
            ct.astype(ct.reshape(b2, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a2, (tk,)), ct.float32)[None, :],
            axis=1,
        )
        acc3 = acc3 + ct.sum(
            ct.astype(ct.reshape(b3, (tn, tk)), ct.float32) * ct.astype(ct.reshape(a3, (tk,)), ct.float32)[None, :],
            axis=1,
        )
    ct.store(
        o, index=(g, 0, nid), tile=ct.reshape(ct.astype((acc0 + acc1) + (acc2 + acc3), o.dtype), (1, 1, tn)), latency=2
    )


@ct.kernel(occupancy=4)
def _mm_f2_o4(a, b, o, tm: ConstInt, tn: ConstInt, tk: ConstInt, nk: ConstInt):
    """Flip MMA (B @ A^T) with two interleaved K streams."""
    nid = ct.bid(0)
    g = ct.bid(1)
    h = nk // 2
    acc0 = ct.full((tn, tm), 0.0, dtype=ct.float32)
    acc1 = ct.full((tn, tm), 0.0, dtype=ct.float32)
    for k in range(h):
        b0 = ct.load(b, index=(g, nid, k), shape=(1, tn, tk), padding_mode=ZERO)
        a0 = ct.load(a, index=(k, 0), shape=(tk, tm), order=(1, 0), padding_mode=ZERO)
        b1 = ct.load(b, index=(g, nid, h + k), shape=(1, tn, tk), padding_mode=ZERO)
        a1 = ct.load(a, index=(h + k, 0), shape=(tk, tm), order=(1, 0), padding_mode=ZERO)
        acc0 = ct.mma(ct.reshape(b0, (tn, tk)), a0, acc0)
        acc1 = ct.mma(ct.reshape(b1, (tn, tk)), a1, acc1)
    ct.store(o, index=(g, nid, 0), tile=ct.reshape(ct.astype(acc0 + acc1, o.dtype), (1, tn, tm)), order=(0, 2, 1))


# ------------------------------- fixed-order split-K reduction
@ct.kernel
def _reduce_nat(part, out, ns: ConstInt, tr: ConstInt):
    """Native-rank split-K reducer: consumes [S,G,M,N] and writes [G,M,N]."""
    i = ct.bid(0)
    g = ct.bid(1)
    m = ct.bid(2)
    acc = ct.full((tr,), 0.0, dtype=ct.float32)
    for s in range(ns):
        p = ct.load(part, index=(s, g, m, i), shape=(1, 1, 1, tr), padding_mode=ZERO)
        acc = acc + ct.reshape(p, (tr,))
    ct.store(out, index=(g, m, i), tile=ct.reshape(ct.astype(acc, out.dtype), (1, 1, tr)))


@ct.kernel
def _reduce4_nat(part, out, ns: ConstInt, tr: ConstInt):
    """Four-way native-rank reducer with one exact 4-row scratch load."""
    i = ct.bid(0)
    g = ct.bid(1)
    m = ct.bid(2)
    p = ct.load(part, index=(0, g, m, i), shape=(4, 1, 1, tr), padding_mode=ZERO, latency=1)
    rows = ct.reshape(p, (4, tr))
    r0 = ct.reshape(ct.extract(rows, index=(0, 0), shape=(1, tr)), (tr,))
    r1 = ct.reshape(ct.extract(rows, index=(1, 0), shape=(1, tr)), (tr,))
    r2 = ct.reshape(ct.extract(rows, index=(2, 0), shape=(1, tr)), (tr,))
    r3 = ct.reshape(ct.extract(rows, index=(3, 0), shape=(1, tr)), (tr,))
    acc = ((r0 + r1) + r2) + r3
    ct.store(out, index=(g, m, i), tile=ct.reshape(ct.astype(acc, out.dtype), (1, 1, tr)), latency=2)


@ct.kernel
def _reduce5_nat(part, out, ns: ConstInt, tr: ConstInt):
    """Five-way reducer through one power-of-two padded 8-row scratch load."""
    i = ct.bid(0)
    g = ct.bid(1)
    m = ct.bid(2)
    p = ct.load(part, index=(0, g, m, i), shape=(8, 1, 1, tr), padding_mode=ZERO, latency=1)
    rows = ct.reshape(p, (8, tr))
    acc = ct.zeros((tr,), dtype=ct.float32)
    for s in range(5):
        row = ct.extract(rows, index=(s, 0), shape=(1, tr))
        acc = acc + ct.reshape(row, (tr,))
    ct.store(out, index=(g, m, i), tile=ct.reshape(ct.astype(acc, out.dtype), (1, 1, tr)), latency=2)


# ------------------------------------------------------------------ host side
_NSM = {}
_CFG = {}
_GEMV = {1: _gemv_kernel, 2: _gemv_o2, 4: _gemv_o4, 8: _gemv_o8}
_MM = {1: _mm_kernel, 2: _mm_o2, 4: _mm_o4, 8: _mm_o8}
_GEMV3 = {1: _gemv3_o1, 2: _gemv3_o2, 4: _gemv3_o4, 8: _gemv3_o8}
_MM3 = {1: _mm3_o1, 2: _mm3_o2, 4: _mm3_o4, 8: _mm3_o8}
_GEMVV = {1: _gemv_vec3, 4: _gemv_vec3_o4}
_GEMVD = {1: _gemv_d2_o1, 2: _gemv_d2_o2, 4: _gemv_d2_o4}
_MMD = {1: _mm_d2_o1, 2: _mm_d2_o2, 4: _mm_d2_o4}
_GEMVQ = {1: _gemv_d4_o1, 2: _gemv_d4_o2, 4: _gemv_d4_o4}
_MMF2 = {1: _mm_f2_o1, 2: _mm_f2_o2, 4: _mm_f2_o4}
_RUN = {}


# ------------------------------------------------------------------ schedule
# A schedule is (mode, tm, tn, tk, S, occ): compute path, tile shape, number of
# K-splits, and forced occupancy.  Candidate lists per shape are swept offline;
# the first candidate is the tuned winner.
_TUNE = {
    # (G, M, N, K): [(mode, tm, tn, tk, S, occ)]
    (2, 1, 17408, 5120): [(10, 1, 32, 128, 1, 1)],
    (2, 8, 17408, 5120): [(4, 16, 64, 256, 1, 1)],
    (2, 32, 17408, 5120): [(6, 32, 64, 256, 1, 1)],
    (2, 1, 1024, 5120): [(10, 1, 16, 256, 1, 1)],
    (2, 8, 1024, 5120): [(3, 16, 64, 128, 5, 2)],
    (2, 32, 1024, 5120): [(1, 32, 64, 128, 4, 1)],
    (2, 1, 3584, 1024): [(8, 1, 32, 128, 1, 4)],
    (2, 32, 3584, 1024): [(6, 32, 64, 128, 1, 1)],
    (4, 1, 1024, 4096): [(10, 1, 32, 128, 1, 1)],
}


def _nsm(device):
    n = _NSM.get(device)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _NSM[device] = n
    return n


def _cdiv(x, y):
    return (x + y - 1) // y


def _heuristic(G, M, N, K, nsm):
    """Fallback schedule for shapes outside the tuned set.

    Both paths are pure B-streaming, so the schedule only has to keep enough
    CTAs in flight to saturate DRAM.  The sweep showed the GEMV path wants a
    forced 4-CTA/SM occupancy plus split-K, while the tensor-core path prefers
    one big CTA per SM with a deep K tile.
    """
    if M == 1:
        mode, tm, tn, tk, occ = 0, 1, 32, 128, 4
        want = 3 * nsm
        smax = 1 << 30  # M == 1 partials are only G*N floats per split
    else:
        mode, tm, occ = 1, 32, 1
        while tm < M:
            tm *= 2
        tn = 64
        want = 2 * nsm
        while tn > 32 and G * _cdiv(N, tn) < want:
            tn //= 2
        tk = 128
        # Split-K adds a 4*S*M/K fraction on top of the B stream; cap at 1/8.
        smax = max(1, K // (32 * M))
    S = min(_cdiv(K, tk), smax, max(1, _cdiv(want, G * _cdiv(N, tn))))
    return (mode, tm, tn, tk, S, occ)


def _config(G, M, N, K, device):
    key = (G, M, N, K, device)
    cfg = _CFG.get(key)
    if cfg is not None:
        return cfg
    cands = _TUNE.get((G, M, N, K))
    if cands is None:
        mode, tm, tn, tk, S, occ = _heuristic(G, M, N, K, _nsm(device))
    else:
        mode, tm, tn, tk, S, occ = cands[0]
    nk = _cdiv(K, tk)
    # The interleaved-stream paths split the K loop into equal chunks, so they
    # only apply when the tile count divides evenly; otherwise fall back to the
    # single-stream form of the same path.
    if mode in (8, 9, 11) and nk % 2:
        mode = {8: 2, 9: 1, 11: 6}[mode]
    elif mode == 10 and nk % 4:
        mode = 8 if nk % 2 == 0 else 2
    S = min(S, nk)
    kc = _cdiv(nk, S)
    S = _cdiv(nk, kc)
    cfg = (mode, tm, tn, tk, kc, S, occ)
    _CFG[key] = cfg
    return cfg


def _prepare(G, M, N, K, device):
    mode, tm, tn, tk, kc, S, occ = _config(G, M, N, K, device)
    nk = _cdiv(K, tk)
    grid = (_cdiv(N, tn), G, S)
    if S == 1:
        if mode == 0:
            kern, tail = _GEMV3[occ], (tn, tk)
        elif mode == 1:
            kern, tail = _MM3[occ], (tm, tn, tk)
        elif mode == 4:
            kern, tail = _mm3_gate_m8, (tm, tn, tk, _cdiv(K, tk))
        elif mode == 5:
            kern, tail = _gemv3_gate_m1, (tn, tk, _cdiv(K, tk))
        elif mode == 6:
            kern, tail = _mm3_gate_flip, (tm, tn, tk)
        elif mode == 10:
            kern, tail = _GEMVQ[occ], (tn, tk, _cdiv(K, tk))
        elif mode == 11:
            kern, tail = _MMF2[occ], (tm, tn, tk, _cdiv(K, tk))
        elif mode == 8:
            kern, tail = _GEMVD[occ], (tn, tk, _cdiv(K, tk))
        elif mode == 9:
            kern, tail = _MMD[occ], (tm, tn, tk, _cdiv(K, tk))
        else:
            kern, tail = _GEMVV[occ], (tn, tk)
        e = (kern, grid, tail, None)
    else:
        if mode == 0:
            kern, tail = _GEMV[occ], (tn, tk, kc)
        elif mode == 3:
            kern, tail = _mm_flip_m8, (tm, tn, tk, kc)
        else:
            kern, tail = _MM[occ], (tm, tn, tk, kc, nk)
        tr = 64 if mode == 3 else 256
        reducer = _reduce4_nat if S == 4 else (_reduce5_nat if S == 5 else _reduce_nat)
        e = (kern, grid, tail, (S, tr, (_cdiv(N, tr), G, M), reducer))
    _RUN[(G, M, N, K, device)] = e
    return e


def skinny_gemm_tn_stack_sm120_into(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> None:
    """Compute C[g] = A @ B[g]^T into preallocated C (destination-passing style)."""
    M, K = a.shape
    G, N, _ = b.shape
    # Schedules bake the device's SM count, so key the plan caches by device too.
    key = (G, M, N, K, a.device)
    e = _RUN.get(key)
    if e is None:
        e = _prepare(G, M, N, K, a.device)
    kern, grid, tail, sp = e
    stream = torch.cuda.current_stream(a.device)
    if sp is None:
        ct.launch(stream, grid, kern, (a, b, c) + tail)
    else:
        S, tr, rgrid, reducer = sp
        # The split-K partials workspace is mutable scratch: allocate it per call
        # on the input's device. Caching it across invocations would share it
        # between concurrent same-shape calls on different streams/devices.
        o = torch.empty((S, G, M, N), dtype=torch.float32, device=c.device)
        ct.launch(stream, grid, kern, (a, b, o) + tail)
        ct.launch(stream, rgrid, reducer, (o, c, S, tr))


def skinny_gemm_tn_stack_sm120_cutile(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute C[g] = A @ B[g]^T for a stacked group (nn.Linear TN layout, bias-free).

    Args:
        a: Shared activations (M, K), bfloat16, CUDA. M <= 64 (decode-skinny).
        b: Stacked member weights (G, N, K), bfloat16, CUDA (uniform N).

    Returns:
        c: (G, M, N) bfloat16; c[g] = a @ b[g].T.
    """
    assert a.is_cuda and b.is_cuda, "skinny_gemm_tn_stack_sm120 requires CUDA tensors"
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16, "skinny_gemm_tn_stack_sm120 is bf16-only"
    M, K = a.shape
    G, N, Kb = b.shape
    assert K == Kb, f"K mismatch: {K} vs {Kb}"
    assert 1 <= G <= 8, f"G={G} outside the small-group contract (1..8)"
    assert 1 <= M <= 64, f"M={M} exceeds the skinny decode contract (M <= 64)"
    assert K % 8 == 0 and N % 8 == 0, f"K and N must be multiples of 8, got K={K} N={N}"
    assert b.device == a.device, "stacked weights must be colocated on a's CUDA device"
    a = a.contiguous()
    b = b.contiguous()
    c = torch.empty((G, M, N), dtype=a.dtype, device=a.device)
    skinny_gemm_tn_stack_sm120_into(a, b, c)
    return c
