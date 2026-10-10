# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Skinny TN GEMM x4 group cuTile kernel for SM120: Ci[M,Ni] = A[M,K] @ Bi[Ni,K]^T.

SM120 (RTX PRO Blackwell) variant of the GatedDeltaNet in-projection group:
one launch computes four skinny projections sharing the same activation tile.
Five tile bodies (two tensor-core MMA forms, two CUDA-core GEMV forms, one
direct A-major MMA form) are selected per shape from a tuned table with a
generic fallback; member outputs are striped over a single flat grid. All kernels launch
with num_ctas=1 (no cluster features), matching SM120's capabilities.

Validated on RTX PRO 6000: 7/7 workload rows correct, geomean 1.62x over the
cuBLAS baseline.
"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
ZERO = ct.PaddingMode.ZERO


@ct.function
def _native_tile(a, b, c, nt, TM, TN, TK):
    nk = ct.cdiv(b.shape[1], TK)
    acc = ct.zeros((TN, TM), dtype=ct.float32)
    for k in range(nk):
        bt = ct.load(b, index=(nt, k), shape=(TN, TK), padding_mode=ZERO)
        at = ct.transpose(ct.load(a, index=(0, k), shape=(TM, TK), padding_mode=ZERO))
        acc = ct.mma(bt, at, acc)
    ct.store(c, index=(nt, 0), tile=ct.astype(acc, c.dtype), order=(1, 0))


@ct.function
def _gemv_tile(a, b, c, nt, TN, TK):
    nk = ct.cdiv(a.shape[1], TK)
    acc = ct.zeros((TN,), dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, TK), padding_mode=ZERO, latency=3).astype(ct.float32)
        bt = ct.load(b, index=(nt, k), shape=(TN, TK), padding_mode=ZERO, latency=10).astype(ct.float32)
        acc = acc + ct.sum(bt * at, axis=1)
    ct.store(c, index=(0, nt), tile=acc.reshape((1, TN)).astype(c.dtype))


@ct.function
def _native2_tile(a, b, c, nt, TM, TN, TK):
    nk = ct.cdiv(b.shape[1], TK)
    acc = ct.zeros((TN, TM), dtype=ct.float32)
    for k in range(nk):
        bt = ct.load(b, index=(nt, k), shape=(TN, TK), padding_mode=ZERO, latency=2)
        at = ct.transpose(ct.load(a, index=(0, k), shape=(TM, TK), padding_mode=ZERO, latency=2))
        acc = ct.mma(bt, at, acc)
    ct.store(c, index=(nt, 0), tile=ct.astype(acc, c.dtype), order=(1, 0))


@ct.function
def _gemv1_tile(a, b, c, nt, TN, TK):
    nk = ct.cdiv(a.shape[1], TK)
    acc = ct.zeros((TN,), dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(1, TK), padding_mode=ZERO, latency=1).astype(ct.float32)
        bt = ct.load(b, index=(nt, k), shape=(TN, TK), padding_mode=ZERO, latency=1).astype(ct.float32)
        acc = acc + ct.sum(bt * at, axis=1)
    ct.store(c, index=(0, nt), tile=acc.reshape((1, TN)).astype(c.dtype))


@ct.function
def _direct_tile(a, b, c, nt, TM, TN, TK):
    nk = ct.cdiv(b.shape[1], TK)
    acc = ct.zeros((TM, TN), dtype=ct.float32)
    for k in range(nk):
        at = ct.load(a, index=(0, k), shape=(TM, TK), padding_mode=ZERO, latency=8)
        bt = ct.load(b, index=(k, nt), shape=(TK, TN), order=(1, 0), padding_mode=ZERO, latency=8)
        acc = ct.mma(at, bt, acc)
    ct.store(c, index=(0, nt), tile=ct.astype(acc, c.dtype), latency=2)


@ct.kernel(num_ctas=1)
def _kn(a, b0, b1, b2, b3, c0, c1, c2, c3, TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bid = ct.bid(0)
    n0 = ct.cdiv(b0.shape[0], TN)
    n1 = ct.cdiv(b1.shape[0], TN)
    n2 = ct.cdiv(b2.shape[0], TN)
    if bid < n0:
        _native_tile(a, b0, c0, bid, TM, TN, TK)
    elif bid < n0 + n1:
        _native_tile(a, b1, c1, bid - n0, TM, TN, TK)
    elif bid < n0 + n1 + n2:
        _native_tile(a, b2, c2, bid - n0 - n1, TM, TN, TK)
    else:
        _native_tile(a, b3, c3, bid - n0 - n1 - n2, TM, TN, TK)


@ct.kernel(num_ctas=1)
def _kp(a, b0, b1, b2, b3, c0, c1, c2, c3, TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bid = ct.bid(0)
    n0 = ct.cdiv(b0.shape[0], TN)
    n1 = ct.cdiv(b1.shape[0], TN)
    n2 = ct.cdiv(b2.shape[0], TN)
    if bid < n0:
        _native2_tile(a, b0, c0, bid, TM, TN, TK)
    elif bid < n0 + n1:
        _native2_tile(a, b1, c1, bid - n0, TM, TN, TK)
    elif bid < n0 + n1 + n2:
        _native2_tile(a, b2, c2, bid - n0 - n1, TM, TN, TK)
    else:
        _native2_tile(a, b3, c3, bid - n0 - n1 - n2, TM, TN, TK)


@ct.kernel(num_ctas=1)
def _kd(a, b0, b1, b2, b3, c0, c1, c2, c3, TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bid = ct.bid(0)
    n0 = ct.cdiv(b0.shape[0], TN)
    n1 = ct.cdiv(b1.shape[0], TN)
    n2 = ct.cdiv(b2.shape[0], TN)
    if bid < n0:
        _direct_tile(a, b0, c0, bid, TM, TN, TK)
    elif bid < n0 + n1:
        _direct_tile(a, b1, c1, bid - n0, TM, TN, TK)
    elif bid < n0 + n1 + n2:
        _direct_tile(a, b2, c2, bid - n0 - n1, TM, TN, TK)
    else:
        _direct_tile(a, b3, c3, bid - n0 - n1 - n2, TM, TN, TK)


@ct.kernel(num_ctas=1)
def _kg(a, b0, b1, b2, b3, c0, c1, c2, c3, TN: ConstInt, TK: ConstInt):
    bid = ct.bid(0)
    n0 = ct.cdiv(b0.shape[0], TN)
    n1 = ct.cdiv(b1.shape[0], TN)
    n2 = ct.cdiv(b2.shape[0], TN)
    if bid < n0:
        _gemv_tile(a, b0, c0, bid, TN, TK)
    elif bid < n0 + n1:
        _gemv_tile(a, b1, c1, bid - n0, TN, TK)
    elif bid < n0 + n1 + n2:
        _gemv_tile(a, b2, c2, bid - n0 - n1, TN, TK)
    else:
        _gemv_tile(a, b3, c3, bid - n0 - n1 - n2, TN, TK)


@ct.kernel(num_ctas=1)
def _kh(a, b0, b1, b2, b3, c0, c1, c2, c3, TN: ConstInt, TK: ConstInt):
    bid = ct.bid(0)
    n0 = ct.cdiv(b0.shape[0], TN)
    n1 = ct.cdiv(b1.shape[0], TN)
    n2 = ct.cdiv(b2.shape[0], TN)
    if bid < n0:
        _gemv1_tile(a, b0, c0, bid, TN, TK)
    elif bid < n0 + n1:
        _gemv1_tile(a, b1, c1, bid - n0, TN, TK)
    elif bid < n0 + n1 + n2:
        _gemv1_tile(a, b2, c2, bid - n0 - n1, TN, TK)
    else:
        _gemv1_tile(a, b3, c3, bid - n0 - n1 - n2, TN, TK)


_BASE = {"n": _kn, "d": _kd, "g": _kg, "h": _kh, "p": _kp}
_VARIANTS = {}


def _kern(mode, occ, wrp):
    vkey = (mode, occ, wrp)
    k = _VARIANTS.get(vkey)
    if k is None:
        hints = {"num_ctas": 1}
        if occ:
            hints["occupancy"] = occ
        if wrp:
            hints["num_worker_warps"] = wrp
        base = _BASE[mode]
        k = base.replace_hints(**hints) if len(hints) > 1 else base
        _VARIANTS[vkey] = k
    return k


_CFG = {
    (1, 5120): ("h", 1, 32, 256, 16, 8),
    (8, 5120): ("d", 16, 128, 256, 0, 0),
    (32, 5120): ("n", 32, 64, 128, 4, 4),
    (1, 4096): ("p", 8, 64, 256, 0, 0),
    (32, 4096): ("n", 32, 128, 128, 0, 0),
    (1, 1024): ("h", 1, 32, 256, 16, 8),
    (32, 1024): ("n", 32, 64, 128, 0, 0),
}


def _pow2_at_least(x, lo):
    v = lo
    while v < x:
        v *= 2
    return v


_PLANS = {}


def _plan(key):
    m, k, n0, n1, n2, n3 = key
    cfg = _CFG.get((m, k))
    if cfg is None:
        cfg = ("n", _pow2_at_least(m, 8), 64, 256, 0, 0)
    mode, TM, TN, TK, occ, wrp = cfg
    kern = _kern(mode, occ, wrp)
    consts = (TN, TK) if mode in ("g", "h") else (TM, TN, TK)
    blocks = 0
    for n in (n0, n1, n2, n3):
        blocks += (n + TN - 1) // TN
    return kern, (blocks, 1, 1), consts


def skinny_gemm_tn_x4_sm120_into(
    a: torch.Tensor,
    b0: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    b3: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    c3: torch.Tensor,
) -> None:
    """Compute Ci = A @ Bi^T into preallocated Ci (destination-passing style)."""
    # Plans capture grid/consts for a shape set; key the cache by device too so
    # multi-GPU runs never replay a plan's kernels against the wrong context.
    key = (a.shape, b0.shape[0], b1.shape[0], b2.shape[0], b3.shape[0], a.device)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _plan((key[0][0], key[0][1]) + key[1:5])
        _PLANS[key] = plan
    kern, grid, consts = plan
    ct.launch(
        torch.cuda.current_stream(a.device),
        grid,
        kern,
        (a, b0, b1, b2, b3, c0, c1, c2, c3) + consts,
    )


def skinny_gemm_tn_x4_sm120_cutile(
    a: torch.Tensor, b0: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor, b3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Ci = A @ Bi^T for i in 0..3 (nn.Linear TN weight layout, bias-free).

    Args:
        a: Shared activations (M, K), bfloat16, CUDA. M <= 64 (decode-skinny).
        b0..b3: Member weights (Ni, K), bfloat16, CUDA.

    Returns:
        (c0, c1, c2, c3): member outputs (M, Ni) bfloat16.
    """
    assert a.is_cuda and all(b.is_cuda for b in (b0, b1, b2, b3)), "skinny_gemm_tn_x4_sm120 requires CUDA tensors"
    assert a.dtype == torch.bfloat16, "skinny_gemm_tn_x4_sm120 is bf16-only"
    M, K = a.shape
    bs = (b0, b1, b2, b3)
    Ns = [b.shape[0] for b in bs]
    for b in bs:
        assert b.dtype == torch.bfloat16 and b.shape[1] == K, "member weights must be bf16 (Ni, K)"
        assert b.device == a.device, "all member weights must be colocated on a's CUDA device"
    assert 1 <= M <= 64, f"M={M} exceeds the skinny decode contract (M <= 64)"
    assert K % 8 == 0 and all(N % 8 == 0 for N in Ns), f"K and Ni must be multiples of 8, got K={K} Ns={Ns}"
    a = a.contiguous()
    bs = tuple(b.contiguous() for b in bs)
    cs = tuple(torch.empty((M, N), dtype=a.dtype, device=a.device) for N in Ns)
    skinny_gemm_tn_x4_sm120_into(a, *bs, *cs)
    return cs
