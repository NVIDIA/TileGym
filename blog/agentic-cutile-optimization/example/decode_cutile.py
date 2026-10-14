# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""A first cuTile paged-KV decode attention kernel (one query token per sequence).

Written to be correct, not fast. One program per (batch, query head); it walks the
sequence's pages in order, computes scores with elementwise multiply-and-sum, and keeps an
online softmax. Grouped query attention: query head h reads KV head h // (H // H_kv).

Shapes:
  q:            (B, H, D) fp16
  k_cache, v_cache: (num_pages, PAGE, H_kv, D) fp16
  block_tables: (B, max_pages) int32, page ids; unused slots must hold a valid page id (0)
  seq_lens:     (B,) int32, number of valid KV tokens per sequence (>= 1)
  out:          (B, H, D) fp16
"""

import math

import cuda.tile as ct
import torch


@ct.kernel
def decode_kernel(
    Q,
    K,
    V,
    BT,
    SEQ,
    OUT,
    scale: float,
    max_pages: int,
    PAGE: ct.Constant[int],
    D: ct.Constant[int],
    GROUP: ct.Constant[int],
):
    b = ct.bid(0)
    h = ct.bid(1)
    kvh = h // GROUP
    q = ct.astype(ct.load(Q, index=(b, h, 0), shape=(1, 1, D)), ct.float32)  # (1,1,D)
    q = ct.reshape(q, (1, D)) * scale
    seq_len = ct.load(SEQ, index=(b,), shape=())  # scalar
    m = ct.full((1,), -math.inf, dtype=ct.float32)
    l = ct.full((1,), 0.0, dtype=ct.float32)
    acc = ct.full((1, D), 0.0, dtype=ct.float32)
    pos0 = ct.arange(PAGE, dtype=ct.int32)
    for p in range(max_pages):
        page = ct.load(BT, index=(b * max_pages + p,), shape=())  # scalar page id
        k = ct.load(K, index=(page, 0, kvh, 0), shape=(1, PAGE, 1, D))
        v = ct.load(V, index=(page, 0, kvh, 0), shape=(1, PAGE, 1, D))
        k = ct.astype(ct.reshape(k, (PAGE, D)), ct.float32)
        v = ct.astype(ct.reshape(v, (PAGE, D)), ct.float32)
        s = ct.sum(k * q, axis=1)  # (PAGE,) scores
        valid = (pos0 + p * PAGE) < seq_len
        s = ct.where(valid, s, ct.full((PAGE,), -math.inf, dtype=ct.float32))
        m_new = ct.maximum(m, ct.max(s, axis=0, keepdims=True))
        alpha = ct.exp(m - m_new)
        pr = ct.exp(s - m_new)  # (PAGE,)
        l = l * alpha + ct.sum(pr, axis=0, keepdims=True)
        pv = ct.sum(ct.reshape(pr, (PAGE, 1)) * v, axis=0, keepdims=True)  # (1, D)
        acc = acc * ct.reshape(alpha, (1, 1)) + pv
        m = m_new
    o = acc / ct.reshape(l, (1, 1))
    ct.store(OUT, index=(b, h, 0), tile=ct.astype(ct.reshape(o, (1, 1, D)), OUT.dtype))


def paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, scale=None):
    B, H, D = q.shape
    num_pages, PAGE, H_kv, D2 = k_cache.shape
    assert D == D2 and H % H_kv == 0 and q.dtype == torch.float16
    assert block_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    bt = block_tables.contiguous().reshape(-1)
    max_pages = int(block_tables.shape[1])
    GROUP = H // H_kv
    args = (q.contiguous(), k_cache, v_cache, bt, seq_lens.contiguous(), out, float(scale), max_pages, PAGE, D, GROUP)
    ct.launch(torch.cuda.current_stream(), (B, H, 1), decode_kernel, args)
    return out


def reference_decode(q, k_cache, v_cache, block_tables, seq_lens, scale=None):
    """Dense PyTorch reference: gather pages, mask by seq_len, softmax in fp32."""
    B, H, D = q.shape
    _, PAGE, H_kv, _ = k_cache.shape
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    S = block_tables.shape[1] * PAGE
    k = k_cache[block_tables.long()].reshape(B, S, H_kv, D).float()  # (B,S,Hkv,D)
    v = v_cache[block_tables.long()].reshape(B, S, H_kv, D).float()
    g = H // H_kv
    k = k.repeat_interleave(g, dim=2)  # (B,S,H,D)
    v = v.repeat_interleave(g, dim=2)
    scores = torch.einsum("bhd,bshd->bhs", q.float(), k) * scale
    pos = torch.arange(S, device=q.device)[None, None, :]
    scores = scores.masked_fill(pos >= seq_lens[:, None, None], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bshd->bhd", p, v).to(q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, H_kv, D, PAGE = 2, 8, 2, 128, 16
    seq_lens = torch.tensor([100, 37], dtype=torch.int32, device="cuda")
    max_pages = math.ceil(int(seq_lens.max()) / PAGE)
    k_cache = torch.randn(64, PAGE, H_kv, D, device="cuda", dtype=torch.float16)
    v_cache = torch.randn(64, PAGE, H_kv, D, device="cuda", dtype=torch.float16)
    block_tables = torch.randperm(64, device="cuda")[: B * max_pages].reshape(B, max_pages).to(torch.int32)
    q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    out = paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens)
    ref = reference_decode(q, k_cache, v_cache, block_tables, seq_lens)
    print("max abs diff vs reference:", (out.float() - ref.float()).abs().max().item())
