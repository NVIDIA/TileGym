# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""cuTile paged-KV decode attention, tuned on NVIDIA B200.

q: (B, H, D) fp16; caches: (pages, PAGE, H_kv, D) fp16;
block_tables: (B, max_pages) int32; seq_lens: (B,) int32, >= 1.
Unused page-table entries must contain a valid page ID, as in the original kernel.
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
    PART,
    STATS,
    scale: float,
    MAX_PAGES: ct.Constant[int],
    PAGE: ct.Constant[int],
    D: ct.Constant[int],
    GROUP: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    TOKENS_PER_SPLIT: ct.Constant[int],
    SINGLE_SPLIT: ct.Constant[bool],
    KV_GROUP: ct.Constant[int],
):
    b, kvh, split = ct.bid(0), ct.bid(1), ct.bid(2)
    q = ct.load(
        Q,
        (b, kvh, 0, 0),
        shape=(1, 1, BLOCK_H, D),
        padding_mode=ct.PaddingMode.ZERO,
    )
    q = ct.transpose(ct.reshape(q, (BLOCK_H, D)))
    cache_head = kvh * KV_GROUP
    seq_len = ct.minimum(ct.load(SEQ, (b,), shape=()), MAX_PAGES * PAGE)
    m = ct.full((BLOCK_H,), -math.inf, dtype=ct.float32)
    l = ct.full((BLOCK_H,), 0.0, dtype=ct.float32)
    acc = ct.full((D, BLOCK_H), 0.0, dtype=ct.float32)
    start = split * TOKENS_PER_SPLIT
    end = ct.minimum(start + TOKENS_PER_SPLIT, seq_len)
    offs = ct.arange(BLOCK_N * KV_GROUP, dtype=ct.int32)
    heads = ct.arange(BLOCK_H, dtype=ct.int32)
    pg = ct.arange(BLOCK_N // PAGE, dtype=ct.int32)
    for pos in range(start, end, BLOCK_N):
        pidx = pos // PAGE + pg
        page_ids = ct.gather(
            BT,
            (b * MAX_PAGES + pidx,),
            mask=pidx < MAX_PAGES,
            padding_value=0,
        )
        k = ct.load_advanced_indexing(
            K,
            (page_ids, ct.Slice(0, PAGE), ct.Slice(cache_head, KV_GROUP), ct.Slice(0, D)),
            padding_mode=ct.PaddingMode.ZERO,
        )
        k = ct.reshape(k, (BLOCK_N * KV_GROUP, D))
        # Put the wide KV extent on MMA's M axis, not the thin query-head extent.
        scores = ct.mma(
            k,
            q,
            ct.full((BLOCK_N * KV_GROUP, BLOCK_H), 0.0, dtype=ct.float32),
        )
        scores = scores * (scale * 1.4426950408889634)
        # Cache rows interleave adjacent KV heads; each query attends only its head.
        valid = ((pos + offs[:, None] // KV_GROUP) < seq_len) & (
            (offs[:, None] % KV_GROUP) == (heads[None, :] // GROUP)
        )
        scores = ct.where(valid, scores, -math.inf)
        # Dummy query columns are discarded on store. Keep their recurrence finite.
        scores = ct.where(heads[None, :] < GROUP * KV_GROUP, scores, 0.0)
        m_new = ct.maximum(m, ct.max(scores, axis=0))
        alpha = ct.exp2(m - m_new)
        probs = ct.exp2(scores - m_new[None, :])
        l = l * alpha + ct.sum(probs, axis=0)
        v = ct.load_advanced_indexing(
            V,
            (page_ids, ct.Slice(0, PAGE), ct.Slice(cache_head, KV_GROUP), ct.Slice(0, D)),
            padding_mode=ct.PaddingMode.ZERO,
        )
        v = ct.transpose(ct.reshape(v, (BLOCK_N * KV_GROUP, D)))
        acc = ct.mma(v, ct.astype(probs, ct.float16), acc * alpha[None, :])
        m = m_new
    if SINGLE_SPLIT:
        result = ct.astype(ct.transpose(acc / l[None, :]), OUT.dtype)
        ct.store(OUT, (b, kvh, 0, 0), ct.reshape(result, (1, 1, BLOCK_H, D)))
    else:
        # Empty splits emit acc=0, l=0, m=-inf; the merge explicitly masks them.
        ct.store(
            PART,
            (b, kvh, split, 0, 0),
            ct.reshape(ct.transpose(acc), (1, 1, 1, BLOCK_H, D)),
        )
        ct.store(STATS, (b, kvh, split, 0, 0), ct.reshape(m, (1, 1, 1, BLOCK_H, 1)))
        ct.store(STATS, (b, kvh, split, 0, 1), ct.reshape(l, (1, 1, 1, BLOCK_H, 1)))


@ct.kernel(occupancy=4)
def merge_kernel(
    PART,
    STATS,
    OUT,
    D: ct.Constant[int],
    GROUP: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    SPLITS: ct.Constant[int],
    TILE_S: ct.Constant[int],
    USE_TMA: ct.Constant[bool],
):
    b, h = ct.bid(0), ct.bid(1)
    kvh, gh = h // GROUP, h % GROUP
    stats = ct.load(
        STATS,
        (b, kvh, 0, gh, 0),
        shape=(1, 1, TILE_S, 1, 2),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=USE_TMA,
    )
    stats = ct.reshape(stats, (TILE_S, 2))
    m = ct.reshape(ct.extract(stats, (0, 0), shape=(TILE_S, 1)), (TILE_S,))
    l = ct.reshape(ct.extract(stats, (0, 1), shape=(TILE_S, 1)), (TILE_S,))
    valid = (ct.arange(TILE_S, dtype=ct.int32) < SPLITS) & (l > 0)
    m = ct.where(valid, m, -math.inf)
    weights = ct.exp2(m - ct.max(m, axis=0))
    part = ct.load(
        PART,
        (b, kvh, 0, gh, 0),
        shape=(1, 1, TILE_S, 1, D),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=USE_TMA,
    )
    part = ct.reshape(part, (TILE_S, D))
    acc = ct.sum(part * weights[:, None], axis=0)
    denom = ct.sum(l * weights, axis=0)
    out = ct.astype(acc / denom, OUT.dtype)
    ct.store(OUT, (b, h, 0), ct.reshape(out, (1, 1, D)), allow_tma=USE_TMA)


def paged_decode_attention(q, k_cache, v_cache, block_tables, seq_lens, scale=None):
    B, H, D = q.shape
    _, PAGE, H_kv, D2 = k_cache.shape
    assert D == D2 and H % H_kv == 0 and q.dtype == torch.float16
    assert block_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    bt = block_tables.contiguous().reshape(-1)
    max_pages = int(block_tables.shape[1])
    group = H // H_kv
    capacity = max_pages * PAGE

    # Offline B200 measurements: packing adjacent KV heads improves page-load
    # efficiency, but large packs underfill the GPU on short/small-batch workloads.
    if B >= 8 and (capacity >= 16384 or capacity <= 2048):
        desired_kv_group = 4
    else:
        desired_kv_group = 2 if capacity >= 8192 else 1
    kv_group = 1
    if group == 4:
        if desired_kv_group >= 4 and H_kv % 4 == 0 and PAGE <= 64:
            kv_group = 4
        elif desired_kv_group >= 2 and H_kv % 2 == 0 and PAGE <= 128:
            kv_group = 2
    kv_blocks = H_kv // kv_group
    combined_heads = group * kv_group
    block_h = max(8, 1 << (combined_heads - 1).bit_length())
    block_n = max(256 // kv_group, PAGE)
    if B < 8:
        tokens_per_split = 1024 if capacity >= 16384 else 256
    else:
        tokens_per_split = 8192 if B >= 64 else 2048
    tokens_per_split = max(tokens_per_split, block_n)
    splits = (capacity + tokens_per_split - 1) // tokens_per_split
    single_split = splits == 1
    if single_split:
        # These arguments are unused in the constant-folded single-split path.
        part = stats = out.reshape(B, kv_blocks, 1, combined_heads, D)
    else:
        part = torch.empty(
            (B, kv_blocks, splits, combined_heads, D),
            device=q.device,
            dtype=torch.float32,
        )
        stats = torch.empty(
            (B, kv_blocks, splits, combined_heads, 2),
            device=q.device,
            dtype=torch.float32,
        )
    stream = torch.cuda.current_stream(q.device)
    args = (
        q.contiguous().reshape(B, kv_blocks, combined_heads, D),
        k_cache,
        v_cache,
        bt,
        seq_lens.contiguous(),
        out.reshape(B, kv_blocks, combined_heads, D),
        part,
        stats,
        float(scale),
        max_pages,
        PAGE,
        D,
        group,
        block_h,
        block_n,
        tokens_per_split,
        single_split,
        kv_group,
    )
    ct.launch(stream, (B, kv_blocks, splits), decode_kernel, args)
    if not single_split:
        # Small merges favor ordinary loads/stores; larger tiles favor TMA.
        merge_args = (
            part,
            stats,
            out,
            D,
            combined_heads,
            block_h,
            splits,
            1 << (splits - 1).bit_length(),
            splits > 16,
        )
        ct.launch(stream, (B, H, 1), merge_kernel, merge_args)
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
