# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# fmt: off
# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/benchmarks


import torch


def sample_actual_seq_lens(max_seqlen, batch_size, device, random_actual_seq_len):
    """
    Get an array of actual sequence lengths for given batch size and max sequence length.
    If random_actual_seq_len is True, sample actual sequence lengths randomly.
    Otherwise, set all actual sequence lengths to max_seqlen.

    Args:
        max_seqlen: Maximum sequence length.
        batch_size: Batch size.
        device: Device to sample on.
        random_actual_seq_len: Whether to sample actual sequence lengths randomly.

    Returns:
        actual_seq_lens: Actual sequence lengths for each batch.
    """
    if random_actual_seq_len:
        actual_seq_lens = torch.randint(
            1, max_seqlen + 1, (batch_size, 1, 1, 1), device=device, dtype=torch.int32
        )
    else:
        actual_seq_lens = torch.full(
            (batch_size, 1, 1, 1), max_seqlen, device=device, dtype=torch.int32
        )
    return actual_seq_lens


def get_paged_kv(batch_size, s_kv, num_kv_heads, page_size, dim_head_qk, dim_head_vo, head_dim_rope, kv_init_dtype, device):

    # Create KV cache
    num_pages_per_seq = (s_kv + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    # Now initialize the page tables
    block_tables = torch.tensor(
        [
            [k + i * num_pages_per_seq for k in range(num_pages_per_seq)]
            for i in range(batch_size)
        ],
        dtype=torch.int32,
        device=device,
    )
    # Initialize KV cache with appropriate shape and stride
    k_cache_shape = (
        total_num_pages,
        num_kv_heads,
        page_size,
        dim_head_qk,
    )
    k_cache = torch.randn(size=k_cache_shape, dtype=kv_init_dtype).to(device)
    if head_dim_rope > 0:
        k_rope_shape = (
            total_num_pages,
            num_kv_heads,
            page_size,
            head_dim_rope,
        )
        k_rope = torch.randn(size=k_rope_shape, dtype=kv_init_dtype).to(device)
        return k_cache, k_rope, block_tables
    else:
        v_cache_shape = (
            total_num_pages,
            num_kv_heads,
            page_size,
            dim_head_vo,
        )
        v_cache = torch.randn(size=v_cache_shape, dtype=kv_init_dtype).to(device)

        return k_cache, v_cache, block_tables

def generate_sample_data(
    batch_size,
    max_seq_len=1024,
    page_size=None,
    dtype=torch.bfloat16,
    group_size=16,
    heads=128,
    head_dim_qk=128,
    head_dim_vo=128,
    head_dim_rope=0,
    device="cuda",
    is_decode=False,
):
    q_init_dtype = torch.float16
    kv_init_dtype = torch.float16

    s_kv = max_seq_len
    num_qo_heads = heads
    num_kv_heads = heads // group_size

    # Sample sequence lengths and create tensors
    actual_seq_lens = sample_actual_seq_lens(
        s_kv, batch_size, device, random_actual_seq_len=True
    )
    cumsum_s_kv = torch.sum(actual_seq_lens)
    q = torch.randn(
        batch_size if is_decode else cumsum_s_kv,
        num_qo_heads,
        head_dim_qk,
        device=device,
        dtype=q_init_dtype,
    )

    actual_seq_offset = torch.arange(0, batch_size + 1, device=device)
    actual_seq_offset[1:] = torch.cumsum(actual_seq_lens.view(-1), dim=0)

    if page_size is None:
        k_cache = torch.randn(
            cumsum_s_kv, num_kv_heads, head_dim_qk, device=device, dtype=kv_init_dtype
        )
        v_cache = torch.randn(
            cumsum_s_kv, num_kv_heads, head_dim_vo, device=device, dtype=kv_init_dtype
        )
        block_tables = None
    else:
        k_cache, v_cache, block_tables = get_paged_kv(
            batch_size, s_kv, num_kv_heads, page_size, head_dim_qk, head_dim_vo, head_dim_rope, kv_init_dtype, device)

    q_indptr = actual_seq_offset * (head_dim_qk * num_qo_heads) # For cuDNN
    k_indptr = actual_seq_offset * (head_dim_qk * num_kv_heads) # For cuDNN
    v_indptr = actual_seq_offset * (head_dim_vo * num_kv_heads) # For cuDNN
    o_indptr = actual_seq_offset * (head_dim_vo * num_qo_heads) # For cuDNN
    lse_indptr = actual_seq_offset * num_qo_heads # For cuDNN
    q_indptr = q_indptr.long()
    k_indptr = k_indptr.long()
    v_indptr = v_indptr.long()
    o_indptr = o_indptr.long()
    lse_indptr = lse_indptr.long()

    head_dim = head_dim_qk + head_dim_rope
    scale = float(1.0 / (head_dim**0.5))
    if head_dim_rope > 0 and page_size is not None:
        q_rope = torch.randn(
            batch_size if is_decode else cumsum_s_kv, num_qo_heads, head_dim_rope, device=device, dtype=kv_init_dtype
        )
        return (
            (q.to(dtype), q_rope.to(dtype)),
            k_cache.to(dtype),
            v_cache.to(dtype),
            scale,
            actual_seq_lens,
            block_tables,
            actual_seq_offset,
            q_indptr,
            k_indptr,
            v_indptr,
            o_indptr,
            lse_indptr,
        )

    else:
        return (
            q.to(dtype),
            k_cache.to(dtype),
            v_cache.to(dtype),
            scale,
            actual_seq_lens,
            block_tables,
            actual_seq_offset,
            q_indptr,
            k_indptr,
            v_indptr,
            o_indptr,
            lse_indptr,
        )
