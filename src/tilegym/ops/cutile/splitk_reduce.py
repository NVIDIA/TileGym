# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import cuda.tile as ct
import torch

from tilegym.backend import register_impl

from .utils import next_power_of_2

# Type aliases for constants
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]


@ct.kernel(occupancy=4)
def splitk_reduce_kernel(
    attn_splitk_out,
    lse_splitk_out,
    attn_out,
    B: ConstInt,
    S_kv: ConstInt,
    num_heads: ConstInt,
    head_dim: ConstInt,
    NUM_KV_SPLITS: ConstInt,
    NUM_KV_SPLITS_POW2: ConstInt,
    TILE_D: ConstInt,
    USE_DOT: ConstBool,
):
    # Get program IDs
    batch_id = ct.bid(0)  # batch index
    head_id = ct.bid(1)  # head index
    tile_id = ct.bid(2)  # tile index

    # Get data type
    dtype = attn_out.dtype

    # Load intermediate attention results with latency hint
    out_splitk = ct.load(
        attn_splitk_out,
        (batch_id, head_id, 0, tile_id),
        shape=(1, 1, NUM_KV_SPLITS_POW2, TILE_D),
        order=(0, 1, 2, 3),
        allow_tma=True,
        latency=2,
    )
    out_splitk = ct.reshape(out_splitk, (NUM_KV_SPLITS_POW2, TILE_D))

    # Load and process lse results
    offs_lse = ct.arange(NUM_KV_SPLITS_POW2, dtype=ct.int32)
    lse_splitk = ct.gather(
        lse_splitk_out,
        (batch_id, head_id, offs_lse),
        padding_value=-math.inf,
    )

    # Compute lse_max
    lse_max = ct.max(lse_splitk)

    # Compute sumexp_normalized_splitk
    sumexp_normalized_splitk = ct.exp2(lse_splitk - lse_max)
    sumexp_normalized_splitk = ct.astype(sumexp_normalized_splitk, ct.float32)

    # Compute sumexp_normalized
    sumexp_normalized = ct.sum(sumexp_normalized_splitk)

    # Compute numerator_normalized
    if USE_DOT:
        mma_result = ct.mma(
            sumexp_normalized_splitk[None, :],
            ct.astype(out_splitk, ct.float32),
            ct.zeros((1, TILE_D), dtype=ct.float32),
        )
        numerator_normalized = ct.extract(mma_result, (0, 0), shape=(1, TILE_D))
    else:
        numerator_normalized = ct.sum(
            out_splitk * ct.reshape(sumexp_normalized_splitk, (NUM_KV_SPLITS_POW2, 1)),
            axis=0,
        )

    # Compute final accumulator
    acc = numerator_normalized / sumexp_normalized

    # Cast to output dtype before storing
    acc = ct.astype(acc, dtype)

    # Store final result with latency hint
    ct.store(
        attn_out,
        index=(batch_id, head_id, tile_id),
        tile=ct.reshape(acc, (1, 1, TILE_D)),
        order=(0, 1, 2),
        allow_tma=True,
        latency=2,
    )


@register_impl("splitk_reduce", backend="cutile")
def splitk_reduce(attn_splitk_out, lse_splitk_out, attn_out, S_kv, **kwargs):
    """
    Reduce the intermediate attention results and lse results into the final output for attention decode
    Args:
        attn_splitk_out: intermediate attention results [B, num_heads, NUM_KV_SPLITS, head_dim]
        lse_splitk_out: intermediate lse results [B, num_heads, NUM_KV_SPLITS]
        attn_out: final output [B, num_heads, head_dim]
        S_kv: sequence length of the key-value tensor, used for boundary check
    """
    B, num_heads, NUM_KV_SPLITS, head_dim = attn_splitk_out.shape
    TILE_D = min(128, next_power_of_2(head_dim))
    NUM_KV_SPLITS_POW2 = next_power_of_2(NUM_KV_SPLITS)

    # Determine if we should use dot product based on conditions
    USE_DOT = NUM_KV_SPLITS_POW2 >= 16

    # Calculate grid dimensions
    grid = (B, num_heads, (head_dim + TILE_D - 1) // TILE_D)

    # Launch kernel
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        splitk_reduce_kernel,
        (
            attn_splitk_out,
            lse_splitk_out,
            attn_out,
            B,
            S_kv,
            num_heads,
            head_dim,
            NUM_KV_SPLITS,
            NUM_KV_SPLITS_POW2,
            TILE_D,
            USE_DOT,
        ),
    )

    return attn_out
