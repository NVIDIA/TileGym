# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.logger import get_logger

logger = get_logger(__name__)


@ct.kernel
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: ct.Constant[int],
    K: ct.Constant[int],
    EM: int,
    num_valid_tokens: int,
    group_n: ct.Constant[int],
    group_k: ct.Constant[int],
    # Tile sizes and configuration (Meta-parameters)
    TILE_SIZE_M: ct.Constant[int],
    TILE_SIZE_N: ct.Constant[int],
    TILE_SIZE_K: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],
    MUL_ROUTED_WEIGHT: ct.Constant[int],
    top_k: ct.Constant[int],
    use_fp8_w8a8: ct.Constant[int],
    use_int8_w8a16: ct.Constant[int],
    even_Ks: ct.Constant[int],
    a_stride_1: ct.Constant[int],  # Original A shape[1] before flatten
    c_stride_1: ct.Constant[int],  # Original C shape[1] before flatten
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        tile. It determines which expert matrix from B should be used for
        each tile in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    TILE_SIZE_M, which is necessary to maintain consistency in tile matrix
    multiplication across different tiles processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `bid` to the tile of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    bid = ct.bid(axis=0)
    num_bid_m = ct.cdiv(EM, TILE_SIZE_M)
    num_bid_n = ct.cdiv(N, TILE_SIZE_N)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + ((bid % num_bid_in_group) % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m

    num_tokens_post_padded = ct.gather(num_tokens_post_padded_ptr, 0, padding_value=0)
    if bid_m * TILE_SIZE_M < num_tokens_post_padded:
        # ----------------------------------------------------------
        # Create offsets for the first tiles of A and B.
        # We will advance these offsets along the K dimension
        # and accumulate.
        offs_token_id = bid_m * TILE_SIZE_M + ct.arange(TILE_SIZE_M, dtype=ct.int32)
        offs_token = ct.gather(sorted_token_ids_ptr, offs_token_id, padding_value=0)
        token_mask = offs_token < num_valid_tokens
        off_experts = ct.load(expert_ids_ptr, index=(bid_m,), shape=(1,))
        off_experts = off_experts.item()
        row_indices = offs_token // top_k
        a_row_offset = row_indices[:, None] * a_stride_1
        mask_a = ct.broadcast_to(token_mask[:, None], (TILE_SIZE_M, TILE_SIZE_K))
        # -----------------------------------------------------------
        # Iterate to compute a tile of the C matrix.
        # We accumulate into a `[TILE_SIZE_M, TILE_SIZE_N]` tile.
        accumulator = ct.full((TILE_SIZE_M, TILE_SIZE_N), 0.0, dtype=ct.float32)
        for k in range(0, ct.cdiv(K, TILE_SIZE_K)):
            # Load the next tile of A and B
            col_indices = k * TILE_SIZE_K + ct.arange(TILE_SIZE_K, dtype=ct.int32)
            a_indices = a_row_offset + col_indices[None, :]
            a = ct.gather(a_ptr, a_indices, padding_value=0)
            b = ct.load(
                b_ptr,
                index=(off_experts, k, bid_n),
                shape=(1, TILE_SIZE_K, TILE_SIZE_N),
                order=(0, 2, 1),
            )
            b = ct.reshape(b, (TILE_SIZE_K, TILE_SIZE_N))

            accumulator = ct.mma(a, b, accumulator)
        if MUL_ROUTED_WEIGHT:
            moe_weight = ct.gather(topk_weights_ptr, offs_token, padding_value=0)
            moe_weight = ct.expand_dims(moe_weight, axis=1)
            accumulator = accumulator * moe_weight
        # -----------------------------------------------------------
        # Write back the tile of the output
        offs_cn = bid_n * TILE_SIZE_N + ct.arange(TILE_SIZE_N, dtype=ct.int32)
        c_offset = c_stride_1 * offs_token[:, None] + offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        accumulator = ct.astype(accumulator, c_ptr.dtype)
        ct.scatter(c_ptr, c_offset, accumulator)


@register_impl("invoke_fused_moe_kernel", "cutile")
def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: ct.DType,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool = False,
    block_shape: Optional[List[int]] = None,
) -> None:
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    padded_size = 0  # only uses when using fp8
    grid = (
        math.ceil(sorted_token_ids.shape[0] / config["TILE_SIZE_M"]) * math.ceil(B.shape[1] / config["TILE_SIZE_N"]),
    )

    K = B.shape[2] - padded_size
    if K % config["TILE_SIZE_K"] == 0:
        even_Ks = True
    else:
        even_Ks = False

    group_n = 0 if block_shape is None else block_shape[0]
    group_k = 0 if block_shape is None else block_shape[1]

    logger.debug(
        f"[cutile] calling fused_moe_kernel, A.shape: {A.shape}, B.shape: {B.shape}, C.shape: {C.shape}, A_scale.shape: {A_scale.shape if A_scale is not None else None}, B_scale.shape: {B_scale.shape if B_scale is not None else None}, topk_weights.shape: {topk_weights.shape}, sorted_token_ids.shape: {sorted_token_ids.shape}, expert_ids.shape: {expert_ids.shape}, num_tokens_post_padded.shape: {num_tokens_post_padded.shape}, mul_routed_weight: {mul_routed_weight}, top_k: {top_k}, config: {config}, compute_type: {compute_type}, use_fp8_w8a8: {use_fp8_w8a8}, use_int8_w8a16: {use_int8_w8a16}, block_shape: {block_shape}"
    )

    # Handle None values for scale tensors
    if A_scale is None:
        A_scale = torch.empty(0, device=A.device, dtype=A.dtype)
    if B_scale is None:
        B_scale = torch.empty(0, device=B.device, dtype=B.dtype)
    topk_weights = topk_weights.view(-1)
    C = C.view(-1, C.shape[2])

    # Save original strides before flattening
    a_stride_1 = A.shape[1]
    c_stride_1 = C.shape[1]

    # Flatten tensors to 1D for gather/scatter operations
    A_flat = A.reshape(-1)
    C_flat = C.reshape(-1)

    ct.launch(
        torch.cuda.current_stream(),
        grid,
        fused_moe_kernel,
        (
            A_flat,
            B,
            C_flat,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],  # N
            B.shape[2] - padded_size,  # K
            sorted_token_ids.shape[0],  # EM
            topk_ids.numel(),  # num_valid_tokens
            group_n,  # group_n
            group_k,  # group_k
            config["TILE_SIZE_M"],  # TILE_SIZE_M
            config["TILE_SIZE_N"],  # TILE_SIZE_N
            config["TILE_SIZE_K"],  # TILE_SIZE_K
            config["GROUP_SIZE_M"],  # GROUP_SIZE_M
            int(mul_routed_weight),  # MUL_ROUTED_WEIGHT (convert bool to int)
            top_k,  # top_k
            int(use_fp8_w8a8),  # use_fp8_w8a8 (convert bool to int)
            int(use_int8_w8a16),  # use_int8_w8a16 (convert bool to int)
            int(even_Ks),  # even_Ks (convert bool to int)
            a_stride_1,  # Original A.shape[1]
            c_stride_1,  # Original C.shape[1]
        ),
    )
