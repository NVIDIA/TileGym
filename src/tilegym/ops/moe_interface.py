# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

from typing import Optional
from typing import Tuple

import torch

from tilegym.backend import dispatch
from tilegym.backend import get_current_backend


@dispatch(
    "invoke_fused_moe_kernel",
)
def invoke_fused_moe_kernel(*args, **kwargs) -> None:
    raise NotImplementedError(f"invoke_fused_moe_kernel is not implemented for this backend: {get_current_backend()}")


@dispatch(
    "moe_align_block_size",
)
def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError(f"moe_align_block_size is not implemented for this backend: {get_current_backend()}")


def fused_moe_torch(A, B, C, topk_weights, topk_ids, mul_routed_weight):
    """
    Fused MoE operation using PyTorch.

    This function performs the fused MoE operation on the input tensors A, B, and C.
    It multiplies the input tensor A by the expert weights B and accumulates the result in C.
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
    - topk_weights: The weights for the topk experts with shape (*, topk), each row sums to 1.
    - topk_ids: The topk ids for each token with shape (*, topk).
    """
    M = A.shape[0]
    N = B.shape[1]
    topk = topk_ids.shape[1]

    acc = torch.einsum("mk, enk -> men", A, B)
    acc_selected = torch.zeros((M, topk, N), device=A.device, dtype=A.dtype)
    # select based on topk_ids and multiply with topk_weights
    for i in range(M):
        acc_selected[i] = acc[i, topk_ids[i]]
    if mul_routed_weight:
        acc_selected = torch.einsum("mkn, mk -> mkn", acc_selected, topk_weights)

    return acc_selected


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    hidden_states: [batch_size * seq_len, moe_intermediate_size] (9, 2048)
    w1: [n_experts, moe_intermediate_size, 2 * hidden_size] (64, 1408 * 2, 2048)
    w2: [n_experts, hidden_size, moe_intermediate_size] (64, 2048, 1408)
    topk_weights: [batch_size * seq_len, top_k] (9, 6)
    topk_ids: [batch_size * seq_len, top_k] (9, 6)
    """
    device = hidden_states.device
    if not w1.is_cuda:
        w1 = w1.to(device)
    if not w2.is_cuda:
        w2 = w2.to(device)
    if not topk_weights.is_cuda:
        topk_weights = topk_weights.to(device)
    if not topk_ids.is_cuda:
        topk_ids = topk_ids.to(device)
    # Check constraints.
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float8_e4m3fn,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]

    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 32768
    M = min(num_tokens, CHUNK_SIZE)

    out_dtype = torch.bfloat16 if hidden_states.dtype == torch.bfloat16 else torch.float16
    intermediate_cache1 = torch.empty(
        (M, topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=out_dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=out_dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], w2.shape[1]),
        device=hidden_states.device,
        dtype=out_dtype,
    )

    compute_type = torch.bfloat16 if hidden_states.dtype == torch.bfloat16 else torch.float16

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states, dtype=out_dtype)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[: tokens_in_chunk * topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        _backend = get_current_backend()
        if _backend == "cutile":
            block_size = config.get("TILE_SIZE_M", 128)
        (sorted_token_ids, expert_ids, num_tokens_post_padded, _, _) = moe_align_block_size(
            curr_topk_ids, block_size, E
        )

        invoke_fused_moe_kernel(
            curr_hidden_states,
            w1,
            intermediate_cache1,
            a1_scale,
            w1_scale,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
        )
        # Lazy import to avoid circular import
        from tilegym.ops.ops import silu_and_mul

        silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)

        if intermediate_cache2.dtype != hidden_states.dtype:
            intermediate_cache2 = intermediate_cache2.to(hidden_states.dtype)

        if use_fp8_w8a8:
            a2_scale = torch.ones(
                [
                    intermediate_cache2.shape[0],
                    intermediate_cache2.shape[1] // config["BLOCK_SIZE_N"],
                ],
                dtype=torch.float32,
                device=intermediate_cache2.device,
            )
        else:
            a2_scale = None

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            a2_scale,
            w2_scale,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
        )
        if topk_ids.shape[1] == 1:
            pass  # we write directly into out_hidden_states
        elif topk_ids.shape[1] == 2:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
            ).squeeze(dim=1)
        elif topk_ids.shape[1] > 2:
            torch.sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                dim=1,
                out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
            )
    return out_hidden_states


def fused_moe_kernel_interface(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
):
    """
    hidden_states: [batch_size, top_k, moe_intermediate_size]
    topk_ids: [batch_size, top_k]
    topk_weights: [batch_size, top_k]
    """
    backend = get_current_backend()
    if backend == "cutile":
        kernel_configs = {
            "TILE_SIZE_M": 128,
            "TILE_SIZE_N": 128,
            "TILE_SIZE_K": 64,
            "GROUP_SIZE_M": 32,
            "num_warps": 8,
            "num_stages": 4,
        }
    assert activation == "silu", "Only silu is supported for now"

    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace,
        config=kernel_configs,
    )
