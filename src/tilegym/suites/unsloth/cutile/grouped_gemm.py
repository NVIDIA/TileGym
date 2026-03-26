# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.

"""
MoE Grouped GEMM CuTile kernels (true persistent version).

Supports: non-permute forward + backward (dX + dW).
Does NOT support: permute_x, permute_y, fuse_mul_post, TMA, autotune.

Strategy:
  - Forward/dX: true persistent kernels — grid=(NUM_SMS,), each SM handles
    multiple tiles via outer for-loop with stride NUM_SMS. Tile-to-expert
    mapping computed inside kernel by walking m_sizes. No host-side
    m_sizes.tolist() or tile counting needed.
  - dW: persistent kernel — iterates over ALL experts inside kernel.
    Grid is (n_tiles * k_tiles) — fixed, no m_sizes dependency.
  - Host only needs NUM_SMS (from device properties) and passes m_sizes
    tensor directly. Zero GPU→CPU sync.
  - Uses ct.gather/ct.scatter for data access (TMA on sliced arrays is
    broken with ct.mma in current CuTile — see task12 record for details).
  - Uses ct.mma() for matrix multiply

"""

import math

import cuda.tile as ct
import cuda.tile_experimental as ct_experimental
import torch

from tilegym.backend import register_impl

from .ct_ops import autotune_configs
from .ct_ops import next_power_of_2

ConstInt = ct.Constant[int]


# ---------------------------------------------------------------------------
# Forward kernel: Y = X @ W^T per expert (true persistent — NUM_SMS grid)
# ---------------------------------------------------------------------------


@ct.kernel
def _grouped_gemm_fwd_kernel_ct(
    X,  # (total_tokens, K)
    W_flat,  # (E*N, K) — expert weights flattened
    Y,  # (total_tokens, N)
    m_sizes,  # (E,) int32 — tokens per expert
    N: ConstInt,
    K: ConstInt,
    TOTAL_TOKENS: ConstInt,
    NUM_EXPERTS: ConstInt,
    NUM_SMS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
):
    """
    Forward: Y[tokens_for_expert] = X[tokens_for_expert] @ W[expert].T

    True persistent kernel: grid=(NUM_SMS,), each SM processes multiple tiles.
    Tile-to-expert mapping computed inside kernel by walking m_sizes.
    No host-side m_sizes.tolist() or tile counting needed.

    Inner loop walks experts to find which expert owns each tile.
    If tidx exceeds actual total tiles, expert_m_size stays 0 → no-op.
    """
    sm_id = ct.bid(0)  # 0..NUM_SMS-1
    num_n_tiles = ct.cdiv(N, BLOCK_N)

    # Compile-time upper bound on tiles per SM.
    # Actual total tiles = sum_e cdiv(m_size_e, BLOCK_M) * num_n_tiles.
    # Since cdiv(a,B) ≤ a/B + 1, the sum ≤ TOTAL_TOKENS/BLOCK_M + NUM_EXPERTS.
    # Using (cdiv(TOTAL_TOKENS, BLOCK_M) + NUM_EXPERTS) as conservative upper bound.
    max_total_tiles = (ct.cdiv(TOTAL_TOKENS, BLOCK_M) + NUM_EXPERTS) * num_n_tiles
    max_per_sm = ct.cdiv(max_total_tiles, NUM_SMS) + 1

    for tile_slot in range(max_per_sm):
        tidx = sm_id + tile_slot * NUM_SMS

        # --- Walk experts to find which expert this tidx belongs to ---
        m_offset = 0  # cumulative token offset
        tiles_before = 0  # cumulative tiles from previous experts
        expert_id = 0
        expert_m_size = 0
        m_local_idx = 0
        n_local_idx = 0

        for e in range(NUM_EXPERTS):
            m_size_tile = ct.load(m_sizes, index=(e,), shape=(1,))
            m_size = m_size_tile.item()
            m_tiles_e = ct.cdiv(m_size, BLOCK_M)
            tiles_e = m_tiles_e * num_n_tiles

            if tidx < tiles_before + tiles_e:
                if expert_m_size == 0:  # first match — acts as "break"
                    expert_id = e
                    expert_m_size = m_size
                    local_pid = tidx - tiles_before
                    m_local_idx = local_pid // num_n_tiles
                    n_local_idx = local_pid % num_n_tiles

            if tidx >= tiles_before + tiles_e:
                m_offset = m_offset + m_size
            tiles_before = tiles_before + tiles_e

        # Only compute GEMM if this tile is valid (tidx < actual total tiles)
        if expert_m_size > 0:
            m_start = m_offset + m_local_idx * BLOCK_M
            n_start = n_local_idx * BLOCK_N
            m_end_valid = m_offset + expert_m_size

            offs_m = m_start + ct.arange(BLOCK_M, dtype=ct.int32)
            offs_n = n_start + ct.arange(BLOCK_N, dtype=ct.int32)
            w_rows = expert_id * N + offs_n

            acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)

            for k_tile in range(ct.cdiv(K, BLOCK_K)):
                k_start = k_tile * BLOCK_K
                offs_k = k_start + ct.arange(BLOCK_K, dtype=ct.int32)

                x_block = ct.gather(
                    X,
                    (offs_m[:, None], offs_k[None, :]),
                    check_bounds=True,
                    padding_value=0,
                )
                w_T = ct.gather(
                    W_flat,
                    (w_rows[None, :], offs_k[:, None]),
                    check_bounds=True,
                    padding_value=0,
                )
                acc = ct.mma(x_block, w_T, acc)

            y_out = ct.astype(acc, Y.dtype)
            scatter_rows = ct.where(offs_m < m_end_valid, offs_m, TOTAL_TOKENS)
            ct.scatter(
                Y,
                (scatter_rows[:, None], offs_n[None, :]),
                y_out,
                check_bounds=True,
            )


# ---------------------------------------------------------------------------
# Backward dX kernel: dX = dY @ W per expert (true persistent — NUM_SMS grid)
# ---------------------------------------------------------------------------


@ct.kernel
def _grouped_gemm_dX_kernel_ct(
    dY,  # (total_tokens, N)
    W_flat,  # (E*N, K) — expert weights flattened
    dX,  # (total_tokens, K) — output
    m_sizes,  # (E,) int32 — tokens per expert
    N: ConstInt,
    K: ConstInt,
    TOTAL_TOKENS: ConstInt,
    NUM_EXPERTS: ConstInt,
    NUM_SMS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
):
    """
    Backward dX: dX[tokens] = dY[tokens] @ W[expert] (no transpose on W).
    True persistent kernel: grid=(NUM_SMS,), each SM processes multiple tiles.
    """
    sm_id = ct.bid(0)
    num_k_tiles = ct.cdiv(K, BLOCK_K)

    # Same conservative upper bound as forward kernel (see comment there).
    max_total_tiles = (ct.cdiv(TOTAL_TOKENS, BLOCK_M) + NUM_EXPERTS) * num_k_tiles
    max_per_sm = ct.cdiv(max_total_tiles, NUM_SMS) + 1

    for tile_slot in range(max_per_sm):
        tidx = sm_id + tile_slot * NUM_SMS

        m_offset = 0
        tiles_before = 0
        expert_id = 0
        expert_m_size = 0
        m_local_idx = 0
        k_local_idx = 0

        for e in range(NUM_EXPERTS):
            m_size_tile = ct.load(m_sizes, index=(e,), shape=(1,))
            m_size = m_size_tile.item()
            m_tiles_e = ct.cdiv(m_size, BLOCK_M)
            tiles_e = m_tiles_e * num_k_tiles

            if tidx < tiles_before + tiles_e:
                if expert_m_size == 0:
                    expert_id = e
                    expert_m_size = m_size
                    local_pid = tidx - tiles_before
                    m_local_idx = local_pid // num_k_tiles
                    k_local_idx = local_pid % num_k_tiles

            if tidx >= tiles_before + tiles_e:
                m_offset = m_offset + m_size
            tiles_before = tiles_before + tiles_e

        if expert_m_size > 0:
            m_start = m_offset + m_local_idx * BLOCK_M
            k_start = k_local_idx * BLOCK_K
            m_end_valid = m_offset + expert_m_size

            offs_m = m_start + ct.arange(BLOCK_M, dtype=ct.int32)
            offs_k = k_start + ct.arange(BLOCK_K, dtype=ct.int32)

            acc = ct.zeros((BLOCK_M, BLOCK_K), dtype=ct.float32)

            for n_tile in range(ct.cdiv(N, BLOCK_N)):
                n_start = n_tile * BLOCK_N
                offs_n = n_start + ct.arange(BLOCK_N, dtype=ct.int32)

                dy_block = ct.gather(
                    dY,
                    (offs_m[:, None], offs_n[None, :]),
                    check_bounds=True,
                    padding_value=0,
                )
                w_rows = expert_id * N + offs_n
                w_block = ct.gather(
                    W_flat,
                    (w_rows[:, None], offs_k[None, :]),
                    check_bounds=True,
                    padding_value=0,
                )
                acc = ct.mma(dy_block, w_block, acc)

            dx_out = ct.astype(acc, dX.dtype)
            scatter_rows = ct.where(offs_m < m_end_valid, offs_m, TOTAL_TOKENS)
            ct.scatter(
                dX,
                (scatter_rows[:, None], offs_k[None, :]),
                dx_out,
                check_bounds=True,
            )


# ---------------------------------------------------------------------------
# Backward dW kernel: dW = dY^T @ X per expert
# ---------------------------------------------------------------------------


@ct.kernel
def _grouped_gemm_dW_kernel_ct(
    X,  # (total_tokens, K)
    dY,  # (total_tokens, N)
    dW,  # (E*N, K) — output, flattened expert dim
    m_sizes,  # (E,) int32 — tokens per expert
    NUM_EXPERTS: ConstInt,
    N: ConstInt,
    K: ConstInt,
    TOTAL_TOKENS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
):
    """
    Backward dW: dW[expert] = dY[tokens].T @ X[tokens]

    Grid: (n_tiles * k_tiles, 1, 1) — iterate over experts inside kernel.
    Each program computes one (N_tile, K_tile) output block across ALL experts.
    """
    pid = ct.bid(0)
    num_n_tiles = ct.cdiv(N, BLOCK_N)
    num_k_tiles = ct.cdiv(K, BLOCK_K)
    # Map pid → (n_tile_idx, k_tile_idx)
    n_tile_idx = pid % num_n_tiles
    k_tile_idx = pid // num_n_tiles

    n_start = n_tile_idx * BLOCK_N
    k_start = k_tile_idx * BLOCK_K
    offs_n = n_start + ct.arange(BLOCK_N, dtype=ct.int32)
    offs_k = k_start + ct.arange(BLOCK_K, dtype=ct.int32)

    m_end = 0
    for expert_idx in range(NUM_EXPERTS):
        m_start_e = m_end
        m_size_tile = ct.load(m_sizes, index=(expert_idx,), shape=(1,))
        m_size = m_size_tile.item()
        m_end = m_start_e + m_size

        # Accumulate dW for this expert
        acc = ct.zeros((BLOCK_N, BLOCK_K), dtype=ct.float32)

        if m_size > 0:
            m_end_e = m_start_e + m_size
            for m_tile in range(ct.cdiv(m_size, BLOCK_M)):
                m_global = m_start_e + m_tile * BLOCK_M
                offs_m = m_global + ct.arange(BLOCK_M, dtype=ct.int32)
                # Mask out-of-range rows to avoid reading other experts' data.
                # dW = dY^T @ X: ALL m rows contribute to every output element,
                # so we must zero-pad rather than relying on output scatter masking.
                offs_m_safe = ct.where(offs_m < m_end_e, offs_m, TOTAL_TOKENS)

                dy_T = ct.gather(
                    dY,
                    (offs_m_safe[None, :], offs_n[:, None]),
                    check_bounds=True,
                    padding_value=0,
                )

                x_block = ct.gather(
                    X,
                    (offs_m_safe[:, None], offs_k[None, :]),
                    check_bounds=True,
                    padding_value=0,
                )

                acc = ct.mma(dy_T, x_block, acc)

        dw_out = ct.astype(acc, dW.dtype)
        expert_n_offset = expert_idx * N
        dw_row_offs = expert_n_offset + offs_n
        dw_col_offs = offs_k
        ct.scatter(
            dW,
            (dw_row_offs[:, None], dw_col_offs[None, :]),
            dw_out,
            check_bounds=True,
        )


# ---------------------------------------------------------------------------
# Host helpers
# ---------------------------------------------------------------------------


# Cache NUM_SMS per device to avoid repeated device property queries.
_num_sms_cache = {}


def _get_num_sms(device):
    """Get number of SMs for the given device, cached."""
    dev_idx = device.index if device.index is not None else torch.cuda.current_device()
    if dev_idx not in _num_sms_cache:
        _num_sms_cache[dev_idx] = torch.cuda.get_device_properties(dev_idx).multi_processor_count
    return _num_sms_cache[dev_idx]


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class GroupedGemmCT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, m_sizes, topk, gather_indices):
        X = X.contiguous()
        W = W.contiguous()

        if W.ndim == 3:
            num_experts = W.shape[0]
            N = W.shape[1]
        else:
            num_experts = m_sizes.shape[0]
            N = W.shape[0] // num_experts

        X_2d = X.view(-1, X.shape[-1])
        W_3d = W.view(num_experts, N, -1)
        total_tokens, K = X_2d.shape
        W_flat = W_3d.reshape(-1, K)  # (E*N, K)

        # Adaptive block sizes for small dimensions.
        # BLOCK_M is tuned to per-expert token count (not total_tokens) to avoid
        # excessive row-padding when tokens_per_expert << BLOCK_M. For example,
        # E=4, T=16: total_tokens=64 would give BLOCK_M=64 (75% padding per expert),
        # but avg_tokens_per_expert=16 gives BLOCK_M=16 (no padding).
        avg_tokens_per_expert = max(1, total_tokens // num_experts)
        BLOCK_M = min(64, max(16, next_power_of_2(avg_tokens_per_expert)))
        BLOCK_N = min(64, max(16, next_power_of_2(N)))
        BLOCK_K = min(64, max(16, next_power_of_2(K)))

        # Ensure m_sizes is int32 on GPU for kernel
        m_sizes_i32 = m_sizes.to(torch.int32) if m_sizes.dtype != torch.int32 else m_sizes

        Y = torch.empty((total_tokens, N), device=X.device, dtype=X.dtype)

        if total_tokens > 0:
            NUM_SMS = _get_num_sms(X.device)
            ct_experimental.autotune_launch(
                torch.cuda.current_stream(),
                grid_fn=lambda cfg: (NUM_SMS, 1, 1),
                kernel=_grouped_gemm_fwd_kernel_ct,
                args_fn=lambda cfg: (
                    X_2d,
                    W_flat,
                    Y,
                    m_sizes_i32,
                    N,
                    K,
                    total_tokens,
                    num_experts,
                    NUM_SMS,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
                hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
                search_space=autotune_configs,
            )

        ctx.save_for_backward(X, W, m_sizes, gather_indices)
        ctx.topk = topk
        ctx.num_experts = num_experts
        ctx.N = N
        ctx.K = K
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY = dY.contiguous()
        X, W, m_sizes, gather_indices = ctx.saved_tensors
        topk = ctx.topk
        num_experts = ctx.num_experts
        N = ctx.N
        K = ctx.K

        X_2d = X.view(-1, X.shape[-1])
        W_3d = W.view(num_experts, N, -1)
        total_tokens = X_2d.shape[0]
        W_flat = W_3d.reshape(-1, K)  # (E*N, K)

        # Adaptive block sizes (same logic as forward)
        avg_tokens_per_expert = max(1, total_tokens // num_experts)
        BLOCK_M = min(64, max(16, next_power_of_2(avg_tokens_per_expert)))
        BLOCK_N = min(64, max(16, next_power_of_2(N)))
        BLOCK_K = min(64, max(16, next_power_of_2(K)))

        # Ensure m_sizes is int32 on GPU
        m_sizes_i32 = m_sizes.to(torch.int32) if m_sizes.dtype != torch.int32 else m_sizes

        NUM_SMS = _get_num_sms(dY.device)

        # ----- dX = dY @ W (true persistent: grid=NUM_SMS) -----
        dX = torch.zeros((total_tokens, K), device=dY.device, dtype=dY.dtype)

        if total_tokens > 0:
            ct_experimental.autotune_launch(
                torch.cuda.current_stream(),
                grid_fn=lambda cfg: (NUM_SMS, 1, 1),
                kernel=_grouped_gemm_dX_kernel_ct,
                args_fn=lambda cfg: (
                    dY.view(-1, N),
                    W_flat,
                    dX,
                    m_sizes_i32,
                    N,
                    K,
                    total_tokens,
                    num_experts,
                    NUM_SMS,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
                hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
                search_space=autotune_configs,
            )

        # ----- dW = dY^T @ X (grid fixed, no m_sizes dependency) -----
        n_tiles = math.ceil(N / BLOCK_N)
        k_tiles_dw = math.ceil(K / BLOCK_K)
        total_dw_tiles = n_tiles * k_tiles_dw
        dW = torch.zeros((num_experts * N, K), device=dY.device, dtype=dY.dtype)

        if total_dw_tiles > 0:
            ct_experimental.autotune_launch(
                torch.cuda.current_stream(),
                grid_fn=lambda cfg: (total_dw_tiles, 1, 1),
                kernel=_grouped_gemm_dW_kernel_ct,
                args_fn=lambda cfg: (
                    X_2d,
                    dY.view(-1, N),
                    dW,
                    m_sizes_i32,
                    num_experts,
                    N,
                    K,
                    total_tokens,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                ),
                hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
                search_space=autotune_configs,
            )

        dW = dW.view(num_experts, N, K)

        if topk > 1:
            # If topk > 1, dX needs reduction along topk dim
            pass  # Simplified: no permute case, tokens already expanded

        return (
            dX,  # X
            dW,  # W
            None,  # m_sizes
            None,  # topk
            None,  # gather_indices
        )


# ---------------------------------------------------------------------------
# Public entry point with dispatch registration
# ---------------------------------------------------------------------------


@register_impl("unsloth.grouped_gemm", backend="cutile")
def grouped_gemm_cutile(
    X: torch.Tensor,
    W: torch.Tensor,
    m_sizes: torch.Tensor,
    topk: int,
    gather_indices: torch.Tensor = None,
    permute_x: bool = False,
    permute_y: bool = False,
    topk_weights: torch.Tensor = None,
    fuse_mul_post: bool = False,
) -> torch.Tensor:
    """
    CuTile grouped GEMM for MoE (simplified: non-permute only).

    """
    assert not permute_x, "CuTile grouped_gemm does not support permute_x yet"
    assert not permute_y, "CuTile grouped_gemm does not support permute_y yet"
    assert not fuse_mul_post, "CuTile grouped_gemm does not support fuse_mul_post yet"

    X = X.view(-1, X.shape[-1])
    m_sizes = m_sizes.view(-1)
    if gather_indices is not None:
        gather_indices = gather_indices.view(-1)

    return GroupedGemmCT.apply(X, W, m_sizes, topk, gather_indices)
