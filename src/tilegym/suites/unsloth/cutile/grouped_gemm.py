# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Original source: https://github.com/unslothai/unsloth
# Original license: Apache License 2.0
# Original copyright: Copyright 2023-present Daniel Han-Chen & the Unsloth team.

"""
MoE Grouped GEMM CuTile kernels (true persistent version).

Supports: forward + backward (dX + dW) with optional permute_x or permute_y.
Does NOT support: fuse_mul_post, TMA, autotune.

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

permute_x support (first GEMM — input gather):
  - Forward: X gathered from original token positions via gather_indices // topk,
    Y stored to sorted (expert-grouped) positions.
  - Backward dX: dY loaded from sorted position, dX stored to original token
    positions via gather_indices.
  - Backward dW: X loaded from original positions via gather_indices // topk,
    dY loaded from sorted positions.
  - When topk > 1: backward dX output is (num_tokens * topk, K), then summed
    along topk dim at host level.

permute_y support (second GEMM — output scatter):
  - Forward: X loaded from sorted position, Y scattered to original token
    position via gather_indices.
  - Backward dX: dY loaded from original token position via gather_indices,
    dX stored to sorted position.
  - Backward dW: X loaded from sorted position, dY loaded from original
    token position via gather_indices.

"""

import math

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.backend import register_impl

from .ct_ops import autotune_configs
from .ct_ops import next_power_of_2

# Module-level tune caches for forward, dX backward, and dW backward
_grouped_gemm_fwd_tune_cache: dict = {}
_grouped_gemm_dX_tune_cache: dict = {}
_grouped_gemm_dW_tune_cache: dict = {}


def _gemm_block_sizes(N, K, avg_tokens):
    """Select block sizes heuristically based on problem shape.

    Uses a single well-chosen config instead of exhaustive search to avoid
    compilation timeouts (each unique block-size combo recompiles the kernel).
    """
    BLOCK_M = min(128, max(64, next_power_of_2(avg_tokens)))
    BLOCK_N = min(128, max(64, next_power_of_2(N)))
    BLOCK_K = min(128, max(64, next_power_of_2(K)))
    return BLOCK_M, BLOCK_N, BLOCK_K


ConstInt = ct.Constant[int]


# ---------------------------------------------------------------------------
# Forward kernel: Y = X @ W^T per expert (true persistent — NUM_SMS grid)
# ---------------------------------------------------------------------------


@ct.kernel
def _grouped_gemm_fwd_kernel_ct(
    X,  # (num_tokens or total_tokens, K) depending on permute_x
    W_flat,  # (E*N, K) — expert weights flattened
    Y,  # (total_tokens, N)
    m_sizes,  # (E,) int32 — tokens per expert
    gather_indices,  # (total_tokens,) int32 — token permutation indices
    N: ConstInt,
    K: ConstInt,
    TOTAL_TOKENS: ConstInt,
    NUM_EXPERTS: ConstInt,
    NUM_SMS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
    PERMUTE_X: ConstInt,
    PERMUTE_Y: ConstInt,
    TOPK: ConstInt,
):
    """
    Forward: Y[tokens_for_expert] = X[tokens_for_expert] @ W[expert].T

    True persistent kernel: grid=(NUM_SMS,), each SM processes multiple tiles.

    When PERMUTE_X=1:
      - X is gathered from original token positions via gather_indices // topk
      - Y is stored to sorted (expert-grouped) positions

    When PERMUTE_Y=1:
      - X is loaded from the sorted (expert-grouped) position
      - Y is scattered to original token positions via gather_indices
    """
    sm_id = ct.bid(0)  # 0..NUM_SMS-1
    num_n_tiles = ct.cdiv(N, BLOCK_N)

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
            n_start = n_local_idx * BLOCK_N
            offs_n = n_start + ct.arange(BLOCK_N, dtype=ct.int32)
            w_rows = expert_id * N + offs_n

            if PERMUTE_X == 1:
                # --- permute_x: gather X from original positions, store Y to sorted ---
                gather_offsets = m_local_idx * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
                indices_to_gather = m_offset + (gather_offsets % expert_m_size)
                expert_token_idx = ct.gather(
                    gather_indices,
                    (indices_to_gather,),
                    check_bounds=True,
                    padding_value=0,
                )
                row_valid = gather_offsets < expert_m_size

                # Load X from original token positions (expert_token_idx // topk)
                load_rows = expert_token_idx // TOPK
                # Store Y to sorted positions (indices_to_gather)
                store_rows = ct.where(row_valid, indices_to_gather, TOTAL_TOKENS)
            elif PERMUTE_Y == 1:
                # --- permute_y: load X from sorted position, store Y via gather_indices ---
                gather_offsets = m_local_idx * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
                indices_to_gather = m_offset + (gather_offsets % expert_m_size)
                expert_token_idx = ct.gather(
                    gather_indices,
                    (indices_to_gather,),
                    check_bounds=True,
                    padding_value=0,
                )
                row_valid = gather_offsets < expert_m_size

                # Load X from sorted positions (indices_to_gather)
                load_rows = indices_to_gather
                # Store Y to original token positions (expert_token_idx)
                store_rows = ct.where(row_valid, expert_token_idx, TOTAL_TOKENS)
            else:
                # --- non-permute: sequential load and store ---
                m_start = m_offset + m_local_idx * BLOCK_M
                m_end_valid = m_offset + expert_m_size
                load_rows = m_start + ct.arange(BLOCK_M, dtype=ct.int32)
                store_rows = ct.where(load_rows < m_end_valid, load_rows, TOTAL_TOKENS)

            acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)

            for k_tile in range(ct.cdiv(K, BLOCK_K)):
                k_start = k_tile * BLOCK_K
                offs_k = k_start + ct.arange(BLOCK_K, dtype=ct.int32)

                x_block = ct.gather(
                    X,
                    (load_rows[:, None], offs_k[None, :]),
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
            ct.scatter(
                Y,
                (store_rows[:, None], offs_n[None, :]),
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
    gather_indices,  # (total_tokens,) int32 — token permutation indices
    N: ConstInt,
    K: ConstInt,
    TOTAL_TOKENS: ConstInt,
    NUM_EXPERTS: ConstInt,
    NUM_SMS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
    PERMUTE_X: ConstInt,
    PERMUTE_Y: ConstInt,
):
    """
    Backward dX: dX[tokens] = dY[tokens] @ W[expert] (no transpose on W).
    True persistent kernel: grid=(NUM_SMS,), each SM processes multiple tiles.

    When PERMUTE_X=1:
      - dY is loaded from sorted (expert-grouped) positions
      - dX is stored to original token positions via gather_indices

    When PERMUTE_Y=1:
      - dY is loaded from original token positions via gather_indices
      - dX is stored to sorted (expert-grouped) positions
    """
    sm_id = ct.bid(0)
    num_k_tiles = ct.cdiv(K, BLOCK_K)

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
            k_start = k_local_idx * BLOCK_K
            offs_k = k_start + ct.arange(BLOCK_K, dtype=ct.int32)

            if PERMUTE_X == 1:
                # --- permute_x: load dY from sorted, store dX to original positions ---
                gather_offsets = m_local_idx * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
                indices_to_gather = m_offset + (gather_offsets % expert_m_size)
                expert_token_idx = ct.gather(
                    gather_indices,
                    (indices_to_gather,),
                    check_bounds=True,
                    padding_value=0,
                )
                row_valid = gather_offsets < expert_m_size

                # Load dY from sorted positions (indices_to_gather)
                dy_load_rows = indices_to_gather
                # Store dX to original token positions (expert_token_idx)
                dx_store_rows = ct.where(row_valid, expert_token_idx, TOTAL_TOKENS)
            elif PERMUTE_Y == 1:
                # --- permute_y: load dY from original positions, store dX to sorted ---
                gather_offsets = m_local_idx * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
                indices_to_gather = m_offset + (gather_offsets % expert_m_size)
                expert_token_idx = ct.gather(
                    gather_indices,
                    (indices_to_gather,),
                    check_bounds=True,
                    padding_value=0,
                )
                row_valid = gather_offsets < expert_m_size

                # Load dY from original token positions (expert_token_idx)
                dy_load_rows = expert_token_idx
                # Store dX to sorted positions (indices_to_gather)
                dx_store_rows = ct.where(row_valid, indices_to_gather, TOTAL_TOKENS)
            else:
                # --- non-permute: sequential load and store ---
                m_start = m_offset + m_local_idx * BLOCK_M
                m_end_valid = m_offset + expert_m_size
                offs_m = m_start + ct.arange(BLOCK_M, dtype=ct.int32)
                dy_load_rows = offs_m
                dx_store_rows = ct.where(offs_m < m_end_valid, offs_m, TOTAL_TOKENS)

            acc = ct.zeros((BLOCK_M, BLOCK_K), dtype=ct.float32)

            for n_tile in range(ct.cdiv(N, BLOCK_N)):
                n_start = n_tile * BLOCK_N
                offs_n = n_start + ct.arange(BLOCK_N, dtype=ct.int32)

                dy_block = ct.gather(
                    dY,
                    (dy_load_rows[:, None], offs_n[None, :]),
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
            ct.scatter(
                dX,
                (dx_store_rows[:, None], offs_k[None, :]),
                dx_out,
                check_bounds=True,
            )


# ---------------------------------------------------------------------------
# Backward dW kernel: dW = dY^T @ X per expert
# ---------------------------------------------------------------------------


@ct.kernel
def _grouped_gemm_dW_kernel_ct(
    X,  # (num_tokens or total_tokens, K)
    dY,  # (total_tokens, N)
    dW,  # (E*N, K) — output, flattened expert dim
    m_sizes,  # (E,) int32 — tokens per expert
    gather_indices,  # (total_tokens,) int32 — token permutation indices
    NUM_EXPERTS: ConstInt,
    N: ConstInt,
    K: ConstInt,
    TOTAL_TOKENS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_K: ConstInt,
    PERMUTE_X: ConstInt,
    PERMUTE_Y: ConstInt,
    TOPK: ConstInt,
):
    """
    Backward dW: dW[expert] = dY[tokens].T @ X[tokens]

    Grid: (n_tiles * k_tiles, 1, 1) — iterate over experts inside kernel.
    Each program computes one (N_tile, K_tile) output block across ALL experts.

    When PERMUTE_X=1:
      - X is loaded from original token positions via gather_indices // topk
      - dY is loaded from sorted positions

    When PERMUTE_Y=1:
      - X is loaded from sorted positions
      - dY is loaded from original token positions via gather_indices
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

                if PERMUTE_X == 1:
                    # --- permute_x: X from original positions, dY from sorted ---
                    gather_offsets = m_tile * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
                    indices_to_gather = m_start_e + (gather_offsets % m_size)
                    expert_token_idx = ct.gather(
                        gather_indices,
                        (indices_to_gather,),
                        check_bounds=True,
                        padding_value=0,
                    )
                    row_valid = gather_offsets < m_size

                    # X loaded from original token positions (expert_token_idx // topk)
                    x_load_rows = ct.where(row_valid, expert_token_idx // TOPK, TOTAL_TOKENS)
                    # dY loaded from sorted positions
                    m_offsets = m_global + ct.arange(BLOCK_M, dtype=ct.int32)
                    dy_load_rows = ct.where(row_valid, m_offsets, TOTAL_TOKENS)
                elif PERMUTE_Y == 1:
                    # --- permute_y: X from sorted, dY from original via gather_indices ---
                    gather_offsets = m_tile * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
                    indices_to_gather = m_start_e + (gather_offsets % m_size)
                    expert_token_idx = ct.gather(
                        gather_indices,
                        (indices_to_gather,),
                        check_bounds=True,
                        padding_value=0,
                    )
                    row_valid = gather_offsets < m_size

                    # X loaded from sorted positions (indices_to_gather)
                    x_load_rows = ct.where(row_valid, indices_to_gather, TOTAL_TOKENS)
                    # dY loaded from original token positions (expert_token_idx)
                    dy_load_rows = ct.where(row_valid, expert_token_idx, TOTAL_TOKENS)
                else:
                    # --- non-permute: sequential access ---
                    offs_m = m_global + ct.arange(BLOCK_M, dtype=ct.int32)
                    offs_m_safe = ct.where(offs_m < m_end_e, offs_m, TOTAL_TOKENS)
                    x_load_rows = offs_m_safe
                    dy_load_rows = offs_m_safe

                dy_T = ct.gather(
                    dY,
                    (dy_load_rows[None, :], offs_n[:, None]),
                    check_bounds=True,
                    padding_value=0,
                )

                x_block = ct.gather(
                    X,
                    (x_load_rows[:, None], offs_k[None, :]),
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
    def forward(
        ctx, X, W, m_sizes, topk, gather_indices, permute_x=False, permute_y=False, dX_only=False, dW_only=False
    ):
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
        _, K = X_2d.shape
        W_flat = W_3d.reshape(-1, K)  # (E*N, K)

        # Determine total_tokens from gather_indices or m_sizes
        if permute_x or permute_y:
            total_tokens = gather_indices.shape[0]
        else:
            total_tokens = X_2d.shape[0]

        # Ensure m_sizes is int32 on GPU for kernel
        m_sizes_i32 = m_sizes.to(torch.int32) if m_sizes.dtype != torch.int32 else m_sizes

        # Block sizes: heuristic selection (single config to avoid recompilation).
        avg_tokens_per_expert = max(1, total_tokens // num_experts)
        BLOCK_M, BLOCK_N, BLOCK_K = _gemm_block_sizes(N, K, avg_tokens_per_expert)

        Y = torch.empty((total_tokens, N), device=X.device, dtype=X.dtype)

        # Prepare gather_indices for kernel (dummy if not permuting)
        if gather_indices is None:
            gather_indices_i32 = torch.arange(total_tokens, dtype=torch.int32, device=X.device)
        else:
            gather_indices_i32 = (
                gather_indices.to(torch.int32) if gather_indices.dtype != torch.int32 else gather_indices
            )

        permute_x_flag = 1 if permute_x else 0
        permute_y_flag = 1 if permute_y else 0

        if total_tokens > 0:
            NUM_SMS = _get_num_sms(X.device)
            fwd_stream = torch.cuda.current_stream()
            fwd_cache_key = (
                total_tokens,
                N,
                K,
                num_experts,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                permute_x_flag,
                permute_y_flag,
                topk,
                X.dtype,
                str(X.device),
            )
            if fwd_cache_key not in _grouped_gemm_fwd_tune_cache:
                result = exhaustive_search(
                    list(autotune_configs()),
                    fwd_stream,
                    lambda cfg: (NUM_SMS, 1, 1),
                    _grouped_gemm_fwd_kernel_ct,
                    lambda cfg: (
                        X_2d,
                        W_flat,
                        Y,
                        m_sizes_i32,
                        gather_indices_i32,
                        N,
                        K,
                        total_tokens,
                        num_experts,
                        NUM_SMS,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        permute_x_flag,
                        permute_y_flag,
                        topk,
                    ),
                    lambda cfg: {"occupancy": cfg.occupancy},
                )
                best_cfg = result.best.config
                _grouped_gemm_fwd_tune_cache[fwd_cache_key] = ct.kernel(
                    _grouped_gemm_fwd_kernel_ct._pyfunc,
                    occupancy=best_cfg.occupancy,
                )
            tuned_fwd_kernel = _grouped_gemm_fwd_tune_cache[fwd_cache_key]
            ct.launch(
                fwd_stream,
                (NUM_SMS, 1, 1),
                tuned_fwd_kernel,
                (
                    X_2d,
                    W_flat,
                    Y,
                    m_sizes_i32,
                    gather_indices_i32,
                    N,
                    K,
                    total_tokens,
                    num_experts,
                    NUM_SMS,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    permute_x_flag,
                    permute_y_flag,
                    topk,
                ),
            )

        ctx.save_for_backward(X, W, m_sizes, gather_indices)
        ctx.topk = topk
        ctx.num_experts = num_experts
        ctx.N = N
        ctx.K = K
        ctx.permute_x = permute_x
        ctx.permute_y = permute_y
        ctx.dX_only = dX_only
        ctx.dW_only = dW_only
        return Y

    @staticmethod
    def backward(ctx, dY):
        dY = dY.contiguous()
        X, W, m_sizes, gather_indices = ctx.saved_tensors
        topk = ctx.topk
        num_experts = ctx.num_experts
        N = ctx.N
        K = ctx.K
        permute_x = ctx.permute_x
        permute_y = ctx.permute_y
        dX_only = ctx.dX_only
        dW_only = ctx.dW_only

        X_2d = X.view(-1, X.shape[-1])
        W_3d = W.view(num_experts, N, -1)
        W_flat = W_3d.reshape(-1, K)  # (E*N, K)

        # Determine total_tokens
        if permute_x or permute_y:
            total_tokens = gather_indices.shape[0]
        else:
            total_tokens = X_2d.shape[0]

        # Ensure m_sizes is int32 on GPU
        m_sizes_i32 = m_sizes.to(torch.int32) if m_sizes.dtype != torch.int32 else m_sizes

        # Prepare gather_indices for kernel
        if gather_indices is None:
            gather_indices_i32 = torch.arange(total_tokens, dtype=torch.int32, device=dY.device)
        else:
            gather_indices_i32 = (
                gather_indices.to(torch.int32) if gather_indices.dtype != torch.int32 else gather_indices
            )

        permute_x_flag = 1 if permute_x else 0
        permute_y_flag = 1 if permute_y else 0

        # Block sizes: heuristic selection (single config to avoid recompilation).
        avg_tokens_per_expert = max(1, total_tokens // num_experts)
        BLOCK_M, BLOCK_N, BLOCK_K = _gemm_block_sizes(N, K, avg_tokens_per_expert)

        NUM_SMS = _get_num_sms(dY.device)
        stream = torch.cuda.current_stream()

        # ----- dX = dY @ W (skip if dW_only) -----
        if not dW_only:
            dX = torch.zeros((total_tokens, K), device=dY.device, dtype=dY.dtype)

            if total_tokens > 0:
                dX_cache_key = (
                    total_tokens,
                    N,
                    K,
                    num_experts,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    permute_x_flag,
                    permute_y_flag,
                    dY.dtype,
                    str(dY.device),
                )
                if dX_cache_key not in _grouped_gemm_dX_tune_cache:
                    result = exhaustive_search(
                        list(autotune_configs()),
                        stream,
                        lambda cfg: (NUM_SMS, 1, 1),
                        _grouped_gemm_dX_kernel_ct,
                        lambda cfg: (
                            dY.view(-1, N),
                            W_flat,
                            dX,
                            m_sizes_i32,
                            gather_indices_i32,
                            N,
                            K,
                            total_tokens,
                            num_experts,
                            NUM_SMS,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_K,
                            permute_x_flag,
                            permute_y_flag,
                        ),
                        lambda cfg: {"occupancy": cfg.occupancy},
                    )
                    best_cfg = result.best.config
                    _grouped_gemm_dX_tune_cache[dX_cache_key] = ct.kernel(
                        _grouped_gemm_dX_kernel_ct._pyfunc,
                        occupancy=best_cfg.occupancy,
                    )
                tuned_dX_kernel = _grouped_gemm_dX_tune_cache[dX_cache_key]
                ct.launch(
                    stream,
                    (NUM_SMS, 1, 1),
                    tuned_dX_kernel,
                    (
                        dY.view(-1, N),
                        W_flat,
                        dX,
                        m_sizes_i32,
                        gather_indices_i32,
                        N,
                        K,
                        total_tokens,
                        num_experts,
                        NUM_SMS,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        permute_x_flag,
                        permute_y_flag,
                    ),
                )

            # topk > 1 with permute_x: multiple expert slots map to same token,
            # sum gradients along topk dimension
            if topk > 1 and permute_x:
                num_tokens = X_2d.shape[0]
                dX = dX.view(num_tokens, topk, -1).sum(dim=1)
        else:
            dX = None

        # ----- dW = dY^T @ X (skip if dX_only) -----
        if not dX_only:
            n_tiles = math.ceil(N / BLOCK_N)
            k_tiles_dw = math.ceil(K / BLOCK_K)
            total_dw_tiles = n_tiles * k_tiles_dw
            dW = torch.zeros((num_experts * N, K), device=dY.device, dtype=dY.dtype)

            if total_dw_tiles > 0:
                dW_cache_key = (
                    total_tokens,
                    N,
                    K,
                    num_experts,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    permute_x_flag,
                    permute_y_flag,
                    topk,
                    dY.dtype,
                    str(dY.device),
                )
                if dW_cache_key not in _grouped_gemm_dW_tune_cache:
                    result = exhaustive_search(
                        list(autotune_configs()),
                        stream,
                        lambda cfg: (total_dw_tiles, 1, 1),
                        _grouped_gemm_dW_kernel_ct,
                        lambda cfg: (
                            X_2d,
                            dY.view(-1, N),
                            dW,
                            m_sizes_i32,
                            gather_indices_i32,
                            num_experts,
                            N,
                            K,
                            total_tokens,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_K,
                            permute_x_flag,
                            permute_y_flag,
                            topk,
                        ),
                        lambda cfg: {"occupancy": cfg.occupancy},
                    )
                    best_cfg = result.best.config
                    _grouped_gemm_dW_tune_cache[dW_cache_key] = ct.kernel(
                        _grouped_gemm_dW_kernel_ct._pyfunc,
                        occupancy=best_cfg.occupancy,
                    )
                tuned_dW_kernel = _grouped_gemm_dW_tune_cache[dW_cache_key]
                ct.launch(
                    stream,
                    (total_dw_tiles, 1, 1),
                    tuned_dW_kernel,
                    (
                        X_2d,
                        dY.view(-1, N),
                        dW,
                        m_sizes_i32,
                        gather_indices_i32,
                        num_experts,
                        N,
                        K,
                        total_tokens,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_K,
                        permute_x_flag,
                        permute_y_flag,
                        topk,
                    ),
                )

            dW = dW.view(num_experts, N, K)
        else:
            dW = None

        return (
            dX,  # X
            dW,  # W
            None,  # m_sizes
            None,  # topk
            None,  # gather_indices
            None,  # permute_x
            None,  # permute_y
            None,  # dX_only
            None,  # dW_only
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
    is_first_gemm: bool = True,
    dX_only: bool = False,
    dW_only: bool = False,
) -> torch.Tensor:
    """
    CuTile grouped GEMM for MoE.

    Supports: non-permute forward/backward, permute_x forward/backward,
              permute_y forward/backward.
    Does NOT support: fuse_mul_post.

    """
    assert not (permute_x and permute_y), "Cannot permute both X and Y"
    assert not fuse_mul_post, "CuTile grouped_gemm does not support fuse_mul_post yet"

    if permute_x or permute_y:
        assert gather_indices is not None, "gather_indices required when permute_x or permute_y is True"

    X = X.view(-1, X.shape[-1])
    m_sizes = m_sizes.view(-1)
    if gather_indices is not None:
        gather_indices = gather_indices.view(-1)

    return GroupedGemmCT.apply(X, W, m_sizes, topk, gather_indices, permute_x, permute_y, dX_only, dW_only)
