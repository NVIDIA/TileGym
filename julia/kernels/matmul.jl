# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# cuTile.jl matrix multiplication kernel
#
# Tile sizes passed as ct.Constant at launch (no @eval needed).
#
# Memory layout (column-major):
#   A(K, M), B(N, K), C(N, M)
#   C[n,m] = sum_k B[n,k] * A[k,m]
#   acc = muladd(b_tile, a_tile, acc) where b_tile is (tn, tk), a_tile is (tk, tm)

using CUDA
import cuTile as ct

#=============================================================================
 Matmul Kernel with ct.Constant tile sizes
 Grid: (num_m_tiles, num_n_tiles)
   bid(1) -> M tile index
   bid(2) -> N tile index
=============================================================================#

function _matmul_kernel(A_jl::ct.TileArray{T, 2},
                        B_jl::ct.TileArray{T, 2},
                        C_jl::ct.TileArray{T, 2},
                        num_k_tiles::Int,
                        TM::Int, TN::Int, TK::Int) where {T}
    # Grid dimensions (1-indexed)
    # A_jl is (K, M), B_jl is (N, K), C_jl is (N, M)
    bid_m = ct.bid(1)      # M tile index
    bid_n = ct.bid(2)      # N tile index

    # Initialize accumulator with Float32 for precision
    acc = ct.full((TN, TM), zero(Float32), Float32)

    # K reduction loop (while, not for)
    k = Int32(1)
    while k <= num_k_tiles
        # Load A_jl tile: shape (K, M), load (tk, tm) at (k, bid_m)
        a_tile = ct.load(A_jl, (k, bid_m), (TK, TM);
                         padding_mode=ct.PaddingMode.Zero)

        # Load B_jl tile: shape (N, K), load (tn, tk) at (bid_n, k)
        b_tile = ct.load(B_jl, (bid_n, k), (TN, TK);
                         padding_mode=ct.PaddingMode.Zero)

        # Convert to TF32 for tensor cores (Float32 inputs only)
        if T === Float32
            a_tile = convert(ct.Tile{ct.TFloat32}, a_tile)
            b_tile = convert(ct.Tile{ct.TFloat32}, b_tile)
        end

        # C_jl[n,m] = sum_k B_jl[n,k] * A_jl[k,m] => muladd(b_tile, a_tile, acc)
        acc = muladd(b_tile, a_tile, acc)
        k += Int32(1)
    end

    # Convert to output type and store
    result = convert(ct.Tile{T}, acc)
    ct.store(C_jl, (bid_n, bid_m), result)

    return nothing
end

#=============================================================================
 Host Function
=============================================================================#

"""
    matmul!(a_ptr, b_ptr, c_ptr, K_dim, M_dim, N_dim, tm, tn, tk)

Launch matmul kernel. Tile sizes are passed as ct.Constant.

Memory layout (column-major):
  A shape: (K, M), B shape: (N, K), C shape: (N, M)
"""
function matmul!(a_ptr::Int, b_ptr::Int, c_ptr::Int,
                 K_dim::Int, M_dim::Int, N_dim::Int,
                 tm::Int, tn::Int, tk::Int)
    # A_jl shape: (K, M), B_jl shape: (N, K), C_jl shape: (N, M)
    A_cu = unsafe_wrap(CuArray{Float32, 2},
                       CUDA.CuPtr{Float32}(UInt(a_ptr)),
                       (K_dim, M_dim); own=false)
    B_cu = unsafe_wrap(CuArray{Float32, 2},
                       CUDA.CuPtr{Float32}(UInt(b_ptr)),
                       (N_dim, K_dim); own=false)
    C_cu = unsafe_wrap(CuArray{Float32, 2},
                       CUDA.CuPtr{Float32}(UInt(c_ptr)),
                       (N_dim, M_dim); own=false)

    num_m_tiles = cld(M_dim, tm)
    num_n_tiles = cld(N_dim, tn)
    num_k_tiles = cld(K_dim, tk)
    grid = (num_m_tiles, num_n_tiles)

    ct.launch(_matmul_kernel, grid, A_cu, B_cu, C_cu,
              ct.Constant(num_k_tiles),
              ct.Constant(tm), ct.Constant(tn), ct.Constant(tk);
              occupancy=2)

    CUDA.synchronize()
    return nothing
end
