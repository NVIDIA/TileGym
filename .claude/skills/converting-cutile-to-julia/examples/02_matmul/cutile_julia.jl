# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Matrix multiplication — cuTile.jl
#
#   C = A @ B
#
# Column-major layout: A_jl(K,M), B_jl(N,K), C_jl(N,M)
#   C_jl[n,m] = sum_k B_jl[n,k] * A_jl[k,m]
#   acc = muladd(b_tile, a_tile, acc)
#
# Uses 2D grid, K-reduction while loop, TF32 for Float32 inputs.
# Matches julia/kernels/matmul.jl

using CUDA
import cuTile as ct

function _matmul_kernel(A_jl::ct.TileArray{T, 2},
                        B_jl::ct.TileArray{T, 2},
                        C_jl::ct.TileArray{T, 2},
                        num_k_tiles::Int,
                        TM::Int, TN::Int, TK::Int) where {T}
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)

    acc = ct.full((TN, TM), zero(Float32), Float32)

    k = Int32(1)
    while k <= num_k_tiles
        a_tile = ct.load(A_jl, (k, bid_m), (TK, TM);
                         padding_mode=ct.PaddingMode.Zero)
        b_tile = ct.load(B_jl, (bid_n, k), (TN, TK);
                         padding_mode=ct.PaddingMode.Zero)

        if T === Float32
            a_tile = convert(ct.Tile{ct.TFloat32}, a_tile)
            b_tile = convert(ct.Tile{ct.TFloat32}, b_tile)
        end

        acc = muladd(b_tile, a_tile, acc)
        k += Int32(1)
    end

    result = convert(ct.Tile{T}, acc)
    ct.store(C_jl, (bid_n, bid_m), result)

    return nothing
end

# ── Host function ────────────────────────────────────────────────────────────

function matmul!(a_ptr::Int, b_ptr::Int, c_ptr::Int,
                 K_dim::Int, M_dim::Int, N_dim::Int,
                 tm::Int, tn::Int, tk::Int)
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

# ── Verify ───────────────────────────────────────────────────────────────────

function verify()
    test_cases = [
        (M=64,  K=64,  N=64),
        (M=128, K=128, N=128),
        (M=256, K=256, N=256),
        (M=100, K=200, N=150),
    ]
    tm, tn, tk = 128, 128, 64
    for tc in test_cases
        A_jl = CUDA.rand(Float32, tc.K, tc.M)
        B_jl = CUDA.rand(Float32, tc.N, tc.K)
        C_jl = CUDA.zeros(Float32, tc.N, tc.M)

        matmul!(Int(pointer(A_jl)), Int(pointer(B_jl)), Int(pointer(C_jl)),
                tc.K, tc.M, tc.N, tm, tn, tk)

        expected = Array(B_jl) * Array(A_jl)
        result = Array(C_jl)
        @assert isapprox(result, expected; atol=1e-1, rtol=1e-2) (
            "matmul failed ($(tc.M)x$(tc.K))@($(tc.K)x$(tc.N))")
        println("  ($(tc.M)x$(tc.K)) @ ($(tc.K)x$(tc.N)): passed")
    end
end

function main()
    println("--- cuTile.jl Matmul Examples ---\n")
    verify()
    println("\n--- All matmul examples passed ---")
end

isinteractive() || main()
