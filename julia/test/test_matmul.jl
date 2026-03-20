# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Tests for cuTile.jl matmul kernel
#
# Memory layout (column-major):
#   A shape: (K, M), B shape: (N, K), C shape: (N, M)
#
#   matmul!(a_ptr, b_ptr, c_ptr, K_dim, M_dim, N_dim, tm, tn, tk)

using Test
using CUDA

const KERNEL_DIR = joinpath(@__DIR__, "..", "kernels")
include(joinpath(KERNEL_DIR, "matmul.jl"))

@testset "Matmul Kernel" begin

    @testset "square matrices" begin
        for n in [64, 128, 256]
            # Col-major layout: A(K,M), B(N,K), C(N,M)
            M, K, N = n, n, n

            # Create the matrices as Julia would see them (col-major interpretation of row-major data)
            # A_jl has shape (K, M), B_jl has shape (N, K), C_jl has shape (N, M)
            A_jl = CUDA.rand(Float32, K, M)
            B_jl = CUDA.rand(Float32, N, K)
            C_jl = CUDA.zeros(Float32, N, M)

            tm, tn, tk = 128, 128, 64

            matmul!(
                Int(pointer(A_jl)),
                Int(pointer(B_jl)),
                Int(pointer(C_jl)),
                K, M, N,
                tm, tn, tk
            )

            # Expected: C_jl[n,m] = sum_k B_jl[n,k] * A_jl[k,m]
            # This is just B_jl * A_jl in matrix notation
            expected = Array(B_jl) * Array(A_jl)
            result = Array(C_jl)
            @test result ≈ expected atol=1e-1 rtol=1e-2
        end
    end

    @testset "rectangular matrices" begin
        test_cases = [
            (M=64, K=128, N=256),
            (M=256, K=64, N=128),
            (M=128, K=256, N=64),
        ]
        for tc in test_cases
            A_jl = CUDA.rand(Float32, tc.K, tc.M)
            B_jl = CUDA.rand(Float32, tc.N, tc.K)
            C_jl = CUDA.zeros(Float32, tc.N, tc.M)

            tm, tn, tk = 128, 128, 64

            matmul!(
                Int(pointer(A_jl)),
                Int(pointer(B_jl)),
                Int(pointer(C_jl)),
                tc.K, tc.M, tc.N,
                tm, tn, tk
            )

            expected = Array(B_jl) * Array(A_jl)
            result = Array(C_jl)
            @test result ≈ expected atol=1e-1 rtol=1e-2
        end
    end

    @testset "non-tile-aligned dimensions" begin
        # Dimensions that don't divide evenly by tile size
        M, K, N = 100, 200, 150

        A_jl = CUDA.rand(Float32, K, M)
        B_jl = CUDA.rand(Float32, N, K)
        C_jl = CUDA.zeros(Float32, N, M)

        tm, tn, tk = 128, 128, 64

        matmul!(
            Int(pointer(A_jl)),
            Int(pointer(B_jl)),
            Int(pointer(C_jl)),
            K, M, N,
            tm, tn, tk
        )

        expected = Array(B_jl) * Array(A_jl)
        result = Array(C_jl)
        @test result ≈ expected atol=1e-1 rtol=1e-2
    end

    @testset "identity multiplication" begin
        # A_jl = I (identity), so C_jl = B_jl * I = B_jl
        M = 128
        K = 128
        N = 128

        # Identity as col-major (K, M)
        A_jl = CuArray(Float32[i == j ? 1.0f0 : 0.0f0 for i in 1:K, j in 1:M])
        B_jl = CUDA.rand(Float32, N, K)
        C_jl = CUDA.zeros(Float32, N, M)

        tm, tn, tk = 128, 128, 64

        matmul!(
            Int(pointer(A_jl)),
            Int(pointer(B_jl)),
            Int(pointer(C_jl)),
            K, M, N,
            tm, tn, tk
        )

        expected = Array(B_jl) * Array(A_jl)
        result = Array(C_jl)
        # TF32 tensor cores have ~1e-3 relative precision
        @test result ≈ expected atol=1e-1 rtol=1e-2
    end

end
