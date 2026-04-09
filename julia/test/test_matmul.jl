# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Tests for cuTile.jl matmul kernel
#
# Standard Julia layout (column-major):
#   A shape: (M, K), B shape: (K, N), C shape: (M, N)
#   C = A * B

using Test
using CUDA

const KERNEL_DIR = joinpath(@__DIR__, "..", "kernels")
include(joinpath(KERNEL_DIR, "matmul.jl"))

@testset "Matmul Kernel" begin

    @testset "square matrices" begin
        for n in [64, 128, 256]
            M, K, N = n, n, n
            A = CUDA.rand(Float32, M, K)
            B = CUDA.rand(Float32, K, N)
            C = CUDA.zeros(Float32, M, N)

            matmul!(C, A, B)

            expected = Array(A) * Array(B)
            @test Array(C) ≈ expected atol=1e-1 rtol=1e-2
        end
    end

    @testset "rectangular matrices" begin
        test_cases = [
            (M=64, K=128, N=256),
            (M=256, K=64, N=128),
            (M=128, K=256, N=64),
        ]
        for tc in test_cases
            A = CUDA.rand(Float32, tc.M, tc.K)
            B = CUDA.rand(Float32, tc.K, tc.N)
            C = CUDA.zeros(Float32, tc.M, tc.N)

            matmul!(C, A, B)

            expected = Array(A) * Array(B)
            @test Array(C) ≈ expected atol=1e-1 rtol=1e-2
        end
    end

    @testset "non-tile-aligned dimensions" begin
        M, K, N = 100, 200, 150
        A = CUDA.rand(Float32, M, K)
        B = CUDA.rand(Float32, K, N)
        C = CUDA.zeros(Float32, M, N)

        matmul!(C, A, B)

        expected = Array(A) * Array(B)
        @test Array(C) ≈ expected atol=1e-1 rtol=1e-2
    end

    @testset "identity multiplication" begin
        M, K, N = 128, 128, 128

        # Identity matrix as (M, K)
        A = CuArray(Float32[i == j ? 1.0f0 : 0.0f0 for i in 1:M, j in 1:K])
        B = CUDA.rand(Float32, K, N)
        C = CUDA.zeros(Float32, M, N)

        matmul!(C, A, B)

        expected = Array(A) * Array(B)
        # TF32 tensor cores have ~1e-3 relative precision
        @test Array(C) ≈ expected atol=1e-1 rtol=1e-2
    end

end
