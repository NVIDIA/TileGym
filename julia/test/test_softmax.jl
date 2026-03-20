# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Tests for cuTile.jl softmax kernels (softmax_tma!, softmax_online!, softmax_chunked!).
#
# Tests create column-major data directly.

using Test
using CUDA

const KERNEL_DIR = joinpath(@__DIR__, "..", "kernels")
include(joinpath(KERNEL_DIR, "softmax.jl"))

"""
    reference_softmax(x::Matrix{Float32}) -> Matrix{Float32}

CPU reference softmax. Input is (M, N) col-major; softmax is computed over dim 2 (columns).
"""
function reference_softmax(x::Matrix{Float32})
    # x is (M, N) — softmax over N (dim 2)
    M, N = size(x)
    out = similar(x)
    for i in 1:M
        row = x[i, :]
        row_max = maximum(row)
        exps = exp.(row .- row_max)
        out[i, :] = exps ./ sum(exps)
    end
    return out
end

"""
Helper to create col-major test data on GPU and return both CPU & GPU versions.
The bridge functions expect col-major (M, N) CuArrays.
"""
function make_test_data(M::Int, N::Int)
    x_cpu = randn(Float32, M, N)
    x_gpu = CuArray(x_cpu)
    return x_cpu, x_gpu
end

function next_power_of_2(n::Int)
    n <= 0 && return 1
    p = 1
    while p < n
        p <<= 1
    end
    return p
end

@testset "Softmax Kernel" begin

    @testset "TMA single-tile (small N)" begin
        test_cases = [
            (M=1,   N=16),
            (M=4,   N=32),
            (M=16,  N=64),
            (M=32,  N=128),
            (M=8,   N=256),
            (M=64,  N=1024),
        ]
        for (M, N) in test_cases
            TILE_SIZE = next_power_of_2(N)
            x_cpu, x_gpu = make_test_data(M, N)

            # Pad to TILE_SIZE columns if needed (NegInf padding is handled by kernel)
            out_gpu = CUDA.zeros(Float32, M, N)

            softmax_tma!(
                Int(pointer(x_gpu)),
                Int(pointer(out_gpu)),
                M, N, TILE_SIZE
            )

            expected = reference_softmax(x_cpu)
            result = Array(out_gpu)
            @test result ≈ expected atol=1e-5 rtol=1e-4
        end
    end

    @testset "Online softmax (large N)" begin
        test_cases = [
            (M=4,  N=2048,  TILE_SIZE=1024),
            (M=8,  N=4096,  TILE_SIZE=1024),
            (M=2,  N=8192,  TILE_SIZE=1024),
        ]
        for (M, N, TILE_SIZE) in test_cases
            tile_num_per_row = cld(N, TILE_SIZE)
            padded_N = tile_num_per_row * TILE_SIZE

            x_cpu = randn(Float32, M, N)
            # Pad with -Inf for unused columns
            x_padded_cpu = fill(-Inf32, M, padded_N)
            x_padded_cpu[:, 1:N] .= x_cpu
            x_padded_gpu = CuArray(x_padded_cpu)
            out_padded_gpu = CUDA.zeros(Float32, M, padded_N)

            softmax_online!(
                Int(pointer(x_padded_gpu)),
                Int(pointer(out_padded_gpu)),
                M, N, TILE_SIZE, tile_num_per_row
            )

            expected = reference_softmax(x_cpu)
            result = Array(out_padded_gpu[:, 1:N])
            @test result ≈ expected atol=1e-4 rtol=1e-3
        end
    end

    @testset "Chunked softmax" begin
        test_cases = [
            (M=4,  N=2048,  TILE_SIZE=1024),
            (M=8,  N=4096,  TILE_SIZE=1024),
            (M=2,  N=1000,  TILE_SIZE=512),
        ]
        for (M, N, TILE_SIZE) in test_cases
            num_chunks = cld(N, TILE_SIZE)
            padded_N = num_chunks * TILE_SIZE

            x_cpu = randn(Float32, M, N)
            x_padded_cpu = fill(-Inf32, M, padded_N)
            x_padded_cpu[:, 1:N] .= x_cpu
            x_padded_gpu = CuArray(x_padded_cpu)
            out_padded_gpu = CUDA.zeros(Float32, M, padded_N)

            softmax_chunked!(
                Int(pointer(x_padded_gpu)),
                Int(pointer(out_padded_gpu)),
                M, N, TILE_SIZE
            )

            expected = reference_softmax(x_cpu)
            result = Array(out_padded_gpu[:, 1:N])
            @test result ≈ expected atol=1e-4 rtol=1e-3
        end
    end

    @testset "Numerical stability (large values)" begin
        M, N = 4, 128
        TILE_SIZE = 128
        # Large values that would overflow naive exp()
        x_cpu = randn(Float32, M, N) .* 100f0
        x_gpu = CuArray(x_cpu)
        out_gpu = CUDA.zeros(Float32, M, N)

        softmax_tma!(
            Int(pointer(x_gpu)),
            Int(pointer(out_gpu)),
            M, N, TILE_SIZE
        )

        result = Array(out_gpu)

        # Softmax output must be valid probabilities
        @test all(isfinite, result)
        @test all(x -> x >= 0, result)
        # Each row should sum to ~1
        for i in 1:M
            @test sum(result[i, :]) ≈ 1.0f0 atol=1e-4
        end

        expected = reference_softmax(x_cpu)
        @test result ≈ expected atol=1e-5 rtol=1e-4
    end

    @testset "Single row" begin
        M, N = 1, 512
        TILE_SIZE = 512

        x_cpu, x_gpu = make_test_data(M, N)
        out_gpu = CUDA.zeros(Float32, M, N)

        softmax_tma!(
            Int(pointer(x_gpu)),
            Int(pointer(out_gpu)),
            M, N, TILE_SIZE
        )

        expected = reference_softmax(x_cpu)
        result = Array(out_gpu)
        @test result ≈ expected atol=1e-5
    end

end
