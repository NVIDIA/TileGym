# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Tests for cuTile.jl softmax kernels

using Test
using CUDA

const KERNEL_DIR = joinpath(@__DIR__, "..", "kernels")
include(joinpath(KERNEL_DIR, "softmax.jl"))

"""
CPU reference softmax. Input is (M, N) col-major; softmax over dim 2 (columns).
"""
function reference_softmax(x::Matrix{Float32})
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
            x_cpu = randn(Float32, M, N)
            x_gpu = CuArray(x_cpu)
            out_gpu = similar(x_gpu)

            softmax_tma!(out_gpu, x_gpu; tile_size=TILE_SIZE)

            expected = reference_softmax(x_cpu)
            @test Array(out_gpu) ≈ expected atol=1e-5 rtol=1e-4
        end
    end

    @testset "Online softmax (large N)" begin
        test_cases = [
            (M=4,  N=2048,  TILE_SIZE=1024),
            (M=8,  N=4096,  TILE_SIZE=1024),
            (M=2,  N=8192,  TILE_SIZE=1024),
        ]
        for (M, N, TILE_SIZE) in test_cases
            x_cpu = randn(Float32, M, N)
            x_gpu = CuArray(x_cpu)
            out_gpu = similar(x_gpu)

            softmax_online!(out_gpu, x_gpu; tile_size=TILE_SIZE)

            expected = reference_softmax(x_cpu)
            @test Array(out_gpu) ≈ expected atol=1e-4 rtol=1e-3
        end
    end

    @testset "Chunked softmax" begin
        test_cases = [
            (M=4,  N=2048,  TILE_SIZE=1024),
            (M=8,  N=4096,  TILE_SIZE=1024),
            (M=2,  N=1000,  TILE_SIZE=512),
        ]
        for (M, N, TILE_SIZE) in test_cases
            x_cpu = randn(Float32, M, N)
            x_gpu = CuArray(x_cpu)
            out_gpu = similar(x_gpu)

            softmax_chunked!(out_gpu, x_gpu; tile_size=TILE_SIZE)

            expected = reference_softmax(x_cpu)
            @test Array(out_gpu) ≈ expected atol=1e-4 rtol=1e-3
        end
    end

    @testset "Numerical stability (large values)" begin
        M, N = 4, 128
        TILE_SIZE = 128
        x_cpu = randn(Float32, M, N) .* 100f0
        x_gpu = CuArray(x_cpu)
        out_gpu = similar(x_gpu)

        softmax_tma!(out_gpu, x_gpu; tile_size=TILE_SIZE)

        result = Array(out_gpu)
        @test all(isfinite, result)
        @test all(x -> x >= 0, result)
        for i in 1:M
            @test sum(result[i, :]) ≈ 1.0f0 atol=1e-4
        end

        expected = reference_softmax(x_cpu)
        @test result ≈ expected atol=1e-5 rtol=1e-4
    end

    @testset "Single row" begin
        M, N = 1, 512
        TILE_SIZE = 512
        x_cpu = randn(Float32, M, N)
        x_gpu = CuArray(x_cpu)
        out_gpu = similar(x_gpu)

        softmax_tma!(out_gpu, x_gpu; tile_size=TILE_SIZE)

        expected = reference_softmax(x_cpu)
        @test Array(out_gpu) ≈ expected atol=1e-5
    end

end
