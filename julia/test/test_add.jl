# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Tests for cuTile.jl add kernel

using Test
using CUDA

const KERNEL_DIR = joinpath(@__DIR__, "..", "kernels")
include(joinpath(KERNEL_DIR, "add.jl"))

@testset "Add Kernel" begin

    @testset "tensor + tensor (alpha=1)" begin
        for n in [128, 1024, 4096, 513]
            x = CUDA.rand(Float32, n)
            y = CUDA.rand(Float32, n)
            out = similar(x)

            add!(out, x, y)

            expected = Array(x) .+ Array(y)
            @test Array(out) ≈ expected atol=1e-5
        end
    end

    @testset "tensor + tensor (alpha=0.5)" begin
        n = 1024
        x = CUDA.rand(Float32, n)
        y = CUDA.rand(Float32, n)
        out = similar(x)

        add!(out, x, y; alpha=0.5f0)

        expected = Array(x) .+ Array(y) .* 0.5f0
        @test Array(out) ≈ expected atol=1e-5
    end

    @testset "tensor + scalar" begin
        for n in [128, 1024, 4096]
            x = CUDA.rand(Float32, n)
            out = similar(x)

            add_scalar!(out, x, 3.14f0)

            expected = Array(x) .+ 3.14f0
            @test Array(out) ≈ expected atol=1e-5
        end
    end

    @testset "tensor + scalar with alpha" begin
        n = 1024
        x = CUDA.rand(Float32, n)
        out = similar(x)

        add_scalar!(out, x, 2.0f0; alpha=0.5f0)

        expected = Array(x) .+ 1.0f0
        @test Array(out) ≈ expected atol=1e-5
    end

end
