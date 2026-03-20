# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Tests for cuTile.jl add kernel
#
# Directly tests the host functions (add!, add_scalar!).

using Test
using CUDA

# Load kernel from julia/kernels/
const KERNEL_DIR = joinpath(@__DIR__, "..", "kernels")
include(joinpath(KERNEL_DIR, "add.jl"))

@testset "Add Kernel" begin

    @testset "tensor + tensor (alpha=1)" begin
        for n in [128, 1024, 4096, 513]
            x = CUDA.rand(Float32, n)
            y = CUDA.rand(Float32, n)
            out = CUDA.zeros(Float32, cld(n, 1024) * 1024)

            # Pad inputs to block-aligned size
            padded_n = cld(n, 1024) * 1024
            x_padded = CUDA.zeros(Float32, padded_n)
            y_padded = CUDA.zeros(Float32, padded_n)
            copyto!(view(x_padded, 1:n), x)
            copyto!(view(y_padded, 1:n), y)

            add!(
                Int(pointer(x_padded)),
                Int(pointer(y_padded)),
                Int(pointer(out)),
                padded_n,
                1.0
            )

            expected = Array(x) .+ Array(y)
            result = Array(out[1:n])
            @test result ≈ expected atol=1e-5
        end
    end

    @testset "tensor + tensor (alpha=0.5)" begin
        n = 1024
        x = CUDA.rand(Float32, n)
        y = CUDA.rand(Float32, n)
        out = CUDA.zeros(Float32, n)

        add!(
            Int(pointer(x)),
            Int(pointer(y)),
            Int(pointer(out)),
            n,
            0.5
        )

        expected = Array(x) .+ Array(y) .* 0.5f0
        result = Array(out)
        @test result ≈ expected atol=1e-5
    end

    @testset "tensor + scalar" begin
        for n in [128, 1024, 4096]
            x = CUDA.rand(Float32, n)
            scalar_val = 3.14
            alpha = 1.0

            padded_n = cld(n, 1024) * 1024
            x_padded = CUDA.zeros(Float32, padded_n)
            out = CUDA.zeros(Float32, padded_n)
            copyto!(view(x_padded, 1:n), x)

            add_scalar!(
                Int(pointer(x_padded)),
                Int(pointer(out)),
                padded_n,
                scalar_val,
                alpha
            )

            expected = Array(x) .+ Float32(scalar_val * alpha)
            result = Array(out[1:n])
            @test result ≈ expected atol=1e-5
        end
    end

    @testset "tensor + scalar with alpha" begin
        n = 1024
        x = CUDA.rand(Float32, n)
        scalar_val = 2.0
        alpha = 0.5

        out = CUDA.zeros(Float32, n)

        add_scalar!(
            Int(pointer(x)),
            Int(pointer(out)),
            n,
            scalar_val,
            alpha
        )

        expected = Array(x) .+ Float32(scalar_val * alpha)
        result = Array(out)
        @test result ≈ expected atol=1e-5
    end

end
