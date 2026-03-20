# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Julia cuTile.jl kernel test runner
#
# Usage:
#   julia --project=julia/ julia/test/runtests.jl
#
# Runs all test_*.jl files in this directory.
# Requires: Julia 1.12+, CUDA.jl, cuTile.jl

using Test

const TEST_DIR = @__DIR__

@testset "TileGym Julia Kernels" begin
    include(joinpath(TEST_DIR, "test_add.jl"))
    include(joinpath(TEST_DIR, "test_matmul.jl"))
    include(joinpath(TEST_DIR, "test_softmax.jl"))
end
