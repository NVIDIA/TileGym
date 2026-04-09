# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# cuTile.jl element-wise addition kernels
#
# Element-wise addition: output = x + y * alpha (tensor+tensor)
#                        output = x + scalar * alpha (tensor+scalar)
#
# Uses ct.load/ct.store TMA pattern for contiguous 1D element-wise ops.

using CUDA
import cuTile as ct

#=============================================================================
 Add Kernel (tensor + tensor): output = x + y * alpha
=============================================================================#

function add_kernel(x::ct.TileArray{T,1}, y::ct.TileArray{T,1},
                    output::ct.TileArray{T,1},
                    alpha::Float32, BLOCK_SIZE::Int) where {T}
    bid = ct.bid(1)

    x_tile = ct.load(x; index=bid, shape=(BLOCK_SIZE,), padding_mode=ct.PaddingMode.Zero)
    y_tile = ct.load(y; index=bid, shape=(BLOCK_SIZE,), padding_mode=ct.PaddingMode.Zero)

    x_f32 = convert(ct.Tile{Float32}, x_tile)
    y_f32 = convert(ct.Tile{Float32}, y_tile)

    # Scalar alpha broadcasts to tile shape automatically
    output_f32 = x_f32 .+ y_f32 .* alpha
    ct.store(output; index=bid, tile=convert(ct.Tile{T}, output_f32))
    return
end

#=============================================================================
 Add Scalar Kernel (tensor + scalar): output = x + scalar_val * alpha
=============================================================================#

function add_scalar_kernel(x::ct.TileArray{T,1}, output::ct.TileArray{T,1},
                           scalar_val::Float32, alpha::Float32,
                           BLOCK_SIZE::Int) where {T}
    bid = ct.bid(1)

    x_tile = ct.load(x; index=bid, shape=(BLOCK_SIZE,), padding_mode=ct.PaddingMode.Zero)
    x_f32 = convert(ct.Tile{Float32}, x_tile)

    output_f32 = x_f32 .+ (scalar_val * alpha)
    ct.store(output; index=bid, tile=convert(ct.Tile{T}, output_f32))
    return
end

#=============================================================================
 Host Functions
=============================================================================#

function add!(output::CuVector{T}, x::CuVector{T}, y::CuVector{T};
              alpha::Float32=1.0f0, block_size::Int=1024) where {T}
    n = length(x)
    grid = cld(n, block_size)
    ct.launch(add_kernel, grid, x, y, output,
              ct.Constant(alpha), ct.Constant(block_size))
    CUDA.synchronize()
    return
end

function add_scalar!(output::CuVector{T}, x::CuVector{T}, scalar_val::Float32;
                     alpha::Float32=1.0f0, block_size::Int=1024) where {T}
    n = length(x)
    grid = cld(n, block_size)
    ct.launch(add_scalar_kernel, grid, x, output,
              ct.Constant(scalar_val), ct.Constant(alpha), ct.Constant(block_size))
    CUDA.synchronize()
    return
end
