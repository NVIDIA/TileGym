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

function _add_kernel(x::ct.TileArray{T,1}, y::ct.TileArray{T,1},
                     output::ct.TileArray{T,1},
                     alpha::Float32, BLOCK_SIZE::Int) where {T}
    bid = ct.bid(1)

    # Load input tiles at block index
    x_tile = ct.load(x, bid, (BLOCK_SIZE,))
    y_tile = ct.load(y, bid, (BLOCK_SIZE,))

    x_f32 = convert(ct.Tile{Float32}, x_tile)
    y_f32 = convert(ct.Tile{Float32}, y_tile)

    # Compute: x + y * alpha
    alpha_tile = ct.full((BLOCK_SIZE,), alpha, Float32)
    y_scaled = y_f32 .* alpha_tile
    output_f32 = x_f32 .+ y_scaled

    ct.store(output, bid, convert(ct.Tile{T}, output_f32))
    return nothing
end

#=============================================================================
 Add Scalar Kernel (tensor + scalar): output = x + scalar_val * alpha
=============================================================================#

function _add_scalar_kernel(x::ct.TileArray{T,1}, output::ct.TileArray{T,1},
                            scalar_val::Float32, alpha::Float32,
                            BLOCK_SIZE::Int) where {T}
    bid = ct.bid(1)

    # Load input tile at block index
    x_tile = ct.load(x, bid, (BLOCK_SIZE,))
    x_f32 = convert(ct.Tile{Float32}, x_tile)

    # Compute: x + scalar * alpha
    scaled_scalar = scalar_val * alpha
    scalar_tile = ct.full((BLOCK_SIZE,), scaled_scalar, Float32)
    output_f32 = x_f32 .+ scalar_tile

    ct.store(output, bid, convert(ct.Tile{T}, output_f32))
    return nothing
end

#=============================================================================
 Host Functions
 Accept raw GPU pointers, wrap as CuArray, launch kernel.
=============================================================================#

function add!(x_ptr::Int, y_ptr::Int, out_ptr::Int,
              n_elements::Int, alpha::Float64)
    BLOCK_SIZE = 1024
    # Pad n_elements to multiple of BLOCK_SIZE
    padded_n = cld(n_elements, BLOCK_SIZE) * BLOCK_SIZE

    x_cu = unsafe_wrap(CuArray{Float32, 1}, CUDA.CuPtr{Float32}(UInt(x_ptr)),
                       (padded_n,); own=false)
    y_cu = unsafe_wrap(CuArray{Float32, 1}, CUDA.CuPtr{Float32}(UInt(y_ptr)),
                       (padded_n,); own=false)
    out_cu = unsafe_wrap(CuArray{Float32, 1}, CUDA.CuPtr{Float32}(UInt(out_ptr)),
                         (padded_n,); own=false)

    grid = cld(padded_n, BLOCK_SIZE)
    ct.launch(_add_kernel, grid, x_cu, y_cu, out_cu,
              ct.Constant(Float32(alpha)), ct.Constant(BLOCK_SIZE))

    CUDA.synchronize()
    return nothing
end

function add_scalar!(x_ptr::Int, out_ptr::Int,
                     n_elements::Int, scalar_val::Float64, alpha::Float64)
    BLOCK_SIZE = 1024
    # Pad n_elements to multiple of BLOCK_SIZE
    padded_n = cld(n_elements, BLOCK_SIZE) * BLOCK_SIZE

    x_cu = unsafe_wrap(CuArray{Float32, 1}, CUDA.CuPtr{Float32}(UInt(x_ptr)),
                       (padded_n,); own=false)
    out_cu = unsafe_wrap(CuArray{Float32, 1}, CUDA.CuPtr{Float32}(UInt(out_ptr)),
                         (padded_n,); own=false)

    grid = cld(padded_n, BLOCK_SIZE)
    ct.launch(_add_scalar_kernel, grid, x_cu, out_cu,
              ct.Constant(Float32(scalar_val)), ct.Constant(Float32(alpha)),
              ct.Constant(BLOCK_SIZE))

    CUDA.synchronize()
    return nothing
end
