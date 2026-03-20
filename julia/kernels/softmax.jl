# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# cuTile.jl softmax kernels
#
# Three strategies:
# 1. TMA single-tile: loads entire row in one ct.load (small N)
# 2. Online softmax: 2-pass column-loop with running max/sum (large N)
# 3. Chunked softmax: 3-pass with gather/scatter (explicit chunking)

using CUDA
import cuTile as ct

#=============================================================================
 Strategy 1: TMA Single-Tile (for small N where TILE_SIZE >= N)
 Loads entire row in one ct.load call with NegInf padding.
=============================================================================#
function _softmax_kernel_tma(output::ct.TileArray{T, 2}, input::ct.TileArray{T, 2},
                             n_rows::Int, n_cols::Int,
                             TILE_SIZE::Int) where {T}
    pid = ct.bid(1)
    num_programs = ct.num_blocks(1)

    row_idx = pid
    while row_idx <= n_rows
        row = ct.load(input, (row_idx, Int32(1)), (1, TILE_SIZE);
                      padding_mode=ct.PaddingMode.NegInf)

        row = convert(ct.Tile{Float32}, row)

        row_max = maximum(row; dims=2)
        row_minus_max = row .- row_max

        numerator = exp.(row_minus_max)
        denominator = sum(numerator; dims=2)
        softmax_output = numerator ./ denominator

        softmax_output = convert(ct.Tile{T}, softmax_output)
        ct.store(output, (row_idx, Int32(1)), softmax_output)

        row_idx += num_programs
    end

    return
end

#=============================================================================
 Strategy 2: Online Softmax (for large N, 2-pass with column-loop)
 Pass 1: streaming max + sum via numerically stable online algorithm
 Pass 2: normalize each tile chunk using final max and sum
=============================================================================#
function _softmax_kernel_online(output::ct.TileArray{T, 2}, input::ct.TileArray{T, 2},
                                n_cols::Int, TILE_SIZE::Int,
                                tile_num_per_row::Int) where {T}
    # One block per row
    row_idx = ct.bid(1)

    # Initialize running max and sum
    m_prev = ct.full((1, 1), -Inf32, Float32)
    l_prev = ct.full((1, 1), 0.0f0, Float32)

    # Pass 1: compute running max and sum
    col_idx = Int32(1)
    while col_idx <= tile_num_per_row
        row_tile = ct.load(input, (row_idx, col_idx), (1, TILE_SIZE))
        row_tile = convert(ct.Tile{Float32}, row_tile)

        tile_max = maximum(row_tile; dims=2)
        m_curr = max.(tile_max, m_prev)

        # Correct old l_prev: l_prev *= exp(m_prev - m_curr)
        exp_diff = exp.(m_prev .- m_curr)
        l_prev = l_prev .* exp_diff

        # Update with current tile: p = exp(row - m_curr)
        p = exp.(row_tile .- m_curr)
        l_curr = sum(p; dims=2)

        l_prev = l_curr .+ l_prev
        m_prev = m_curr

        col_idx += Int32(1)
    end

    # Pass 2: compute actual softmax values
    col_idx = Int32(1)
    while col_idx <= tile_num_per_row
        row_tile = ct.load(input, (row_idx, col_idx), (1, TILE_SIZE))
        row_tile = convert(ct.Tile{Float32}, row_tile)

        row_minus_max = row_tile .- m_prev
        numerator = exp.(row_minus_max)
        softmax_output = numerator ./ l_prev

        softmax_output = convert(ct.Tile{T}, softmax_output)
        ct.store(output, (row_idx, col_idx), softmax_output)

        col_idx += Int32(1)
    end

    return
end

#=============================================================================
 Strategy 3: Chunked Softmax (3-pass with ct.load/ct.store)
 Pass 1: find row maximum across all chunks
 Pass 2: compute sum of exp(x - max) across all chunks
 Pass 3: compute final softmax = exp(x - max) / sum and store back

 Tile size passed as ct.Constant (no @eval needed).
=============================================================================#

function _softmax_kernel_chunked(output::ct.TileArray{T, 2}, input::ct.TileArray{T, 2},
                                 num_chunks::Int, TILE_SIZE::Int) where {T}
    # One block per row
    row_idx = ct.bid(1)

    row_max = ct.full((1, 1), -Inf32, Float32)
    denominator = ct.full((1, 1), 0.0f0, Float32)

    # Pass 1: Find maximum across all chunks
    col_idx = Int32(1)
    while col_idx <= num_chunks
        chunk = ct.load(input, (row_idx, col_idx), (1, TILE_SIZE))
        chunk = convert(ct.Tile{Float32}, chunk)
        chunk_max = maximum(chunk; dims=2)
        row_max = max.(row_max, chunk_max)
        col_idx += Int32(1)
    end

    # Pass 2: Compute denominator (sum of all exp values)
    col_idx = Int32(1)
    while col_idx <= num_chunks
        chunk = ct.load(input, (row_idx, col_idx), (1, TILE_SIZE))
        chunk = convert(ct.Tile{Float32}, chunk)
        row_minus_max = chunk .- row_max
        numerator = exp.(row_minus_max)
        exponentials_sum = sum(numerator; dims=2)
        denominator = denominator .+ exponentials_sum
        col_idx += Int32(1)
    end

    # Pass 3: Compute final softmax and store
    col_idx = Int32(1)
    while col_idx <= num_chunks
        chunk = ct.load(input, (row_idx, col_idx), (1, TILE_SIZE))
        chunk = convert(ct.Tile{Float32}, chunk)
        row_minus_max = chunk .- row_max
        numerator = exp.(row_minus_max)
        softmax_output = numerator ./ denominator
        softmax_output = convert(ct.Tile{T}, softmax_output)
        ct.store(output, (row_idx, col_idx), softmax_output)
        col_idx += Int32(1)
    end

    return
end

#=============================================================================
 Host Functions
=============================================================================#

"""
    softmax_tma!(input_ptr, output_ptr, M, N, TILE_SIZE)

TMA single-tile strategy. TILE_SIZE must be >= N.
"""
function softmax_tma!(input_ptr::Int, output_ptr::Int, M::Int, N::Int, TILE_SIZE::Int)
    input_cu = unsafe_wrap(CuArray{Float32, 2}, CUDA.CuPtr{Float32}(UInt(input_ptr)), (M, N); own=false)
    output_cu = unsafe_wrap(CuArray{Float32, 2}, CUDA.CuPtr{Float32}(UInt(output_ptr)), (M, N); own=false)

    ct.launch(_softmax_kernel_tma, M, output_cu, input_cu,
              ct.Constant(M), ct.Constant(N), ct.Constant(TILE_SIZE);
              occupancy=2)

    CUDA.synchronize()
    return nothing
end

"""
    softmax_online!(input_ptr, output_ptr, M, N, TILE_SIZE, tile_num_per_row)

Online softmax strategy. Processes row in TILE_SIZE chunks.
Input must be padded to tile_num_per_row * TILE_SIZE columns.
"""
function softmax_online!(input_ptr::Int, output_ptr::Int,
                         M::Int, N::Int, TILE_SIZE::Int, tile_num_per_row::Int)
    # N here is the padded column count (tile_num_per_row * TILE_SIZE)
    padded_N = tile_num_per_row * TILE_SIZE
    input_cu = unsafe_wrap(CuArray{Float32, 2}, CUDA.CuPtr{Float32}(UInt(input_ptr)), (M, padded_N); own=false)
    output_cu = unsafe_wrap(CuArray{Float32, 2}, CUDA.CuPtr{Float32}(UInt(output_ptr)), (M, padded_N); own=false)

    # One block per row for online variant
    ct.launch(_softmax_kernel_online, M, output_cu, input_cu,
              ct.Constant(N), ct.Constant(TILE_SIZE), ct.Constant(tile_num_per_row);
              occupancy=2)

    CUDA.synchronize()
    return nothing
end

"""
    softmax_chunked!(input_ptr, output_ptr, M, N, TILE_SIZE)

Chunked softmax strategy. TILE_SIZE is passed as ct.Constant.
"""
function softmax_chunked!(input_ptr::Int, output_ptr::Int, M::Int, N::Int, TILE_SIZE::Int)
    num_chunks = cld(N, TILE_SIZE)
    # Pad N to multiple of TILE_SIZE for ct.load alignment
    padded_N = num_chunks * TILE_SIZE
    input_cu = unsafe_wrap(CuArray{Float32, 2}, CUDA.CuPtr{Float32}(UInt(input_ptr)), (M, padded_N); own=false)
    output_cu = unsafe_wrap(CuArray{Float32, 2}, CUDA.CuPtr{Float32}(UInt(output_ptr)), (M, padded_N); own=false)

    ct.launch(_softmax_kernel_chunked, M, output_cu, input_cu,
              ct.Constant(num_chunks), ct.Constant(TILE_SIZE);
              occupancy=4)

    CUDA.synchronize()
    return nothing
end

const softmax! = softmax_tma!
