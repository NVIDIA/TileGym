# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# cuTile.jl softmax kernels
#
# Three strategies:
# 1. TMA single-tile: loads entire row in one ct.load (small N)
# 2. Online softmax: 2-pass column-loop with running max/sum (large N)
# 3. Chunked softmax: 3-pass with explicit chunking (large N)

using CUDA
import cuTile as ct

#=============================================================================
 Strategy 1: TMA Single-Tile (for small N where TILE_SIZE >= N)
 Loads entire row in one ct.load call with NegInf padding.
 Uses persistent scheduling: each block processes multiple rows.
=============================================================================#
function softmax_kernel_tma(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                            TILE_SIZE::Int) where {T}
    ct.@compiler_options occupancy=2

    pid = ct.bid(1)
    num_programs = ct.num_blocks(1)
    n_rows = size(input, 1)

    row_idx = pid
    while row_idx <= n_rows
        row = ct.load(input; index=(row_idx, Int32(1)), shape=(1, TILE_SIZE),
                      padding_mode=ct.PaddingMode.NegInf)
        row = convert(ct.Tile{Float32}, row)

        row_max = maximum(row; dims=2)
        numerator = exp.(row .- row_max)
        denominator = sum(numerator; dims=2)
        softmax_output = numerator ./ denominator

        ct.store(output; index=(row_idx, Int32(1)),
                 tile=convert(ct.Tile{T}, softmax_output))
        row_idx += num_programs
    end
    return
end

#=============================================================================
 Strategy 2: Online Softmax (for large N, 2-pass with column-loop)
 Pass 1: streaming max + sum via numerically stable online algorithm
 Pass 2: normalize each tile chunk using final max and sum
=============================================================================#
function softmax_kernel_online(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                               TILE_SIZE::Int) where {T}
    row_idx = ct.bid(1)
    num_col_tiles = ct.num_tiles(input, 2, (1, TILE_SIZE))

    m_prev = fill(-Inf32, (1, 1))
    l_prev = zeros(Float32, 1, 1)

    # Pass 1: compute running max and sum
    for col_idx in Int32(1):num_col_tiles
        row_tile = ct.load(input; index=(row_idx, col_idx), shape=(1, TILE_SIZE),
                          padding_mode=ct.PaddingMode.NegInf)
        row_tile = convert(ct.Tile{Float32}, row_tile)

        tile_max = maximum(row_tile; dims=2)
        m_curr = max.(tile_max, m_prev)

        # Correct old sum: l_prev *= exp(m_prev - m_curr)
        l_prev = l_prev .* exp.(m_prev .- m_curr)

        # Update with current tile
        p = exp.(row_tile .- m_curr)
        l_prev = sum(p; dims=2) .+ l_prev
        m_prev = m_curr
    end

    # Pass 2: compute actual softmax values
    for col_idx in Int32(1):num_col_tiles
        row_tile = ct.load(input; index=(row_idx, col_idx), shape=(1, TILE_SIZE),
                          padding_mode=ct.PaddingMode.NegInf)
        row_tile = convert(ct.Tile{Float32}, row_tile)

        numerator = exp.(row_tile .- m_prev)
        softmax_output = numerator ./ l_prev

        ct.store(output; index=(row_idx, col_idx),
                 tile=convert(ct.Tile{T}, softmax_output))
    end
    return
end

#=============================================================================
 Strategy 3: Chunked Softmax (3-pass with gather/scatter)
 Uses index-based access with bounds checking (matches Python TileGym).
 Pass 1: find row maximum across all chunks
 Pass 2: compute sum of exp(x - max) across all chunks
 Pass 3: compute final softmax = exp(x - max) / sum and scatter back
=============================================================================#
function softmax_kernel_chunked(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                                n_cols::Int, TILE_SIZE::Int) where {T}
    ct.@compiler_options occupancy=4

    row_idx = ct.bid(1)
    num_chunks = (n_cols + TILE_SIZE - Int32(1)) ÷ Int32(TILE_SIZE)
    col_offsets_base = ct.arange(TILE_SIZE)
    row_tile = ct.Tile(row_idx)

    row_max = fill(-Inf32, (1,))
    denominator = zeros(Float32, TILE_SIZE)

    # Pass 1: Find maximum across all chunks
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        col_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ col_offsets_base
        chunk = ct.gather(input, (row_tile, col_indices);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        chunk_max = maximum(chunk)
        row_max = max.(row_max, ct.Tile(chunk_max))
    end

    # Pass 2: Compute denominator (sum of all exp values)
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        col_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ col_offsets_base
        chunk = ct.gather(input, (row_tile, col_indices);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        numerator = exp.(chunk .- row_max)
        denominator = denominator .+ numerator
    end
    denom_sum = ct.Tile(sum(denominator))

    # Pass 3: Compute final softmax and scatter
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        col_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ col_offsets_base
        chunk = ct.gather(input, (row_tile, col_indices);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        softmax_output = exp.(chunk .- row_max) ./ denom_sum
        ct.scatter(output, (row_tile, col_indices), convert(ct.Tile{T}, softmax_output);
                  check_bounds=true)
    end
    return
end

#=============================================================================
 Host Functions
=============================================================================#

"""
    softmax_tma!(output, input; tile_size)

TMA single-tile strategy. tile_size must be >= size(input, 2).
"""
function softmax_tma!(output::CuMatrix{T}, input::CuMatrix{T};
                      tile_size::Int=1024) where {T}
    M = size(input, 1)
    ct.launch(softmax_kernel_tma, M, output, input, ct.Constant(tile_size))
    CUDA.synchronize()
    return
end

"""
    softmax_online!(output, input; tile_size)

Online softmax strategy. Processes row in tile_size chunks.
"""
function softmax_online!(output::CuMatrix{T}, input::CuMatrix{T};
                         tile_size::Int=1024) where {T}
    M = size(input, 1)
    ct.launch(softmax_kernel_online, M, output, input, ct.Constant(tile_size))
    CUDA.synchronize()
    return
end

"""
    softmax_chunked!(output, input; tile_size)

Chunked softmax strategy (3-pass, gather/scatter).
"""
function softmax_chunked!(output::CuMatrix{T}, input::CuMatrix{T};
                          tile_size::Int=1024) where {T}
    M, N = size(input)
    ct.launch(softmax_kernel_chunked, M, output, input,
              ct.Constant(N), ct.Constant(tile_size))
    CUDA.synchronize()
    return
end

const softmax! = softmax_tma!
