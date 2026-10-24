# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

# cuTile.jl softmax kernels
#
# Layout: input and output are (N, M) column-major matrices. Softmax runs over
# dim 1 (length N, contiguous in memory) independently for each of the M
# columns. This is the same memory as a row-major (M, N) tensor in Python with
# softmax over its last axis — reversing the dims keeps the reduction axis
# contiguous.
#
# Three strategies:
# 1. TMA single-tile: loads an entire column in one ct.load (TILE_SIZE >= N)
# 2. Online softmax: 2-pass loop over column chunks with running max/sum (large N)
# 3. Chunked softmax: 3-pass with explicit chunking via gather/scatter (large N)

using CUDA
using cuTile: cuTile
import cuTile as ct

#=============================================================================
 Strategy 1: TMA Single-Tile (for small N where TILE_SIZE >= N)
 Loads entire column in one ct.load call with NegInf padding.
 Uses persistent scheduling: each block processes multiple columns.
=============================================================================#
function softmax_kernel_tma(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                            TILE_SIZE::Int) where {T}
    pid = ct.bid(1)
    num_programs = ct.num_blocks(1)
    n_cols = size(input, 2)

    col_idx = pid
    while col_idx <= n_cols
        col = ct.load(input; index=(Int32(1), col_idx), shape=(TILE_SIZE, 1),
                      padding_mode=ct.PaddingMode.NegInf)
        col = convert(ct.Tile{Float32}, col)

        col_max = maximum(col; dims=1)
        numerator = exp.(col .- col_max)
        denominator = sum(numerator; dims=1)
        softmax_output = numerator ./ denominator

        ct.store(output; index=(Int32(1), col_idx),
                 tile=convert(ct.Tile{T}, softmax_output))
        col_idx += num_programs
    end
    return
end

#=============================================================================
 Strategy 2: Online Softmax (for large N, 2-pass loop over column chunks)
 Pass 1: streaming max + sum via numerically stable online algorithm
 Pass 2: normalize each tile chunk using final max and sum
=============================================================================#
function softmax_kernel_online(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                               TILE_SIZE::Int) where {T}
    ct.@compiler_options occupancy=8

    col_idx = ct.bid(1)
    num_row_tiles = ct.num_tiles(input, 1, (TILE_SIZE, 1))

    m_prev = fill(-Inf32, (1, 1))
    l_prev = zeros(Float32, 1, 1)

    # Pass 1: compute running max and sum
    for row_idx in Int32(1):num_row_tiles
        tile = ct.load(input; index=(row_idx, col_idx), shape=(TILE_SIZE, 1),
                       padding_mode=ct.PaddingMode.NegInf)
        tile = convert(ct.Tile{Float32}, tile)

        tile_max = maximum(tile; dims=1)
        m_curr = max.(tile_max, m_prev)

        # Correct old sum: l_prev *= exp(m_prev - m_curr)
        l_prev = l_prev .* exp.(m_prev .- m_curr)

        # Update with current tile
        p = exp.(tile .- m_curr)
        l_prev = sum(p; dims=1) .+ l_prev
        m_prev = m_curr
    end

    # Pass 2: compute actual softmax values
    for row_idx in Int32(1):num_row_tiles
        tile = ct.load(input; index=(row_idx, col_idx), shape=(TILE_SIZE, 1),
                       padding_mode=ct.PaddingMode.NegInf)
        tile = convert(ct.Tile{Float32}, tile)

        numerator = exp.(tile .- m_prev)
        softmax_output = numerator ./ l_prev

        ct.store(output; index=(row_idx, col_idx),
                 tile=convert(ct.Tile{T}, softmax_output))
    end
    return
end

#=============================================================================
 Strategy 3: Chunked Softmax (3-pass with gather/scatter)
 Uses index-based access with bounds checking (matches Python TileGym).
 Pass 1: find column maximum across all chunks
 Pass 2: compute sum of exp(x - max) across all chunks
 Pass 3: compute final softmax = exp(x - max) / sum and scatter back
=============================================================================#
function softmax_kernel_chunked(output::ct.TileArray{T,2}, input::ct.TileArray{T,2},
                                n_rows::Int, TILE_SIZE::Int) where {T}
    ct.@compiler_options occupancy=4

    col_idx = ct.bid(1)
    # Narrow to Int32 so the chunk loop stays in i32 (no i64↔i32 truncation per offset)
    num_chunks = Int32((n_rows + TILE_SIZE - Int32(1)) ÷ Int32(TILE_SIZE))
    row_offsets_base = ct.arange(TILE_SIZE)
    col_tile = ct.Tile(col_idx)

    col_max = fill(-Inf32, (1,))
    denominator = zeros(Float32, TILE_SIZE)

    # Pass 1: Find maximum across all chunks
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
        chunk = ct.gather(input, (row_indices, col_tile);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        chunk_max = maximum(chunk)
        col_max = max.(col_max, ct.Tile(chunk_max))
    end

    # Pass 2: Compute denominator (sum of all exp values)
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
        chunk = ct.gather(input, (row_indices, col_tile);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        numerator = exp.(chunk .- col_max)
        denominator = denominator .+ numerator
    end
    denom_sum = ct.Tile(sum(denominator))

    # Pass 3: Compute final softmax and scatter
    for chunk_idx in Int32(0):num_chunks - Int32(1)
        row_indices = ct.broadcast_to(ct.Tile(chunk_idx * Int32(TILE_SIZE)), (TILE_SIZE,)) .+ row_offsets_base
        chunk = ct.gather(input, (row_indices, col_tile);
                         check_bounds=true, padding_value=T(-Inf))
        chunk = convert(ct.Tile{Float32}, chunk)
        softmax_output = exp.(chunk .- col_max) ./ denom_sum
        ct.scatter(output, (row_indices, col_tile), convert(ct.Tile{T}, softmax_output);
                  check_bounds=true)
    end
    return
end

#=============================================================================
 Host Functions

 All take (N, M) matrices and compute softmax over dim 1. Launches are
 asynchronous on the task-bound stream; reading the result (e.g. Array(output))
 or CUDA.synchronize() waits for completion.
=============================================================================#

"""
    softmax_tma!(output, input; tile_size)

TMA single-tile strategy. tile_size must be >= size(input, 1).
"""
function softmax_tma!(output::CuMatrix{T}, input::CuMatrix{T};
                      tile_size::Int=nextpow(2, size(input, 1))) where {T}
    M = size(input, 2)
    # Tiles up to 2048 run best at occupancy 8; larger tiles need the registers.
    occupancy = tile_size <= 2048 ? 8 : 4
    @cuda backend=cuTile blocks=M occupancy=occupancy softmax_kernel_tma(output, input, ct.Constant(tile_size))
    return
end

"""
    softmax_online!(output, input; tile_size)

Online softmax strategy. Processes each column in tile_size chunks.
"""
function softmax_online!(output::CuMatrix{T}, input::CuMatrix{T};
                         tile_size::Int=min(2048, nextpow(2, size(input, 1)))) where {T}
    M = size(input, 2)
    @cuda backend=cuTile blocks=M softmax_kernel_online(output, input, ct.Constant(tile_size))
    return
end

"""
    softmax_chunked!(output, input; tile_size)

Chunked softmax strategy (3-pass, gather/scatter).
"""
function softmax_chunked!(output::CuMatrix{T}, input::CuMatrix{T};
                          tile_size::Int=min(1024, nextpow(2, size(input, 1)))) where {T}
    N, M = size(input)
    @cuda backend=cuTile blocks=M softmax_kernel_chunked(output, input, ct.Constant(N), ct.Constant(tile_size))
    return
end

"""
    softmax!(output, input)

Softmax over dim 1, using the strategy that is fastest for N = size(input, 1)
on B200: chunked below 4096, online from 4096 up.
"""
function softmax!(output::CuMatrix{T}, input::CuMatrix{T}) where {T}
    if size(input, 1) < 4096
        softmax_chunked!(output, input)
    else
        softmax_online!(output, input)
    end
    return
end
