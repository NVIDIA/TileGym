# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from functools import lru_cache

import cuda.tile as ct
import torch

from tilegym.backend import register_impl
from tilegym.logger import get_logger

logger = get_logger(__name__)

# These help in clearly indicating that certain kernel parameters are compile-time constants,
# which cuTile uses for optimization and code generation.
ConstInt = ct.Constant[int]


# --- The FFT Kernel Implementation ---
# This kernel contains the core, low-level cuTile logic for the FFT computation.
# It implements a multi-dimensional FFT using tensor factorization, which breaks
# down a large 1D FFT into a series of smaller FFTs and permutations.
@ct.kernel
def fft_kernel(
    x_packed_in,
    y_packed_out,
    W0,
    W1,
    W2,
    T0,
    T1,  # W and T matrices are pre-computed as single tensors
    N: ConstInt,
    F0: ConstInt,
    F1: ConstInt,
    F2: ConstInt,
    BS: ConstInt,
    D: ConstInt,
):  # D is the atom_packing_dim for memory packing
    """
    cuTile kernel for a multi-dimensional FFT using tensor factorization.
    It expects packed real/imaginary parts of the input and pre-computed factors
    (W and T matrices).

    Args:
        x_packed_in: Input tensor with real/imaginary parts packed for efficient memory access.
                     Expected shape: (Batch_Size, N * 2 // D, D).
        y_packed_out: Output tensor to store FFT results, also in a packed format.
                      Expected shape: (Batch_Size, N * 2 // D, D).
        W0, W1, W2: Rotation matrices (Discrete Fourier Transform matrices) for each
                    of the three logical dimensions (F0, F1, F2). These are pre-computed.
        T0, T1: Twiddle factors for inter-dimensional permutations and phase adjustments.
                These are also pre-computed.
        N (ConstInt): Total FFT size (e.g., 256, 1024).
        F0, F1, F2 (ConstInt): Factors of N, such that N = F0 * F1 * F2. These define
                               the logical 3D shape for the FFT decomposition.
        BS (ConstInt): Number of batch rows each block processes.
        D (ConstInt): Atom packing dimension. This parameter controls how the real and
                      imaginary data are interleaved and packed into memory for optimal
                      coalesced access on the GPU.
    """
    # Pre-calculate products of factors for convenience in reshaping and indexing.
    F0F1 = F0 * F1
    F1F2 = F1 * F2
    F0F2 = F0 * F2

    bid = ct.bid(0)  # Get the Batch ID for the current block.
    # In this kernel, each block processes BS consecutive rows of the batch.

    # --- Load Input Data ---
    # Load input data for the current batch from `x_packed_in`.
    # `x_packed_in` is initially (BS, N * 2 // D, D) due to the packing scheme.
    # `ct.load` reads the specified tile from global memory.
    # Then, `ct.reshape` transforms it to (BS, N, 2) to logically separate
    # the real and imaginary components for each of the N elements.
    # Every tile in this kernel is loaded once and consumed from registers, so
    # plain vector loads and stores are used throughout instead of TMA.
    X_ri = ct.reshape(ct.load(x_packed_in, index=(bid, 0, 0), shape=(BS, N * 2 // D, D), allow_tma=False), (BS, N, 2))

    # Split the real (X_r) and imaginary (X_i) parts into separate tensors.
    # `ct.extract` pulls out the specific component (real at index 0, imag at index 1).
    # Reshape them into the logical 3D structure (BS, F0, F1, F2) for the FFT computation.
    X_r = ct.reshape(ct.extract(X_ri, index=(0, 0, 0), shape=(BS, N, 1)), (BS, F0, F1, F2))
    X_i = ct.reshape(ct.extract(X_ri, index=(0, 0, 1), shape=(BS, N, 1)), (BS, F0, F1, F2))

    # --- Load Rotation (W) and Twiddle (T) Matrices ---
    # These matrices are pre-computed by the caller and passed to the kernel
    # as global memory tensors. They are loaded into the kernel's local scope
    # (e.g., shared memory or registers) and their interleaved real/imaginary parts
    # are split for use in complex arithmetic.

    # W0 (F0 x F0) - Rotation matrix for the first dimension's DFT.
    # Loaded as (F0, F0*2) real/imag interleaved, then reshaped to (F0, F0, 2).
    W0_ri_loaded = ct.reshape(ct.load(W0, index=(0, 0), shape=(F0, F0 * 2), allow_tma=False), (F0, F0, 2))
    # Extract and reshape real and imaginary parts. The (1, F0, F0) shape
    # allows for broadcasting during matrix multiplication with X_r/X_i.
    W0_r_tile = ct.reshape(ct.extract(W0_ri_loaded, index=(0, 0, 0), shape=(F0, F0, 1)), (1, F0, F0))
    W0_i_tile = ct.reshape(ct.extract(W0_ri_loaded, index=(0, 0, 1), shape=(F0, F0, 1)), (1, F0, F0))

    # W1 (F1 x F1) - Rotation matrix for the second dimension's DFT.
    W1_ri_loaded = ct.reshape(ct.load(W1, index=(0, 0), shape=(F1, F1 * 2), allow_tma=False), (F1, F1, 2))
    W1_r_tile = ct.reshape(ct.extract(W1_ri_loaded, index=(0, 0, 0), shape=(F1, F1, 1)), (1, F1, F1))
    W1_i_tile = ct.reshape(ct.extract(W1_ri_loaded, index=(0, 0, 1), shape=(F1, F1, 1)), (1, F1, F1))

    # W2 (F2 x F2) - Rotation matrix for the third dimension's DFT.
    W2_ri_loaded = ct.reshape(ct.load(W2, index=(0, 0), shape=(F2, F2 * 2), allow_tma=False), (F2, F2, 2))
    W2_r_tile = ct.reshape(ct.extract(W2_ri_loaded, index=(0, 0, 0), shape=(F2, F2, 1)), (1, F2, F2))
    W2_i_tile = ct.reshape(ct.extract(W2_ri_loaded, index=(0, 0, 1), shape=(F2, F2, 1)), (1, F2, F2))

    # T0 (F0 x F1F2) - Twiddle factors applied after the first contraction stage.
    # Loaded as (F0, F1F2*2), then reshaped to (F0, F1F2, 2).
    T0_ri_loaded = ct.reshape(ct.load(T0, index=(0, 0), shape=(F0, F1F2 * 2), allow_tma=False), (F0, F1F2, 2))
    # Reshape to (N, 1) to align with the flattened data for element-wise multiplication.
    T0_r_tile = ct.reshape(ct.extract(T0_ri_loaded, index=(0, 0, 0), shape=(F0, F1F2, 1)), (N, 1))
    T0_i_tile = ct.reshape(ct.extract(T0_ri_loaded, index=(0, 0, 1), shape=(F0, F1F2, 1)), (N, 1))

    # T1 (F1 x F2) - Twiddle factors applied after the second contraction stage.
    T1_ri_loaded = ct.reshape(ct.load(T1, index=(0, 0), shape=(F1, F2 * 2), allow_tma=False), (F1, F2, 2))
    # Reshape to (F1F2, 1) for element-wise multiplication.
    T1_r_tile = ct.reshape(ct.extract(T1_ri_loaded, index=(0, 0, 0), shape=(F1, F2, 1)), (F1F2, 1))
    T1_i_tile = ct.reshape(ct.extract(T1_ri_loaded, index=(0, 0, 1), shape=(F1, F2, 1)), (F1F2, 1))

    # --- CT0: Contract over the first dimension (F0) ---
    # Reshape X_r and X_i to (BS, F0, F1F2) to prepare for matrix multiplication with W0.
    X_r = ct.reshape(X_r, (BS, F0, F1F2))
    X_i = ct.reshape(X_i, (BS, F0, F1F2))
    # Perform complex matrix multiplication: (A+iB)(C+iD) = (AC-BD) + i(AD+BC).
    # The result is then reshaped back to (BS, N, 1) to align with T0 for twiddling.
    X_r_ = ct.reshape(ct.matmul(W0_r_tile, X_r) - ct.matmul(W0_i_tile, X_i), (BS, N, 1))
    X_i_ = ct.reshape(ct.matmul(W0_i_tile, X_r) + ct.matmul(W0_r_tile, X_i), (BS, N, 1))

    # --- Twiddle & Permute 0 ---
    # Apply twiddle factors T0 element-wise to the complex results.
    X_r = T0_r_tile * X_r_ - T0_i_tile * X_i_
    X_i = T0_i_tile * X_r_ + T0_r_tile * X_i_
    # Permute dimensions from (BS, F0, F1, F2) to (BS, F1, F2, F0)
    # to prepare the data for the next contraction stage.
    X_r = ct.permute(ct.reshape(X_r, (BS, F0, F1, F2)), (0, 2, 3, 1))
    X_i = ct.permute(ct.reshape(X_i, (BS, F0, F1, F2)), (0, 2, 3, 1))

    # --- CT1: Contract over the second dimension (F1) ---
    # Reshape X_r and X_i to (BS, F1, F0F2) for matrix multiplication with W1.
    X_r = ct.reshape(X_r, (BS, F1, F0F2))
    X_i = ct.reshape(X_i, (BS, F1, F0F2))
    # Perform complex matrix multiplication.
    # The result is reshaped to (BS, F1F2, F0) to align with T1 for twiddling.
    X_r_ = ct.reshape(ct.matmul(W1_r_tile, X_r) - ct.matmul(W1_i_tile, X_i), (BS, F1F2, F0))
    X_i_ = ct.reshape(ct.matmul(W1_i_tile, X_r) + ct.matmul(W1_r_tile, X_i), (BS, F1F2, F0))

    # --- Twiddle & Permute 1 ---
    # Apply twiddle factors T1 element-wise.
    X_r = T1_r_tile * X_r_ - T1_i_tile * X_i_
    X_i = T1_i_tile * X_r_ + T1_r_tile * X_i_
    # Permute dimensions from (BS, F1, F2, F0) to (BS, F2, F0, F1)
    # to prepare the data for the final contraction stage.
    X_r = ct.permute(ct.reshape(X_r, (BS, F1, F2, F0)), (0, 2, 3, 1))
    X_i = ct.permute(ct.reshape(X_i, (BS, F1, F2, F0)), (0, 2, 3, 1))

    # --- CT2: Contract over the third dimension (F2) ---
    # Reshape X_r and X_i to (BS, F2, F0F1) for matrix multiplication with W2.
    X_r = ct.reshape(X_r, (BS, F2, F0F1))
    X_i = ct.reshape(X_i, (BS, F2, F0F1))
    # Perform complex matrix multiplication.
    X_r_ = ct.matmul(W2_r_tile, X_r) - ct.matmul(W2_i_tile, X_i)
    X_i_ = ct.matmul(W2_i_tile, X_r) + ct.matmul(W2_r_tile, X_i)

    # --- Final Permutation and Reshape ---
    # Permute back to the original logical order (BS, F0, F1, F2) from (BS, F2, F0, F1).
    X_r = ct.permute(ct.reshape(X_r_, (BS, F2, F0, F1)), (0, 1, 3, 2))
    X_i = ct.permute(ct.reshape(X_i_, (BS, F2, F0, F1)), (0, 1, 3, 2))
    # Reshape to (BS, N, 1) for real and imaginary parts separately,
    # flattening the 3D logical structure back to a 1D representation per batch item.
    X_r = ct.reshape(X_r, (BS, N, 1))
    X_i = ct.reshape(X_i, (BS, N, 1))

    # --- Final Reshape and Store Output ---
    # Concatenate the real and imaginary parts along the last axis to form (BS, N, 2).
    # Then reshape to the packed output format (BS, N * 2 // D, D) for storing to global memory.
    Y_ri = ct.reshape(ct.cat((X_r, X_i), axis=-1), (BS, N * 2 // D, D))
    ct.store(y_packed_out, index=(bid, 0, 0), tile=Y_ri, allow_tma=False)


@lru_cache(maxsize=None)
def _get_fft_kernel(occupancy):
    return fft_kernel.replace_hints(occupancy=occupancy) if occupancy is not None else fft_kernel


@register_impl("fft", backend="cutile")
def fft(x, decomp, W0, W1, W2, T0, T1, atom, BS, occupancy, **kwargs):
    x_ri = torch.view_as_real(x)
    F0, F1, F2 = decomp
    y_ri = torch.empty_like(x_ri)
    batch, N, _ = x_ri.shape
    assert x_ri.shape[2] == 2
    D = N * 2  # one tile per interleaved row
    grid = (math.ceil(batch / BS), 1, 1)
    kernel = _get_fft_kernel(occupancy)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        kernel,
        (
            x_ri.view(batch, N * 2 // D, D),
            y_ri.view(batch, N * 2 // D, D),
            W0.view(F0, F0 * 2),
            W1.view(F1, F1 * 2),
            W2.view(F2, F2 * 2),
            T0.view(F0, F1 * F2 * 2),
            T1.view(F1, F2 * 2),
            N,
            F0,
            F1,
            F2,
            BS,
            D,
        ),
    )
    return torch.view_as_complex(y_ri)
