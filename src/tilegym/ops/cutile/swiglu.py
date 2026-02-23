# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT


import cuda.tile as ct
import torch
import torch.nn as nn
from cuda.tile._numeric_semantics import RoundingMode as RMd

from tilegym.backend import register_impl

from .utils import next_power_of_2

PAD_ZERO = ct.PaddingMode.ZERO

def sigmoid(x):
    denom = ct.add(1.0, ct.exp(-x), flush_to_zero=True)
    return ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=RMd.APPROX)


def silu(x):
    return x * sigmoid(x)


def ceildiv(a, b):
    return -(a // -b)


@ct.kernel
def swiglu_forward_kernel(a, b, c, TILE_SIZE: ct.Constant[int]):
    row = ct.bid(0)
    offsets = ct.arange(TILE_SIZE, dtype=ct.int32)

    a_tile = ct.gather(a, (row, offsets), check_bounds=True, padding_value=0.0)
    b_tile = ct.gather(b, (row, offsets), check_bounds=True, padding_value=0.0)

    # Forward uses fast math knobs for throughput on Blackwell.
    a_tile_f32 = a_tile.astype(ct.float32)
    c_tile = silu(a_tile_f32).astype(a.dtype) * b_tile
    ct.scatter(c, (row, offsets), c_tile, check_bounds=True)


def swiglu_forward(a, b):
    """
    a: (batch_size, seq_len, intermediate_size)
    b: (batch_size, seq_len, intermediate_size)
    """
    ori_shape = a.shape
    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    TILE_SIZE = next_power_of_2(n_cols)
    grid = (n_rows,)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        swiglu_forward_kernel,
        (
            a,
            b,
            c,
            TILE_SIZE,
        ),
    )
    return c.view(*ori_shape)


# Backward kernel for swiglu
# Forward: c = silu(a) * b
# da = dc * b * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
#    = dc * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
# db = dc * silu(a)
@ct.kernel
def swiglu_backward_kernel(dc, a, b, da, db, TILE_SIZE: ct.Constant[int]):
    row = ct.bid(0)
    col = ct.bid(1)

    dc_tile = ct.load(dc, index=(row, col), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
    a_tile = ct.load(a, index=(row, col), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
    b_tile = ct.load(b, index=(row, col), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)

    # Convert to float32 for precision
    dc_tile = dc_tile.astype(ct.float32)
    a_tile_f32 = a_tile.astype(ct.float32)
    b_tile_f32 = b_tile.astype(ct.float32)

    # Compute sigmoid(a) and silu(a)
    sigmoid_a = 1.0 / (1.0 + ct.exp(-a_tile_f32))
    silu_a = a_tile_f32 * sigmoid_a

    # db = dc * silu(a)
    db_tile = dc_tile * silu_a
    ct.store(db, index=(row, col), tile=db_tile.astype(a.dtype))

    # da = dc * b * sigmoid(a) * (1 + a * (1 - sigmoid(a)))
    one_minus_sigmoid = 1.0 - sigmoid_a
    silu_grad = sigmoid_a * (1.0 + a_tile_f32 * one_minus_sigmoid)
    da_tile = dc_tile * b_tile_f32 * silu_grad
    ct.store(da, index=(row, col), tile=da_tile.astype(a.dtype))


def swiglu_backward(dc, a, b):
    """
    Backward pass for SwiGLU operation.

    Args:
        dc: Gradient w.r.t. output c, shape (batch_size, seq_len, intermediate_size)
        a: Original input a (gate), same shape as dc
        b: Original input b (up), same shape as dc

    Returns:
        Tuple of (da, db) with same shapes as inputs
    """
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols).contiguous()
    a = a.view(-1, n_cols).contiguous()
    b = b.view(-1, n_cols).contiguous()
    n_rows = dc.shape[0]

    da = torch.empty_like(a)
    db = torch.empty_like(b)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    TILE_N = ceildiv(NUM_SMS, n_rows)
    TILE_SIZE = next_power_of_2(int(n_cols / TILE_N))
    grid = (n_rows, ceildiv(n_cols, TILE_SIZE), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        swiglu_backward_kernel,
        (dc, a, b, da, db, TILE_SIZE),
    )
    return da.view(*ori_shape), db.view(*ori_shape)


class SiLUMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        da, db = swiglu_backward(dc, a, b)
        return da, db


class SwiGLUMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if config.hidden_act not in ["silu", "swish"]:
            raise ValueError(f"Activation function {config.hidden_act} not supported.")

    def forward(self, x):
        return self.down_proj(SiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


@register_impl("get_swiglu_module", backend="cutile")
def get_swiglu_module():
    return SwiGLUMLP


@register_impl("get_swiglu", backend="cutile")
def get_swiglu():
    return swiglu_forward
