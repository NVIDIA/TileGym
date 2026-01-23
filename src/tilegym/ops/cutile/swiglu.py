# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT


import cuda.tile as ct
import torch
import torch.nn as nn

from tilegym.backend import register_impl

from .utils import next_power_of_2

PAD_ZERO = ct.PaddingMode.ZERO


def sigmoid(x):
    return 1.0 / (1.0 + ct.exp(-x))


def silu(x):
    return x * sigmoid(x)


@ct.kernel
def swiglu_forward_kernel(a, b, c, TILE_SIZE: ct.Constant[int]):
    row = ct.bid(0)
    col = ct.bid(1)

    a_tile = ct.load(a, index=(row, col), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)
    b_tile = ct.load(b, index=(row, col), shape=(1, TILE_SIZE), padding_mode=PAD_ZERO)

    # Sigmoid requires type float32
    c_tile = silu(a_tile.astype(ct.float32)).astype(a.dtype) * b_tile
    ct.store(c, index=(row, col), tile=c_tile)


def ceildiv(a, b):
    return -(a // -b)


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

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    TILE_N = ceildiv(NUM_SMS, n_rows)
    TILE_SIZE = next_power_of_2(int(n_cols / TILE_N))
    grid = (n_rows, ceildiv(n_cols, TILE_SIZE), 1)
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        swiglu_forward_kernel,
        (
            a.data,
            b.data,
            c.data,
            TILE_SIZE,
        ),
    )
    return c.view(*ori_shape)


class SiLUMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        c = swiglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dc):
        raise NotImplementedError("SwiGLU backward is not implemented.")


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
