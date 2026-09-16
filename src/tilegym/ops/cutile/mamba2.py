# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Mamba-2 SSD forward and backward cuTile kernels."""

import math
from types import SimpleNamespace

import cuda.tile as ct
import torch
from cuda.tile.tune import exhaustive_search

from tilegym.autotune import is_autotune_disabled
from tilegym.backend import register_impl

LOG2E = math.log2(math.e)


NEG_BIG = -1.0e30


def _co_tile(
    batch,
    head,
    group,
    chunk_idx,
    head_tile_idx,
    x,
    b,
    c,
    d,
    dt,
    decay_cumsum,
    hiddens,
    out,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    state_size: ct.Constant[int],
    H_LAT: ct.Constant[int],
):
    num_state_tiles = state_size // BLOCK_STATE
    cb_tile = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
    out_tile = ct.full((chunk_size, BLOCK_HEAD), 0.0, dtype=ct.float32)
    for state_tile_idx in range(num_state_tiles):
        b_tile = ct.load(
            b,
            index=(batch, chunk_idx, group, state_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_STATE),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((chunk_size, BLOCK_STATE))
        c_tile = ct.load(
            c,
            index=(batch, chunk_idx, group, state_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_STATE),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((chunk_size, BLOCK_STATE))
        cb_tile = ct.mma(c_tile, b_tile.transpose(), cb_tile)
        h_tile = ct.load(
            hiddens,
            index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
            shape=(1, 1, 1, BLOCK_HEAD, BLOCK_STATE),
            latency=H_LAT,
        ).reshape((BLOCK_HEAD, BLOCK_STATE))
        out_tile = ct.mma(c_tile, h_tile.transpose().astype(c_tile.dtype), out_tile)
    d_tile = ct.load(
        decay_cumsum, index=(batch, head, chunk_idx), shape=(1, 1, chunk_size), padding_mode=ct.PaddingMode.ZERO
    ).reshape((chunk_size,))
    cb_tile = cb_tile * ct.exp2(d_tile[:, None] - d_tile[None, :])
    idx = ct.arange(chunk_size, dtype=ct.int32)
    cb_tile = ct.where(idx[:, None] >= idx[None, :], cb_tile, 0.0)
    out_tile = out_tile * ct.exp2(d_tile[:, None])
    dt_tile = ct.load(
        dt, index=(batch, head, chunk_idx), shape=(1, 1, chunk_size), padding_mode=ct.PaddingMode.ZERO
    ).reshape((chunk_size,))
    x_tile = ct.load(
        x,
        index=(batch, chunk_idx, head, head_tile_idx),
        shape=(1, chunk_size, 1, BLOCK_HEAD),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((chunk_size, BLOCK_HEAD))
    dt_x = dt_tile[:, None] * x_tile
    out_tile = ct.mma(cb_tile.astype(ct.tfloat32), dt_x.astype(ct.tfloat32), out_tile)
    d_val = ct.load(d, index=(head,), shape=(1,))
    out_tile = out_tile + d_val * x_tile
    ct.store(
        out,
        index=(batch, chunk_idx, head, head_tile_idx),
        tile=out_tile.reshape((1, chunk_size, 1, BLOCK_HEAD)).astype(out.dtype),
    )


def _co_swap_tile(
    batch,
    head,
    group,
    chunk_idx,
    head_tile_idx,
    x,
    b,
    c,
    d,
    dt,
    decay_cumsum,
    hiddens,
    out,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    state_size: ct.Constant[int],
    LATENCY: ct.Constant[int],
    H_LAT: ct.Constant[int],
):
    num_state_tiles = state_size // BLOCK_STATE
    cb_tile = ct.full((chunk_size, chunk_size), 0.0, dtype=ct.float32)
    out_T_tile = ct.full((BLOCK_HEAD, chunk_size), 0.0, dtype=ct.float32)
    for state_tile_idx in range(num_state_tiles):
        b_tile = ct.load(
            b,
            index=(batch, chunk_idx, group, state_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_STATE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LATENCY,
        ).reshape((chunk_size, BLOCK_STATE))
        c_tile = ct.load(
            c,
            index=(batch, chunk_idx, group, state_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_STATE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LATENCY,
        ).reshape((chunk_size, BLOCK_STATE))
        cb_tile = ct.mma(c_tile, b_tile.transpose(), cb_tile)
        h_tile = ct.load(
            hiddens,
            index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
            shape=(1, 1, 1, BLOCK_HEAD, BLOCK_STATE),
            latency=H_LAT,
        ).reshape((BLOCK_HEAD, BLOCK_STATE))
        out_T_tile = ct.mma(h_tile.astype(c_tile.dtype), c_tile.transpose(), out_T_tile)
    d_tile = ct.load(
        decay_cumsum, index=(batch, head, chunk_idx), shape=(1, 1, chunk_size), padding_mode=ct.PaddingMode.ZERO
    ).reshape((chunk_size,))
    cb_tile = cb_tile * ct.exp2(d_tile[:, None] - d_tile[None, :])
    idx = ct.arange(chunk_size, dtype=ct.int32)
    cb_tile = ct.where(idx[:, None] >= idx[None, :], cb_tile, 0.0)
    out_T_tile = out_T_tile * ct.exp2(d_tile[None, :])
    dt_tile = ct.load(
        dt, index=(batch, head, chunk_idx), shape=(1, 1, chunk_size), padding_mode=ct.PaddingMode.ZERO
    ).reshape((chunk_size,))
    x_tile = ct.load(
        x,
        index=(batch, chunk_idx, head, head_tile_idx),
        shape=(1, chunk_size, 1, BLOCK_HEAD),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((chunk_size, BLOCK_HEAD))
    dt_x = dt_tile[:, None] * x_tile
    out_T_tile = ct.mma(dt_x.transpose().astype(ct.tfloat32), cb_tile.transpose().astype(ct.tfloat32), out_T_tile)
    d_val = ct.load(d, index=(head,), shape=(1,))
    out_T_tile = out_T_tile + d_val * x_tile.transpose()
    out_tile = out_T_tile.transpose()
    ct.store(
        out,
        index=(batch, chunk_idx, head, head_tile_idx),
        tile=out_tile.reshape((1, chunk_size, 1, BLOCK_HEAD)).astype(out.dtype),
    )


def _compute_out_tiles(
    kind,
    x,
    b,
    c,
    d,
    dt,
    decay_cumsum,
    hiddens,
    out,
    chunk_size,
    BLOCK_STATE,
    BLOCK_HEAD,
    num_heads_per_group,
    state_size,
    LATENCY,
    H_LAT,
    persistent,
):
    if persistent:
        start_idx = ct.bid(0)
        num_sms = ct.num_blocks(0)
        num_chunks = c.shape[1] // chunk_size
        num_head_tiles = x.shape[-1] // BLOCK_HEAD
        num_heads = x.shape[2]
        batch_size = c.shape[0]
        total = num_chunks * num_head_tiles * num_heads * batch_size
        for tile_idx in range(start_idx, total, num_sms):
            head = tile_idx % num_heads
            idx = tile_idx // num_heads
            head_tile_idx = idx % num_head_tiles
            idx = idx // num_head_tiles
            chunk_idx = idx % num_chunks
            batch = idx // num_chunks
            group = head // num_heads_per_group
            if kind == 0:
                _co_tile(
                    batch,
                    head,
                    group,
                    chunk_idx,
                    head_tile_idx,
                    x,
                    b,
                    c,
                    d,
                    dt,
                    decay_cumsum,
                    hiddens,
                    out,
                    chunk_size,
                    BLOCK_STATE,
                    BLOCK_HEAD,
                    state_size,
                    H_LAT,
                )
            else:
                _co_swap_tile(
                    batch,
                    head,
                    group,
                    chunk_idx,
                    head_tile_idx,
                    x,
                    b,
                    c,
                    d,
                    dt,
                    decay_cumsum,
                    hiddens,
                    out,
                    chunk_size,
                    BLOCK_STATE,
                    BLOCK_HEAD,
                    state_size,
                    LATENCY,
                    H_LAT,
                )
    else:
        chunk_idx, head_tile_idx, batch_head = ct.bid(0), ct.bid(1), ct.bid(2)
        batch = batch_head // x.shape[2]
        head = batch_head % x.shape[2]
        group = head // num_heads_per_group
        if kind == 0:
            _co_tile(
                batch,
                head,
                group,
                chunk_idx,
                head_tile_idx,
                x,
                b,
                c,
                d,
                dt,
                decay_cumsum,
                hiddens,
                out,
                chunk_size,
                BLOCK_STATE,
                BLOCK_HEAD,
                state_size,
                H_LAT,
            )
        else:
            _co_swap_tile(
                batch,
                head,
                group,
                chunk_idx,
                head_tile_idx,
                x,
                b,
                c,
                d,
                dt,
                decay_cumsum,
                hiddens,
                out,
                chunk_size,
                BLOCK_STATE,
                BLOCK_HEAD,
                state_size,
                LATENCY,
                H_LAT,
            )


def _compute_dd_dcb_dx_tile(
    batch: int,
    head: int,
    group: int,
    chunk_idx: int,
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    dt: torch.Tensor,
    decay_cumsum: torch.Tensor,
    dout: torch.Tensor,
    dhiddens: torch.Tensor,
    dx: torch.Tensor,
    ddt: torch.Tensor,
    dd: torch.Tensor,
    ddecay_cumsum: torch.Tensor,
    dcb: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    state_size: ct.Constant[int],
    head_size: ct.Constant[int],
    LOAD_LAT: ct.Constant[int],
    STORE_LAT: ct.Constant[int],
    DOUT_LAT: ct.Constant[int],
    DHID_LAT: ct.Constant[int],
) -> None:
    num_head_tiles = head_size // BLOCK_HEAD
    num_state_tiles = state_size // BLOCK_STATE
    f32 = ct.float32

    d_tile = ct.load(
        decay_cumsum,
        index=(batch, head, chunk_idx),
        shape=(1, 1, chunk_size),
        padding_mode=ct.PaddingMode.ZERO,
        latency=LOAD_LAT,
    ).reshape((chunk_size,))
    dt_tile = (
        ct.load(
            dt,
            index=(batch, head, chunk_idx),
            shape=(1, 1, chunk_size),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LAT,
        )
        .reshape((chunk_size,))
        .astype(f32)
    )
    d_last = ct.extract(d_tile, index=(chunk_size - 1,), shape=(1,))
    d_val = ct.load(d, index=(head,), shape=(1,))

    idx = ct.arange(chunk_size, dtype=ct.int32)
    causal_mask_t = idx[None, :] >= idx[:, None]

    W = ct.exp2(ct.where(causal_mask_t, d_tile[None, :] - d_tile[:, None], NEG_BIG))
    coef = ct.exp2(d_last - d_tile)

    xdoT = ct.full((chunk_size, chunk_size), 0.0, dtype=f32)
    rawT = ct.full((chunk_size, chunk_size), 0.0, dtype=f32)
    for head_tile_idx in range(num_head_tiles):
        x_tile = ct.load(
            x,
            index=(batch, chunk_idx, head, head_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_HEAD),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LAT,
        ).reshape((chunk_size, BLOCK_HEAD))
        dout_tile = ct.load(
            dout,
            index=(batch, chunk_idx, head, head_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_HEAD),
            padding_mode=ct.PaddingMode.ZERO,
            latency=DOUT_LAT,
        ).reshape((chunk_size, BLOCK_HEAD))
        xdoT = ct.mma(x_tile, dout_tile.transpose(), xdoT)

        bdh = ct.full((chunk_size, BLOCK_HEAD), 0.0, dtype=f32)
        for state_tile_idx in range(num_state_tiles):
            b_tile = ct.load(
                b,
                index=(batch, chunk_idx, group, state_tile_idx),
                shape=(1, chunk_size, 1, BLOCK_STATE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=LOAD_LAT,
            ).reshape((chunk_size, BLOCK_STATE))
            if head_tile_idx == 0:
                c_tile = ct.load(
                    c,
                    index=(batch, chunk_idx, group, state_tile_idx),
                    shape=(1, chunk_size, 1, BLOCK_STATE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=LOAD_LAT,
                ).reshape((chunk_size, BLOCK_STATE))
                rawT = ct.mma(b_tile, c_tile.transpose(), rawT)

            dhid_tile = ct.load(
                dhiddens,
                index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
                shape=(1, 1, 1, BLOCK_HEAD, BLOCK_STATE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=DHID_LAT,
            ).reshape((BLOCK_HEAD, BLOCK_STATE))
            bdh = ct.mma(b_tile, dhid_tile.transpose().astype(b.dtype), bdh)

        cbT = rawT * W
        dxtd = ct.mma(cbT.astype(dout_tile.dtype), dout_tile, ct.full((chunk_size, BLOCK_HEAD), 0.0, dtype=f32))
        dx_tile = d_val * dout_tile + (dxtd + bdh * coef[:, None]) * dt_tile[:, None]
        ct.store(
            dx,
            index=(batch, chunk_idx, head, head_tile_idx),
            tile=dx_tile.reshape((1, chunk_size, 1, BLOCK_HEAD)).astype(dx.dtype),
            latency=STORE_LAT,
        )

    dd_row = ct.sum(ct.where(idx[None, :] == idx[:, None], xdoT, 0.0), axis=1)
    ct.store(dd, index=(batch, head, chunk_idx), tile=ct.sum(dd_row, axis=0).reshape((1, 1, 1)), latency=STORE_LAT)

    dcbT_out = xdoT * W * dt_tile[:, None]
    ct.store(
        dcb,
        index=(batch, chunk_idx, head, 0),
        tile=dcbT_out.reshape((1, chunk_size, 1, chunk_size)).astype(dcb.dtype),
        latency=STORE_LAT,
    )

    dcbcb = dcbT_out * rawT
    rs = ct.sum(dcbcb, axis=1)
    ct.store(
        ddecay_cumsum,
        index=(batch, chunk_idx, head),
        tile=(ct.sum(dcbcb, axis=0) - rs).reshape((1, chunk_size, 1)),
        latency=STORE_LAT,
    )
    ct.store(
        ddt,
        index=(batch, chunk_idx, head),
        tile=(rs / dt_tile).reshape((1, chunk_size, 1)).astype(ddt.dtype),
        latency=STORE_LAT,
    )


def _compute_da_db_dc_ddt_tile(
    batch: int,
    head: int,
    group: int,
    chunk_idx: int,
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dt: torch.Tensor,
    decay_cumsum: torch.Tensor,
    hiddens: torch.Tensor,
    dout: torch.Tensor,
    dhiddens: torch.Tensor,
    dcb: torch.Tensor,
    ddecay_cumsum: torch.Tensor,
    dd_last_partials: torch.Tensor,
    da: torch.Tensor,
    db: torch.Tensor,
    dc: torch.Tensor,
    ddt_partial: torch.Tensor,
    ddt: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    state_size: ct.Constant[int],
    head_size: ct.Constant[int],
    ND_LEN: ct.Constant[int],
    LOAD_LAT: ct.Constant[int],
    STORE_LAT: ct.Constant[int],
    REDUCE_HEADS: ct.Constant[bool],
) -> None:
    num_head_tiles = head_size // BLOCK_HEAD
    num_state_tiles = state_size // BLOCK_STATE

    d_tile = ct.load(
        decay_cumsum,
        index=(batch, head, chunk_idx),
        shape=(1, 1, chunk_size),
        padding_mode=ct.PaddingMode.ZERO,
        latency=LOAD_LAT,
    ).reshape((chunk_size,))
    d_last = ct.extract(d_tile, index=(chunk_size - 1,), shape=(1,))
    dt_tile = ct.load(
        dt,
        index=(batch, head, chunk_idx),
        shape=(1, 1, chunk_size),
        padding_mode=ct.PaddingMode.ZERO,
        latency=LOAD_LAT,
    ).reshape((chunk_size,))
    dcb_tile = ct.load(
        dcb,
        index=(batch, chunk_idx, head, 0),
        shape=(1, chunk_size, 1, chunk_size),
        padding_mode=ct.PaddingMode.ZERO,
        latency=LOAD_LAT,
    ).reshape((chunk_size, chunk_size))
    dd_tile = ct.load(
        ddecay_cumsum,
        index=(batch, chunk_idx, head),
        shape=(1, chunk_size, 1),
        padding_mode=ct.PaddingMode.ZERO,
        latency=LOAD_LAT,
    ).reshape((chunk_size,))
    ddt_tile = (
        ct.load(
            ddt_partial,
            index=(batch, chunk_idx, head),
            shape=(1, chunk_size, 1),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LAT,
        )
        .reshape((chunk_size,))
        .astype(ct.float32)
    )

    partials = ct.load(
        dd_last_partials,
        index=(batch, head, chunk_idx, 0),
        shape=(1, 1, 1, ND_LEN),
        padding_mode=ct.PaddingMode.ZERO,
        latency=LOAD_LAT,
    ).reshape((ND_LEN,))
    dd_last = ct.sum(partials, axis=0) * ct.exp2(d_last)
    acc_dg = ct.full((chunk_size,), 0.0, dtype=ct.float32)
    for state_tile_idx in range(num_state_tiles):
        b_tile = ct.load(
            b,
            index=(batch, chunk_idx, group, state_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_STATE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LAT,
        ).reshape((chunk_size, BLOCK_STATE))
        c_tile = ct.load(
            c,
            index=(batch, chunk_idx, group, state_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_STATE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LAT,
        ).reshape((chunk_size, BLOCK_STATE))

        db_tile = ct.full((chunk_size, BLOCK_STATE), 0.0, dtype=ct.float32)
        dc_tile = ct.full((chunk_size, BLOCK_STATE), 0.0, dtype=ct.float32)
        db_tile = ct.mma(dcb_tile.astype(c_tile.dtype), c_tile, db_tile)
        dc_tile = ct.mma(dcb_tile.transpose().astype(b_tile.dtype), b_tile, dc_tile)

        dc_d = ct.full((chunk_size, BLOCK_STATE), 0.0, dtype=ct.float32)
        db_dt_d = ct.full((chunk_size, BLOCK_STATE), 0.0, dtype=ct.float32)
        for head_tile_idx in range(num_head_tiles):
            dout_tile = ct.load(
                dout,
                index=(batch, chunk_idx, head, head_tile_idx),
                shape=(1, chunk_size, 1, BLOCK_HEAD),
                padding_mode=ct.PaddingMode.ZERO,
                latency=LOAD_LAT,
            ).reshape((chunk_size, BLOCK_HEAD))
            hid_tile = ct.load(
                hiddens,
                index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
                shape=(1, 1, 1, BLOCK_HEAD, BLOCK_STATE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=LOAD_LAT,
            ).reshape((BLOCK_HEAD, BLOCK_STATE))

            dc_d = ct.mma(dout_tile, hid_tile.astype(dout_tile.dtype), dc_d)

            dhid_tile = ct.load(
                dhiddens,
                index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
                shape=(1, 1, 1, BLOCK_HEAD, BLOCK_STATE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=LOAD_LAT,
            ).reshape((BLOCK_HEAD, BLOCK_STATE))

            x_tile = ct.load(
                x,
                index=(batch, chunk_idx, head, head_tile_idx),
                shape=(1, chunk_size, 1, BLOCK_HEAD),
                padding_mode=ct.PaddingMode.ZERO,
                latency=LOAD_LAT,
            ).reshape((chunk_size, BLOCK_HEAD))

            db_dt_d = ct.mma(x_tile, dhid_tile.astype(x_tile.dtype), db_dt_d)

        dc_d = dc_d * ct.exp2(d_tile[:, None])
        dc_tile += dc_d
        dd_tile += ct.sum(dc_d * c_tile.astype(ct.float32), axis=1)

        db_dt_d = db_dt_d * ct.exp2(d_last - d_tile[:, None])
        db_tile += db_dt_d * dt_tile[:, None].astype(ct.float32)
        ddt_tile += ct.sum(db_dt_d * b_tile.astype(ct.float32), axis=1)
        unreduced_dg = db_dt_d * b_tile.astype(ct.float32) * dt_tile[:, None].astype(ct.float32)

        _r1 = ct.sum(unreduced_dg, axis=1)
        dd_tile -= _r1
        acc_dg += _r1

        if REDUCE_HEADS:
            dc.tiled_view((1, chunk_size, 1, BLOCK_STATE)).atomic_store_add(
                (batch, chunk_idx, group, state_tile_idx), dc_tile.reshape((1, chunk_size, 1, BLOCK_STATE))
            )
            db.tiled_view((1, chunk_size, 1, BLOCK_STATE)).atomic_store_add(
                (batch, chunk_idx, group, state_tile_idx), db_tile.reshape((1, chunk_size, 1, BLOCK_STATE))
            )
        else:
            ct.store(
                dc,
                index=(batch, chunk_idx, head, state_tile_idx),
                tile=dc_tile.reshape((1, chunk_size, 1, BLOCK_STATE)).astype(dc.dtype),
                latency=STORE_LAT,
            )
            ct.store(
                db,
                index=(batch, chunk_idx, head, state_tile_idx),
                tile=db_tile.reshape((1, chunk_size, 1, BLOCK_STATE)).astype(db.dtype),
                latency=STORE_LAT,
            )

    dd_last += ct.sum(acc_dg, axis=0)
    cumsum_result = ct.cumsum(dd_tile, axis=0)

    dd_tile = dd_last + ct.extract(cumsum_result, index=(chunk_size - 1,), shape=(1,)) - cumsum_result + dd_tile

    a_val = ct.load(a, index=(head,), shape=(1,))
    ddt_tile += dd_tile * a_val
    da_val = ct.sum(dt_tile * dd_tile)
    ct.store(
        ddt,
        index=(batch, chunk_idx, head),
        tile=ddt_tile.reshape((1, chunk_size, 1)).astype(ddt.dtype),
        latency=STORE_LAT,
    )
    ct.store(
        da,
        index=(batch, head, chunk_idx),
        tile=da_val.reshape((1, 1, 1)).astype(da.dtype),
        latency=STORE_LAT,
    )


def _chunk_cumsum_autotune_configs(num_heads):
    for BLOCK_H in [4, 8, 16]:
        for occupancy in [4, 5, 6, 7, 8]:
            if num_heads % BLOCK_H == 0:
                yield SimpleNamespace(BLOCK_H=BLOCK_H, occupancy=occupancy)


def _fwd_pass_state_autotune_configs(state_size, head_size, chunk_size):
    variants = [(False, True), (True, True)]
    if chunk_size >= 128:
        variants.append((False, False))
    for SWAP, BHT in variants:
        for BLOCK_STATE in [16, 32, 64]:
            if state_size % BLOCK_STATE != 0:
                continue
            for BLOCK_HEAD in [min(64, head_size)]:
                if head_size % BLOCK_HEAD != 0:
                    continue
                for LOAD_LAT in [4, 6]:
                    for BDD_LAT in [2, 4]:
                        for STORE_LAT in [2]:
                            yield SimpleNamespace(
                                SWAP=SWAP,
                                BHT=BHT,
                                BLOCK_STATE=BLOCK_STATE,
                                BLOCK_HEAD=BLOCK_HEAD,
                                occupancy=2,
                                num_ctas=1,
                                LOAD_LAT=LOAD_LAT,
                                BDD_LAT=BDD_LAT,
                                STORE_LAT=STORE_LAT,
                            )


def _compute_out_configs(head_size, state_size):
    block_head = min(64, head_size)
    if head_size % block_head != 0:
        return
    for BLOCK_STATE in [16, 32, 64, 128]:
        if state_size % BLOCK_STATE != 0:
            continue
        for occupancy in [1, 2]:
            for persistent in [True, False]:
                for H_LAT in [3, 4, 5, 6]:
                    yield SimpleNamespace(
                        BLOCK_STATE=BLOCK_STATE,
                        BLOCK_HEAD=block_head,
                        occupancy=occupancy,
                        persistent=persistent,
                        H_LAT=H_LAT,
                    )


def _compute_out_swap_configs(head_size, state_size):
    block_head = min(64, head_size)
    if head_size % block_head != 0:
        return
    for BLOCK_STATE in [16, 32, 64, 128]:
        if state_size % BLOCK_STATE != 0:
            continue
        for occupancy in [1, 2]:
            for LATENCY in [3, 4, 6]:
                for H_LAT in [3, 4, 5]:
                    yield SimpleNamespace(
                        BLOCK_STATE=BLOCK_STATE,
                        BLOCK_HEAD=block_head,
                        occupancy=occupancy,
                        persistent=True,
                        LATENCY=LATENCY,
                        H_LAT=H_LAT,
                    )


def _bwd_pass_state_base_configs(head_size):
    for BLOCK_STATE in [16, 32, 64]:
        for BLOCK_HEAD in [min(64, head_size)]:
            for occupancy in [2]:
                for num_ctas in [1]:
                    for LOAD_LAT in [4, 5, 6]:
                        for STORE_LAT in [1, 2]:
                            for CD_LAT in [3, 5, 6]:
                                if CD_LAT > LOAD_LAT:
                                    continue
                                yield SimpleNamespace(
                                    BLOCK_STATE=BLOCK_STATE,
                                    BLOCK_HEAD=BLOCK_HEAD,
                                    occupancy=occupancy,
                                    num_ctas=num_ctas,
                                    LOAD_LAT=LOAD_LAT,
                                    STORE_LAT=STORE_LAT,
                                    CD_LAT=CD_LAT,
                                )


def _bwd_pass_state_autotune_configs(state_size, head_size):
    for cfg in _bwd_pass_state_base_configs(head_size):
        if (
            head_size % cfg.BLOCK_HEAD
            or state_size % cfg.BLOCK_STATE
            or (cfg.BLOCK_HEAD * cfg.BLOCK_STATE) % _PARTIALS_PER_TILE
        ):
            continue
        for hid_lat in [3, 5, 6]:
            yield SimpleNamespace(**vars(cfg), HID_LAT=hid_lat)


_PRUNED = {
    64: ((32, 64), (3, 4, 5), (5, 6, 7), (5, 7)),
    128: ((32, 64, 128), (4, 5), (5, 7), (5, 6, 7)),
    256: ((32, 64, 128), (4, 5), (5, 6), (5, 6)),
    512: ((32, 128), (4, 5), (7,), (5, 6)),
}

_FALLBACK = ((16, 32, 64, 128), (3, 4, 5), (5, 6, 7), (5, 6, 7))


def _compute_dd_dcb_dx_autotune_configs(state_size, head_size):
    bs_d, load_d, dout_d, dhid_d = _PRUNED.get(state_size, _FALLBACK)
    for occ in (1, 2):
        for BLOCK_STATE in bs_d:
            if state_size % BLOCK_STATE:
                continue
            for LOAD_LAT in load_d:
                for DOUT_LAT in dout_d:
                    for DHID_LAT in dhid_d:
                        for STORE_LAT in (1,):
                            yield SimpleNamespace(
                                BLOCK_STATE=BLOCK_STATE,
                                BLOCK_HEAD=head_size,
                                occupancy=occ,
                                persistent=True,
                                LOAD_LAT=LOAD_LAT,
                                STORE_LAT=STORE_LAT,
                                DOUT_LAT=DOUT_LAT,
                                DHID_LAT=DHID_LAT,
                            )


def _compute_da_db_dc_ddt_autotune_configs(head_size, state_size):
    BLOCK_HEAD = min(64, head_size)
    if head_size % BLOCK_HEAD:
        return
    for BLOCK_STATE in [16, 32]:
        if state_size % BLOCK_STATE:
            continue
        for occ in [2, 4]:
            for pers in [True, False]:
                for LOAD_LAT in [3, 4, 5, 6]:
                    for STORE_LAT in [1, 2]:
                        yield SimpleNamespace(
                            BLOCK_STATE=BLOCK_STATE,
                            BLOCK_HEAD=BLOCK_HEAD,
                            occupancy=occ,
                            persistent=pers,
                            LOAD_LAT=LOAD_LAT,
                            STORE_LAT=STORE_LAT,
                        )


@ct.kernel
def chunk_cumsum_kernel(
    a: torch.Tensor,
    dt: torch.Tensor,
    out_bth: torch.Tensor,
    out_bht: torch.Tensor,
    dt_bht: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
) -> None:
    chunk_idx, head_tile_idx, batch = ct.bid(0), ct.bid(1), ct.bid(2)
    dt_raw = ct.load(
        dt, index=(batch, chunk_idx, head_tile_idx), shape=(1, chunk_size, BLOCK_H), padding_mode=ct.PaddingMode.ZERO
    )
    dt_tile = dt_raw.astype(ct.float32)
    a_tile = ct.load(a, index=(head_tile_idx,), shape=(BLOCK_H,), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    d_tile = ct.cumsum(a_tile * dt_tile, axis=1) * LOG2E
    ct.store(out_bth, index=(batch, chunk_idx, head_tile_idx), tile=d_tile.astype(out_bth.dtype))
    d_T = ct.permute(d_tile, (0, 2, 1))
    ct.store(out_bht, index=(batch, head_tile_idx, chunk_idx), tile=d_T.astype(out_bht.dtype))
    dt_T = ct.permute(dt_raw, (0, 2, 1))
    ct.store(dt_bht, index=(batch, head_tile_idx, chunk_idx), tile=dt_T.astype(dt_bht.dtype))


@ct.kernel
def fwd_pass_state_kernel(
    x: torch.Tensor,
    b: torch.Tensor,
    dt: torch.Tensor,
    decay_cumsum: torch.Tensor,
    init_state: torch.Tensor,
    hiddens: torch.Tensor,
    hid_narrow: torch.Tensor,
    final_state: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    num_heads_per_group: ct.Constant[int],
    LOAD_LAT: ct.Constant[int],
    STORE_LAT: ct.Constant[int],
    BDD_LAT: ct.Constant[int],
    SWAP: ct.Constant[bool],
    BHT: ct.Constant[bool],
    DUAL: ct.Constant[bool],
) -> None:
    head_tile_idx, state_tile_idx, batch_head = ct.bid(0), ct.bid(1), ct.bid(2)
    batch = batch_head // x.shape[2]
    head = batch_head % x.shape[2]
    group = head // num_heads_per_group

    if SWAP:
        hid_tile = ct.load(
            init_state,
            index=(batch, head, state_tile_idx, head_tile_idx),
            shape=(1, 1, BLOCK_STATE, BLOCK_HEAD),
            order=(0, 1, 3, 2),
            latency=LOAD_LAT,
        ).reshape((BLOCK_STATE, BLOCK_HEAD))
    else:
        hid_tile = ct.load(
            init_state,
            index=(batch, head, head_tile_idx, state_tile_idx),
            shape=(1, 1, BLOCK_HEAD, BLOCK_STATE),
            latency=LOAD_LAT,
        ).reshape((BLOCK_HEAD, BLOCK_STATE))

    num_chunks = ct.cdiv(b.shape[1], chunk_size)
    for chunk_idx in range(num_chunks):
        if SWAP:
            ct.store(
                hiddens,
                index=(batch, head, chunk_idx, state_tile_idx, head_tile_idx),
                tile=hid_tile.reshape((1, 1, 1, BLOCK_STATE, BLOCK_HEAD)).astype(hiddens.dtype),
                order=(0, 1, 2, 4, 3),
                latency=STORE_LAT,
            )
            if DUAL:
                ct.store(
                    hid_narrow,
                    index=(batch, head, chunk_idx, state_tile_idx, head_tile_idx),
                    tile=hid_tile.reshape((1, 1, 1, BLOCK_STATE, BLOCK_HEAD)).astype(hid_narrow.dtype),
                    order=(0, 1, 2, 4, 3),
                    latency=STORE_LAT,
                )
        else:
            ct.store(
                hiddens,
                index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
                tile=hid_tile.reshape((1, 1, 1, BLOCK_HEAD, BLOCK_STATE)).astype(hiddens.dtype),
                latency=STORE_LAT,
            )
            if DUAL:
                ct.store(
                    hid_narrow,
                    index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
                    tile=hid_tile.reshape((1, 1, 1, BLOCK_HEAD, BLOCK_STATE)).astype(hid_narrow.dtype),
                    latency=STORE_LAT,
                )

        x_tile = ct.load(
            x,
            index=(batch, chunk_idx, head, head_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_HEAD),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LAT,
        ).reshape((chunk_size, BLOCK_HEAD))

        if SWAP:
            b_tile_t = ct.load(
                b,
                index=(batch, state_tile_idx, group, chunk_idx),
                shape=(1, BLOCK_STATE, 1, chunk_size),
                order=(0, 3, 2, 1),
                padding_mode=ct.PaddingMode.ZERO,
                latency=BDD_LAT,
            ).reshape((BLOCK_STATE, chunk_size))
        else:
            b_tile = ct.load(
                b,
                index=(batch, chunk_idx, group, state_tile_idx),
                shape=(1, chunk_size, 1, BLOCK_STATE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=BDD_LAT,
            ).reshape((chunk_size, BLOCK_STATE))

        if BHT:
            dt_tile = ct.load(
                dt,
                index=(batch, head, chunk_idx),
                shape=(1, 1, chunk_size),
                padding_mode=ct.PaddingMode.ZERO,
                latency=BDD_LAT,
            ).reshape((chunk_size,))
            d_tile = ct.load(
                decay_cumsum,
                index=(batch, head, chunk_idx),
                shape=(1, 1, chunk_size),
                padding_mode=ct.PaddingMode.ZERO,
                latency=BDD_LAT,
            ).reshape((chunk_size,))
        else:
            dt_tile = ct.load(
                dt,
                index=(batch, chunk_idx, head),
                shape=(1, chunk_size, 1),
                padding_mode=ct.PaddingMode.ZERO,
                latency=BDD_LAT,
            ).reshape((chunk_size,))
            d_tile = ct.load(
                decay_cumsum,
                index=(batch, chunk_idx, head),
                shape=(1, chunk_size, 1),
                padding_mode=ct.PaddingMode.ZERO,
                latency=BDD_LAT,
            ).reshape((chunk_size,))
        d_last = ct.extract(d_tile, index=(chunk_size - 1,), shape=(1,))

        if SWAP:
            b_dt_d_t = b_tile_t * (dt_tile * ct.exp2(d_last - d_tile))[None, :]
            hid_tile = ct.exp2(d_last) * hid_tile
            hid_tile = ct.mma(b_dt_d_t.astype(b.dtype), x_tile, hid_tile)
        else:
            b_dt_d = b_tile * (dt_tile * ct.exp2(d_last - d_tile))[:, None]
            hid_tile = ct.exp2(d_last) * hid_tile
            hid_tile = ct.mma(x_tile.transpose(), b_dt_d.astype(b.dtype), hid_tile)

    if SWAP:
        ct.store(
            final_state,
            index=(batch, head, state_tile_idx, head_tile_idx),
            tile=hid_tile.reshape((1, 1, BLOCK_STATE, BLOCK_HEAD)),
            order=(0, 1, 3, 2),
            latency=STORE_LAT,
        )
    else:
        ct.store(
            final_state,
            index=(batch, head, head_tile_idx, state_tile_idx),
            tile=hid_tile.reshape((1, 1, BLOCK_HEAD, BLOCK_STATE)),
            latency=STORE_LAT,
        )


@ct.kernel
def compute_out_kernel(
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    dt: torch.Tensor,
    decay_cumsum: torch.Tensor,
    hiddens: torch.Tensor,
    out: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    num_heads_per_group: ct.Constant[int],
    state_size: ct.Constant[int],
    H_LAT: ct.Constant[int],
    persistent: ct.Constant[bool],
) -> None:
    _compute_out_tiles(
        0,
        x,
        b,
        c,
        d,
        dt,
        decay_cumsum,
        hiddens,
        out,
        chunk_size,
        BLOCK_STATE,
        BLOCK_HEAD,
        num_heads_per_group,
        state_size,
        3,
        H_LAT,
        persistent,
    )


@ct.kernel
def compute_out_swap_kernel(
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    dt: torch.Tensor,
    decay_cumsum: torch.Tensor,
    hiddens: torch.Tensor,
    out: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    num_heads_per_group: ct.Constant[int],
    state_size: ct.Constant[int],
    LATENCY: ct.Constant[int],
    H_LAT: ct.Constant[int],
    persistent: ct.Constant[bool],
) -> None:
    _compute_out_tiles(
        1,
        x,
        b,
        c,
        d,
        dt,
        decay_cumsum,
        hiddens,
        out,
        chunk_size,
        BLOCK_STATE,
        BLOCK_HEAD,
        num_heads_per_group,
        state_size,
        LATENCY,
        H_LAT,
        persistent,
    )


_REGROUP_BLOCK_HEAD = 64
_REGROUP_MIN_BLOCK_STATE = 8
_PARTIALS_PER_TILE = 128


@ct.kernel
def bwd_pass_state_kernel(
    c: torch.Tensor,
    decay_cumsum: torch.Tensor,
    dout: torch.Tensor,
    dfinal_state: torch.Tensor,
    dhiddens: torch.Tensor,
    hiddens: torch.Tensor,
    dd_last_partials: torch.Tensor,
    dinit_state: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    num_heads_per_group: ct.Constant[int],
    LOAD_LAT: ct.Constant[int],
    STORE_LAT: ct.Constant[int],
    CD_LAT: ct.Constant[int],
    HID_LAT: ct.Constant[int],
    WITH_PARTIALS: ct.Constant[bool],
) -> None:
    head_tile_idx, state_tile_idx, batch_head = ct.bid(0), ct.bid(1), ct.bid(2)
    batch = batch_head // dout.shape[2]
    head = batch_head % dout.shape[2]
    group = head // num_heads_per_group

    dhid_tile = ct.load(
        dfinal_state,
        index=(batch, head, head_tile_idx, state_tile_idx),
        shape=(1, 1, BLOCK_HEAD, BLOCK_STATE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=LOAD_LAT,
    ).reshape((BLOCK_HEAD, BLOCK_STATE))
    num_chunks = ct.cdiv(dout.shape[1], chunk_size)
    for i in range(num_chunks):
        chunk_idx = num_chunks - 1 - i
        ct.store(
            dhiddens,
            index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
            tile=dhid_tile.reshape((1, 1, 1, BLOCK_HEAD, BLOCK_STATE)).astype(dhiddens.dtype),
            latency=STORE_LAT,
        )
        if WITH_PARTIALS:
            hid_tile = ct.load(
                hiddens,
                index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx),
                shape=(1, 1, 1, BLOCK_HEAD, BLOCK_STATE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=HID_LAT,
            ).reshape((BLOCK_HEAD, BLOCK_STATE))

            hid_dhid = hid_tile * dhid_tile
            if BLOCK_HEAD == _REGROUP_BLOCK_HEAD and BLOCK_STATE % _REGROUP_MIN_BLOCK_STATE == 0:
                by_mode = hid_dhid.reshape((4, 2, 8, BLOCK_STATE // 8, 4, 2))
                thread_major = ct.permute(by_mode, (0, 2, 4, 1, 3, 5)).reshape(
                    (_PARTIALS_PER_TILE, BLOCK_HEAD * BLOCK_STATE // _PARTIALS_PER_TILE)
                )
            else:
                thread_major = hid_dhid.reshape((_PARTIALS_PER_TILE, BLOCK_HEAD * BLOCK_STATE // _PARTIALS_PER_TILE))
            dd_last_partial = ct.sum(thread_major, axis=1)

            ct.store(
                dd_last_partials,
                index=(batch, head, chunk_idx, head_tile_idx, state_tile_idx, 0),
                tile=dd_last_partial.reshape((1, 1, 1, 1, 1, _PARTIALS_PER_TILE)),
                latency=STORE_LAT,
            )
        dout_tile_t = ct.load(
            dout,
            index=(batch, head_tile_idx, head, chunk_idx),
            shape=(1, BLOCK_HEAD, 1, chunk_size),
            order=(0, 3, 2, 1),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LAT,
        ).reshape((BLOCK_HEAD, chunk_size))
        c_tile = ct.load(
            c,
            index=(batch, chunk_idx, group, state_tile_idx),
            shape=(1, chunk_size, 1, BLOCK_STATE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=CD_LAT,
        ).reshape((chunk_size, BLOCK_STATE))
        d_tile = ct.load(
            decay_cumsum,
            index=(batch, chunk_idx, head),
            shape=(1, chunk_size, 1),
            padding_mode=ct.PaddingMode.ZERO,
            latency=CD_LAT,
        ).reshape((chunk_size,))
        d_last = ct.extract(d_tile, index=(chunk_size - 1,), shape=(1,))

        c_d = c_tile * ct.exp2(d_tile)[:, None]
        dhid_tile = ct.exp2(d_last) * dhid_tile
        dhid_tile = ct.mma(dout_tile_t, c_d.astype(dout.dtype), dhid_tile)

    ct.store(
        dinit_state,
        index=(batch, head, head_tile_idx, state_tile_idx),
        tile=dhid_tile.reshape((1, 1, BLOCK_HEAD, BLOCK_STATE)),
        latency=STORE_LAT,
    )


@ct.kernel
def compute_dd_dcb_dx_kernel(
    x: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    dt: torch.Tensor,
    decay_cumsum: torch.Tensor,
    dout: torch.Tensor,
    dhiddens: torch.Tensor,
    dx: torch.Tensor,
    ddt: torch.Tensor,
    dd: torch.Tensor,
    ddecay_cumsum: torch.Tensor,
    dcb: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    num_heads_per_group: ct.Constant[int],
    state_size: ct.Constant[int],
    head_size: ct.Constant[int],
    LOAD_LAT: ct.Constant[int],
    STORE_LAT: ct.Constant[int],
    DOUT_LAT: ct.Constant[int],
    DHID_LAT: ct.Constant[int],
    persistent: ct.Constant[bool],
) -> None:
    if persistent:
        start_idx = ct.bid(0)
        num_sms = ct.num_blocks(0)
        num_chunks = c.shape[1] // chunk_size
        num_heads = x.shape[2]
        batch_size = c.shape[0]
        num_total_tiles = num_chunks * num_heads * batch_size
        for tile_idx in range(start_idx, num_total_tiles, num_sms):
            head = tile_idx % num_heads
            idx = tile_idx // num_heads
            chunk_idx = idx % num_chunks
            batch = idx // num_chunks
            group = head // num_heads_per_group
            _compute_dd_dcb_dx_tile(
                batch=batch,
                head=head,
                group=group,
                chunk_idx=chunk_idx,
                x=x,
                b=b,
                c=c,
                d=d,
                dt=dt,
                decay_cumsum=decay_cumsum,
                dout=dout,
                dhiddens=dhiddens,
                dx=dx,
                ddt=ddt,
                dd=dd,
                ddecay_cumsum=ddecay_cumsum,
                dcb=dcb,
                chunk_size=chunk_size,
                BLOCK_STATE=BLOCK_STATE,
                BLOCK_HEAD=BLOCK_HEAD,
                state_size=state_size,
                head_size=head_size,
                LOAD_LAT=LOAD_LAT,
                STORE_LAT=STORE_LAT,
                DOUT_LAT=DOUT_LAT,
                DHID_LAT=DHID_LAT,
            )
    else:
        chunk_idx, head, batch = ct.bid(0), ct.bid(1), ct.bid(2)
        group = head // num_heads_per_group
        _compute_dd_dcb_dx_tile(
            batch=batch,
            head=head,
            group=group,
            chunk_idx=chunk_idx,
            x=x,
            b=b,
            c=c,
            d=d,
            dt=dt,
            decay_cumsum=decay_cumsum,
            dout=dout,
            dhiddens=dhiddens,
            dx=dx,
            ddt=ddt,
            dd=dd,
            ddecay_cumsum=ddecay_cumsum,
            dcb=dcb,
            chunk_size=chunk_size,
            BLOCK_STATE=BLOCK_STATE,
            BLOCK_HEAD=BLOCK_HEAD,
            state_size=state_size,
            head_size=head_size,
            LOAD_LAT=LOAD_LAT,
            STORE_LAT=STORE_LAT,
            DOUT_LAT=DOUT_LAT,
            DHID_LAT=DHID_LAT,
        )


@ct.kernel
def compute_da_db_dc_ddt_kernel(
    x: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    dt: torch.Tensor,
    decay_cumsum: torch.Tensor,
    hiddens: torch.Tensor,
    dout: torch.Tensor,
    dhiddens: torch.Tensor,
    dcb: torch.Tensor,
    ddecay_cumsum: torch.Tensor,
    dd_last_partials: torch.Tensor,
    da: torch.Tensor,
    db: torch.Tensor,
    dc: torch.Tensor,
    ddt_partial: torch.Tensor,
    ddt: torch.Tensor,
    chunk_size: ct.Constant[int],
    BLOCK_STATE: ct.Constant[int],
    BLOCK_HEAD: ct.Constant[int],
    num_heads_per_group: ct.Constant[int],
    state_size: ct.Constant[int],
    head_size: ct.Constant[int],
    ND_LEN: ct.Constant[int],
    LOAD_LAT: ct.Constant[int],
    STORE_LAT: ct.Constant[int],
    persistent: ct.Constant[bool],
    REDUCE_HEADS: ct.Constant[bool],
) -> None:
    if persistent:
        start_idx = ct.bid(0)
        num_sms = ct.num_blocks(0)
        num_chunks = c.shape[1] // chunk_size
        num_heads = x.shape[2]
        batch_size = c.shape[0]
        num_total_tiles = num_chunks * num_heads * batch_size
        for tile_idx in range(start_idx, num_total_tiles, num_sms):
            head = tile_idx % num_heads
            idx = tile_idx // num_heads
            chunk_idx = idx % num_chunks
            batch = idx // num_chunks
            group = head // num_heads_per_group
            _compute_da_db_dc_ddt_tile(
                batch=batch,
                head=head,
                group=group,
                chunk_idx=chunk_idx,
                x=x,
                a=a,
                b=b,
                c=c,
                dt=dt,
                decay_cumsum=decay_cumsum,
                hiddens=hiddens,
                dout=dout,
                dhiddens=dhiddens,
                dcb=dcb,
                ddecay_cumsum=ddecay_cumsum,
                dd_last_partials=dd_last_partials,
                da=da,
                db=db,
                dc=dc,
                ddt_partial=ddt_partial,
                ddt=ddt,
                chunk_size=chunk_size,
                BLOCK_STATE=BLOCK_STATE,
                BLOCK_HEAD=BLOCK_HEAD,
                state_size=state_size,
                head_size=head_size,
                ND_LEN=ND_LEN,
                LOAD_LAT=LOAD_LAT,
                STORE_LAT=STORE_LAT,
                REDUCE_HEADS=REDUCE_HEADS,
            )
    else:
        head, chunk_idx, batch = ct.bid(0), ct.bid(1), ct.bid(2)
        group = head // num_heads_per_group

        _compute_da_db_dc_ddt_tile(
            batch=batch,
            head=head,
            group=group,
            chunk_idx=chunk_idx,
            x=x,
            a=a,
            b=b,
            c=c,
            dt=dt,
            decay_cumsum=decay_cumsum,
            hiddens=hiddens,
            dout=dout,
            dhiddens=dhiddens,
            dcb=dcb,
            ddecay_cumsum=ddecay_cumsum,
            dd_last_partials=dd_last_partials,
            da=da,
            db=db,
            dc=dc,
            ddt_partial=ddt_partial,
            ddt=ddt,
            chunk_size=chunk_size,
            BLOCK_STATE=BLOCK_STATE,
            BLOCK_HEAD=BLOCK_HEAD,
            state_size=state_size,
            head_size=head_size,
            ND_LEN=ND_LEN,
            LOAD_LAT=LOAD_LAT,
            STORE_LAT=STORE_LAT,
            REDUCE_HEADS=REDUCE_HEADS,
        )


_chunk_cumsum_tune_cache = {}
_fwd_pass_state_tune_cache = {}
_compute_out_tune_cache = {}
_fwd_pass_state_dual_tune_cache = {}
_bwd_pass_state_tune_cache = {}
_compute_dd_dcb_dx_tune_cache = {}
_compute_da_db_dc_ddt_tune_cache = {}


def _validate_chunk_size(num_tokens, chunk_size):
    if chunk_size <= 0 or num_tokens % chunk_size != 0:
        raise ValueError("num_tokens must be divisible by a positive chunk_size")


def _tuning_key(*args):
    return tuple(
        (tuple(arg.shape), tuple(arg.stride()), arg.dtype, str(arg.device)) if torch.is_tensor(arg) else arg
        for arg in args
    )


def narrow_dtype(state_dtype, *consumer_dtypes):
    """Use the common consumer dtype, or retain the state dtype when consumers differ."""
    first = consumer_dtypes[0]
    return first if all(d == first for d in consumer_dtypes) else state_dtype


def _num_sms():
    return torch.cuda.get_device_properties("cuda").multi_processor_count


def _chunk_cumsum(a, dt, chunk_size=64):
    """dt:[b,t,h] -> (decay_bth [b,t,h], decay_bht [b,h,t], dt_bht [b,h,t])."""
    batch_size, num_tokens, num_heads = dt.shape
    num_chunks = ct.cdiv(num_tokens, chunk_size)
    out_bth = torch.empty_like(dt, dtype=torch.float32)
    out_bht = torch.empty(batch_size, num_heads, num_tokens, dtype=torch.float32, device=dt.device)
    dt_bht = torch.empty(batch_size, num_heads, num_tokens, dtype=dt.dtype, device=dt.device)

    def args_fn(cfg):
        return (a, dt, out_bth, out_bht, dt_bht, chunk_size, cfg.BLOCK_H)

    def grid_fn(cfg):
        return (num_chunks, ct.cdiv(num_heads, cfg.BLOCK_H), batch_size)

    stream = torch.cuda.current_stream()

    def hints_fn(cfg):
        return {"occupancy": cfg.occupancy}

    disabled = is_autotune_disabled()
    cache_key = _tuning_key(a, dt, chunk_size, disabled)
    if cache_key not in _chunk_cumsum_tune_cache:
        configs = list(_chunk_cumsum_autotune_configs(num_heads=num_heads))
        if not configs:
            raise ValueError("Unsupported Mamba shape: autotune search space is empty")
        if disabled:
            best_cfg = configs[0]
        else:
            result = exhaustive_search(configs, stream, grid_fn, chunk_cumsum_kernel, args_fn, hints_fn)
            best_cfg = result.best.config
        _chunk_cumsum_tune_cache[cache_key] = (best_cfg, chunk_cumsum_kernel.replace_hints(**hints_fn(best_cfg)))
    best_cfg, tuned_kernel = _chunk_cumsum_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))
    return out_bth, out_bht, dt_bht


def _fwd_pass_state(x, b, dt, dt_bht, decay_bth, decay_bht, init_state, chunk_size=64, state_dtype=torch.float32):
    """Return chunk-entry and final states using the supplied BTH and BHT layouts."""
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups = b.shape[2]
    state_size = b.shape[-1]
    num_heads_per_group = num_heads // num_groups
    num_chunks = num_tokens // chunk_size
    hiddens = x.new_empty(batch_size, num_heads, num_chunks, head_size, state_size, dtype=state_dtype)
    final_state = torch.empty_like(init_state)

    def args_fn(cfg):
        dt_k = dt_bht if cfg.BHT else dt
        dec_k = decay_bht if cfg.BHT else decay_bth
        return (
            x,
            b,
            dt_k,
            dec_k,
            init_state,
            hiddens,
            hiddens,
            final_state,
            chunk_size,
            cfg.BLOCK_STATE,
            cfg.BLOCK_HEAD,
            num_heads_per_group,
            cfg.LOAD_LAT,
            cfg.STORE_LAT,
            cfg.BDD_LAT,
            bool(cfg.SWAP),
            bool(cfg.BHT),
            False,
        )

    def grid_fn(cfg):
        return (head_size // cfg.BLOCK_HEAD, state_size // cfg.BLOCK_STATE, batch_size * num_heads)

    stream = torch.cuda.current_stream()

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    disabled = is_autotune_disabled()
    cache_key = _tuning_key(x, b, dt, dt_bht, decay_bth, decay_bht, init_state, chunk_size, state_dtype, disabled)
    if cache_key not in _fwd_pass_state_tune_cache:
        configs = list(
            _fwd_pass_state_autotune_configs(state_size=state_size, head_size=head_size, chunk_size=chunk_size)
        )
        if not configs:
            raise ValueError("Unsupported Mamba shape: autotune search space is empty")
        if disabled:
            best_cfg = configs[0]
        else:
            result = exhaustive_search(configs, stream, grid_fn, fwd_pass_state_kernel, args_fn, hints_fn)
            best_cfg = result.best.config
        _fwd_pass_state_tune_cache[cache_key] = (best_cfg, fwd_pass_state_kernel.replace_hints(**hints_fn(best_cfg)))
    best_cfg, tuned_kernel = _fwd_pass_state_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))
    return hiddens, final_state


def _compute_out(x, b, c, d, dt_bht, decay_bht, hiddens, chunk_size=64):
    """Compute outputs from BHT step sizes and decays."""
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups = b.shape[2]
    state_size = b.shape[-1]
    num_heads_per_group = num_heads // num_groups
    out = torch.empty_like(x)
    use_swap = chunk_size == 64
    kernel = compute_out_swap_kernel if use_swap else compute_out_kernel
    cfg_gen = _compute_out_swap_configs if use_swap else _compute_out_configs

    def args_fn(cfg):
        common = (
            x,
            b,
            c,
            d,
            dt_bht,
            decay_bht,
            hiddens,
            out,
            chunk_size,
            cfg.BLOCK_STATE,
            cfg.BLOCK_HEAD,
            num_heads_per_group,
            state_size,
        )
        if use_swap:
            return common + (cfg.LATENCY, cfg.H_LAT, cfg.persistent)
        return common + (cfg.H_LAT, cfg.persistent)

    def grid_fn(cfg):
        if cfg.persistent:
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            return (cfg.occupancy * num_sms, 1, 1)
        num_chunks = num_tokens // chunk_size
        num_head_tiles = head_size // cfg.BLOCK_HEAD
        return (num_chunks, num_head_tiles, batch_size * num_heads)

    stream = torch.cuda.current_stream()

    def hints_fn(cfg):
        return {"occupancy": cfg.occupancy}

    disabled = is_autotune_disabled()
    cache_key = _tuning_key(x, b, c, d, dt_bht, decay_bht, hiddens, chunk_size, disabled)
    if cache_key not in _compute_out_tune_cache:
        configs = list(cfg_gen(head_size=head_size, state_size=state_size))
        if not configs:
            raise ValueError("Unsupported Mamba shape: autotune search space is empty")
        if disabled:
            best_cfg = configs[0]
        else:
            result = exhaustive_search(configs, stream, grid_fn, kernel, args_fn, hints_fn)
            best_cfg = result.best.config
        _compute_out_tune_cache[cache_key] = (best_cfg, kernel.replace_hints(**hints_fn(best_cfg)))
    best_cfg, tuned_kernel = _compute_out_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))
    return out


def _fwd_pass_state_dual(
    x, b, dt, dt_bht, decay_bth, decay_bht, init_state, chunk_size=64, state_dtype=torch.float32, narrow=torch.bfloat16
):
    """Return wide and narrowed chunk-entry states and the final state."""
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups = b.shape[2]
    state_size = b.shape[-1]
    num_heads_per_group = num_heads // num_groups
    num_chunks = num_tokens // chunk_size
    dual = narrow != state_dtype

    hiddens = x.new_empty(batch_size, num_heads, num_chunks, head_size, state_size, dtype=state_dtype)
    hid_narrow = (
        x.new_empty(batch_size, num_heads, num_chunks, head_size, state_size, dtype=narrow) if dual else hiddens
    )
    final_state = torch.empty_like(init_state)

    def args_fn(cfg):
        dt_k = dt_bht if cfg.BHT else dt
        dec_k = decay_bht if cfg.BHT else decay_bth
        return (
            x,
            b,
            dt_k,
            dec_k,
            init_state,
            hiddens,
            hid_narrow,
            final_state,
            chunk_size,
            cfg.BLOCK_STATE,
            cfg.BLOCK_HEAD,
            num_heads_per_group,
            cfg.LOAD_LAT,
            cfg.STORE_LAT,
            cfg.BDD_LAT,
            cfg.SWAP,
            cfg.BHT,
            dual,
        )

    def grid_fn(cfg):
        return (head_size // cfg.BLOCK_HEAD, state_size // cfg.BLOCK_STATE, batch_size * num_heads)

    stream = torch.cuda.current_stream()

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    disabled = is_autotune_disabled()
    cache_key = _tuning_key(
        x, b, dt, dt_bht, decay_bth, decay_bht, init_state, chunk_size, state_dtype, narrow, disabled
    )
    if cache_key not in _fwd_pass_state_dual_tune_cache:
        configs = list(_fwd_pass_state_autotune_configs(state_size, head_size, chunk_size))
        if not configs:
            raise ValueError("Unsupported Mamba shape: autotune search space is empty")
        if disabled:
            best_cfg = configs[0]
        else:
            result = exhaustive_search(configs, stream, grid_fn, fwd_pass_state_kernel, args_fn, hints_fn)
            best_cfg = result.best.config
        _fwd_pass_state_dual_tune_cache[cache_key] = (
            best_cfg,
            fwd_pass_state_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _fwd_pass_state_dual_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))
    return hiddens, hid_narrow, final_state


def _bwd_pass_state(c, decay_cumsum, dout, dfinal_state, hiddens=None, chunk_size=64, narrow=torch.bfloat16):
    """Return state gradients and optional forward/backward state-product partials."""
    batch_size, num_tokens, num_heads, head_size = dout.shape
    state_size = c.shape[-1]
    num_heads_per_group = num_heads // c.shape[2]
    num_chunks = num_tokens // chunk_size
    dhiddens = torch.empty(batch_size, num_heads, num_chunks, head_size, state_size, dtype=narrow, device=c.device)
    dinit_state = torch.empty_like(dfinal_state)
    with_partials = hiddens is not None
    space = list(_bwd_pass_state_autotune_configs(state_size, head_size))

    def tiling(cfg):
        return head_size // cfg.BLOCK_HEAD, state_size // cfg.BLOCK_STATE

    dd_last_partials = {
        t: torch.empty(
            batch_size, num_heads, num_chunks, t[0], t[1], _PARTIALS_PER_TILE, dtype=torch.float32, device=c.device
        )
        for t in ({tiling(cfg) for cfg in space} if with_partials else ())
    }

    won = None

    def args_fn(cfg):
        nonlocal won
        won = tiling(cfg)
        return (
            c,
            decay_cumsum,
            dout,
            dfinal_state,
            dhiddens,
            hiddens if with_partials else dhiddens,
            dd_last_partials[won] if with_partials else dhiddens,
            dinit_state,
            chunk_size,
            cfg.BLOCK_STATE,
            cfg.BLOCK_HEAD,
            num_heads_per_group,
            cfg.LOAD_LAT,
            cfg.STORE_LAT,
            cfg.CD_LAT,
            cfg.HID_LAT,
            with_partials,
        )

    def grid_fn(cfg):
        return tiling(cfg) + (batch_size * num_heads,)

    stream = torch.cuda.current_stream()

    def hints_fn(cfg):
        return {"num_ctas": cfg.num_ctas, "occupancy": cfg.occupancy}

    disabled = is_autotune_disabled()
    cache_key = _tuning_key(c, decay_cumsum, dout, dfinal_state, hiddens, chunk_size, narrow, disabled)
    if cache_key not in _bwd_pass_state_tune_cache:
        configs = list(space)
        if not configs:
            raise ValueError("Unsupported Mamba shape: autotune search space is empty")
        if disabled:
            best_cfg = configs[0]
        else:
            result = exhaustive_search(configs, stream, grid_fn, bwd_pass_state_kernel, args_fn, hints_fn)
            best_cfg = result.best.config
        _bwd_pass_state_tune_cache[cache_key] = (best_cfg, bwd_pass_state_kernel.replace_hints(**hints_fn(best_cfg)))
    best_cfg, tuned_kernel = _bwd_pass_state_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))
    if not with_partials:
        return dhiddens, None, dinit_state, 0
    n_partials = won[0] * won[1] * _PARTIALS_PER_TILE
    return (
        dhiddens,
        dd_last_partials[won].view(batch_size, num_heads, num_chunks, n_partials),
        dinit_state,
        n_partials,
    )


def _compute_dd_dcb_dx(x, b, c, d, dt_bht, decay_bht, dout, dhiddens, chunk_size=64, dcb_dtype=None):
    """Return dx, partial ddt, dd, partial decay gradients, and transposed dcb."""
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups = b.shape[2]
    state_size = b.shape[-1]
    num_heads_per_group = num_heads // num_groups

    dx = torch.empty_like(x)
    dd = d.new_empty(batch_size, num_heads, num_tokens // chunk_size, dtype=torch.float32)
    ddt = dt_bht.new_empty(batch_size, num_tokens, num_heads)
    ddecay_cumsum = dt_bht.new_empty(batch_size, num_tokens, num_heads, dtype=torch.float32)

    assert b.dtype == c.dtype, f"b and c must have the same dtype, got {b.dtype}/{c.dtype}"
    dcb = c.new_empty(batch_size, num_tokens, num_heads, chunk_size, dtype=c.dtype if dcb_dtype is None else dcb_dtype)

    def args_fn(cfg):
        return (
            x,
            b,
            c,
            d,
            dt_bht,
            decay_bht,
            dout,
            dhiddens,
            dx,
            ddt,
            dd,
            ddecay_cumsum,
            dcb,
            chunk_size,
            cfg.BLOCK_STATE,
            cfg.BLOCK_HEAD,
            num_heads_per_group,
            state_size,
            head_size,
            cfg.LOAD_LAT,
            cfg.STORE_LAT,
            cfg.DOUT_LAT,
            cfg.DHID_LAT,
            cfg.persistent,
        )

    def grid_fn(cfg):
        if cfg.persistent:
            return (cfg.occupancy * _num_sms(), 1, 1)
        return (num_tokens // chunk_size, num_heads, batch_size)

    stream = torch.cuda.current_stream()

    def hints_fn(cfg):
        return {"occupancy": cfg.occupancy}

    disabled = is_autotune_disabled()
    cache_key = _tuning_key(x, b, c, d, dt_bht, decay_bht, dout, dhiddens, chunk_size, dcb_dtype, disabled)
    if cache_key not in _compute_dd_dcb_dx_tune_cache:
        configs = list(_compute_dd_dcb_dx_autotune_configs(state_size=state_size, head_size=head_size))
        if not configs:
            raise ValueError("Unsupported Mamba shape: autotune search space is empty")
        if disabled:
            best_cfg = configs[0]
        else:
            result = exhaustive_search(configs, stream, grid_fn, compute_dd_dcb_dx_kernel, args_fn, hints_fn)
            best_cfg = result.best.config
        _compute_dd_dcb_dx_tune_cache[cache_key] = (
            best_cfg,
            compute_dd_dcb_dx_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _compute_dd_dcb_dx_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))
    return (dx, ddt, dd.sum(dim=[0, 2]), ddecay_cumsum, dcb)


def _compute_da_db_dc_ddt(
    x,
    a,
    b,
    c,
    dt_bht,
    decay_bht,
    hiddens,
    dout,
    dhiddens,
    dcb,
    ddecay_cumsum,
    ddt,
    partials,
    n_partials,
    chunk_size=64,
    grouped=None,
):
    """Return da, db, dc and ddt; grouped=False keeps db/dc per head."""
    batch_size, num_tokens, num_heads, head_size = x.shape
    num_groups = b.shape[2]
    state_size = b.shape[-1]
    num_heads_per_group = num_heads // num_groups
    num_chunks = num_tokens // chunk_size
    grouped = (num_groups != num_heads) if grouped is None else grouped

    da = a.new_empty(batch_size, num_heads, num_chunks, dtype=torch.float32)
    if grouped:
        db = torch.empty(batch_size, num_tokens, num_groups, state_size, dtype=torch.float32, device=b.device)
        dc = torch.empty(batch_size, num_tokens, num_groups, state_size, dtype=torch.float32, device=c.device)
    else:
        db = b.new_empty(batch_size, num_tokens, num_heads, state_size)
        dc = c.new_empty(batch_size, num_tokens, num_heads, state_size)

    ddt_partial = ddt
    ddt = torch.empty_like(ddt_partial)

    def args_fn(cfg):
        if grouped:
            db.zero_()
            dc.zero_()
        return (
            x,
            a,
            b,
            c,
            dt_bht,
            decay_bht,
            hiddens,
            dout,
            dhiddens,
            dcb,
            ddecay_cumsum,
            partials,
            da,
            db,
            dc,
            ddt_partial,
            ddt,
            chunk_size,
            cfg.BLOCK_STATE,
            cfg.BLOCK_HEAD,
            num_heads_per_group,
            state_size,
            head_size,
            n_partials,
            cfg.LOAD_LAT,
            cfg.STORE_LAT,
            cfg.persistent,
            grouped,
        )

    def grid_fn(cfg):
        if cfg.persistent:
            return (cfg.occupancy * _num_sms(), 1, 1)
        return (num_heads, num_chunks, batch_size)

    stream = torch.cuda.current_stream()

    def hints_fn(cfg):
        return {"occupancy": cfg.occupancy}

    disabled = is_autotune_disabled()
    cache_key = _tuning_key(
        x,
        a,
        b,
        c,
        dt_bht,
        decay_bht,
        hiddens,
        dout,
        dhiddens,
        dcb,
        ddecay_cumsum,
        ddt_partial,
        partials,
        n_partials,
        chunk_size,
        grouped,
        disabled,
    )
    if cache_key not in _compute_da_db_dc_ddt_tune_cache:
        configs = list(_compute_da_db_dc_ddt_autotune_configs(head_size, state_size))
        if not configs:
            raise ValueError("Unsupported Mamba shape: autotune search space is empty")
        if disabled:
            best_cfg = configs[0]
        else:
            result = exhaustive_search(configs, stream, grid_fn, compute_da_db_dc_ddt_kernel, args_fn, hints_fn)
            best_cfg = result.best.config
        _compute_da_db_dc_ddt_tune_cache[cache_key] = (
            best_cfg,
            compute_da_db_dc_ddt_kernel.replace_hints(**hints_fn(best_cfg)),
        )
    best_cfg, tuned_kernel = _compute_da_db_dc_ddt_tune_cache[cache_key]
    ct.launch(stream, grid_fn(best_cfg), tuned_kernel, args_fn(best_cfg))
    if grouped:
        return da.sum(dim=[0, 2]), db.to(b.dtype), dc.to(c.dtype), ddt
    return da.sum(dim=[0, 2]), db, dc, ddt


@register_impl("mamba2_chunk_forward", backend="cutile")
def mamba2_chunk_forward(x, a, b, c, d, dt, init_state, chunk_size=64):
    """Return the output and final state using chunk-entry states in c.dtype."""
    _validate_chunk_size(x.shape[1], chunk_size)
    decay_bth, decay_bht, dt_bht = _chunk_cumsum(a, dt, chunk_size=chunk_size)
    hiddens, final_state = _fwd_pass_state(
        x=x,
        b=b,
        dt=dt,
        dt_bht=dt_bht,
        decay_bth=decay_bth,
        decay_bht=decay_bht,
        init_state=init_state,
        chunk_size=chunk_size,
        state_dtype=c.dtype,
    )
    out = _compute_out(x=x, b=b, c=c, d=d, dt_bht=dt_bht, decay_bht=decay_bht, hiddens=hiddens, chunk_size=chunk_size)
    return out, final_state


@register_impl("mamba2_chunk_backward", backend="cutile")
def mamba2_chunk_backward(x, a, b, c, d, dt, init_state, dout, dfinal_state, chunk_size=64, state_dtype=torch.float32):
    """Return dx, da, grouped db/dc, dd, ddt and dinit_state."""
    _validate_chunk_size(x.shape[1], chunk_size)
    decay_bth, decay_bht, dt_bht = _chunk_cumsum(a=a, dt=dt, chunk_size=chunk_size)

    hiddens, hiddens_narrow, _ = _fwd_pass_state_dual(
        x=x,
        b=b,
        dt=dt,
        dt_bht=dt_bht,
        decay_bth=decay_bth,
        decay_bht=decay_bht,
        init_state=init_state,
        chunk_size=chunk_size,
        state_dtype=state_dtype,
        narrow=narrow_dtype(state_dtype, dout.dtype),
    )

    dhiddens, partials, dinit_state, n_partials = _bwd_pass_state(
        c=c,
        decay_cumsum=decay_bth,
        dout=dout,
        dfinal_state=dfinal_state,
        hiddens=hiddens,
        chunk_size=chunk_size,
        narrow=narrow_dtype(state_dtype, b.dtype, x.dtype),
    )

    dx, ddt_partial, dd, ddecay_partial, dcb = _compute_dd_dcb_dx(
        x=x, b=b, c=c, d=d, dt_bht=dt_bht, decay_bht=decay_bht, dout=dout, dhiddens=dhiddens, chunk_size=chunk_size
    )

    da, db, dc, ddt = _compute_da_db_dc_ddt(
        x=x,
        a=a,
        b=b,
        c=c,
        dt_bht=dt_bht,
        decay_bht=decay_bht,
        hiddens=hiddens_narrow,
        dout=dout,
        dhiddens=dhiddens,
        dcb=dcb,
        ddecay_cumsum=ddecay_partial,
        ddt=ddt_partial,
        partials=partials,
        n_partials=n_partials,
        chunk_size=chunk_size,
    )
    return dx, da, db, dc, dd, ddt, dinit_state
