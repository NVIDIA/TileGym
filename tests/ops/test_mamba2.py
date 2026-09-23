# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
from dataclasses import dataclass
from typing import Any

import pytest
import torch

einops = pytest.importorskip("einops", reason="einops is required for mamba2 tests")
repeat = einops.repeat
rearrange = einops.rearrange

import tilegym

from .. import common

LOG2E = math.log2(math.e)

# ---------------------------------------------------------------------------
# Problem size configurations
# (batch_size, num_tokens, num_groups, num_heads, state_size, head_size,
#  chunk_size)
# ---------------------------------------------------------------------------

_DTYPES = [
    pytest.param(torch.bfloat16, id="bf16"),
]


_E2E_SHAPE_CONFIGS = [
    pytest.param(1, 64, 1, 4, 16, 32, 64, id="minimal"),
    pytest.param(2, 128, 1, 4, 16, 32, 64, id="batch2"),
    pytest.param(1, 64, 2, 4, 16, 32, 64, id="groups2"),
    pytest.param(2, 128, 2, 8, 32, 64, 64, id="standard"),
    pytest.param(1, 192, 1, 4, 16, 32, 64, id="T192"),
    pytest.param(2, 256, 2, 8, 64, 64, 64, id="large"),
    pytest.param(2, 64, 1, 8, 16, 64, 64, id="single-chunk"),
    pytest.param(1, 192, 1, 4, 64, 64, 64, id="three-chunks"),
    pytest.param(1, 256, 4, 4, 128, 64, 128, id="chunk128-ungrouped"),
]


@dataclass(frozen=True)
class Mamba2ProblemSpec:
    batch_size: int
    num_tokens: int
    num_groups: int
    num_heads: int
    state_size: int
    head_size: int
    chunk_size: int
    dtype: torch.dtype
    state_dtype: torch.dtype

    def get_sample_inputs(self, *, with_reference=True, with_backward=True) -> dict[str, Any]:
        """Generate all tensors any Mamba-2 kernel might need."""
        x = torch.randn(
            self.batch_size,
            self.num_tokens,
            self.num_heads,
            self.head_size,
            device="cuda",
            dtype=self.dtype,
        )

        a = -torch.empty(self.num_heads, dtype=torch.float32, device="cuda").uniform_(1, 16)
        b = torch.randn(
            self.batch_size,
            self.num_tokens,
            self.num_groups,
            self.state_size,
            device="cuda",
            dtype=self.dtype,
        )
        c = torch.randn(
            self.batch_size,
            self.num_tokens,
            self.num_groups,
            self.state_size,
            device="cuda",
            dtype=self.dtype,
        )

        d = torch.randn(self.num_heads, device="cuda")
        dt = torch.randn(
            self.batch_size,
            self.num_tokens,
            self.num_heads,
            device="cuda",
            dtype=self.dtype,
        )
        x = torch.nn.functional.silu(x)
        b = torch.nn.functional.silu(b)
        c = torch.nn.functional.silu(c)
        dt = torch.nn.functional.softplus(dt)

        init_state = self._init_hidden()
        if not with_reference and not with_backward:
            return dict(x=x, a=a, b=b, c=c, d=d, dt=dt, init_state=init_state, chunk_size=self.chunk_size)

        dfinal_state = self._init_hidden()
        dout = torch.randn(
            self.batch_size,
            self.num_tokens,
            self.num_heads,
            self.head_size,
            device="cuda",
            dtype=self.dtype,
        )

        if not with_reference:
            return dict(
                x=x,
                a=a,
                b=b,
                c=c,
                d=d,
                dt=dt,
                init_state=init_state,
                dout=dout,
                dfinal_state=dfinal_state,
                chunk_size=self.chunk_size,
            )

        decay_cumsum = self._compute_ref_decay_cumsum(a, dt, chunk_size=self.chunk_size)
        hiddens, final_state = self._compute_ref_pass_state(
            x=x,
            b=b,
            dt=dt,
            decay_cumsum=decay_cumsum,
            init_state=init_state,
            chunk_size=self.chunk_size,
        )
        out = self._compute_ref_out(
            x=x,
            b=b,
            c=c,
            d=d,
            dt=dt,
            decay_cumsum=decay_cumsum,
            hiddens=hiddens,
            chunk_size=self.chunk_size,
        )

        dhiddens, dinit_state = self._compute_ref_bwd_pass_state(
            c=c,
            decay_cumsum=decay_cumsum,
            dout=dout,
            dfinal_state=dfinal_state,
            chunk_size=self.chunk_size,
        )

        dx, dd, ddt_partial, ddecay_cumsum_partial, dcb = self._compute_ref_dd_dcb_dx(
            x=x,
            b=b,
            c=c,
            d=d,
            dt=dt,
            decay_cumsum=decay_cumsum,
            dout=dout,
            dhiddens=dhiddens,
            chunk_size=self.chunk_size,
        )

        da, db, dc, ddt = self._compute_ref_da_db_dc_ddt(
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
            ddecay_cumsum=ddecay_cumsum_partial,
            ddt=ddt_partial,
            chunk_size=self.chunk_size,
        )

        return {
            "x": x,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "dt": dt,
            "init_state": init_state,
            "chunk_size": self.chunk_size,
            "decay_cumsum": decay_cumsum * LOG2E,
            "decay_cumsum_loge": decay_cumsum,
            "hiddens": hiddens,
            "final_state": final_state,
            "out": out,
            "dout": dout,
            "dfinal_state": dfinal_state,
            "dhiddens": dhiddens,
            "ddt_partial": ddt_partial,
            "ddecay_cumsum_partial": ddecay_cumsum_partial,
            "dcb": dcb,
            "dx": dx,
            "da": da,
            "db": db,
            "dc": dc,
            "dd": dd,
            "ddt": ddt,
            "dinit_state": dinit_state,
        }

    def _init_hidden(self) -> torch.Tensor:
        """Initialize states through an SSM recurrence to keep numeric tests in distribution."""

        x = torch.randn(
            self.batch_size,
            self.chunk_size,
            self.num_heads,
            self.head_size,
            device="cuda",
            dtype=self.dtype,
        )
        a = -torch.empty(self.num_heads, dtype=torch.float32, device="cuda").uniform_(1, 16)
        b = torch.randn(
            self.batch_size,
            self.chunk_size,
            self.num_groups,
            self.state_size,
            device="cuda",
            dtype=self.dtype,
        )
        dt = torch.randn(
            self.batch_size,
            self.num_tokens,
            self.num_heads,
            device="cuda",
            dtype=self.dtype,
        )
        x = torch.nn.functional.silu(x)
        b = torch.nn.functional.silu(b)
        dt = torch.nn.functional.softplus(dt)
        decay = a * dt
        decay_cumsum = decay.cumsum(dim=1)
        init_state = torch.zeros(
            self.batch_size,
            self.num_heads,
            self.head_size,
            self.state_size,
            device="cuda",
            dtype=self.state_dtype,
        )
        _, last_hidden = self._compute_ref_pass_state(
            x=x,
            b=b,
            dt=dt,
            decay_cumsum=decay_cumsum,
            init_state=init_state,
            chunk_size=self.chunk_size,
        )
        return last_hidden

    @staticmethod
    def _compute_ref_decay_cumsum(a: torch.Tensor, dt: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        Inputs:
            a (num_heads), dtype is float32
            dt (batch_size, num_tokens, num_heads), dtype is bfloat16

        Output:
            decay_cumsum (batch_size, num_tokens, num_heads), dtype is float32
        """
        num_tokens = dt.shape[1]
        dt = dt.to(torch.float32)
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        padded_num_tokens = num_chunks * chunk_size
        decay_padded = a * dt
        if num_tokens < padded_num_tokens:
            decay_padded = torch.nn.functional.pad(decay_padded, (0, 0, 0, 0, 0, padded_num_tokens - num_tokens))
        decay_chunk = rearrange(decay_padded, "b (n c) h -> b n c h", c=chunk_size)
        decay_cumsum = decay_chunk.float().cumsum(dim=2)
        decay_cumsum = rearrange(decay_cumsum, "b n c h -> b (n c) h")
        decay_cumsum = decay_cumsum[:, :num_tokens].contiguous()
        return decay_cumsum

    @staticmethod
    def _compute_ref_pass_state(
        x: torch.Tensor,
        b: torch.Tensor,
        dt: torch.Tensor,
        decay_cumsum: torch.Tensor,
        init_state: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
            b (batch_size, num_tokens, num_groups, state_size), dtype is bfloat16
            dt (batch_size, num_tokens, num_heads), dtype is bfloat16
            decay_cumsum (batch_size, num_tokens, num_heads), dtype is float32
            init_state (batch_size, num_heads, head_size, state_size), dtype is float32

        Output:
            hiddens shape (batch_size, num_heads, num_chunks, head_size, state_size), dtype is bfloat16
            final_state shape (batch_size, num_heads, head_size, state_size), dtype is float32
        """

        x = x.to(torch.float32)
        b = b.to(torch.float32)
        dt = dt.to(torch.float32)

        _, num_tokens, num_groups, _ = b.shape
        num_heads = x.shape[2]
        b = repeat(b, "b n g k -> b n (g f) k", f=num_heads // num_groups).float()
        hid = init_state

        hiddens = []

        num_chunks = num_tokens // chunk_size
        for chunk_idx in range(num_chunks):
            hiddens.append(hid.clone())
            start_idx = chunk_idx * chunk_size
            x_chunk = x[:, start_idx : start_idx + chunk_size].transpose(1, 2)
            b_chunk = b[:, start_idx : start_idx + chunk_size].transpose(1, 2)
            dt_chunk = dt[:, start_idx : start_idx + chunk_size].transpose(1, 2)
            d_chunk = decay_cumsum[:, start_idx : start_idx + chunk_size].transpose(1, 2)
            g_last = d_chunk[:, :, -1]
            b_dt_d_chunk = b_chunk * (dt_chunk * torch.exp(g_last[..., None] - d_chunk))[..., None]
            hid = hid * torch.exp(g_last[..., None, None])
            hid = hid + x_chunk.transpose(-1, -2) @ b_dt_d_chunk.to(x_chunk.dtype)

        hiddens = torch.stack(hiddens, dim=2)
        return hiddens, hid

    @staticmethod
    def _compute_ref_out(
        x: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        dt: torch.Tensor,
        decay_cumsum: torch.Tensor,
        hiddens: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        """
        Inputs:
            x (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
            b (batch_size, num_tokens, num_groups, state_size), dtype is bfloat16
            c (batch_size, num_tokens, num_groups, state_size), dtype is bfloat16
            d (num_heads), dtype is float32
            dt (batch_size, num_tokens, num_heads), dtype is bfloat16
            decay_cumsum (batch_size, num_tokens, num_heads), dtype is float32
            hiddens (batch_size, num_heads, num_chunks, head_size, state_size), dtype is bfloat16

        Output:
            output shape (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
        """

        x = x.to(torch.float32)
        b = b.to(torch.float32)
        c = c.to(torch.float32)
        dt = dt.to(torch.float32)
        hiddens = hiddens.to(torch.float32)

        num_chunks = hiddens.shape[2]
        num_heads = x.shape[2]
        num_groups = c.shape[2]

        b = repeat(b, "b n g k -> b n (g f) k", f=num_heads // num_groups).float()
        c = repeat(c, "b n g k -> b n (g f) k", f=num_heads // num_groups).float()

        out = torch.zeros_like(x)

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, c.shape[1])

            x_chunk = x[:, start_idx:end_idx].transpose(1, 2)
            b_chunk = b[:, start_idx:end_idx].transpose(1, 2)
            c_chunk = c[:, start_idx:end_idx].transpose(1, 2)
            dt_chunk = dt[:, start_idx:end_idx].transpose(1, 2)
            d_chunk = decay_cumsum[:, start_idx:end_idx].transpose(1, 2)
            h_chunk = hiddens[:, :, chunk_idx]

            cb = c_chunk @ b_chunk.transpose(-1, -2)
            cb = cb * torch.tril(torch.exp(d_chunk[..., None] - d_chunk[..., None, :]))
            cb = torch.tril(cb)

            c_d = c_chunk * torch.exp(d_chunk[..., None])
            out_chunk = c_d @ h_chunk.transpose(-1, -2)
            out_chunk = out_chunk + cb @ (dt_chunk[..., None] * x_chunk)
            out_chunk = out_chunk + d[None, :, None, None] * x_chunk
            out[:, start_idx:end_idx] = out_chunk.transpose(1, 2)

        return out

    @staticmethod
    def _compute_ref_bwd_pass_state(
        c: torch.Tensor,
        decay_cumsum: torch.Tensor,
        dout: torch.Tensor,
        dfinal_state: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference implementation (pure PyTorch) of the backward pass state kernel.

        Inputs:
            c (batch_size, num_tokens, num_groups, state_size), dtype is bfloat16
            decay_cumsum (batch_size, num_tokens, num_heads), dtype is float32
            dout (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
            dfinal_state (batch_size, num_heads, head_size, state_size), dtype is float32

        Output:
            dhiddens shape (batch_size, num_heads, num_chunks, head_size, state_size),
                dtype is bfloat16
            dinit_state shape (batch_size, num_heads, head_size, state_size),
                dtype is float32
        """
        _, num_tokens, num_groups, _ = c.shape
        num_heads = dout.shape[2]
        dhid = dfinal_state

        c = repeat(c, "b n g k -> b n (g f) k", f=num_heads // num_groups).float()
        dout = dout.float()

        dhiddens = []
        num_chunks = num_tokens // chunk_size
        for chunk_idx in range(num_chunks - 1, -1, -1):
            dhiddens.append(dhid.clone())
            start_idx = chunk_idx * chunk_size
            c_chunk = c[:, start_idx : start_idx + chunk_size].transpose(1, 2)
            d_chunk = decay_cumsum[:, start_idx : start_idx + chunk_size].transpose(1, 2)
            g_last = d_chunk[..., -1]
            do_chunk = dout[:, start_idx : start_idx + chunk_size].transpose(1, 2)

            c_d = c_chunk * torch.exp(d_chunk[..., None])
            dhid = dhid * torch.exp(g_last[..., None, None]) + do_chunk.transpose(-1, -2) @ c_d.to(do_chunk.dtype)

        dhiddens = torch.stack(list(reversed(dhiddens)), dim=2)
        return dhiddens, dhid

    @staticmethod
    def _compute_ref_dd_dcb_dx(
        x: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
        dt: torch.Tensor,
        decay_cumsum: torch.Tensor,
        dout: torch.Tensor,
        dhiddens: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
            b (batch_size, num_tokens, num_groups, state_size), dtype is bfloat16
            c (batch_size, num_tokens, num_groups, state_size), dtype is bfloat16
            d (num_heads), dtype is float32
            dt (batch_size, num_tokens, num_heads), dtype is bfloat16
            decay_cumsum (batch_size, num_tokens, num_heads), dtype is float32
            dout (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
            dhiddens (batch_size, num_heads, num_chunks, head_size, state_size), dtype is float32

        Output:
            dx shape (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
            dd shape (num_heads), dtype is float32
            ddt (partial gradient) shape (batch_size, num_tokens, num_heads), dtype is bfloat16
            ddecay_cumsum (partial gradient) shape (batch_size, num_tokens, num_heads), dtype is float32
            dcb shape (batch_size, num_tokens, num_heads, chunk_size), dtype is float32
        """
        batch_size, num_tokens, num_heads, _ = x.shape
        num_groups = c.shape[2]

        x = x.to(torch.float32)
        b = repeat(b, "b n g k -> b n (g f) k", f=num_heads // num_groups).float()
        c = repeat(c, "b n g k -> b n (g f) k", f=num_heads // num_groups).float()
        dt = dt.to(torch.float32)
        dout = dout.to(torch.float32)

        dd = torch.zeros_like(d)
        dcb = torch.empty(
            batch_size,
            num_tokens,
            num_heads,
            chunk_size,
            device=x.device,
            dtype=torch.float32,
        )
        dx = torch.empty_like(x)
        ddt = torch.empty_like(dt)
        ddecay_cumsum = torch.empty_like(decay_cumsum)
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_tokens)
            actual_chunk_size = end_idx - start_idx

            x_chunk = x[:, start_idx:end_idx].transpose(1, 2)
            b_chunk = b[:, start_idx:end_idx].transpose(1, 2)
            c_chunk = c[:, start_idx:end_idx].transpose(1, 2)
            dt_chunk = dt[:, start_idx:end_idx].transpose(1, 2)
            d_chunk = decay_cumsum[:, start_idx:end_idx].transpose(1, 2)
            do_chunk = dout[:, start_idx:end_idx].transpose(1, 2)
            dhid_chunk = dhiddens[:, :, chunk_idx]

            cb = (c_chunk @ b_chunk.transpose(-1, -2)) * torch.exp(d_chunk[..., None] - d_chunk[..., None, :])
            cb = torch.tril(cb)

            dx_chunk = do_chunk * d[None, :, None, None]
            dd += (do_chunk * x_chunk).sum(dim=[0, 2, 3])

            xt = dt_chunk[..., None] * x_chunk
            dcb_chunk = do_chunk @ xt.transpose(-1, -2)
            dxt = cb.transpose(-1, -2) @ do_chunk

            dx_chunk += dxt * dt_chunk[..., None]
            ddt_chunk = (dxt * x_chunk).sum(dim=-1)

            dcb_chunk = torch.tril(dcb_chunk)
            dcb_cb = dcb_chunk * cb
            dd_chunk = dcb_cb.sum(dim=-1) - dcb_cb.sum(dim=-2)
            dcb_chunk = dcb_chunk * torch.exp(d_chunk[..., None] - d_chunk[..., None, :])
            # Mask again because the upper-triangle exponential can overflow.

            dcb_chunk = torch.tril(dcb_chunk)

            g_last = d_chunk[..., -1]
            b_dt_d = b_chunk * dt_chunk[..., None] * (torch.exp(g_last[..., None] - d_chunk))[..., None]
            dx_chunk += b_dt_d @ dhid_chunk.transpose(-1, -2)

            dx[:, start_idx:end_idx] = dx_chunk.transpose(1, 2)
            ddt[:, start_idx:end_idx] = ddt_chunk.transpose(1, 2)
            ddecay_cumsum[:, start_idx:end_idx] = dd_chunk.transpose(1, 2)
            dcb[:, start_idx:end_idx, :, :actual_chunk_size] = dcb_chunk.transpose(1, 2)

        return dx, dd, ddt, ddecay_cumsum, dcb

    @staticmethod
    def _compute_ref_da_db_dc_ddt(
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
        ddt: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs:
            x (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
            a (num_heads), dtype is float32
            b (batch_size, num_tokens, num_groups, state_size), dtype is bfloat16
            c (batch_size, num_tokens, num_groups, state_size), dtype is bfloat16
            dt (batch_size, num_tokens, num_heads), dtype is bfloat16
            decay_cumsum (batch_size, num_tokens, num_heads), dtype is float32
            hiddens (batch_size, num_heads, num_chunks, head_size, state_size), dtype is float32
            dout (batch_size, num_tokens, num_heads, head_size), dtype is bfloat16
            dhiddens (batch_size, num_heads, num_chunks, head_size, state_size), dtype is float32
            dcb shape (batch_size, num_tokens, num_heads, chunk_size), dtype is float32
            ddecay_cumsum (batch_size, num_tokens, num_heads), dtype is float32
            ddt (batch_size, num_tokens, num_heads), dtype is bfloat16

        Output:
            da (num_heads), dtype is float32
            db (unreduced) (batch_size, num_tokens, num_heads, state_size), dtype is bfloat16
            dc (unreduced) (batch_size, num_tokens, num_heads, state_size), dtype is bfloat16
            ddt shape (batch_size, num_tokens, num_heads), dtype is bfloat16
        """
        _, num_tokens, num_heads, _ = x.shape
        num_groups = c.shape[2]

        x = x.to(torch.float32)
        b = repeat(b, "b n g k -> b n (g f) k", f=num_heads // num_groups).float()
        c = repeat(c, "b n g k -> b n (g f) k", f=num_heads // num_groups).float()
        dt = dt.to(torch.float32)
        dout = dout.to(torch.float32)

        da = torch.zeros_like(a)
        db = torch.empty_like(b)
        dc = torch.empty_like(c)
        ddt_out = torch.empty_like(dt)

        num_chunks = (num_tokens + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_tokens)
            actual_chunk_size = end_idx - start_idx

            x_chunk = x[:, start_idx:end_idx].transpose(1, 2)
            b_chunk = b[:, start_idx:end_idx].transpose(1, 2)
            c_chunk = c[:, start_idx:end_idx].transpose(1, 2)
            dt_chunk = dt[:, start_idx:end_idx].transpose(1, 2)
            d_chunk = decay_cumsum[:, start_idx:end_idx].transpose(1, 2)
            hid_chunk = hiddens[:, :, chunk_idx]
            do_chunk = dout[:, start_idx:end_idx].transpose(1, 2)
            dhid_chunk = dhiddens[:, :, chunk_idx]
            dcb_chunk = dcb[:, start_idx:end_idx, :, :actual_chunk_size].transpose(1, 2)
            ddt_chunk = ddt[:, start_idx:end_idx].transpose(1, 2).clone()
            dd_chunk = ddecay_cumsum[:, start_idx:end_idx].transpose(1, 2).clone()

            dcg = do_chunk @ hid_chunk
            dc_chunk = dcg * torch.exp(d_chunk[..., None])
            dd_chunk += (dcg * c_chunk * torch.exp(d_chunk[..., None])).sum(dim=-1)

            dc_chunk += dcb_chunk @ b_chunk
            db_chunk = dcb_chunk.transpose(-1, -2) @ c_chunk

            g_last = d_chunk[..., -1]
            dg_last = (dhid_chunk * hid_chunk * torch.exp(g_last[..., None, None])).sum(dim=[-1, -2])
            db_dt_d = x_chunk @ dhid_chunk

            db_chunk += db_dt_d * dt_chunk[..., None] * (torch.exp(g_last[..., None] - d_chunk))[..., None]
            ddt_chunk += (db_dt_d * b_chunk * (torch.exp(g_last[..., None] - d_chunk))[..., None]).sum(dim=-1)
            dd_chunk += -(
                db_dt_d * b_chunk * dt_chunk[..., None] * (torch.exp(g_last[..., None] - d_chunk))[..., None]
            ).sum(dim=-1)
            dg_last += (
                db_dt_d * b_chunk * dt_chunk[..., None] * (torch.exp(g_last[..., None] - d_chunk))[..., None]
            ).sum(dim=[-1, -2])

            dd_chunk[..., -1] += dg_last

            dd_chunk = dd_chunk.sum(dim=-1, keepdim=True) - dd_chunk.cumsum(dim=-1) + dd_chunk
            ddt_chunk += dd_chunk * a[None, :, None]
            da += (dt_chunk * dd_chunk).sum(dim=[0, 2])

            db[:, start_idx:end_idx] = db_chunk.transpose(1, 2)
            dc[:, start_idx:end_idx] = dc_chunk.transpose(1, 2)
            ddt_out[:, start_idx:end_idx] = ddt_chunk.transpose(1, 2)

        return da, db, dc, ddt_out


@pytest.fixture
def _ieee_reference_math(monkeypatch):
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "0")
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.usefixtures("_ieee_reference_math")
class Test_Mamba2_ChunkForward(common.PyTestCase):
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize(
        "batch_size,num_tokens,num_groups,num_heads,state_size,head_size,chunk_size", _E2E_SHAPE_CONFIGS
    )
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op(
        self, batch_size, num_tokens, num_groups, num_heads, state_size, head_size, chunk_size, dtype, arch, backend
    ):
        self.setUp()
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")
        torch.manual_seed(42)
        spec = Mamba2ProblemSpec(
            batch_size, num_tokens, num_groups, num_heads, state_size, head_size, chunk_size, dtype, torch.float32
        )
        with torch.no_grad():
            r = spec.get_sample_inputs()
            result = tilegym.ops.mamba2_chunk_forward(
                r["x"],
                r["a"],
                r["b"],
                r["c"],
                r["d"],
                r["dt"],
                r["init_state"],
                chunk_size=chunk_size,
            )
            out, final = result
            torch.testing.assert_close(out.float(), r["out"].float(), atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(final.float(), r["final_state"].float(), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("T", [2**i for i in range(10, 19)])
    @pytest.mark.parametrize("S", [64, 128, 256, 512])
    @pytest.mark.parametrize("framework", ["cutile"])
    def test_perf(self, T, S, arch, framework, record_property):
        self.setUp()
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")
        torch.manual_seed(0)
        chunk_size = 128
        spec = Mamba2ProblemSpec(4, T, 1, 8, S, 64, chunk_size, torch.bfloat16, torch.float32)
        with torch.no_grad():
            r = spec.get_sample_inputs(with_reference=False, with_backward=False)
            framework_fn = lambda: tilegym.ops.mamba2_chunk_forward(
                r["x"],
                r["a"],
                r["b"],
                r["c"],
                r["d"],
                r["dt"],
                r["init_state"],
                chunk_size=chunk_size,
            )
            result = common.benchmark_framework(framework, framework_fn, use_cupti=True)
            record_property("benchmark", result)


@pytest.mark.usefixtures("_ieee_reference_math")
class Test_Mamba2_ChunkBackward(common.PyTestCase):
    @pytest.mark.timeout(1200)
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize(
        "batch_size,num_tokens,num_groups,num_heads,state_size,head_size,chunk_size", _E2E_SHAPE_CONFIGS
    )
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op(
        self, batch_size, num_tokens, num_groups, num_heads, state_size, head_size, chunk_size, dtype, arch, backend
    ):
        self.setUp()
        try:
            tilegym.set_backend(backend)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")
        torch.manual_seed(42)
        spec = Mamba2ProblemSpec(
            batch_size, num_tokens, num_groups, num_heads, state_size, head_size, chunk_size, dtype, torch.float32
        )
        with torch.no_grad():
            r = spec.get_sample_inputs()
            result = tilegym.ops.mamba2_chunk_backward(
                r["x"],
                r["a"],
                r["b"],
                r["c"],
                r["d"],
                r["dt"],
                r["init_state"],
                r["dout"],
                r["dfinal_state"],
                chunk_size=chunk_size,
            )
            expected = (r["dx"], r["da"], r["dd"], r["ddt"], r["dinit_state"])
            tolerances = (4e-2, 1e-2, 1e-2, 8e-2, 1e-2)
            for actual, ref, tolerance in zip((*result[:2], *result[4:]), expected, tolerances):
                torch.testing.assert_close(actual.float(), ref.float(), atol=tolerance, rtol=tolerance)
            for actual, per_head, atol in zip(result[2:4], (r["db"], r["dc"]), (5e-2, 4e-2)):
                per_head = per_head.reshape(batch_size, num_tokens, num_groups, num_heads // num_groups, state_size)
                expected = per_head.float().sum(3)
                torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=4e-2)

    @pytest.mark.timeout(1200)
    @pytest.mark.parametrize("T", [2**i for i in range(10, 19)])
    @pytest.mark.parametrize("S", [64, 128, 256, 512])
    @pytest.mark.parametrize("framework", ["cutile"])
    def test_perf(self, T, S, arch, framework, record_property):
        self.setUp()
        try:
            tilegym.set_backend(framework)
        except Exception as exc:
            pytest.skip(f"Backend is not supported: {exc}")
        torch.manual_seed(0)
        chunk_size = 128 if S == 512 else 64
        spec = Mamba2ProblemSpec(4, T, 1, 8, S, 64, chunk_size, torch.bfloat16, torch.float32)
        with torch.no_grad():
            r = spec.get_sample_inputs(with_reference=False)
            framework_fn = lambda: tilegym.ops.mamba2_chunk_backward(
                r["x"],
                r["a"],
                r["b"],
                r["c"],
                r["d"],
                r["dt"],
                r["init_state"],
                r["dout"],
                r["dfinal_state"],
                chunk_size=chunk_size,
            )
            result = common.benchmark_framework(framework, framework_fn, use_cupti=True)
            record_property("benchmark", result)
