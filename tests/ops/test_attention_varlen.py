# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math
import os
from typing import Optional
from typing import Tuple

import pytest
import torch

import tilegym
from tilegym.ops import fmha_varlen

from .. import common

_backends = ["cutile"]
_perf_backends = _backends + ["pytorch"]


_perf_dtypes = [torch.float16]


USE_FULL_CONFIG = os.environ.get("USE_FULL_CONFIG", "0") == "1"


def get_configs_full():
    return [
        (
            batch,
            num_heads,
            num_head_groups,
            max_q_seq_len,
            max_kv_seq_len,
            head_dim,
            is_causal,
            dtype,
            varlen,
        )
        for batch in [1, 10, 30]
        for num_heads in [4, 32]
        for num_head_groups in [1, 4]
        for max_q_seq_len in [1, 128, 257]
        for max_kv_seq_len in [256, 1024, 2048]
        for head_dim in [128]
        for is_causal in [True, False]
        for dtype in [torch.bfloat16]
        for varlen in [True, False]
    ]


def get_configs_test():
    return [
        (
            batch,
            num_heads,
            num_head_groups,
            max_q_seq_len,
            max_kv_seq_len,
            head_dim,
            is_causal,
            dtype,
            varlen,
        )
        for batch in [4]
        for num_heads in [32]
        for num_head_groups in [1]
        for max_q_seq_len in [128, 257]
        for max_kv_seq_len in [496, 2064]
        for head_dim in [128]
        for is_causal in [True, False]
        for dtype in [torch.bfloat16]
        for varlen in [
            True,
            False,
        ]
    ]


def _process_attention_output(output, q_lens=None, ref_mask=None, layout="bnsd", dropout_mask=None):
    """
    Process attention output, set invalid positions to 0 to skip comparison

    Args:
        output: Attention output tensor, shape (batch, num_heads, seq_len, head_dim)
        q_lens: Query sequence actual length, shape (batch)
        ref_mask: Reference mask, shape (batch, num_heads, seq_len, kv_seq_len)

    Returns:
        Processed output tensor
    """
    dtype = output.dtype
    if dtype == torch.float8_e5m2:
        output = output.float()

    processed_output = output.clone()

    if q_lens is not None:
        s_dim = layout.find("s")
        output_s_broadcast_shape = [1] * 4
        output_s_broadcast_shape[2] = output.shape[s_dim]  # bnsd
        q_len_mask = (
            torch.arange(output.shape[s_dim], device=output.device).view(output_s_broadcast_shape)
            >= q_lens[:, None, None, None]
        )
        q_len_mask = torch.einsum(f"bnsd->{layout}", q_len_mask).contiguous()
        processed_output.masked_fill_(q_len_mask, 0)

    # Combine ref_mask and dropout_mask with OR logic
    combined_mask = None
    if ref_mask is not None:
        combined_mask = ref_mask
    if dropout_mask is not None:
        # Convert dropout_mask: True(keep) -> False(mask), False(drop) -> True(mask)
        dropout_mask_inverted = ~dropout_mask
        if combined_mask is not None:
            # OR logic: if either mask is True, the position is masked
            combined_mask = torch.logical_or(combined_mask, dropout_mask_inverted)
        else:
            combined_mask = dropout_mask_inverted

    if combined_mask is not None:
        # Check if entire rows are masked (all positions in the last dimension are True)
        fully_masked_rows = combined_mask.all(dim=-1).unsqueeze(-1)  # (batch, num_heads, q_seq_len)

        fully_masked_expanded = (
            torch.einsum(f"bnsd->{layout}", fully_masked_rows).contiguous().expand_as(processed_output)
        )
        processed_output.masked_fill_(fully_masked_expanded, 0)
    return processed_output.to(dtype)


def _get_data(*shape, dtype, device, mean=0.1, normal_std=0.2):
    if dtype == torch.float8_e5m2:
        return torch.empty(*shape, dtype=torch.float16, device=device).normal_(mean, normal_std).to(dtype)
    return torch.empty(*shape, dtype=dtype, device=device).normal_(mean, normal_std)


def _get_qkv(
    batch,
    q_heads,
    kv_heads,
    q_seq_len,
    kv_seq_len,
    head_dim,
    device,
    dtype,
    mean=0.1,
    normal_std=0.2,
):
    q = _get_data(
        batch,
        q_heads,
        q_seq_len,
        head_dim,
        dtype=dtype,
        device=device,
        mean=mean,
        normal_std=normal_std,
    )
    k = _get_data(
        batch,
        kv_heads,
        kv_seq_len,
        head_dim,
        dtype=dtype,
        device=device,
        mean=mean,
        normal_std=normal_std,
    )
    v = _get_data(
        batch,
        kv_heads,
        kv_seq_len,
        head_dim,
        dtype=dtype,
        device=device,
        mean=mean,
        normal_std=normal_std,
    )
    return q, k, v


def _generate_causal_mask(
    batch: int,
    num_heads: int,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    device: torch.device,
    q_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate causal mask

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        max_q_seq_len: Maximum query sequence length
        max_kv_seq_len: Maximum key-value sequence length
        device: Device
        q_lens: Query sequence actual length
        kv_lens: Key-value sequence actual length

    Returns:
        Causal mask with shape (batch, num_heads, max_q_seq_len, max_kv_seq_len)
    """
    batch_masks = []
    for b in range(batch):
        q_len = q_lens[b].item() if q_lens is not None else max_q_seq_len
        kv_len = kv_lens[b].item() if kv_lens is not None else max_kv_seq_len
        causal_offset = kv_len - q_len
        assert kv_len >= q_len, f"kv_len ({kv_len}) should >= q_len ({q_len}) for batch {b}"

        # Generate causal mask for current batch
        batch_mask = (
            torch.triu(
                torch.ones((max_q_seq_len, max_kv_seq_len), device=device),
                diagonal=1 + causal_offset,
            )
            .expand(num_heads, max_q_seq_len, max_kv_seq_len)
            .contiguous()
            .bool()
        )
        batch_masks.append(batch_mask)
    return torch.stack(batch_masks, dim=0)


def _get_mask(
    causal: bool = False,
    max_q_seq_len: int = 0,
    max_kv_seq_len: int = 0,
    batch: int = 0,
    num_heads: int = 0,
    device: torch.device = torch.device("cuda"),
    q_lens: Optional[torch.Tensor] = None,
    kv_lens: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """
    Generate the causal mask for the reference implementation.

    Returns None when causal is False: the kernel applies causal masking
    itself, so only the reference needs an explicit mask.

    Returns:
        Optional[torch.Tensor]: Mask for the reference implementation, or None.
    """
    ref_mask = None
    # Generate causal mask
    if causal is not False:
        ref_mask = _generate_causal_mask(batch, num_heads, max_q_seq_len, max_kv_seq_len, device, q_lens, kv_lens)

    ref_mask = ref_mask.clone() if ref_mask is not None else None

    return ref_mask


# for correctness test
def _get_varlen_rand(
    varlen: bool,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    batch: int,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.int32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    q_lens = torch.randint(max_q_seq_len // 2, max_q_seq_len, (batch,), device=device, dtype=dtype) if varlen else None
    if varlen:
        kv_lens = torch.empty((batch,), device=device, dtype=dtype)
        for i in range(batch):
            kv_lens[i] = torch.randint(q_lens[i].item(), max_kv_seq_len, (1,), device=device, dtype=dtype)
    else:
        kv_lens = None
    return q_lens, kv_lens


# for prof
def _get_varlen_fix(
    varlen: bool,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    batch: int,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.int32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    q_lens = torch.full((batch,), max_q_seq_len, device=device, dtype=dtype) if varlen else None
    kv_lens = torch.full((batch,), max_kv_seq_len, device=device, dtype=dtype) if varlen else None
    return q_lens, kv_lens


class Test_FMHA_varlen(common.PyTestCase):
    @staticmethod
    def reference(q, k, v, scaling=None, attention_mask=None, is_causal=False):
        dtype = q.dtype
        if dtype == torch.float8_e5m2:
            q = q.float()
            k = k.float()
            v = v.float()

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal, scale=scaling
        )
        return ref.to(dtype)

    @staticmethod
    def einsum_reference(
        q,
        k,
        v,
        scaling,
        q_lens=None,
        kv_lens=None,
        mask=None,
        bias=None,
        dropout=0.0,
        layout="bnsd",
        dropout_mask=None,
    ):
        dtype = q.dtype
        if dtype == torch.float8_e5m2:
            q = q.float()
            k = k.float()
            v = v.float()
            if bias is not None:
                bias = bias.float()

        q_layout = layout.replace("s", "i")
        k_layout = layout.replace("s", "j")
        inner_layout = "bnij"
        assert inner_layout[2:] == "ij"
        head_dim = layout.find("n")
        num_heads_q = q.shape[head_dim]
        num_heads_kv = k.shape[head_dim]
        if num_heads_q == num_heads_kv or num_heads_kv == 1:
            pass
        else:
            assert num_heads_q % num_heads_kv == 0
            num_head_groups = int(num_heads_q / num_heads_kv)
            k = torch.repeat_interleave(k, num_head_groups, dim=head_dim)
            v = torch.repeat_interleave(v, num_head_groups, dim=head_dim)

        p = torch.einsum(f"{q_layout},{k_layout}->{inner_layout}", q, k)
        p = p * scaling
        if bias is not None:
            p = p + bias

        if mask is not None:
            p.masked_fill_(mask[:, :, :, :], torch.finfo(p.dtype).min)

        if q_lens is not None:
            batch_dim = inner_layout.find("b")
            q_len_shape = 4 * [1]
            q_len_shape[batch_dim] = -1

            q_seq_dim = inner_layout.find("i")
            q_seq_idx_shape = 4 * [1]
            q_seq_idx_shape[q_seq_dim] = -1

            q_seq_idx = torch.arange(p.shape[q_seq_dim], device=p.device)
            q_len_mask = q_lens.view(q_len_shape) <= q_seq_idx.view(q_seq_idx_shape)
            p.masked_fill_(q_len_mask.expand_as(p), -float("inf"))

        if kv_lens is not None:
            batch_dim = inner_layout.find("b")
            kv_len_shape = 4 * [1]
            kv_len_shape[batch_dim] = -1

            kv_seq_dim = inner_layout.find("j")
            kv_seq_idx_shape = 4 * [1]
            kv_seq_idx_shape[kv_seq_dim] = -1

            kv_seq_idx = torch.arange(p.shape[kv_seq_dim], device=p.device)
            kv_len_mask = kv_lens.view(kv_len_shape) <= kv_seq_idx.view(kv_seq_idx_shape)
            p.masked_fill_(kv_len_mask.expand_as(p), -float("inf"))

        p = torch.softmax(p, dim=inner_layout.find("j"), dtype=torch.float32).to(v.dtype)

        if q_lens is not None:
            p = p.masked_fill(q_len_mask.expand_as(p), 0)
        if dropout > 0.0:
            p = torch.where(dropout_mask, p, 0.0) * float(1.0 / (1.0 - dropout))
        v_layout = layout.replace("s", "j")
        ref_out = torch.einsum(f"{inner_layout},{v_layout}->{q_layout}", p, v)
        return ref_out.to(dtype)

    @pytest.mark.parametrize(
        "batch, num_heads, num_head_groups, max_q_seq_len, max_kv_seq_len, head_dim, is_causal, dtype, varlen",
        get_configs_full() if USE_FULL_CONFIG else get_configs_test(),
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        batch,
        num_heads,
        num_head_groups,
        max_q_seq_len,
        max_kv_seq_len,
        head_dim,
        is_causal,
        dtype,
        arch,
        varlen,
        backend: str,
    ):
        if arch in ["sm120", "sm121"]:
            pytest.skip("Skip on sm120, sm121: limited shared memory size.")

        # Set the backend based on backend parameter
        tilegym.set_backend(backend)

        if max_q_seq_len > max_kv_seq_len:
            pytest.skip("max_q_seq_len should <= max_kv_seq_len")
        if num_heads % num_head_groups != 0:
            pytest.skip("num_heads should be divisible by num_head_groups")

        # Create random input tensors
        self.setUp()
        device = torch.device("cuda")

        kv_heads = num_head_groups
        q, k, v = _get_qkv(batch, num_heads, kv_heads, max_q_seq_len, max_kv_seq_len, head_dim, device, dtype)
        q_lens, kv_lens = _get_varlen_rand(varlen, max_q_seq_len, max_kv_seq_len, batch, device)
        ref_mask = _get_mask(
            causal=is_causal,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            batch=batch,
            num_heads=num_heads,
            device=device,
            q_lens=q_lens,
            kv_lens=kv_lens,
        )

        # Calculate scaling factor
        sm_scale = 1.0 / math.sqrt(head_dim)
        self.assertCorrectness(
            fmha_varlen,
            self.einsum_reference,
            {
                "q": q,
                "k": k,
                "v": v,
                "q_lens": q_lens,
                "kv_lens": kv_lens,
                "scaling": sm_scale,
            },
            extra_ref_kwargs={
                "mask": ref_mask,
            },
            extra_test_kwargs={
                "is_causal": is_causal,
            },
            atol=1e-1,
            rtol=1e-1,
            check_stride=False,
            output_processor=lambda ind, output, fn_kwargs, extra_test_kwargs, extra_ref_kwargs: (
                output if ind != 0 else _process_attention_output(output, fn_kwargs["q_lens"], extra_ref_kwargs["mask"])
            ),
        )

    @pytest.mark.parametrize(
        "batch,heads,q_seq_len,kv_seq_len,head_dim,dtype",
        [
            (4, 32, q_seq_len, kv_seq_len, 128, dtype)
            for dtype in _perf_dtypes
            for q_seq_len in [2**9, 2**12]
            for kv_seq_len in [2**13]
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("is_causal", [True])
    @pytest.mark.parametrize("varlen", [True, False])
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf_varlen(
        self,
        batch,
        heads,
        q_seq_len,
        kv_seq_len,
        head_dim,
        dtype,
        is_causal,
        backend: str,
        varlen,
        record_property,
    ):
        if not torch.cuda.is_available():
            pytest.skip("CUDA support required")
        if torch.cuda.get_device_capability() in [(12, 0), (12, 1)] and q_seq_len == 2**12 and kv_seq_len >= 2**13:
            pytest.skip("Skip OOM on sm120/sm121: q_seq=4096 + kv_seq>=8192 exceeds VRAM")

        self.setUp()
        device = torch.device("cuda")
        q, k, v = _get_qkv(batch, heads, heads, q_seq_len, kv_seq_len, head_dim, device, dtype)
        q_lens, kv_lens = _get_varlen_fix(varlen, q_seq_len, kv_seq_len, batch, device)

        ref_mask = _get_mask(
            causal=is_causal,
            max_q_seq_len=q_seq_len,
            max_kv_seq_len=kv_seq_len,
            batch=batch,
            num_heads=heads,
            device=device,
            q_lens=q_lens,
            kv_lens=kv_lens,
        )

        sm_scale = 1.0 / math.sqrt(head_dim)

        if backend in _backends:
            tilegym.set_backend(backend)
            backend_fn = lambda: fmha_varlen(
                q=q, k=k, v=v, q_lens=q_lens, kv_lens=kv_lens, scaling=sm_scale, is_causal=is_causal
            )
        elif backend == "pytorch":
            backend_fn = lambda: self.einsum_reference(
                q=q, k=k, v=v, q_lens=q_lens, kv_lens=kv_lens, scaling=sm_scale, mask=ref_mask
            )
        else:
            pytest.skip(f"Backend {backend} not supported")

        if backend != "pytorch":
            atol = 1e-1
            rtol = 1e-1
            self.assertCorrectness(
                backend_fn,
                lambda: self.einsum_reference(
                    q=q, k=k, v=v, q_lens=q_lens, kv_lens=kv_lens, scaling=sm_scale, mask=ref_mask
                ),
                kwargs={},
                atol=atol,
                rtol=rtol,
                check_stride=False,
                output_processor=lambda ind, output, fn_kwargs, extra_test_kwargs, extra_ref_kwargs: (
                    output if ind != 0 else _process_attention_output(output, q_lens, ref_mask)
                ),
            )

        result = common.benchmark_framework(backend, backend_fn, min_rep=50)
        record_property("benchmark", result)

        # Explicit cleanup to prevent OOM
        del q, k, v, backend_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
