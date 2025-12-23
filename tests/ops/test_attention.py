# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

from tilegym.backend import set_backend
from tilegym.ops import fmha_interface

from .. import common


def get_data(
    *shape,
    dtype,
    device,
    mean=0.0,
    normal_std=1.0,
):
    if dtype == torch.float8_e5m2:
        out = torch.empty(*shape, dtype=torch.float16, device=device).normal_(mean, normal_std).to(dtype)
    else:
        out = torch.empty(*shape, dtype=dtype, device=device).normal_(mean, normal_std)
    return out


class Test_FMHA(common.PyTestCase):
    @staticmethod
    def reference(q, k, v, scaling=None, attention_mask=None, is_causal=False):
        if q.dtype == torch.float8_e5m2:
            ref = torch.nn.functional.scaled_dot_product_attention(
                q.float(),
                k.float(),
                v.float(),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=is_causal,
                scale=scaling,
            )
            return ref.to(q.dtype)

        ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal, scale=scaling
        )
        return ref

    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "batch_size, num_heads, seq_len, head_dim, is_causal, dtype",
        [
            (1, 1, 9, 128, False, torch.bfloat16),
            (1, 32, 2047, 128, True, torch.float16),
            (2, 32, 4095, 128, True, torch.bfloat16),
            (2, 32, 4095, 128, True, torch.float8_e5m2),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        is_causal,
        dtype,
        backend,
        arch,
    ):
        if arch in ["sm120", "sm121"]:
            pytest.skip("Skip on sm120, sm121: limited shared memory size.")
        if arch in ["sm80"] and dtype == torch.float8_e5m2:
            pytest.skip("Skip on sm80: float8_e5m2 is not supported")
        try:
            set_backend(backend)
        except Exception as e:
            pytest.skip(f"Backend is not supported: {e}")
        self.setUp()
        # Create random input tensors
        q = get_data(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        k = get_data(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device="cuda",
            dtype=dtype,
        )
        v = get_data(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device="cuda",
            dtype=dtype,
        )

        # Calculate scaling factor
        sm_scale = 1.0 / math.sqrt(head_dim)
        if dtype == torch.float8_e5m2:
            atol = 3
            rtol = 0
        else:
            atol = 5e-2
            rtol = 1e-2
        try:
            self.assertCorrectness(
                fmha_interface,
                self.reference,
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "scaling": sm_scale,
                    "is_causal": is_causal,
                },
                atol=atol,
                rtol=rtol,
                check_stride=False,
            )
        except NotImplementedError as e:
            pytest.skip(f"Skip due to not-implemented: {e}")
