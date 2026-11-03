# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch
import torch.nn.functional as F

import tilegym
from tests import common
from tilegym.suites.liger.ops import multi_token_attention


def _reference_multi_token_attention(scores, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """
    PyTorch float32 reference for multi-token attention.

    1. Apply causal -inf mask (future positions → -inf)
    2. Softmax
    3. Conv2d
    4. Apply causal zero mask (future positions → 0)
    """
    scores_f = scores.float()
    L = scores_f.shape[-1]
    # Causal mask: upper triangular → -inf
    mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1)
    # Expand mask to match scores shape (*, L, L)
    for _ in range(scores_f.dim() - 2):
        mask = mask.unsqueeze(0)
    mask = mask.expand_as(scores_f)
    scores_masked = scores_f.masked_fill(mask, -1e9)
    probs = torch.softmax(scores_masked, dim=-1)

    out_conv = F.conv2d(
        probs,
        weight.float(),
        bias.float() if bias is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )

    # Zero out future positions
    zero_mask = torch.tril(torch.ones(L, L, device=scores.device, dtype=torch.float32))
    for _ in range(out_conv.dim() - 2):
        zero_mask = zero_mask.unsqueeze(0)
    zero_mask = zero_mask.expand_as(out_conv)
    out = out_conv * zero_mask
    return out.to(scores.dtype)


class Test_Liger_MultiTokenAttention(common.PyTestCase):
    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]

    @pytest.mark.parametrize(
        "batch_channels_L, dtype",
        [
            ((2, 1, 8), torch.float32),
            ((1, 2, 16), torch.float32),
            ((2, 1, 8), torch.float16),
            ((2, 1, 8), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, batch_channels_L, dtype, backend, monkeypatch):
        """Test multi-token attention forward with 1x1 conv weight."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        batch, channels, L = batch_channels_L

        # Use identity-like 1×1 conv weight
        weight = torch.ones(channels, channels, 1, 1, dtype=dtype, device=device)
        scores = torch.randn(batch, channels, L, L, dtype=dtype, device=device)

        def fw():
            return multi_token_attention(scores.clone(), weight)

        def ref():
            return _reference_multi_token_attention(scores, weight)

        self.assertCorrectness(fw, ref, kwargs={}, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "batch_channels_L, dtype",
        [
            ((2, 1, 8), torch.float32),
            ((1, 1, 16), torch.float32),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, batch_channels_L, dtype, backend, monkeypatch):
        """Test backward pass."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        batch, channels, L = batch_channels_L

        weight = torch.ones(channels, channels, 1, 1, dtype=dtype, device=device, requires_grad=True)
        scores = torch.randn(batch, channels, L, L, dtype=dtype, device=device, requires_grad=True)

        out = multi_token_attention(scores, weight)
        out.sum().backward()

        assert scores.grad is not None
        assert weight.grad is not None

    @pytest.mark.parametrize(
        "batch, channels, L, groups, dtype",
        [
            (2, 1, 8, 1, torch.float32),  # baseline: CH=1, groups=1 (mm path)
            (2, 2, 8, 1, torch.float32),  # CH=2, groups=1: mm path generalises to C > 1
            (2, 2, 8, 2, torch.float32),  # CH=2, groups=2 depthwise: exercises cuDNN fallback
            (2, 2, 8, 2, torch.bfloat16),  # bf16, depthwise: fallback + bf16 precision
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward_conv_params(self, batch, channels, L, groups, dtype, backend, monkeypatch):
        """Backward gradient correctness for varied channel/group configurations.

        Specifically exercises:
        - CH > 1 with groups=1: mm-based conv backward must generalise to C_in > 1
        - groups > 1 (depthwise conv): mm path is incorrect; cuDNN fallback must be used
        """
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        torch.manual_seed(42)

        # weight shape for grouped conv: (C_out, C_in // groups, kH, kW)
        weight_shape = (channels, channels // groups, 1, 1)
        s = torch.randn(batch, channels, L, L, dtype=dtype, device=device)
        w = torch.randn(*weight_shape, dtype=dtype, device=device)

        # fp32 reference
        s_ref = s.float().detach().requires_grad_(True)
        w_ref = w.float().detach().requires_grad_(True)
        _reference_multi_token_attention(s_ref, w_ref, groups=groups).sum().backward()

        # tilegym implementation
        s_out = s.detach().requires_grad_(True)
        w_out = w.detach().requires_grad_(True)
        multi_token_attention(s_out, w_out, groups=groups).sum().backward()

        # fp32 is tight; bf16 runs natively with inherent precision loss vs fp32 ref
        atol = 1e-4 if dtype == torch.float32 else 0.1
        rtol = 1e-3 if dtype == torch.float32 else 0.1

        assert torch.allclose(s_out.grad.float(), s_ref.grad, atol=atol, rtol=rtol), (
            f"scores.grad mismatch (dtype={dtype}, groups={groups}, "
            f"max_err={(s_out.grad.float() - s_ref.grad).abs().max().item():.5f})"
        )
        assert torch.allclose(w_out.grad.float(), w_ref.grad, atol=atol, rtol=rtol), (
            f"weight.grad mismatch (dtype={dtype}, groups={groups}, "
            f"max_err={(w_out.grad.float() - w_ref.grad).abs().max().item():.5f})"
        )

    @pytest.mark.parametrize(
        "batch_channels_L, dtype",
        [
            ((2, 1, 8), torch.float32),
            ((2, 1, 8), torch.float16),
            ((2, 1, 8), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op_with_bias(self, batch_channels_L, dtype, backend, monkeypatch):
        """Test multi-token attention forward with bias."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        batch, channels, L = batch_channels_L

        weight = torch.ones(channels, channels, 1, 1, dtype=dtype, device=device)
        bias = torch.zeros(channels, dtype=dtype, device=device)
        scores = torch.randn(batch, channels, L, L, dtype=dtype, device=device)

        def fw():
            return multi_token_attention(scores.clone(), weight, bias=bias)

        def ref():
            return _reference_multi_token_attention(scores, weight, bias=bias)

        self.assertCorrectness(fw, ref, kwargs={}, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize(
        "shape, dtype",
        [
            ((2, 4, 32, 32), torch.bfloat16),
            ((2, 4, 64, 64), torch.bfloat16),
            ((2, 4, 128, 128), torch.bfloat16),
            ((2, 4, 256, 256), torch.bfloat16),
            ((2, 4, 512, 512), torch.bfloat16),
            ((2, 4, 1024, 1024), torch.bfloat16),
            ((2, 4, 2048, 2048), torch.bfloat16),
            ((2, 4, 4096, 4096), torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(self, shape, dtype, backend, record_property, monkeypatch):
        self.setUp()
        device = torch.device("cuda")
        batch, c_in, L, _ = shape

        weight = torch.randn(c_in, c_in, 3, 3, dtype=dtype, device=device)
        scores = torch.randn(batch, c_in, L, L, dtype=dtype, device=device, requires_grad=True)

        if backend in self._backends:
            if not tilegym.is_backend_available(backend):
                pytest.skip(f"Backend {backend} not available")
            tilegym.set_backend(backend)
            framework_fn = lambda: multi_token_attention(scores, weight, groups=1, padding=1)
        elif backend == "pytorch":
            framework_fn = lambda: _reference_multi_token_attention(scores, weight, groups=1, padding=1)
        else:
            pytest.skip(f"Backend {backend} not supported")

        if backend != "pytorch":
            self.assertCorrectness(
                framework_fn,
                lambda: _reference_multi_token_attention(scores, weight, groups=1, padding=1),
                kwargs={},
                atol=1e-2,
                rtol=1e-2,
            )

        result = common.benchmark_framework(backend, framework_fn)
        record_property("benchmark", result)

        del scores, weight, framework_fn
        torch.cuda.empty_cache()
        gc.collect()
