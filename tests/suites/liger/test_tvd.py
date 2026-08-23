# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tests import common
from tilegym.suites.liger.ops import tvd


class _TorchTVDReference:
    """PyTorch float32 reference for TVD."""

    def __init__(self, reduction="batchmean", ignore_index=-100):
        self.reduction = reduction
        self.ignore_index = ignore_index

    def __call__(self, p, q, shift_labels=None):
        p_f = p.float()
        q_f = q.float()
        tvd_val = torch.abs(p_f - q_f) / 2.0
        n_non_ignore = p_f.size(0)

        if shift_labels is not None:
            tvd_val = torch.where(
                shift_labels.unsqueeze(1) != self.ignore_index,
                tvd_val,
                torch.zeros_like(tvd_val),
            )
            n_non_ignore = (shift_labels != self.ignore_index).sum().item()
            if n_non_ignore == 0:
                return torch.tensor(0.0, device=p.device)

        if self.reduction == "mean":
            return torch.sum(tvd_val) / (n_non_ignore * p_f.size(1))
        elif self.reduction == "sum":
            return torch.sum(tvd_val)
        elif self.reduction == "none":
            return tvd_val.to(p.dtype)
        else:  # batchmean
            return torch.sum(tvd_val) / n_non_ignore


class Test_Liger_TVD(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize(
        "B, T, V",
        [
            # Shapes from Liger test/transformers/test_tvd.py
            (1, 4096, 32000),
            (41, 401, 1271),
            (3, 423, 32000),
        ],
    )
    @pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean", "none"])
    @pytest.mark.parametrize(
        "dtype, atol, rtol",
        [
            (torch.float32, 1e-5, 1e-5),
            (torch.bfloat16, 1e-3, 1e-3),
        ],
    )
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, B, T, V, reduction, dtype, atol, rtol, backend, monkeypatch):
        """Test TVD forward correctness."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        torch.manual_seed(0)

        BT = B * T
        p = torch.randn(BT, V, device=device, dtype=dtype)
        q = torch.randn(BT, V, device=device).softmax(dim=-1).to(dtype)

        ref_fn = _TorchTVDReference(reduction=reduction)
        out_ref = ref_fn(p, q)

        out_test = tvd(p, q, reduction=reduction)

        if reduction == "none":
            assert torch.allclose(out_test.float(), out_ref.float(), atol=atol, rtol=rtol), (
                f"Forward mismatch: max_diff={((out_test.float() - out_ref.float()).abs().max()).item():.6f}"
            )
        else:
            assert torch.allclose(out_test.float(), out_ref.float(), atol=atol, rtol=rtol), (
                f"Forward mismatch: test={out_test.item():.6f} ref={out_ref.item():.6f}"
            )

    @pytest.mark.parametrize(
        "B, T, V",
        [
            (1, 4096, 32000),
            (41, 401, 1271),
            (3, 423, 32000),
        ],
    )
    @pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean"])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_backward(self, B, T, V, reduction, backend, monkeypatch):
        """Test TVD backward pass (gradient w.r.t. p)."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        torch.manual_seed(0)
        BT = B * T

        p_data = torch.randn(BT, V, device=device, dtype=torch.float32)
        q_data = torch.randn(BT, V, device=device).softmax(dim=-1)

        # Reference
        p_ref = p_data.clone().requires_grad_(True)
        ref_fn = _TorchTVDReference(reduction=reduction)
        out_ref = ref_fn(p_ref, q_data)
        out_ref.backward()

        # Test
        p_test = p_data.clone().requires_grad_(True)
        out_test = tvd(p_test, q_data, reduction=reduction)
        out_test.backward()

        assert p_test.grad is not None
        assert torch.allclose(p_test.grad.float(), p_ref.grad.float(), atol=1e-5, rtol=1e-5), (
            f"Backward mismatch: max_diff={((p_test.grad - p_ref.grad).abs().max()).item():.6f}"
        )

    @pytest.mark.parametrize(
        "B, T, V",
        [
            (1, 4096, 32000),
            (41, 401, 1271),
            (3, 423, 32000),
        ],
    )
    @pytest.mark.parametrize("reduction", ["batchmean", "sum", "mean", "none"])
    @pytest.mark.parametrize("ignore_index", [-100, 0, 1])
    @pytest.mark.parametrize("backend", _backends)
    def test_op_with_ignore_index(self, B, T, V, reduction, ignore_index, backend, monkeypatch):
        """Test TVD with shift_labels and ignore_index."""
        self.setUp()
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")

        device = torch.device("cuda")
        torch.manual_seed(0)
        BT = B * T

        p = torch.randn(BT, V, device=device, dtype=torch.float32)
        q = torch.randn(BT, V, device=device).softmax(dim=-1)

        labels = torch.randint(0, V, (BT,), device=device, dtype=torch.long)
        # Assign some positions to ignore_index
        n_ignore = max(1, BT // 4)
        idx = torch.randperm(BT)[:n_ignore]
        labels[idx] = ignore_index

        ref_fn = _TorchTVDReference(reduction=reduction, ignore_index=ignore_index)
        out_ref = ref_fn(p, q, shift_labels=labels)

        out_test = tvd(p, q, shift_labels=labels, reduction=reduction, ignore_index=ignore_index)

        atol, rtol = 1e-5, 1e-5
        if reduction == "none":
            assert torch.allclose(out_test.float(), out_ref.float(), atol=atol, rtol=rtol), (
                f"With-ignore forward mismatch: max_diff="
                f"{((out_test.float() - out_ref.float()).abs().max()).item():.6f}"
            )
        else:
            assert torch.allclose(out_test.float(), out_ref.float(), atol=atol, rtol=rtol), (
                f"With-ignore forward mismatch: test={out_test.item():.6f} ref={out_ref.item():.6f}"
            )
