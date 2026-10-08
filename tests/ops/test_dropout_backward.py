# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

import tilegym

from .. import common


def reference_mask(shape, seed, p, device):
    """Evaluate the documented signed-int32 hash independently using int64 tensors."""
    size = torch.Size(shape).numel()
    offsets = torch.arange(size, dtype=torch.int64)

    def int32(value):
        return (value + 2**31) % 2**32 - 2**31

    mixed_seed = int32(int(seed) * 2654435761)
    value = int32(offsets * 1103515245 + mixed_seed)
    value = int32(value ^ (value >> 16))
    value = int32(value ^ int32(value << 8))
    value = int32(value ^ (value >> 4))
    random = (value & 0x7FFFFFFF).float() / 2147483647.0
    return (random > p).reshape(shape).to(device)


class Test_DropoutBackward(common.PyTestCase):
    _backends = ["cutile"]

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape,seed,p",
        [
            ((), 0, 0.5),
            ((1,), 11, 0.2),
            ((1023,), 99, 0.5),
            ((1025,), -7, 0.8),
            ((17, 63), 2**63 - 1, 0.2),
            ((2, 17, 31), 11, 0.5),
            ((65537,), 99, 0.4),
            ((0,), 99, 0.5),
        ],
    )
    @pytest.mark.parametrize("inplace", [False, True])
    def test_op_backward(self, shape, seed, p, dtype, inplace, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        self.setUp()
        leaf = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        # Include genuine input zeros, which must not be confused with dropped lanes.
        with torch.no_grad():
            leaf.reshape(-1)[::3] = 0
        x = leaf.clone() if inplace else leaf
        original = x.detach().clone()
        dy = torch.randn_like(x)
        original_dy = dy.clone()
        mask = reference_mask(shape, seed, p, x.device)
        y = tilegym.ops.dropout(x, seed, p=p, inplace=inplace)
        expected_y = torch.where(mask, original.float() / (1 - p), 0).to(dtype)
        expected_dx = torch.where(mask, dy.float() / (1 - p), 0).to(dtype)
        torch.testing.assert_close(y, expected_y, rtol=1e-2, atol=1e-3)
        if inplace:
            assert y is x
        y.backward(dy)
        torch.testing.assert_close(leaf.grad, expected_dx, rtol=1e-2, atol=1e-3)
        torch.testing.assert_close(dy, original_dy, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("training,p", [(False, 0.5), (False, 1.0), (True, 0.0), (True, 1.0)])
    @pytest.mark.parametrize("inplace", [False, True])
    def test_op_boundaries(self, training, p, inplace, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        self.setUp()
        leaf = torch.randn((3, 7), device="cuda", requires_grad=True)
        x = leaf.clone() if inplace else leaf
        original = x.detach().clone()
        y = tilegym.ops.dropout(x, 0, p=p, training=training, inplace=inplace)
        active = training and p == 1.0
        torch.testing.assert_close(y, torch.zeros_like(x) if active else original, rtol=0, atol=0)
        y.sum().backward()
        torch.testing.assert_close(leaf.grad, torch.zeros_like(x) if active else torch.ones_like(x), rtol=0, atol=0)
        if not training or inplace or p == 0:
            assert y is x

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("gradient_layout", ["transpose", "expand"])
    def test_op_strided_gradient(self, gradient_layout, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        self.setUp()
        x = torch.randn((17, 63), device="cuda", requires_grad=True)
        dy = (
            torch.randn((63, 17), device="cuda").t()
            if gradient_layout == "transpose"
            else torch.randn((), device="cuda").expand_as(x)
        )
        assert not dy.is_contiguous()
        original_dy = dy.clone()
        y = tilegym.ops.dropout(x, 99, p=0.5)
        y.backward(dy)
        mask = reference_mask(x.shape, 99, 0.5, x.device)
        torch.testing.assert_close(x.grad, torch.where(mask, original_dy * 2, 0), rtol=0, atol=0)
        torch.testing.assert_close(dy, original_dy, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_higher_order_gradient(self, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        self.setUp()
        x = torch.randn(1025, device="cuda", requires_grad=True)
        dy = torch.randn_like(x, requires_grad=True)
        y = tilegym.ops.dropout(x, 11, p=0.5)
        dx = torch.autograd.grad(y, x, dy, create_graph=True)[0]
        ddy = torch.autograd.grad(dx, dy, torch.ones_like(dx))[0]
        mask = reference_mask(x.shape, 11, 0.5, x.device)
        torch.testing.assert_close(ddy, mask.float() * 2, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_repeated_backward(self, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        self.setUp()
        x = torch.ones(1025, device="cuda", requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda tensor: saved.append(tensor) or tensor, lambda t: t):
            y = tilegym.ops.dropout(x, 11, p=0.5)
        assert saved == []  # Backward requires no saved input, output, or mask.
        # An intervening launch with another seed must not affect this context.
        tilegym.ops.dropout(x, 99, p=0.5)
        mask = reference_mask(x.shape, 11, 0.5, x.device)
        for _ in range(2):
            dy = torch.randn_like(x)
            dx = torch.autograd.grad(y, x, dy, retain_graph=True)[0]
            torch.testing.assert_close(dx, torch.where(mask, dy * 2, 0), rtol=0, atol=0)

    @pytest.mark.parametrize("backend", _backends)
    def test_op_inplace_nonleaf_view(self, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        leaf = torch.ones(34, device="cuda", requires_grad=True)
        x = (leaf * 3)[:17]
        y = tilegym.ops.dropout(x, 11, inplace=True)
        assert y is x
        y.sum().backward()
        expected = torch.zeros_like(leaf)
        expected[:17] = reference_mask(x.shape, 11, 0.5, x.device).float() * 6
        torch.testing.assert_close(leaf.grad, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("view", [False, True])
    def test_op_inplace_leaf(self, view, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        x = torch.ones(17, device="cuda", requires_grad=True)
        if view:
            x = x.view(17)
        original = x.detach().clone()
        with pytest.raises(RuntimeError, match="leaf"):
            tilegym.ops.dropout(x, 11, inplace=True)
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("view_kind", ["split", "no_grad"])
    def test_op_inplace_forbidden_view(self, view_kind, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        base = torch.ones(34, device="cuda", requires_grad=True) * 3
        if view_kind == "split":
            x = base.split(17)[0]
        else:
            with torch.no_grad():
                x = base.view(34)
        original = base.detach().clone()
        with pytest.raises(RuntimeError, match="view"):
            tilegym.ops.dropout(x, 11, inplace=True)
        torch.testing.assert_close(base, original, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("view", [False, True])
    def test_op_inplace_no_grad(self, view, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        x = torch.ones(17, device="cuda", requires_grad=True)
        if view:
            x = x.view(17)
        version = x._version
        with torch.no_grad():
            y = tilegym.ops.dropout(x, 11, inplace=True)
        assert y is x
        assert x._version == version + 1
        mask = reference_mask(x.shape, 11, 0.5, x.device)
        torch.testing.assert_close(x, mask.float() * 2, rtol=0, atol=0)

    @pytest.mark.parametrize("backend", _backends)
    @pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
    def test_op_invalid_probability(self, p, backend, arch):
        if not tilegym.is_backend_available(backend):
            pytest.skip(f"Backend {backend} is not available")
        tilegym.set_backend(backend)
        with pytest.raises(ValueError, match="probability"):
            tilegym.ops.dropout(torch.ones(1, device="cuda"), 11, p=p)
