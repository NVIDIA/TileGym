# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import math

import pytest
import torch

import tilegym
from tests.common import PyTestCase
from tests.common import benchmark_framework
from tilegym.ops import fft


class Test_FFT(PyTestCase):
    _backends = ["cutile"]
    _perf_backends = _backends + ["pytorch"]
    _atoms = ["tma"]

    @staticmethod
    def cutile_impl(x, decomp, W0, W1, W2, T0, T1, atom, BS, occupancy):
        """Implementation using the cuTile FFT kernel"""
        tilegym.set_backend("cutile")
        return fft(x, decomp, W0, W1, W2, T0, T1, atom, BS, occupancy)

    @staticmethod
    def torch_fft(x, decomp, W0, W1, W2, T0, T1, atom, BS, occupancy):
        """Reference: torch.fft.fft computed in complex64, cast back to the input dtype"""
        return torch.fft.fft(x.to(torch.complex64), dim=-1).to(x.dtype)

    @staticmethod
    def make_twiddles(decomp, precision):
        """Generate twiddle factors for FFT computation"""
        F0, F1, F2 = decomp
        N = F0 * F1 * F2
        F1F2 = F1 * F2

        def twiddles(rows: int, cols: int, factor: int):
            (I, J) = torch.meshgrid(torch.arange(rows, device="cuda"), torch.arange(cols, device="cuda"), indexing="ij")
            W = torch.exp(-2 * math.pi * 1j * (I * J) / factor)
            return torch.view_as_real(W)

        # Generate twiddle factors for each dimension
        W0_ri = twiddles(F0, F0, F0).to(precision)
        W1_ri = twiddles(F1, F1, F1).to(precision)
        W2_ri = twiddles(F2, F2, F2).to(precision)
        # Generate twiddle factors for dimension transitions
        T0_ri = twiddles(F0, F1F2, N).to(precision)
        # Because of the data repetition, we can use a smaller twiddle factor
        T1_ri = twiddles(F1, F2, F1F2).to(precision)
        return (W0_ri, W1_ri, W2_ri, T0_ri, T1_ri)

    @pytest.mark.parametrize(
        "batch_size,decomp,dtype",
        [
            (2, (16, 16, 16), torch.complex32),
            (64, (16, 16, 16), torch.complex32),
            (128, (8, 8, 16), torch.complex32),
            (16 * 1024, (16, 16, 16), torch.complex32),
        ],
    )
    @pytest.mark.parametrize("BS", [1, 2, 4, 8])
    @pytest.mark.parametrize("occupancy", [1, 2, 3, 4])
    @pytest.mark.parametrize("atom", _atoms)
    def test_op(self, batch_size, decomp, dtype, atom, BS, occupancy, arch, request):
        """Test FFT functional correctness against torch.fft.fft"""
        self.setUp()
        if not tilegym.backend.is_backend_available("cutile"):
            pytest.skip("cuTile backend is not available")
        N = math.prod(decomp)
        x = torch.rand((batch_size, N), dtype=dtype, device="cuda")

        # Determine precision for twiddle factors
        precision = torch.float32 if dtype == torch.complex64 else torch.float16
        weights = self.make_twiddles(decomp, precision)
        fft_args = {
            "x": x,
            "decomp": decomp,
            "W0": weights[0],
            "W1": weights[1],
            "W2": weights[2],
            "T0": weights[3],
            "T1": weights[4],
            "atom": atom,
            "BS": BS,
            "occupancy": occupancy,
        }

        # fp16 MMA accumulation over N up to 4096 needs a loose tolerance against the complex64 reference
        self.assertCorrectness(self.cutile_impl, self.torch_fft, fft_args, rtol=5e-2, atol=5e-1)

    @pytest.mark.parametrize(
        "batch_size,decomp,dtype",
        [
            (16 * 1024, (16, 16, 16), torch.complex32),
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("atom", _atoms)
    @pytest.mark.parametrize("BS", [1])
    @pytest.mark.parametrize("occupancy", [4])
    @pytest.mark.parametrize("framework", _perf_backends)
    def test_perf(
        self,
        batch_size,
        decomp,
        dtype,
        framework,
        atom,
        BS,
        occupancy,
        record_property,
    ):
        """Benchmark the FFT op on each backend"""
        self.setUp()
        N = math.prod(decomp)
        x = torch.rand((batch_size, N), dtype=dtype, device="cuda")

        # Determine precision for twiddle factors
        precision = torch.float32 if dtype == torch.complex64 else torch.float16
        weights = self.make_twiddles(decomp, precision)

        # Prepare arguments
        fft_args = {
            "x": x,
            "decomp": decomp,
            "W0": weights[0],
            "W1": weights[1],
            "W2": weights[2],
            "T0": weights[3],
            "T1": weights[4],
        }

        if framework in self._backends:
            if tilegym.is_backend_available(framework):
                tilegym.set_backend(framework)
            else:
                pytest.skip(f"Backend {framework} is not available")

        if framework == "pytorch":
            framework_fn = lambda: torch.fft.fft(x, dim=-1)
        elif framework in self._backends:
            framework_fn = lambda: tilegym.ops.fft(**fft_args, atom=atom, BS=BS, occupancy=occupancy)
        else:
            pytest.skip(f"Framework {framework} not supported")

        res = benchmark_framework(framework, framework_fn)

        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del x, weights, fft_args, framework_fn
        torch.cuda.empty_cache()
        import gc

        gc.collect()
