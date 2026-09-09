# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

import gc

import pytest
import torch

import tilegym
from tilegym.backend import is_backend_available

from .. import common

# Base matmul input metadata: (m, n, k, offset_a, offset_b, transpose_a, transpose_b, dtype)
_MATMUL_BASE_INPUTS = [
    (1024, 1024, 1024, 0, 0, False, False, torch.bfloat16),
    (1024, 1024, 1023, 0, 0, False, False, torch.bfloat16),
    (8192, 8192, 8192, 0, 0, False, False, torch.bfloat16),
    (8, 8, 8, 0, 0, False, False, torch.bfloat16),
    (3072, 6144, 2720, 0, 0, False, False, torch.bfloat16),
]


def _build_matmul_test_op_params(backends):
    """(backend, use_tma, static_persistent, m, n, k, offset_a, offset_b, transpose_a, transpose_b, dtype) params."""
    params = [
        (backend, use_tma, static_persistent, *inp)
        for backend in backends
        for use_tma in (True, False)
        for static_persistent in (True, False)
        for inp in _MATMUL_BASE_INPUTS
        # cutile only exposes the TMA matmul kernel; its use_tma=False path is unused.
        if not (backend == "cutile" and not use_tma)
    ]
    return params


class Test_Matmul(common.PyTestCase):
    @staticmethod
    def reference(a, b, trans_a=False, trans_b=False):
        if trans_a:
            a = a.t()
        if trans_b:
            b = b.t()
        if a.dtype == torch.float8_e4m3fn:
            # NOTE: float8_e4m3fn is not supported in pytorch, so we convert it to float16 and then convert it back to float8_e4m3fn
            # This is a workaround to avoid torch error
            a_fp16 = a.to(torch.float16)
            b_fp16 = b.to(torch.float16)
            return (a_fp16 @ b_fp16).to(torch.float8_e4m3fn)
        else:
            return a @ b

    @staticmethod
    def prepare_data(m, n, k, trans_a, trans_b, offset_a, offset_b, dtype):
        device = torch.device("cuda")

        assert offset_a <= 64
        assert offset_b <= 64

        a_size = m * k + offset_a
        b_size = k * n + offset_b
        if dtype == torch.float8_e4m3fn:
            a = torch.rand(a_size, device=device, dtype=torch.float16, requires_grad=False).normal_(std=0.3).to(dtype)
            b = torch.rand(b_size, device=device, dtype=torch.float16, requires_grad=False).normal_(std=0.3).to(dtype)
        else:
            a = torch.rand(a_size, device=device, dtype=dtype, requires_grad=True)
            b = torch.rand(b_size, device=device, dtype=dtype, requires_grad=True)

        if trans_a:
            a = a[offset_a:].view(k, m).detach().contiguous().requires_grad_()
        else:
            a = a[offset_a:].view(m, k).detach().contiguous().requires_grad_()
        if trans_b:
            b = b[offset_b:].view(n, k).detach().contiguous().requires_grad_()
        else:
            b = b[offset_b:].view(k, n).detach().contiguous().requires_grad_()

        alignment_a = common.get_tensor_alignment(a) % 64
        alignment_b = common.get_tensor_alignment(b) % 64

        assert alignment_a == offset_a * a.element_size()
        assert alignment_b == offset_b * b.element_size()
        return a, b

    _backends = ["cutile"]
    if is_backend_available("tilecpp"):
        _backends = _backends + ["tilecpp"]
    if is_backend_available("cutile-rs"):
        _backends = _backends + ["cutile-rs"]
    _perf_backends = _backends + ["pytorch"]
    _test_op_params = _build_matmul_test_op_params(_backends)

    @pytest.mark.parametrize(
        "backend, use_tma, static_persistent, m, n, k, offset_a, offset_b, transpose_a, transpose_b, dtype",
        _test_op_params,
        ids=[
            f"{p[0]}-use_tma={p[1]}-static_persistent={p[2]}-" + "-".join(str(x) for x in p[3:])
            for p in _test_op_params
        ],
    )
    def test_op(
        self,
        m,
        n,
        k,
        offset_a,
        offset_b,
        transpose_a,
        transpose_b,
        dtype,
        static_persistent,
        use_tma,
        backend,
        arch,
        request,
    ):
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        if backend == "cutile-rs":
            if transpose_a or transpose_b:
                pytest.skip("cutile-rs matmul does not support transpose")
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                pytest.skip(f"cutile-rs matmul does not support dtype {dtype}")
        if arch in ["sm120", "sm121"] and n >= 6144:
            pytest.skip("Skip due to global memory OOM")
        if k == 1023:
            pytest.skip("Skip matmul due to result mismatch when cannot divide BLOCK")
        self.setUp()
        a, b = self.prepare_data(m, n, k, transpose_a, transpose_b, offset_a, offset_b, dtype)
        self.assertCorrectness(
            tilegym.ops.matmul,
            self.reference,
            {
                "a": a,
                "b": b,
                "trans_a": transpose_a,
                "trans_b": transpose_b,
            },
            extra_test_kwargs={
                "static_persistent": static_persistent,
                "use_tma": use_tma,
            },
            gradient=torch.rand_like,
            atol=1e-2,
            rtol=1e-2,
        )

    @pytest.mark.parametrize(
        "m,n,k,offset_a,offset_b,dtype",
        [
            (2**i, 2**i, 2**i, 0, 0, dtype)
            for i in list(range(11, 16)) + [6, 8]
            for dtype in ([torch.float16, torch.float32, torch.float8_e4m3fn])
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("transpose_a", [False, True])
    @pytest.mark.parametrize("transpose_b", [False, True])
    @pytest.mark.parametrize("static_persistent", [False, True])
    @pytest.mark.parametrize("use_tma", [False] if torch.cuda.get_device_capability()[0] == 8 else [True])
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(
        self,
        m,
        n,
        k,
        offset_a,
        offset_b,
        transpose_a,
        transpose_b,
        static_persistent,
        use_tma,
        dtype,
        backend,
        record_property,
    ):
        self.setUp()
        if backend == "cutile-rs":
            if transpose_a or transpose_b:
                pytest.skip("cutile-rs matmul does not support transpose")
            if dtype not in (torch.float16, torch.bfloat16, torch.float32):
                pytest.skip(f"cutile-rs matmul does not support dtype {dtype}")
        # Enforce SM80 restrictions: use_tma=False, static_persistent=False, dtype=float16 only
        if torch.cuda.get_device_capability()[0] == 8:
            if use_tma != False or static_persistent != False or dtype != torch.float16:
                pytest.skip(
                    "SM80 restriction: use_tma must be False, static_persistent must be False, and dtype must be float16"
                )

        # Skip FP8 for pytorch reference (no native support)
        if dtype == torch.float8_e4m3fn and backend == "pytorch":
            pytest.skip("Skip float8_e4m3fn because pytorch reference doesn't support it")
        # xfail on sm121 for 32768x32768 matmul due to performance
        if torch.cuda.get_device_capability() == (12, 1) and m == 32768:
            pytest.skip("32768x32768 matmul takes too long on sm121")
        if torch.cuda.get_device_capability() == (12, 0) and m == 32768:
            pytest.skip("Skip OOM on B20X (sm120): 32768³ matmul exceeds 32 GiB VRAM")
        if torch.cuda.get_device_capability()[0] == 8 and m == 32768:
            pytest.skip("Skip 32768x32768 matmul on A100 (sm80) due to OOM")
        if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] == 8:
            pytest.skip("Skip due to sm80 not support fp8 type")
        if backend == "cutile" and not static_persistent and transpose_a:
            pytest.skip("Cutile transpose_a is not supported when static_persistent is False")

        a, b = self.prepare_data(m, n, k, transpose_a, transpose_b, offset_a, offset_b, dtype)
        kernel_kwargs = {
            "trans_a": transpose_a,
            "trans_b": transpose_b,
            "static_persistent": static_persistent,
            "use_tma": use_tma,
        }
        if backend == "pytorch":
            # Detach so the output does not require grad; this keeps the benchmark to the forward pass only.
            _a = a.detach()
            _b = b.detach()
            backend_fn = lambda: self.reference(_a, _b, transpose_a, transpose_b)
        elif tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
            if backend == "cutile" and transpose_b:
                pytest.skip("[matmul] cutile transpose_b is not supported")
            backend_fn = lambda: tilegym.ops.matmul(a, b, **kernel_kwargs)
        else:
            pytest.skip(f"Backend {backend} is not available")
        skip_correctness = backend == "pytorch"
        if not skip_correctness:
            output_processor = None
            if dtype == torch.float8_e4m3fn:
                atol = 1
                rtol = 1
                # float8 doesn't support autograd, disable requires_grad for correctness check
                a = a.detach()
                b = b.detach()
            else:
                atol = 1e-2
                rtol = 1e-2
            self.assertCorrectness(
                backend_fn,
                lambda: self.reference(a, b, trans_a=transpose_a, trans_b=transpose_b),
                kwargs={},
                atol=atol,
                rtol=rtol,
                output_processor=output_processor,
            )
        try:
            res = common.benchmark_framework(backend, backend_fn, use_cudagraph=False)
        except torch.OutOfMemoryError as e:
            pytest.skip(f"OOM during benchmark: {e}")
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del a, b, backend_fn
        if "kernel_configs" in locals():
            del kernel_configs
        torch.cuda.empty_cache()
        gc.collect()

    @pytest.mark.parametrize(
        "model,m,n,k,offset_a,offset_b,dtype",
        [
            ("gpt3-40b", 4096, 8192, 2728, 0, 0, torch.float16),
            ("gpt3-7b", 4096, 4096, 5440, 0, 0, torch.float16),
            ("t5-3b", 12288, 2048, 2560, 0, 0, torch.float16),
            ("t5-11b", 12288, 4096, 2560, 0, 0, torch.float16),
            ("t5-23b", 4096, 5120, 2720, 0, 0, torch.float16),
            ("t5-41b", 3072, 6144, 2720, 0, 0, torch.float16),
        ],
        ids=lambda x: str(x) if isinstance(x, list) else x.__name__ if hasattr(x, "__name__") else str(x),
    )
    @pytest.mark.parametrize("static_persistent", [True, False])
    @pytest.mark.parametrize("use_tma", [True])
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf_llm(
        self,
        model,
        m,
        n,
        k,
        offset_a,
        offset_b,
        dtype,
        static_persistent,
        use_tma,
        backend,
        record_property,
    ):
        self.setUp()
        if backend == "cutile-rs" and dtype not in (torch.float16, torch.bfloat16, torch.float32):
            pytest.skip(f"cutile-rs matmul does not support dtype {dtype}")
        if torch.cuda.get_device_capability()[0] == 8:
            pytest.skip("Skip on sm80")

        a, b = self.prepare_data(m, n, k, False, False, offset_a, offset_b, dtype)
        kernel_kwargs = {
            "trans_a": False,
            "trans_b": False,
            "static_persistent": static_persistent,
            "use_tma": use_tma,
        }
        if backend == "pytorch":
            backend_fn = lambda: self.reference(a, b)
        elif tilegym.is_backend_available(backend):
            try:
                tilegym.set_backend(backend)
            except Exception as e:
                pytest.skip(f"Backend {backend} is not available: {e}")
            backend_fn = lambda: tilegym.ops.matmul(a, b, **kernel_kwargs)
        else:
            pytest.skip(f"Backend {backend} is not available")
        skip_correctness = backend == "pytorch"
        if not skip_correctness:
            self.assertCorrectness(
                backend_fn,
                lambda: self.reference(a, b),
                kwargs={},
                atol=1e-2,
                rtol=1e-2,
            )
        res = common.benchmark_framework(backend, backend_fn, use_cudagraph=False)
        record_property("benchmark", res)

        # Explicit cleanup to prevent OOM
        del a, b, backend_fn
        if "kernel_configs" in locals():
            del kernel_configs
        torch.cuda.empty_cache()
        gc.collect()


def get_weight_shapes(tp_size=1):
    """Get weight shapes for DeepSeek-V3-like models."""
    # NOTE: The weight shapes only works for DeepSeek-V3.
    # cannot TP
    total = [
        (512 + 64, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (7168, 16384),
        (7168, 18432),
        # Runtime fused/TP-split shapes from SGLang real serving GEMM logs (DeepSeek R1)
        (1536 + 512 + 64, 7168),  # fused_qkv_a_proj: q_a + kv_a + nope fused
        (24576 // 8, 1536),  # q_b_proj at tp=8 (n_tp shape 24576/8)
        (7168, 2048 // 8),  # kv_b_o_proj at tp=8 (k_tp shape 2048/8)
    ]
    # N can TP
    n_tp = [
        (18432 * 2, 7168),
        ((128 + 64) * 128, 7168),
        (128 * (128 + 128), 512),
        (24576, 1536),
        (4096, 7168),
    ]
    # K can TP
    k_tp = [(7168, 18432), (7168, 16384), (7168, 2048)]

    weight_shapes = []
    for t in total:
        weight_shapes.append(t)
    for n_t in n_tp:
        new_t = (n_t[0] // tp_size, n_t[1])
        weight_shapes.append(new_t)
    for k_t in k_tp:
        new_t = (k_t[0], k_t[1] // tp_size)
        weight_shapes.append(new_t)
    # eliminate duplicate weight shapes
    weight_shapes = list(set(weight_shapes))
    return weight_shapes


class Test_W8A8BlockFp8Matmul(common.PyTestCase):
    _backends = ["cutile"]
    # Backends reached through the tilegym dispatcher.
    _dispatch_backends = _backends
    _perf_backends = _backends + ["pytorch"]

    @staticmethod
    def reference(A_fp8, B_fp8, As, Bs, block_size=[128, 128]):
        """Vectorized PyTorch reference for block-wise FP8 matmul.

        Replaces the O(N/block_n × K/block_k) Python loop with a single
        torch.mm call, eliminating tens-of-thousands of kernel launches for
        large DeepSeek-V3 weight shapes (e.g. 36864×7168, block=32×32 had
        258048 loop iterations).

        Math: C[m,n] = Σ_k  A[m,k]·As[m, k//bk]  ·  B[n,k]·Bs[n//bn, k//bk]
              = (A ⊙ scale_A) @ (B ⊙ scale_B).T   where scales are broadcast
              per tile — factorises into one fused matmul.
        """
        block_n, block_k = block_size

        *batch_dims, K = A_fp8.shape
        N, K_B = B_fp8.shape
        assert K == K_B, f"K dimension mismatch: {K} vs {K_B}"

        M = A_fp8.view(-1, K).shape[0]
        K_tiles = (K + block_k - 1) // block_k
        N_tiles = (N + block_n - 1) // block_n
        K_pad = K_tiles * block_k
        N_pad = N_tiles * block_n

        A_f = A_fp8.float().view(M, K)
        B_f = B_fp8.float()  # [N, K]

        # Pad to tile boundaries when N or K is not a multiple of block size
        if K_pad != K:
            A_f = torch.nn.functional.pad(A_f, (0, K_pad - K))
            B_f = torch.nn.functional.pad(B_f, (0, K_pad - K))
        if N_pad != N:
            B_f = torch.nn.functional.pad(B_f, (0, 0, 0, N_pad - N))

        # Scale A: As[M, K_tiles] broadcast over block_k elements per tile
        A_scaled = (A_f.view(M, K_tiles, block_k) * As.view(M, K_tiles, 1)).view(M, K_pad)

        # Scale B: Bs[N_tiles, K_tiles] broadcast over (block_n, block_k) per tile
        B_scaled = (B_f.view(N_tiles, block_n, K_tiles, block_k) * Bs.view(N_tiles, 1, K_tiles, 1)).view(N_pad, K_pad)

        # Single matmul — O(1) Python overhead vs O(N_tiles * K_tiles) loops
        C = torch.mm(A_scaled, B_scaled[:N].T)

        output_shape = list(batch_dims) + [N]
        return C.view(output_shape)

    @staticmethod
    def prepare_data(M, N, K, block_n, block_k, dtype=torch.float16):
        """Prepare test data for benchmarking."""
        device = torch.device("cuda")
        factor_for_scale = 1e-2
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        # Create FP8 data
        A_fp32 = (torch.rand(M, K, dtype=torch.float32, device=device).normal_(mean=0.0, std=0.3) - 0.5) * 2 * fp8_max
        A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        B_fp32 = (torch.rand(N, K, dtype=torch.float32, device=device).normal_(mean=0.0, std=0.3) - 0.5) * 2 * fp8_max
        B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        # Create scales
        n_tiles = (N + block_n - 1) // block_n
        k_tiles = (K + block_k - 1) // block_k

        As = torch.rand(M, k_tiles, dtype=torch.float32, device=device) * factor_for_scale
        Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32, device=device) * factor_for_scale

        return A_fp8, B_fp8, As, Bs

    @pytest.mark.slow
    @pytest.mark.parametrize("batch_size", [32, 256])
    @pytest.mark.parametrize("weight_shapes", get_weight_shapes(tp_size=1))
    @pytest.mark.parametrize(
        "block_n,block_k",
        [
            (32, 32),
            (128, 128),
        ],
    )
    @pytest.mark.parametrize("output_dtype", [torch.float32])
    @pytest.mark.parametrize("backend", _backends)
    def test_op(self, batch_size, weight_shapes, block_n, block_k, output_dtype, arch, backend, request):
        if arch == "sm80":
            pytest.skip("FP8 is not supported on sm80 (Ampere).")
        if tilegym.is_backend_available(backend):
            tilegym.set_backend(backend)
        else:
            pytest.skip(f"Backend {backend} is not available")
        self.setUp()

        # Get weight shapes for this tp_size
        N, K = weight_shapes

        # Prepare test data
        A_fp8, B_fp8, As, Bs = self.prepare_data(batch_size, N, K, block_n, block_k, output_dtype)
        block_size = [block_n, block_k]

        # Test with reference implementation
        self.assertCorrectness(
            lambda A, B, As, Bs, block_size, output_dtype: tilegym.ops.w8a8_block_fp8_matmul(
                A, B, As, Bs, block_size=block_size, output_dtype=output_dtype
            ),
            lambda A, B, As, Bs, block_size, output_dtype: self.reference(A, B, As, Bs, block_size).to(output_dtype),
            {
                "A": A_fp8,
                "B": B_fp8,
                "As": As,
                "Bs": Bs,
                "block_size": block_size,
                "output_dtype": output_dtype,
            },
            rtol=9e-1,  # Higher tolerance for FP8 precision
            atol=9e-1,  # Higher absolute tolerance
        )

        # Cleanup between iterations to prevent OOM
        del A_fp8, B_fp8, As, Bs, block_size
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("batch_size", [32, 4096])
    @pytest.mark.parametrize("weight_shapes", get_weight_shapes(tp_size=1))
    @pytest.mark.parametrize(
        "block_n,block_k",
        [
            (32, 32),
            (128, 128),
        ],
    )
    # bf16 is the realistic dtype for DeepSeek-V3 inference.
    @pytest.mark.parametrize("output_dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", _perf_backends)
    def test_perf(
        self,
        batch_size,
        weight_shapes,
        block_n,
        block_k,
        output_dtype,
        arch,
        backend,
        record_property,
    ):
        self.setUp()
        if arch == "sm80":
            pytest.skip("FP8 is not supported on sm80 (Ampere).")
        # The autotune pruner drops configs where BLOCK_M > batch_size, so a small
        # batch_size directly caps BLOCK_M.  sm90 WGMMA and sm100 tcgen05.mma both
        # require M >= 64 (and block-scaled variants require M >= 128), so any
        # batch_size < 128 cannot exercise Hopper or Blackwell MMA and is misleading
        # as a regression signal on sm90+.
        if batch_size < 128 and torch.cuda.get_device_capability()[0] >= 9:
            pytest.skip(
                f"batch_size={batch_size} caps BLOCK_M < 128, below sm90 WGMMA / "
                "sm100 tcgen05.mma minimum (M >= 64). Not a valid regression signal "
                "on Hopper or Blackwell."
            )
        N, K = weight_shapes

        # Prepare test data
        A_fp8, B_fp8, As, Bs = self.prepare_data(batch_size, N, K, block_n, block_k, output_dtype)
        block_size = [block_n, block_k]
        backend_fn = None
        if backend in self._dispatch_backends:
            tilegym.set_backend(backend)
            backend_fn = lambda: tilegym.ops.w8a8_block_fp8_matmul(
                A_fp8,
                B_fp8,
                As,
                Bs,
                block_size=block_size,
                output_dtype=output_dtype,
            )
        elif backend == "pytorch":
            backend_fn = lambda: self.reference(A_fp8, B_fp8, As, Bs, block_size).to(output_dtype)
        if backend_fn is None:
            pytest.skip(f"Backend {backend} not supported")

        # Run benchmarks
        res = common.benchmark_framework(backend, backend_fn, use_cudagraph=False)

        # Record results for reporting
        record_property("benchmark", res)
        torch.cuda.synchronize()

        # pytorch IS self.reference.
        _skip_correctness = ("pytorch",)
        if backend not in _skip_correctness:
            ref_out = self.reference(A_fp8, B_fp8, As, Bs, block_size).to(output_dtype)
            actual_out = backend_fn()
            torch.testing.assert_close(actual_out, ref_out, atol=9e-1, rtol=9e-1)

        # Explicit cleanup to prevent OOM
        del A_fp8, B_fp8, As, Bs, backend_fn, block_size
        torch.cuda.empty_cache()
        gc.collect()
