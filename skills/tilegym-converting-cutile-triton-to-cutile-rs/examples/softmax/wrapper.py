# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0

"""
cutile-rs softmax backend via FFI to the shared libcutile_kernels.so.

This is the canonical Python wrapper template for Agent D, which owns the host
boundary in the current conversion pipeline. Agent B writes device kernel.rs;
Agent D writes ffi.rs, this Python wrapper shape, backend wiring, and the
correctness report.

It demonstrates:
    1. Shared cdylib loading via bind_kernel_function_cffi (one
       libcutile_kernels.so for every op; the op cdefs only its own symbol).
    2. A cffi cdef (_FFI_CDEF) that exactly matches ffi.rs; tensors cross as
       `const TensorDesc*` and are packed with make_tensor_desc (dtype/shape/
       strides travel inside the descriptor — no ctypes argtypes list to drift).
    3. dtype/parameter gates that raise for unsupported cases.
    4. fixed kernel_configs bypass for debugging.
    5. CUPTI autotune with torch.empty-only allocations inside kernel_fn.
    6. an autograd-aware public wrapper.
    7. carrying autotune-config compile options (occupancy / num_cta_in_cga)
       through to the FFI so a tuned CGA cluster size actually takes effect.
    8. passing the tensor's device ordinal (device_id) so multi-GPU launches
       target the right device.

Important wrapper rules:
    - cutile-rs is FORWARD-ONLY. If input.requires_grad is True, detach it
      first and return a leaf tensor (no grad_fn). Do NOT wrap forward in
      torch.autograd.Function; backward via a PyTorch formula adds large
      slowdown on the combined fwd/bwd ratio and obscures the kernel-level
      perf signal we want to evolve. Backward correctness tests must skip
      the cutile-rs backend explicitly.
    - detach() is essentially free, so apply it unconditionally when
      requires_grad is True.
    - Match launch grid semantics to the Rust kernel body. This softmax
      template is persistent: it may launch fewer CTAs than logical rows
      because kernel.rs loops over remaining rows with a grid-stride loop.
      For a non-persistent one-block-per-row kernel, use grid=n_rows. Do not
      cap such a grid by occupancy or num_sms.
    - For a persistent grid-stride kernel, size the launch to a resident-CTA
      budget, not to a bare one-CTA-per-SM cap. The safe pattern is:
          grid = min(total_work_tiles, num_sms * resident_ctas_per_sm / cga)
      where total_work_tiles is the loop's logical tile count, usually row
      tiles for reductions/norms. A wrapper like `min(num_sms, total_tiles)`
      is correctness-safe but can starve memory-bound persistent kernels.
    - Keep grid-budget knobs separate from compile options. More launched CTAs
      do not imply `CompileOptions::occupancy` should be pinned, and a compile
      occupancy hint does not by itself increase a too-small launch grid.
    - Do not run a fixed-config canary in the hot path before autotune_launch.
      CUPTI sums every GPU kernel inside the measured function; a canary plus
      cached best-config launch doubles the measured time.
    - Compile options (occupancy / num_cta_in_cga) follow analysis.json /
      Agent A's best_config, NOT a blanket default:
        * If the reference autotune space carries an EXPLICIT num_ctas /
          num_cta_in_cga (e.g. matmul ships num_ctas in {1,2,4}), you MUST
          forward that per-config value to the FFI so the CGA cluster size
          takes effect. Passing the auto sentinel here silently pins the
          cluster to the compiler default (=1) and regresses persistent /
          tensor-core kernels ~1.6x. This is the #1 perf trap for GEMM.
        * Only when the reference genuinely uses num_ctas=None / "compiler
          auto-pick" (e.g. some elementwise / decode kernels) do you leave the
          field None and pass the auto sentinel. Decide per-kernel from
          analysis.json; do not default everything to auto.
"""

from types import SimpleNamespace

import torch

from tilegym.backend import register_impl
from tilegym.backend.cutile_rs.autotuner import autotune_launch
from tilegym.backend.cutile_rs.utils import bind_kernel_function_cffi
from tilegym.backend.cutile_rs.utils import check_rc
from tilegym.backend.cutile_rs.utils import get_num_sm
from tilegym.backend.cutile_rs.utils import make_tensor_desc

_KERNEL = "softmax"
_FFI_NAME = "cutile_softmax"
# C-declaration source of truth for the cffi boundary — keep in sync with the
# `cutile_softmax` signature in ffi.rs. Tensors cross as `const TensorDesc*`
# (the shared TensorDesc typedef is prepended by bind_kernel_function_cffi); m/n
# and row/col strides are read off the descriptors inside the FFI.
_FFI_CDEF = """
int32_t cutile_softmax(
    const TensorDesc* y, const TensorDesc* x,
    int32_t bm, int32_t bn, int32_t latency,
    int32_t num_cta_in_cga, int32_t occupancy,
    int32_t grid_size, int32_t device_id, uint64_t raw_stream);
"""

# Supported dtypes (wrapper input validation; the dtype code is packed by
# make_tensor_desc into TensorDesc.dtype).
_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# FFI convention: a value >0 calls the CompileOptions setter with that value;
# a value <=0 (the _AUTO_COMPILE_OPTION sentinel) means "leave it at the
# compiler default". ffi.rs MUST honor this guard:
#
#   let mut opts = CompileOptions::default();
#   if occupancy > 0 { opts = opts.occupancy(occupancy); }
#   if num_cta_in_cga > 0 { opts = opts.num_cta_in_cga(num_cta_in_cga); }
#
# IMPORTANT: the sentinel is for kernels whose reference genuinely uses
# num_ctas=None / auto. It is NOT a blanket default. If analysis.json /
# best_config records an explicit num_cta_in_cga (matmul does), set the
# _COMPILE_* values below (or sweep them in _configs) so a REAL cluster size
# reaches the kernel. Defaulting these to None on a kernel whose reference
# tunes num_ctas pins the cluster to 1 and is the dominant GEMM perf regression.
_AUTO_COMPILE_OPTION = -1

# Grid-budget knobs for this persistent softmax template. These control how
# many CTAs are launched for a grid-stride kernel; they are SEPARATE from the
# compile-option cluster knob below.
#
# For memory-bound persistent reductions/norms, 1 CTA/SM is often not enough
# even though it is correct. Launch a resident-CTA budget (num_sms * a
# resident-per-SM factor, clamped to row tiles) and forward a matching
# compile occupancy for the static_persistent variant.
# Do not cargo-cult those values globally; copy the resident-grid pattern below
# and sweep only when E/C localizes a CTA-count-sensitive gap.
_GRID_OCCUPANCY = 2
_GRID_NUM_CTA_IN_CGA = 1

# Compile-time knobs. softmax's reference uses compiler auto-pick, so these stay
# None HERE. For your op, set them from analysis.json / best_config: if the
# reference autotunes num_ctas (e.g. matmul ships {1,2,4}), DO NOT leave these
# None; carry the per-config value into the FFI (see _configs below).
_COMPILE_OCCUPANCY = None
_COMPILE_NUM_CTA_IN_CGA = None


def _compile_option_value(value) -> int:
    if value is None:
        return _AUTO_COMPILE_OPTION
    return int(value)


def _configs(m: int, n: int):
    """Autotune search space.

    Start with analysis.json reference configs for the real op, then optionally
    add expansions after correctness passes.

    Compile-option fields per config:
      - softmax's reference auto-picks, so OCCUPANCY / NUM_CTA_IN_CGA stay None.
      - For an op whose reference autotunes num_ctas (matmul ships {1,2,4}),
        carry the real value, e.g.:
            SimpleNamespace(BM=256, BN=256, BK=64, NUM_CTA_IN_CGA=2, OCCUPANCY=1)
            SimpleNamespace(BM=256, BN=256, BK=64, NUM_CTA_IN_CGA=4, OCCUPANCY=1)
        so the autotuner explores live CGA cluster sizes. Do NOT fill these with
        None on such an op; that pins the cluster to the compiler default (=1).
    """
    del m
    out = []
    for bm in (64, 128):
        for bn in (128, 256):
            # kernel contract: N <= BN (the BN-wide tile must cover the row;
            # padding_value=0 handles N < BN). Keep only configs where BN >= N.
            if bn >= n:
                for latency in (2, 4):
                    out.append(
                        SimpleNamespace(
                            BM=bm,
                            BN=bn,
                            LATENCY=latency,
                            OCCUPANCY=_COMPILE_OCCUPANCY,
                            NUM_CTA_IN_CGA=_COMPILE_NUM_CTA_IN_CGA,
                        )
                    )
    if not out:
        out.append(
            SimpleNamespace(
                BM=64,
                BN=128,
                LATENCY=2,
                OCCUPANCY=_COMPILE_OCCUPANCY,
                NUM_CTA_IN_CGA=_COMPILE_NUM_CTA_IN_CGA,
            )
        )
    return out


def _resident_persistent_grid_size(
    total_work_tiles: int,
    *,
    resident_ctas_per_sm: int = _GRID_OCCUPANCY,
    cta_cluster: int = _GRID_NUM_CTA_IN_CGA,
) -> int:
    """Grid for kernels that explicitly grid-stride over logical work tiles.

    The matching Rust body must contain logic equivalent to:

        for tile_id in (bid_x..total_work_tiles).step_by(grid_x as usize) { ... }

    `total_work_tiles` is the loop's logical bound, not a product of unrelated
    axes. For row-wise reductions/norms this is usually cdiv(M, TILE_SIZE_M).
    For softmax it is cdiv(M, BM). If the Rust body handles only
    get_tile_block_id().0 once, do not use this helper; use
    _full_coverage_grid(n_blocks) instead.

    The important perf point is resident budget, not bare SM count. A
    grid-stride kernel launched with min(num_sms, total_work_tiles) is often
    correct but can leave memory-bound kernels at ~1 CTA/SM.
    """
    total_work_tiles = int(total_work_tiles)
    resident_ctas_per_sm = max(int(resident_ctas_per_sm), 1)
    cta_cluster = max(int(cta_cluster), 1)
    num_sms = get_num_sm()
    cta_budget = max((num_sms * resident_ctas_per_sm) // cta_cluster, 1)
    return min(total_work_tiles, cta_budget)


def _persistent_grid_size(m: int, bm: int) -> int:
    """Softmax-specific persistent grid helper."""
    num_tiles_m = (m + bm - 1) // bm
    return _resident_persistent_grid_size(num_tiles_m)


def _full_coverage_grid(n_blocks: int) -> int:
    """Grid for non-persistent kernels: one CTA per logical block/row."""
    return int(n_blocks)


def _run_ffi(
    y,
    x2,
    m: int,
    n: int,
    bm: int,
    bn: int,
    latency: int,
    compile_occupancy=None,
    compile_num_cta_in_cga=None,
):
    """One FFI launch. Used by fixed-config and autotune paths."""
    ffi, lib = bind_kernel_function_cffi(_KERNEL, _FFI_CDEF)
    _dev = x2.device
    device_id = _dev.index if _dev.index is not None else torch.cuda.current_device()
    raw_stream = torch.cuda.current_stream(device=_dev).cuda_stream

    # This template's kernel is persistent, so a resident-CTA capped grid is
    # correct here. For row-wise elementwise ports, pass _full_coverage_grid
    # unless kernel.rs has a grid-stride loop over rows.
    grid_size = _persistent_grid_size(m, bm)

    # Keep the descriptors alive until after the FFI call (cffi frees on GC).
    yd = make_tensor_desc(ffi, y)
    xd = make_tensor_desc(ffi, x2)
    rc = lib.cutile_softmax(
        yd,
        xd,
        int(bm),
        int(bn),
        int(latency),
        _compile_option_value(compile_num_cta_in_cga),
        _compile_option_value(compile_occupancy),
        int(grid_size),
        int(device_id),
        int(raw_stream),
    )
    check_rc(rc, _FFI_NAME)
    return y


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    return dim


def _validate_inputs(x: torch.Tensor, dim: int) -> int:
    if not x.is_cuda:
        raise ValueError("cutile-rs softmax: input must be a CUDA tensor")
    dim = _normalize_dim(dim, x.ndim)
    if dim != x.ndim - 1:
        raise NotImplementedError(
            "cutile-rs softmax template supports only the last dimension. "
            "For another op, either implement the layout natively or raise."
        )
    if x.dtype not in _DTYPES:
        raise NotImplementedError(
            f"cutile-rs softmax: dtype {x.dtype} not supported (extend ffi.rs dtype dispatch and make_tensor_desc together)."
        )
    return dim


def _forward_impl(x: torch.Tensor, dim: int = -1, kernel_configs=None):
    """Forward implementation shared by plain and autograd paths.

    Keep expensive copies out of the autotune lambda. A single input contiguous
    conversion here is acceptable only when it is part of the kernel contract.
    """
    dim = _validate_inputs(x, dim)

    x_contig = x.contiguous()
    n = int(x_contig.shape[-1])
    if n == 0 or x_contig.numel() == 0:
        return torch.empty_like(x_contig)
    m = int(x_contig.numel() // n)
    x2 = x_contig.reshape(m, n)

    def launch_with_cfg(cfg):
        if int(cfg.BN) < n:
            raise NotImplementedError(
                f"cutile-rs softmax: kernel requires N<=BN (N={n}, BN={cfg.BN}); "
                "use a larger BN or add an inner N-tile loop"
            )
        y_local = torch.empty_like(x2)
        _run_ffi(
            y_local,
            x2,
            m,
            n,
            int(cfg.BM),
            int(cfg.BN),
            int(cfg.LATENCY),
            compile_occupancy=getattr(cfg, "OCCUPANCY", _COMPILE_OCCUPANCY),
            compile_num_cta_in_cga=getattr(cfg, "NUM_CTA_IN_CGA", _COMPILE_NUM_CTA_IN_CGA),
        )
        return y_local

    if kernel_configs:
        cfg = SimpleNamespace(
            BM=int(kernel_configs.get("BM", kernel_configs.get("bm", 128))),
            BN=int(kernel_configs.get("BN", kernel_configs.get("bn", 128))),
            LATENCY=int(kernel_configs.get("LATENCY", kernel_configs.get("latency", 4))),
            OCCUPANCY=kernel_configs.get("OCCUPANCY", kernel_configs.get("occupancy", _COMPILE_OCCUPANCY)),
            NUM_CTA_IN_CGA=kernel_configs.get(
                "NUM_CTA_IN_CGA",
                kernel_configs.get("num_cta_in_cga", _COMPILE_NUM_CTA_IN_CGA),
            ),
        )
        return launch_with_cfg(cfg).reshape(x_contig.shape)

    configs = _configs(m, n)

    def kernel_fn(cfg):
        # Autotune rule: allocate outputs with torch.empty only. Do not clone,
        # zeros, ones, or make additional contiguous copies here. Do not launch a
        # separate canary before this lambda; autotune_launch calls this function
        # for warmup/timing and then once with the winning cached config.
        return launch_with_cfg(cfg)

    result = autotune_launch(
        kernel_fn=kernel_fn,
        configs=configs,
        key=(m, n, x.dtype),
        kernel_name=_KERNEL,
    )
    return result.output.reshape(x_contig.shape)


@register_impl("softmax", backend="cutile-rs")
def softmax(x: torch.Tensor, dim: int = -1, **kwargs):
    """Softmax via cutile-rs FFI (forward-only).

    Unsupported semantic cases raise. Do not use a PyTorch forward fallback;
    that hides conversion gaps.
    """
    kernel_configs = kwargs.get("kernel_configs", None)
    _validate_inputs(x, dim)

    if x.requires_grad:
        x = x.detach()

    return _forward_impl(x, dim=dim, kernel_configs=kernel_configs)


# Copy patterns for common generated wrappers:
#
# Direct-launch / no-autotune attention-style kernels:
#
#   Decide per-kernel from Agent A's analysis.json / best_config:
#     - reference num_ctas=None / "compiler auto-pick"  -> keep _COMPILE_* = None
#       and pass the auto sentinel; do NOT pass 1 "to seem safe" (caps resident
#       CTAs, regresses high-CTA decode shapes).
#     - reference autotunes num_ctas (matmul ships {1,2,4}) -> carry the real
#       per-config value through to the FFI. Leaving it None pins the CGA
#       cluster to the compiler default (=1) and regresses GEMM ~1.6x.
#
# FFI sentinel pattern:
#
#   let mut opts = CompileOptions::default();
#   if occupancy > 0 { opts = opts.occupancy(occupancy); }
#   if num_cta_in_cga > 0 { opts = opts.num_cta_in_cga(num_cta_in_cga); }
#   let op = kernel(...).compile_options(opts);
#
# Persistent reductions / norm-style launch:
#
#   Only use this pattern when kernel.rs contains a grid-stride loop over row
#   tiles, e.g.:
#       for cur_bid in (bid_x..cdiv(M, TILE_SIZE_M)).step_by(grid_x as usize) { ... }
#
#   tile_size_m = ...
#   num_row_tiles = _ceil_div(m, tile_size_m)
#   grid_size = _resident_persistent_grid_size(
#       num_row_tiles,
#       resident_ctas_per_sm=_SP_GRID_OCCUPANCY,
#       cta_cluster=_SP_GRID_NUM_CTA_IN_CGA,
#   )
#
#   If E/C localizes a gap to only this persistent variant while IR matches,
#   suspect the host grid before rewriting kernel math: a too-small resident-CTA
#   budget (about 1 CTA/SM) is a common cause. Raise _SP_GRID_OCCUPANCY and the
#   matching compile occupancy, clamped to row tiles.
#   Keep num_cta_in_cga auto unless analysis.json/reference has an explicit
#   cluster value.
#
# SiLU-and-mul / row-wise elementwise launch:
#
#   n_rows = int(x_flat.shape[0])
#   grid_size = _full_coverage_grid(n_rows)
#
#   This is mandatory unless kernel.rs contains:
#       for row in (bid_x..n_rows).step_by(grid_x as usize) { ... }
#
#   A perf geomean that is much faster than reference while correctness fails is
#   often a partial-output launch. Check the first mismatched row against
#   get_num_sm() * occupancy.
#
# Autotuned row-wise wrappers:
#
#   def kernel_fn(cfg):
#       y_local = torch.empty((n_rows, hidden_size), dtype=x.dtype, device=x.device)
#       _run_ffi(x2, y_local, compile_occupancy=getattr(cfg, "OCCUPANCY", None))
#       return y_local
#
#   result = autotune_launch(kernel_fn=kernel_fn, configs=configs, key=(...), kernel_name=_KERNEL)
#
#   Do not allocate a separate y2 and do not call _run_ffi once as a canary
#   before autotune_launch. That causes two GPU launches per measured forward on
#   cache hits and creates an artificial ~2x CUPTI regression.
#
# Backward is OUT OF SCOPE for cutile-rs in this skill version.
#
# All cutile-rs wrappers must detach grad-enabled inputs and return a
# leaf tensor; the FFI call only emits the forward direction. Test patches
# for ops with backward variants must skip the `cutile-rs` backend for any
# backward-correctness / backward-perf parametrizations.
#
# In-place reference ops:
#
#   The detached input may not be modified in-place if the original required
#   grad. Clone before the FFI call:
#       x = x.detach().clone() if x.requires_grad else x.contiguous()
