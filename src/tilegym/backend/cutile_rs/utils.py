# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

#

"""cutile-rs FFI utilities: lazy build + cffi loader for the shared
libcutile_kernels.so, plus tensor packing and small device helpers.

All cutile-rs kernels build into ONE cdylib (ops/cutile_rs/cutile_kernels/),
rebuilt on demand from the pure-.rs op sources (see _build_kernels and
CUTILE_RS_AUTOBUILD). Wrappers bind an op via bind_kernel_function_cffi and pack
tensors with make_tensor_desc.

Public API:
    bind_kernel_function_cffi() — (ffi, lib) for an op, cdef'd over the shared .so
    make_tensor_desc()          — pack a torch.Tensor into a cffi TensorDesc*
    next_power_of_2() / get_num_sm() / check_rc() — small helpers

Environment variables:
    CUTILE_RS_AUTOBUILD   — "0" disables the stale-source rebuild (default on)
    CUTILE_RS_KERNELS_DIR — override the cutile_kernels crate location
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

import torch

# Logger fallback — same pattern as autotune_launch.py.
try:
    from tilegym.logger import get_logger  # type: ignore

    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# ─── Shared cdylib loading (libcutile_kernels.so) ──────────────────────────
#
# All cutile-rs kernels build into ONE cdylib (ops/cutile_rs/cutile_kernels/);
# per-op sources are pure-.rs in ops/cutile_rs/<op>_kernel/ and are include!'d
# by the crate's src/lib.rs. Wrappers bind via bind_kernel_function_cffi (below).


def _kernels_crate_dir() -> str | None:
    """The single cdylib crate aggregating ALL cutile-rs kernels.

    ``backend/cutile_rs/`` -> ``../../ops/cutile_rs/cutile_kernels/`` (override:
    ``$CUTILE_RS_KERNELS_DIR``). Per-op sources live as pure-.rs in the sibling
    ``ops/cutile_rs/<op>_kernel/`` dirs and are ``include!``d by the crate's
    src/lib.rs. One crate -> one ``libcutile_kernels.so`` exporting every
    ``cutile_<op>`` symbol; no shared cutile-rs checkout, no ``CUTILE_RS_DIR``.
    """
    override = os.environ.get("CUTILE_RS_KERNELS_DIR")
    if override and os.path.isfile(os.path.join(override, "Cargo.toml")):
        return override
    crate = os.path.join(os.path.dirname(__file__), "..", "..", "ops", "cutile_rs", "cutile_kernels")
    return crate if os.path.isfile(os.path.join(crate, "Cargo.toml")) else None


def _ops_src_root() -> str:
    """ops/cutile_rs/ — holds the crate + every <op>_kernel/ source dir."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "ops", "cutile_rs")


def _shared_so_path(crate_dir: str, profile: str = "release") -> str:
    """The one cdylib for all ops: <crate>/target/<profile>/libcutile_kernels.so."""
    return os.path.join(crate_dir, "target", profile, "libcutile_kernels.so")


# ── Auto-build on stale source (default ON; opt-out via CUTILE_RS_AUTOBUILD=0) ─
def _autobuild_enabled() -> bool:
    # Rebuild the .so whenever a .rs/.toml is newer than it, by default — editing
    # a kernel then re-running "just works". Disable with CUTILE_RS_AUTOBUILD=0
    # (e.g. to pin a prebuilt .so, or where cargo is intentionally absent).
    return os.environ.get("CUTILE_RS_AUTOBUILD", "1").lower() not in ("0", "false", "no")


def _so_stale(so_path: str) -> bool:
    """Stale if the .so is missing or older than any .rs/.toml under
    ops/cutile_rs/ (the crate src + every <op>_kernel/ source). target/ skipped.
    """
    if not os.path.isfile(so_path):
        return True
    so_mtime = os.path.getmtime(so_path)
    for root, _, files in os.walk(_ops_src_root()):
        if "target" in root.split(os.sep):
            continue
        for f in files:
            if f.endswith((".rs", ".toml")) and os.path.getmtime(os.path.join(root, f)) > so_mtime:
                return True
    return False


_BUILD_ENV_KEYS_KERNEL = (
    "HOME",
    "PATH",
    "USER",
    "TERM",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "LD_LIBRARY_PATH",
    "CC",
    "CXX",
    "RUSTC",
    "RUSTFLAGS",
    "CUDA_TOOLKIT_PATH",
    "LIBCLANG_PATH",
)


def _build_kernels(crate_dir: str) -> None:
    """``cargo build --release`` the single cutile_kernels crate into its own
    target/ (crates.io deps; no cutile-rs checkout). File-locked + re-checks
    staleness inside the lock so concurrent callers don't rebuild redundantly.
    """
    import fcntl

    so_path = _shared_so_path(crate_dir)
    os.makedirs(os.path.dirname(so_path), exist_ok=True)
    lock_path = os.path.join(crate_dir, ".build.lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        if not _so_stale(so_path):
            return
        logger.info(f"cutile-rs: building libcutile_kernels.so (cargo build --release) in {crate_dir} ...")
        t0 = time.time()
        env = {k: os.environ[k] for k in _BUILD_ENV_KEYS_KERNEL if k in os.environ}
        # cuda-bindings' build.rs runs bindgen, which locates libclang via clang-sys.
        # clang-sys already searches the standard install dirs (/usr/lib/llvm-*/lib,
        # /usr/lib/x86_64-linux-gnu, ...), so an apt-installed libclang-dev is found
        # without help. Set LIBCLANG_PATH explicitly only for a non-standard clang.
        # cuda-bindings' build.rs reads $CUDA_TOOLKIT_PATH (panics if unset) and
        # passes -I$CUDA_TOOLKIT_PATH/include to bindgen; clang supplies its own
        # builtin headers, so no extra bindgen flags are needed. The runtime
        # entrypoint may point CUDA_TOOLKIT_PATH at a headerless runtime CUDA (for
        # tileiras), so honor a pre-set value only when it actually carries the
        # CUDA headers, else fall back to the standard /usr/local/cuda toolkit.
        cuda_toolkit = env.get("CUDA_TOOLKIT_PATH") or "/usr/local/cuda"
        if not os.path.isfile(os.path.join(cuda_toolkit, "include", "cuda.h")):
            cuda_toolkit = "/usr/local/cuda"
        env["CUDA_TOOLKIT_PATH"] = cuda_toolkit
        # Resolve cargo to an absolute path (avoid PATH hijacking).
        cargo_bin = shutil.which("cargo")
        if cargo_bin is None:
            raise RuntimeError(
                "cutile-rs: cargo not found on PATH (install Rust, or set CUTILE_RS_AUTOBUILD=0 to use a prebuilt .so)."
            )
        try:
            subprocess.run(
                [cargo_bin, "build", "--release"],
                cwd=crate_dir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=900,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "cutile-rs: cargo not found on PATH (install Rust, or set CUTILE_RS_AUTOBUILD=0 to use a prebuilt .so)."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"cutile-rs: cargo build timed out (>900s) in {crate_dir}.") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"cutile-rs: cargo build failed (exit {e.returncode}) in {crate_dir}:\n{(e.stderr or '')[-2000:]}"
            ) from e
        logger.info(f"cutile-rs: built libcutile_kernels.so in {time.time() - t0:.1f}s")


def _ensure_built_and_path() -> str:
    """Resolve the shared .so path, autobuilding if enabled+stale.

    The staleness check runs before the first load, so each process builds (when
    needed) and then loads current kernels. The cffi handle cache is dropped on a rebuild, but note this does NOT hot-reload within a live process
    that has already dlopen'd the .so: re-loading the same path reuses the OS
    dlopen mapping, so an in-place rebuild only takes effect in the next process.
    """
    crate_dir = _kernels_crate_dir()
    if crate_dir is None:
        raise RuntimeError(
            "cutile-rs kernels crate not found (expected ops/cutile_rs/cutile_kernels/ with a Cargo.toml)."
        )
    so_path = _shared_so_path(crate_dir)
    if _autobuild_enabled() and _so_stale(so_path):
        _build_kernels(crate_dir)
        _cffi_kernel_libs.clear()
    if not os.path.isfile(so_path):
        raise RuntimeError(
            f"libcutile_kernels.so not found at {so_path}. "
            f"Auto-build is disabled (CUTILE_RS_AUTOBUILD=0) or failed; "
            f"build it with:  cd {crate_dir} && cargo build --release."
        )
    return so_path


# ─── cffi (ABI-mode) loader ─────────────────────────────────────────────────
#
# Lighter, drift-resistant alternative to a hand-written ctypes argtypes list:
# declare the C signature ONCE as an inline cdef in the wrapper (keep in sync
# with the op's ffi.rs). All ops share one libcutile_kernels.so; each op gets
# its own cffi handle (cdef'ing only its symbol) over that shared .so.

_cffi_kernel_libs: dict[str, tuple] = {}


# Generic C-ABI tensor descriptor shared by all cutile-rs ops (the "packer"
# side). MUST stay in sync with `TensorDesc` in ops/cutile_rs/ffi_util.rs
# (#[repr(C)], MAX_DIMS=4, strides in ELEMENTS, dtype: 0=f32/1=f16/2=bf16). It is
# prepended to every op's cdef in bind_kernel_function_cffi, so an op signature
# can just take `const TensorDesc*` args.
_TENSORDESC_MAX_DIMS = 4
_TENSORDESC_CDEF = """
typedef struct {
    uint64_t ptr;
    int32_t  ndim;
    int64_t  shape[4];
    int64_t  strides[4];
    int32_t  dtype;
} TensorDesc;
"""

# torch.dtype -> TensorDesc.dtype code. Keep in sync with ffi_util::dtype_str.
# int32 (code 3) is for integer index tensors (e.g. attention start offsets).
_DTYPE_CODE = {"torch.float32": 0, "torch.float16": 1, "torch.bfloat16": 2, "torch.int32": 3}


def make_tensor_desc(ffi, t):
    """Pack a torch.Tensor into a cffi ``TensorDesc *`` (the generic Python
    "packer"; the Rust "unpacker" is ``ffi_util::borrow_tensor``).

    ``ffi`` must be the FFI() returned by :func:`bind_kernel_function_cffi` (it
    has the shared TensorDesc typedef). The returned cdata must be kept alive by
    the caller until after the FFI call (cffi frees it when GC'd).
    """
    if not t.is_cuda:
        raise ValueError("cutile-rs: make_tensor_desc requires a CUDA tensor (got device " + str(t.device) + ")")
    if t.dim() > _TENSORDESC_MAX_DIMS:
        raise ValueError(f"cutile-rs TensorDesc supports <= {_TENSORDESC_MAX_DIMS} dims, got {t.dim()}")
    code = _DTYPE_CODE.get(str(t.dtype))
    if code is None:
        raise NotImplementedError(f"cutile-rs: dtype {t.dtype} not supported")
    d = ffi.new("TensorDesc *")
    d.ptr = t.data_ptr()
    d.ndim = t.dim()
    for i in range(t.dim()):
        d.shape[i] = int(t.shape[i])
        d.strides[i] = int(t.stride(i))
    d.dtype = code
    return d


def bind_kernel_function_cffi(kernel: str, cdef: str):
    """cffi ABI-mode loader. Returns ``(ffi, lib)``; call ``lib.cutile_<op>(...)``
    (returns a plain int rc — feed to :func:`check_rc`). Loads the shared
    libcutile_kernels.so and cdefs this op's signature. The shared TensorDesc
    typedef is prepended automatically, so ops can take ``const TensorDesc*``
    args and pack tensors via :func:`make_tensor_desc`.

    Usage::
        ffi, lib = bind_kernel_function_cffi("matmul", _FFI_CDEF)
        cd, ad, bd = (make_tensor_desc(ffi, x) for x in (out, a, b))
        rc = lib.cutile_matmul(cd, ad, bd, ...)
        check_rc(rc, "cutile_matmul")
    """
    from cffi import FFI

    if kernel in _cffi_kernel_libs:
        return _cffi_kernel_libs[kernel]
    so_path = _ensure_built_and_path()
    ffi = FFI()
    ffi.cdef(_TENSORDESC_CDEF + cdef)
    lib = ffi.dlopen(so_path)
    _cffi_kernel_libs[kernel] = (ffi, lib)
    logger.info(f"Loaded cutile-rs library (cffi) from {so_path}")
    return ffi, lib


# ─── Common math/device utilities ──────────────────────────────────────────

# Per-device SM-count cache (multi-GPU correct — keyed by device id, not a single
# global that would return device 0's count for every device).
_NUM_SM: dict[int, int] = {}


def get_num_sm() -> int:
    """Cached SM count for the CURRENT CUDA device (per-device cache)."""
    dev = torch.cuda.current_device()
    if dev not in _NUM_SM:
        _NUM_SM[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    return _NUM_SM[dev]


def next_power_of_2(n: int) -> int:
    """Integer next power of 2 (e.g. 1000 → 1024, 1024 → 1024)."""
    if n < 1:
        raise ValueError(f"next_power_of_2: n must be >= 1, got {n}")
    # bit_length is arbitrary-precision (the fixed 32-bit shift chain breaks for n > 2**32).
    return 1 << (n - 1).bit_length()


def check_rc(rc: int, fn_name: str) -> None:
    """Assert FFI return code is 0; raise RuntimeError otherwise."""
    if rc != 0:
        raise RuntimeError(f"{fn_name} returned error code {rc}")
