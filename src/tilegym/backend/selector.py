# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Unified Backend Selector
Used to manage backend implementations of various operations in TileGym library
"""

import functools
import os
from typing import Dict
from typing import Set

from tilegym.logger import get_logger

logger = get_logger(__name__)


def is_nvt_available():
    try:
        import triton.backends.tileir

        tileir_exists = True
    except ImportError:
        tileir_exists = False
    return tileir_exists and int(os.environ.get("ENABLE_TILE", -1)) == 1


try:
    import cuda.tile as ct
    import cuda.tile.tune  # noqa: F401  # required by every op under ops/cutile/

    CUTILE_AVAILABLE = True

except ImportError:
    import warnings

    warnings.warn(
        "Failed to import cuda.tile / cuda.tile.tune, CUDA Tile backend is not available. "
        "To enable it: `pip install cuda-tile` "
        "(see https://github.com/NVIDIA/cutile-python)"
    )
    CUTILE_AVAILABLE = False


def is_cutile_available():
    if os.environ.get("TILEGYM_DISABLE_CUTILE") == "1":
        print("[TileGym] TILEGYM_DISABLE_CUTILE=1; CUDA Tile backend force-disabled")
        return False
    return CUTILE_AVAILABLE


_TILECPP_MIN_NVCC = (13, 3)


def _nvcc_version_supported() -> bool:
    """Return True iff a usable nvcc with a supported CUDA version is found.

    Resolution order: ``$TILECPP_NVCC_PATH`` first, then ``nvcc`` on PATH.
    The release version reported by ``nvcc --version`` must be at least
    ``_TILECPP_MIN_NVCC`` (currently 13.3).
    """
    import re
    import shutil
    import subprocess

    nvcc = os.environ.get("TILECPP_NVCC_PATH", "nvcc")
    if not os.path.isabs(nvcc):
        resolved = shutil.which(nvcc)
        if resolved is None:
            return False
        nvcc = resolved
    elif not os.path.exists(nvcc):
        return False

    try:
        result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    m = re.search(r"release\s+(\d+)\.(\d+)", result.stdout)
    if not m:
        return False
    return (int(m.group(1)), int(m.group(2))) >= _TILECPP_MIN_NVCC


def _check_tilecpp_module_importable():
    """Cheap eager check: can we locate and import the TileCpp _cuda_utils module?

    Does NOT spawn any subprocess, so it is safe to call at module load time
    even on hosts without nvcc / without CUDA. Returns ``(ok, err)`` where
    ``err`` is the captured exception when ``ok`` is False.
    """
    try:
        from importlib import util as importlib_util
        from pathlib import Path

        _tilecpp_cuda_utils_path = Path(__file__).resolve().parents[1] / "ops" / "tilecpp" / "utils" / "_cuda_utils.py"
        _tilecpp_cuda_utils_spec = importlib_util.spec_from_file_location(
            "_tilegym_tilecpp_cuda_utils_availability",
            _tilecpp_cuda_utils_path,
        )
        if _tilecpp_cuda_utils_spec is None or _tilecpp_cuda_utils_spec.loader is None:
            raise ImportError("Failed to locate TileCpp _cuda_utils module")
        _tilecpp_cuda_utils = importlib_util.module_from_spec(_tilecpp_cuda_utils_spec)
        _tilecpp_cuda_utils_spec.loader.exec_module(_tilecpp_cuda_utils)
        if not hasattr(_tilecpp_cuda_utils, "TileCppKernel"):
            raise ImportError("TileCppKernel is not available")
    except (ImportError, FileNotFoundError) as err:
        return False, err
    return True, None


_TILECPP_MODULE_IMPORTABLE, _tilecpp_unavailable_err = _check_tilecpp_module_importable()


@functools.cache
def is_tilecpp_available() -> bool:
    """Check if the CUDA Tile C++ backend is available.

    The expensive ``nvcc --version`` subprocess is deferred to the first call
    of this function (cached thereafter), so ``import tilegym`` on a non-CUDA
    host has no subprocess overhead. The check is invoked by the dispatcher
    on the first actual tilecpp dispatch. When the check fails, a
    ``UserWarning`` is emitted at the caller's frame (``stacklevel=2``) and
    suppressed for subsequent calls.
    """
    import warnings

    if not _TILECPP_MODULE_IMPORTABLE:
        warnings.warn(
            f"TileCpp backend is not available: {_tilecpp_unavailable_err}",
            stacklevel=2,
        )
        return False
    if not _nvcc_version_supported():
        warnings.warn(
            f"TileCpp backend is not available: nvcc >= {_TILECPP_MIN_NVCC[0]}.{_TILECPP_MIN_NVCC[1]} "
            "is required (set TILECPP_NVCC_PATH or install CUDA "
            f"{_TILECPP_MIN_NVCC[0]}.{_TILECPP_MIN_NVCC[1]} or newer on PATH)",
            stacklevel=2,
        )
        return False
    return True


def is_cutile_rs_available() -> bool:
    """Check if the cutile-rs (Rust FFI) backend is available.

    All cutile-rs kernels build into ONE cdylib
    ``ops/cutile_rs/cutile_kernels/target/release/libcutile_kernels.so``
    (crates.io ``cutile`` deps; per-op sources are pure-.rs in
    ``ops/cutile_rs/<op>_kernel/``). No shared cutile-rs checkout, no
    ``CUTILE_RS_DIR``.

    Permissive probe (Rule 35): available if ``cargo`` is on PATH (the wrapper
    builds the crate lazily at dispatch) OR an up-to-date prebuilt
    ``libcutile_kernels.so`` already exists. libclang / CUDA headers / ``tileiras``
    are validated lazily at dispatch, not here.
    """
    import shutil

    from .cutile_rs.utils import _autobuild_enabled
    from .cutile_rs.utils import _kernels_crate_dir
    from .cutile_rs.utils import _shared_so_path
    from .cutile_rs.utils import _so_stale

    crate_dir = _kernels_crate_dir()
    if crate_dir is None:
        return False
    so_path = _shared_so_path(crate_dir)
    if not _autobuild_enabled():
        # Auto-build disabled (CUTILE_RS_AUTOBUILD=0): only a present, pinned .so is usable.
        return os.path.isfile(so_path)
    # Auto-build enabled: buildable if cargo is on PATH; otherwise accept a prebuilt
    # .so only when present AND not stale (it cannot be rebuilt without cargo).
    if shutil.which("cargo") is not None:
        return True
    return not _so_stale(so_path)


_AVAILABLE_BACKENDS: Set[str] = set()
_CURRENT_BACKENDS: str = "cutile"


def _check_backends_availability() -> Dict[str, bool]:
    availability = {
        "cutile": is_cutile_available(),
        "triton": True,
        "tilecpp": _TILECPP_MODULE_IMPORTABLE,
        "cutile-rs": is_cutile_rs_available(),
    }
    return availability


def _initialize_available_backends():
    global _AVAILABLE_BACKENDS
    global _CURRENT_BACKENDS
    backend_availability = _check_backends_availability()

    for backend, available in backend_availability.items():
        if available:
            _AVAILABLE_BACKENDS.add(backend)


def _load_from_environment():
    """CUTILE_TUTORIALS_BACKEND=xxx"""
    global _CURRENT_BACKENDS
    backend = os.environ.get("CUTILE_TUTORIALS_BACKEND", _CURRENT_BACKENDS)
    if backend in _AVAILABLE_BACKENDS:
        _CURRENT_BACKENDS = backend
    else:
        raise ValueError(f"Unknown backend: {backend}, available backends: {_AVAILABLE_BACKENDS}")


def get_available_backends() -> Set[str]:
    return _AVAILABLE_BACKENDS


def get_available_triton_backend() -> str:
    if is_nvt_available():
        return "nvt"
    return "oait"


def get_current_backend() -> str:
    return _CURRENT_BACKENDS


def set_backend(backend: str) -> None:
    """set the backend for ops"""
    global _CURRENT_BACKENDS
    if backend not in _AVAILABLE_BACKENDS:
        raise ValueError(f"Unknown backend: {backend}, available backends: {_AVAILABLE_BACKENDS}")
    # tilecpp is in _AVAILABLE_BACKENDS based on a cheap module-importability
    # check; verify the full runtime requirement (nvcc >= 13.3) here so callers
    # opting in to tilecpp fail fast instead of silently falling back at dispatch.
    if backend == "tilecpp" and not is_tilecpp_available():
        raise ValueError(
            f"Backend 'tilecpp' is not available on this system: nvcc >= "
            f"{_TILECPP_MIN_NVCC[0]}.{_TILECPP_MIN_NVCC[1]} is required "
            "(set TILECPP_NVCC_PATH or install CUDA "
            f"{_TILECPP_MIN_NVCC[0]}.{_TILECPP_MIN_NVCC[1]} or newer on PATH)"
        )
    _CURRENT_BACKENDS = backend
    logger.info(f"Set backend to {backend}")


def is_backend_available(backend: str) -> bool:
    """check if the backend is available"""
    if backend not in _AVAILABLE_BACKENDS:
        return False
    # tilecpp's entry in _AVAILABLE_BACKENDS reflects only the cheap module-
    # importability check; the runtime nvcc>=13.3 requirement is verified
    # lazily here (cached) so test gates like
    # ``if is_backend_available("tilecpp"):`` skip on hosts without nvcc.
    if backend == "tilecpp":
        return is_tilecpp_available()
    return True


def assert_backend_available(backend: str) -> None:
    """assert the backend is available"""
    if not is_backend_available(backend):
        raise ValueError(f"Backend {backend} is not available, available backends: {_AVAILABLE_BACKENDS}")


_initialize_available_backends()
_load_from_environment()
