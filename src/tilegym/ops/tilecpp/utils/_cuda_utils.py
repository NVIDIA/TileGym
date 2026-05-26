# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""
Common utilities for CUDA Tile C++ kernel compilation and execution.

This module provides:
- Data type mapping between PyTorch and CUDA C++
- Scalar value conversion for kernel arguments
- CUDA source compilation to cubin via nvcc
- Kernel caching and management
- Kernel launch utilities
"""

import fcntl
import hashlib
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from cuda.bindings import driver
from cuda.core import Device
from cuda.core import LaunchConfig
from cuda.core import ObjectCode
from cuda.core import Stream
from cuda.core import launch

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Path to CUDA Tile C++ headers (.cuh files)
TILECPP_HEADER_PATH = Path(__file__).parent.parent

# Path to nvcc compiler
NVCC_PATH = os.environ.get("TILECPP_NVCC_PATH", "nvcc")


def _get_current_device_arch() -> int:
    """Return the current CUDA device architecture as an integer, e.g. 100 or 103."""
    device_id = torch.cuda.current_device()
    dev = Device(device_id)
    dev.set_current()
    major, minor = dev.compute_capability
    return major * 10 + minor


def _get_current_device_arch_flag() -> str:
    """Return the nvcc architecture flag value for the current CUDA device."""
    return f"sm_{_get_current_device_arch()}"


# Cache directory for compiled cubins
# Use TILECPP_CACHE_DIR env var if set, otherwise use ~/.cache/tilecpp (matching XDG convention)
def _get_cache_dir() -> Path:
    env_dir = os.environ.get("TILECPP_CACHE_DIR", "")
    if env_dir:
        cache_dir = Path(env_dir)
    else:
        # Use XDG cache dir or fallback to ~/.cache
        xdg_cache = os.environ.get("XDG_CACHE_HOME", "")
        if xdg_cache:
            cache_dir = Path(xdg_cache) / "tilecpp"
        else:
            cache_dir = Path.home() / ".cache" / "tilecpp"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


CACHE_DIR = _get_cache_dir()


class _FileLock:
    """Cross-process file lock using fcntl.

    Used to serialize cubin compilation so that in a multiprocessing
    environment (e.g. tensor-parallel workers) only one process compiles a
    given kernel while others wait and reuse the result.

    The lock file lives next to the cubin (``<cubin_path>.lock``).  The lock
    is *advisory* — it only coordinates processes that use this class.
    """

    def __init__(self, path: Path, timeout: float = 600):
        self._lock_path = Path(str(path) + ".lock")
        self._timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> "_FileLock":
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = -1
                    raise TimeoutError(f"Timed out waiting for lock: {self._lock_path}")
                time.sleep(0.1)

    def __exit__(self, *exc):
        if self._fd >= 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically via a temp file + rename.

    This ensures that concurrent readers never see a partially-written cubin.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, data)
        os.close(fd)
        fd = -1
        os.replace(tmp, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# Global kernel cache: cache_key -> (cubin_path, mangled_name, scalar_converter)
# We cache the path and name, not the kernel object, because kernel objects are tied to contexts
_global_kernel_cache: Dict[str, Tuple[Path, str, Callable]] = {}


def is_cache_disabled() -> bool:
    """Check if cubin caching is disabled via TILECPP_DISABLE_CACHE."""
    return os.environ.get("TILECPP_DISABLE_CACHE", "0") != "0"


def is_verbose_autotune() -> bool:
    """Check if verbose output is enabled via TILECPP_VERBOSE_AUTOTUNE."""
    return os.environ.get("TILECPP_VERBOSE_AUTOTUNE", "0") != "0"


def should_save_source() -> bool:
    """Check if source saving is enabled via TILECPP_SAVE_SRC."""
    return os.environ.get("TILECPP_SAVE_SRC", "0") != "0"


# =============================================================================
# Data Type Handling
# =============================================================================


def _float_to_bfloat16_bits(val: float) -> np.uint16:
    """Convert float to bfloat16 bit representation via float32 with rounding.

    bfloat16 is the upper 16 bits of float32, with round-to-nearest-even.
    This matches torch's bfloat16 conversion exactly.
    """
    f32_bits = np.float32(val).view(np.uint32)

    # Round to nearest even (banker's rounding)
    round_bit = (f32_bits >> 16) & 1
    sticky_bits = f32_bits & 0xFFFF
    bf16_bits = f32_bits >> 16

    # Round up if: sticky > 0.5, or sticky == 0.5 and round bit is 1 (odd)
    if sticky_bits > 0x8000 or (sticky_bits == 0x8000 and round_bit):
        bf16_bits += 1

    return np.uint16(bf16_bits)


def _float_to_half_bits(val: float) -> np.uint16:
    """Convert float to float16 bit representation."""
    return np.float16(val).view(np.uint16)


# Mapping from torch dtype to (C++ type name, scalar conversion function)
# The conversion function takes a Python scalar and returns the appropriate numpy type
DTYPE_MAP: Dict[torch.dtype, Tuple[str, Callable]] = {
    torch.float32: ("float", lambda x: np.float32(x)),
    torch.float64: ("double", lambda x: np.float64(x)),
    torch.float16: ("__half", _float_to_half_bits),
    torch.bfloat16: ("__nv_bfloat16", _float_to_bfloat16_bits),
    torch.int32: ("int", lambda x: np.int32(x)),
    torch.int64: ("long long", lambda x: np.int64(x)),
    torch.float8_e4m3fn: ("__nv_fp8_e4m3", lambda x: np.uint8(x)),
    torch.float8_e5m2: ("__nv_fp8_e5m2", lambda x: np.uint8(x)),
}


def get_dtype_info(dtype: torch.dtype) -> Tuple[str, Callable]:
    """Get C++ type name and scalar conversion function for a torch dtype.

    Args:
        dtype: PyTorch data type

    Returns:
        Tuple of (cpp_type_name, scalar_converter_function)

    Raises:
        ValueError: If dtype is not supported
    """
    if dtype not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype}. Supported: {list(DTYPE_MAP.keys())}")
    return DTYPE_MAP[dtype]


def get_cpp_type(dtype: torch.dtype) -> str:
    """Get C++ type name for a torch dtype."""
    return get_dtype_info(dtype)[0]


def get_scalar_converter(dtype: torch.dtype) -> Callable:
    """Get scalar conversion function for a torch dtype."""
    return get_dtype_info(dtype)[1]


# =============================================================================
# CUDA Compilation
# =============================================================================


def get_source_hash(source: str) -> str:
    """Get hash of source code for caching."""
    return hashlib.md5(source.encode()).hexdigest()[:16]


def compile_cuda_to_cubin(
    header_path: Path,
    output_path: Path,
    template_instantiations: Optional[List[str]] = None,
    include_paths: Optional[List[Path]] = None,
) -> Path:
    """
    Compile CUDA header to a cubin file using nvcc.

    Generates a temporary .cu file that includes the header and adds
    explicit template instantiations.

    Args:
        header_path: Path to the .cuh header file
        output_path: Path to write the cubin file
        template_instantiations: List of explicit template instantiation statements
        include_paths: Additional include paths for compilation

    Returns:
        Path to the compiled cubin file

    Raises:
        RuntimeError: If compilation fails
    """
    # Generate wrapper .cu file content that includes the header
    header_name = header_path.name
    wrapper_source = f"// Auto-generated wrapper for {header_name}\n"
    wrapper_source += f'#include "{header_name}"\n\n'

    if template_instantiations:
        wrapper_source += "// Explicit template instantiations\n"
        for inst in template_instantiations:
            wrapper_source += f"{inst}\n"

    # Write wrapper .cu file using the same base name as the cubin
    cu_file = output_path.with_suffix(".cu")
    with open(cu_file, "w") as f:
        f.write(wrapper_source)

    try:
        # Build nvcc command
        gpu_arch = _get_current_device_arch_flag()
        cmd = [
            NVCC_PATH,
            "-tilecubin",
            f"-arch={gpu_arch}",
            "-std=c++20",
            "--tile-only",
            "-o",
            str(output_path),
        ]

        # Add include paths - include tilecpp header directory for .cuh files
        all_include_paths = list(include_paths or [])
        if TILECPP_HEADER_PATH not in all_include_paths:
            all_include_paths.insert(0, TILECPP_HEADER_PATH)

        for inc_path in all_include_paths:
            cmd.extend(["-I", str(inc_path)])

        cmd.append(str(cu_file))

        logger.debug(f"Compiling CUDA kernel: {' '.join(cmd)}")
        # Save source file alongside cubin if TILECPP_SAVE_SRC is set
        if should_save_source():
            source_output_path = output_path.with_suffix(".cu")
            source_output_path.write_text(wrapper_source)
            logger.debug(f"Saved source to: {source_output_path}")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        if result.stderr:
            logger.debug(f"nvcc stderr: {result.stderr}")

        logger.debug(f"Compiled cubin to: {output_path}")

        return output_path

    except subprocess.CalledProcessError as e:
        logger.error(f"nvcc compilation failed:\nstdout: {e.stdout}\nstderr: {e.stderr}")
        raise RuntimeError(f"CUDA compilation failed: {e.stderr}") from e

    finally:
        # Clean up temporary .cu file (keep if TILECPP_SAVE_SRC is set)
        if not should_save_source():
            cu_file.unlink(missing_ok=True)


def get_kernel_name_from_cubin(cubin_bytes: bytes, expected_kernel_name: str, cubin_path: str = None) -> str:
    """Extract the kernel name from a cubin file.

    Args:
        cubin_bytes: Raw bytes of the cubin file
        expected_kernel_name: Base name of the kernel to find (will match mangled names containing this)
        cubin_path: Optional path to the cubin file (for error messages)

    Returns:
        The mangled kernel name found in the cubin

    Raises:
        RuntimeError: If no functions are found or module loading fails
    """
    err, module = driver.cuModuleLoadData(cubin_bytes)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to load module: {err}, cubin path: {cubin_path}")

    err, func_count = driver.cuModuleGetFunctionCount(module)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to get function count: {err}")

    if func_count == 0:
        raise RuntimeError("No functions found in cubin")

    err, functions = driver.cuModuleEnumerateFunctions(func_count, module)
    if err != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Failed to enumerate functions: {err}")

    # Find the function with the expected name
    for func in functions:
        err, kernel_name_bytes = driver.cuFuncGetName(func)
        if err != driver.CUresult.CUDA_SUCCESS:
            continue
        kernel_name = kernel_name_bytes.decode() if isinstance(kernel_name_bytes, bytes) else kernel_name_bytes
        if expected_kernel_name in kernel_name:
            return kernel_name

    # Return first function if no match found
    err, kernel_name_bytes = driver.cuFuncGetName(functions[0])
    kernel_name = kernel_name_bytes.decode() if isinstance(kernel_name_bytes, bytes) else kernel_name_bytes
    return kernel_name


# =============================================================================
# Kernel Management
# =============================================================================


class TileCppKernel:
    """
    Manages a CUDA Tile C++ kernel including compilation, caching, and launching.

    Example usage:
        # Define the kernel
        kernel_def = TileCppKernel(
            source_path=Path(__file__).parent / "add.cuh",
            kernel_name="add_kernel",
            template_params=["float", 1024],  # Type and block size
            signature="float*, float*, float*, int, float",
        )

        # Get compiled kernel for a specific dtype
        kernel = kernel_def.get_kernel(torch.float32)

        # Launch the kernel
        kernel_def.launch(
            grid=(num_blocks,),
            args=[x_ptr, y_ptr, out_ptr, n_elements, alpha],
            dtype=torch.float32,
        )
    """

    def __init__(
        self,
        source_path: Path,
        kernel_name: str,
        include_paths: Optional[List[Path]] = None,
    ):
        """
        Initialize a CUDA Tile C++ kernel definition.

        Args:
            source_path: Path to the .cuh header file
            kernel_name: Name of the kernel function (base name, before template parameters)
            include_paths: Additional include paths for compilation
        """
        self.source_path = source_path
        self.kernel_name = kernel_name
        self.include_paths = include_paths or []
        self._source_hash_cache: Optional[str] = None

    @property
    def source_hash(self) -> str:
        """Get hash of the header source for cache invalidation."""
        if self._source_hash_cache is None:
            source_content = self.source_path.read_text()
            self._source_hash_cache = get_source_hash(source_content)
        return self._source_hash_cache

    def _make_cache_key(self, template_key: str, device_id: int = None) -> str:
        """Create a cache key for the kernel, scoped to the current device."""
        if device_id is None:
            device_id = torch.cuda.current_device()
        return f"{self.kernel_name}_{template_key}_dev{device_id}"

    def _make_cubin_path(self, template_key: str, arch: int, source_hash: str) -> Path:
        """Create a deterministic path for the cached cubin file.

        The path contains no PID, so cubins are shared across processes and
        persist across program restarts.  File locking ensures safe concurrent
        compilation.
        """
        safe_key = template_key.replace(" ", "_").replace(",", "_")
        basename = f"{self.kernel_name}_{safe_key}_{arch}_{source_hash}"
        return CACHE_DIR / f"{basename}.cubin"

    def _compile_kernel(
        self,
        cubin_path: Path,
        all_params: List[str],
        signature: str,
        cpp_type: str,
    ) -> bytes:
        """Compile the kernel and atomically write the cubin to *cubin_path*.

        Returns the cubin bytes.  Uses a temporary output path so that nvcc
        never writes directly to the final location — this prevents other
        processes from reading a partially-written file.
        """
        template_args = ", ".join(all_params)
        sig_with_type = signature.replace("{T}", cpp_type)
        template_inst = f"template __tile_global__ void {self.kernel_name}<{template_args}>({sig_with_type});"

        tmp_cubin = cubin_path.with_suffix(".tmp.cubin")
        try:
            compile_cuda_to_cubin(
                self.source_path,
                tmp_cubin,
                template_instantiations=[template_inst],
                include_paths=self.include_paths,
            )
            cubin_bytes = tmp_cubin.read_bytes()
            _atomic_write_bytes(cubin_path, cubin_bytes)
        finally:
            tmp_cubin.unlink(missing_ok=True)

        return cubin_bytes

    def get_kernel(
        self,
        dtype: torch.dtype,
        template_params: List[Union[int, str]],
        signature: str,
    ) -> Tuple[Any, str, Callable]:
        """
        Get or compile a kernel for the specified dtype and template parameters.

        Args:
            dtype: PyTorch data type for the kernel
            template_params: List of template parameters (excluding type, which is derived from dtype)
            signature: Function signature for explicit instantiation (with {T} placeholder for type)

        Returns:
            Tuple of (kernel, mangled_name, scalar_converter)
        """
        cpp_type, scalar_converter = get_dtype_info(dtype)

        # Build template key for caching
        # Handle boolean parameters specially (Python True/False -> C++ true/false)
        def to_cpp_param(p):
            if isinstance(p, bool):
                return "true" if p else "false"
            return str(p)

        all_params = [cpp_type] + [to_cpp_param(p) for p in template_params]
        template_key = "_".join(all_params)

        # Get device architecture
        device_id = torch.cuda.current_device()
        arch = _get_current_device_arch()

        # Cache key includes device_id since kernel handles are context-specific
        cache_key = self._make_cache_key(template_key, device_id)

        # Check if caching is disabled
        cache_disabled = is_cache_disabled()
        verbose = is_verbose_autotune()

        # Get current CUDA context handle to validate cache
        err, cur_ctx = driver.cuCtxGetCurrent()

        # Check cache for cubin bytes, mangled name, and kernel object.
        # Kernel handles are tied to a specific CUDA context, so we must
        # also validate the context matches before reusing a cached handle.
        # cubin_bytes are cached in memory to avoid repeated file I/O.
        cache_hit = False
        cached_cubin_bytes = None
        cached_mangled_name = None
        if not cache_disabled and cache_key in _global_kernel_cache:
            cached_ctx, cached_cubin_bytes, cached_mangled_name, cached_converter, kernel = _global_kernel_cache[
                cache_key
            ]
            if cached_ctx == cur_ctx:
                cache_hit = True
                mangled_name = cached_mangled_name
                if verbose:
                    logger.info(f"[TileCpp] Using cached kernel: {cache_key}")
            else:
                if verbose:
                    logger.info(f"[TileCpp] Context changed, reloading kernel: {cache_key}")

        if not cache_hit:
            if cached_cubin_bytes is not None:
                # Context changed but cubin bytes are in memory — just reload the kernel
                cubin_bytes = cached_cubin_bytes
                mangled_name = cached_mangled_name
                if verbose:
                    logger.info(f"[TileCpp] Reloading kernel from in-memory cubin for new context: {cache_key}")
            else:
                source_hash = get_source_hash(self.source_hash + template_key)
                cubin_path = self._make_cubin_path(template_key, arch, source_hash)

                if cache_disabled:
                    if verbose:
                        logger.info(f"[TileCpp] Cache disabled, recompiling: {cache_key}")
                    cubin_bytes = self._compile_kernel(cubin_path, all_params, signature, cpp_type)
                else:
                    # Use a file lock so only one process compiles a given
                    # kernel while others wait and reuse the result.
                    with _FileLock(cubin_path):
                        if cubin_path.exists():
                            if verbose:
                                logger.info(f"[TileCpp] Using on-disk cached cubin: {cache_key}")
                            cubin_bytes = cubin_path.read_bytes()
                        else:
                            if verbose:
                                logger.info(f"[TileCpp] Compiling kernel: {cache_key}")
                            cubin_bytes = self._compile_kernel(cubin_path, all_params, signature, cpp_type)

                mangled_name = get_kernel_name_from_cubin(cubin_bytes, self.kernel_name, cubin_path=str(cubin_path))

            # Load kernel from in-memory cubin bytes
            mod = ObjectCode.from_cubin(cubin_bytes)
            kernel = mod.get_kernel(mangled_name)

            # Cache cubin bytes in memory (not file path) so reloads avoid file I/O
            if not cache_disabled:
                _global_kernel_cache[cache_key] = (cur_ctx, cubin_bytes, mangled_name, scalar_converter, kernel)

        return kernel, mangled_name, scalar_converter

    def launch(
        self,
        grid: Union[int, Tuple[int, ...], List[int]],
        kernel: Any,
        args: List[Any],
        stream: Optional[Any] = None,
    ):
        """
        Launch the kernel with the given arguments.

        Args:
            grid: Grid dimensions (number of blocks)
            kernel: The compiled kernel object
            args: List of kernel arguments (should be numpy types or uint64 pointers)
            stream: Optional CUDA stream (uses current torch stream if not provided)
        """
        # Get fresh device for this thread/context
        device_id = torch.cuda.current_device()
        dev = Device(device_id)
        dev.set_current()

        if stream is None:
            # Use the current PyTorch CUDA stream for compatibility with CUDA graphs
            # and proper stream ordering with PyTorch operations
            torch_stream = torch.cuda.current_stream()
            stream = Stream.from_handle(torch_stream.cuda_stream)

        # Tile kernels use block=1
        config = LaunchConfig(grid=grid, block=1)

        launch(stream, config, kernel, *args)


def make_kernel_args(*tensors_and_scalars: Union[torch.Tensor, Tuple[Any, torch.dtype]]) -> List:
    """
    Convert tensors and scalars to kernel arguments.

    For tensors: extracts data_ptr as np.uint64
    For scalars: tuple of (value, dtype) - converts using dtype's scalar converter
    For numpy types: passes through as-is

    Args:
        *tensors_and_scalars: Mix of:
            - torch.Tensor: converted to np.uint64(data_ptr())
            - (scalar, dtype): converted using dtype's scalar converter
            - numpy scalar: passed through

    Returns:
        List of kernel arguments ready for launch()
    """
    args = []
    for item in tensors_and_scalars:
        if isinstance(item, torch.Tensor):
            args.append(np.uint64(item.data_ptr()))
        elif isinstance(item, tuple) and len(item) == 2:
            value, dtype = item
            if isinstance(dtype, torch.dtype):
                converter = get_scalar_converter(dtype)
                args.append(converter(value))
            else:
                args.append(item)
        elif isinstance(item, (np.integer, np.floating, np.uint16, np.int32, np.int64, np.float32, np.float64)):
            args.append(item)
        else:
            # Assume it's already in the right format
            args.append(item)
    return args
