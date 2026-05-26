<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->




# CUDA Tile C++ Backend

The CUDA Tile C++ backend provides CUDA Tile C++ kernel implementations for TileGym operations.

## Set up

CUDA Tile C++ requires CUDA Toolkit 13.3 or newer. Install the latest CUDA Toolkit
available for your platform, and make sure `nvcc` from that toolkit is on
your `PATH`.

```
# Example: use a CUDA 13.3+ toolkit installed under /usr/local.
export PATH=/usr/local/cuda-13.3/bin:$PATH
export TILECPP_NVCC_PATH=/usr/local/cuda-13.3/bin/nvcc

# Verify nvcc is visible.
nvcc --version

# Run a test, you should see a CUDA Tile C++ (TileCpp) column in the report table
python tests/benchmark/bench_swiglu.py
```

## Environment Variables

### Cache Configuration


| Variable                | Default            | Description                                                                                                                 |
| ----------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| `TILECPP_CACHE_DIR`     | `~/.cache/tilecpp` | Directory for caching compiled cubin files. If not set, uses `$XDG_CACHE_HOME/tilecpp` or falls back to `~/.cache/tilecpp`. |
| `TILECPP_DISABLE_CACHE` | `0`                | Set to `1` to disable cubin caching and force recompilation on every run. Useful for development/debugging.                 |


### Compiler Configuration


| Variable            | Default | Description                                                                                                        |
| ------------------- | ------- | ------------------------------------------------------------------------------------------------------------------ |
| `TILECPP_NVCC_PATH` | `nvcc`  | Path to the nvcc compiler. Override if nvcc is not in your PATH or you want to use a specific version.             |
| `TILECPP_SAVE_SRC`  | `0`     | Set to `1` to save generated CUDA source files alongside compiled cubins. Useful for debugging compilation issues. |


### Autotuning


| Variable                   | Default | Description                                                                                             |
| -------------------------- | ------- | ------------------------------------------------------------------------------------------------------- |
| `TILECPP_AUTOTUNE`         | `0`     | Set to `1` to enable autotuning for kernel configurations. When disabled, uses default configurations.  |
| `TILECPP_VERBOSE_AUTOTUNE` | `0`     | Set to `1` to enable verbose output during autotuning, showing configuration trials and timing results. |


## Adding a New CUDA Tile C++ Kernel to TileGym

This section is only about integrating a CUDA Tile C++ kernel into TileGym.

CUDA Tile C++ operators normally have two pieces:

1. A CUDA Tile C++ kernel in `src/tilegym/ops/tilecpp/<op>.cuh`.
2. A Python binding in `src/tilegym/ops/tilecpp/<op>.py` that compiles, launches,
  and registers the kernel with TileGym.

The `.cuh` file contains the `__tile_global__` kernel and any helper tile code.
Prefer making compile-time constants template parameters when they affect tile
shapes or loop structure. Keep the kernel signature limited to runtime pointers
and scalar values that must be passed at launch time.

```cpp
#pragma once

#include <cuda_tile.h>

template<typename T, int BLOCK_M, int BLOCK_N>
__tile_global__ void my_kernel(const T* __restrict__ x, T* __restrict__ y, int n) {
    namespace ct = cuda::tiles;
    // Tile code goes here.
}
```

The Python file creates a `TileCppKernel`, requests a specialized kernel with
`get_kernel(...)`, launches it with device pointers/scalars, and registers the
public TileGym op for the `tilecpp` backend.

```python
from pathlib import Path

import numpy as np
import torch

from tilegym.backend import register_impl
from tilegym.ops.tilecpp.utils._cuda_utils import TileCppKernel

_my_kernel = TileCppKernel(
    source_path=Path(__file__).parent / "my_op.cuh",
    kernel_name="my_kernel",
)


def _launch_my_kernel(x: torch.Tensor, y: torch.Tensor, block_m: int, block_n: int):
    kernel, _, _ = _my_kernel.get_kernel(
        dtype=x.dtype,
        template_params=[block_m, block_n],
        signature="const {T}*, {T}*, int",
    )
    _my_kernel.launch(
        grid=(1, 1, 1),
        kernel=kernel,
        args=[
            np.uint64(x.data_ptr()),
            np.uint64(y.data_ptr()),
            np.int32(x.numel()),
        ],
    )


@register_impl("my_op", backend="tilecpp")
def my_op(x: torch.Tensor, **kwargs):
    y = torch.empty_like(x)
    _launch_my_kernel(x, y, block_m=128, block_n=128)
    return y
```

Make sure `src/tilegym/ops/tilecpp/__init__.py` imports the new Python module
when the backend is available. Add or extend tests under `tests/ops/` so the
same operation can run with `backend="tilecpp"`, and add benchmark coverage
under `tests/benchmark/` when there is a corresponding CuTile benchmark.

## Compiling a `.cuh` Kernel Standalone with nvcc 13.3+

You can compile a CUDA Tile C++ `.cuh` kernel directly with the CUDA 13.3+ toolkit
without going through TileGym. This is useful for verifying a kernel builds
cleanly outside the framework or sharing a self-contained reproducer.

You need one extra `.cu` driver file that:

1. Includes the `.cuh` so the template is in scope.
2. Adds at least one **explicit template instantiation**.
3. Provides host-side setup: device buffers, `cudaMemcpy`, the kernel
  launch, and copy-back/cleanup.

Example driver (`my_op_main.cu`) for the `my_kernel` template shown earlier:

```cpp
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#include "my_op.cuh"

template __tile_global__ void my_kernel<float, 128, 128>(
    const float* __restrict__, float* __restrict__, int);

int main() {
    constexpr int N = 1 << 20;
    std::vector<float> h_x(N, 1.0f), h_y(N);

    float *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    /* Tile C++ kernels are tile-centric: the launch always uses
     * block=1, and the kernel uses ct::bid() for parallelism.  The
     * grid covers ceil(N / BLOCK_SIZE) tiles. */
    dim3 grid((N + 127) / 128), block(1);
    my_kernel<float, 128, 128><<<grid, block>>>(d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("y[0] = %f\n", h_y[0]);

    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
```

Compile with nvcc 13.3 or newer. Set `-arch` to match your target GPU
(`sm_80` and newer architectures are supported):

```bash
/usr/local/cuda-13.3/bin/nvcc \
    -enable-tile \
    -std=c++20 \
    -arch=sm_100 \
    -I src/tilegym/ops/tilecpp \
    my_op_main.cu \
    -o my_op_main

./my_op_main
```

The `-enable-tile` flag turns on the Tile C++ extensions (`__tile_global__`,
the `cuda::tiles` namespace, etc.); without it nvcc treats the `.cuh` as
plain CUDA and rejects the tile syntax.

The same toolchain can produce a cubin-only artifact (the form TileGym caches
internally) by adding `-tilecubin --tile-only` and dropping the host driver
code from the `.cu` file.

## Cache Management

The CUDA Tile C++ cache stores compiled cubin files to avoid recompilation. Cache files are named using a hash of the source code and template parameters.

To clear the cache:

```bash
rm -rf ~/.cache/tilecpp/*
```
