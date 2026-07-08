# Environment Block for Agent Prompts

## Required env vars

Everything now lives in the tilegym checkout — there is no separate cutile-rs
checkout and no `CUTILE_RS_ROOT`. The kernel Rust and the aggregated
`cutile_kernels` crate live under `$TILEGYM_PATH/src/tilegym/ops/cutile_rs/`,
build against crates.io pins (`cutile="=0.2.0"`, …; Rule 33), and produce a
single `libcutile_kernels.so`. Set these once per shell session;
`scripts/preflight.sh` verifies them.

```bash
# ─── Git repo ─────────────────────────────────────────────────────────────
export TILEGYM_PATH=<your-tilegym-checkout>     # tests + Python wrappers + cutile_rs Rust

# ─── Toolchain (existence-checked) ────────────────────────────────────────
export TILEIRAS_BIN=<path-to-tileiras>          # cuTile compiler binary (IR->cubin at launch)
export CUDA_TILE_OPT_BIN=<path-to-cuda-tile-opt> # MLIR canonicalizer (Agent B/C)
export TRITON_TILEIR_PYTHONPATH=<triton-tileir/python>  # Triton-TileIR backend bindings
export CUDA_TOOLKIT_PATH=/usr/local/cuda        # CUDA root (cuda-bindings build.rs); default /usr/local/cuda
```

> **Building the `cutile_kernels` crate.** The crate uses PINNED crates.io
> dependencies (no path deps), so there is nothing to sed-replace. The Python
> loader autobuilds it on first use — `CUTILE_RS_AUTOBUILD` is ON by default;
> set it to `0` to use a prebuilt library. Override the crate directory (default
> `$TILEGYM_PATH/src/tilegym/ops/cutile_rs/cutile_kernels`) with
> `CUTILE_RS_KERNELS_DIR`. The `cuda-bindings` build step reads
> `CUDA_TOOLKIT_PATH` (default `/usr/local/cuda`); `tileiras` lowers IR to cubin
> at launch time.

## CRITICAL: tileiras alignment

cutile-python and cutile-rs MUST use the SAME `tileiras`. The binary pointed at
by `$TILEIRAS_BIN` must come BEFORE `/usr/local/cuda/bin/` on PATH so cuTile-py
JIT picks it up.

## Full manual environment

```bash
# Required env vars (incl. CUDA_TOOLKIT_PATH) — see top of file.

# tileiras must precede system CUDA on PATH
export PATH=$(dirname "$TILEIRAS_BIN"):$PATH

# Triton-TileIR bindings + tilegym source on PYTHONPATH
export PYTHONPATH=$TRITON_TILEIR_PYTHONPATH:$TILEGYM_PATH/src:$PYTHONPATH

export ENABLE_TILE=1
export TILEIR_ENABLE_FTZ=1
export TILEIR_ENABLE_APPROX=1

# Benchmark defaults
export CUPTI=1
export WARMUP=100
export REP=50
export MIN_REP=2

# CUDA_TOOLKIT_PATH is one of the required vars set at the top of this file.
```

## Tool paths

```bash
# tileiras (for cutile-rs JIT → cubin, and cutile-python compilation)
which tileiras   # MUST be: $TILEIRAS_BIN
                 # NOT: /usr/local/cuda/bin/tileiras

# cuda-tile-opt (for canonicalize/CSE on dumped IR — used by Agent B / Agent C)
echo "$CUDA_TILE_OPT_BIN"
```

## Verification

```bash
which tileiras
# Expected: $TILEIRAS_BIN

python -c "import triton; print(triton.__file__)"
# Expected: contains $TRITON_TILEIR_PYTHONPATH

python -c "import tilegym; print('OK')"
# Expected: OK

cd "$TILEGYM_PATH/src/tilegym/ops/cutile_rs/cutile_kernels" && cargo build --release 2>&1 | tail -1
# Expected: Finished `release` profile ...   (produces libcutile_kernels.so)
```
