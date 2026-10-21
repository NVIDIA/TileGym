<!--- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# Agentic cuTile paged-KV decode optimization on B200

Run your own agentic cuTile optimization experiment on a B200 and collect baseline and optimized performance data. The bundled `solution/` is the best correct implementation from our Codex run, supplied so you can reproduce the reported results. Remove it from your experiment workspace before starting your own agent run.

- `example/`: original kernel, five pytest tests, and six-workload benchmark.
- `solution/`: supplied implementation and reproduction instructions in [solution/README.md](solution/README.md).
- `prompt.txt`: optimization task with a 90-minute budget.
- `scripts/validate_extended.py`: six benchmark-shape numerical checks and 28 additional interface/numerical checks.

## Environment and setup

NVIDIA B200; host driver 570.195.03, container CUDA compatibility library 615.65.02. Container image: `nvcr.io/nvidia/pytorch:26.08-py3`. Python 3.12.3, PyTorch 2.14.0 (CUDA 13.0), cuda-tile 1.6.0, nvidia-cuda-tileiras 13.4.92, nvidia-cudnn-cu13 9.27.0.42.

Inside a GPU-enabled container:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.14.0 cuda-tile==1.6.0 pytest==9.1.1
pip install --no-deps nvidia-cuda-tileiras==13.4.92 nvidia-cudnn-cu13==9.27.0.42
```

## Run your own experiment

Start from a fresh checkout containing the original `example/decode_cutile.py`. From this package directory, remove the supplied solution before starting:

```bash
rm -rf solution
mkdir results
```

`solution/` contains the optimized kernel, reproduction instructions, and a recorded performance table. Delete the whole directory before asking the agent to optimize.

To provide the optimization wiki/skills, clone NVIDIA/TileGym separately and copy its `wiki/` and `skills/` into your workspace's `example/` directory. For Claude Code discovery, create `example/.claude/skills` pointing to `../skills`; configure the equivalent skill discovery in other harnesses.

### 1. Run the optimization agent

Start your coding agent in `example/` with access to the B200 and the environment above. Submit the task in `../prompt.txt`: optimize `decode_cutile.py`, preserve its interface and numerical tolerances, validate and benchmark each candidate, and retain the best correct implementation within the 90-minute budget. The `/goal` prefix belongs to the original harness; use your agent's task submission mechanism if it does not support that command.

Ask the agent to save its initial baseline with `python bench_decode.py --json ../results/baseline.json` before changing the kernel. Keep the tests and benchmark unchanged.

### 2. Validate and collect final data

After the agent finishes, run from `example/`:

```bash
python -m pytest test_decode.py
python ../scripts/validate_extended.py
python bench_decode.py --json ../results/final.json
cp decode_cutile.py ../results/optimized_decode_cutile.py
```

Check that both validation commands pass before comparing performance. The extended script checks the six benchmark shapes and 28 additional interface/numerical cases. Record your GPU, driver, container image, package versions, agent/model, and optimization budget alongside the results.

Match rows by `case` in `baseline.json` and `final.json`. Speedup over the original is `baseline.cutile_ms / final.cutile_ms`; speed relative to SDPA is `final.ref_ms / final.cutile_ms`. Use the geometric mean of the six original-to-final ratios to summarize the workload set. These newly generated files contain your experiment's measurements.
