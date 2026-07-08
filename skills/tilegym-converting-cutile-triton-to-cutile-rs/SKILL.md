---
name: tilegym-converting-cutile-triton-to-cutile-rs
description: |
  Use this skill to convert, port, or translate Triton-TileIR (nvtriton) or cuTile-Python GPU kernels to cutile-rs (Rust). The orchestrator runs scripts/preflight.sh, then drives a bounded Agent A -> B -> D -> E pipeline (Agent C is diagnostic, Agent F optional), delegating all kernel/host/correctness/perf work to sub-agents and routing by each stage's single-line VERDICT.
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "TileGym Team <TileGym@nvidia.com>"
---

# Converting Triton-TileIR (nvtriton) / cuTile-Python to cutile-rs

## Instructions

Hard safety rule for every participant: never emit a Bash command containing the destructive recursive-delete token spelled as `r` + `m` + space + dash-r-f. Use `mkdir -p X && find X -mindepth 1 -delete 2>/dev/null` for cleanup.

Use this skill when a user asks to add a `cutile-rs` backend to a tilegym kernel or to port a Triton-TileIR/cuTile-Python kernel to cutile-rs. The orchestrator is a router, not a worker: it delegates all kernel writing, compiling, correctness, perf, and IR diff work to sub-agents.

## First Tool Calls

Copy this checklist into TodoWrite immediately:

```
[ ] 1. Bash: cd "$TILEGYM_PATH" && bash .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/scripts/preflight.sh
[ ] 2. Agent: spawn Agent A with the minimal-pointer template below
[ ] 3. Check Agent A's <VALIDATOR_OUTPUT> block and final VERDICT line
[ ] 4. Route by the table in this file
[ ] 5. Continue happy path B -> D -> E; spawn C ONLY after D **FAIL** or E **INVESTIGATE/BLOCKED**. **E `DONE` is terminal → PIPELINE_COMPLETE; never spawn C/D(2)/F after E DONE** (geomean ≤ 1.05 is the authoritative gate — do NOT reinterpret "≤ baseline" / a (1.0,1.05] geomean / an info-only slow row as a reason to diagnose). For D **BLOCKED**, route by `block_class` in correctness.md **exactly as the Routing Table below**: `host` (wiring / ffi.rs / .so / wrapper / backend registration) -> Agent D(2) (D owns registration; C and B cannot fix it), `kernel` -> Agent C, `env` -> STOP
[ ] 6. Run scripts/validate_kernel.sh {kernel_name} only after the pipeline has produced A/B/C-if-run/D/E artifacts
```

The orchestrator does not read `agents/agent_*.md`. It copies the stage-specific Step 0 file set below into the sub-agent prompt and passes concrete prior artifact paths. Do not add a generic common reference preload: Agent B needs conversion docs; Agent D/E usually do not.

## Agent Step 0 File Sets

**Path convention:** all skill-internal paths in this skill are written relative
to the tilegym root (`$TILEGYM_PATH`, e.g. `/workspace/tilegym`), which is the
assumed current working directory for every agent. So `.agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/...`
resolves from `$TILEGYM_PATH`. Read/cite skill files with these relative paths.
Any bash snippet that invokes a skill `scripts/*.sh` is prefixed with
`cd "$TILEGYM_PATH" &&` so it runs from the root even if a prior step changed the
directory (e.g. a cargo build in the kernel-out dir).

Use exactly the relevant set when filling `{agent_step0_files}` in the spawn template. These lists are intentionally different by stage.

Agent A reads only dump/environment guidance:

```text
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/agents/agent_a.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/env-block.md
```

Agent B reads conversion references, examples, and A's concrete reference artifacts:

```text
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/agents/agent_b.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/coding-rules.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/op-mapping.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/concepts/strided-view-to-partition-view.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/concepts/transpose-support.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/concepts/tensor-vs-pointer-pattern.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/softmax/walkthrough.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/softmax/kernel.rs
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/softmax/softmax_pipeline.rs
- $CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reference/analysis.json
- the concrete structural reference IR paths from Agent A: either reference/reference.mlir or each top-level reference/reference_{variant}.mlir named in prior_artifact_paths
- optional dtype-only supplements under reference/supplements/ only when analysis.json names them
```

Agent C reads IR-diff references plus B outputs. If the route came from D/E, include the relevant failure/perf report paths too.

```text
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/agents/agent_c.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/ir-diff-checklist.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/coding-rules.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/op-mapping.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/concepts/strided-view-to-partition-view.md
- $CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reference/analysis.json
- the concrete reference/generated canonical IR pair(s) from prior_artifact_paths
- $CUTILE_KERNEL_OUT_ROOT/{kernel_name}/kernel.rs
- $CUTILE_KERNEL_OUT_ROOT/{kernel_name}/ffi.rs
- $CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reports/agent_b.md
```

Agent D reads only the concrete templates it needs:

```text
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/agents/agent_d.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/softmax/ffi.rs       (raw-pointer launch template)
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/bmm/ffi.rs           (TILED-output template — read-only &Tensor output + partition_full_mut; NO &mut Tensor)
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/examples/softmax/wrapper.py   (wrapper + argtypes + autotune)
- $CUTILE_KERNEL_OUT_ROOT/{kernel_name}/kernel.rs                  (B's kernel — entry sigs to launch)
- $CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reference/analysis.json    (autotune configs for wrapper)
- $TILEGYM_PATH/src/tilegym/ops/cutile_rs/__init__.py + $TILEGYM_PATH/src/tilegym/backend/selector.py   (backend-reg files D edits)
```
On-demand only (not batched): references/coding-rules.md (if a SKIP needs a citation),
tests/ops/test_{kernel_name}.py (when adding the cutile-rs parametrization).

Agent E reads benchmark instructions only:

```text
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/agents/agent_e.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/performance-checklist.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/coding-rules.md
```

Agent F reads residual-perf diagnostics:

```text
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/agents/agent_f.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/performance-checklist.md
- .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/references/coding-rules.md
```

## Minimal Agent Spawn Template

Use this shape for every sub-agent. Do not paste the full agent file into the prompt, and do not paste B conversion references into non-B stages.

```text
Agent(
  description="Agent {X} - {short title} for {kernel_name}",
  subagent_type="general-purpose",
  prompt="""You are Agent {X}. Execute the {X}-stage of the cutile-rs conversion pipeline for kernel `{kernel_name}`.

STEP 0 (MANDATORY first tool call): Read these files in ONE batch Read call:
{agent_step0_files}

Then follow agent_{x}.md's procedure for `{kernel_name}`. If agent_{x}.md asks for an additional report because this is a failure/perf route, read only the concrete report path named below or in prior_artifact_paths.

Prior-stage artifacts already on disk:
{prior_artifact_paths}

OUTPUT PROTOCOL:
1. Write the stage report to $CUTILE_KERNEL_OUT_ROOT/{kernel_name}/reports/agent_{x}.md.
2. Write the stage artifacts required by agent_{x}.md.
3. Run only this stage's validators from agent_{x}.md's closing contract and capture stdout/stderr.
   - Agent B runs validate_agent_b.sh and its compile/canonicalization checks.
   - Agent B does NOT run validate_kernel.sh; that is the final aggregate gate after D/E.
4. Return only a literal <VALIDATOR_OUTPUT>...</VALIDATOR_OUTPUT> block followed immediately by `VERDICT: X`.

If you cannot complete the stage, write SKILL_METHODOLOGY_BLOCKED.txt with the reason and return `VERDICT: BLOCKED`."""
)
```

## Routing Table

The orchestrator checks the validator block first, then applies this table. It does not infer fixes from prose.

Device/host split (this is why the fork exists; the rows below are the single
source of truth for where each verdict routes): Agent B writes the DEVICE kernel
only (kernel.rs + passing in-Rust pipeline test); Agent D owns the entire HOST
boundary — ffi.rs (C-ABI launcher), the cdylib .so build, the Python wrapper +
backend wiring, AND correctness. Agents D and C emit the `fail_class:` /
`block_class:` (D) or `owner:` (C) line that selects the matching row.

```text
Stage              VERDICT                      Action
-----              -------                      ------
Agent A            PASS                         spawn Agent B
Agent A            FAIL_FIXABLE | BLOCKED       STOP

Agent B(1)         COMPILED                     spawn Agent D
Agent B(1)         FAIL_COMPILE | FAIL_FIXABLE | BLOCKED
                                                STOP

Agent D(1)         ALL_PASS                     spawn Agent E
Agent D(1)         FAIL (fail_class: host)      spawn Agent D(2) — D owns ffi.rs/.so/wrapper/launch fix.
                                                Read `host_fix:` in correctness.md.
Agent D(1)         FAIL (fail_class: kernel)    spawn Agent C — device-math fault; C diagnoses → B(2).
                                                Read `agent_b_followup:` in correctness.md.
Agent D(1)         BLOCKED (block_class: host)
                                                spawn Agent D(2) — D's own wiring (0 tests collected,
                                                import error, argtype/CUDA-700). NOT a C/B problem.
Agent D(1)         BLOCKED (block_class: kernel)
                                                spawn Agent C (in-kernel panic / missing .so / FFI gap)
Agent D(1)         BLOCKED (block_class: env)
                                                STOP — external cause (missing toolchain / GPU /
                                                driver); not fixable by C, B, or D.

Agent E(1)         DONE                         PIPELINE_COMPLETE
Agent E(1)         INVESTIGATE | BLOCKED        spawn Agent C to diagnose the perf root cause,
                                                if C has not run; otherwise PIPELINE_COMPLETE

Agent C            FAIL_FIXABLE (owner: host)    spawn Agent D(2) if D has not been respawned — the fix is
                                                host/Python-side (autotune config, grid budget,
                                                num_cta_in_cga/occupancy VALUE, layout). otherwise PIPELINE_COMPLETE
Agent C            FAIL_FIXABLE (owner: kernel)  spawn Agent B(2) if B has not been respawned — the fix is
                                                kernel.rs device math/IR. otherwise PIPELINE_COMPLETE
Agent C            PASS | PASS_WITH_NOTES        PIPELINE_COMPLETE
Agent C            FAIL_COMPILER_BUG | BLOCKED   PIPELINE_COMPLETE

Agent B(2)         COMPILED                     spawn Agent D(2)
Agent B(2)         FAIL_COMPILE | FAIL_FIXABLE | BLOCKED
                                                STOP

Agent D(2)         ALL_PASS                     spawn Agent E(2)
Agent D(2)         FAIL | BLOCKED               PIPELINE_COMPLETE

Agent E(2)         DONE | INVESTIGATE | BLOCKED PIPELINE_COMPLETE

Agent F            FIXABLE | ALIGNED | BLOCKED  PIPELINE_COMPLETE
```

If a `fail_class:` / `block_class:` (from D) or `owner:` (from C) line is absent,
fall back to the kernel route (C / B) — but the agents are required to emit it,
and host faults misrouted to B waste a full stage.

Meaning of Agent C verdicts:

- `FAIL_FIXABLE` means "a concrete next edit is owned now." C MUST tag the owner in `generated/diff_report.md` with an `owner: host` or `owner: kernel` line, where `host` = ffi.rs launcher / output-partition ABI / launch grid / autotune value / wrapper, and `kernel` = kernel.rs device math / IR. The orchestrator routes that owner strictly per the Routing Table above — do not restate the target agent here. Action items go in `generated/diff_report.md`.
- `PASS` means the IR/perf/correctness diagnostic found no fixable owner.
- `PASS_WITH_NOTES` means the remaining differences are known gaps, INFO-only deltas, accepted semantic equivalents, or residual perf notes. Do not respawn for this verdict.
- `FAIL_COMPILER_BUG` and `BLOCKED` stop the in-run loop; reflection mines the reports.

## Spawn Caps

Hard caps per pipeline run:

| Agent | Max semantic runs |
|---|---:|
| A | 1 |
| B | 2 |
| C | 1 |
| D | 2 |
| E | 2 |
| F | 1 |

A same-agent validator-block repair is also capped at one per letter. Validator repair never cascades to another agent letter.

## Mechanical Validator Acceptance Gate

After every sub-agent return:

1. Extract the first enumerated `VERDICT:` line.
2. Look for a literal `<VALIDATOR_OUTPUT>` ... `</VALIDATOR_OUTPUT>` block.
3. If the block exists and every parseable `exit=<n>` / `exit_code=<n>` is zero, accept the return and route by verdict.
4. If the block is missing but the final verdict is a clean success marker (`PASS`, `COMPILED`, `ALL_PASS`, `DONE`) and the return contains no failure marker (`FAIL`, `ERROR`, `Traceback`, `panic`, `exit=1`, `exit_code=1`, `nonzero`, or `timeout`), accept the verdict, record `SOFT_VALIDATOR_MISSING`, and do not respawn for formatting alone.
5. If a block exists with a non-zero exit, or no clean fallback applies, respawn only the same agent letter once for validator repair. The repair prompt tells it to reuse existing artifacts and return the same semantic verdict plus a valid validator block.
6. If the repair still fails, proceed with `SOFT_VALIDATOR_FAILED` unless the semantic verdict itself blocks. Never cascade validator repair across agents.

This gate is mechanical. It does not authorize the orchestrator to run cargo, pytest, benchmarks, IR diffs, or source edits inline.

## Final Aggregate Gate

Run this only after the route has reached `PIPELINE_COMPLETE`:

```bash
cd "$TILEGYM_PATH" && bash .agents/skills/tilegym-converting-cutile-triton-to-cutile-rs/scripts/validate_kernel.sh {kernel_name}
```

`validate_kernel.sh` is a 17-file end-to-end checklist across A/B/C-if-run/D/E outputs. It is not an Agent B validator because B runs before correctness and performance artifacts exist.

If final validation fails, record the output in the orchestrator summary. Do not repair by running cargo, pytest, benchmarks, or editing files inline; if another agent is needed, it must follow the routing table and spawn caps.

## Pipeline Order

1. Run `scripts/preflight.sh` first. It checks required env vars and toolchain paths.
2. Spawn Agent A. Agent A writes reference IR, analysis.json, tolerance, and baseline perf logs.
3. Spawn Agent B. Agent B writes the device kernel.rs, kernel-only Cargo project, generated IR, and a build log, and proves the kernel with a passing in-Rust pipeline test + a host_launch_contract for D. B does NOT write ffi.rs, build the .so, write the Python wrapper, or register the backend.
4. Spawn Agent D. Agent D writes ffi.rs (C-ABI launcher) + builds the cdylib .so, then the Python wrapper + backend wiring (ops/cutile_rs/<name>.py, ops/__init__, selector.py, test parametrization), then runs tilegym correctness.
5. Spawn Agent E. Agent E runs CUPTI-backed performance comparison.
6. Spawn Agent C only when D/E produce a non-DONE semantic result. C diagnoses, tags `owner: host|kernel`, and writes `generated/diff_report.md`.
7. On a fixable result, respawn the owner that C's `owner:` line (or D's `fail_class:`) names, exactly as the Routing Table maps it — do not re-derive the target here. ffi.rs is D-owned, never B's, so a host fault never routes to B; D self-routes its own FAIL via `fail_class:`.
8. Stop after E(2) or any cap exhaustion. Reflection handles multi-iteration learning.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `preflight.sh` | Verify env vars and toolchain paths | none |
| `validate_analysis.sh` | Validate Agent A analysis.json | `{kernel_name}` |
| `validate_ir.sh` | Validate a reference/generated MLIR file | `{path/to/.mlir} {kernel_name}` |
| `validate_agent_b.sh` | Validate B-stage kernel/build/wrapper artifacts | `{kernel_name}` |
| `validate_agent_d.sh` | Validate D correctness report | `{kernel_name}` |
| `validate_agent_e.sh` | Validate E performance report | `{kernel_name}` |
| `validate_agent_f.sh` | Validate F residual investigation report | `{kernel_name}` |
| `validate_diff_report.sh` | Validate Agent C diff_report.md | `{kernel_name}` |
| `validate_perf_log.sh` | Validate CUPTI perf log records | `{path/to/perf_log.txt} {label}` |
| `validate_kernel.sh` | Final aggregate output contract | `{kernel_name}` |
| `diff_ir.sh` | Side-by-side IR op/attribute diff | `{ref.mlir} {gen.mlir}` |

## Examples

Two fully worked conversions ship under `examples/` as read-only templates the
sub-agents cite (per the Agent Step 0 File Sets) — never copied blindly:

- `examples/softmax/` — **raw-pointer launch** (`kernel.rs`, `ffi.rs`,
  `wrapper.py`, `softmax_pipeline.rs`, `walkthrough.md`). The pattern for
  elementwise / reduction ops that launch over raw device pointers.
- `examples/bmm/` — **tiled-output launch** (`kernel.rs`, `ffi.rs`,
  `wrapper.py`, `walkthrough.md`). The template for matmul-class ops:
  read-only `&Tensor` output + `partition_full_mut` (never `&mut Tensor`).

## STOP List

The orchestrator must not:

- Paste an agent file into a spawn prompt.
- Add common conversion references to every agent prompt.
- Summarize or reinterpret agent instructions when spawning.
- Respawn an agent beyond the caps above.
- Spawn B(2) after C `PASS` or `PASS_WITH_NOTES`.
- Run `validate_kernel.sh` from Agent B's validator block.
- Edit `kernel.rs`, `ffi.rs`, `wrapper.py`, tests, Cargo.toml, or generated IR. (This bans the **orchestrator** from editing. Sub-agents edit their own artifacts per their agent files — Agent D writes ffi.rs / wrapper / test parametrization; Agent E may make the one narrow `test_perf` backend-parametrize wiring described in its Step 0.5. Neither is an orchestrator edit.)
- Run cargo, pytest, benchmarks, nsys, ncu, or IR diff inline.
- Parse sub-agent return text beyond validator-block mechanics and final verdict.
- Use final aggregate validation as a reason to bypass the routing table.

For detailed rationale, retry policy, and critical conversion rules, read `references/pipeline.md`.
