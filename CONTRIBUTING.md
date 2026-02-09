<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

## Contributing to TileGym

Thank you for your interest in contributing to TileGym!
This document explains the main ways you can help and what we expect from contributions.

### 1. Ways to contribute

- **Report issues**
  - Use the [issue tracker](https://github.com/NVIDIA/TileGym/issues) to report bugs, request features, or suggest improvements.
  - Include clear steps to reproduce, expected vs. actual behavior, and environment details when possible.

- **Contribute code**
  - See [Code contributions](#2-code-contributions) for the end-to-end workflow and expectations.
  - See [Contributing kernels](#3-contributing-kernels) for kernel work (new kernels vs. optimizing existing kernels).

Before starting non-trivial work, please check for existing issues and consider opening a discussion/issue so we can align on scope and design.

### 2. Code contributions

If you plan to submit code changes (new features, bug fixes, refactors):

1. **Review the Roadmap**
   - **Before contributing**, please review the [ROADMAP.md](ROADMAP.md) to understand:
     - Current operator support status (what's available, in progress, or planned)
     - Contribution opportunities and priority areas
     - Which kernels need help (marked as "🙋 Help Wanted")
   - This helps ensure your contribution aligns with project priorities and avoids duplicate work.

2. **Read the project README**
   - Review the project-level [`README.md`](README.md) for build, install, and basic usage instructions.
3. **Pick or propose an issue**
   - Look for existing issues that match what you want to do, or create a new issue describing your proposal.
   - Comment on the issue to indicate you are working on it.
4. **Discuss significant changes first**
   - For larger features or intrusive refactors, outline your approach in the issue so maintainers can provide feedback early.
5. **Implement the change**
   - Follow the existing coding style and patterns in the affected modules.
   - Add or update tests to cover new behavior.
   - If you are contributing kernel code, follow [Contributing kernels](#3-contributing-kernels).
6. **Format your code**
   - Run `./format.sh` from the repository root to ensure your changes pass CI format checks.
7. **Open a pull request**
   - Keep PRs focused on a single logical change.
   - Describe what the PR does, how you tested it, and any potential user-facing impact.
8. **Respond to review**
   - Address comments, push updates, and keep the discussion on the PR/issue.

### 3. Contributing kernels

There are **two common situations** when contributing kernel code:

- [Situation A: Add a new kernel (or a new op)](#situation-a-add-a-new-kernel-or-a-new-op)
- [Situation B: Optimize or update an existing kernel](#situation-b-optimize-or-update-an-existing-kernel)

#### Situation A: Add a new kernel (or a new op)

If you are adding a **new kernel** (new `@ct.kernel` / new op implementation) that is not yet validated by the core team, it should go through the experimental-kernel flow.

New cuTile kernel contributions should first be placed in the `experimental/` directories. Once the TileGym team has fully verified functional correctness and performance, kernels will be promoted from `experimental/` into the main source tree.

##### Directory structure

```
src/tilegym/ops/cutile/experimental/<kernel_name>.py   # kernel implementation
tests/ops/experimental/test_<kernel_name>.py            # correctness tests
tests/benchmark/experimental/bench_<kernel_name>.py     # performance benchmarks
```

##### Step-by-step: Adding a new kernel

###### a. Create the kernel file

Create `src/tilegym/ops/cutile/experimental/<kernel_name>.py`.

- Import `experimental_kernel` via `from tilegym.experimental import experimental_kernel`.
- Apply the `@experimental_kernel` decorator **before** `@ct.kernel`.
- Use `@register_impl` on the op entry-point function.
- See [`src/tilegym/ops/cutile/experimental/mhc.py`](src/tilegym/ops/cutile/experimental/mhc.py) for an example.

The `@experimental_kernel` decorator marks a kernel as experimental so that a one-time message is printed the first time the kernel is launched via `ct.launch`. Three usage forms:

```python
@experimental_kernel                         # bare — auto-generates message from kernel name (recommended)
@experimental_kernel()                       # empty parens — same as bare
@experimental_kernel("Custom message text")  # custom message
```

- The bare form (recommended) auto-generates a message that includes the kernel's function name.
- The message prints **once per kernel per process** at the first `ct.launch` call.
- The decorator must be placed **before** `@ct.kernel`.

###### b. Add the public interface (if introducing a new op)

If your kernel introduces a **new public op name** (a new dispatch key that does not already exist), you need to add it to the unified API layer in [`src/tilegym/ops/ops.py`](src/tilegym/ops/ops.py):

1. Add a new function decorated with `@dispatch("<op_name>")`.
2. Keep the public API signature stable and aligned with your implementation.
3. Ensure the dispatch key string matches exactly: `@dispatch("<op_name>")` in `ops.py` must match `@register_impl("<op_name>", backend=...)` in your backend implementation.

If you are providing a new backend implementation for an op that **already exists** in `ops.py`, you usually do not need to modify `ops.py`.

###### c. Register exports

In [`src/tilegym/ops/cutile/__init__.py`](src/tilegym/ops/cutile/__init__.py):

1. Add an import for your module (for example `from .experimental import <module>`) so the module is loaded and your `@register_impl(...)` registration runs.
2. If you want functions to be directly accessible from `tilegym.ops.cutile`, add `from .experimental.<module> import <function>`.
3. Add any directly-exported function names to `__all__`.

###### d. Create tests

Create `tests/ops/experimental/test_<kernel_name>.py`.

- Inherit from `common.PyTestCase`.
- Implement a `reference()` method (or similar) that computes the expected result using PyTorch.
- Use `@pytest.mark.parametrize` for shape/dtype coverage and `self.assertCorrectness()` for validation.
- Add cases that cover typical shapes, edge cases, and mixed-precision scenarios where relevant.
- Make sure tests pass locally with `pytest` before opening the PR.
- See [`tests/ops/README.md`](tests/ops/README.md) for full testing guidelines.

###### e. Create benchmarks

Create `tests/benchmark/experimental/bench_<kernel_name>.py`.

Benchmarks in `tests/benchmark/experimental/` are auto-discovered by `run_all.sh` and `run_all_json.py` — no extra registration is needed.

#### Situation B: Optimize or update an existing kernel

If you are **optimizing or updating an existing kernel** (for performance, correctness, maintainability, readability, portability, etc.) for an op that already exists, you typically only need to update the existing implementation file(s) and tests/benchmarks.

- **No `@experimental_kernel` marker needed**: you do **not** need to add `@experimental_kernel` when you are improving an already-existing kernel implementation. If the kernel already carries `@experimental_kernel`, keep the existing behavior unless maintainers request otherwise.
- **No `ops.py` or registration changes needed** in most cases, because the op is already dispatched. This includes helper kernels that are called inside an already-registered implementation.
- **Follow the existing coding style** and patterns in the affected modules.
- **Add or update tests and benchmarks** to verify your changes if needed: correctness tests under `tests/ops/` and benchmarks under `tests/benchmark/`.

### 4. First-time contributors: CLA required

To accept your contribution, we need a signed Contributor License Agreement (CLA) on file.

1. Locate the CLA at [`LICENSES/CLA.md`](LICENSES/CLA.md) in this repository.
2. Fill it out and sign.
3. Email the signed CLA to `TileGym@nvidia.com` with subject: `TileGym CLA Submission`.
4. Wait for confirmation from the TileGym team before your PR can be merged.

### 5. Review & merge process

- Maintainers will review your PR, suggest changes if needed, and approve once it meets project standards.
- CI and tests must pass before merge.
- Focused, well-described, and well-tested PRs are much easier and faster to review.

If anything in this document is unclear or missing, feel free to comment on issues and ask for clarifications!
