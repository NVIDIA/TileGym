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

  Code contribution is currently not open until we establish our Github CI. Please stay tuned.

  - See [Code contributions](#2-code-contributions) for the end-to-end workflow and expectations.
  - See [Testing contributions](#3-testing-contributions-especially-op-tests) if your change adds or modifies ops or tests.

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
6. **Open a pull request**
   - Keep PRs focused on a single logical change.
   - Describe what the PR does, how you tested it, and any potential user-facing impact.
7. **Respond to review**
   - Address comments, push updates, and keep the discussion on the PR/issue.

### 3. Testing contributions (especially op tests)

Robust tests are critical for functional correctness and performance coverage.

- **If you are contributing new op tests or modifying existing ones**, please:
   - Ensure new tests follow the structure in [`tests/ops/README.md`](tests/ops/README.md).
   - Add cases that cover typical shapes, edge cases, and mixed-precision scenarios where relevant.
   - Make sure tests pass locally with `pytest` before opening the PR.

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
