<!--- SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. --->

<!--- SPDX-License-Identifier: MIT --->

# GitHub Workflows & Infrastructure

This directory contains CI/CD workflows, utility scripts, and infrastructure tests for the TileGym repository.

## Workflows

### `tilegym-ci.yml`
**Main CI workflow** - Builds Docker images and runs tests.

**Jobs:**
- `config` - Parses PR body for CI configuration options
- `build` - Builds `tilegym` Docker image and pushes to GHCR
- `test-ops` - Runs ops tests (`pytest -s tests/ops`)
- `test-benchmark` - Runs benchmark tests sequentially (`tests/benchmark/run_all.sh`)
- `promote-to-latest` - Tags passing nightly builds with `latest` and `<SHA>-verified`

**Scripts used:**
- `scripts/parse_pr_config.py` - Parse PR body config
- `scripts/check_image_exists.py` - Skip nightly builds if `latest` already points to current SHA (tests passed previously)

**Test Results:**
- **ops-test-results:** JUnit XML + HTML report with test pass/fail status (visible in "Checks" tab)
- **benchmark-results:** Individual `*_results.txt` files containing performance tables with TFLOPS/GBps metrics for each benchmark (downloadable artifacts)
- **Benchmark summary:** Formatted markdown tables visible in the workflow "Summary" tab

---

### `tilegym-ci-infra-tests.yml`
**Infrastructure validation** - Ensures code quality and validates CI scripts.

**Jobs:**
- `python-formatting` - Runs `ruff` for import sorting and format checks
- `spdx-headers-check` - Verifies all source files have SPDX license headers
- `utility-scripts-tests` - Runs pytest on all infrastructure tests

**Triggers:** Push to `main`, push to `pull-request/*` branches

**Tests:**
- All utility scripts in `scripts/`

---

### `tilegym-ghcr-cleanup.yml`
**GHCR maintenance** - Cleans up old Docker images to save storage.

**Jobs:**
- `cleanup` - Deletes stale PR images and untracked images

**Triggers:** Daily at 2 AM UTC, manual

**Scripts used:**
- `scripts/cleanup_stale_images.py` - Delete closed PR and untracked images

**Cleanup rules:**
- Images for closed PRs (`pr-*` tags)
- Untracked images (no `pr-*`, `latest`, or `-verified` tags, older than 7 days)
- Verified images (`*-verified` tags) are kept indefinitely

---

## Scripts

Located in `scripts/`, these Python utilities are used by workflows:

- **`parse_pr_config.py`** - Extract CI configuration from PR descriptions
- **`check_image_exists.py`** - Check if Docker images exist in GHCR
- **`cleanup_stale_images.py`** - Delete stale Docker images from GHCR
- **`format_benchmark_summary.py`** - Parse benchmark results and format as markdown tables for GitHub Actions summary
- **`check_spdx_headers.py`** - Check and add SPDX license headers to source files
- **`utils.py`** - Shared utilities (GitHub token, API headers, outputs)

All scripts have comprehensive docstrings and are fully tested.

---

## Infrastructure Tests

Located in `infra_tests/`, these pytest-based tests validate all CI scripts including:

- PR config parsing logic
- Image existence checks and latest tag validation
- Image cleanup logic (verified tag preservation, untracked image detection)
- SPDX header detection and addition
- Shared utility functions

**Run locally:**
```bash
pytest .github/infra_tests/ -v
```

Tests are independent of the main TileGym package (no torch/CUDA dependencies).

**Check SPDX headers locally:**
```bash
# Check all files have headers
python3 .github/scripts/check_spdx_headers.py --action check

# Add missing headers
python3 .github/scripts/check_spdx_headers.py --action write
```

**Test results:** Available in GitHub Actions UI under "Checks" tab and as downloadable artifacts (`infra-test-results`).

---

## PR Configuration

Control CI behavior by adding a YAML config block to your PR description:

```yaml
config:
  build: true
  test: ["ops", "benchmark"]
```

**Options:**
- `build: false` - Skip build, pull latest from GHCR
- `test: ["ops"]` - Run only ops tests
- `test: []` - Skip all tests

See `.github/pull_request_template.md` for the full template.

---

## Docker Images

**Nightly images:** `ghcr.io/<owner>/tilegym:<SHA>`, `nightly-<DATETIME>`  
**Verified images:** `ghcr.io/<owner>/tilegym:<SHA>-verified` (permanent proof tests passed)  
**Latest verified:** `ghcr.io/<owner>/tilegym:latest` (points to newest passing build)

**Tagging strategy:**
- Build pushes: `<SHA>`, `nightly-<DATETIME>`
- After tests pass: `latest` and `<SHA>-verified` tags are added
- `latest` moves to newest passing build
- `<SHA>-verified` is permanent (useful for auditing and rollbacks)
- Nightly builds skip if `latest` already points to current SHA

