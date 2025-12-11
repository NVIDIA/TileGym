# GitHub Actions Scripts

Utility scripts for GitHub Actions workflows. All scripts are tested in `../.github/infra_tests/`.

## Scripts Overview

- **`parse_pr_config.py`** - Parse PR body for CI configuration
- **`check_image_exists.py`** - Check if Docker image exists in GHCR
- **`cleanup_stale_images.py`** - Clean up old/orphaned Docker images
- **`utils.py`** - Shared utilities (token handling, GitHub API headers, output writing)

---

## parse_pr_config.py

Parses CI configuration from PR body to control which jobs run.

### Usage in PR

Add this to your PR description:

```yaml
config:
  build: true
  test: ["ops", "benchmark"]
```

### Configuration Options

- **`build`** (boolean): Whether to build the Docker image
  - `true`: Build from scratch (default)
  - `false`: Pull latest from GHCR (must have been built previously)

- **`test`** (list): Which test suites to run
  - `["ops", "benchmark"]`: Run both (default)
  - `["ops"]`: Only run ops tests
  - `["benchmark"]`: Only run benchmark tests
  - `[]`: Skip all tests

### Examples

**Quick iteration - skip build:**
```yaml
config:
  build: false
  test: ["ops"]
```

**Only benchmarks:**
```yaml
config:
  build: true
  test: ["benchmark"]
```

**Skip CI entirely:**
```yaml
config:
  build: false
  test: []
```

### Notes

- On first run, you must use `build: true` to create the cached image
- If no config is found, defaults to building everything

---

## check_image_exists.py

Checks if a Docker image with a specific tag exists in GHCR to skip redundant nightly builds.

### Environment Variables

- `REGISTRY_IMAGE`: Full registry path (e.g., `ghcr.io/nvidia/tilegym-transformers`)
- `IMAGE_TAG`: Tag to check (typically commit SHA)
- `GITHUB_TOKEN`: GitHub token for authentication
- `IS_PR`: Whether running in PR context (`true`/`false`)

### Outputs

- `skipped`: `true` if image exists and build should be skipped, `false` otherwise

### Behavior

- **PR context**: Always returns `skipped=false` (always build for PRs)
- **Nightly/main context**: Checks if image exists, skips if found

---

## cleanup_stale_images.py

Cleans up stale Docker images from GHCR to save storage and keep registry organized.

### Environment Variables

- `GITHUB_TOKEN`: GitHub token for authentication
- `GITHUB_REPOSITORY_OWNER`: Repository owner (org or user)
- `GITHUB_REPOSITORY`: Full repository name
- `PACKAGE_NAME`: GHCR package name (e.g., `tilegym-transformers-pr`)
- `ORPHAN_DAYS_THRESHOLD`: Days to keep orphaned images (default: 7)

### Cleanup Rules

1. **Closed PR images**: Deletes images tagged with `pr-*` where PR is closed
2. **Orphaned images**: Deletes images without `pr-*` or `latest` tags older than threshold

### Schedule

Runs daily at 2 AM UTC via `cleanup-stale-images.yml` workflow.

---

## utils.py

Shared utility functions used across all scripts.

### Functions

- **`get_github_token()`**: Get and validate GitHub token from environment
- **`write_github_output(key, value)`**: Write key-value pair to GitHub Actions output
- **`get_github_api_headers(token)`**: Get standard GitHub API headers with authentication

### Usage

```python
from utils import get_github_token, write_github_output, get_github_api_headers

token = get_github_token()
headers = get_github_api_headers(token)
write_github_output("my_output", "my_value")
```

---

## Testing

All scripts have comprehensive unit tests in `../.github/infra_tests/`:

- `test_parse_pr_config.py` (6 tests)
- `test_check_image_exists.py` (6 tests)
- `test_cleanup_stale_images.py` (6 tests)
- `test_utils.py` (5 tests)

Run tests locally:
```bash
pytest .github/infra_tests/ -v
```

Tests run automatically in the `ci-infra-tests` workflow on every push.

