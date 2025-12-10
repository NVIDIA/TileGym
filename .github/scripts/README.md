# GitHub Actions Scripts

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

