# Development Scripts

## Setup

First, create and activate the virtual environment:

```bash
# Create venv with Python 3.13 (or available Python)
make venv

# Activate venv
source venv/bin/activate

# Install dependencies
make dev
```

The scripts automatically detect and use the venv if available.

## Pre-Commit Check Scripts

These scripts help ensure your code passes all CI checks before committing.

### Quick Check (Recommended for Development)

```bash
./scripts/quick-check.sh
```

Runs fast checks only:
- ✅ Black formatting
- ✅ Ruff linting
- ✅ mypy type checking

**Use this during development for rapid feedback (< 10 seconds)**

### Full Check (Before Committing)

```bash
./scripts/check.sh
```

Runs all checks:
- ✅ Black formatting
- ✅ Ruff linting
- ✅ mypy type checking
- ✅ Unit tests with pytest

**Use this before committing to ensure CI will pass (~ 30 seconds)**

### Integration Tests (Optional)

```bash
python3 -m pytest -v -m integration
```

Runs integration tests that require:
- Network connectivity
- API credentials (for Binance testnet)
- May take several minutes

**These tests are skipped in regular CI to prevent hanging**

## Fixing Issues

### Black Formatting Issues

```bash
python3 -m black bot/ tests/
```

### Ruff Linting Issues

```bash
python3 -m ruff check --fix bot/ tests/
```

### Type Checking Issues

Review mypy errors and fix type annotations manually.

## CI Configuration

The GitHub Actions workflow mirrors these checks:

1. **lint** job: Black + Ruff
2. **typecheck** job: mypy
3. **test** job: pytest (unit tests only, 10-minute timeout)

Integration tests are skipped by default to prevent hanging.
