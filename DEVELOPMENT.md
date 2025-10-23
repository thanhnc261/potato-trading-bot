# Development Guide - Quick Reference

## Setup (One Time)

```bash
# Create virtual environment with Python 3.13
make venv

# Activate it
source venv/bin/activate

# Install dependencies
make dev
```

## Daily Workflow

### Before You Start

```bash
# Activate venv (if not already activated)
source venv/bin/activate
```

### While Developing

```bash
# Quick checks (10s) - run frequently
make quick-check

# Or use the script directly
./scripts/quick-check.sh
```

### Before Committing

```bash
# Full checks including tests (30s)
make full-check

# Or use make
make ci
```

### If Checks Fail

```bash
# Fix formatting
make lint

# Check types
make typecheck

# Run specific tests
make test
```

## Common Commands

### Testing
```bash
make test                # Unit tests only (fast, default)
make test-integration    # Integration tests (slow, requires credentials)
make test-all           # All tests
make test-cov           # Tests with coverage report
```

### Code Quality
```bash
make lint               # Auto-fix formatting and linting
make typecheck          # Run mypy type checking
make ci                 # Run all checks (lint + typecheck + test)
```

### Quick Checks
```bash
make quick-check        # Fast: Black + Ruff + mypy (~10s)
make full-check         # Full: quick-check + unit tests (~30s)
```

### Cleanup
```bash
make clean              # Remove build artifacts
make clean-venv         # Remove virtual environment
```

## Python Version

**Using:** Python 3.13.0 (in venv/)
**Supports:** Python 3.11, 3.12, 3.13
**CI Tests:** All three versions

## File Structure

```
venv/                    # Virtual environment (gitignored)
scripts/
  ├── check.sh          # Full pre-commit checks
  └── quick-check.sh    # Fast lint + typecheck
tests/
  ├── unit/            # Unit tests (always run)
  └── integration/     # Integration tests (skipped by default)
```

## Tips

1. **Always activate venv** before working:
   ```bash
   source venv/bin/activate
   ```

2. **Run quick-check frequently** during development:
   ```bash
   make quick-check  # or ./scripts/quick-check.sh
   ```

3. **Run full-check before committing**:
   ```bash
   make full-check  # or ./scripts/check.sh
   ```

4. **Integration tests are skipped by default** (they can hang):
   ```bash
   # Run only if you have credentials and time
   make test-integration
   ```

5. **Use Make tab completion** to discover commands:
   ```bash
   make <TAB><TAB>
   ```

## Troubleshooting

### "command not found: make"
Use scripts directly:
```bash
./scripts/quick-check.sh
./scripts/check.sh
```

### "No module named X"
Activate venv and reinstall:
```bash
source venv/bin/activate
make dev
```

### Tests hang
Integration tests are slow. Run unit tests only:
```bash
make test  # Only runs tests/unit/
```

### Wrong Python version
Recreate venv:
```bash
make clean-venv
make venv
make dev
```

## See Also

- [SETUP.md](SETUP.md) - Detailed setup instructions
- [scripts/README.md](scripts/README.md) - Check scripts documentation
- [CLAUDE.md](CLAUDE.md) - Project architecture guide
- [README.md](README.md) - Project overview
