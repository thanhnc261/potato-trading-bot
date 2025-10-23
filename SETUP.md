# Development Setup Guide

## Requirements

- Python 3.11, 3.12, or 3.13 (3.13 recommended)
- Git
- Make (optional, but recommended)

## Quick Setup

### 1. Create Virtual Environment

```bash
# Using Make (recommended)
make venv

# Or manually with Python 3.13
python3.13 -m venv venv

# Or with Homebrew Python on macOS
/opt/homebrew/bin/python3.13 -m venv venv
```

### 2. Activate Virtual Environment

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Using Make
make dev

# Or manually
pip install -r requirements.txt -r requirements-dev.txt
```

### 4. Verify Installation

```bash
# Run quick checks
./scripts/quick-check.sh

# Or using Make
make quick-check
```

## Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and set your credentials:

```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_TESTNET=true
```

## Development Workflow

### Before Committing

Run all checks:

```bash
# Quick checks (lint + typecheck) - ~10 seconds
./scripts/quick-check.sh

# Full checks (lint + typecheck + tests) - ~30 seconds
./scripts/check.sh

# Or using Make
make ci
```

### Running Tests

```bash
# Unit tests only (default, fast)
make test

# Integration tests (slow, requires API credentials)
make test-integration

# All tests
make test-all

# With coverage
make test-cov
```

### Code Formatting

```bash
# Auto-fix formatting and linting
make lint

# Or manually
python -m black bot/ tests/
python -m ruff check --fix bot/ tests/
```

### Type Checking

```bash
make typecheck

# Or manually
python -m mypy bot/
```

## Available Make Commands

```bash
make help                 # Show all available commands
make venv                 # Create virtual environment
make dev                  # Install dev dependencies
make ci                   # Run all CI checks
make quick-check          # Quick lint + typecheck
make full-check           # Full checks including tests
make test                 # Run unit tests
make test-integration     # Run integration tests
make test-all             # Run all tests
make lint                 # Format and lint code
make typecheck            # Run type checking
make clean                # Remove build artifacts
make clean-venv           # Remove virtual environment
```

## Python Version Support

This project supports Python 3.11, 3.12, and 3.13:

- **pyproject.toml**: `requires-python = ">=3.11,<3.14"`
- **CI**: Tests run on all three versions
- **Development**: Python 3.13 recommended for latest features

## Troubleshooting

### Python Version Not Found

If `python3.13` is not available:

```bash
# macOS with Homebrew
brew install python@3.13

# Or use available Python 3.11+
python3 -m venv venv
```

### Dependencies Installation Fails

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Then retry
pip install -r requirements.txt -r requirements-dev.txt
```

### Tests Hang

Integration tests require network connectivity and may be slow:

```bash
# Skip integration tests (default)
pytest tests/unit/

# Run with timeout (30s per test)
pytest tests/ --timeout=30
```

### Import Errors

Ensure you're using the virtual environment:

```bash
# Check Python location
which python

# Should be: /path/to/project/venv/bin/python
# If not, activate venv:
source venv/bin/activate
```

## IDE Setup

### VS Code

Add to `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black"
}
```

### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add → Existing Environment
3. Select `venv/bin/python`

## Next Steps

1. ✅ Setup complete - ready to develop!
2. Read [CLAUDE.md](CLAUDE.md) for architecture overview
3. Check [scripts/README.md](scripts/README.md) for check scripts
4. See [README.md](README.md) for project details
