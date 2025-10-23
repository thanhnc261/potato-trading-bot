#!/bin/bash
# Quick pre-commit checks (lint + typecheck only, no tests)
# Use this for rapid feedback during development

set -e

echo "⚡ Running quick checks (lint + typecheck)..."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    PYTHON="python"
else
    PYTHON="python3"
fi

FAILED=0

# Black
echo "1️⃣  Black formatting..."
if $PYTHON -m black --check bot/ tests/ --quiet; then
    echo -e "${GREEN}✅ Black${NC}"
else
    echo -e "${RED}❌ Black${NC}"
    FAILED=1
fi

# Ruff
echo "2️⃣  Ruff linting..."
if $PYTHON -m ruff check bot/ tests/ --quiet; then
    echo -e "${GREEN}✅ Ruff${NC}"
else
    echo -e "${RED}❌ Ruff${NC}"
    FAILED=1
fi

# mypy
echo "3️⃣  Type checking..."
if $PYTHON -m mypy bot/ --no-error-summary 2>/dev/null; then
    echo -e "${GREEN}✅ mypy${NC}"
else
    echo -e "${RED}❌ mypy${NC}"
    FAILED=1
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All quick checks passed!${NC}"
    echo "Run './scripts/check.sh' for full test suite"
    exit 0
else
    echo -e "${RED}❌ Some checks failed${NC}"
    echo "Run './scripts/check.sh' for detailed output"
    exit 1
fi
