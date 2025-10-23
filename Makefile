.PHONY: help venv install dev clean lint typecheck test ci run-paper run-dev backtest

# Detect Python command - use venv if available
PYTHON := $(shell if [ -f venv/bin/python ]; then echo venv/bin/python; else echo python3; fi)
PIP := $(shell if [ -f venv/bin/pip ]; then echo venv/bin/pip; else echo pip; fi)

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv: ## Create Python 3.13 virtual environment
	@if [ ! -d venv ]; then \
		echo "Creating venv with Python 3.13..."; \
		if command -v python3.13 >/dev/null 2>&1; then \
			python3.13 -m venv venv; \
		elif [ -f /opt/homebrew/bin/python3.13 ]; then \
			/opt/homebrew/bin/python3.13 -m venv venv; \
		else \
			echo "Python 3.13 not found, using python3..."; \
			python3 -m venv venv; \
		fi; \
		echo "Upgrading pip..."; \
		venv/bin/pip install --upgrade pip setuptools wheel; \
		echo "✅ Virtual environment created at ./venv"; \
		echo "Activate with: source venv/bin/activate"; \
	else \
		echo "Virtual environment already exists at ./venv"; \
	fi

install: venv ## Install production dependencies
	$(PIP) install -r requirements.txt

dev: venv ## Install development dependencies
	$(PIP) install -r requirements.txt -r requirements-dev.txt

clean: ## Remove build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf htmlcov/
	rm -f .coverage

clean-venv: ## Remove virtual environment
	rm -rf venv/

lint: ## Run code formatters and linters
	$(PYTHON) -m black bot/ tests/
	$(PYTHON) -m ruff check --fix bot/ tests/

typecheck: ## Run type checking
	$(PYTHON) -m mypy bot/

test: ## Run unit tests only (skip integration and slow tests)
	$(PYTHON) -m pytest tests/unit/ -v -m "not integration and not slow"

test-integration: ## Run integration tests
	$(PYTHON) -m pytest tests/integration/ -v -m integration

test-all: ## Run all tests (unit + integration, skip slow)
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest tests/unit/ -v --cov=bot --cov-report=html --cov-report=term

ci: lint typecheck test ## Run all CI checks (lint + typecheck + unit tests)

quick-check: ## Run quick checks (lint + typecheck only)
	@./scripts/quick-check.sh

full-check: ## Run full checks (lint + typecheck + unit tests)
	@./scripts/check.sh

run-paper: ## Run bot in paper trading mode
	$(PYTHON) -m bot.cli run --profile paper

run-dev: ## Run bot in development mode
	$(PYTHON) -m bot.cli run --profile dev

backtest: ## Run backtest (usage: make backtest SYMBOL=BTCUSDT)
	$(PYTHON) -m bot.cli backtest --symbol $(or $(SYMBOL),BTCUSDT) --start 2021-01-01 --end 2022-01-01

symbols: ## List available trading symbols
	$(PYTHON) -m bot.cli symbols

validate-config: ## Validate configuration
	$(PYTHON) -m bot.cli config validate --profile $(or $(PROFILE),paper)

docker-build: ## Build Docker image
	docker build -t ai-trading-bot:latest .

docker-run: ## Run bot in Docker
	docker run --env-file .env ai-trading-bot:latest
