.PHONY: help install dev clean lint typecheck test ci run-paper run-dev backtest

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -e .

dev: ## Install development dependencies
	pip install -e ".[dev]"

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

lint: ## Run code formatters and linters
	black bot/ tests/
	ruff check --fix bot/ tests/

typecheck: ## Run type checking
	mypy bot/

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=bot --cov-report=html --cov-report=term

ci: lint typecheck test ## Run all CI checks (lint + typecheck + test)

run-paper: ## Run bot in paper trading mode
	python -m bot.cli run --profile paper

run-dev: ## Run bot in development mode
	python -m bot.cli run --profile dev

backtest: ## Run backtest (usage: make backtest SYMBOL=BTCUSDT)
	python -m bot.cli backtest --symbol $(or $(SYMBOL),BTCUSDT) --start 2021-01-01 --end 2022-01-01

symbols: ## List available trading symbols
	python -m bot.cli symbols

validate-config: ## Validate configuration
	python -m bot.cli config validate --profile $(or $(PROFILE),paper)

docker-build: ## Build Docker image
	docker build -t ai-trading-bot:latest .

docker-run: ## Run bot in Docker
	docker run --env-file .env ai-trading-bot:latest
