#!/usr/bin/env python3
"""
Import tasks to vibe-kanban for Potato Trading Bot project.

Usage:
    python scripts/import_to_vibe_kanban.py <project_id> [--phase {phase1,phase2,all}]
"""

import argparse
import requests
import sys
from typing import Dict

VIBE_KANBAN_API = "http://127.0.0.1:57281/api"

PHASE_1_TASKS = [
    {
        "title": "BOT-01 - Setup Exchange Adapter",
        "description": """Implement Binance exchange adapter with robust error handling and testnet/mainnet support.

**Acceptance Criteria**:
- Create bot/execution/adapters/binance.py implementing base exchange interface
- Support REST API calls for account info, balance, orders
- Implement error handling with retries and exponential backoff
- Add testnet/mainnet toggle via environment variable
- Rate limiting to respect exchange API limits
- Unit tests for adapter methods
- Integration test with Binance testnet

**Estimated Time**: 3-4 days
**Priority**: High
**Phase**: Phase 1 - Foundation
""",
    },
    {
        "title": "BOT-02 - Market Data Streaming",
        "description": """Implement real-time market data ingestion via WebSocket or REST polling.

**Acceptance Criteria**:
- Create bot/data/market_data.py module
- WebSocket connection for real-time price updates
- Support multiple symbols simultaneously
- Store data efficiently using PyArrow
- Handle connection drops and reconnection
- Data normalization to standard format
- Tests for data stream handling

**Estimated Time**: 3-4 days
**Priority**: High
**Phase**: Phase 1 - Foundation
""",
    },
    {
        "title": "BOT-03 - Logging Infrastructure",
        "description": """Implement structured JSON logging with correlation IDs for trade lifecycle tracing.

**Acceptance Criteria**:
- Create logging configuration module
- Structured JSON log format
- Correlation IDs for request tracing
- Log levels: DEBUG, INFO, WARNING, ERROR
- Daily log rotation with 30-day retention
- Separate log files for trades, system, errors
- Integration with all modules

**Estimated Time**: 2 days
**Priority**: High
**Phase**: Phase 1 - Foundation
""",
    },
    {
        "title": "BOT-04 - RiskManager - Pre-trade Checks",
        "description": """Implement comprehensive risk management with pre-trade validation checks.

**Acceptance Criteria**:
- Create bot/risk/risk_manager.py
- Order book depth analysis for slippage estimation
- Liquidity validation (position < 1% of daily volume)
- Position sizing based on volatility (ATR)
- Global stop-loss (portfolio loss threshold)
- Correlation exposure checks
- Time-based trading restrictions (avoid off-hours)
- Risk check results logging
- Unit tests for all risk checks

**Risk Parameters** (configurable):
- max_position_size_pct: 0.03  # 3% per trade
- max_total_exposure_pct: 0.25  # 25% total
- max_slippage_pct: 0.005      # 0.5% max slippage
- min_liquidity_ratio: 0.01    # Position < 1% daily volume

**Estimated Time**: 4-5 days
**Priority**: Critical
**Phase**: Phase 1 - Foundation
""",
    },
    {
        "title": "BOT-05 - Emergency Stop System",
        "description": """Implement kill-switch system for catastrophic scenarios with automated triggers.

**Acceptance Criteria**:
- Create bot/risk/emergency_stop.py
- Monitor for flash crashes (>10% price move in <5 min)
- Detect exchange API failures
- Track portfolio drawdown in real-time
- Data quality monitoring (stale data, NaN values)
- On trigger: cancel all orders
- On trigger: close all positions at market
- On trigger: halt trading engine
- Send alerts via Telegram/Email
- Manual override capability
- Tests for all trigger conditions

**Trigger Conditions**:
- Flash crash detected
- Exchange API unreachable for >30 seconds
- Portfolio drawdown exceeds 10%
- Data feed stale (>60 seconds old)
- Consecutive API failures (>5)

**Estimated Time**: 2-3 days
**Priority**: Critical
**Phase**: Phase 1 - Foundation
""",
    },
    {
        "title": "BOT-06 - Basic Technical Strategy",
        "description": """Implement simple rule-based strategy to test the trading pipeline.

**Acceptance Criteria**:
- Create bot/core/strategy.py with base strategy class
- Implement moving average crossover OR RSI-based strategy
- Strategy generates buy/sell/hold signals
- Include position entry/exit logic
- Configuration support (strategy parameters)
- Backtestable implementation
- Unit tests for signal generation

**Example Strategy**:
- MA Crossover: Buy when MA20 crosses above MA50, Sell when crosses below
- OR RSI: Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)

**Estimated Time**: 2-3 days
**Priority**: Medium
**Phase**: Phase 1 - Foundation
""",
    },
    {
        "title": "BOT-07 - Execution Orchestrator",
        "description": """Build order execution system that routes decisions to exchange and manages order lifecycle.

**Acceptance Criteria**:
- Create bot/execution/orchestrator.py
- Place market orders via exchange adapter
- Handle order responses (filled, partial, rejected)
- Order status tracking and updates
- Integration with RiskManager (pre-trade checks)
- Integration with EmergencyStop
- Order deduplication (prevent double orders)
- Async order execution (non-blocking)
- Comprehensive logging of order lifecycle
- Tests for order placement and handling

**Order Flow**:
1. Strategy generates signal
2. RiskManager validates trade
3. Orchestrator creates order
4. Exchange adapter executes
5. Orchestrator tracks status
6. Update portfolio state

**Estimated Time**: 3-4 days
**Priority**: High
**Phase**: Phase 1 - Foundation
""",
    },
    {
        "title": "BOT-08 - Data Quality Monitor",
        "description": """Validate market data quality before making trading decisions.

**Acceptance Criteria**:
- Create bot/data/quality_monitor.py
- Price sanity checks (no >10% spikes without validation)
- Volume anomaly detection (>5x average)
- Timestamp freshness validation (<60 seconds old)
- NaN/missing data detection
- Data quality score calculation
- Halt trading on poor data quality
- Alert on data issues

**Estimated Time**: 2 days
**Priority**: High
**Phase**: Phase 1 - Foundation
""",
    },
]

PHASE_2_TASKS = [
    {
        "title": "BOT-09 - Technical Analyzer Implementation",
        "description": """Build comprehensive technical analysis module with multiple indicators.

**Acceptance Criteria**:
- [ ] Create `bot/core/technical_analyzer.py`
- [ ] Integrate TA-Lib or `ta` library
- [ ] Implement indicators: RSI, MACD, Bollinger Bands, MA20/50/200, ATR
- [ ] Support multi-timeframe analysis (1h, 4h, 1d)
- [ ] Pattern detection (support/resistance levels)
- [ ] Efficient caching of indicator calculations
- [ ] Tests comparing outputs to known values
- [ ] Use `make lint`, `make typecheck`, and `make testall` to verify before commit

**Estimated Time**: 3-4 days
**Priority**: High
**Phase**: Phase 2 - Traditional Strategy Expansion & Backtesting
""",
    },
    {
        "title": "BOT-10 - Backtesting Engine",
        "description": """Build historical simulation engine with realistic slippage and latency modeling.

**Acceptance Criteria**:
- [ ] Create `bot/data/backtesting.py`
- [ ] Load historical data from CSV/Parquet
- [ ] Replay data bar-by-bar or tick-by-tick
- [ ] Simulate order execution with slippage
- [ ] Model execution delays
- [ ] Calculate performance metrics (win rate, Sharpe, drawdown)
- [ ] Support multiple strategies and symbols
- [ ] Generate performance reports
- [ ] CLI command: `python -m bot.cli backtest`
- [ ] Use `make lint`, `make typecheck`, and `make testall` to verify before commit

**Estimated Time**: 4-5 days
**Priority**: High
**Phase**: Phase 2 - Traditional Strategy Expansion & Backtesting
""",
    },
    {
        "title": "BOT-11 - Multi-Strategy Support",
        "description": """Extend StrategyContext to support multiple concurrent strategies.

**Acceptance Criteria**:
- [ ] Create `bot/core/strategy_context.py`
- [ ] Support multiple strategy instances
- [ ] Per-strategy configuration
- [ ] Isolated state per strategy
- [ ] Strategy-level position tracking
- [ ] Aggregated risk management across strategies
- [ ] Use `make lint`, `make typecheck`, and `make testall` to verify before commit

**Estimated Time**: 3 days
**Priority**: Medium
**Phase**: Phase 2 - Traditional Strategy Expansion & Backtesting
""",
    },
    {
        "title": "BOT-12 - Paper Trading Mode",
        "description": """Implement simulated broker for paper trading.

**Acceptance Criteria**:
- [ ] Simulated portfolio tracking
- [ ] Virtual order execution
- [ ] Realistic fill simulation
- [ ] Track P/L without real money
- [ ] CLI command: `python -m bot.cli run --profile paper`
- [ ] Use `make lint`, `make typecheck`, and `make testall` to verify before commit

**Estimated Time**: 2-3 days
**Priority**: High
**Phase**: Phase 2 - Traditional Strategy Expansion & Backtesting
""",
    },
]

PHASE_DEFINITIONS = {
    "phase1": {
        "label": "Phase 1 - Foundation",
        "tasks": PHASE_1_TASKS,
    },
    "phase2": {
        "label": "Phase 2 - Traditional Strategy Expansion & Backtesting",
        "tasks": PHASE_2_TASKS,
    },
}

PHASE_SEQUENCE = ["phase1", "phase2"]


def create_task(project_id: str, task: Dict) -> bool:
    """Create a task in vibe-kanban."""
    url = f"{VIBE_KANBAN_API}/tasks"
    payload = {
        "project_id": project_id,
        "title": task["title"],
        "description": task["description"],
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Created: {task['title']}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to create '{task['title']}': {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import tasks to vibe-kanban for the Potato Trading Bot project."
    )
    parser.add_argument(
        "project_id",
        help="UUID of the target vibe-kanban project (create it first in the UI).",
    )
    parser.add_argument(
        "--phase",
        default="all",
        help="Which phase to import (phase1, phase2, 1, 2, or all). Defaults to all.",
    )
    return parser.parse_args()


def normalize_phase_name(raw_phase: str) -> str:
    """Normalize user-supplied phase filters to canonical keys."""
    if not raw_phase:
        return "all"

    value = raw_phase.strip().lower()
    if value in {"all", "both"}:
        return "all"

    phase_aliases = {
        "1": "phase1",
        "phase1": "phase1",
        "foundation": "phase1",
        "2": "phase2",
        "phase2": "phase2",
        "traditional": "phase2",
        "backtesting": "phase2",
    }

    return phase_aliases.get(value, "")


def main():
    args = parse_args()
    project_id = args.project_id

    phase_key = normalize_phase_name(args.phase)
    if not phase_key:
        valid_phases = ", ".join(["phase1", "phase2", "all"])
        print(f"Unrecognized phase '{args.phase}'. Valid options: {valid_phases}.")
        sys.exit(1)

    phases_to_import = PHASE_SEQUENCE if phase_key == "all" else [phase_key]

    total_tasks_requested = sum(
        len(PHASE_DEFINITIONS[phase]["tasks"]) for phase in phases_to_import
    )

    print(
        f"Importing {total_tasks_requested} task(s) to project {project_id} "
        f"across {', '.join(PHASE_DEFINITIONS[phase]['label'] for phase in phases_to_import)}."
    )

    overall_success = 0
    for phase in phases_to_import:
        definition = PHASE_DEFINITIONS[phase]
        tasks = definition["tasks"]
        label = definition["label"]

        print(f"\n--- {label} ---")
        print(f"Total tasks to import: {len(tasks)}\n")

        phase_success = 0
        for task in tasks:
            if create_task(project_id, task):
                phase_success += 1

        overall_success += phase_success
        print(
            f"\nCompleted {phase_success}/{len(tasks)} task(s) for {label}."
        )

    print(f"\n{'=' * 60}")
    print(
        f"Import complete: {overall_success}/{total_tasks_requested} task(s) created"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
