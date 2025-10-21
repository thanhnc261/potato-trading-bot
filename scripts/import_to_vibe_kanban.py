#!/usr/bin/env python3
"""
Import tasks to vibe-kanban for Potato Trading Bot project.

Usage:
    python scripts/import_to_vibe_kanban.py <project_id>
"""

import sys
import requests
from typing import List, Dict

VIBE_KANBAN_API = "http://127.0.0.1:52718/api"

# Phase 1 Tasks
PHASE_1_TASKS = [
    {
        "title": "Setup Exchange Adapter",
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
        "title": "Market Data Streaming",
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
        "title": "Logging Infrastructure",
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
        "title": "RiskManager - Pre-trade Checks",
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
        "title": "Emergency Stop System",
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
        "title": "Basic Technical Strategy",
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
        "title": "Execution Orchestrator",
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
        "title": "Data Quality Monitor",
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/import_to_vibe_kanban.py <project_id>")
        print("\nFirst, create a project in vibe-kanban:")
        print("  Name: Potato Trading Bot")
        print("  Git Repo: /Users/thanhnguyen/Dev/Coin/bot")
        print("\nThen run this script with the project ID.")
        sys.exit(1)

    project_id = sys.argv[1]

    print(f"Importing Phase 1 tasks to project {project_id}...")
    print(f"Total tasks to import: {len(PHASE_1_TASKS)}\n")

    success_count = 0
    for task in PHASE_1_TASKS:
        if create_task(project_id, task):
            success_count += 1

    print(f"\n{'='*60}")
    print(f"Import complete: {success_count}/{len(PHASE_1_TASKS)} tasks created")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
