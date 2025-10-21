# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-enhanced cryptocurrency trading bot built with Python 3.11+. The system combines traditional algorithmic trading with advanced AI components (LLMs, multi-agent decision making, and reinforcement learning) to execute trades on crypto exchanges (primarily Binance). The architecture emphasizes **modularity**, **asynchronous processing**, and **safety-first design**.

**Key Differentiators:**
- Multi-agent LLM system for collaborative decision-making (Analyst, Risk, Devil's Advocate, Supervisor agents)
- PPO-based reinforcement learning for risk-aware trade adjustments
- Multi-modal analysis: technical indicators + sentiment (Reddit, news, Fear & Greed) + LLM predictions
- Two-tier execution: fast rule-based path for obvious signals, AI-assisted path for nuanced decisions
- Comprehensive risk management with VaR/CVaR, slippage estimation, and emergency stop systems

## Core Architecture

### Module Organization

```
bot/
├── core/           # Core trading logic and strategies
│   ├── strategy.py          # Base strategy classes and StrategyContext
│   ├── technical_analyzer.py   # Indicator computation (RSI, MACD, Bollinger, etc.)
│   └── strategy_context.py     # Multi-strategy orchestration
├── ai/             # AI/ML components
│   ├── llm_manager.py          # Multi-provider LLM interface (OpenAI, Anthropic, local)
│   ├── ai_trend_predictor.py   # LLM-powered multi-timeframe forecasting
│   ├── sentiment_engine.py     # Multi-source sentiment aggregation
│   ├── decision_engine.py      # Multi-agent decision fusion
│   └── ppo_adjuster.py         # Risk-aware PPO reinforcement learning
├── risk/           # Risk management
│   ├── risk_manager.py         # Pre-trade checks, position sizing, VaR/CVaR
│   ├── var_calculator.py       # Value-at-Risk calculations
│   └── emergency_stop.py       # Kill-switch for catastrophic scenarios
├── execution/      # Order execution and exchange adapters
│   ├── orchestrator.py         # Decision router and order management
│   ├── adapters/
│   │   ├── binance.py          # Binance exchange adapter
│   │   └── base.py             # Base exchange interface
│   └── order_manager.py        # Order lifecycle management
├── data/           # Data handling
│   ├── market_data.py          # Market data streaming/polling
│   ├── quality_monitor.py      # Data validation and quality checks
│   └── backtesting.py          # Historical simulation and PPO training
├── config/         # Configuration
│   ├── models.py               # Pydantic configuration models
│   └── profiles/               # YAML configuration profiles
│       ├── dev.yaml
│       ├── paper.yaml
│       └── prod.yaml
├── interfaces/     # User interfaces
│   ├── api.py                  # FastAPI web service
│   ├── cli.py                  # Typer-based CLI
│   └── dashboard/              # Optional React dashboard
└── tests/          # Test suite
    ├── unit/
    ├── integration/
    └── backtest/
```

## Technology Stack

**Core:**
- Python 3.11+ (asyncio for concurrent operations)
- FastAPI (async web framework for API and monitoring)
- Pydantic (configuration validation)
- PyArrow (efficient in-memory data handling)

**Trading & Data:**
- CCXT (multi-exchange support)
- TA-Lib or `ta` (technical indicators)
- pandas (data manipulation)

**AI/ML:**
- OpenAI/Anthropic SDKs (LLM APIs)
- Stable-Baselines3 (PPO reinforcement learning)
- Optional: llama.cpp or HuggingFace Transformers (local models)

**Observability:**
- Prometheus (metrics)
- Grafana (dashboards)
- structlog or loguru (structured logging)
- python-telegram-bot (alerts)

**CLI & UI:**
- Typer (command-line interface)
- ReactJS (optional web dashboard)

## Development Commands

### Setup
```bash
# Install dependencies using uv (preferred) or pip
uv pip install -r requirements.txt

# Install dev dependencies
uv pip install -r requirements-dev.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Core Operations
```bash
# Start bot in paper trading mode
python -m bot.cli run --profile paper

# Start bot in live mode (use with caution!)
python -m bot.cli run --profile prod

# Run backtest on historical data
python -m bot.cli backtest --symbol BTCUSDT --strategy AITrend --start 2021-01-01 --end 2022-01-01

# List available trading symbols
python -m bot.cli symbols

# Validate configuration
python -m bot.cli config validate --profile paper
```

### Testing
```bash
# Run all tests
pytest

# Run specific test module
pytest tests/unit/test_risk_manager.py

# Run with coverage
pytest --cov=bot --cov-report=html

# Run integration tests (requires test exchange credentials)
pytest tests/integration/ -v
```

### Code Quality
```bash
# Format code
black bot/ tests/
ruff check --fix bot/ tests/

# Type checking
mypy bot/

# Run all checks
make ci  # Runs lint + typecheck + tests
```

## Configuration System

### Multi-Profile Support
The bot uses YAML configuration files with Pydantic validation. Profiles are located in `config/profiles/`:

- `dev.yaml` - Local development with paper trading
- `paper.yaml` - Paper trading with test credentials
- `prod.yaml` - Live trading (requires real credentials)

### Key Configuration Sections

```yaml
# Example structure (see config/profiles/ for full examples)
bot:
  name: "AI Trading Bot"
  environment: "paper"  # dev | paper | prod

exchange:
  primary: "binance"
  api_key: "${BINANCE_API_KEY}"  # From environment
  api_secret: "${BINANCE_API_SECRET}"
  testnet: true

llm:
  providers:
    - name: "openai"
      model: "gpt-4"
      api_key: "${OPENAI_API_KEY}"
    - name: "anthropic"
      model: "claude-3-sonnet"
      api_key: "${ANTHROPIC_API_KEY}"
  fallback_chain: ["openai", "anthropic", "local"]
  max_monthly_cost_usd: 100
  cache_ttl_minutes: 60

risk:
  max_position_size_pct: 0.03  # 3% per trade
  max_total_exposure_pct: 0.25  # 25% total
  max_daily_loss_pct: 0.02
  max_slippage_pct: 0.005
  var_confidence: 0.95

strategies:
  - name: "AITrendFollower"
    symbols: ["BTCUSDT", "ETHUSDT"]
    timeframes: ["1h", "4h", "1d"]
    enabled: true
```

### Environment Variables
Required in `.env` file:
```bash
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true  # Use testnet for development

OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Optional
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id
```

## Key Design Patterns

### 1. Two-Tier Decision Making
To handle LLM latency (2-8 seconds per call), the bot uses a dual-path approach:

- **Fast Path**: Rule-based decisions for obvious signals (execute in <100ms)
- **AI Path**: Multi-agent analysis for complex scenarios (15-60 seconds)

AI predictions are **pre-computed** every 5-15 minutes and cached, so execution doesn't wait for LLM calls.

### 2. Multi-Agent Decision System
Trade decisions flow through multiple specialized agents:

1. **Analyst Agent**: LLM-powered trend prediction using technical + sentiment data
2. **Risk Agent**: Evaluates risk metrics (VaR, correlation, position sizing)
3. **Devil's Advocate Agent**: Challenges the analyst's decision to reduce groupthink
4. **Supervisor Agent**: Final veto power, ensures consensus or flags conflicts

This mirrors a real trading desk structure and reduces false signals.

### 3. Risk-First Architecture
Every trade passes through **RiskManager** before execution:

- Pre-trade checks: slippage estimation, liquidity validation, order book depth
- Dynamic position sizing based on volatility (ATR) and portfolio risk
- Circuit breakers: halt trading after consecutive losses or drawdown thresholds
- Emergency stop system: automatic shutdown on catastrophic conditions

**Safe defaults**: Even if AI suggests aggressive trades, RiskManager enforces hard limits.

### 4. PPO-Based Risk Adjustment
Optional reinforcement learning module that fine-tunes decisions:

- Trained offline using backtesting engine
- Adjusts position sizes based on learned risk-reward patterns
- Can reduce confidence or skip trades when risk metrics are unfavorable
- Runs in advisory mode initially before full integration

### 5. Multi-Modal Data Fusion
AITrendPredictor combines:
- **Technical indicators**: RSI, MACD, Bollinger Bands, moving averages, ATR
- **Sentiment data**: Reddit (PRAW), news RSS feeds, Fear & Greed Index, exchange funding rates
- **LLM reasoning**: Synthesizes all inputs into actionable predictions with confidence scores

Each data source is weighted and cached to optimize cost and latency.

## Critical Safety Features

### Emergency Stop System
Monitors for:
- Flash crashes or extreme price moves
- Exchange API failures or connectivity issues
- Portfolio drawdown exceeding thresholds
- Data quality issues (stale data, NaN values)

Actions:
1. Cancel all open orders
2. Close all positions at market
3. Halt trading engine
4. Send alerts (Telegram, email)

### Data Quality Validation
Before executing trades, validate:
- Price sanity (no >10% spikes without cause)
- Volume anomalies (sudden 5x volume increase)
- Timestamp freshness (data not >60 seconds old)
- No missing data (NaN values)

### LLM Response Validation
- Circuit breaker: disable provider after 5 consecutive failures
- Response validator: check for hallucinations, invalid JSON, out-of-range values
- Structured output enforcement: strict JSON schema in system prompts
- Cost tracking: halt AI if monthly budget exceeded

## Implementation Phases

The project follows an 8-phase rollout (18-20 weeks total):

**Phase 1 (Weeks 1-3)**: Foundation - Exchange adapter, basic execution, RiskManager, emergency stop
**Phase 2 (Weeks 4-5)**: Technical strategies and backtesting framework
**Phase 3 (Weeks 6-7)**: LLM integration with multi-provider support
**Phase 4 (Weeks 8-9)**: Sentiment engine and multi-modal data fusion
**Phase 5 (Weeks 10-12)**: Multi-agent decision system
**Phase 6 (Weeks 13-15)**: PPO reinforcement learning and training
**Phase 7 (Weeks 16-17)**: Monitoring, dashboards, and UI
**Phase 8 (Weeks 18-20)**: Testing, deployment, and gradual rollout with real capital

**Current Phase**: Foundation (Phase 1) - Setting up project structure

## Testing Strategy

### Unit Tests
- Event bus load testing (1000+ messages/second)
- Risk manager checks (slippage, VaR, position limits)
- Order deduplication and idempotency
- Technical indicator calculations vs known results
- LLM manager fallback chain

### Integration Tests
- Binance testnet connectivity
- WebSocket stream handling
- End-to-end order placement and fill tracking
- Multi-agent decision flow

### Backtesting
- Run strategies on 2+ years of historical data
- Model realistic slippage based on order book depth
- Compare AI-enhanced vs traditional baseline
- Target: Traditional strategies achieve ~48-52% win rate before adding AI

### Paper Trading
Mandatory before live:
- Minimum 4-8 weeks of paper trading
- Test all failure scenarios (API outages, bad data, edge cases)
- Verify emergency stop triggers correctly
- Monitor for any unexpected behavior

### Gradual Rollout
Live trading progression:
1. Week 1: $100 capital
2. Week 2-3: $500 if no major issues
3. Month 2+: Gradually scale to target capital
4. Only increase after 3+ profitable months

## Monitoring & Observability

### Structured Logging
- JSON logs with correlation IDs for trade lifecycle tracing
- Log levels: DEBUG (development), INFO (production), ERROR (alerts)
- Rotation: daily with 30-day retention

### Prometheus Metrics
Exposed at `/metrics`:
- Portfolio value, PnL, win rate
- Trade counts and frequency
- LLM call latency and costs
- API failure rates
- Emergency stop triggers

### Grafana Dashboards
Key panels:
- Real-time P/L and open positions
- AI prediction confidence over time
- Risk metrics (current VaR, exposure)
- System health (API latency, data freshness)

### Telegram Alerts
Instant notifications for:
- Emergency stop triggered
- Large loss events (>2% single trade)
- LLM budget approaching limit
- Consecutive losses threshold
- Daily performance summaries

## Common Development Tasks

### Adding a New Strategy
1. Create class in `core/strategy.py` inheriting from `StrategyBase`
2. Implement `generate_signals()` method
3. Add configuration in `config/profiles/{profile}.yaml`
4. Write unit tests in `tests/unit/test_strategies.py`
5. Backtest before deploying: `python -m bot.cli backtest --strategy YourStrategy`

### Adding a New Exchange
1. Create adapter in `execution/adapters/{exchange}.py` implementing base interface
2. Add exchange config section in YAML
3. Update `execution/orchestrator.py` to route to new exchange
4. Test connectivity in paper mode

### Adding a New LLM Provider
1. Add provider config in `config/profiles/{profile}.yaml`
2. Implement adapter in `ai/llm_manager.py`
3. Add to fallback chain
4. Test with dummy predictions

### Tuning Risk Parameters
Edit `config/profiles/{profile}.yaml`:
```yaml
risk:
  max_position_size_pct: 0.03  # Adjust per-trade risk
  max_daily_loss_pct: 0.02     # Adjust daily stop-loss
  var_confidence: 0.95         # Adjust VaR confidence level
```

No code changes needed - restart bot with new config.

## Performance Expectations

### Realistic Targets (Year 1)
- Monthly return: 1-3% (not 4-6% as initially projected)
- Win rate: 48-54% (not 58-65%)
- Sharpe ratio: 0.8-1.3 (not 2.08)
- Max drawdown: -15% to -25% (not -7.9%)

**Success criteria:**
- Beat buy-and-hold strategy
- Sharpe > 0.7
- Survive 6+ months without catastrophic loss
- Max drawdown < -30%

### AI Enhancement Goals
- Sentiment analysis: +2-5% win rate improvement over baseline
- Multi-agent system: Reduce false positives, improve decision confidence
- PPO adjuster: +0.1-0.2 Sharpe ratio improvement through risk optimization

**Note**: LLMs are not magic - they analyze sentiment and synthesize data but cannot predict prices with high accuracy. Treat AI as an enhancement to solid traditional strategies, not a replacement.

## Security & Best Practices

### API Key Management
- Never commit API keys to git
- Use environment variables or secret management tools
- Rotate keys regularly
- Use IP allowlists on exchange accounts

### Pre-Mainnet Checklist
- [ ] IP allowlist configured on exchange
- [ ] System time synchronized (critical for signatures)
- [ ] Fee tier verified
- [ ] Minimum notional requirements checked
- [ ] Paper trading >4 weeks without issues
- [ ] Emergency stop tested
- [ ] Monitoring and alerts configured
- [ ] Backup and recovery procedures documented

### Production Deployment
- Use Docker for consistent environments
- Deploy on GCP or similar cloud provider
- Set up health checks and auto-restart
- Configure Prometheus/Grafana monitoring
- Enable log aggregation
- Test failover scenarios

## Troubleshooting

### Common Issues

**LLM calls timing out:**
- Check cache hit rate - should be >70%
- Reduce frequency of AI predictions
- Enable fast-path execution for more scenarios
- Consider using faster, smaller models

**High slippage on trades:**
- Increase `min_liquidity_ratio` in risk config
- Trade only during high-volume periods
- Reduce position sizes
- Check order book depth before execution

**Emergency stop triggering frequently:**
- Review logs for root cause
- Adjust circuit breaker thresholds if too sensitive
- Verify data quality checks aren't too strict
- Check exchange connectivity stability

**Backtests don't match live performance:**
- Ensure slippage modeling is realistic
- Check that data feed matches live source
- Verify execution delays are modeled
- Look for overfitting in strategy parameters

## Support & Documentation

- Technical design: [AI-Enhanced Crypto Trading Bot_ Technical Design & Implementation Plan.pdf](AI-Enhanced Crypto Trading Bot_ Technical Design & Implementation Plan.pdf)
- Expert review: [Claude_ai-trading-bot-review.md](Claude_ai-trading-bot-review.md)
- API docs: Available at `/docs` when FastAPI server is running
- Logs: `logs/` directory with daily rotation

**Philosophy**: This bot prioritizes **not losing money** over making money. 10% strategy, 90% risk management. Every decision passes through multiple safety layers. When in doubt, the system errs on the side of caution.
