# AI-Enhanced Cryptocurrency Trading Bot

An advanced cryptocurrency trading bot that combines traditional algorithmic trading with cutting-edge AI technologies:
- Multi-agent LLM system for collaborative decision-making
- PPO-based reinforcement learning for risk-aware trade optimization
- Multi-modal analysis (technical indicators + sentiment + AI predictions)
- Comprehensive risk management with emergency stop systems

## Current Status

**Phase**: Foundation (Phase 1)
**Version**: 0.1.0
**Status**: Initial setup and architecture design

## Quick Start

### Prerequisites
- Python 3.11+
- Exchange API keys (Binance testnet recommended for development)
- LLM API keys (OpenAI or Anthropic)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd bot/potato-trading-bot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Running the Bot

```bash
# Paper trading (recommended for testing)
python -m bot.cli run --profile paper

# Backtest a strategy
python -m bot.cli backtest --symbol BTCUSDT --start 2021-01-01 --end 2022-01-01
```

## Documentation

- **[CLAUDE.md](../CLAUDE.md)** - Comprehensive guide for developers and Claude Code
- **[Technical Design](../AI-Enhanced%20Crypto%20Trading%20Bot_%20Technical%20Design%20&%20Implementation%20Plan.pdf)** - Full architecture and implementation plan
- **[Expert Review](../Claude_ai-trading-bot-review.md)** - Critical review and realistic expectations
- **[Task Breakdown](../TASK_BREAKDOWN.md)** - Detailed task breakdown for all phases

## Architecture

```
AI Trading Intelligence Layer
├── LLM Core (Multi-provider: GPT-4, Claude)
├── Sentiment Analyzer (Reddit, News, Fear & Greed)
├── Technical Analyzer (RSI, MACD, Bollinger, etc.)
├── Decision Fusion Engine (Multi-agent orchestration)
└── Risk-Aware PPO Adjuster (Reinforcement Learning)

Trading Execution Layer
├── Risk Manager (VaR/CVaR, position sizing, pre-trade checks)
├── Execution Orchestrator (Multi-exchange support)
└── Emergency Stop System (Kill-switch for disasters)
```

## Key Features

### Multi-Agent Decision System
- **Analyst Agent**: LLM-powered trend predictions
- **Risk Agent**: Quantitative risk evaluation
- **Devil's Advocate**: Challenges consensus to reduce groupthink
- **Supervisor Agent**: Final approval with veto power

### Risk Management
- Pre-trade checks: slippage, liquidity, order book depth
- Dynamic position sizing based on volatility
- Circuit breakers and emergency stop
- VaR/CVaR calculations

### Two-Tier Execution
- Fast path (<100ms) for obvious signals
- AI-assisted path (5-60s) for complex analysis
- Pre-computed predictions to minimize latency

## Development Roadmap

- [x] **Phase 1** (Weeks 1-3): Foundation and core trading engine
- [ ] **Phase 2** (Weeks 4-5): Traditional strategies and backtesting
- [ ] **Phase 3** (Weeks 6-7): LLM integration
- [ ] **Phase 4** (Weeks 8-9): Multi-modal sentiment analysis
- [ ] **Phase 5** (Weeks 10-12): Multi-agent decision system
- [ ] **Phase 6** (Weeks 13-15): PPO reinforcement learning
- [ ] **Phase 7** (Weeks 16-17): Monitoring and UI
- [ ] **Phase 8** (Weeks 18-20): Testing and gradual rollout

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bot --cov-report=html

# Run specific test module
pytest tests/unit/test_risk_manager.py -v
```

## Safety & Risk

⚠️ **Important**: This bot trades with real money. Please:

1. **Start with testnet** - Use Binance testnet for all development
2. **Paper trade extensively** - Minimum 4-8 weeks before live
3. **Start small** - Begin with $100-500 in live mode
4. **Monitor closely** - Check daily during first month
5. **Set strict limits** - Configure conservative risk parameters

**Realistic Expectations**:
- Monthly returns: 1-3% (not 4-6%)
- Win rate: 48-54% (not 58-65%)
- Max drawdown: -15% to -25% (not -7.9%)

**Success means**: Beating buy-and-hold, surviving 6+ months, Sharpe > 0.7

## Configuration

The bot uses YAML configuration profiles in `config/profiles/`:

- `dev.yaml` - Local development
- `paper.yaml` - Paper trading
- `prod.yaml` - Live trading

See [CLAUDE.md](../CLAUDE.md) for detailed configuration options.

## Monitoring

- **Logs**: Structured JSON logs in `logs/` directory
- **Metrics**: Prometheus metrics at `:9090/metrics`
- **Dashboard**: Grafana dashboards (optional)
- **Alerts**: Telegram notifications for critical events

## Contributing

This is currently a private project. For collaboration:

1. Follow the phased implementation plan
2. Write tests for all new features
3. Run `make ci` before committing
4. Never commit API keys or secrets

## License

Proprietary - All rights reserved

## Support

For issues or questions, refer to:
- [CLAUDE.md](../CLAUDE.md) for development guidance
- [Technical design](../AI-Enhanced%20Crypto%20Trading%20Bot_%20Technical%20Design%20&%20Implementation%20Plan.pdf) for architecture details
- [Expert review](../Claude_ai-trading-bot-review.md) for realistic expectations

---

**Philosophy**: This bot prioritizes **not losing money** over making money. 10% strategy, 90% risk management.
