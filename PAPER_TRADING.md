# Paper Trading Mode Documentation

This document describes the paper trading (simulated trading) functionality implemented for the Potato Trading Bot.

## Overview

Paper trading allows you to test trading strategies with virtual money in real market conditions without financial risk. The system:

- ✅ Simulates a complete exchange environment in-memory
- ✅ Connects to live market data feeds
- ✅ Executes virtual orders with realistic slippage and commissions
- ✅ Tracks portfolio performance and P/L
- ✅ Provides comprehensive reporting and metrics

## Quick Start

### 1. Create Configuration File

Create a `config_paper.yaml` file (see `config_paper.yaml` in the repository for a complete example):

```yaml
name: "My Trading Bot - Paper Mode"
environment: "paper"

exchange:
  name: "binance"
  api_key: "dummy_key"  # Not used for paper trading but required
  api_secret: "dummy_secret"
  testnet: true

strategy:
  type: "rsi"
  enabled: true
  parameters:
    rsi_period: 14
    oversold_threshold: 30
    overbought_threshold: 70
    stop_loss_pct: 0.02
    take_profit_pct: 0.04
    position_size_pct: 0.10

risk:
  max_position_size_pct: 0.10
  max_daily_loss_pct: 0.05
```

### 2. Run Paper Trading

There are two CLI commands available:

#### Option 1: Using `bot run --profile paper`

```bash
python -m bot.cli run --profile paper --config config_paper.yaml
```

With custom settings:

```bash
python -m bot.cli run --profile paper \
  --config config_paper.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --capital 50000 \
  --max-runtime 3600
```

#### Option 2: Using `bot paper` command

```bash
python -m bot.cli paper --config config_paper.yaml
```

With custom settings:

```bash
python -m bot.cli paper \
  --config config_paper.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --capital 50000 \
  --interval 1.0 \
  --max-runtime 3600
```

### 3. Monitor Performance

The bot will log performance metrics every 60 seconds:

```
performance_report:
  initial_value: 10000.0
  current_value: 10245.50
  pnl: 245.50
  pnl_pct: 2.455
  total_trades: 5
  total_commission: 12.45
  total_slippage: 8.23
```

### 4. Stop and View Results

Press `Ctrl+C` to stop. A comprehensive session report will be saved to:

```
paper_trading_results/session_YYYYMMDD_HHMMSS.json
```

## Architecture

### Components

#### 1. **SimulatedExchange** (`bot/execution/adapters/simulated.py`)

Complete in-memory exchange simulator that:
- Maintains virtual portfolio balances
- Executes orders with realistic slippage
- Calculates commissions and fees
- Tracks order lifecycle (pending → filled)
- Provides P/L metrics

Key Features:
```python
# Initialize with virtual capital
exchange = SimulatedExchange(
    initial_balances={"USDT": Decimal("10000")},
    commission_rate=0.001,  # 0.1%
    slippage_factor=0.001,  # 0.1%
)

# Update market prices (from live data)
exchange.update_market_price("BTCUSDT", Decimal("50000"))

# Execute orders
order = await exchange.create_order(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("0.1"),
)

# Get performance metrics
metrics = exchange.get_performance_metrics()
```

#### 2. **PaperTradingRunner** (`bot/execution/paper_trading.py`)

Orchestrates the entire paper trading session:
- Connects to live market data streams
- Runs strategy signal generation
- Executes orders through simulated exchange
- Tracks performance in real-time
- Generates comprehensive reports

Main Loop:
```python
while running:
    1. Receive market data ticks
    2. Update price buffers
    3. Generate trading signals
    4. Execute orders if signal detected
    5. Report performance metrics
    6. Sleep for update interval
```

#### 3. **Market Data Integration**

Uses the existing `MarketDataStream` to fetch live prices:
- Connects to Binance (or other exchanges via CCXT)
- Subscribes to symbol tickers
- Buffers OHLCV data for strategy analysis
- Updates simulated exchange prices

### Realistic Order Execution

The simulator models realistic order execution:

#### Slippage Calculation

```python
def _calculate_slippage(price, side, quantity, volatility):
    # Base slippage (0.1% default)
    base = price * 0.001

    # Volatility impact
    vol = price * volatility * 0.5

    # Order size impact
    size = price * min(quantity / 100, 0.002)

    # Random component
    random = base * random.uniform(0, 0.5)

    total = base + vol + size + random

    # Always unfavorable
    if side == BUY:
        execution_price = price + total
    else:
        execution_price = price - total
```

#### Commission

- Taker fee: 0.1% (default)
- Maker fee: 0.09% (default)
- Applied to notional value

## CLI Commands

### `python -m bot.cli run --profile paper`

Main command supporting both paper and live trading (live not yet implemented).

**Required Arguments:**
- `--config, -c`: Path to configuration file

**Optional Arguments:**
- `--profile, -p`: Trading profile (default: `paper`)
  - `paper`: Simulated trading
  - `live`: Real trading (not implemented)
- `--symbols, -s`: Comma-separated symbols (default: `BTCUSDT`)
- `--capital`: Initial virtual capital (default: `10000`)
- `--max-runtime`: Maximum runtime in seconds (optional)

**Examples:**

```bash
# Basic usage
python -m bot.cli run --profile paper --config config.yaml

# Multiple symbols with custom capital
python -m bot.cli run --profile paper \
  -c config.yaml \
  -s BTCUSDT,ETHUSDT,BNBUSDT \
  --capital 100000

# Run for 1 hour
python -m bot.cli run --profile paper \
  -c config.yaml \
  --max-runtime 3600
```

### `python -m bot.cli paper`

Dedicated paper trading command with simplified interface.

**Required Arguments:**
- `--config, -c`: Path to configuration file

**Optional Arguments:**
- `--symbols, -s`: Comma-separated symbols (default: `BTCUSDT`)
- `--capital`: Initial virtual capital (default: `10000`)
- `--interval`: Market data update interval in seconds (default: `1.0`)
- `--max-runtime`: Maximum runtime in seconds (optional)
- `--profile`: Profile name for logging (default: `paper`)

**Examples:**

```bash
# Basic usage
python -m bot.cli paper --config config.yaml

# Custom settings
python -m bot.cli paper \
  --config config.yaml \
  --symbols BTCUSDT,ETHUSDT \
  --capital 50000 \
  --interval 0.5
```

## Session Reports

When you stop a paper trading session, a comprehensive JSON report is generated:

```json
{
  "config": {
    "symbols": ["BTCUSDT"],
    "initial_capital": 10000.0,
    "strategy": "rsi",
    "strategy_params": {
      "rsi_period": 14,
      "oversold_threshold": 30,
      "overbought_threshold": 70
    }
  },
  "metrics": {
    "session": {
      "start_time": "2025-10-28T10:00:00",
      "duration_seconds": 3600,
      "signals_generated": 15,
      "orders_executed": 10
    },
    "performance": {
      "initial_value": 10000.0,
      "current_value": 10245.50,
      "pnl": 245.50,
      "pnl_pct": 2.455,
      "total_trades": 5,
      "total_commission": 12.45,
      "total_slippage": 8.23
    }
  },
  "trades": [
    {
      "id": "trade_abc123",
      "symbol": "BTCUSDT",
      "side": "buy",
      "price": 50125.50,
      "quantity": 0.1,
      "commission": 5.01,
      "timestamp": "2025-10-28T10:15:23"
    }
  ]
}
```

## Risk Management Integration

Paper trading integrates with the existing risk management system:

- ✅ Position size limits
- ✅ Total exposure limits
- ✅ Daily loss limits
- ✅ Slippage checks
- ✅ Liquidity checks

All risk checks are applied to virtual orders before execution.

## Performance Metrics

### Portfolio Metrics

- **Initial Value**: Starting capital
- **Current Value**: Mark-to-market portfolio value
- **P/L**: Absolute profit/loss
- **P/L %**: Percentage return
- **Total Trades**: Number of completed trades
- **Total Commission**: Total fees paid
- **Total Slippage**: Total slippage costs

### Trading Metrics

- **Signals Generated**: Total trading signals
- **Orders Executed**: Successfully executed orders
- **Open Orders**: Currently pending orders
- **Win Rate**: (calculated from trade history)
- **Profit Factor**: (calculated from trade history)

## Limitations and Considerations

### Current Limitations

1. **Long Positions Only**: Currently only supports long (buy) positions
2. **Market Orders Only**: Limit orders placed but not actively monitored for fills
3. **Single Symbol Per Strategy**: Each strategy instance handles one symbol
4. **No Order Book Depth**: Slippage is modeled but not based on real order book

### Future Enhancements

- [ ] Short selling support
- [ ] Limit order matching engine
- [ ] Multi-symbol strategies
- [ ] Order book depth simulation
- [ ] Advanced slippage models
- [ ] Portfolio rebalancing

## Testing

The implementation has been verified with:

```bash
# Code formatting
make lint

# Type checking
make typecheck

# Unit tests (20 passed)
make test-all
```

All tests pass except for one pre-existing test in `test_notifications.py` (unrelated to paper trading).

## Integration with Existing Code

Paper trading seamlessly integrates with:

- ✅ **Strategies**: RSI, MA Crossover, custom strategies
- ✅ **Risk Manager**: Full risk management system
- ✅ **Execution Orchestrator**: Order lifecycle management
- ✅ **Market Data**: Live data streaming
- ✅ **Configuration**: Unified configuration system
- ✅ **Logging**: Structured logging with correlation IDs

## Troubleshooting

### "No market price available"

**Problem**: Exchange not receiving price updates

**Solution**: Ensure market data stream is connected and subscribed to correct symbols

```python
# Check connection status
await market_stream.connect()
await market_stream.subscribe(["BTCUSDT"])
```

### "Insufficient balance"

**Problem**: Virtual account doesn't have enough capital

**Solution**: Increase initial capital or reduce position sizes

```yaml
# In config file
risk:
  position_size_pct: 0.05  # Reduce to 5%
```

### Orders not executing

**Problem**: Risk checks failing

**Solution**: Review risk management logs to see which checks are failing

```bash
# Check logs
tail -f logs/bot.log | grep risk_check
```

## Example Session

Here's a typical paper trading session:

```bash
$ python -m bot.cli run --profile paper --config config_paper.yaml

Starting Paper Trading Session

Configuration: config_paper.yaml
Profile: PAPER
Symbols: BTCUSDT
Initial Capital: $10,000.00 USDT
Strategy: rsi
Max Runtime: Unlimited (Ctrl+C to stop)

Starting bot... Press Ctrl+C to stop

[INFO] paper_trading_session_starting
[INFO] simulated_exchange_connected initial_value=10000.0
[INFO] market_data_stream_connected exchange=binance
[INFO] trading_loop_started

[INFO] entering_position symbol=BTCUSDT side=LONG size=0.1 price=50000
[INFO] position_entered_successfully order_id=1

[INFO] performance_report pnl=125.50 pnl_pct=1.255 total_trades=1

[INFO] exiting_position symbol=BTCUSDT reason=take_profit
[INFO] position_exited_successfully pnl=195.30

^C
[INFO] keyboard_interrupt_received
[INFO] paper_trading_session_stopping
[INFO] session_report_saved output_path=paper_trading_results/session_20251028_120000.json

Paper Trading Session Completed!
```

## Conclusion

The paper trading system provides a complete simulated trading environment for strategy development and testing. It combines realistic order execution, live market data, and comprehensive performance tracking to help validate strategies before risking real capital.

For questions or issues, please refer to the main project documentation or open an issue on GitHub.
