# Trading Strategy Module

**Location**: `bot/core/strategy.py`

## Overview

The strategy module provides a flexible framework for implementing technical trading strategies. It includes:

- **Base Strategy Class**: Abstract interface for all strategies
- **RSI Strategy**: Trades based on Relative Strength Index (oversold/overbought)
- **MA Crossover Strategy**: Trades based on Moving Average crossovers (golden/death cross)
- **Position Management**: Entry/exit logic with stop loss and take profit
- **Backtesting Support**: Reset functionality for iterative backtesting

## Key Features

✅ **Configurable Parameters**: All strategies support custom configuration
✅ **Position Management**: Automatic stop loss and take profit calculation
✅ **Signal Generation**: BUY, SELL, HOLD signals with confidence scores
✅ **Metadata Rich**: All signals include indicator values and reasons
✅ **Backtesting Ready**: Reset functionality for iterative testing
✅ **Type Safe**: Full type hints and Pydantic configuration models

## Quick Start

### RSI Strategy

```python
from bot.core.strategy import RSIStrategy, Signal
import pandas as pd

# Create strategy with default config
strategy = RSIStrategy()

# Or with custom config
config = {
    "rsi_period": 14,
    "oversold_threshold": 30,
    "overbought_threshold": 70,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
}
strategy = RSIStrategy(config=config)

# Generate signal from OHLCV data
signal = strategy.generate_signal(ohlcv_data)

if signal.signal == Signal.BUY:
    # Calculate position size
    size = strategy.get_position_size(signal, capital=10000)

    # Enter position
    position = strategy.enter_position(signal, size)

    print(f"Entered {position.side} at {position.entry_price}")
    print(f"Stop Loss: {position.stop_loss}")
    print(f"Take Profit: {position.take_profit}")
```

### MA Crossover Strategy

```python
from bot.core.strategy import MovingAverageCrossoverStrategy

# Create strategy
config = {
    "fast_period": 20,
    "slow_period": 50,
    "ma_type": "sma",  # or "ema"
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.04,
}
strategy = MovingAverageCrossoverStrategy(config=config)

# Generate signals
signal = strategy.generate_signal(ohlcv_data)

print(f"Signal: {signal.signal}")
print(f"Reason: {signal.reason}")
print(f"Fast MA: {signal.metadata['fast_ma']}")
print(f"Slow MA: {signal.metadata['slow_ma']}")
```

## Strategy Details

### RSI Strategy

**Logic**:
- **BUY** when RSI < oversold_threshold (default 30) - market is oversold
- **SELL** when RSI > overbought_threshold (default 70) - market is overbought
- **HOLD** otherwise - RSI in neutral zone

**Configuration Parameters**:
```python
{
    "rsi_period": 14,              # RSI calculation period
    "oversold_threshold": 30,      # Buy threshold
    "overbought_threshold": 70,    # Sell threshold
    "stop_loss_pct": 0.02,         # 2% stop loss
    "take_profit_pct": 0.04,       # 4% take profit
    "position_size_pct": 0.1,      # 10% of capital per trade
}
```

**Signal Metadata**:
- `rsi`: Current RSI value
- `oversold_threshold`: Configured oversold threshold
- `overbought_threshold`: Configured overbought threshold

**Use Cases**:
- Mean reversion trading
- Identifying oversold/overbought conditions
- Short-term trading in ranging markets

### MA Crossover Strategy

**Logic**:
- **BUY** when fast MA crosses above slow MA (golden cross) - bullish signal
- **SELL** when fast MA crosses below slow MA (death cross) - bearish signal
- **HOLD** when no crossover detected

**Configuration Parameters**:
```python
{
    "fast_period": 20,             # Fast MA period
    "slow_period": 50,             # Slow MA period
    "ma_type": "sma",              # "sma" or "ema"
    "stop_loss_pct": 0.02,         # 2% stop loss
    "take_profit_pct": 0.04,       # 4% take profit
    "position_size_pct": 0.1,      # 10% of capital per trade
}
```

**Signal Metadata**:
- `fast_ma`: Current fast MA value
- `slow_ma`: Current slow MA value
- `fast_period`: Fast MA period
- `slow_period`: Slow MA period
- `ma_type`: MA type (sma/ema)

**Use Cases**:
- Trend following
- Identifying trend reversals
- Medium to long-term trading

## Position Management

### Automatic Stop Loss & Take Profit

Both strategies automatically calculate stop loss and take profit levels:

```python
# For LONG positions
stop_loss = entry_price * (1 - stop_loss_pct)
take_profit = entry_price * (1 + take_profit_pct)

# For SHORT positions
stop_loss = entry_price * (1 + stop_loss_pct)
take_profit = entry_price * (1 - take_profit_pct)
```

### Manual Override

You can override automatic levels:

```python
position = strategy.enter_position(
    signal,
    size=0.5,
    stop_loss=49000.0,      # Manual stop loss
    take_profit=52000.0,    # Manual take profit
)
```

### Exit Conditions

Check if position should be exited:

```python
should_exit, reason = strategy.should_exit(current_price, position)

if should_exit:
    exit_details = strategy.exit_position(
        exit_price=current_price,
        exit_timestamp=timestamp,
        reason=reason
    )

    print(f"PnL: ${exit_details['pnl']:.2f}")
    print(f"PnL %: {exit_details['pnl_pct']:.2%}")
```

## Backtesting

Strategies support backtesting through the `reset()` method:

```python
strategy = RSIStrategy()
trades = []

for window in rolling_windows:
    signal = strategy.generate_signal(window)

    # Entry logic
    if signal.signal == Signal.BUY and not strategy.current_position:
        position = strategy.enter_position(signal, size)

    # Exit logic
    elif strategy.current_position:
        should_exit, reason = strategy.should_exit(current_price, strategy.current_position)
        if should_exit:
            exit_details = strategy.exit_position(current_price, timestamp, reason)
            trades.append(exit_details)

# Reset for next backtest run
strategy.reset()
```

See `bot/core/examples/strategy_example.py` for a complete backtesting example.

## Data Requirements

All strategies require OHLCV data in pandas DataFrame format:

```python
data = pd.DataFrame({
    "timestamp": [...],  # Unix timestamp in milliseconds
    "open": [...],       # Open prices
    "high": [...],       # High prices
    "low": [...],        # Low prices
    "close": [...],      # Close prices
    "volume": [...],     # Trading volume
})
```

**Minimum Data Requirements**:
- RSI Strategy: Requires at least `rsi_period` periods (default 14)
- MA Crossover: Requires at least `slow_period` periods (default 50)

## Signal Structure

All signals follow the `StrategySignal` structure:

```python
@dataclass
class StrategySignal:
    signal: Signal              # BUY, SELL, or HOLD
    timestamp: int              # Signal timestamp (ms)
    price: float                # Current price
    confidence: float           # Confidence score (0-1)
    metadata: dict              # Strategy-specific data
    reason: str                 # Human-readable reason
```

**Confidence Scores**:
- RSI: Based on distance from threshold (0.0 - 1.0)
- MA Crossover: Based on distance between MAs
- HOLD signals typically have confidence of 0.5

## Configuration with Pydantic

Strategies can be configured through Pydantic models:

```python
from bot.config.models import StrategyConfig, StrategyType

config = StrategyConfig(
    type=StrategyType.RSI,
    enabled=True,
    parameters={
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
    }
)

# Create strategy from config
if config.type == StrategyType.RSI:
    strategy = RSIStrategy(config=config.parameters)
elif config.type == StrategyType.MA_CROSSOVER:
    strategy = MovingAverageCrossoverStrategy(config=config.parameters)
```

## Creating Custom Strategies

Extend `BaseStrategy` to create custom strategies:

```python
from bot.core.strategy import BaseStrategy, Signal, StrategySignal

class MyCustomStrategy(BaseStrategy):
    def __init__(self, config: dict | None = None):
        super().__init__(name="My Custom Strategy", config=config)
        # Initialize your indicators

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        # Implement your signal generation logic
        if condition:
            return StrategySignal(
                signal=Signal.BUY,
                timestamp=timestamp,
                price=price,
                confidence=0.8,
                metadata={"indicator_value": value},
                reason="Custom condition met"
            )
        return StrategySignal(
            signal=Signal.HOLD,
            timestamp=timestamp,
            price=price,
            confidence=0.5,
            metadata={},
            reason="No signal"
        )

    def should_exit(self, current_price: float, position: Position) -> tuple[bool, str]:
        # Implement your exit logic
        # Check stop loss and take profit
        # Return (should_exit, reason)
        return False, ""
```

## Testing

Comprehensive test suite in `tests/unit/core/test_strategy.py`:

```bash
# Run strategy tests
pytest tests/unit/core/test_strategy.py -v

# Run with coverage
pytest tests/unit/core/test_strategy.py --cov=bot.core.strategy
```

**Test Coverage**:
- Signal generation (BUY, SELL, HOLD)
- Position entry/exit
- Stop loss and take profit
- Configuration validation
- Edge cases (insufficient data, etc.)
- Backtesting workflows

## Performance Considerations

**Memory Efficiency**:
- Strategies only store current position state
- Historical data is passed by reference
- Use `reset()` to clear state between backtest runs

**Computational Efficiency**:
- RSI: O(n) where n is rsi_period
- MA Crossover: O(n) where n is slow_period
- Indicators are calculated using optimized `ta` library

**Concurrency**:
- Strategies are NOT thread-safe
- Create separate instances for concurrent execution
- Use locks if sharing strategy instances

## Integration with Trading Pipeline

```python
from bot.core.strategy import RSIStrategy
from bot.data.market_data import MarketDataStream
from bot.execution.adapters.binance import BinanceAdapter
from bot.risk.risk_manager import RiskManager

# Initialize components
strategy = RSIStrategy()
market_data = MarketDataStream()
exchange = BinanceAdapter()
risk_manager = RiskManager()

# Trading loop
async def trading_loop():
    # Get latest data
    data = await market_data.buffer.get_latest("BTC/USDT", limit=50)

    # Generate signal
    signal = strategy.generate_signal(data)

    if signal.signal == Signal.BUY and not strategy.current_position:
        # Check risk constraints
        if risk_manager.can_open_position():
            # Calculate position size
            capital = await exchange.get_balance()
            size = strategy.get_position_size(signal, capital)

            # Execute order
            order = await exchange.place_market_order("BTC/USDT", "buy", size)

            # Enter position in strategy
            position = strategy.enter_position(signal, size)

    elif strategy.current_position:
        current_price = data["close"].iloc[-1]
        should_exit, reason = strategy.should_exit(current_price, strategy.current_position)

        if should_exit:
            # Execute exit order
            await exchange.place_market_order("BTC/USDT", "sell", strategy.current_position.size)

            # Exit position in strategy
            exit_details = strategy.exit_position(current_price, timestamp, reason)
```

## References

- **Technical Indicators**: Uses `ta` library - [https://github.com/bukosabino/ta](https://github.com/bukosabino/ta)
- **RSI**: Relative Strength Index - [Investopedia](https://www.investopedia.com/terms/r/rsi.asp)
- **MA Crossover**: Moving Average Crossover - [Investopedia](https://www.investopedia.com/articles/active-trading/052014/how-use-moving-average-buy-stocks.asp)

## Next Steps

1. **Add More Strategies**: Implement MACD, Bollinger Bands, etc.
2. **Parameter Optimization**: Use grid search or genetic algorithms
3. **Multi-Strategy**: Combine multiple strategies with voting
4. **Machine Learning**: Integrate ML models for signal enhancement
5. **Real-time Monitoring**: Add Prometheus metrics for strategy performance

## Support

For examples, see:
- `bot/core/examples/strategy_example.py` - Complete usage examples
- `tests/unit/core/test_strategy.py` - Comprehensive test suite

For issues or questions, refer to the main project documentation.
