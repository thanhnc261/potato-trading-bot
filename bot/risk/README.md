# Risk Management Module

Comprehensive risk management system with pre-trade validation, position sizing, and portfolio monitoring.

## Features

### 1. Pre-Trade Validation Checks

The RiskManager performs seven critical checks before allowing any trade:

#### Time-Based Restrictions
- **Purpose**: Prevent trading during off-hours or low-liquidity periods
- **Configurable**: Trading hours (start/end time) and allowed days
- **Default**: 24/7 trading enabled
- **Use Case**: Avoid weekends or specific hours with low market activity

#### Position Size Validation
- **Purpose**: Limit individual position size to prevent over-concentration
- **Parameter**: `max_position_size_pct` (default: 3% of portfolio)
- **Check**: Position value ≤ max_position_value
- **Use Case**: Risk no more than 3% of capital on a single trade

#### Total Exposure Validation
- **Purpose**: Limit aggregate exposure across all open positions
- **Parameter**: `max_total_exposure_pct` (default: 25% of portfolio)
- **Check**: Sum of all positions + new position ≤ max_exposure
- **Use Case**: Prevent over-leveraging the portfolio

#### Slippage Estimation
- **Purpose**: Estimate price impact based on order book depth
- **Parameter**: `max_slippage_pct` (default: 0.5%)
- **Method**: Analyzes top 50 order book levels to calculate weighted average execution price
- **Check**: Estimated slippage ≤ max_slippage
- **Use Case**: Reject trades that would move the market significantly

#### Liquidity Validation
- **Purpose**: Ensure position size is appropriate for market liquidity
- **Parameter**: `min_liquidity_ratio` (default: 1% of daily volume)
- **Method**: Compares position value to 24-hour trading volume
- **Check**: Position value / daily_volume ≤ min_liquidity_ratio
- **Use Case**: Prevent taking positions that are too large relative to market depth

#### Portfolio Stop-Loss
- **Purpose**: Global circuit breaker to prevent catastrophic losses
- **Parameter**: `max_daily_loss_pct` (default: 2% per day)
- **Method**: Tracks daily P&L and halts trading if threshold breached
- **Reset**: Automatically resets at midnight UTC
- **Use Case**: Stop all trading if portfolio loses 2% in a day

#### Correlation Exposure
- **Purpose**: Prevent over-concentration in correlated assets
- **Parameter**: `high_correlation_threshold` (default: 0.7)
- **Method**: Maintains correlation matrix of open positions
- **Check**: Total correlated exposure ≤ 70% of max_total_exposure
- **Use Case**: Avoid having multiple highly correlated positions (e.g., BTC and ETH)

### 2. ATR-Based Position Sizing

Dynamic position sizing based on Average True Range (ATR) volatility:

```python
position_size = (portfolio_value * risk_pct) / (ATR * multiplier)
```

**Benefits**:
- Automatically adjusts position size based on volatility
- Lower position sizes for volatile assets
- Higher position sizes for stable assets
- Maintains consistent risk per trade

**Parameters**:
- `risk_per_trade_pct`: Risk per trade (default: 1%)
- `atr_multiplier`: Stop-loss distance multiplier (default: 2.0x)
- `atr_period`: Number of periods for ATR calculation (default: 14)

### 3. Portfolio Tracking

Real-time tracking of:
- Current portfolio value
- Daily P&L
- Open positions and exposure
- Price history for correlation analysis

**Methods**:
- `update_position()`: Add or remove position
- `update_portfolio_value()`: Update portfolio value and P&L
- `update_price_history()`: Track prices for correlation analysis
- `get_risk_metrics()`: Get comprehensive risk metrics

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- `numpy>=1.26.0` - For ATR calculations
- `pandas>=2.1.0` - For correlation analysis
- `structlog>=23.2.0` - For logging
- `pydantic>=2.5.0` - For configuration models

### Setup

1. Configure risk parameters in `config/examples/risk_config.yaml`
2. Adjust parameters based on your risk tolerance
3. Initialize RiskManager with exchange interface and configuration

## Usage

### Basic Usage

```python
import asyncio
from decimal import Decimal
from bot.config.models import RiskConfig
from bot.risk.risk_manager import RiskManager
from bot.interfaces.exchange import OrderSide

async def main():
    # Create risk configuration
    risk_config = RiskConfig(
        max_position_size_pct=0.03,  # 3% per trade
        max_total_exposure_pct=0.25,  # 25% total
        max_daily_loss_pct=0.02,     # 2% daily loss limit
        max_slippage_pct=0.005,      # 0.5% max slippage
        min_liquidity_ratio=0.01,    # 1% of daily volume
    )

    # Initialize RiskManager
    risk_manager = RiskManager(
        exchange=exchange,
        config=risk_config,
        initial_portfolio_value=Decimal("100000"),
    )

    # Validate trade
    result = await risk_manager.validate_trade(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("0.5"),
    )

    # Check result
    if result.approved:
        print("✓ Trade approved")
        # Execute trade
    else:
        print("✗ Trade rejected")
        for check in result.get_failed_checks():
            print(f"  - {check.check_name}: {check.message}")
```

### Configure Trading Hours

```python
from datetime import time

# Only trade during market hours (9 AM - 5 PM UTC, weekdays)
risk_manager.set_trading_hours(
    start_time=time(9, 0),   # 9:00 AM UTC
    end_time=time(17, 0),    # 5:00 PM UTC
    trading_days={0, 1, 2, 3, 4},  # Monday-Friday
)
```

### ATR-Based Position Sizing

```python
# Calculate recommended position size
position_size = await risk_manager.calculate_position_size_atr(
    symbol="BTCUSDT",
    risk_per_trade_pct=0.01,  # Risk 1% per trade
    atr_multiplier=2.0,       # 2x ATR stop-loss
)

print(f"Recommended position: {position_size} BTC")
```

### Portfolio Tracking

```python
# Add position
risk_manager.update_position("BTCUSDT", Decimal("5000"), add=True)

# Update portfolio value
risk_manager.update_portfolio_value(Decimal("105000"))  # +5k profit

# Get risk metrics
metrics = risk_manager.get_risk_metrics()
print(f"Total exposure: {metrics['total_exposure']}")
print(f"Daily P&L: {metrics['daily_pnl']}")
```

### Multiple Concurrent Validations

```python
# Validate multiple trades concurrently
trades = [
    {"symbol": "BTCUSDT", "side": OrderSide.BUY, "quantity": Decimal("0.1")},
    {"symbol": "ETHUSDT", "side": OrderSide.BUY, "quantity": Decimal("1.0")},
    {"symbol": "ADAUSDT", "side": OrderSide.SELL, "quantity": Decimal("1000")},
]

# Run all validations in parallel
results = await asyncio.gather(*[
    risk_manager.validate_trade(
        symbol=trade["symbol"],
        side=trade["side"],
        quantity=trade["quantity"],
    )
    for trade in trades
])

# Process results
for result in results:
    if result.approved:
        print(f"✓ {result.symbol}: Approved")
    else:
        print(f"✗ {result.symbol}: Rejected")
```

## Configuration

### Risk Parameters

See `config/examples/risk_config.yaml` for complete configuration options.

Key parameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `max_position_size_pct` | 0.03 (3%) | 0.001 - 1.0 | Maximum position size |
| `max_total_exposure_pct` | 0.25 (25%) | 0.001 - 1.0 | Maximum total exposure |
| `max_daily_loss_pct` | 0.02 (2%) | 0.001 - 0.5 | Daily loss limit |
| `max_slippage_pct` | 0.005 (0.5%) | 0.0 - 0.1 | Maximum slippage |
| `min_liquidity_ratio` | 0.01 (1%) | 0.001 - 1.0 | Position to volume ratio |

### Conservative Configuration

For conservative risk management:

```yaml
risk:
  max_position_size_pct: 0.02    # 2% per trade
  max_total_exposure_pct: 0.15   # 15% total
  max_daily_loss_pct: 0.01       # 1% daily limit
  max_slippage_pct: 0.003        # 0.3% slippage
  min_liquidity_ratio: 0.005     # 0.5% of volume
```

### Aggressive Configuration

For aggressive risk management:

```yaml
risk:
  max_position_size_pct: 0.05    # 5% per trade
  max_total_exposure_pct: 0.40   # 40% total
  max_daily_loss_pct: 0.05       # 5% daily limit
  max_slippage_pct: 0.01         # 1% slippage
  min_liquidity_ratio: 0.02      # 2% of volume
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/unit/risk/test_risk_manager.py -v
```

Test coverage includes:
- ✓ Time restrictions (3 tests)
- ✓ Position size validation (3 tests)
- ✓ Total exposure validation (3 tests)
- ✓ Slippage estimation (3 tests)
- ✓ Liquidity validation (3 tests)
- ✓ Portfolio stop-loss (4 tests)
- ✓ Correlation exposure (3 tests)
- ✓ ATR position sizing (3 tests)
- ✓ Trade validation (3 tests)
- ✓ Portfolio tracking (7 tests)
- ✓ Result dataclasses (4 tests)

**Total: 39 comprehensive unit tests**

## Examples

See `bot/risk/examples/risk_manager_example.py` for a complete working example demonstrating:

1. Basic pre-trade validation
2. ATR-based position sizing
3. Portfolio tracking and metrics
4. Multiple concurrent validations
5. Emergency stop scenario

Run the example:

```bash
python bot/risk/examples/risk_manager_example.py
```

## Architecture

### Class Hierarchy

```
RiskManager
├── validate_trade() - Main entry point
├── _check_time_restrictions()
├── _check_position_size()
├── _check_total_exposure()
├── _check_slippage()
├── _check_liquidity()
├── _check_portfolio_stop_loss()
├── _check_correlation_exposure()
├── calculate_position_size_atr()
├── update_position()
├── update_portfolio_value()
├── update_price_history()
└── get_risk_metrics()
```

### Data Models

```
RiskCheckResult
├── check_name: str
├── status: RiskCheckStatus
├── passed: bool
├── message: str
├── details: Dict
├── value: Optional[float]
└── threshold: Optional[float]

TradeValidationResult
├── approved: bool
├── results: List[RiskCheckResult]
├── correlation_id: str
├── timestamp: datetime
├── symbol: str
├── side: OrderSide
├── quantity: Decimal
└── estimated_value: Decimal
```

## Logging

All risk checks are logged with structured logging using `structlog`:

### Log Levels

- **INFO**: Successful validations, configuration changes, portfolio updates
- **WARNING**: Failed checks that allow trading (warnings), missing data
- **ERROR**: System errors, exceptions in checks

### Correlation Tracking

Every trade validation gets a unique correlation ID for tracing:

```json
{
  "event": "trade_validation_started",
  "symbol": "BTCUSDT",
  "side": "buy",
  "quantity": "0.5",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T14:30:00.123Z"
}
```

### Risk Check Logging

Each risk check logs its result:

```json
{
  "event": "risk_check_completed",
  "check_name": "position_size",
  "status": "passed",
  "value": 0.025,
  "threshold": 0.03,
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Performance

### Optimizations

1. **Concurrent Checks**: All risk checks run in parallel using `asyncio.gather()`
2. **Caching**: Order book and volume data cached for 60 seconds
3. **Efficient Data Structures**: Price history limited to 1000 most recent points
4. **Correlation Updates**: Correlation matrix updates hourly (configurable)

### Benchmarks

Typical performance on modern hardware:

- Single trade validation: ~50-100ms
- ATR position sizing: ~10-20ms
- Portfolio metrics: <1ms
- Correlation matrix update: ~100-200ms (for 10 symbols)

## Best Practices

### 1. Start Conservative

Begin with conservative risk parameters and gradually increase as you gain confidence:

```python
risk_config = RiskConfig(
    max_position_size_pct=0.01,   # 1% per trade
    max_total_exposure_pct=0.10,  # 10% total
    max_daily_loss_pct=0.01,      # 1% daily limit
)
```

### 2. Monitor Risk Metrics

Regularly check risk metrics:

```python
metrics = risk_manager.get_risk_metrics()
logger.info("risk_metrics", **metrics)
```

### 3. Use ATR Position Sizing

Let volatility determine position size:

```python
size = await risk_manager.calculate_position_size_atr(
    symbol=symbol,
    risk_per_trade_pct=0.01,
)
```

### 4. Set Trading Hours

Avoid low-liquidity periods:

```python
risk_manager.set_trading_hours(
    start_time=time(8, 0),   # 8 AM UTC
    end_time=time(22, 0),    # 10 PM UTC
    trading_days={0, 1, 2, 3, 4},  # Weekdays only
)
```

### 5. Handle Validation Results

Always check validation results before executing trades:

```python
result = await risk_manager.validate_trade(...)

if result.approved:
    # Execute trade
    await exchange.create_order(...)

    # Update risk manager
    risk_manager.update_position(symbol, position_value, add=True)
else:
    # Log rejection
    logger.warning("trade_rejected",
                   failed_checks=[c.check_name for c in result.get_failed_checks()])
```

### 6. Track All Positions

Keep RiskManager synchronized with actual positions:

```python
# When opening a position
risk_manager.update_position(symbol, value, add=True)

# When closing a position
risk_manager.update_position(symbol, value, add=False)

# Update portfolio value regularly
risk_manager.update_portfolio_value(current_value)
```

## Troubleshooting

### Common Issues

#### 1. All Trades Rejected

**Problem**: All trades are being rejected

**Solutions**:
- Check if daily loss limit has been breached
- Verify trading hours configuration
- Check if portfolio exposure is at maximum
- Review risk parameter configuration

#### 2. Slippage Check Always Fails

**Problem**: Slippage check fails for all trades

**Solutions**:
- Increase `max_slippage_pct` parameter
- Use smaller order sizes
- Trade more liquid pairs
- Check order book data quality

#### 3. Correlation Data Missing

**Problem**: Correlation checks are skipped

**Solutions**:
- Ensure price history is being updated
- Call `update_price_history()` regularly
- Wait for minimum data points (20+)
- Check correlation matrix update interval

#### 4. ATR Position Sizing Returns Default

**Problem**: ATR sizing falls back to default

**Solutions**:
- Ensure sufficient price history (14+ points)
- Verify price history is being updated correctly
- Check for ATR calculation errors in logs

## Future Enhancements

Potential future additions:

1. **VaR (Value at Risk) Calculation**: Historical and parametric VaR
2. **Monte Carlo Simulations**: Portfolio risk simulation
3. **Greeks for Options**: Delta, gamma, vega exposure
4. **Multi-Timeframe Analysis**: Risk across different timeframes
5. **Machine Learning Risk Scoring**: ML-based risk assessment
6. **Real-Time Market Impact**: Live market impact estimation
7. **Cross-Exchange Risk**: Aggregate risk across multiple exchanges
8. **Regulatory Compliance**: Built-in compliance checks

## Support

For issues, questions, or contributions:

1. Check the examples: `bot/risk/examples/risk_manager_example.py`
2. Review test cases: `tests/unit/risk/test_risk_manager.py`
3. Check logs for detailed error messages
4. Review configuration: `config/examples/risk_config.yaml`

## License

See main project LICENSE file.
