# Logging Infrastructure

Comprehensive structured logging system with correlation ID tracking, multiple outputs, and automatic rotation.

## Features

- ✅ **Structured JSON logging** using `structlog`
- ✅ **Correlation IDs** for request tracing across the application
- ✅ **Multiple log outputs**: console, system logs, trade logs, error logs
- ✅ **Daily log rotation** with 30-day retention
- ✅ **Performance-optimized** async I/O
- ✅ **Type-safe configuration** with Pydantic
- ✅ **Environment-specific** settings (dev, paper, prod)

## Quick Start

### Basic Usage

```python
from pathlib import Path
from bot.core.logging_config import setup_logging, get_logger

# Initialize logging
setup_logging(
    log_dir=Path("logs"),
    log_level="INFO",
    console_level="INFO"
)

# Get a logger
log = get_logger(__name__)

# Log messages
log.info("bot_started", symbol="BTCUSDT", balance=10000.0)
log.warning("high_volatility", symbol="ETHUSDT", volatility=0.15)
log.error("api_error", exchange="binance", error_code=429)
```

### With Correlation IDs

```python
from bot.core.logging_config import CorrelationContext, get_logger

log = get_logger(__name__)

# All logs within this context will have the same correlation_id
with CorrelationContext() as correlation_id:
    log.info("processing_order", order_id="12345")
    # ... do work ...
    log.info("order_completed", order_id="12345", status="FILLED")

# Correlation ID is automatically cleared after context exits
```

### Trade Logging

```python
from bot.core.logging_config import log_trade, log_order

# Log a trade event (goes to trades.log)
log_trade(
    action="ENTRY",
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.01,
    price=50000.0,
    order_id="order-123",
    strategy="ema_crossover"
)

# Log an order event
log_order(
    order_id="order-123",
    status="FILLED",
    symbol="BTCUSDT",
    side="BUY",
    quantity=0.01,
    filled_price=50050.0,
    slippage_pct=0.001
)
```

## Configuration

### Using YAML Configuration

```yaml
# config/profiles/dev.yaml
logging:
  log_dir: "logs"
  log_level: "DEBUG"
  console_level: "DEBUG"
  enable_json: false      # Human-readable for dev
  enable_colors: true
  rotation_when: "midnight"
  rotation_interval: 1
  backup_count: 7         # Keep 7 days
```

### Loading Configuration

```python
from pathlib import Path
import yaml
from bot.config.models import BotConfig
from bot.core.logging_config import setup_logging

# Load config from YAML
with open("config/profiles/dev.yaml") as f:
    config_dict = yaml.safe_load(f)

config = BotConfig(**config_dict["bot"])

# Setup logging from config
setup_logging(
    log_dir=config.logging.log_dir,
    log_level=config.logging.log_level.value,
    console_level=config.logging.console_level.value,
    enable_json=config.logging.enable_json,
    enable_colors=config.logging.enable_colors
)
```

## Log Files

The logging system creates three separate log files:

### 1. system.log
General application logs (INFO, WARNING, ERROR)

```json
{
  "timestamp": "2025-10-22T00:00:00.000000Z",
  "level": "info",
  "event": "bot_started",
  "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "module": "bot.main",
  "function": "start_bot",
  "line": 42,
  "environment": "paper",
  "version": "0.1.0"
}
```

### 2. trades.log
Trade-specific events (separate from system logs)

```json
{
  "timestamp": "2025-10-22T00:01:00.000000Z",
  "level": "info",
  "event": "trade_event",
  "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "action": "ENTRY",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "quantity": 0.01,
  "price": 50000.0,
  "order_id": "order-123",
  "strategy": "ema_crossover"
}
```

### 3. errors.log
ERROR and CRITICAL level logs only

```json
{
  "timestamp": "2025-10-22T00:02:00.000000Z",
  "level": "error",
  "event": "exchange_api_error",
  "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "module": "bot.execution.adapters.binance",
  "function": "place_order",
  "line": 156,
  "error_code": 429,
  "error_message": "Rate limit exceeded",
  "exception": "..."
}
```

## Advanced Usage

### Manual Correlation ID Management

```python
from bot.core.logging_config import (
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id
)

# Set correlation ID manually
set_correlation_id("my-custom-id-123")

# Get current correlation ID
current_id = get_correlation_id()

# Clear correlation ID
clear_correlation_id()
```

### Nested Correlation Contexts

```python
from bot.core.logging_config import CorrelationContext, get_logger

log = get_logger(__name__)

with CorrelationContext("outer-request"):
    log.info("processing_batch")

    for item in items:
        # Each item gets its own correlation ID
        with CorrelationContext(f"item-{item.id}"):
            log.info("processing_item", item_id=item.id)
            process_item(item)

    # Back to outer correlation ID
    log.info("batch_completed")
```

### Logging with Structured Data

```python
log = get_logger(__name__)

# Log with structured context
log.info(
    "market_analysis_completed",
    symbol="BTCUSDT",
    timeframe="1h",
    indicators={
        "rsi": 65.4,
        "macd": 0.0023,
        "bollinger_upper": 51000,
        "bollinger_lower": 49000
    },
    signal="BUY",
    confidence=0.75
)
```

### Exception Logging

```python
log = get_logger(__name__)

try:
    risky_operation()
except Exception as e:
    log.exception(
        "operation_failed",
        operation="risky_operation",
        error_type=type(e).__name__
    )
    # Exception traceback is automatically included
```

## Log Rotation

Logs are automatically rotated daily at midnight (configurable):

```
logs/
├── system.log              # Current day
├── system.log.2025-10-21   # Previous day
├── system.log.2025-10-20
├── ...                     # Up to backup_count days
├── trades.log
├── trades.log.2025-10-21
└── errors.log
```

## Performance Considerations

1. **Async I/O**: Log writes are non-blocking
2. **Buffering**: Log handlers use buffered I/O
3. **Lazy Formatting**: String formatting only happens if log level is enabled
4. **Structured Data**: Native dict logging avoids string formatting overhead

```python
# Good - structured logging (efficient)
log.info("processing_order", order_id=order_id, quantity=qty)

# Avoid - string formatting (slower)
log.info(f"Processing order {order_id} with quantity {qty}")
```

## Environment-Specific Settings

### Development
```yaml
logging:
  log_level: "DEBUG"
  console_level: "DEBUG"
  enable_json: false      # Human-readable
  enable_colors: true
  backup_count: 7         # 7 days
```

### Paper Trading
```yaml
logging:
  log_level: "INFO"
  console_level: "INFO"
  enable_json: true       # Structured for analysis
  enable_colors: true
  backup_count: 30        # 30 days
```

### Production
```yaml
logging:
  log_level: "INFO"
  console_level: "WARNING"  # Less verbose
  enable_json: true         # Structured logs
  enable_colors: false      # No ANSI codes in production
  backup_count: 90          # 90 days for compliance
```

## Testing

Run the logging tests:

```bash
cd potato-trading-bot
pytest tests/unit/core/test_logging_config.py -v
```

## Integration with Other Modules

### In Exchange Adapter

```python
from bot.core.logging_config import get_logger, CorrelationContext

class BinanceAdapter:
    def __init__(self):
        self.log = get_logger(__name__)

    async def place_order(self, order):
        with CorrelationContext() as correlation_id:
            self.log.info(
                "placing_order",
                order_id=order.id,
                symbol=order.symbol,
                side=order.side
            )

            try:
                result = await self._execute_order(order)
                self.log.info("order_placed", order_id=order.id, status="NEW")
                return result
            except Exception as e:
                self.log.exception("order_failed", order_id=order.id)
                raise
```

### In Strategy

```python
from bot.core.logging_config import get_logger, log_trade

class Strategy:
    def __init__(self):
        self.log = get_logger(__name__)

    def on_signal(self, signal):
        self.log.info(
            "signal_generated",
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.confidence
        )

        if signal.should_execute:
            log_trade(
                action="ENTRY",
                symbol=signal.symbol,
                side=signal.side,
                quantity=signal.quantity,
                price=signal.price,
                strategy=self.name
            )
```

## Troubleshooting

### Logs not appearing

Check that logging is initialized before any log calls:
```python
from bot.core.logging_config import setup_logging

# Must be called before get_logger()
setup_logging(Path("logs"))
```

### Log directory permissions

Ensure the log directory is writable:
```bash
mkdir -p logs
chmod 755 logs
```

### Testing log output

For debugging, disable JSON and enable DEBUG level:
```python
setup_logging(
    log_dir=Path("logs"),
    log_level="DEBUG",
    enable_json=False,
    enable_colors=True
)
```

## Best Practices

1. **Always use correlation IDs** for request/trade lifecycle tracking
2. **Use structured logging** (key=value) instead of string formatting
3. **Log at appropriate levels**:
   - DEBUG: Detailed diagnostic information
   - INFO: General information (starts, stops, state changes)
   - WARNING: Something unexpected but handled
   - ERROR: Error conditions that need attention
   - CRITICAL: Critical errors causing shutdown

4. **Include context**: symbol, order_id, strategy name, etc.
5. **Don't log sensitive data**: API keys, secrets, passwords
6. **Use trade logger** for trade events to keep them separate
7. **Test logging** in unit tests to ensure it works correctly

## Example: Complete Trade Lifecycle

```python
from bot.core.logging_config import (
    setup_logging,
    get_logger,
    log_order,
    log_trade,
    CorrelationContext
)

# Initialize
setup_logging(Path("logs"))
log = get_logger(__name__)

# Start trade with correlation ID
with CorrelationContext() as correlation_id:
    log.info("analyzing_market", symbol="BTCUSDT")

    # Generate signal
    log.info("signal_generated", signal="BUY", confidence=0.8)

    # Place order
    log_order(
        order_id="order-123",
        status="NEW",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.01
    )

    # Order filled
    log_order(
        order_id="order-123",
        status="FILLED",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.01,
        filled_price=50050.0
    )

    # Log trade
    log_trade(
        action="ENTRY",
        symbol="BTCUSDT",
        side="BUY",
        quantity=0.01,
        price=50050.0,
        order_id="order-123"
    )

    log.info("trade_completed", order_id="order-123")
```

All these log entries will share the same `correlation_id`, making it easy to trace the entire trade lifecycle.
