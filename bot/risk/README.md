# Emergency Stop System

Comprehensive kill-switch system for catastrophic scenarios with automated triggers and manual override capabilities.

## Features

### Automated Triggers

1. **Flash Crash Detection** - Detects rapid price movements >10% in <5 minutes
2. **Exchange API Failure** - Monitors API connectivity and triggers if unreachable >30 seconds
3. **Portfolio Drawdown** - Tracks portfolio value and triggers if drawdown exceeds 10%
4. **Data Quality Monitoring** - Detects stale data (>60 seconds old) and invalid values (NaN, negative prices)
5. **Consecutive API Failures** - Triggers after 5 consecutive API failures

### Emergency Actions

When triggered, the system automatically:
1. Cancels all open orders across all symbols
2. Closes all positions at market price
3. Halts the trading engine
4. Sends alerts via configured channels (Telegram/Email)

### Alert System

- **Telegram Integration** - Instant alerts to configured chat IDs with formatted messages
- **Email Notifications** - HTML and plain text emails to configured recipients
- **Detailed Context** - All alerts include trigger type, severity, timestamp, and correlation ID

### Manual Override

- **Manual Trigger** - Operators can manually trigger emergency stop with a reason
- **Manual Resume** - Operators can resume trading after verifying conditions
- **Auto-Recovery** - Optional automatic recovery after configurable delay

## Usage Example

```python
from decimal import Decimal
from bot.risk.emergency_stop import EmergencyStopManager, EmergencyConfig
from bot.risk.notifications import NotificationManager, TelegramConfig

# Configure emergency stop
emergency_config = EmergencyConfig(
    flash_crash_threshold_pct=0.10,      # 10% price move
    api_failure_threshold_seconds=30,     # 30 seconds timeout
    max_drawdown_pct=0.10,                # 10% max drawdown
)

# Configure Telegram alerts
telegram_config = TelegramConfig(
    bot_token="YOUR_BOT_TOKEN",
    chat_ids=["YOUR_CHAT_ID"],
)

notification_manager = NotificationManager(telegram_config=telegram_config)

# Create emergency manager
emergency_manager = EmergencyStopManager(
    exchange=exchange,
    initial_portfolio_value=Decimal("10000"),
    config=emergency_config,
    alert_callback=notification_manager.send_alert,
)

# Start monitoring
await emergency_manager.start_monitoring()

# Update price data
await emergency_manager.update_price("BTCUSDT", Decimal("50000"))

# Check if halted before trading
if emergency_manager.is_halted():
    print("Trading halted!")
```

See full documentation in docstrings and tests.
