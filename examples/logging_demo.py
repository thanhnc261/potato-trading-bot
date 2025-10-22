"""
Demo script showing logging functionality.

Run this to verify logging is working correctly.
"""

from pathlib import Path
import sys
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.core.logging_config import (
    setup_logging,
    get_logger,
    log_trade,
    log_order,
    CorrelationContext,
)


def main():
    """Run logging demo."""
    # Create temporary log directory for demo
    log_dir = Path(tempfile.mkdtemp(prefix="bot_logs_"))
    print(f"📁 Logging to: {log_dir}\n")

    # Initialize logging
    setup_logging(
        log_dir=log_dir,
        log_level="DEBUG",
        console_level="INFO",
        enable_json=False,  # Human-readable for demo
        enable_colors=True,
    )

    log = get_logger(__name__)

    # Demo 1: Basic logging
    print("=" * 60)
    print("Demo 1: Basic Logging")
    print("=" * 60)

    log.debug("debug_message", detail="This is debug level")
    log.info("bot_started", version="0.1.0", environment="demo")
    log.warning("high_volatility_detected", symbol="BTCUSDT", volatility=0.15)
    log.error("api_rate_limit", exchange="binance", retry_after=60)

    print()

    # Demo 2: Correlation IDs
    print("=" * 60)
    print("Demo 2: Correlation ID Tracking")
    print("=" * 60)

    with CorrelationContext() as correlation_id:
        log.info("trade_lifecycle_start", trade_id="trade-001")
        log.info("analyzing_market", symbol="ETHUSDT")
        log.info("signal_generated", signal="BUY", confidence=0.8)
        log.info("trade_lifecycle_end", trade_id="trade-001")
        print(f"\n✓ All logs above share correlation_id: {correlation_id}\n")

    # Demo 3: Trade logging
    print("=" * 60)
    print("Demo 3: Trade Logging (separate file)")
    print("=" * 60)

    with CorrelationContext("trade-correlation-456"):
        log_order(
            order_id="order-123",
            status="NEW",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
        )

        log_order(
            order_id="order-123",
            status="FILLED",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
            filled_price=50050.0,
            slippage_pct=0.001,
        )

        log_trade(
            action="ENTRY",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.01,
            price=50050.0,
            order_id="order-123",
            strategy="demo_strategy",
        )

    print()

    # Demo 4: Structured data
    print("=" * 60)
    print("Demo 4: Structured Data")
    print("=" * 60)

    log.info(
        "market_analysis_completed",
        symbol="BTCUSDT",
        timeframe="1h",
        indicators={
            "rsi": 65.4,
            "macd": 0.0023,
            "bollinger_upper": 51000,
            "bollinger_lower": 49000,
        },
        signal="BUY",
        confidence=0.75,
    )

    print()

    # Demo 5: Exception logging
    print("=" * 60)
    print("Demo 5: Exception Logging")
    print("=" * 60)

    try:
        # Simulate an error
        result = 1 / 0
    except Exception as e:
        log.exception(
            "calculation_error",
            operation="division",
            error_type=type(e).__name__,
        )

    print()

    # Show log files
    print("=" * 60)
    print("Log Files Created")
    print("=" * 60)

    for log_file in ["system.log", "trades.log", "errors.log"]:
        file_path = log_dir / log_file
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✓ {log_file} ({size} bytes)")

            # Show first few lines
            print(f"\n  Preview of {log_file}:")
            with open(file_path) as f:
                lines = [line.strip() for line in f.readlines()[:3]]
                for line in lines:
                    print(f"  {line[:100]}...")  # Truncate long lines

            print()

    print()
    print("=" * 60)
    print("✅ Logging demo completed successfully!")
    print("=" * 60)
    print(f"\nFull logs available at: {log_dir}")


if __name__ == "__main__":
    main()
