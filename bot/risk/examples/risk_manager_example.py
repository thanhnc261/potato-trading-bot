"""
Example usage of the RiskManager for pre-trade validation.

This example demonstrates:
1. Setting up RiskManager with configuration
2. Performing pre-trade validation checks
3. Dynamic position sizing with ATR
4. Portfolio tracking and risk metrics
5. Time-based trading restrictions
"""

import asyncio
from datetime import time as datetime_time
from decimal import Decimal
from typing import Any

from bot.config.models import RiskConfig
from bot.execution.adapters.binance import BinanceAdapter
from bot.interfaces.exchange import OrderSide
from bot.risk.risk_manager import RiskManager


async def main():
    """Demonstrate RiskManager usage."""

    # 1. Initialize exchange adapter
    print("=" * 60)
    print("Risk Manager Example")
    print("=" * 60)

    exchange = BinanceAdapter(
        api_key="your_api_key",
        api_secret="your_api_secret",
        testnet=True,
    )

    await exchange.connect()
    print("✓ Connected to exchange")

    # 2. Create risk configuration
    risk_config = RiskConfig(
        max_position_size_pct=0.03,  # 3% per trade
        max_total_exposure_pct=0.25,  # 25% total
        max_daily_loss_pct=0.02,  # 2% daily loss limit
        max_slippage_pct=0.005,  # 0.5% max slippage
        min_liquidity_ratio=0.01,  # Position < 1% of daily volume
        var_confidence=0.95,
        enable_emergency_stop=True,
    )
    print("✓ Risk configuration created")

    # 3. Initialize RiskManager
    initial_portfolio_value = Decimal("100000")  # $100,000 starting capital
    risk_manager = RiskManager(
        exchange=exchange,
        config=risk_config,
        initial_portfolio_value=initial_portfolio_value,
    )
    print(f"✓ RiskManager initialized with ${initial_portfolio_value:,}")

    # 4. Configure trading hours (optional)
    # Example: Only trade during market hours (9 AM - 5 PM UTC, weekdays)
    risk_manager.set_trading_hours(
        start_time=datetime_time(9, 0),  # 9:00 AM UTC
        end_time=datetime_time(17, 0),  # 5:00 PM UTC
        trading_days={0, 1, 2, 3, 4},  # Monday-Friday
    )
    print("✓ Trading hours configured (9 AM - 5 PM UTC, weekdays)")

    print("\n" + "=" * 60)
    print("Example 1: Pre-Trade Validation")
    print("=" * 60)

    # 5. Perform pre-trade validation
    symbol = "BTCUSDT"
    side = OrderSide.BUY
    quantity = Decimal("0.5")  # 0.5 BTC

    print(f"\nValidating trade: {side.value} {quantity} {symbol}")

    validation_result = await risk_manager.validate_trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=None,  # Use current market price
    )

    # 6. Check validation result
    print("\nValidation Result:")
    print(f"  Approved: {validation_result.approved}")
    print(f"  Correlation ID: {validation_result.correlation_id}")
    print(f"  Estimated Value: ${validation_result.estimated_value:,.2f}")
    print(f"  Timestamp: {validation_result.timestamp}")

    # 7. Display individual check results
    print("\nIndividual Checks:")
    for check in validation_result.results:
        status_symbol = "✓" if check.passed else "✗"
        print(f"  {status_symbol} {check.check_name}: {check.message}")
        if check.value is not None and check.threshold is not None:
            print(f"      Value: {check.value:.2%} | Threshold: {check.threshold:.2%}")

    # 8. Handle failed checks
    if not validation_result.approved:
        print("\n⚠ Trade rejected! Failed checks:")
        for failed_check in validation_result.get_failed_checks():
            print(f"  - {failed_check.check_name}: {failed_check.message}")
    else:
        print("\n✓ Trade approved!")

    # 9. Display warnings (if any)
    warnings = validation_result.get_warnings()
    if warnings:
        print(f"\n⚠ Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"  - {warning.check_name}: {warning.message}")

    print("\n" + "=" * 60)
    print("Example 2: ATR-Based Position Sizing")
    print("=" * 60)

    # 10. Simulate price history for ATR calculation
    print(f"\nSimulating price history for {symbol}...")
    for i in range(20):
        price = 50000.0 + (i * 100)  # Simulated prices
        risk_manager.update_price_history(symbol, price)
    print("✓ Added 20 price points")

    # 11. Calculate dynamic position size using ATR
    recommended_size = await risk_manager.calculate_position_size_atr(
        symbol=symbol,
        risk_per_trade_pct=0.01,  # Risk 1% per trade
        atr_multiplier=2.0,  # 2x ATR for stop-loss
    )

    print("\nATR-Based Position Sizing:")
    print(f"  Recommended quantity: {recommended_size:.4f} {symbol.split('USDT')[0]}")
    print("  Risk per trade: 1% of portfolio")
    print("  ATR multiplier: 2.0x")

    print("\n" + "=" * 60)
    print("Example 3: Portfolio Tracking")
    print("=" * 60)

    # 12. Simulate opening positions
    print("\nSimulating portfolio activity...")

    # Open first position
    risk_manager.update_position("BTCUSDT", Decimal("5000"), add=True)
    print("✓ Opened position: BTCUSDT ($5,000)")

    # Open second position
    risk_manager.update_position("ETHUSDT", Decimal("3000"), add=True)
    print("✓ Opened position: ETHUSDT ($3,000)")

    # Update portfolio value (simulated P&L)
    risk_manager.update_portfolio_value(Decimal("102000"))  # +$2,000 profit
    print("✓ Portfolio value updated: $102,000 (+$2,000)")

    # 13. Get risk metrics
    metrics = risk_manager.get_risk_metrics()

    print("\nCurrent Risk Metrics:")
    print(f"  Portfolio Value: {metrics['portfolio_value']}")
    print(f"  Daily P&L: {metrics['daily_pnl']}")
    print(f"  Open Positions: {metrics['open_positions']}")
    print(f"  Total Exposure: {metrics['total_exposure']}")
    print(f"  Exposure %: {metrics['exposure_pct']}")
    print(f"  Max Exposure Limit: {metrics['max_exposure_pct']}")

    print("\nPosition Details:")
    for symbol, value in metrics["positions"].items():
        print(f"  - {symbol}: ${value}")

    print("\n" + "=" * 60)
    print("Example 4: Multiple Validations")
    print("=" * 60)

    # 14. Validate multiple trades
    trades: list[dict[str, Any]] = [
        {"symbol": "BTCUSDT", "side": OrderSide.BUY, "quantity": Decimal("0.1")},
        {"symbol": "ETHUSDT", "side": OrderSide.BUY, "quantity": Decimal("1.0")},
        {"symbol": "ADAUSDT", "side": OrderSide.SELL, "quantity": Decimal("1000")},
    ]

    print("\nValidating multiple trades concurrently...")

    # Validate all trades concurrently
    validation_tasks = [
        risk_manager.validate_trade(
            symbol=str(trade["symbol"]),
            side=OrderSide(trade["side"]),
            quantity=Decimal(trade["quantity"]),
        )
        for trade in trades
    ]

    results = await asyncio.gather(*validation_tasks)

    # Display results
    print("\nValidation Results:")
    for i, result in enumerate(results):
        trade = trades[i]
        side = OrderSide(trade["side"])
        status = "✓ APPROVED" if result.approved else "✗ REJECTED"
        print(f"  {i+1}. {trade['symbol']} {side.value} {trade['quantity']}: {status}")

        if not result.approved:
            failed = result.get_failed_checks()
            print(f"      Failed checks: {', '.join([c.check_name for c in failed])}")

    # 15. Summary statistics
    approved_count = sum(1 for r in results if r.approved)
    rejected_count = len(results) - approved_count

    print("\nSummary:")
    print(f"  Total trades validated: {len(results)}")
    print(f"  Approved: {approved_count}")
    print(f"  Rejected: {rejected_count}")
    print(f"  Approval rate: {(approved_count / len(results) * 100):.1f}%")

    print("\n" + "=" * 60)
    print("Example 5: Emergency Stop (Daily Loss Limit)")
    print("=" * 60)

    # 16. Simulate daily loss scenario
    print("\nSimulating daily loss scenario...")

    # Simulate a -3% loss (exceeds the 2% daily loss limit)
    loss_amount = initial_portfolio_value * Decimal("0.03")
    new_portfolio_value = initial_portfolio_value - loss_amount
    risk_manager.update_portfolio_value(new_portfolio_value)

    print(f"Portfolio dropped to ${new_portfolio_value:,.2f} (-${loss_amount:,.2f})")

    # Try to validate a new trade
    validation = await risk_manager.validate_trade(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("0.1"),
    )

    print("\nAttempting new trade after daily loss limit breach:")
    print(f"  Approved: {validation.approved}")

    # Find the stop-loss check
    stop_loss_check = next(
        (r for r in validation.results if r.check_name == "portfolio_stop_loss"),
        None,
    )

    if stop_loss_check and not stop_loss_check.passed:
        print(f"  ⚠ {stop_loss_check.message}")
        print(f"  Daily loss: {stop_loss_check.details['daily_loss_pct']}")
        print(f"  Max allowed: {stop_loss_check.details['max_daily_loss_pct']}")
        print("\n  🛑 Emergency stop activated! Trading halted.")

    # Cleanup
    print("\n" + "=" * 60)
    await exchange.disconnect()
    print("✓ Disconnected from exchange")
    print("\nExample completed!")


if __name__ == "__main__":
    # Note: This example requires valid exchange credentials
    # For testing, use paper trading or testnet mode
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
    except Exception as e:
        print(f"\n\nError running example: {e}")
        print("Note: Make sure you have valid exchange credentials configured")
