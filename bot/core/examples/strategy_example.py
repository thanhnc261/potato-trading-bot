"""
Example usage of trading strategies.

This script demonstrates:
- Creating and configuring strategies
- Generating trading signals
- Managing positions
- Backtesting workflows
"""

import time

import numpy as np
import pandas as pd
from structlog import get_logger

from bot.core.strategy import (
    MovingAverageCrossoverStrategy,
    RSIStrategy,
    Signal,
)

logger = get_logger(__name__)


def generate_sample_data(n_periods: int = 200, trend: str = "neutral") -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.

    Args:
        n_periods: Number of periods to generate
        trend: Trend type ("up", "down", "neutral", "volatile")

    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(42)

    # Base parameters
    base_price = 50000.0
    timestamps = [int(time.time() * 1000) + i * 60000 for i in range(n_periods)]

    # Generate prices based on trend
    if trend == "up":
        trend_component = np.linspace(0, 0.2, n_periods)
        volatility = 0.01
    elif trend == "down":
        trend_component = np.linspace(0, -0.15, n_periods)
        volatility = 0.01
    elif trend == "volatile":
        trend_component = np.zeros(n_periods)
        volatility = 0.03
    else:  # neutral
        trend_component = np.zeros(n_periods)
        volatility = 0.015

    # Generate price movements
    returns = np.random.normal(0, volatility, n_periods) + trend_component / n_periods
    prices = base_price * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * 0.999,
            "high": prices * 1.002,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.random.uniform(100, 1000, n_periods),
        }
    )


def demo_rsi_strategy():
    """Demonstrate RSI strategy usage"""
    logger.info("=== RSI Strategy Demo ===")

    # Create RSI strategy with custom config
    config = {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "position_size_pct": 0.1,
    }
    strategy = RSIStrategy(config=config)

    # Generate sample data with downtrend (to trigger oversold)
    data = generate_sample_data(n_periods=100, trend="down")

    # Generate signal
    signal = strategy.generate_signal(data)

    logger.info(
        "rsi_signal_generated",
        signal=signal.signal.value,
        price=signal.price,
        confidence=signal.confidence,
        rsi=signal.metadata.get("rsi"),
        reason=signal.reason,
    )

    # Enter position if BUY signal
    if signal.signal == Signal.BUY:
        capital = 10000.0
        size = strategy.get_position_size(signal, capital)
        position = strategy.enter_position(signal, size)

        logger.info(
            "position_entered",
            side=position.side.value,
            entry_price=position.entry_price,
            size=position.size,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
        )

        # Simulate price movement and check exit conditions
        take_profit_price = position.take_profit if position.take_profit else position.entry_price * 1.04
        test_prices = [
            position.entry_price * 0.99,  # Small loss
            position.entry_price * 1.01,  # Small profit
            take_profit_price * 1.001,  # Take profit hit
        ]

        for price in test_prices:
            should_exit, reason = strategy.should_exit(price, position)
            logger.info(
                "exit_check",
                current_price=price,
                should_exit=should_exit,
                reason=reason,
            )

            if should_exit:
                exit_details = strategy.exit_position(
                    exit_price=price,
                    exit_timestamp=signal.timestamp + 60000,
                    reason=reason,
                )
                logger.info("position_exited", **exit_details)
                break


def demo_ma_crossover_strategy():
    """Demonstrate MA Crossover strategy usage"""
    logger.info("=== MA Crossover Strategy Demo ===")

    # Create MA Crossover strategy
    config = {
        "fast_period": 20,
        "slow_period": 50,
        "ma_type": "sma",
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
    }
    strategy = MovingAverageCrossoverStrategy(config=config)

    # Generate sample data with uptrend (to trigger golden cross)
    data = generate_sample_data(n_periods=150, trend="up")

    # Generate signals progressively to detect crossover
    logger.info("scanning_for_crossover")

    for i in range(60, len(data), 10):
        window_data = data.iloc[: i + 1]
        signal = strategy.generate_signal(window_data)

        if signal.signal in [Signal.BUY, Signal.SELL]:
            logger.info(
                "crossover_detected",
                signal=signal.signal.value,
                price=signal.price,
                fast_ma=signal.metadata.get("fast_ma"),
                slow_ma=signal.metadata.get("slow_ma"),
                reason=signal.reason,
            )
            break


def demo_backtesting():
    """Demonstrate backtesting workflow"""
    logger.info("=== Backtesting Demo ===")

    # Create strategy
    strategy = RSIStrategy()

    # Generate volatile data for backtesting
    data = generate_sample_data(n_periods=200, trend="volatile")

    # Track performance metrics
    trades = []
    total_pnl = 0.0
    capital = 10000.0

    # Simulate rolling window backtesting
    window_size = 50

    for i in range(window_size, len(data)):
        window_data = data.iloc[i - window_size : i + 1]
        signal = strategy.generate_signal(window_data)
        current_price = float(window_data["close"].iloc[-1])

        # Entry logic
        if signal.signal == Signal.BUY and strategy.current_position is None:
            size = strategy.get_position_size(signal, capital)
            strategy.enter_position(signal, size)

            logger.info(
                "backtest_entry",
                period=i,
                price=signal.price,
                rsi=signal.metadata.get("rsi"),
            )

        # Exit logic
        elif strategy.current_position is not None:
            should_exit, reason = strategy.should_exit(current_price, strategy.current_position)

            # Check for opposite signal
            if signal.signal == Signal.SELL:
                should_exit = True
                reason = "Opposite signal"

            if should_exit:
                exit_details = strategy.exit_position(
                    exit_price=current_price,
                    exit_timestamp=signal.timestamp,
                    reason=reason,
                )

                trades.append(exit_details)
                total_pnl += exit_details["pnl"]

                logger.info(
                    "backtest_exit",
                    period=i,
                    pnl=exit_details["pnl"],
                    pnl_pct=exit_details["pnl_pct"],
                    reason=reason,
                )

    # Calculate performance metrics
    if trades:
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(winning_trades) / len(trades) * 100

        avg_win = (
            sum(t["pnl"] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        )
        avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        logger.info(
            "backtest_results",
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=f"{win_rate:.2f}%",
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            final_capital=capital + total_pnl,
        )
    else:
        logger.info("backtest_results", message="No trades executed")


def main():
    """Run all strategy examples"""
    # Setup basic logging
    import structlog

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )

    # Run demos
    demo_rsi_strategy()
    print("\n")
    demo_ma_crossover_strategy()
    print("\n")
    demo_backtesting()


if __name__ == "__main__":
    main()
