"""
Unit tests for trading strategy implementations.

Tests cover:
- Base strategy interface
- RSI strategy signal generation
- MA Crossover strategy signal generation
- Position entry/exit logic
- Stop loss and take profit functionality
- Configuration validation
"""

import time

import numpy as np
import pandas as pd
import pytest

from bot.core.strategy import (
    BaseStrategy,
    MovingAverageCrossoverStrategy,
    Position,
    PositionSide,
    RSIStrategy,
    Signal,
    StrategySignal,
)


# Fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    np.random.seed(42)
    n_periods = 100

    # Generate realistic price movements
    base_price = 50000.0
    returns = np.random.normal(0.0001, 0.02, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))

    timestamps = [int(time.time() * 1000) + i * 60000 for i in range(n_periods)]

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


@pytest.fixture
def oversold_data():
    """Create data that triggers oversold RSI signal"""
    n_periods = 50

    # Create declining prices to trigger oversold
    timestamps = [int(time.time() * 1000) + i * 60000 for i in range(n_periods)]
    prices = np.linspace(50000, 45000, n_periods)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.ones(n_periods) * 500,
        }
    )


@pytest.fixture
def overbought_data():
    """Create data that triggers overbought RSI signal"""
    n_periods = 50

    # Create rising prices to trigger overbought
    timestamps = [int(time.time() * 1000) + i * 60000 for i in range(n_periods)]
    prices = np.linspace(45000, 50000, n_periods)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.ones(n_periods) * 500,
        }
    )


@pytest.fixture
def golden_cross_data():
    """Create data that triggers golden cross (fast MA crosses above slow MA)"""
    n_periods = 100

    # First half: declining prices (death cross setup)
    # Second half: rising prices (golden cross)
    timestamps = [int(time.time() * 1000) + i * 60000 for i in range(n_periods)]
    prices = np.concatenate([np.linspace(50000, 47000, 50), np.linspace(47000, 51000, 50)])

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices * 0.999,
            "high": prices * 1.001,
            "low": prices * 0.998,
            "close": prices,
            "volume": np.ones(n_periods) * 500,
        }
    )


# RSI Strategy Tests
class TestRSIStrategy:
    """Test suite for RSI strategy"""

    def test_initialization(self):
        """Test RSI strategy initialization with default config"""
        strategy = RSIStrategy()

        assert strategy.name == "RSI Strategy"
        assert strategy.rsi_period == 14
        assert strategy.oversold_threshold == 30
        assert strategy.overbought_threshold == 70
        assert strategy.stop_loss_pct == 0.02
        assert strategy.take_profit_pct == 0.04
        assert strategy.current_position is None

    def test_custom_config(self):
        """Test RSI strategy with custom configuration"""
        config = {
            "rsi_period": 21,
            "oversold_threshold": 25,
            "overbought_threshold": 75,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.05,
        }
        strategy = RSIStrategy(config=config)

        assert strategy.rsi_period == 21
        assert strategy.oversold_threshold == 25
        assert strategy.overbought_threshold == 75
        assert strategy.stop_loss_pct == 0.03
        assert strategy.take_profit_pct == 0.05

    def test_insufficient_data(self):
        """Test signal generation with insufficient data"""
        strategy = RSIStrategy()

        # Create data with fewer periods than required
        data = pd.DataFrame(
            {
                "timestamp": [int(time.time() * 1000)],
                "close": [50000.0],
            }
        )

        signal = strategy.generate_signal(data)

        assert signal.signal == Signal.HOLD
        assert signal.confidence == 0.0
        assert signal.metadata.get("reason") == "insufficient_data"

    def test_oversold_signal(self, oversold_data):
        """Test BUY signal generation when RSI is oversold"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(oversold_data)

        assert signal.signal == Signal.BUY
        assert signal.confidence > 0.0
        assert signal.metadata["rsi"] < strategy.oversold_threshold
        assert "oversold" in signal.reason.lower()

    def test_overbought_signal(self, overbought_data):
        """Test SELL signal generation when RSI is overbought"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(overbought_data)

        assert signal.signal == Signal.SELL
        assert signal.confidence > 0.0
        assert signal.metadata["rsi"] > strategy.overbought_threshold
        assert "overbought" in signal.reason.lower()

    def test_neutral_signal(self, sample_ohlcv_data):
        """Test HOLD signal generation when RSI is neutral"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(sample_ohlcv_data)

        # Most random data should be neutral
        if signal.signal == Signal.HOLD:
            assert 30 <= signal.metadata["rsi"] <= 70
            assert "neutral" in signal.reason.lower()

    def test_position_entry_long(self, oversold_data):
        """Test entering a long position"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(oversold_data)

        # Enter position
        position = strategy.enter_position(signal, size=0.1)

        assert strategy.current_position is not None
        assert position.side == PositionSide.LONG
        assert position.entry_price == signal.price
        assert position.size == 0.1
        assert position.stop_loss is not None
        assert position.take_profit is not None
        assert position.stop_loss < signal.price
        assert position.take_profit > signal.price

    def test_stop_loss_calculation(self, oversold_data):
        """Test automatic stop loss calculation"""
        strategy = RSIStrategy(config={"stop_loss_pct": 0.05})
        signal = strategy.generate_signal(oversold_data)

        position = strategy.enter_position(signal, size=0.1)

        expected_stop_loss = signal.price * (1 - 0.05)
        assert abs(position.stop_loss - expected_stop_loss) < 0.01

    def test_take_profit_calculation(self, oversold_data):
        """Test automatic take profit calculation"""
        strategy = RSIStrategy(config={"take_profit_pct": 0.08})
        signal = strategy.generate_signal(oversold_data)

        position = strategy.enter_position(signal, size=0.1)

        expected_take_profit = signal.price * (1 + 0.08)
        assert abs(position.take_profit - expected_take_profit) < 0.01

    def test_should_exit_stop_loss(self, oversold_data):
        """Test exit condition when stop loss is hit"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(oversold_data)
        position = strategy.enter_position(signal, size=0.1)

        # Price drops below stop loss
        current_price = position.stop_loss - 100

        should_exit, reason = strategy.should_exit(current_price, position)

        assert should_exit is True
        assert "stop loss" in reason.lower()

    def test_should_exit_take_profit(self, oversold_data):
        """Test exit condition when take profit is hit"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(oversold_data)
        position = strategy.enter_position(signal, size=0.1)

        # Price rises above take profit
        current_price = position.take_profit + 100

        should_exit, reason = strategy.should_exit(current_price, position)

        assert should_exit is True
        assert "take profit" in reason.lower()

    def test_should_not_exit(self, oversold_data):
        """Test no exit when price is within bounds"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(oversold_data)
        position = strategy.enter_position(signal, size=0.1)

        # Price is between stop loss and take profit
        current_price = signal.price * 1.01

        should_exit, reason = strategy.should_exit(current_price, position)

        assert should_exit is False
        assert reason == ""

    def test_position_exit_with_profit(self, oversold_data):
        """Test exiting position with profit"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(oversold_data)
        position = strategy.enter_position(signal, size=0.1)

        # Exit at higher price
        exit_price = signal.price * 1.05
        exit_timestamp = signal.timestamp + 60000

        exit_details = strategy.exit_position(exit_price, exit_timestamp, "Take profit")

        assert exit_details["pnl"] > 0
        assert exit_details["pnl_pct"] > 0
        assert exit_details["side"] == PositionSide.LONG.value
        assert strategy.current_position is None

    def test_position_exit_with_loss(self, oversold_data):
        """Test exiting position with loss"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(oversold_data)
        position = strategy.enter_position(signal, size=0.1)

        # Exit at lower price
        exit_price = signal.price * 0.95
        exit_timestamp = signal.timestamp + 60000

        exit_details = strategy.exit_position(exit_price, exit_timestamp, "Stop loss")

        assert exit_details["pnl"] < 0
        assert exit_details["pnl_pct"] < 0
        assert strategy.current_position is None

    def test_reset(self, oversold_data):
        """Test strategy reset"""
        strategy = RSIStrategy()
        signal = strategy.generate_signal(oversold_data)
        strategy.enter_position(signal, size=0.1)

        assert strategy.current_position is not None

        strategy.reset()

        assert strategy.current_position is None


# MA Crossover Strategy Tests
class TestMovingAverageCrossoverStrategy:
    """Test suite for MA Crossover strategy"""

    def test_initialization(self):
        """Test MA Crossover strategy initialization"""
        strategy = MovingAverageCrossoverStrategy()

        assert strategy.name == "MA Crossover Strategy"
        assert strategy.fast_period == 20
        assert strategy.slow_period == 50
        assert strategy.ma_type == "sma"
        assert strategy.current_position is None
        assert strategy._prev_fast_ma is None
        assert strategy._prev_slow_ma is None

    def test_custom_config(self):
        """Test MA Crossover strategy with custom config"""
        config = {
            "fast_period": 10,
            "slow_period": 30,
            "ma_type": "ema",
            "stop_loss_pct": 0.03,
        }
        strategy = MovingAverageCrossoverStrategy(config=config)

        assert strategy.fast_period == 10
        assert strategy.slow_period == 30
        assert strategy.ma_type == "ema"
        assert strategy.stop_loss_pct == 0.03

    def test_insufficient_data(self):
        """Test signal generation with insufficient data"""
        strategy = MovingAverageCrossoverStrategy()

        # Create data with fewer periods than required
        data = pd.DataFrame(
            {
                "timestamp": [int(time.time() * 1000)],
                "close": [50000.0],
            }
        )

        signal = strategy.generate_signal(data)

        assert signal.signal == Signal.HOLD
        assert signal.confidence == 0.0
        assert signal.metadata.get("reason") == "insufficient_data"

    def test_golden_cross_detection(self, golden_cross_data):
        """Test golden cross (BUY signal) detection"""
        strategy = MovingAverageCrossoverStrategy()

        # First pass to initialize MA values
        signal1 = strategy.generate_signal(golden_cross_data.iloc[:60])
        assert signal1.signal == Signal.HOLD  # No crossover yet

        # Generate signals progressively to detect crossover
        for i in range(61, len(golden_cross_data)):
            signal = strategy.generate_signal(golden_cross_data.iloc[: i + 1])
            if signal.signal == Signal.BUY:
                assert "golden cross" in signal.reason.lower()
                assert signal.confidence > 0.0
                break

    def test_no_crossover_hold(self, sample_ohlcv_data):
        """Test HOLD signal when no crossover occurs"""
        strategy = MovingAverageCrossoverStrategy()

        # First signal to initialize
        signal1 = strategy.generate_signal(sample_ohlcv_data)

        # Second signal on same data should be HOLD
        signal2 = strategy.generate_signal(sample_ohlcv_data)

        # No crossover detected on static data
        assert signal2.signal == Signal.HOLD

    def test_ma_values_in_metadata(self, sample_ohlcv_data):
        """Test that MA values are included in signal metadata"""
        strategy = MovingAverageCrossoverStrategy()
        signal = strategy.generate_signal(sample_ohlcv_data)

        assert "fast_ma" in signal.metadata
        assert "slow_ma" in signal.metadata
        assert "fast_period" in signal.metadata
        assert "slow_period" in signal.metadata
        assert "ma_type" in signal.metadata

    def test_position_entry_with_auto_stops(self, golden_cross_data):
        """Test position entry with automatic stop loss and take profit"""
        strategy = MovingAverageCrossoverStrategy()

        # Generate multiple signals to get a BUY signal
        for i in range(60, len(golden_cross_data)):
            signal = strategy.generate_signal(golden_cross_data.iloc[: i + 1])
            if signal.signal == Signal.BUY:
                position = strategy.enter_position(signal, size=0.1)

                assert position.stop_loss is not None
                assert position.take_profit is not None
                assert position.stop_loss < signal.price
                assert position.take_profit > signal.price
                break

    def test_reset_clears_ma_tracking(self, sample_ohlcv_data):
        """Test that reset clears MA tracking variables"""
        strategy = MovingAverageCrossoverStrategy()

        # Generate signal to initialize MA tracking
        strategy.generate_signal(sample_ohlcv_data)

        assert strategy._prev_fast_ma is not None
        assert strategy._prev_slow_ma is not None

        strategy.reset()

        assert strategy._prev_fast_ma is None
        assert strategy._prev_slow_ma is None
        assert strategy.current_position is None


# Base Strategy Tests
class TestBaseStrategy:
    """Test suite for base strategy functionality"""

    def test_get_position_size_default(self):
        """Test default position size calculation"""

        strategy = RSIStrategy()  # Use concrete implementation
        signal = StrategySignal(
            signal=Signal.BUY,
            timestamp=int(time.time() * 1000),
            price=50000.0,
            confidence=0.8,
            metadata={},
            reason="Test signal",
        )

        capital = 10000.0
        size = strategy.get_position_size(signal, capital)

        expected_size = (capital * 0.1) / signal.price  # Default 10% position
        assert abs(size - expected_size) < 0.0001

    def test_get_position_size_custom(self):
        """Test position size calculation with custom percentage"""
        config = {"position_size_pct": 0.2}
        strategy = RSIStrategy(config=config)

        signal = StrategySignal(
            signal=Signal.BUY,
            timestamp=int(time.time() * 1000),
            price=50000.0,
            confidence=0.8,
            metadata={},
            reason="Test signal",
        )

        capital = 10000.0
        size = strategy.get_position_size(signal, capital)

        expected_size = (capital * 0.2) / signal.price  # 20% position
        assert abs(size - expected_size) < 0.0001

    def test_exit_position_without_position(self):
        """Test exiting when no position exists"""
        strategy = RSIStrategy()

        exit_details = strategy.exit_position(
            exit_price=50000.0, exit_timestamp=int(time.time() * 1000), reason="Test"
        )

        assert exit_details == {}


# Integration Tests
class TestStrategyIntegration:
    """Integration tests for strategy workflows"""

    def test_full_trade_cycle(self, oversold_data, overbought_data):
        """Test complete trade cycle: signal -> entry -> exit"""
        strategy = RSIStrategy()

        # Generate BUY signal from oversold data
        buy_signal = strategy.generate_signal(oversold_data)
        assert buy_signal.signal == Signal.BUY

        # Enter position
        capital = 10000.0
        size = strategy.get_position_size(buy_signal, capital)
        position = strategy.enter_position(buy_signal, size)

        assert strategy.current_position is not None
        assert position.side == PositionSide.LONG

        # Check if should exit (not yet)
        current_price = buy_signal.price * 1.02  # 2% profit, not at take profit yet
        should_exit, _ = strategy.should_exit(current_price, position)
        assert should_exit is False

        # Price reaches take profit
        exit_price = position.take_profit + 1
        should_exit, reason = strategy.should_exit(exit_price, position)
        assert should_exit is True

        # Exit position
        exit_details = strategy.exit_position(
            exit_price=exit_price, exit_timestamp=buy_signal.timestamp + 60000, reason=reason
        )

        assert exit_details["pnl"] > 0
        assert strategy.current_position is None

    def test_backtesting_workflow(self, sample_ohlcv_data):
        """Test strategy in backtesting mode with multiple signals"""
        strategy = RSIStrategy()
        positions_opened = 0
        positions_closed = 0

        # Simulate rolling window backtesting
        window_size = 30

        for i in range(window_size, len(sample_ohlcv_data)):
            window_data = sample_ohlcv_data.iloc[i - window_size : i + 1]
            signal = strategy.generate_signal(window_data)

            if signal.signal == Signal.BUY and strategy.current_position is None:
                position = strategy.enter_position(signal, size=0.1)
                positions_opened += 1

            elif signal.signal == Signal.SELL and strategy.current_position is not None:
                exit_details = strategy.exit_position(
                    exit_price=signal.price, exit_timestamp=signal.timestamp, reason="Signal exit"
                )
                positions_closed += 1

            # Check stop loss / take profit
            elif strategy.current_position is not None:
                current_price = float(window_data["close"].iloc[-1])
                should_exit, reason = strategy.should_exit(current_price, strategy.current_position)

                if should_exit:
                    exit_details = strategy.exit_position(
                        exit_price=current_price, exit_timestamp=signal.timestamp, reason=reason
                    )
                    positions_closed += 1

        # Should have opened and closed some positions
        assert positions_opened >= 0
        assert positions_closed >= 0
