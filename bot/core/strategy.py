"""
Trading strategy module with base strategy class and technical implementations.

This module provides:
- Base strategy interface for signal generation
- RSI-based strategy implementation
- Moving Average Crossover strategy implementation
- Position entry/exit logic
- Configuration support for strategy parameters
- Backtestable implementation

Architecture:
- Abstract base strategy class defining the interface
- Concrete implementations for specific strategies
- Signal types: BUY, SELL, HOLD
- Position management with entry/exit rules
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
from structlog import get_logger

logger = get_logger(__name__)


class Signal(str, Enum):
    """Trading signal types"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class PositionSide(str, Enum):
    """Position side types"""

    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class StrategySignal:
    """
    Strategy signal with metadata.

    Attributes:
        signal: Signal type (BUY, SELL, HOLD)
        timestamp: Signal timestamp in milliseconds
        price: Current price at signal generation
        confidence: Signal confidence score (0-1)
        metadata: Additional signal metadata (e.g., indicator values)
        reason: Human-readable reason for the signal
    """

    signal: Signal
    timestamp: int
    price: float
    confidence: float
    metadata: dict[str, Any]
    reason: str


@dataclass
class Position:
    """
    Trading position representation.

    Attributes:
        side: Position side (LONG, SHORT, NONE)
        entry_price: Entry price
        entry_timestamp: Entry timestamp in milliseconds
        size: Position size
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        metadata: Additional position metadata
    """

    side: PositionSide
    entry_price: float
    entry_timestamp: int
    size: float
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] | None = None


class BaseStrategy(ABC):
    """
    Abstract base strategy class.

    All strategies must implement:
    - generate_signal: Generate trading signal from market data
    - should_exit: Check if current position should be exited
    - get_position_size: Calculate position size for entry

    Strategies receive OHLCV data and generate trading signals.
    """

    def __init__(self, name: str, config: dict[str, Any] | None = None):
        """
        Initialize strategy.

        Args:
            name: Strategy name
            config: Strategy configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.current_position: Position | None = None

        logger.info("strategy_initialized", name=name, config=config)

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """
        Generate trading signal from market data.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume, timestamp

        Returns:
            StrategySignal with signal type and metadata
        """
        pass

    @abstractmethod
    def should_exit(self, current_price: float, position: Position) -> tuple[bool, str]:
        """
        Check if current position should be exited.

        Args:
            current_price: Current market price
            position: Current position

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        pass

    def get_position_size(self, signal: StrategySignal, capital: float) -> float:
        """
        Calculate position size for entry.

        Default implementation uses fixed percentage of capital.
        Override for custom position sizing logic.

        Args:
            signal: Trading signal
            capital: Available capital

        Returns:
            Position size in base currency units
        """
        position_pct = float(self.config.get("position_size_pct", 0.1))
        return capital * position_pct / signal.price

    def enter_position(
        self,
        signal: StrategySignal,
        size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Position:
        """
        Enter a new position.

        Args:
            signal: Trading signal that triggered entry
            size: Position size
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price

        Returns:
            Position object
        """
        side = PositionSide.LONG if signal.signal == Signal.BUY else PositionSide.SHORT

        position = Position(
            side=side,
            entry_price=signal.price,
            entry_timestamp=signal.timestamp,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata=signal.metadata,
        )

        self.current_position = position

        logger.info(
            "position_entered",
            strategy=self.name,
            side=side.value,
            entry_price=signal.price,
            size=size,
            reason=signal.reason,
        )

        return position

    def exit_position(self, exit_price: float, exit_timestamp: int, reason: str) -> dict[str, Any]:
        """
        Exit current position and calculate PnL.

        Args:
            exit_price: Exit price
            exit_timestamp: Exit timestamp in milliseconds
            reason: Reason for exit

        Returns:
            Dictionary with exit details and PnL
        """
        if not self.current_position:
            logger.warning("no_position_to_exit", strategy=self.name)
            return {}

        position = self.current_position

        # Calculate PnL
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * position.size
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:  # SHORT
            pnl = (position.entry_price - exit_price) * position.size
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        exit_details = {
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_timestamp": position.entry_timestamp,
            "exit_timestamp": exit_timestamp,
            "side": position.side.value,
            "size": position.size,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
            "holding_period_ms": exit_timestamp - position.entry_timestamp,
        }

        logger.info(
            "position_exited",
            strategy=self.name,
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )

        self.current_position = None
        return exit_details

    def reset(self) -> None:
        """Reset strategy state (useful for backtesting)"""
        self.current_position = None
        logger.debug("strategy_reset", strategy=self.name)


class RSIStrategy(BaseStrategy):
    """
    RSI-based trading strategy.

    Strategy Logic:
    - BUY signal when RSI < oversold_threshold (default 30) - oversold condition
    - SELL signal when RSI > overbought_threshold (default 70) - overbought condition
    - HOLD signal otherwise
    - Exit when opposite signal is triggered or stop loss/take profit is hit

    Configuration Parameters:
    - rsi_period: RSI calculation period (default 14)
    - oversold_threshold: RSI threshold for buy signal (default 30)
    - overbought_threshold: RSI threshold for sell signal (default 70)
    - stop_loss_pct: Stop loss percentage from entry (default 0.02 = 2%)
    - take_profit_pct: Take profit percentage from entry (default 0.04 = 4%)
    - position_size_pct: Position size as % of capital (default 0.1 = 10%)
    """

    DEFAULT_CONFIG = {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "position_size_pct": 0.1,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize RSI strategy.

        Args:
            config: Strategy configuration parameters
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name="RSI Strategy", config=merged_config)

        # Extract config parameters
        self.rsi_period = self.config["rsi_period"]
        self.oversold_threshold = self.config["oversold_threshold"]
        self.overbought_threshold = self.config["overbought_threshold"]
        self.stop_loss_pct = self.config["stop_loss_pct"]
        self.take_profit_pct = self.config["take_profit_pct"]

    def _calculate_rsi(self, data: pd.DataFrame) -> Any:
        """
        Calculate RSI indicator.

        Args:
            data: DataFrame with 'close' column

        Returns:
            Series with RSI values
        """
        import ta

        return ta.momentum.RSIIndicator(close=data["close"], window=self.rsi_period).rsi()

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """
        Generate trading signal based on RSI.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume, timestamp

        Returns:
            StrategySignal with signal type and metadata
        """
        if len(data) < self.rsi_period:
            logger.warning(
                "insufficient_data_for_rsi",
                data_length=len(data),
                required=self.rsi_period,
            )
            return StrategySignal(
                signal=Signal.HOLD,
                timestamp=int(data["timestamp"].iloc[-1]) if "timestamp" in data.columns else 0,
                price=float(data["close"].iloc[-1]),
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
                reason="Insufficient data for RSI calculation",
            )

        # Calculate RSI
        rsi_values = self._calculate_rsi(data)
        current_rsi = float(rsi_values.iloc[-1])
        current_price = float(data["close"].iloc[-1])
        timestamp = int(data["timestamp"].iloc[-1]) if "timestamp" in data.columns else 0

        # Generate signal based on RSI thresholds
        if current_rsi < self.oversold_threshold:
            # Oversold - BUY signal
            confidence = 1.0 - (current_rsi / self.oversold_threshold)
            signal = Signal.BUY
            reason = f"RSI oversold: {current_rsi:.2f} < {self.oversold_threshold}"

        elif current_rsi > self.overbought_threshold:
            # Overbought - SELL signal
            confidence = (current_rsi - self.overbought_threshold) / (
                100 - self.overbought_threshold
            )
            signal = Signal.SELL
            reason = f"RSI overbought: {current_rsi:.2f} > {self.overbought_threshold}"

        else:
            # Neutral zone - HOLD
            signal = Signal.HOLD
            confidence = 0.5
            reason = f"RSI neutral: {current_rsi:.2f}"

        logger.debug(
            "signal_generated",
            strategy=self.name,
            signal=signal.value,
            rsi=current_rsi,
            confidence=confidence,
        )

        return StrategySignal(
            signal=signal,
            timestamp=timestamp,
            price=current_price,
            confidence=confidence,
            metadata={
                "rsi": current_rsi,
                "oversold_threshold": self.oversold_threshold,
                "overbought_threshold": self.overbought_threshold,
            },
            reason=reason,
        )

    def should_exit(self, current_price: float, position: Position) -> tuple[bool, str]:
        """
        Check if current position should be exited.

        Exit conditions:
        1. Stop loss hit
        2. Take profit hit
        3. RSI crosses opposite threshold (managed by signal generation)

        Args:
            current_price: Current market price
            position: Current position

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        # Check stop loss
        if position.stop_loss:
            if position.side == PositionSide.LONG and current_price <= position.stop_loss:
                return True, f"Stop loss hit: {current_price} <= {position.stop_loss}"
            elif position.side == PositionSide.SHORT and current_price >= position.stop_loss:
                return True, f"Stop loss hit: {current_price} >= {position.stop_loss}"

        # Check take profit
        if position.take_profit:
            if position.side == PositionSide.LONG and current_price >= position.take_profit:
                return True, f"Take profit hit: {current_price} >= {position.take_profit}"
            elif position.side == PositionSide.SHORT and current_price <= position.take_profit:
                return True, f"Take profit hit: {current_price} <= {position.take_profit}"

        return False, ""

    def enter_position(
        self,
        signal: StrategySignal,
        size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Position:
        """
        Enter position with automatic stop loss and take profit calculation.

        Args:
            signal: Trading signal
            size: Position size
            stop_loss: Optional manual stop loss (overrides auto calculation)
            take_profit: Optional manual take profit (overrides auto calculation)

        Returns:
            Position object
        """
        # Calculate stop loss and take profit if not provided
        if stop_loss is None:
            if signal.signal == Signal.BUY:
                stop_loss = signal.price * (1 - self.stop_loss_pct)
            else:  # SELL/SHORT
                stop_loss = signal.price * (1 + self.stop_loss_pct)

        if take_profit is None:
            if signal.signal == Signal.BUY:
                take_profit = signal.price * (1 + self.take_profit_pct)
            else:  # SELL/SHORT
                take_profit = signal.price * (1 - self.take_profit_pct)

        return super().enter_position(signal, size, stop_loss, take_profit)


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover trading strategy.

    Strategy Logic:
    - BUY signal when fast MA crosses above slow MA (golden cross)
    - SELL signal when fast MA crosses below slow MA (death cross)
    - HOLD signal otherwise
    - Exit when opposite signal is triggered or stop loss/take profit is hit

    Configuration Parameters:
    - fast_period: Fast MA period (default 20)
    - slow_period: Slow MA period (default 50)
    - ma_type: MA type - 'sma' or 'ema' (default 'sma')
    - stop_loss_pct: Stop loss percentage from entry (default 0.02 = 2%)
    - take_profit_pct: Take profit percentage from entry (default 0.04 = 4%)
    - position_size_pct: Position size as % of capital (default 0.1 = 10%)
    """

    DEFAULT_CONFIG = {
        "fast_period": 20,
        "slow_period": 50,
        "ma_type": "sma",
        "stop_loss_pct": 0.02,
        "take_profit_pct": 0.04,
        "position_size_pct": 0.1,
    }

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Moving Average Crossover strategy.

        Args:
            config: Strategy configuration parameters
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name="MA Crossover Strategy", config=merged_config)

        # Extract config parameters
        self.fast_period = self.config["fast_period"]
        self.slow_period = self.config["slow_period"]
        self.ma_type = self.config["ma_type"]
        self.stop_loss_pct = self.config["stop_loss_pct"]
        self.take_profit_pct = self.config["take_profit_pct"]

        # Track previous MA values for crossover detection
        self._prev_fast_ma: float | None = None
        self._prev_slow_ma: float | None = None

    def _calculate_ma(self, data: pd.DataFrame, period: int) -> Any:
        """
        Calculate moving average.

        Args:
            data: DataFrame with 'close' column
            period: MA period

        Returns:
            Series with MA values
        """
        import ta

        if self.ma_type == "ema":
            return ta.trend.EMAIndicator(close=data["close"], window=period).ema_indicator()
        else:  # SMA
            return ta.trend.SMAIndicator(close=data["close"], window=period).sma_indicator()

    def generate_signal(self, data: pd.DataFrame) -> StrategySignal:
        """
        Generate trading signal based on MA crossover.

        Args:
            data: OHLCV DataFrame with columns: open, high, low, close, volume, timestamp

        Returns:
            StrategySignal with signal type and metadata
        """
        if len(data) < self.slow_period:
            logger.warning(
                "insufficient_data_for_ma",
                data_length=len(data),
                required=self.slow_period,
            )
            return StrategySignal(
                signal=Signal.HOLD,
                timestamp=int(data["timestamp"].iloc[-1]) if "timestamp" in data.columns else 0,
                price=float(data["close"].iloc[-1]),
                confidence=0.0,
                metadata={"reason": "insufficient_data"},
                reason="Insufficient data for MA calculation",
            )

        # Calculate MAs
        fast_ma = self._calculate_ma(data, self.fast_period)
        slow_ma = self._calculate_ma(data, self.slow_period)

        current_fast_ma = float(fast_ma.iloc[-1])
        current_slow_ma = float(slow_ma.iloc[-1])
        current_price = float(data["close"].iloc[-1])
        timestamp = int(data["timestamp"].iloc[-1]) if "timestamp" in data.columns else 0

        # Detect crossover
        signal = Signal.HOLD
        confidence = 0.5
        reason = "No crossover detected"

        if self._prev_fast_ma is not None and self._prev_slow_ma is not None:
            # Golden cross: fast MA crosses above slow MA
            if self._prev_fast_ma <= self._prev_slow_ma and current_fast_ma > current_slow_ma:
                signal = Signal.BUY
                confidence = min(1.0, (current_fast_ma - current_slow_ma) / current_slow_ma)
                reason = f"Golden cross: Fast MA ({current_fast_ma:.2f}) crossed above Slow MA ({current_slow_ma:.2f})"

            # Death cross: fast MA crosses below slow MA
            elif self._prev_fast_ma >= self._prev_slow_ma and current_fast_ma < current_slow_ma:
                signal = Signal.SELL
                confidence = min(1.0, (current_slow_ma - current_fast_ma) / current_slow_ma)
                reason = f"Death cross: Fast MA ({current_fast_ma:.2f}) crossed below Slow MA ({current_slow_ma:.2f})"

            # Check if still in bullish/bearish trend
            elif current_fast_ma > current_slow_ma:
                reason = f"Bullish trend: Fast MA ({current_fast_ma:.2f}) above Slow MA ({current_slow_ma:.2f})"
            else:
                reason = f"Bearish trend: Fast MA ({current_fast_ma:.2f}) below Slow MA ({current_slow_ma:.2f})"

        # Update previous values for next iteration
        self._prev_fast_ma = current_fast_ma
        self._prev_slow_ma = current_slow_ma

        logger.debug(
            "signal_generated",
            strategy=self.name,
            signal=signal.value,
            fast_ma=current_fast_ma,
            slow_ma=current_slow_ma,
            confidence=confidence,
        )

        return StrategySignal(
            signal=signal,
            timestamp=timestamp,
            price=current_price,
            confidence=confidence,
            metadata={
                "fast_ma": current_fast_ma,
                "slow_ma": current_slow_ma,
                "fast_period": self.fast_period,
                "slow_period": self.slow_period,
                "ma_type": self.ma_type,
            },
            reason=reason,
        )

    def should_exit(self, current_price: float, position: Position) -> tuple[bool, str]:
        """
        Check if current position should be exited.

        Exit conditions:
        1. Stop loss hit
        2. Take profit hit
        3. Opposite crossover (managed by signal generation)

        Args:
            current_price: Current market price
            position: Current position

        Returns:
            Tuple of (should_exit: bool, reason: str)
        """
        # Check stop loss
        if position.stop_loss:
            if position.side == PositionSide.LONG and current_price <= position.stop_loss:
                return True, f"Stop loss hit: {current_price} <= {position.stop_loss}"
            elif position.side == PositionSide.SHORT and current_price >= position.stop_loss:
                return True, f"Stop loss hit: {current_price} >= {position.stop_loss}"

        # Check take profit
        if position.take_profit:
            if position.side == PositionSide.LONG and current_price >= position.take_profit:
                return True, f"Take profit hit: {current_price} >= {position.take_profit}"
            elif position.side == PositionSide.SHORT and current_price <= position.take_profit:
                return True, f"Take profit hit: {current_price} <= {position.take_profit}"

        return False, ""

    def enter_position(
        self,
        signal: StrategySignal,
        size: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Position:
        """
        Enter position with automatic stop loss and take profit calculation.

        Args:
            signal: Trading signal
            size: Position size
            stop_loss: Optional manual stop loss (overrides auto calculation)
            take_profit: Optional manual take profit (overrides auto calculation)

        Returns:
            Position object
        """
        # Calculate stop loss and take profit if not provided
        if stop_loss is None:
            if signal.signal == Signal.BUY:
                stop_loss = signal.price * (1 - self.stop_loss_pct)
            else:  # SELL/SHORT
                stop_loss = signal.price * (1 + self.stop_loss_pct)

        if take_profit is None:
            if signal.signal == Signal.BUY:
                take_profit = signal.price * (1 + self.take_profit_pct)
            else:  # SELL/SHORT
                take_profit = signal.price * (1 - self.take_profit_pct)

        return super().enter_position(signal, size, stop_loss, take_profit)

    def reset(self) -> None:
        """Reset strategy state including MA tracking"""
        super().reset()
        self._prev_fast_ma = None
        self._prev_slow_ma = None
