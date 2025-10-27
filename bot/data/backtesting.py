"""
Backtesting engine for historical strategy simulation.

This module provides a comprehensive backtesting framework with:
- Historical data loading from CSV/Parquet
- Bar-by-bar and tick-by-tick replay
- Realistic order execution simulation with slippage and latency
- Performance metrics calculation (win rate, Sharpe, drawdown)
- Multi-strategy and multi-symbol support
- Detailed performance reports

Architecture:
- BacktestEngine: Main simulation engine
- BacktestConfig: Configuration for backtest runs
- BacktestResults: Results container with metrics
- SimulatedExchange: Exchange simulator with realistic execution
"""

import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from structlog import get_logger

from bot.core.strategy import BaseStrategy, Position, RSIStrategy, Signal, StrategySignal

logger = get_logger(__name__)


class ReplayMode(str, Enum):
    """Data replay mode for backtesting"""

    BAR = "bar"  # Bar-by-bar (OHLCV candles)
    TICK = "tick"  # Tick-by-tick (individual trades/quotes)


@dataclass
class TradeRecord:
    """
    Record of a completed trade with entry and exit details.

    Attributes:
        symbol: Trading symbol
        side: Position side (LONG/SHORT)
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        entry_price: Entry price
        exit_price: Exit price
        size: Position size
        pnl: Profit/loss in absolute terms
        pnl_pct: Profit/loss percentage
        commission: Total commission paid
        slippage: Total slippage cost
        reason: Exit reason
        holding_period: Duration of trade in seconds
    """

    symbol: str
    side: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    reason: str
    holding_period: float

    def to_dict(self) -> dict[str, Any]:
        """Convert trade record to dictionary"""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_time": (
                self.entry_time.isoformat()
                if isinstance(self.entry_time, datetime)
                else str(self.entry_time)
            ),
            "exit_time": (
                self.exit_time.isoformat()
                if isinstance(self.exit_time, datetime)
                else str(self.exit_time)
            ),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "size": self.size,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "commission": self.commission,
            "slippage": self.slippage,
            "reason": self.reason,
            "holding_period": self.holding_period,
        }


@dataclass
class BacktestConfig:
    """
    Configuration for backtest execution.

    Attributes:
        data_file: Path to historical data file (CSV or Parquet)
        strategy_name: Name of strategy to backtest
        symbol: Trading symbol
        initial_capital: Starting capital
        commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%)
        slippage_factor: Slippage factor for execution (e.g., 0.001 = 0.1%)
        start_date: Optional start date for backtest
        end_date: Optional end date for backtest
        replay_mode: Replay mode (bar or tick)
        execution_delay_ms: Simulated execution delay in milliseconds
        strategy_config: Strategy-specific configuration
    """

    data_file: str
    strategy_name: str
    symbol: str
    initial_capital: float = 10000.0
    commission_rate: float = 0.001
    slippage_factor: float = 0.001
    start_date: datetime | None = None
    end_date: datetime | None = None
    replay_mode: str = "bar"
    execution_delay_ms: int = 100
    strategy_config: dict[str, Any] | None = None


@dataclass
class BacktestResults:
    """
    Container for backtest results and performance metrics.

    Attributes:
        config: Backtest configuration used
        trades: List of completed trades
        equity_curve: Time series of portfolio value
        metrics: Performance metrics dictionary
        strategy_name: Strategy name
        symbol: Trading symbol
        start_time: Backtest start time
        end_time: Backtest end time
    """

    config: BacktestConfig
    trades: list[TradeRecord]
    equity_curve: pd.DataFrame
    metrics: dict[str, Any]
    strategy_name: str
    symbol: str
    start_time: datetime
    end_time: datetime

    def save_report(self, output_file: str) -> None:
        """
        Save backtest results to JSON file.

        Args:
            output_file: Path to output file
        """
        report = {
            "config": {
                "data_file": self.config.data_file,
                "strategy_name": self.config.strategy_name,
                "symbol": self.config.symbol,
                "initial_capital": self.config.initial_capital,
                "commission_rate": self.config.commission_rate,
                "slippage_factor": self.config.slippage_factor,
                "replay_mode": self.config.replay_mode,
                "execution_delay_ms": self.config.execution_delay_ms,
            },
            "summary": {
                "strategy": self.strategy_name,
                "symbol": self.symbol,
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "duration_days": (self.end_time - self.start_time).days,
            },
            "metrics": self.metrics,
            "trades": [trade.to_dict() for trade in self.trades],
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("backtest_report_saved", output_file=output_file)

        # Save trades to CSV
        if self.trades:
            trades_df = pd.DataFrame([trade.to_dict() for trade in self.trades])
            csv_file = Path(output_file).parent / "trades.csv"
            trades_df.to_csv(csv_file, index=False)
            logger.info("trades_csv_saved", csv_file=str(csv_file))

        # Save equity curve
        equity_file = Path(output_file).parent / "equity_curve.csv"
        self.equity_curve.to_csv(equity_file)
        logger.info("equity_curve_saved", equity_file=str(equity_file))


class SimulatedExchange:
    """
    Simulated exchange for backtesting with realistic execution modeling.

    Features:
    - Slippage simulation based on volatility
    - Execution delay modeling
    - Commission calculation
    - Order fill simulation
    """

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_factor: float = 0.001,
        execution_delay_ms: int = 100,
    ):
        """
        Initialize simulated exchange.

        Args:
            commission_rate: Commission rate per trade
            slippage_factor: Slippage factor for execution
            execution_delay_ms: Execution delay in milliseconds
        """
        self.commission_rate = commission_rate
        self.slippage_factor = slippage_factor
        self.execution_delay_ms = execution_delay_ms

        logger.info(
            "simulated_exchange_initialized",
            commission_rate=commission_rate,
            slippage_factor=slippage_factor,
            execution_delay_ms=execution_delay_ms,
        )

    def calculate_slippage(
        self,
        price: float,
        side: str,
        volatility: float = 0.01,
    ) -> tuple[float, float]:
        """
        Calculate realistic slippage based on market conditions.

        Slippage model:
        - Base slippage from slippage_factor
        - Additional slippage based on volatility
        - Random component for realism
        - Always unfavorable to the trader

        Args:
            price: Order price
            side: Order side (LONG/SHORT)
            volatility: Current market volatility

        Returns:
            Tuple of (execution_price, slippage_cost)
        """
        # Base slippage
        base_slippage = price * self.slippage_factor

        # Volatility-adjusted slippage (higher volatility = more slippage)
        vol_slippage = price * volatility * 0.5

        # Random component (0-50% of base slippage)
        random_slippage = base_slippage * random.uniform(0, 0.5)

        # Total slippage
        total_slippage = base_slippage + vol_slippage + random_slippage

        # Apply slippage (always unfavorable)
        if side == "LONG":
            # Buy: price increases
            execution_price = price + total_slippage
        else:
            # Sell: price decreases
            execution_price = price - total_slippage

        slippage_cost = abs(execution_price - price) * 1.0  # Per unit

        return execution_price, slippage_cost

    def execute_order(
        self,
        signal: StrategySignal,
        size: float,
        current_volatility: float = 0.01,
    ) -> tuple[float, float, float]:
        """
        Simulate order execution with slippage and commission.

        Args:
            signal: Trading signal
            size: Order size
            current_volatility: Current market volatility

        Returns:
            Tuple of (execution_price, total_commission, total_slippage)
        """
        # Simulate execution delay
        if self.execution_delay_ms > 0:
            # In backtest, we just record the delay
            # In reality, price might have moved during this time
            pass

        # Determine side
        side = "LONG" if signal.signal == Signal.BUY else "SHORT"

        # Calculate slippage
        execution_price, slippage_per_unit = self.calculate_slippage(
            signal.price,
            side,
            current_volatility,
        )

        # Calculate commission
        notional_value = execution_price * size
        commission = notional_value * self.commission_rate

        # Total slippage cost
        total_slippage = slippage_per_unit * size

        logger.debug(
            "order_executed",
            side=side,
            requested_price=signal.price,
            execution_price=execution_price,
            size=size,
            commission=commission,
            slippage=total_slippage,
        )

        return execution_price, commission, total_slippage


class BacktestEngine:
    """
    Main backtesting engine for strategy simulation.

    Features:
    - Historical data loading from CSV/Parquet
    - Bar-by-bar and tick-by-tick replay
    - Strategy signal generation
    - Order execution simulation
    - Performance tracking
    - Multi-strategy and multi-symbol support
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
        """
        self.config = config
        self.exchange = SimulatedExchange(
            commission_rate=config.commission_rate,
            slippage_factor=config.slippage_factor,
            execution_delay_ms=config.execution_delay_ms,
        )

        # Portfolio state
        self.capital = config.initial_capital
        self.initial_capital = config.initial_capital
        self.position: Position | None = None
        self.trades: list[TradeRecord] = []

        # Equity tracking
        self.equity_curve: list[dict[str, Any]] = []

        # Strategy
        self.strategy = self._create_strategy()

        logger.info(
            "backtest_engine_initialized",
            strategy=config.strategy_name,
            symbol=config.symbol,
            initial_capital=config.initial_capital,
        )

    def _create_strategy(self) -> BaseStrategy:
        """
        Create strategy instance based on configuration.

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy name is not recognized
        """
        strategy_name = self.config.strategy_name.lower()
        strategy_config = self.config.strategy_config

        if strategy_name == "rsi":
            return RSIStrategy(config=strategy_config)
        elif strategy_name == "ma_crossover":
            from bot.core.strategy import MovingAverageCrossoverStrategy

            return MovingAverageCrossoverStrategy(config=strategy_config)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy_name}")

    def load_data(self) -> pd.DataFrame:
        """
        Load historical data from CSV or Parquet file.

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If data file does not exist
        """
        data_file = Path(self.config.data_file)

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        logger.info("loading_historical_data", file=str(data_file))

        # Load based on file extension
        if data_file.suffix == ".csv":
            df = pd.read_csv(data_file)
        elif data_file.suffix == ".parquet":
            df = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")

        # Ensure required columns exist
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert timestamp to datetime if needed
        if df["timestamp"].dtype == "int64":
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        elif df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Filter by date range
        if self.config.start_date:
            df = df[df["timestamp"] >= self.config.start_date]
        if self.config.end_date:
            df = df[df["timestamp"] <= self.config.end_date]

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            "historical_data_loaded",
            rows=len(df),
            start=df["timestamp"].iloc[0],
            end=df["timestamp"].iloc[-1],
        )

        return df

    def calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """
        Calculate rolling volatility for slippage estimation.

        Args:
            data: OHLCV DataFrame
            window: Rolling window for volatility calculation

        Returns:
            Current volatility estimate
        """
        if len(data) < window:
            return 0.01  # Default volatility

        returns = data["close"].pct_change().dropna()
        if len(returns) < window:
            return 0.01

        volatility = returns.iloc[-window:].std()
        return float(volatility) if not pd.isna(volatility) else 0.01

    def process_bar(
        self,
        bar: pd.Series,
        historical_data: pd.DataFrame,
        bar_index: int,
    ) -> None:
        """
        Process a single bar of data.

        Args:
            bar: Current bar (OHLCV)
            historical_data: Historical data up to current bar
            bar_index: Index of current bar
        """
        current_price = float(bar["close"])
        current_time = pd.to_datetime(bar["timestamp"])

        # Check if we need to exit current position
        if self.position:
            should_exit, exit_reason = self.strategy.should_exit(current_price, self.position)

            if should_exit:
                self._exit_position(current_price, current_time, exit_reason, bar)

        # Generate signal if no position
        if not self.position:
            # Generate signal with historical data
            signal = self.strategy.generate_signal(historical_data)

            if signal.signal in [Signal.BUY, Signal.SELL]:
                self._enter_position(signal, bar)

        # Record equity
        portfolio_value = self.capital
        if self.position:
            # Mark-to-market position value
            if self.position.side.value == "long":
                position_value = (current_price - self.position.entry_price) * self.position.size
            else:  # short
                position_value = (self.position.entry_price - current_price) * self.position.size

            portfolio_value += position_value

        self.equity_curve.append(
            {
                "timestamp": current_time,
                "equity": portfolio_value,
                "cash": self.capital,
                "position_value": portfolio_value - self.capital,
            }
        )

    def _enter_position(self, signal: StrategySignal, bar: pd.Series) -> None:
        """
        Enter a new position based on signal.

        Args:
            signal: Trading signal
            bar: Current bar data
        """
        # Calculate position size
        position_size = self.strategy.get_position_size(signal, self.capital)

        if position_size <= 0:
            logger.warning("invalid_position_size", size=position_size)
            return

        # Calculate current volatility for slippage
        volatility = self.calculate_volatility(
            pd.DataFrame([bar]), window=1
        )  # Use close price volatility

        # Execute order
        execution_price, commission, slippage = self.exchange.execute_order(
            signal,
            position_size,
            volatility,
        )

        # Calculate total cost
        total_cost = (execution_price * position_size) + commission + (slippage * position_size)

        # Check if we have enough capital
        if total_cost > self.capital:
            logger.warning("insufficient_capital", required=total_cost, available=self.capital)
            return

        # Deduct from capital
        self.capital -= total_cost

        # Enter position via strategy
        self.position = self.strategy.enter_position(
            signal=signal,
            size=position_size,
            stop_loss=None,  # Strategy will calculate
            take_profit=None,  # Strategy will calculate
        )

        # Update position with actual execution price
        self.position.entry_price = execution_price

        logger.info(
            "position_entered",
            symbol=self.config.symbol,
            side=self.position.side.value,
            size=position_size,
            entry_price=execution_price,
            commission=commission,
            slippage=slippage,
        )

    def _exit_position(
        self,
        exit_price: float,
        exit_time: datetime,
        reason: str,
        bar: pd.Series,
    ) -> None:
        """
        Exit current position.

        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Exit reason
            bar: Current bar data
        """
        if not self.position:
            return

        position = self.position

        # Calculate volatility for slippage
        volatility = self.calculate_volatility(pd.DataFrame([bar]), window=1)

        # Simulate exit order execution
        exit_signal = StrategySignal(
            signal=Signal.SELL if position.side.value == "long" else Signal.BUY,
            timestamp=int(exit_time.timestamp() * 1000),
            price=exit_price,
            confidence=1.0,
            metadata={},
            reason=reason,
        )

        execution_price, commission, slippage = self.exchange.execute_order(
            exit_signal,
            position.size,
            volatility,
        )

        # Calculate PnL
        if position.side.value == "long":
            gross_pnl = (execution_price - position.entry_price) * position.size
            pnl_pct = (execution_price - position.entry_price) / position.entry_price
        else:  # short
            gross_pnl = (position.entry_price - execution_price) * position.size
            pnl_pct = (position.entry_price - execution_price) / position.entry_price

        # Net PnL after commission and slippage
        net_pnl = gross_pnl - commission - (slippage * position.size)

        # Update capital
        position_value = execution_price * position.size
        self.capital += position_value - commission - (slippage * position.size)

        # Record trade
        entry_time = pd.to_datetime(position.entry_timestamp, unit="ms")
        holding_period = (exit_time - entry_time).total_seconds()

        trade = TradeRecord(
            symbol=self.config.symbol,
            side=position.side.value,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=execution_price,
            size=position.size,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            slippage=slippage * position.size,
            reason=reason,
            holding_period=holding_period,
        )

        self.trades.append(trade)

        logger.info(
            "position_exited",
            symbol=self.config.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            exit_price=execution_price,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            reason=reason,
        )

        # Clear position
        self.position = None
        self.strategy.reset()

    def calculate_metrics(self) -> dict[str, Any]:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if not self.trades:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "final_capital": self.capital,
            }

        equity_df = pd.DataFrame(self.equity_curve)

        # Total return
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        # Annual return
        days = (equity_df["timestamp"].iloc[-1] - equity_df["timestamp"].iloc[0]).days
        years = days / 365.25 if days > 0 else 1
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return

        # Returns for ratio calculations
        equity_df["returns"] = equity_df["equity"].pct_change()
        returns = equity_df["returns"].dropna()

        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized

        # Sortino ratio (downside deviation)
        sortino_ratio = 0.0
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)

        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0

        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        avg_win = total_wins / len(winning_trades) if winning_trades else 0.0
        avg_loss = -total_losses / len(losing_trades) if losing_trades else 0.0

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "total_wins": total_wins,
            "total_losses": -total_losses,
            "final_capital": self.capital,
            "total_commission": sum(t.commission for t in self.trades),
            "total_slippage": sum(t.slippage for t in self.trades),
        }

        return metrics

    def run(self) -> BacktestResults:
        """
        Run the backtest simulation.

        Returns:
            BacktestResults with performance metrics

        Raises:
            Exception: If backtest execution fails
        """
        logger.info("backtest_started", config=self.config)
        start_time = time.time()

        try:
            # Load historical data
            data = self.load_data()

            if len(data) == 0:
                raise ValueError("No data available for backtesting")

            backtest_start = data["timestamp"].iloc[0]
            backtest_end = data["timestamp"].iloc[-1]

            # Reset strategy state
            self.strategy.reset()

            # Process data based on replay mode
            if self.config.replay_mode == ReplayMode.BAR:
                # Bar-by-bar replay
                for i in range(len(data)):
                    # Historical data up to current bar (inclusive)
                    historical = data.iloc[: i + 1]

                    # Current bar
                    bar = data.iloc[i]

                    # Process bar
                    self.process_bar(bar, historical, i)

            elif self.config.replay_mode == ReplayMode.TICK:
                # Tick-by-tick replay (simulate by processing OHLC as ticks)
                for i in range(len(data)):
                    historical = data.iloc[: i + 1]
                    bar = data.iloc[i]

                    # Simulate tick sequence: open -> high -> low -> close
                    tick_prices = [
                        float(bar["open"]),
                        float(bar["high"]),
                        float(bar["low"]),
                        float(bar["close"]),
                    ]

                    for tick_price in tick_prices:
                        # Update close price for signal generation
                        bar["close"] = tick_price
                        self.process_bar(bar, historical, i)

            # Close any remaining position
            if self.position:
                final_price = float(data["close"].iloc[-1])
                final_time = pd.to_datetime(data["timestamp"].iloc[-1])
                final_bar = data.iloc[-1]
                self._exit_position(
                    final_price,
                    final_time,
                    "end_of_backtest",
                    final_bar,
                )

            # Calculate metrics
            metrics = self.calculate_metrics()

            # Create results
            equity_df = pd.DataFrame(self.equity_curve)

            results = BacktestResults(
                config=self.config,
                trades=self.trades,
                equity_curve=equity_df,
                metrics=metrics,
                strategy_name=self.config.strategy_name,
                symbol=self.config.symbol,
                start_time=backtest_start,
                end_time=backtest_end,
            )

            elapsed_time = time.time() - start_time

            logger.info(
                "backtest_completed",
                total_trades=len(self.trades),
                final_capital=self.capital,
                total_return=metrics["total_return"],
                elapsed_time=elapsed_time,
            )

            return results

        except Exception as e:
            logger.error("backtest_failed", error=str(e))
            raise
