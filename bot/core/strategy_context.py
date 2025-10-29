"""
Multi-Strategy Context Manager.

This module provides comprehensive support for running multiple concurrent trading strategies
with isolated state management, per-strategy configuration, and aggregated risk control.

Features:
- Support for multiple concurrent strategy instances
- Per-strategy configuration and customization
- Isolated state tracking per strategy
- Strategy-level position tracking
- Aggregated risk management across all strategies
- Portfolio-level exposure and correlation monitoring
- Individual strategy performance metrics
- Centralized signal aggregation and conflict resolution

Architecture:
- StrategyInstance: Wrapper for strategy with isolated state
- StrategyContext: Main orchestrator for multi-strategy management
- Position tracking at both strategy and portfolio levels
- Risk aggregation across all strategies
- Conflict resolution for overlapping signals

Usage:
    >>> from bot.core.strategy import RSIStrategy, MovingAverageCrossoverStrategy
    >>> from bot.risk.risk_manager import RiskManager
    >>> from bot.execution.orchestrator import ExecutionOrchestrator
    >>>
    >>> # Initialize context
    >>> context = StrategyContext(
    ...     risk_manager=risk_manager,
    ...     orchestrator=orchestrator,
    ...     initial_capital=Decimal("10000")
    ... )
    >>>
    >>> # Add strategies
    >>> context.add_strategy(
    ...     name="rsi_btc",
    ...     strategy=RSIStrategy({"rsi_period": 14}),
    ...     symbol="BTCUSDT",
    ...     allocation_pct=0.3,  # 30% of capital
    ...     config={"max_position_size_pct": 0.05}
    ... )
    >>>
    >>> context.add_strategy(
    ...     name="ma_eth",
    ...     strategy=MovingAverageCrossoverStrategy(),
    ...     symbol="ETHUSDT",
    ...     allocation_pct=0.4,  # 40% of capital
    ... )
    >>>
    >>> # Process market data for all strategies
    >>> await context.process_all_strategies(market_data)
    >>>
    >>> # Get aggregated metrics
    >>> metrics = context.get_portfolio_metrics()
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import pandas as pd
from structlog import get_logger

from bot.core.logging_config import CorrelationContext
from bot.core.strategy import BaseStrategy, Position, PositionSide, Signal, StrategySignal
from bot.execution.orchestrator import ExecutionOrchestrator
from bot.interfaces.exchange import OrderSide
from bot.risk.risk_manager import RiskManager

logger = get_logger(__name__)


class SignalConflictResolution(str, Enum):
    """Strategy for resolving conflicting signals from multiple strategies."""

    FIRST_WINS = "first_wins"  # First strategy to generate signal wins
    HIGHEST_CONFIDENCE = "highest_confidence"  # Strategy with highest confidence wins
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by strategy allocation
    VETO = "veto"  # Any HOLD signal vetoes execution
    UNANIMOUS = "unanimous"  # All strategies must agree


class StrategyState(str, Enum):
    """State of a strategy instance."""

    ACTIVE = "active"  # Strategy is actively trading
    PAUSED = "paused"  # Strategy is paused (no new positions)
    STOPPED = "stopped"  # Strategy is stopped (close existing positions)
    ERROR = "error"  # Strategy encountered an error


@dataclass
class StrategyPerformance:
    """
    Performance metrics for a strategy instance.

    Attributes:
        total_trades: Total number of trades executed
        winning_trades: Number of profitable trades
        losing_trades: Number of losing trades
        total_pnl: Total profit/loss in base currency
        total_pnl_pct: Total P&L as percentage of allocated capital
        win_rate: Percentage of winning trades
        avg_win: Average profit per winning trade
        avg_loss: Average loss per losing trade
        profit_factor: Ratio of gross profit to gross loss
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Maximum drawdown from peak
        avg_holding_period_ms: Average holding period in milliseconds
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    total_pnl_pct: float = 0.0
    win_rate: float = 0.0
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: Decimal = Decimal("0")
    avg_holding_period_ms: float = 0.0

    def update_from_trade(self, pnl: Decimal, holding_period_ms: int) -> None:
        """
        Update performance metrics with a completed trade.

        Args:
            pnl: Profit/loss from the trade
            holding_period_ms: Trade holding period in milliseconds
        """
        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1

        # Recalculate metrics
        self._recalculate_metrics()

    def _recalculate_metrics(self) -> None:
        """Recalculate derived performance metrics."""
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades

        # Note: avg_win, avg_loss, profit_factor, sharpe_ratio, max_drawdown
        # would require tracking individual trade results
        # This is a simplified version - full implementation would need trade history


@dataclass
class StrategyInstance:
    """
    Wrapper for a strategy with isolated state and configuration.

    Each strategy instance maintains:
    - Its own strategy object and configuration
    - Isolated position tracking
    - Performance metrics
    - Allocated capital
    - Trading symbol
    - State (active/paused/stopped)
    """

    name: str
    strategy: BaseStrategy
    symbol: str
    allocation_pct: float
    allocated_capital: Decimal
    config: dict[str, Any] = field(default_factory=dict)

    # State management
    state: StrategyState = StrategyState.ACTIVE
    current_position: Position | None = None
    position_value: Decimal = Decimal("0")

    # Performance tracking
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)

    # Trade history
    trade_history: list[dict[str, Any]] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_signal_at: datetime | None = None
    last_trade_at: datetime | None = None

    # Error tracking
    error_count: int = 0
    last_error: str | None = None
    last_error_at: datetime | None = None

    def update_allocation(self, total_capital: Decimal) -> None:
        """
        Update allocated capital based on allocation percentage.

        Args:
            total_capital: Total portfolio capital
        """
        self.allocated_capital = total_capital * Decimal(str(self.allocation_pct))
        logger.info(
            "strategy_allocation_updated",
            strategy=self.name,
            allocation_pct=f"{self.allocation_pct:.2%}",
            allocated_capital=str(self.allocated_capital),
        )

    def record_error(self, error: str) -> None:
        """
        Record an error for this strategy.

        Args:
            error: Error message
        """
        self.error_count += 1
        self.last_error = error
        self.last_error_at = datetime.now(UTC)

        if self.error_count >= 5:  # Threshold for auto-pause
            self.state = StrategyState.ERROR
            logger.error(
                "strategy_auto_paused_due_to_errors",
                strategy=self.name,
                error_count=self.error_count,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            "name": self.name,
            "strategy_type": self.strategy.__class__.__name__,
            "symbol": self.symbol,
            "allocation_pct": f"{self.allocation_pct:.2%}",
            "allocated_capital": str(self.allocated_capital),
            "state": self.state.value,
            "position_value": str(self.position_value),
            "has_position": self.current_position is not None,
            "performance": {
                "total_trades": self.performance.total_trades,
                "winning_trades": self.performance.winning_trades,
                "losing_trades": self.performance.losing_trades,
                "total_pnl": str(self.performance.total_pnl),
                "win_rate": f"{self.performance.win_rate:.2%}",
            },
            "error_count": self.error_count,
            "created_at": self.created_at.isoformat(),
        }


class StrategyContext:
    """
    Multi-strategy context manager with aggregated risk management.

    This class orchestrates multiple trading strategies, manages their isolated state,
    aggregates risk across all strategies, and coordinates order execution.

    Features:
    - Add/remove strategy instances dynamically
    - Process market data for all or specific strategies
    - Aggregate signals with conflict resolution
    - Track positions at both strategy and portfolio levels
    - Monitor portfolio-level exposure and risk
    - Individual strategy performance tracking
    - State management (pause/resume strategies)
    """

    def __init__(
        self,
        risk_manager: RiskManager,
        orchestrator: ExecutionOrchestrator,
        initial_capital: Decimal,
        conflict_resolution: SignalConflictResolution = SignalConflictResolution.HIGHEST_CONFIDENCE,
        max_concurrent_positions: int = 5,
        max_strategy_allocation_pct: float = 0.5,
    ):
        """
        Initialize multi-strategy context.

        Args:
            risk_manager: Risk manager for portfolio-level risk control
            orchestrator: Execution orchestrator for order management
            initial_capital: Initial portfolio capital
            conflict_resolution: Strategy for resolving signal conflicts
            max_concurrent_positions: Maximum concurrent positions across all strategies
            max_strategy_allocation_pct: Maximum allocation per strategy (default 50%)
        """
        self.risk_manager = risk_manager
        self.orchestrator = orchestrator
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.conflict_resolution = conflict_resolution
        self.max_concurrent_positions = max_concurrent_positions
        self.max_strategy_allocation_pct = max_strategy_allocation_pct

        # Strategy management
        self._strategies: dict[str, StrategyInstance] = {}
        self._strategy_lock = asyncio.Lock()

        # Portfolio tracking
        self._total_allocation_pct: float = 0.0
        self._portfolio_positions: dict[str, list[str]] = {}  # symbol -> [strategy_names]
        self._portfolio_pnl = Decimal("0")

        # Performance tracking
        self._total_trades = 0
        self._correlation_matrix: pd.DataFrame | None = None

        logger.info(
            "strategy_context_initialized",
            initial_capital=str(initial_capital),
            conflict_resolution=conflict_resolution.value,
            max_concurrent_positions=max_concurrent_positions,
        )

    def add_strategy(
        self,
        name: str,
        strategy: BaseStrategy,
        symbol: str,
        allocation_pct: float,
        config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add a new strategy instance to the context.

        Args:
            name: Unique name for the strategy instance
            strategy: Strategy object to add
            symbol: Trading symbol for this strategy
            allocation_pct: Percentage of capital allocated to this strategy (0.0-1.0)
            config: Optional per-strategy configuration overrides

        Returns:
            True if strategy added successfully, False otherwise
        """
        # Validation
        if name in self._strategies:
            logger.error("strategy_already_exists", name=name)
            return False

        if allocation_pct <= 0 or allocation_pct > self.max_strategy_allocation_pct:
            logger.error(
                "invalid_allocation_pct",
                name=name,
                allocation_pct=allocation_pct,
                max_allowed=self.max_strategy_allocation_pct,
            )
            return False

        if self._total_allocation_pct + allocation_pct > 1.0:
            logger.error(
                "total_allocation_exceeds_100_percent",
                name=name,
                current_total=self._total_allocation_pct,
                new_allocation=allocation_pct,
            )
            return False

        # Calculate allocated capital
        allocated_capital = self.current_capital * Decimal(str(allocation_pct))

        # Create strategy instance
        instance = StrategyInstance(
            name=name,
            strategy=strategy,
            symbol=symbol,
            allocation_pct=allocation_pct,
            allocated_capital=allocated_capital,
            config=config or {},
        )

        # Add to tracking
        self._strategies[name] = instance
        self._total_allocation_pct += allocation_pct

        logger.info(
            "strategy_added",
            name=name,
            strategy_type=strategy.__class__.__name__,
            symbol=symbol,
            allocation_pct=f"{allocation_pct:.2%}",
            allocated_capital=str(allocated_capital),
            total_allocation=f"{self._total_allocation_pct:.2%}",
        )

        return True

    def remove_strategy(self, name: str, close_position: bool = True) -> bool:
        """
        Remove a strategy instance from the context.

        Args:
            name: Name of the strategy to remove
            close_position: Whether to close any open positions (default True)

        Returns:
            True if strategy removed successfully, False otherwise
        """
        if name not in self._strategies:
            logger.error("strategy_not_found", name=name)
            return False

        instance = self._strategies[name]

        # Check for open position
        if instance.current_position and close_position:
            logger.warning(
                "removing_strategy_with_open_position",
                name=name,
                symbol=instance.symbol,
                position_side=instance.current_position.side.value,
            )
            # Note: Actual position closing would be handled by orchestrator
            # This is just a warning - implementation would need async close

        # Update total allocation
        self._total_allocation_pct -= instance.allocation_pct

        # Remove from tracking
        del self._strategies[name]

        logger.info(
            "strategy_removed",
            name=name,
            total_allocation=f"{self._total_allocation_pct:.2%}",
        )

        return True

    def get_strategy(self, name: str) -> StrategyInstance | None:
        """
        Get a strategy instance by name.

        Args:
            name: Strategy name

        Returns:
            StrategyInstance or None if not found
        """
        return self._strategies.get(name)

    def list_strategies(self) -> list[str]:
        """
        Get list of all strategy names.

        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())

    def pause_strategy(self, name: str) -> bool:
        """
        Pause a strategy (no new positions, existing positions remain).

        Args:
            name: Strategy name

        Returns:
            True if paused successfully, False otherwise
        """
        instance = self._strategies.get(name)
        if not instance:
            return False

        instance.state = StrategyState.PAUSED
        logger.info("strategy_paused", name=name)
        return True

    def resume_strategy(self, name: str) -> bool:
        """
        Resume a paused strategy.

        Args:
            name: Strategy name

        Returns:
            True if resumed successfully, False otherwise
        """
        instance = self._strategies.get(name)
        if not instance:
            return False

        if instance.state == StrategyState.ERROR:
            logger.warning("cannot_resume_strategy_in_error_state", name=name)
            return False

        instance.state = StrategyState.ACTIVE
        logger.info("strategy_resumed", name=name)
        return True

    async def process_strategy_signal(
        self, name: str, market_data: pd.DataFrame
    ) -> StrategySignal | None:
        """
        Process market data for a specific strategy and generate signal.

        Args:
            name: Strategy name
            market_data: OHLCV market data for the strategy's symbol

        Returns:
            StrategySignal or None if strategy not active or error occurred
        """
        async with self._strategy_lock:
            instance = self._strategies.get(name)
            if not instance:
                logger.error("strategy_not_found", name=name)
                return None

            # Check if strategy is active
            if instance.state != StrategyState.ACTIVE:
                logger.debug(
                    "strategy_not_active",
                    name=name,
                    state=instance.state.value,
                )
                return None

            try:
                # Generate signal
                signal = instance.strategy.generate_signal(market_data)
                instance.last_signal_at = datetime.now(UTC)

                logger.info(
                    "strategy_signal_generated",
                    strategy=name,
                    symbol=instance.symbol,
                    signal=signal.signal.value,
                    confidence=signal.confidence,
                    reason=signal.reason,
                )

                return signal

            except Exception as e:
                instance.record_error(str(e))
                logger.error(
                    "strategy_signal_generation_failed",
                    strategy=name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return None

    async def execute_strategy_signal(
        self, name: str, signal: StrategySignal, market_data: pd.DataFrame
    ) -> bool:
        """
        Execute a strategy signal (enter or exit position).

        Args:
            name: Strategy name
            signal: Trading signal from the strategy
            market_data: Current market data

        Returns:
            True if execution successful, False otherwise
        """
        async with self._strategy_lock:
            instance = self._strategies.get(name)
            if not instance:
                return False

            with CorrelationContext() as correlation_id:
                try:
                    # Check for exit conditions first
                    if instance.current_position:
                        should_exit, exit_reason = instance.strategy.should_exit(
                            signal.price, instance.current_position
                        )

                        if should_exit:
                            # Exit position
                            success = await self._exit_position(
                                instance,
                                signal.price,
                                signal.timestamp,
                                exit_reason,
                                correlation_id,
                            )
                            return success

                    # Check for entry signal
                    if signal.signal in [Signal.BUY, Signal.SELL] and not instance.current_position:
                        # Calculate position size
                        position_size = instance.strategy.get_position_size(
                            signal, float(instance.allocated_capital)
                        )

                        # Enter position
                        success = await self._enter_position(
                            instance, signal, position_size, correlation_id
                        )
                        return success

                    return True  # HOLD signal or no action needed

                except Exception as e:
                    instance.record_error(str(e))
                    logger.error(
                        "strategy_signal_execution_failed",
                        strategy=name,
                        error=str(e),
                        error_type=type(e).__name__,
                        correlation_id=correlation_id,
                    )
                    return False

    async def _enter_position(
        self,
        instance: StrategyInstance,
        signal: StrategySignal,
        size: float,
        correlation_id: str,
    ) -> bool:
        """
        Enter a new position for a strategy.

        Args:
            instance: Strategy instance
            signal: Trading signal
            size: Position size
            correlation_id: Correlation ID for tracking

        Returns:
            True if position entered successfully, False otherwise
        """
        # Check portfolio-level constraints
        current_positions = sum(
            1 for inst in self._strategies.values() if inst.current_position is not None
        )

        if current_positions >= self.max_concurrent_positions:
            logger.warning(
                "max_concurrent_positions_reached",
                strategy=instance.name,
                current_positions=current_positions,
                max_positions=self.max_concurrent_positions,
            )
            return False

        # Execute order via orchestrator
        side = OrderSide.BUY if signal.signal == Signal.BUY else OrderSide.SELL

        result = await self.orchestrator.execute_market_order(
            symbol=instance.symbol,
            side=side,
            quantity=Decimal(str(size)),
            correlation_id=correlation_id,
            metadata={
                "strategy": instance.name,
                "signal_confidence": signal.confidence,
                "signal_reason": signal.reason,
            },
        )

        if not result.success:
            logger.error(
                "position_entry_failed",
                strategy=instance.name,
                symbol=instance.symbol,
                error=result.error_message,
            )
            return False

        # Update strategy position
        position = instance.strategy.enter_position(signal, size)
        instance.current_position = position
        instance.position_value = Decimal(str(size * signal.price))
        instance.last_trade_at = datetime.now(UTC)

        # Track in portfolio
        if instance.symbol not in self._portfolio_positions:
            self._portfolio_positions[instance.symbol] = []
        self._portfolio_positions[instance.symbol].append(instance.name)

        logger.info(
            "position_entered",
            strategy=instance.name,
            symbol=instance.symbol,
            side=position.side.value,
            entry_price=position.entry_price,
            size=size,
            correlation_id=correlation_id,
        )

        return True

    async def _exit_position(
        self,
        instance: StrategyInstance,
        exit_price: float,
        exit_timestamp: int,
        reason: str,
        correlation_id: str,
    ) -> bool:
        """
        Exit an existing position for a strategy.

        Args:
            instance: Strategy instance
            exit_price: Exit price
            exit_timestamp: Exit timestamp
            reason: Reason for exit
            correlation_id: Correlation ID for tracking

        Returns:
            True if position exited successfully, False otherwise
        """
        if not instance.current_position:
            return False

        position = instance.current_position

        # Execute order via orchestrator
        side = OrderSide.SELL if position.side == PositionSide.LONG else OrderSide.BUY

        result = await self.orchestrator.execute_market_order(
            symbol=instance.symbol,
            side=side,
            quantity=Decimal(str(position.size)),
            correlation_id=correlation_id,
            metadata={"strategy": instance.name, "exit_reason": reason},
        )

        if not result.success:
            logger.error(
                "position_exit_failed",
                strategy=instance.name,
                symbol=instance.symbol,
                error=result.error_message,
            )
            return False

        # Calculate P&L and update performance
        exit_details = instance.strategy.exit_position(exit_price, exit_timestamp, reason)

        pnl = Decimal(str(exit_details.get("pnl", 0)))
        holding_period = exit_details.get("holding_period_ms", 0)

        instance.performance.update_from_trade(pnl, holding_period)
        instance.trade_history.append(exit_details)
        instance.current_position = None
        instance.position_value = Decimal("0")

        # Update portfolio tracking
        self._portfolio_pnl += pnl
        self._total_trades += 1

        # Remove from portfolio positions
        if instance.symbol in self._portfolio_positions:
            if instance.name in self._portfolio_positions[instance.symbol]:
                self._portfolio_positions[instance.symbol].remove(instance.name)
            if not self._portfolio_positions[instance.symbol]:
                del self._portfolio_positions[instance.symbol]

        logger.info(
            "position_exited",
            strategy=instance.name,
            symbol=instance.symbol,
            exit_price=exit_price,
            pnl=str(pnl),
            pnl_pct=f"{exit_details.get('pnl_pct', 0):.2%}",
            reason=reason,
            correlation_id=correlation_id,
        )

        return True

    async def process_all_strategies(
        self, market_data_by_symbol: dict[str, pd.DataFrame], resolve_conflicts: bool = True
    ) -> dict[str, StrategySignal]:
        """
        Process market data for all active strategies and optionally resolve conflicts.

        Args:
            market_data_by_symbol: Dictionary mapping symbols to OHLCV data
            resolve_conflicts: Whether to apply conflict resolution (default: True)

        Returns:
            Dictionary mapping strategy names to their generated signals.
            If resolve_conflicts is True, signals are filtered by conflict resolution.
            If resolve_conflicts is False, all signals are returned (may have conflicts).
        """
        signals: dict[str, StrategySignal] = {}

        for name, instance in self._strategies.items():
            if instance.state != StrategyState.ACTIVE:
                continue

            market_data = market_data_by_symbol.get(instance.symbol)
            if market_data is None:
                logger.warning(
                    "no_market_data_for_strategy",
                    strategy=name,
                    symbol=instance.symbol,
                )
                continue

            signal = await self.process_strategy_signal(name, market_data)
            if signal:
                signals[name] = signal

        # Apply conflict resolution if requested
        if resolve_conflicts and len(signals) > 1:
            signals = self.resolve_signal_conflicts(signals)

        return signals

    def resolve_signal_conflicts(
        self, signals: dict[str, StrategySignal]
    ) -> dict[str, StrategySignal]:
        """
        Resolve conflicts when multiple strategies generate signals for the same symbol.

        Args:
            signals: Dictionary mapping strategy names to their signals

        Returns:
            Dictionary with resolved signals (one per symbol)
        """
        if not signals:
            return {}

        # Group signals by symbol
        signals_by_symbol: dict[str, list[tuple[str, StrategySignal]]] = {}
        for strategy_name, signal in signals.items():
            # Get symbol from strategy instance
            strategy_instance = self._strategies.get(strategy_name)
            if strategy_instance is None:
                continue
            symbol = strategy_instance.symbol
            if symbol not in signals_by_symbol:
                signals_by_symbol[symbol] = []
            signals_by_symbol[symbol].append((strategy_name, signal))

        resolved_signals: dict[str, StrategySignal] = {}

        # Resolve conflicts for each symbol
        for symbol, symbol_signals in signals_by_symbol.items():
            # If only one signal for this symbol, no conflict
            if len(symbol_signals) == 1:
                strategy_name, signal = symbol_signals[0]
                resolved_signals[strategy_name] = signal
                continue

            # Multiple signals for same symbol - apply conflict resolution
            logger.info(
                "resolving_signal_conflict",
                symbol=symbol,
                num_signals=len(symbol_signals),
                resolution_mode=self.conflict_resolution.value,
            )

            resolved_signal = None

            if self.conflict_resolution == SignalConflictResolution.FIRST_WINS:
                # First strategy wins
                strategy_name, resolved_signal = symbol_signals[0]
                logger.debug(
                    "conflict_resolved_first_wins",
                    winner=strategy_name,
                    signal=resolved_signal.signal.value,
                )

            elif self.conflict_resolution == SignalConflictResolution.HIGHEST_CONFIDENCE:
                # Highest confidence wins
                winner = max(symbol_signals, key=lambda x: x[1].confidence)
                strategy_name, resolved_signal = winner
                logger.debug(
                    "conflict_resolved_highest_confidence",
                    winner=strategy_name,
                    confidence=resolved_signal.confidence,
                    signal=resolved_signal.signal.value,
                )

            elif self.conflict_resolution == SignalConflictResolution.WEIGHTED_AVERAGE:
                # Weighted average by strategy allocation
                # For now, use highest confidence as fallback
                # TODO: Implement proper weighted averaging for position size
                winner = max(symbol_signals, key=lambda x: x[1].confidence)
                strategy_name, resolved_signal = winner
                logger.debug(
                    "conflict_resolved_weighted_average",
                    winner=strategy_name,
                    note="using_highest_confidence_fallback",
                )

            elif self.conflict_resolution == SignalConflictResolution.VETO:
                # Any HOLD signal vetoes execution
                from bot.core.strategy import Signal

                has_hold = any(sig[1].signal == Signal.HOLD for sig in symbol_signals)
                if has_hold:
                    logger.debug("conflict_resolved_veto", result="vetoed_by_hold")
                    # Don't add any signal for this symbol
                    continue
                else:
                    # Use highest confidence among non-HOLD signals
                    winner = max(symbol_signals, key=lambda x: x[1].confidence)
                    strategy_name, resolved_signal = winner
                    logger.debug(
                        "conflict_resolved_veto",
                        winner=strategy_name,
                        result="no_veto",
                    )

            elif self.conflict_resolution == SignalConflictResolution.UNANIMOUS:
                # All strategies must agree
                from bot.core.strategy import Signal

                all_signals = [sig[1].signal for sig in symbol_signals]
                if len(set(all_signals)) == 1:
                    # All agree - use highest confidence
                    winner = max(symbol_signals, key=lambda x: x[1].confidence)
                    strategy_name, resolved_signal = winner
                    logger.debug(
                        "conflict_resolved_unanimous",
                        winner=strategy_name,
                        result="unanimous_agreement",
                    )
                else:
                    logger.debug(
                        "conflict_resolved_unanimous",
                        result="no_unanimous_agreement",
                        signals=[s.value for s in all_signals],
                    )
                    # No agreement - don't execute
                    continue

            if resolved_signal:
                resolved_signals[strategy_name] = resolved_signal

        return resolved_signals

    def get_portfolio_metrics(self) -> dict[str, Any]:
        """
        Get comprehensive portfolio metrics across all strategies.

        Returns:
            Dictionary containing portfolio-level metrics
        """
        total_position_value = sum(inst.position_value for inst in self._strategies.values())

        active_positions = sum(
            1 for inst in self._strategies.values() if inst.current_position is not None
        )

        total_strategy_pnl = sum(inst.performance.total_pnl for inst in self._strategies.values())

        strategy_metrics = {name: inst.to_dict() for name, inst in self._strategies.items()}

        risk_metrics = self.risk_manager.get_risk_metrics()

        return {
            "portfolio": {
                "initial_capital": str(self.initial_capital),
                "current_capital": str(self.current_capital),
                "total_pnl": str(self._portfolio_pnl),
                "total_pnl_pct": f"{float(self._portfolio_pnl / self.initial_capital):.2%}",
                "total_position_value": str(total_position_value),
                "active_positions": active_positions,
                "max_concurrent_positions": self.max_concurrent_positions,
            },
            "strategies": {
                "total_strategies": len(self._strategies),
                "active_strategies": sum(
                    1 for inst in self._strategies.values() if inst.state == StrategyState.ACTIVE
                ),
                "paused_strategies": sum(
                    1 for inst in self._strategies.values() if inst.state == StrategyState.PAUSED
                ),
                "total_allocation_pct": f"{self._total_allocation_pct:.2%}",
                "details": strategy_metrics,
            },
            "performance": {
                "total_trades": self._total_trades,
                "total_strategy_pnl": str(total_strategy_pnl),
            },
            "risk": risk_metrics,
        }

    def reset_all_strategies(self) -> None:
        """Reset all strategy states (useful for backtesting)."""
        for instance in self._strategies.values():
            instance.strategy.reset()
            instance.current_position = None
            instance.position_value = Decimal("0")
            instance.performance = StrategyPerformance()
            instance.trade_history = []

        self._portfolio_positions = {}
        self._portfolio_pnl = Decimal("0")
        self._total_trades = 0

        logger.info("all_strategies_reset")
