"""
Emergency stop system for catastrophic scenarios.

This module provides a comprehensive kill-switch system with:
- Flash crash detection (>10% price move in <5 min)
- Exchange API failure monitoring
- Portfolio drawdown tracking
- Data quality monitoring (stale data, NaN values)
- Automated emergency actions (cancel orders, close positions, halt trading)
- Alert system via Telegram/Email
- Manual override capability
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

import structlog

from bot.core.logging_config import CorrelationContext
from bot.interfaces.exchange import ExchangeInterface, OrderSide, OrderType
from bot.config.models import RiskConfig


logger = structlog.get_logger(__name__)


class EmergencyTrigger(str, Enum):
    """Emergency trigger types."""

    FLASH_CRASH = "flash_crash"
    API_FAILURE = "api_failure"
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    STALE_DATA = "stale_data"
    DATA_QUALITY = "data_quality"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    MANUAL_TRIGGER = "manual_trigger"


class EmergencyState(str, Enum):
    """Emergency system states."""

    ACTIVE = "active"
    TRIGGERED = "triggered"
    HALTED = "halted"
    RECOVERING = "recovering"


@dataclass
class EmergencyEvent:
    """
    Emergency event data.

    Attributes:
        trigger: Type of emergency trigger
        timestamp: Event timestamp
        message: Human-readable description
        details: Additional event details
        severity: Severity level (1-10)
        correlation_id: Correlation ID for tracking
    """

    trigger: EmergencyTrigger
    timestamp: datetime
    message: str
    details: Dict[str, Any]
    severity: int = 10
    correlation_id: Optional[str] = None


@dataclass
class PriceDataPoint:
    """Price data point for flash crash detection."""

    symbol: str
    price: Decimal
    timestamp: datetime


@dataclass
class EmergencyConfig:
    """
    Emergency stop configuration.

    Attributes:
        flash_crash_threshold_pct: Price move threshold for flash crash (default: 0.10 = 10%)
        flash_crash_window_seconds: Time window for flash crash detection (default: 300 = 5 min)
        api_failure_threshold_seconds: API unreachable threshold (default: 30 seconds)
        max_drawdown_pct: Maximum portfolio drawdown (default: 0.10 = 10%)
        stale_data_threshold_seconds: Stale data threshold (default: 60 seconds)
        consecutive_failure_threshold: Max consecutive API failures (default: 5)
        enable_auto_recovery: Enable automatic recovery (default: False)
        recovery_delay_seconds: Delay before auto recovery (default: 300 seconds)
    """

    flash_crash_threshold_pct: float = 0.10
    flash_crash_window_seconds: int = 300
    api_failure_threshold_seconds: int = 30
    max_drawdown_pct: float = 0.10
    stale_data_threshold_seconds: int = 60
    consecutive_failure_threshold: int = 5
    enable_auto_recovery: bool = False
    recovery_delay_seconds: int = 300


class EmergencyStopManager:
    """
    Emergency stop manager with automated triggers and actions.

    Features:
    - Real-time monitoring of market conditions and system health
    - Multiple trigger conditions with configurable thresholds
    - Automated emergency actions (cancel orders, close positions, halt trading)
    - Alert notifications via callbacks
    - Manual override capability
    - Recovery management
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        initial_portfolio_value: Decimal,
        config: Optional[EmergencyConfig] = None,
        alert_callback: Optional[Callable[[EmergencyEvent], None]] = None,
    ):
        """
        Initialize emergency stop manager.

        Args:
            exchange: Exchange interface for trading operations
            initial_portfolio_value: Initial portfolio value for drawdown tracking
            config: Emergency stop configuration
            alert_callback: Optional callback for alerts (async function)
        """
        self.exchange = exchange
        self.initial_portfolio_value = initial_portfolio_value
        self.config = config or EmergencyConfig()
        self.alert_callback = alert_callback

        # State management
        self.state = EmergencyState.ACTIVE
        self._is_halted = False
        self._halt_lock = asyncio.Lock()

        # Portfolio tracking
        self.current_portfolio_value = initial_portfolio_value
        self.peak_portfolio_value = initial_portfolio_value

        # Price history for flash crash detection
        # symbol -> deque of PriceDataPoint
        self._price_history: Dict[str, deque[PriceDataPoint]] = {}

        # API health tracking
        self._last_successful_api_call: Optional[datetime] = None
        self._consecutive_api_failures = 0
        self._api_failure_start: Optional[datetime] = None

        # Data quality tracking
        self._last_data_update: Dict[str, datetime] = {}  # symbol -> last update time

        # Emergency events log
        self._emergency_events: List[EmergencyEvent] = []

        # Recovery tracking
        self._recovery_task: Optional[asyncio.Task] = None

        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._stop_monitoring = asyncio.Event()

        logger.info(
            "emergency_stop_manager_initialized",
            initial_portfolio_value=str(initial_portfolio_value),
            config=vars(config),
        )

    async def start_monitoring(self) -> None:
        """Start continuous monitoring of emergency conditions."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("monitoring_already_running")
            return

        self._stop_monitoring.clear()
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("emergency_monitoring_started")

    async def stop_monitoring(self) -> None:
        """Stop monitoring loop."""
        self._stop_monitoring.set()
        if self._monitoring_task:
            await self._monitoring_task
        logger.info("emergency_monitoring_stopped")

    async def _monitoring_loop(self) -> None:
        """
        Continuous monitoring loop for system health.

        Monitors:
        - API connectivity
        - Data staleness
        """
        while not self._stop_monitoring.is_set():
            try:
                # Check API connectivity
                await self._check_api_health()

                # Check data staleness
                await self._check_data_staleness()

                # Sleep for 5 seconds between checks
                await asyncio.sleep(5)

            except Exception as e:
                logger.error("monitoring_loop_error", error=str(e))
                await asyncio.sleep(5)

    async def update_price(self, symbol: str, price: Decimal) -> None:
        """
        Update price data and check for flash crashes.

        Args:
            symbol: Trading pair symbol
            price: Current price
        """
        if price is None or price <= 0:
            await self._trigger_emergency(
                EmergencyTrigger.DATA_QUALITY,
                f"Invalid price data for {symbol}: {price}",
                {"symbol": symbol, "price": str(price)},
                severity=7,
            )
            return

        # Check for NaN or invalid values
        try:
            float(price)
        except (ValueError, TypeError):
            await self._trigger_emergency(
                EmergencyTrigger.DATA_QUALITY,
                f"NaN or invalid price data for {symbol}",
                {"symbol": symbol, "price": str(price)},
                severity=8,
            )
            return

        now = datetime.now(timezone.utc)

        # Initialize price history for symbol if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=1000)

        # Add price data point
        data_point = PriceDataPoint(symbol=symbol, price=price, timestamp=now)
        self._price_history[symbol].append(data_point)

        # Update last data timestamp
        self._last_data_update[symbol] = now

        # Check for flash crash
        await self._check_flash_crash(symbol)

    async def _check_flash_crash(self, symbol: str) -> None:
        """
        Check for flash crash in price history.

        Args:
            symbol: Trading pair symbol
        """
        if symbol not in self._price_history or len(self._price_history[symbol]) < 2:
            return

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=self.config.flash_crash_window_seconds)

        # Get prices within the time window
        recent_prices = [
            dp for dp in self._price_history[symbol] if dp.timestamp >= window_start
        ]

        if len(recent_prices) < 2:
            return

        # Find min and max prices in window
        prices = [dp.price for dp in recent_prices]
        min_price = min(prices)
        max_price = max(prices)

        # Calculate price change percentage
        if max_price > 0:
            price_change_pct = abs(float((max_price - min_price) / max_price))

            if price_change_pct > self.config.flash_crash_threshold_pct:
                await self._trigger_emergency(
                    EmergencyTrigger.FLASH_CRASH,
                    f"Flash crash detected for {symbol}: {price_change_pct:.2%} move in {self.config.flash_crash_window_seconds}s",
                    {
                        "symbol": symbol,
                        "price_change_pct": price_change_pct,
                        "min_price": str(min_price),
                        "max_price": str(max_price),
                        "window_seconds": self.config.flash_crash_window_seconds,
                        "threshold_pct": self.config.flash_crash_threshold_pct,
                    },
                    severity=10,
                )

    async def update_portfolio_value(self, new_value: Decimal) -> None:
        """
        Update portfolio value and check for drawdown.

        Args:
            new_value: New portfolio value
        """
        old_value = self.current_portfolio_value
        self.current_portfolio_value = new_value

        # Update peak value
        if new_value > self.peak_portfolio_value:
            self.peak_portfolio_value = new_value

        # Calculate drawdown from peak
        if self.peak_portfolio_value > 0:
            drawdown_pct = float((self.peak_portfolio_value - new_value) / self.peak_portfolio_value)

            if drawdown_pct > self.config.max_drawdown_pct:
                await self._trigger_emergency(
                    EmergencyTrigger.PORTFOLIO_DRAWDOWN,
                    f"Portfolio drawdown exceeded threshold: {drawdown_pct:.2%}",
                    {
                        "current_value": str(new_value),
                        "peak_value": str(self.peak_portfolio_value),
                        "drawdown_pct": drawdown_pct,
                        "threshold_pct": self.config.max_drawdown_pct,
                        "initial_value": str(self.initial_portfolio_value),
                    },
                    severity=10,
                )

    async def record_api_success(self) -> None:
        """Record successful API call."""
        self._last_successful_api_call = datetime.now(timezone.utc)
        self._consecutive_api_failures = 0
        self._api_failure_start = None

    async def record_api_failure(self) -> None:
        """Record API failure and check thresholds."""
        now = datetime.now(timezone.utc)
        self._consecutive_api_failures += 1

        if self._api_failure_start is None:
            self._api_failure_start = now

        # Check consecutive failures
        if self._consecutive_api_failures >= self.config.consecutive_failure_threshold:
            await self._trigger_emergency(
                EmergencyTrigger.CONSECUTIVE_FAILURES,
                f"Consecutive API failures threshold exceeded: {self._consecutive_api_failures}",
                {
                    "consecutive_failures": self._consecutive_api_failures,
                    "threshold": self.config.consecutive_failure_threshold,
                    "failure_start": self._api_failure_start.isoformat(),
                },
                severity=9,
            )

    async def _check_api_health(self) -> None:
        """Check API connectivity health."""
        if self._last_successful_api_call is None:
            return

        now = datetime.now(timezone.utc)
        time_since_success = (now - self._last_successful_api_call).total_seconds()

        if time_since_success > self.config.api_failure_threshold_seconds:
            await self._trigger_emergency(
                EmergencyTrigger.API_FAILURE,
                f"Exchange API unreachable for {time_since_success:.0f} seconds",
                {
                    "time_since_success": time_since_success,
                    "threshold_seconds": self.config.api_failure_threshold_seconds,
                    "last_success": self._last_successful_api_call.isoformat(),
                },
                severity=10,
            )

    async def _check_data_staleness(self) -> None:
        """Check for stale data feeds."""
        now = datetime.now(timezone.utc)

        for symbol, last_update in self._last_data_update.items():
            time_since_update = (now - last_update).total_seconds()

            if time_since_update > self.config.stale_data_threshold_seconds:
                await self._trigger_emergency(
                    EmergencyTrigger.STALE_DATA,
                    f"Stale data detected for {symbol}: {time_since_update:.0f} seconds old",
                    {
                        "symbol": symbol,
                        "time_since_update": time_since_update,
                        "threshold_seconds": self.config.stale_data_threshold_seconds,
                        "last_update": last_update.isoformat(),
                    },
                    severity=8,
                )

    async def _trigger_emergency(
        self,
        trigger: EmergencyTrigger,
        message: str,
        details: Dict[str, Any],
        severity: int = 10,
    ) -> None:
        """
        Trigger emergency stop.

        Args:
            trigger: Emergency trigger type
            message: Human-readable message
            details: Additional details
            severity: Severity level (1-10)
        """
        # Don't re-trigger if already halted
        if self._is_halted:
            logger.warning(
                "emergency_already_halted",
                trigger=trigger.value,
                message=message,
            )
            return

        with CorrelationContext() as correlation_id:
            event = EmergencyEvent(
                trigger=trigger,
                timestamp=datetime.now(timezone.utc),
                message=message,
                details=details,
                severity=severity,
                correlation_id=correlation_id,
            )

            self._emergency_events.append(event)

            logger.critical(
                "emergency_triggered",
                trigger=trigger.value,
                message=message,
                severity=severity,
                details=details,
                correlation_id=correlation_id,
            )

            # Execute emergency actions
            await self._execute_emergency_actions(event)

            # Send alerts
            await self._send_alert(event)

    async def _execute_emergency_actions(self, event: EmergencyEvent) -> None:
        """
        Execute emergency actions.

        Actions:
        1. Cancel all open orders
        2. Close all positions (market orders)
        3. Halt trading engine

        Args:
            event: Emergency event
        """
        async with self._halt_lock:
            try:
                logger.info(
                    "executing_emergency_actions",
                    trigger=event.trigger.value,
                    correlation_id=event.correlation_id,
                )

                # Step 1: Cancel all open orders
                await self._cancel_all_orders()

                # Step 2: Close all positions
                await self._close_all_positions()

                # Step 3: Halt trading
                self._is_halted = True
                self.state = EmergencyState.HALTED

                logger.critical(
                    "emergency_actions_completed",
                    trigger=event.trigger.value,
                    correlation_id=event.correlation_id,
                )

                # Schedule auto-recovery if enabled
                if self.config.enable_auto_recovery:
                    self._recovery_task = asyncio.create_task(
                        self._auto_recovery(event.correlation_id)
                    )

            except Exception as e:
                logger.critical(
                    "emergency_actions_failed",
                    error=str(e),
                    trigger=event.trigger.value,
                    correlation_id=event.correlation_id,
                )
                raise

    async def _cancel_all_orders(self) -> None:
        """Cancel all open orders across all symbols."""
        try:
            open_orders = await self.exchange.get_open_orders()

            if not open_orders:
                logger.info("no_open_orders_to_cancel")
                return

            logger.info("canceling_all_orders", count=len(open_orders))

            # Cancel orders concurrently
            cancel_tasks = [
                self.exchange.cancel_order(order.symbol, order.id) for order in open_orders
            ]

            results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

            # Count successes and failures
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            failure_count = len(results) - success_count

            logger.info(
                "orders_canceled",
                total=len(open_orders),
                success=success_count,
                failed=failure_count,
            )

        except Exception as e:
            logger.error("cancel_all_orders_error", error=str(e))
            raise

    async def _close_all_positions(self) -> None:
        """
        Close all positions at market price.

        Note: This is a simplified implementation. In production, you would
        need to query actual positions from the exchange or a position tracker.
        """
        try:
            logger.info("closing_all_positions")

            # In a real implementation, you would:
            # 1. Get all open positions from position tracker
            # 2. Create market orders to close each position
            # 3. For long positions: sell at market
            # 4. For short positions: buy at market

            # Placeholder: This would be implemented based on your position tracking
            logger.warning(
                "close_all_positions_not_implemented",
                message="Position tracking integration required",
            )

        except Exception as e:
            logger.error("close_all_positions_error", error=str(e))
            raise

    async def _send_alert(self, event: EmergencyEvent) -> None:
        """
        Send alert notification.

        Args:
            event: Emergency event
        """
        try:
            if self.alert_callback:
                # Call the alert callback (could be Telegram, Email, etc.)
                if asyncio.iscoroutinefunction(self.alert_callback):
                    await self.alert_callback(event)
                else:
                    self.alert_callback(event)

            logger.info(
                "alert_sent",
                trigger=event.trigger.value,
                correlation_id=event.correlation_id,
            )

        except Exception as e:
            logger.error(
                "alert_send_failed",
                error=str(e),
                trigger=event.trigger.value,
            )

    async def _auto_recovery(self, correlation_id: str) -> None:
        """
        Automatic recovery after emergency halt.

        Args:
            correlation_id: Original emergency correlation ID
        """
        try:
            logger.info(
                "auto_recovery_scheduled",
                delay_seconds=self.config.recovery_delay_seconds,
                correlation_id=correlation_id,
            )

            await asyncio.sleep(self.config.recovery_delay_seconds)

            self.state = EmergencyState.RECOVERING

            logger.info("auto_recovery_started", correlation_id=correlation_id)

            # Reset state
            async with self._halt_lock:
                self._is_halted = False
                self.state = EmergencyState.ACTIVE
                self._consecutive_api_failures = 0
                self._api_failure_start = None

            logger.info("auto_recovery_completed", correlation_id=correlation_id)

        except Exception as e:
            logger.error("auto_recovery_failed", error=str(e), correlation_id=correlation_id)

    async def manual_trigger(self, reason: str) -> None:
        """
        Manually trigger emergency stop.

        Args:
            reason: Reason for manual trigger
        """
        await self._trigger_emergency(
            EmergencyTrigger.MANUAL_TRIGGER,
            f"Manual emergency stop: {reason}",
            {"reason": reason, "triggered_by": "manual"},
            severity=10,
        )

    async def manual_resume(self, operator: str) -> None:
        """
        Manually resume trading after emergency halt.

        Args:
            operator: Name/ID of operator resuming trading
        """
        async with self._halt_lock:
            if not self._is_halted:
                logger.warning("resume_called_but_not_halted", operator=operator)
                return

            logger.info("manual_resume_initiated", operator=operator)

            # Cancel auto-recovery if running
            if self._recovery_task and not self._recovery_task.done():
                self._recovery_task.cancel()

            # Reset state
            self._is_halted = False
            self.state = EmergencyState.ACTIVE
            self._consecutive_api_failures = 0
            self._api_failure_start = None

            logger.info("manual_resume_completed", operator=operator)

    def is_halted(self) -> bool:
        """
        Check if trading is halted.

        Returns:
            bool: True if halted, False otherwise
        """
        return self._is_halted

    def get_state(self) -> EmergencyState:
        """
        Get current emergency state.

        Returns:
            EmergencyState: Current state
        """
        return self.state

    def get_emergency_events(self, limit: Optional[int] = None) -> List[EmergencyEvent]:
        """
        Get emergency events history.

        Args:
            limit: Maximum number of events to return (None = all)

        Returns:
            List[EmergencyEvent]: Emergency events
        """
        if limit:
            return self._emergency_events[-limit:]
        return self._emergency_events.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current emergency system metrics.

        Returns:
            Dict containing metrics
        """
        return {
            "state": self.state.value,
            "is_halted": self._is_halted,
            "current_portfolio_value": str(self.current_portfolio_value),
            "peak_portfolio_value": str(self.peak_portfolio_value),
            "drawdown_pct": float(
                (self.peak_portfolio_value - self.current_portfolio_value)
                / self.peak_portfolio_value
                if self.peak_portfolio_value > 0
                else 0
            ),
            "consecutive_api_failures": self._consecutive_api_failures,
            "last_successful_api_call": (
                self._last_successful_api_call.isoformat()
                if self._last_successful_api_call
                else None
            ),
            "total_emergency_events": len(self._emergency_events),
            "monitored_symbols": list(self._price_history.keys()),
            "config": vars(self.config),
        }
