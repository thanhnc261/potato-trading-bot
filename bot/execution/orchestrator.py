"""
Execution Orchestrator for Trading Bot.

This module provides the core order execution system that routes trading decisions
to the exchange and manages the complete order lifecycle.

Order Flow:
1. Strategy generates signal
2. RiskManager validates trade
3. Orchestrator creates order
4. Exchange adapter executes
5. Orchestrator tracks status
6. Update portfolio state

Features:
- Async order execution (non-blocking)
- Order deduplication (prevents double orders)
- Comprehensive order lifecycle tracking
- Integration with RiskManager (pre-trade checks)
- Integration with EmergencyStop
- Order status monitoring and updates
- Comprehensive logging with correlation IDs
"""

import asyncio
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from bot.core.logging_config import CorrelationContext, get_logger, log_order
from bot.interfaces.exchange import (
    ExchangeInterface,
    InsufficientBalanceError,
    InvalidOrderError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot.risk.emergency_stop import EmergencyStopManager
from bot.risk.risk_manager import RiskManager

logger = get_logger(__name__)


class OrderRequestStatus(Enum):
    """Status of an order request in the orchestrator."""

    PENDING = "pending"  # Request created, awaiting validation
    VALIDATING = "validating"  # Risk validation in progress
    VALIDATED = "validated"  # Passed risk checks
    REJECTED = "rejected"  # Failed risk checks or validation
    SUBMITTING = "submitting"  # Sending to exchange
    SUBMITTED = "submitted"  # Successfully submitted to exchange
    FILLED = "filled"  # Order completely filled
    PARTIALLY_FILLED = "partially_filled"  # Order partially filled
    CANCELLED = "cancelled"  # Order cancelled
    FAILED = "failed"  # Order submission failed
    EXPIRED = "expired"  # Order expired


@dataclass
class OrderRequest:
    """
    Represents an order request before execution.

    This is the internal representation used by the orchestrator
    to track order requests through the lifecycle.
    """

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.GTC
    client_order_id: str | None = None
    correlation_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Lifecycle tracking
    status: OrderRequestStatus = OrderRequestStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Execution tracking
    exchange_order_id: str | None = None
    exchange_order: Order | None = None
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None

    # Rejection/failure tracking
    rejection_reason: str | None = None
    failure_reason: str | None = None

    def update_status(self, status: OrderRequestStatus, reason: str | None = None) -> None:
        """Update the status of the order request."""
        self.status = status
        self.updated_at = datetime.now()

        if status == OrderRequestStatus.REJECTED:
            self.rejection_reason = reason
        elif status == OrderRequestStatus.FAILED:
            self.failure_reason = reason

    def get_order_hash(self) -> str:
        """
        Generate a unique hash for this order request.

        Used for deduplication - identical orders will produce the same hash.
        Includes symbol, side, type, quantity, and price (if applicable).
        """
        hash_components = [
            self.symbol,
            self.side.value,
            self.order_type.value,
            str(self.quantity),
        ]

        if self.price is not None:
            hash_components.append(str(self.price))

        if self.stop_price is not None:
            hash_components.append(str(self.stop_price))

        hash_string = "|".join(hash_components)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging and serialization."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "time_in_force": self.time_in_force.value,
            "client_order_id": self.client_order_id,
            "correlation_id": self.correlation_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "exchange_order_id": self.exchange_order_id,
            "filled_quantity": str(self.filled_quantity),
            "average_price": str(self.average_price) if self.average_price else None,
            "rejection_reason": self.rejection_reason,
            "failure_reason": self.failure_reason,
            "metadata": self.metadata,
        }


@dataclass
class OrderExecutionResult:
    """
    Result of an order execution attempt.

    Returned by the orchestrator after attempting to execute an order.
    """

    success: bool
    order_request: OrderRequest
    order: Order | None = None
    error_message: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {
            "success": self.success,
            "correlation_id": self.correlation_id,
            "order_request": self.order_request.to_dict(),
            "error_message": self.error_message,
        }

        if self.order:
            result["order"] = {
                "order_id": self.order.id,
                "client_order_id": self.order.client_order_id,
                "symbol": self.order.symbol,
                "status": self.order.status.value,
                "filled_quantity": str(self.order.filled_quantity),
                "average_price": (
                    str(self.order.average_price) if self.order.average_price else None
                ),
            }

        return result


class ExecutionOrchestrator:
    """
    Orchestrates order execution with risk management and emergency stop integration.

    Features:
    - Pre-trade risk validation via RiskManager
    - Emergency stop integration
    - Order deduplication
    - Async order execution
    - Order lifecycle tracking
    - Portfolio state updates
    - Comprehensive logging with correlation IDs

    Example:
        >>> orchestrator = ExecutionOrchestrator(
        ...     exchange=binance_adapter,
        ...     risk_manager=risk_manager,
        ...     emergency_stop=emergency_stop
        ... )
        >>> await orchestrator.start()
        >>>
        >>> # Execute a market buy order
        >>> result = await orchestrator.execute_market_order(
        ...     symbol="BTCUSDT",
        ...     side=OrderSide.BUY,
        ...     quantity=Decimal("0.001")
        ... )
        >>> if result.success:
        ...     print(f"Order executed: {result.order.order_id}")
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        risk_manager: RiskManager,
        emergency_stop: EmergencyStopManager | None = None,
        deduplication_window_seconds: int = 60,
        order_status_check_interval_seconds: int = 5,
        max_status_checks: int = 20,
    ) -> None:
        """
        Initialize the execution orchestrator.

        Args:
            exchange: Exchange adapter for order execution
            risk_manager: Risk manager for pre-trade validation
            emergency_stop: Emergency stop manager (optional)
            deduplication_window_seconds: Time window for order deduplication (default 60s)
            order_status_check_interval_seconds: Interval for checking order status (default 5s)
            max_status_checks: Maximum number of status checks before timeout (default 20)
        """
        self._exchange = exchange
        self._risk_manager = risk_manager
        self._emergency_stop = emergency_stop

        # Configuration
        self._deduplication_window = deduplication_window_seconds
        self._status_check_interval = order_status_check_interval_seconds
        self._max_status_checks = max_status_checks

        # Order tracking
        self._active_orders: dict[str, OrderRequest] = {}  # order_id -> OrderRequest
        self._order_hash_timestamps: dict[str, float] = {}  # order_hash -> timestamp
        self._order_lock = asyncio.Lock()

        # Order callbacks
        self._order_callbacks: list[Callable[[Order], None]] = []

        # State tracking
        self._running = False
        self._monitoring_task: asyncio.Task[None] | None = None

        logger.info(
            "ExecutionOrchestrator initialized",
            exchange=exchange.__class__.__name__,
            deduplication_window_seconds=deduplication_window_seconds,
        )

    async def start(self) -> None:
        """Start the orchestrator and background monitoring tasks."""
        if self._running:
            logger.warning("ExecutionOrchestrator already running")
            return

        self._running = True

        # Start background monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_orders())

        logger.info("ExecutionOrchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator and cleanup."""
        if not self._running:
            return

        self._running = False

        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("ExecutionOrchestrator stopped")

    def add_order_callback(self, callback: Callable[[Order], None]) -> None:
        """
        Add a callback to be invoked when order status changes.

        Args:
            callback: Async function that receives Order objects
        """
        self._order_callbacks.append(callback)
        logger.debug("Added order callback", callback=callback.__name__)

    async def execute_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderExecutionResult:
        """
        Execute a market order.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            side: Order side (BUY or SELL)
            quantity: Order quantity
            correlation_id: Optional correlation ID for tracking
            metadata: Optional metadata for the order

        Returns:
            OrderExecutionResult with execution details
        """
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            time_in_force=TimeInForce.GTC,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        return await self._execute_order(order_request)

    async def execute_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        time_in_force: TimeInForce = TimeInForce.GTC,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderExecutionResult:
        """
        Execute a limit order.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            side: Order side (BUY or SELL)
            quantity: Order quantity
            price: Limit price
            time_in_force: Time in force (default GTC)
            correlation_id: Optional correlation ID for tracking
            metadata: Optional metadata for the order

        Returns:
            OrderExecutionResult with execution details
        """
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        return await self._execute_order(order_request)

    async def execute_stop_loss_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        stop_price: Decimal,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OrderExecutionResult:
        """
        Execute a stop-loss market order.

        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            side: Order side (BUY or SELL)
            quantity: Order quantity
            stop_price: Stop trigger price
            correlation_id: Optional correlation ID for tracking
            metadata: Optional metadata for the order

        Returns:
            OrderExecutionResult with execution details
        """
        order_request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=OrderType.STOP_LOSS,
            quantity=quantity,
            stop_price=stop_price,
            time_in_force=TimeInForce.GTC,
            correlation_id=correlation_id,
            metadata=metadata or {},
        )

        return await self._execute_order(order_request)

    async def cancel_order(
        self, order_id: str, symbol: str, correlation_id: str | None = None
    ) -> bool:
        """
        Cancel an active order.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            correlation_id: Optional correlation ID for tracking

        Returns:
            True if cancellation successful, False otherwise
        """
        with CorrelationContext(correlation_id):
            try:
                logger.info(
                    "Cancelling order",
                    order_id=order_id,
                    symbol=symbol,
                )

                # Cancel on exchange
                cancelled_order = await self._exchange.cancel_order(
                    order_id=order_id, symbol=symbol
                )

                # Update tracking
                async with self._order_lock:
                    if order_id in self._active_orders:
                        order_request = self._active_orders[order_id]
                        order_request.update_status(OrderRequestStatus.CANCELLED)
                        order_request.exchange_order = cancelled_order

                        # Remove from active orders
                        del self._active_orders[order_id]

                logger.info(
                    "Order cancelled successfully",
                    order_id=order_id,
                    symbol=symbol,
                )

                # Log order event
                log_order(
                    order_id=order_id,
                    symbol=symbol,
                    side=cancelled_order.side.value,
                    order_type=cancelled_order.type.value,
                    quantity=float(cancelled_order.quantity),
                    status="cancelled",
                    correlation_id=correlation_id,
                )

                return True

            except Exception as e:
                logger.error(
                    "Failed to cancel order",
                    order_id=order_id,
                    symbol=symbol,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return False

    async def cancel_all_orders(self, correlation_id: str | None = None) -> int:
        """
        Cancel all active orders.

        Args:
            correlation_id: Optional correlation ID for tracking

        Returns:
            Number of orders successfully cancelled
        """
        with CorrelationContext(correlation_id):
            logger.info("Cancelling all active orders")

            async with self._order_lock:
                active_order_ids = list(self._active_orders.keys())

            # Cancel all orders concurrently
            cancel_tasks = [
                self.cancel_order(
                    order_id=order_request.exchange_order_id,
                    symbol=order_request.symbol,
                    correlation_id=correlation_id,
                )
                for order_request in self._active_orders.values()
                if order_request.exchange_order_id
            ]

            results = await asyncio.gather(*cancel_tasks, return_exceptions=True)

            # Count successful cancellations
            successful_cancellations = sum(1 for result in results if result is True)

            logger.info(
                "Cancelled active orders",
                total_orders=len(active_order_ids),
                successful_cancellations=successful_cancellations,
            )

            return successful_cancellations

    async def get_order_status(
        self, order_id: str, symbol: str, correlation_id: str | None = None
    ) -> Order | None:
        """
        Get the current status of an order from the exchange.

        Args:
            order_id: Exchange order ID
            symbol: Trading pair symbol
            correlation_id: Optional correlation ID for tracking

        Returns:
            Order object or None if not found
        """
        with CorrelationContext(correlation_id):
            try:
                order = await self._exchange.get_order(order_id=order_id, symbol=symbol)

                # Update tracking if we're tracking this order
                async with self._order_lock:
                    if order_id in self._active_orders:
                        order_request = self._active_orders[order_id]
                        order_request.exchange_order = order
                        order_request.filled_quantity = order.filled_quantity

                        if order.average_price:
                            order_request.average_price = order.average_price

                        # Update status based on exchange order status
                        self._update_request_status_from_order(order_request, order)

                return order

            except Exception as e:
                logger.error(
                    "Failed to get order status",
                    order_id=order_id,
                    symbol=symbol,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return None

    def get_active_orders(self) -> list[OrderRequest]:
        """
        Get all currently active order requests.

        Returns:
            List of active OrderRequest objects
        """
        return list(self._active_orders.values())

    def get_order_metrics(self) -> dict[str, Any]:
        """
        Get metrics about order execution.

        Returns:
            Dictionary with order execution metrics
        """
        total_orders = len(self._active_orders)
        status_counts: dict[str, int] = {}

        for order_request in self._active_orders.values():
            status = order_request.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_active_orders": total_orders,
            "status_counts": status_counts,
            "deduplication_cache_size": len(self._order_hash_timestamps),
            "running": self._running,
        }

    async def _execute_order(self, order_request: OrderRequest) -> OrderExecutionResult:
        """
        Internal method to execute an order through the full lifecycle.

        Order Lifecycle:
        1. Generate correlation ID if not provided
        2. Check for duplicate orders
        3. Validate with RiskManager
        4. Check emergency stop status
        5. Submit to exchange
        6. Track order status
        7. Update portfolio state

        Args:
            order_request: The order request to execute

        Returns:
            OrderExecutionResult with execution details
        """
        # Generate correlation ID if not provided
        if not order_request.correlation_id:
            with CorrelationContext() as correlation_id:
                order_request.correlation_id = correlation_id
        else:
            correlation_id = order_request.correlation_id

        with CorrelationContext(correlation_id):
            logger.info(
                "Executing order request",
                symbol=order_request.symbol,
                side=order_request.side.value,
                order_type=order_request.order_type.value,
                quantity=str(order_request.quantity),
                price=str(order_request.price) if order_request.price else None,
            )

            # Step 1: Check for duplicate orders
            if await self._is_duplicate_order(order_request):
                order_request.update_status(OrderRequestStatus.REJECTED, "Duplicate order detected")
                logger.warning(
                    "Duplicate order rejected",
                    order_hash=order_request.get_order_hash(),
                )
                return OrderExecutionResult(
                    success=False,
                    order_request=order_request,
                    error_message="Duplicate order detected",
                    correlation_id=correlation_id,
                )

            # Step 2: Validate with RiskManager
            order_request.update_status(OrderRequestStatus.VALIDATING)

            if not await self._validate_with_risk_manager(order_request):
                order_request.update_status(
                    OrderRequestStatus.REJECTED, order_request.rejection_reason
                )
                logger.warning(
                    "Order rejected by risk manager",
                    reason=order_request.rejection_reason,
                )
                return OrderExecutionResult(
                    success=False,
                    order_request=order_request,
                    error_message=order_request.rejection_reason,
                    correlation_id=correlation_id,
                )

            order_request.update_status(OrderRequestStatus.VALIDATED)

            # Step 3: Check emergency stop
            if self._emergency_stop and self._emergency_stop.is_halted():
                order_request.update_status(OrderRequestStatus.REJECTED, "Emergency stop is active")
                logger.warning("Order rejected due to emergency stop")
                return OrderExecutionResult(
                    success=False,
                    order_request=order_request,
                    error_message="Emergency stop is active",
                    correlation_id=correlation_id,
                )

            # Step 4: Submit to exchange
            order_request.update_status(OrderRequestStatus.SUBMITTING)

            try:
                order = await self._submit_to_exchange(order_request)

                if not order:
                    order_request.update_status(
                        OrderRequestStatus.FAILED, order_request.failure_reason
                    )
                    return OrderExecutionResult(
                        success=False,
                        order_request=order_request,
                        error_message=order_request.failure_reason,
                        correlation_id=correlation_id,
                    )

                # Update order request with exchange details
                order_request.exchange_order_id = order.id
                order_request.exchange_order = order
                order_request.filled_quantity = order.filled_quantity

                if order.average_price:
                    order_request.average_price = order.average_price

                order_request.update_status(OrderRequestStatus.SUBMITTED)

                # Track order
                async with self._order_lock:
                    self._active_orders[order.id] = order_request

                # Update order status based on initial response
                self._update_request_status_from_order(order_request, order)

                # Step 5: Update portfolio state
                await self._update_portfolio_state(order_request, order)

                # Step 6: Invoke callbacks
                await self._invoke_order_callbacks(order)

                logger.info(
                    "Order executed successfully",
                    order_id=order.id,
                    status=order.status.value,
                    filled_quantity=str(order.filled_quantity),
                )

                # Log order event
                log_order(
                    order_id=order.id,
                    symbol=order.symbol,
                    side=order.side.value,
                    order_type=order.type.value,
                    quantity=float(order.quantity),
                    price=str(order.price) if order.price else None,
                    status=order.status.value,
                    correlation_id=correlation_id,
                )

                return OrderExecutionResult(
                    success=True,
                    order_request=order_request,
                    order=order,
                    correlation_id=correlation_id,
                )

            except Exception as e:
                order_request.update_status(OrderRequestStatus.FAILED, str(e))
                logger.error(
                    "Order execution failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                return OrderExecutionResult(
                    success=False,
                    order_request=order_request,
                    error_message=str(e),
                    correlation_id=correlation_id,
                )

    async def _is_duplicate_order(self, order_request: OrderRequest) -> bool:
        """
        Check if this order is a duplicate based on recent order history.

        Uses order hash and timestamp to detect duplicates within the deduplication window.

        Args:
            order_request: The order request to check

        Returns:
            True if duplicate detected, False otherwise
        """
        order_hash = order_request.get_order_hash()
        current_time = time.time()

        async with self._order_lock:
            # Clean up old entries outside the deduplication window
            expired_hashes = [
                h
                for h, timestamp in self._order_hash_timestamps.items()
                if current_time - timestamp > self._deduplication_window
            ]

            for h in expired_hashes:
                del self._order_hash_timestamps[h]

            # Check if this order hash exists in recent history
            if order_hash in self._order_hash_timestamps:
                return True

            # Add to tracking
            self._order_hash_timestamps[order_hash] = current_time

            return False

    async def _validate_with_risk_manager(self, order_request: OrderRequest) -> bool:
        """
        Validate order with RiskManager.

        Args:
            order_request: The order request to validate

        Returns:
            True if validation passed, False otherwise
        """
        try:
            # Get current price for validation
            current_price = await self._exchange.get_ticker_price(order_request.symbol)

            # Validate trade with RiskManager
            validation_result = await self._risk_manager.validate_trade(
                symbol=order_request.symbol,
                side=order_request.side,
                quantity=order_request.quantity,
                price=current_price,
            )

            if not validation_result.approved:
                # Build rejection reason from failed checks
                failed_checks = validation_result.get_failed_checks()
                rejection_reasons = [check.message for check in failed_checks]
                order_request.rejection_reason = "; ".join(rejection_reasons)

                logger.warning(
                    "Risk validation failed",
                    symbol=order_request.symbol,
                    failed_checks=[check.check_name for check in failed_checks],
                )

                return False

            # Log any warnings
            warnings = validation_result.get_warnings()
            if warnings:
                logger.warning(
                    "Risk validation warnings",
                    symbol=order_request.symbol,
                    warnings=[w.message for w in warnings],
                )

            return True

        except Exception as e:
            order_request.rejection_reason = f"Risk validation error: {e!s}"
            logger.error(
                "Risk validation error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    async def _submit_to_exchange(self, order_request: OrderRequest) -> Order | None:
        """
        Submit order to exchange.

        Args:
            order_request: The order request to submit

        Returns:
            Order object if successful, None otherwise
        """
        try:
            # Submit to exchange
            order = await self._exchange.create_order(
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                quantity=order_request.quantity,
                price=order_request.price,
                stop_price=order_request.stop_price,
                time_in_force=order_request.time_in_force,
            )

            logger.info(
                "Order submitted to exchange",
                order_id=order.id,
                symbol=order.symbol,
                status=order.status.value,
            )

            return order

        except InsufficientBalanceError as e:
            order_request.failure_reason = f"Insufficient balance: {e!s}"
            logger.error("Insufficient balance for order", error=str(e))
            return None

        except InvalidOrderError as e:
            order_request.failure_reason = f"Invalid order: {e!s}"
            logger.error("Invalid order parameters", error=str(e))
            return None

        except Exception as e:
            order_request.failure_reason = f"Exchange error: {e!s}"
            logger.error(
                "Failed to submit order to exchange",
                error=str(e),
                error_type=type(e).__name__,
            )
            return None

    def _update_request_status_from_order(self, order_request: OrderRequest, order: Order) -> None:
        """
        Update order request status based on exchange order status.

        Args:
            order_request: The order request to update
            order: The exchange order
        """
        if order.status == OrderStatus.CLOSED:
            order_request.update_status(OrderRequestStatus.FILLED)
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            order_request.update_status(OrderRequestStatus.PARTIALLY_FILLED)
        elif order.status == OrderStatus.CANCELED:
            order_request.update_status(OrderRequestStatus.CANCELLED)
        elif order.status == OrderStatus.REJECTED:
            order_request.update_status(OrderRequestStatus.REJECTED)
        elif order.status == OrderStatus.EXPIRED:
            order_request.update_status(OrderRequestStatus.EXPIRED)

    async def _update_portfolio_state(self, order_request: OrderRequest, order: Order) -> None:
        """
        Update portfolio state in RiskManager after order execution.

        Args:
            order_request: The executed order request
            order: The exchange order
        """
        try:
            # Calculate position value
            if order.filled_quantity > 0 and order.average_price:
                position_value = order.filled_quantity * order.average_price

                # Update position in RiskManager
                if order.side == OrderSide.BUY:
                    self._risk_manager.update_position(
                        symbol=order.symbol, position_value=position_value, add=True
                    )
                    logger.debug(
                        "Updated long position",
                        symbol=order.symbol,
                        value=str(position_value),
                    )
                else:  # SELL
                    self._risk_manager.update_position(
                        symbol=order.symbol, position_value=position_value, add=False
                    )
                    logger.debug(
                        "Updated short position",
                        symbol=order.symbol,
                        value=str(position_value),
                    )

        except Exception as e:
            logger.error(
                "Failed to update portfolio state",
                order_id=order.id,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _invoke_order_callbacks(self, order: Order) -> None:
        """
        Invoke registered order callbacks.

        Args:
            order: The order to pass to callbacks
        """
        for callback in self._order_callbacks:
            try:
                # Check if callback is async
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(
                    "Order callback failed",
                    callback=callback.__name__,
                    error=str(e),
                    error_type=type(e).__name__,
                )

    async def _monitor_orders(self) -> None:
        """
        Background task to monitor active orders and update their status.

        Runs continuously while the orchestrator is active.
        """
        logger.info("Order monitoring task started")

        while self._running:
            try:
                await asyncio.sleep(self._status_check_interval)

                # Get snapshot of active orders
                async with self._order_lock:
                    active_orders = list(self._active_orders.items())

                # Check status of each active order
                for order_id, order_request in active_orders:
                    if not order_request.exchange_order_id:
                        continue

                    try:
                        # Get current order status
                        order = await self._exchange.get_order(
                            order_id=order_request.exchange_order_id,
                            symbol=order_request.symbol,
                        )

                        # Update order request
                        order_request.exchange_order = order
                        order_request.filled_quantity = order.filled_quantity

                        if order.average_price:
                            order_request.average_price = order.average_price

                        # Update status
                        old_status = order_request.status
                        self._update_request_status_from_order(order_request, order)

                        # Log status changes
                        if order_request.status != old_status:
                            logger.info(
                                "Order status updated",
                                order_id=order_id,
                                old_status=old_status.value,
                                new_status=order_request.status.value,
                                filled_quantity=str(order.filled_quantity),
                            )

                            # Invoke callbacks for status change
                            await self._invoke_order_callbacks(order)

                        # Remove completed orders from active tracking
                        if order_request.status in [
                            OrderRequestStatus.FILLED,
                            OrderRequestStatus.CANCELLED,
                            OrderRequestStatus.REJECTED,
                            OrderRequestStatus.EXPIRED,
                        ]:
                            async with self._order_lock:
                                if order_id in self._active_orders:
                                    del self._active_orders[order_id]

                                    logger.info(
                                        "Order removed from active tracking",
                                        order_id=order_id,
                                        final_status=order_request.status.value,
                                    )

                    except Exception as e:
                        logger.error(
                            "Failed to check order status",
                            order_id=order_id,
                            error=str(e),
                            error_type=type(e).__name__,
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Order monitoring error",
                    error=str(e),
                    error_type=type(e).__name__,
                )

        logger.info("Order monitoring task stopped")
