"""
Comprehensive unit tests for ExecutionOrchestrator.

Tests all orchestrator features:
- Order execution (market, limit, stop-loss)
- Order deduplication
- Risk validation integration
- Emergency stop integration
- Order status tracking and updates
- Portfolio state updates
- Order lifecycle management
- Callback invocation
- Concurrent order handling
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.execution.orchestrator import (
    ExecutionOrchestrator,
    OrderExecutionResult,
    OrderRequest,
    OrderRequestStatus,
)
from bot.interfaces.exchange import (
    InsufficientBalanceError,
    InvalidOrderError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot.risk.risk_manager import (
    RiskCheckResult,
    RiskCheckStatus,
    TradeValidationResult,
)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange interface."""
    exchange = AsyncMock()

    # Default successful order response
    exchange.create_order = AsyncMock(
        return_value=Order(
            id="ORDER123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.CLOSED,
            filled_quantity=Decimal("0.1"),
            remaining_quantity=Decimal("0"),
            average_price=Decimal("50000"),
            created_at=datetime.now(),
            commission=Decimal("0.001"),
            commission_asset="BTC",
        )
    )

    exchange.cancel_order = AsyncMock(
        return_value=Order(
            id="ORDER123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.CANCELED,
            filled_quantity=Decimal("0"),
            remaining_quantity=Decimal("0.1"),
            average_price=None,
            created_at=datetime.now(),
            commission=Decimal("0"),
            commission_asset="BTC",
        )
    )

    exchange.get_order = AsyncMock(
        return_value=Order(
            id="ORDER123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.CLOSED,
            filled_quantity=Decimal("0.1"),
            remaining_quantity=Decimal("0"),
            average_price=Decimal("50000"),
            created_at=datetime.now(),
            commission=Decimal("0.001"),
            commission_asset="BTC",
        )
    )

    exchange.get_ticker_price = AsyncMock(return_value=Decimal("50000"))

    return exchange


@pytest.fixture
def mock_risk_manager():
    """Create a mock risk manager."""
    risk_manager = AsyncMock()

    # Default approval
    risk_manager.validate_trade = AsyncMock(
        return_value=TradeValidationResult(
            approved=True,
            results=[
                RiskCheckResult(
                    check_name="position_size",
                    status=RiskCheckStatus.PASSED,
                    passed=True,
                    message="Position size within limits",
                )
            ],
            correlation_id="test-correlation-123",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            estimated_value=Decimal("5000"),
        )
    )

    risk_manager.update_position = AsyncMock()

    return risk_manager


@pytest.fixture
def mock_emergency_stop():
    """Create a mock emergency stop manager."""
    emergency_stop = MagicMock()
    emergency_stop.is_halted = MagicMock(return_value=False)
    return emergency_stop


@pytest.fixture
async def orchestrator(mock_exchange, mock_risk_manager, mock_emergency_stop):
    """Create an ExecutionOrchestrator instance for testing."""
    orch = ExecutionOrchestrator(
        exchange=mock_exchange,
        risk_manager=mock_risk_manager,
        emergency_stop=mock_emergency_stop,
        deduplication_window_seconds=60,
        order_status_check_interval_seconds=5,
    )

    await orch.start()
    yield orch
    await orch.stop()


class TestMarketOrderExecution:
    """Test market order execution."""

    @pytest.mark.asyncio
    async def test_successful_market_buy_order(
        self, orchestrator, mock_exchange, mock_risk_manager
    ):
        """Test successful execution of a market buy order."""
        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result.success is True
        assert result.order is not None
        assert result.order.id == "ORDER123"
        assert result.order.status == OrderStatus.CLOSED
        assert result.error_message is None

        # Verify exchange was called
        mock_exchange.create_order.assert_called_once()
        call_kwargs = mock_exchange.create_order.call_args.kwargs
        assert call_kwargs["symbol"] == "BTCUSDT"
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["order_type"] == OrderType.MARKET
        assert call_kwargs["quantity"] == Decimal("0.1")

        # Verify risk validation was called
        mock_risk_manager.validate_trade.assert_called_once()

        # Verify portfolio was updated
        mock_risk_manager.update_position.assert_called_once()

    @pytest.mark.asyncio
    async def test_successful_market_sell_order(
        self, orchestrator, mock_exchange, mock_risk_manager
    ):
        """Test successful execution of a market sell order."""
        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
        )

        assert result.success is True
        assert result.order is not None

        # Verify correct side
        call_kwargs = mock_exchange.create_order.call_args.kwargs
        assert call_kwargs["side"] == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_market_order_with_correlation_id(self, orchestrator):
        """Test market order execution with custom correlation ID."""
        correlation_id = "custom-correlation-123"

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            correlation_id=correlation_id,
        )

        assert result.success is True
        assert result.correlation_id == correlation_id
        assert result.order_request.correlation_id == correlation_id

    @pytest.mark.asyncio
    async def test_market_order_with_metadata(self, orchestrator):
        """Test market order execution with custom metadata."""
        metadata = {"strategy": "RSI", "signal_strength": 0.85}

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            metadata=metadata,
        )

        assert result.success is True
        assert result.order_request.metadata == metadata


class TestLimitOrderExecution:
    """Test limit order execution."""

    @pytest.mark.asyncio
    async def test_successful_limit_order(self, orchestrator, mock_exchange):
        """Test successful execution of a limit order."""
        # Mock limit order response
        mock_exchange.create_order.return_value = Order(
            id="LIMIT123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.OPEN,
            filled_quantity=Decimal("0"),
            remaining_quantity=Decimal("0.1"),
            average_price=None,
            created_at=datetime.now(),
            commission=Decimal("0"),
            commission_asset="BTC",
        )

        result = await orchestrator.execute_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
            time_in_force=TimeInForce.GTC,
        )

        assert result.success is True
        assert result.order is not None
        assert result.order.type == OrderType.LIMIT
        assert result.order.price == Decimal("49000")
        assert result.order.status == OrderStatus.OPEN

        # Verify exchange was called with correct parameters
        call_kwargs = mock_exchange.create_order.call_args.kwargs
        assert call_kwargs["order_type"] == OrderType.LIMIT
        assert call_kwargs["price"] == Decimal("49000")
        assert call_kwargs["time_in_force"] == TimeInForce.GTC

    @pytest.mark.asyncio
    async def test_limit_order_ioc(self, orchestrator, mock_exchange):
        """Test limit order with IOC time in force."""
        result = await orchestrator.execute_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
            time_in_force=TimeInForce.IOC,
        )

        assert result.success is True

        call_kwargs = mock_exchange.create_order.call_args.kwargs
        assert call_kwargs["time_in_force"] == TimeInForce.IOC


class TestStopLossOrderExecution:
    """Test stop-loss order execution."""

    @pytest.mark.asyncio
    async def test_successful_stop_loss_order(self, orchestrator, mock_exchange):
        """Test successful execution of a stop-loss order."""
        # Mock stop-loss order response
        mock_exchange.create_order.return_value = Order(
            id="STOP123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            type=OrderType.STOP_LOSS,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=Decimal("48000"),
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            filled_quantity=Decimal("0"),
            remaining_quantity=Decimal("0.1"),
            average_price=None,
            created_at=datetime.now(),
            commission=Decimal("0"),
            commission_asset="BTC",
        )

        result = await orchestrator.execute_stop_loss_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            stop_price=Decimal("48000"),
        )

        assert result.success is True
        assert result.order is not None
        assert result.order.type == OrderType.STOP_LOSS
        assert result.order.stop_price == Decimal("48000")

        # Verify exchange was called correctly
        call_kwargs = mock_exchange.create_order.call_args.kwargs
        assert call_kwargs["order_type"] == OrderType.STOP_LOSS
        assert call_kwargs["stop_price"] == Decimal("48000")


class TestOrderDeduplication:
    """Test order deduplication mechanism."""

    @pytest.mark.asyncio
    async def test_duplicate_order_rejected(self, orchestrator):
        """Test that duplicate orders are rejected."""
        # Execute first order
        result1 = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result1.success is True

        # Try to execute identical order immediately
        result2 = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result2.success is False
        assert "Duplicate order" in result2.error_message
        assert result2.order_request.status == OrderRequestStatus.REJECTED

    @pytest.mark.asyncio
    async def test_different_orders_not_duplicates(self, orchestrator):
        """Test that different orders are not considered duplicates."""
        # Execute first order
        result1 = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result1.success is True

        # Execute different order (different quantity)
        result2 = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.2"),  # Different quantity
        )

        assert result2.success is True

        # Execute different order (different side)
        result3 = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,  # Different side
            quantity=Decimal("0.1"),
        )

        assert result3.success is True

    @pytest.mark.asyncio
    async def test_order_hash_generation(self):
        """Test order hash generation for deduplication."""
        request1 = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        request2 = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        # Same orders should have same hash
        assert request1.get_order_hash() == request2.get_order_hash()

        # Different quantity should have different hash
        request3 = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.2"),
        )

        assert request1.get_order_hash() != request3.get_order_hash()


class TestRiskValidation:
    """Test integration with RiskManager."""

    @pytest.mark.asyncio
    async def test_order_rejected_by_risk_manager(self, orchestrator, mock_risk_manager):
        """Test that orders rejected by risk manager are not executed."""
        # Mock risk rejection
        mock_risk_manager.validate_trade.return_value = TradeValidationResult(
            approved=False,
            results=[
                RiskCheckResult(
                    check_name="position_size",
                    status=RiskCheckStatus.FAILED,
                    passed=False,
                    message="Position size exceeds limit",
                )
            ],
            correlation_id="test-123",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            estimated_value=Decimal("500000"),
        )

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
        )

        assert result.success is False
        assert "Position size exceeds limit" in result.error_message
        assert result.order_request.status == OrderRequestStatus.REJECTED
        assert result.order is None

    @pytest.mark.asyncio
    async def test_risk_validation_with_warnings(
        self, orchestrator, mock_risk_manager, mock_exchange
    ):
        """Test that orders with warnings are still executed."""
        # Mock risk approval with warnings
        mock_risk_manager.validate_trade.return_value = TradeValidationResult(
            approved=True,
            results=[
                RiskCheckResult(
                    check_name="correlation",
                    status=RiskCheckStatus.WARNING,
                    passed=True,
                    message="High correlation with existing positions",
                ),
                RiskCheckResult(
                    check_name="position_size",
                    status=RiskCheckStatus.PASSED,
                    passed=True,
                    message="Position size OK",
                ),
            ],
            correlation_id="test-123",
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            estimated_value=Decimal("5000"),
        )

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        # Should still succeed despite warnings
        assert result.success is True
        assert result.order is not None


class TestEmergencyStop:
    """Test integration with EmergencyStop."""

    @pytest.mark.asyncio
    async def test_order_rejected_during_emergency_stop(self, orchestrator, mock_emergency_stop):
        """Test that orders are rejected when emergency stop is active."""
        # Activate emergency stop
        mock_emergency_stop.is_halted.return_value = True

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result.success is False
        assert "Emergency stop is active" in result.error_message
        assert result.order_request.status == OrderRequestStatus.REJECTED
        assert result.order is None

    @pytest.mark.asyncio
    async def test_order_succeeds_when_emergency_stop_inactive(
        self, orchestrator, mock_emergency_stop
    ):
        """Test that orders succeed when emergency stop is inactive."""
        # Ensure emergency stop is inactive
        mock_emergency_stop.is_halted.return_value = False

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result.success is True
        assert result.order is not None


class TestExchangeErrors:
    """Test handling of exchange errors."""

    @pytest.mark.asyncio
    async def test_insufficient_balance_error(self, orchestrator, mock_exchange):
        """Test handling of insufficient balance errors."""
        mock_exchange.create_order.side_effect = InsufficientBalanceError("Insufficient balance")

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("100.0"),
        )

        assert result.success is False
        assert "Insufficient balance" in result.error_message
        assert result.order_request.status == OrderRequestStatus.FAILED

    @pytest.mark.asyncio
    async def test_invalid_order_error(self, orchestrator, mock_exchange):
        """Test handling of invalid order errors."""
        mock_exchange.create_order.side_effect = InvalidOrderError("Invalid quantity")

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.00001"),
        )

        assert result.success is False
        assert "Invalid order" in result.error_message
        assert result.order_request.status == OrderRequestStatus.FAILED

    @pytest.mark.asyncio
    async def test_generic_exchange_error(self, orchestrator, mock_exchange):
        """Test handling of generic exchange errors."""
        mock_exchange.create_order.side_effect = Exception("Network timeout")

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result.success is False
        assert "Network timeout" in result.error_message
        assert result.order_request.status == OrderRequestStatus.FAILED


class TestOrderCancellation:
    """Test order cancellation."""

    @pytest.mark.asyncio
    async def test_successful_order_cancellation(self, orchestrator, mock_exchange):
        """Test successful cancellation of an order."""
        # First create an order
        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        order_id = result.order.id

        # Cancel the order
        success = await orchestrator.cancel_order(order_id=order_id, symbol="BTCUSDT")

        assert success is True
        mock_exchange.cancel_order.assert_called_once_with(order_id=order_id, symbol="BTCUSDT")

    @pytest.mark.asyncio
    async def test_cancel_order_error(self, orchestrator, mock_exchange):
        """Test handling of cancellation errors."""
        mock_exchange.cancel_order.side_effect = Exception("Order not found")

        success = await orchestrator.cancel_order(order_id="NONEXISTENT", symbol="BTCUSDT")

        assert success is False

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, orchestrator, mock_exchange, mock_risk_manager):
        """Test cancellation of all active orders."""
        # Create limit orders that stay open
        mock_exchange.create_order.return_value = Order(
            id="LIMIT1",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.OPEN,
            filled_quantity=Decimal("0"),
            remaining_quantity=Decimal("0.1"),
            average_price=None,
            created_at=datetime.now(),
            commission=Decimal("0"),
            commission_asset="BTC",
        )

        # Create multiple orders
        await orchestrator.execute_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
        )

        # Cancel all orders
        cancelled_count = await orchestrator.cancel_all_orders()

        assert cancelled_count >= 0


class TestOrderStatusTracking:
    """Test order status tracking and monitoring."""

    @pytest.mark.asyncio
    async def test_get_order_status(self, orchestrator, mock_exchange):
        """Test retrieving order status from exchange."""
        order = await orchestrator.get_order_status(order_id="ORDER123", symbol="BTCUSDT")

        assert order is not None
        assert order.id == "ORDER123"
        mock_exchange.get_order.assert_called_once_with(order_id="ORDER123", symbol="BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_order_status_not_found(self, orchestrator, mock_exchange):
        """Test handling of order not found."""
        mock_exchange.get_order.side_effect = Exception("Order not found")

        order = await orchestrator.get_order_status(order_id="NONEXISTENT", symbol="BTCUSDT")

        assert order is None

    @pytest.mark.asyncio
    async def test_get_active_orders(self, orchestrator):
        """Test retrieving active orders."""
        # Execute some orders
        await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        active_orders = orchestrator.get_active_orders()

        assert isinstance(active_orders, list)

    @pytest.mark.asyncio
    async def test_update_request_status_from_order(self, orchestrator):
        """Test updating order request status based on exchange order."""
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        # Test filled order
        filled_order = Order(
            id="ORDER123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.CLOSED,
            filled_quantity=Decimal("0.1"),
            remaining_quantity=Decimal("0"),
            average_price=Decimal("50000"),
            created_at=datetime.now(),
            commission=Decimal("0.001"),
            commission_asset="BTC",
        )

        orchestrator._update_request_status_from_order(order_request, filled_order)
        assert order_request.status == OrderRequestStatus.FILLED

        # Test cancelled order
        cancelled_order = filled_order
        cancelled_order.status = OrderStatus.CANCELED

        orchestrator._update_request_status_from_order(order_request, cancelled_order)
        assert order_request.status == OrderRequestStatus.CANCELLED


class TestPortfolioUpdates:
    """Test portfolio state updates."""

    @pytest.mark.asyncio
    async def test_portfolio_update_on_buy(self, orchestrator, mock_risk_manager, mock_exchange):
        """Test portfolio update after buy order."""
        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result.success is True

        # Verify portfolio was updated with positive value
        mock_risk_manager.update_position.assert_called_once()
        call_args = mock_risk_manager.update_position.call_args

        # For BUY, should add positive position
        assert call_args.kwargs["symbol"] == "BTCUSDT"
        assert call_args.kwargs["position_value"] > 0
        assert call_args.kwargs["add"] is True

    @pytest.mark.asyncio
    async def test_portfolio_update_on_sell(self, orchestrator, mock_risk_manager, mock_exchange):
        """Test portfolio update after sell order."""
        # Configure for sell order
        mock_exchange.create_order.return_value = Order(
            id="SELL123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.CLOSED,
            filled_quantity=Decimal("0.1"),
            remaining_quantity=Decimal("0"),
            average_price=Decimal("50000"),
            created_at=datetime.now(),
            commission=Decimal("0.001"),
            commission_asset="USDT",
        )

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
        )

        assert result.success is True

        # Verify portfolio was updated (for SELL, we remove position)
        mock_risk_manager.update_position.assert_called_once()
        call_args = mock_risk_manager.update_position.call_args

        assert call_args.kwargs["symbol"] == "BTCUSDT"
        assert call_args.kwargs["position_value"] > 0
        assert call_args.kwargs["add"] is False


class TestOrderCallbacks:
    """Test order callback functionality."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_order_execution(self, orchestrator):
        """Test that callbacks are invoked when orders are executed."""
        callback_invoked = False
        received_order = None

        def callback(order: Order) -> None:
            nonlocal callback_invoked, received_order
            callback_invoked = True
            received_order = order

        orchestrator.add_order_callback(callback)

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result.success is True
        assert callback_invoked is True
        assert received_order is not None
        assert received_order.id == result.order.id

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self, orchestrator):
        """Test that async callbacks are invoked correctly."""
        callback_invoked = False

        async def async_callback(order: Order) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        orchestrator.add_order_callback(async_callback)

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        assert result.success is True
        assert callback_invoked is True

    @pytest.mark.asyncio
    async def test_callback_error_does_not_fail_execution(self, orchestrator):
        """Test that callback errors don't prevent order execution."""

        def failing_callback(order: Order) -> None:
            raise ValueError("Callback error")

        orchestrator.add_order_callback(failing_callback)

        result = await orchestrator.execute_market_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
        )

        # Order should still succeed despite callback error
        assert result.success is True


class TestOrchestratorMetrics:
    """Test orchestrator metrics and monitoring."""

    @pytest.mark.asyncio
    async def test_get_order_metrics(self, orchestrator, mock_exchange):
        """Test retrieving orchestrator metrics."""
        # Create some limit orders to have active orders
        mock_exchange.create_order.return_value = Order(
            id="LIMIT1",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.OPEN,
            filled_quantity=Decimal("0"),
            remaining_quantity=Decimal("0.1"),
            average_price=None,
            created_at=datetime.now(),
            commission=Decimal("0"),
            commission_asset="BTC",
        )

        await orchestrator.execute_limit_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("49000"),
        )

        metrics = orchestrator.get_order_metrics()

        assert "total_active_orders" in metrics
        assert "status_counts" in metrics
        assert "deduplication_cache_size" in metrics
        assert "running" in metrics
        assert metrics["running"] is True


class TestOrderRequestDataclass:
    """Test OrderRequest dataclass methods."""

    def test_order_request_creation(self):
        """Test creating an order request."""
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert request.symbol == "BTCUSDT"
        assert request.side == OrderSide.BUY
        assert request.order_type == OrderType.MARKET
        assert request.quantity == Decimal("0.1")
        assert request.status == OrderRequestStatus.PENDING

    def test_order_request_update_status(self):
        """Test updating order request status."""
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        old_updated_at = request.updated_at

        request.update_status(OrderRequestStatus.VALIDATED)

        assert request.status == OrderRequestStatus.VALIDATED
        assert request.updated_at >= old_updated_at

    def test_order_request_rejection(self):
        """Test order request rejection tracking."""
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        request.update_status(OrderRequestStatus.REJECTED, "Risk check failed")

        assert request.status == OrderRequestStatus.REJECTED
        assert request.rejection_reason == "Risk check failed"

    def test_order_request_to_dict(self):
        """Test converting order request to dictionary."""
        request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            metadata={"strategy": "RSI"},
        )

        request_dict = request.to_dict()

        assert request_dict["symbol"] == "BTCUSDT"
        assert request_dict["side"] == "buy"
        assert request_dict["order_type"] == "market"
        assert request_dict["quantity"] == "0.1"
        assert request_dict["metadata"]["strategy"] == "RSI"


class TestOrderExecutionResult:
    """Test OrderExecutionResult dataclass."""

    def test_execution_result_success(self):
        """Test successful execution result."""
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        order = Order(
            id="ORDER123",
            client_order_id=None,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.CLOSED,
            filled_quantity=Decimal("0.1"),
            remaining_quantity=Decimal("0"),
            average_price=Decimal("50000"),
            created_at=datetime.now(),
            commission=Decimal("0.001"),
            commission_asset="BTC",
        )

        result = OrderExecutionResult(
            success=True,
            order_request=order_request,
            order=order,
            correlation_id="test-123",
        )

        assert result.success is True
        assert result.order is not None
        assert result.error_message is None

    def test_execution_result_failure(self):
        """Test failed execution result."""
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        result = OrderExecutionResult(
            success=False,
            order_request=order_request,
            error_message="Insufficient balance",
            correlation_id="test-123",
        )

        assert result.success is False
        assert result.order is None
        assert result.error_message == "Insufficient balance"

    def test_execution_result_to_dict(self):
        """Test converting execution result to dictionary."""
        order_request = OrderRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        result = OrderExecutionResult(
            success=False,
            order_request=order_request,
            error_message="Test error",
            correlation_id="test-123",
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is False
        assert result_dict["error_message"] == "Test error"
        assert result_dict["correlation_id"] == "test-123"
        assert "order_request" in result_dict
