"""
Integration tests for Binance exchange adapter.

These tests run against the Binance testnet and require valid API credentials.
They test real API interactions, order flow, and data retrieval.

To run these tests:
1. Create a Binance testnet account at https://testnet.binance.vision/
2. Generate API keys
3. Set environment variables:
   - BINANCE_TESTNET_API_KEY
   - BINANCE_TESTNET_API_SECRET
4. Run: pytest tests/integration/execution/adapters/test_binance_integration.py -v

Note: These tests may be slow due to rate limiting and network latency.
"""

import asyncio
import os
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from bot.execution.adapters.binance import BinanceAdapter
from bot.interfaces.exchange import (
    ExchangeAPIError,
    ExchangeAuthenticationError,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

# Skip all tests if API credentials not available
pytestmark = pytest.mark.skipif(
    not os.getenv("BINANCE_TESTNET_API_KEY") or not os.getenv("BINANCE_TESTNET_API_SECRET"),
    reason="Binance testnet credentials not available. Set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET environment variables.",
)


@pytest.fixture
async def adapter():
    """Create and connect to Binance testnet adapter."""
    api_key = os.getenv("BINANCE_TESTNET_API_KEY")
    api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

    adapter = BinanceAdapter(api_key=api_key, api_secret=api_secret, testnet=True)

    await adapter.connect()
    yield adapter
    await adapter.disconnect()


class TestBinanceIntegration:
    """Integration tests for Binance adapter."""

    @pytest.mark.asyncio
    async def test_connection_and_authentication(self, adapter):
        """Test successful connection and authentication."""
        # Connection already established in fixture
        assert adapter.session is not None

        # Verify we can make authenticated requests
        account_info = await adapter.get_account_info()
        assert account_info is not None
        assert account_info.can_trade is not None

    @pytest.mark.asyncio
    async def test_connectivity_check(self, adapter):
        """Test connectivity check."""
        result = await adapter.test_connectivity()
        assert result is True

    @pytest.mark.asyncio
    async def test_get_account_info(self, adapter):
        """Test retrieving account information."""
        account_info = await adapter.get_account_info()

        assert account_info.account_type in ["SPOT", "MARGIN", "FUTURES"]
        assert isinstance(account_info.can_trade, bool)
        assert isinstance(account_info.can_withdraw, bool)
        assert isinstance(account_info.can_deposit, bool)
        assert isinstance(account_info.balances, list)
        assert account_info.maker_commission is not None
        assert account_info.taker_commission is not None

    @pytest.mark.asyncio
    async def test_get_balance(self, adapter):
        """Test retrieving account balances."""
        balances = await adapter.get_balance()

        assert isinstance(balances, list)
        # All balances should have positive total
        for balance in balances:
            assert balance.total > 0
            assert balance.free >= 0
            assert balance.locked >= 0
            assert balance.total == balance.free + balance.locked

    @pytest.mark.asyncio
    async def test_get_balance_specific_asset(self, adapter):
        """Test retrieving balance for specific asset."""
        # Get all balances first to find an asset we have
        all_balances = await adapter.get_balance()

        if all_balances:
            test_asset = all_balances[0].asset
            specific_balance = await adapter.get_balance(test_asset)

            assert len(specific_balance) <= 1  # Should be 0 or 1
            if specific_balance:
                assert specific_balance[0].asset == test_asset

    @pytest.mark.asyncio
    async def test_get_ticker_price(self, adapter):
        """Test retrieving ticker price."""
        # BTC/USDT should always be available
        price = await adapter.get_ticker_price("BTCUSDT")

        assert isinstance(price, Decimal)
        assert price > 0

    @pytest.mark.asyncio
    async def test_create_and_cancel_limit_order(self, adapter):
        """
        Test creating and canceling a limit order.

        Uses a price far from market to avoid accidental fills.
        """
        # Get current BTC price
        current_price = await adapter.get_ticker_price("BTCUSDT")

        # Create a buy order at 50% of current price (won't fill)
        order_price = (current_price * Decimal("0.5")).quantize(Decimal("0.01"))

        # Create limit order
        order = await adapter.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),  # Minimum order size
            price=order_price,
            time_in_force=TimeInForce.GTC,
        )

        assert order.id is not None
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.type == OrderType.LIMIT
        assert order.status in [OrderStatus.OPEN, OrderStatus.PENDING]
        assert order.quantity == Decimal("0.001")
        assert order.price == order_price

        # Wait a moment to ensure order is registered
        import asyncio

        await asyncio.sleep(1)

        # Cancel the order
        canceled_order = await adapter.cancel_order(symbol="BTCUSDT", order_id=order.id)

        assert canceled_order.id == order.id
        assert canceled_order.status == OrderStatus.CANCELED

    @pytest.mark.asyncio
    async def test_get_order(self, adapter):
        """Test retrieving order information."""
        # Create an order first
        current_price = await adapter.get_ticker_price("BTCUSDT")
        order_price = (current_price * Decimal("0.5")).quantize(Decimal("0.01"))

        created_order = await adapter.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=order_price,
            time_in_force=TimeInForce.GTC,
        )

        # Retrieve the order
        retrieved_order = await adapter.get_order(symbol="BTCUSDT", order_id=created_order.id)

        assert retrieved_order.id == created_order.id
        assert retrieved_order.symbol == created_order.symbol
        assert retrieved_order.side == created_order.side

        # Clean up
        await adapter.cancel_order(symbol="BTCUSDT", order_id=created_order.id)

    @pytest.mark.asyncio
    async def test_get_open_orders(self, adapter):
        """Test retrieving open orders."""
        # Cancel any existing open orders first
        existing_orders = await adapter.get_open_orders("BTCUSDT")
        for order in existing_orders:
            await adapter.cancel_order(symbol="BTCUSDT", order_id=order.id)

        # Create a test order
        current_price = await adapter.get_ticker_price("BTCUSDT")
        order_price = (current_price * Decimal("0.5")).quantize(Decimal("0.01"))

        created_order = await adapter.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=order_price,
            time_in_force=TimeInForce.GTC,
        )

        # Get open orders
        open_orders = await adapter.get_open_orders("BTCUSDT")

        assert len(open_orders) >= 1
        assert any(order.id == created_order.id for order in open_orders)

        # Clean up
        await adapter.cancel_order(symbol="BTCUSDT", order_id=created_order.id)

    @pytest.mark.asyncio
    async def test_get_order_history(self, adapter):
        """Test retrieving order history."""
        # Get recent order history
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        orders = await adapter.get_order_history(
            symbol="BTCUSDT", start_time=start_time, end_time=end_time, limit=10
        )

        assert isinstance(orders, list)
        # Each order should have required fields
        for order in orders:
            assert order.id is not None
            assert order.symbol == "BTCUSDT"
            assert order.status is not None

    @pytest.mark.asyncio
    async def test_get_trades(self, adapter):
        """Test retrieving trade history."""
        # Get recent trades
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        trades = await adapter.get_trades(
            symbol="BTCUSDT", start_time=start_time, end_time=end_time, limit=10
        )

        assert isinstance(trades, list)
        # Each trade should have required fields
        for trade in trades:
            assert trade.id is not None
            assert trade.symbol == "BTCUSDT"
            assert trade.price > 0
            assert trade.quantity > 0

    @pytest.mark.asyncio
    async def test_rate_limiting(self, adapter):
        """Test that rate limiting works without errors."""
        # Make multiple rapid requests
        tasks = []
        for _ in range(10):
            tasks.append(adapter.get_ticker_price("BTCUSDT"))

        # Should complete without rate limit errors
        prices = await asyncio.gather(*tasks)

        assert len(prices) == 10
        for price in prices:
            assert isinstance(price, Decimal)
            assert price > 0

    @pytest.mark.asyncio
    async def test_invalid_authentication(self):
        """Test that invalid credentials raise authentication error."""
        invalid_adapter = BinanceAdapter(
            api_key="invalid_key", api_secret="invalid_secret", testnet=True
        )

        with pytest.raises(ExchangeAuthenticationError):
            await invalid_adapter.connect()

    @pytest.mark.asyncio
    async def test_multiple_symbols(self, adapter):
        """Test operations across multiple trading pairs."""
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

        # Get prices for multiple symbols
        prices = {}
        for symbol in symbols:
            price = await adapter.get_ticker_price(symbol)
            prices[symbol] = price
            assert price > 0

        # Verify we got all prices
        assert len(prices) == len(symbols)


class TestBinanceIntegrationErrorHandling:
    """Integration tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_symbol(self, adapter):
        """Test handling of invalid trading symbol."""
        from bot.interfaces.exchange import ExchangeAPIError

        with pytest.raises(ExchangeAPIError):
            await adapter.get_ticker_price("INVALID_SYMBOL")

    @pytest.mark.asyncio
    async def test_invalid_order_parameters(self, adapter):
        """Test handling of invalid order parameters."""
        from bot.interfaces.exchange import InvalidOrderError

        # Try to create limit order without price
        with pytest.raises(InvalidOrderError):
            await adapter.create_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.001"),
                # Missing price parameter
            )

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, adapter):
        """Test canceling an order that doesn't exist."""
        from bot.interfaces.exchange import OrderNotFoundError

        with pytest.raises((OrderNotFoundError, ExchangeAPIError)):
            await adapter.cancel_order(
                symbol="BTCUSDT", order_id="99999999999"  # Non-existent order ID
            )


if __name__ == "__main__":
    """
    Run integration tests manually.

    Usage:
        export BINANCE_TESTNET_API_KEY="your_key"
        export BINANCE_TESTNET_API_SECRET="your_secret"
        python -m pytest tests/integration/execution/adapters/test_binance_integration.py -v
    """
    print(
        "Run with pytest: pytest tests/integration/execution/adapters/test_binance_integration.py -v"
    )
