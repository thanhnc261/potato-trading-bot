"""
Unit tests for Binance exchange adapter.

Tests cover:
- Connection and authentication
- Rate limiting
- Error handling and retries
- Order operations
- Account information retrieval
- Response parsing
"""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError, ClientSession

from bot.execution.adapters.binance import BinanceAdapter, RateLimiter
from bot.interfaces.exchange import (
    ExchangeAPIError,
    ExchangeAuthenticationError,
    ExchangeConnectionError,
    InsufficientBalanceError,
    InvalidOrderError,
    OrderNotFoundError,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


class TestRateLimiter:
    """Test rate limiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test that rate limiter allows requests within the limit."""
        limiter = RateLimiter(max_requests=10, time_window=1.0)

        # Should allow first 10 requests without delay
        start_time = asyncio.get_event_loop().time()
        for _ in range(10):
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should complete almost instantly (allow small margin for execution time)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_delays_excess_requests(self):
        """Test that rate limiter delays requests exceeding the limit."""
        limiter = RateLimiter(max_requests=5, time_window=1.0)

        # Use up all tokens
        for _ in range(5):
            await limiter.acquire()

        # Next request should be delayed
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should wait approximately 0.2 seconds (1/5 of time window)
        assert 0.15 < elapsed < 0.3

    @pytest.mark.asyncio
    async def test_rate_limiter_refills_tokens_over_time(self):
        """Test that rate limiter refills tokens over time."""
        limiter = RateLimiter(max_requests=10, time_window=1.0)

        # Use up all tokens
        for _ in range(10):
            await limiter.acquire()

        # Wait for half the time window
        await asyncio.sleep(0.5)

        # Should be able to make ~5 more requests without much delay
        start_time = asyncio.get_event_loop().time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = asyncio.get_event_loop().time() - start_time

        assert elapsed < 0.2


class TestBinanceAdapter:
    """Test Binance adapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create a Binance adapter instance for testing."""
        return BinanceAdapter(api_key="test_api_key", api_secret="test_api_secret", testnet=True)

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp session."""
        session = AsyncMock(spec=ClientSession)
        return session

    def test_initialization_testnet(self, adapter):
        """Test adapter initialization with testnet."""
        assert adapter.api_key == "test_api_key"
        assert adapter.api_secret == "test_api_secret"
        assert adapter.testnet is True
        assert adapter.base_url == BinanceAdapter.TESTNET_BASE_URL

    def test_initialization_mainnet(self):
        """Test adapter initialization with mainnet."""
        adapter = BinanceAdapter(
            api_key="test_api_key", api_secret="test_api_secret", testnet=False
        )
        assert adapter.testnet is False
        assert adapter.base_url == BinanceAdapter.MAINNET_BASE_URL

    def test_generate_signature(self, adapter):
        """Test HMAC signature generation."""
        params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "quantity": "1.0",
            "price": "50000.0",
            "timestamp": 1234567890,
        }

        signature = adapter._generate_signature(params)

        # Signature should be a 64-character hexadecimal string
        assert isinstance(signature, str)
        assert len(signature) == 64
        assert all(c in "0123456789abcdef" for c in signature)

    def test_sanitize_params(self, adapter):
        """Test parameter sanitization for logging."""
        params = {
            "symbol": "BTCUSDT",
            "signature": "secret_signature_value",
            "timestamp": 1234567890,
        }

        sanitized = adapter._sanitize_params(params)

        assert sanitized["symbol"] == "BTCUSDT"
        assert sanitized["timestamp"] == 1234567890
        assert sanitized["signature"] == "***"

    @pytest.mark.asyncio
    async def test_connect_success(self, adapter, mock_session):
        """Test successful connection to Binance."""
        # Mock session creation
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Mock test_connectivity
            with patch.object(adapter, "test_connectivity", return_value=True):
                # Mock get_account_info
                mock_account_info = MagicMock()
                with patch.object(adapter, "get_account_info", return_value=mock_account_info):
                    await adapter.connect()

                    assert adapter.session is not None

    @pytest.mark.asyncio
    async def test_connect_failure_connectivity(self, adapter, mock_session):
        """Test connection failure due to connectivity issues."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(adapter, "test_connectivity", return_value=False):
                with pytest.raises(ExchangeConnectionError, match="Failed to connect"):
                    await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_failure_authentication(self, adapter, mock_session):
        """Test connection failure due to authentication issues."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            with patch.object(adapter, "test_connectivity", return_value=True):
                with patch.object(
                    adapter,
                    "get_account_info",
                    side_effect=ExchangeAuthenticationError("Invalid API key"),
                ):
                    with pytest.raises(ExchangeAuthenticationError):
                        await adapter.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self, adapter, mock_session):
        """Test disconnection from Binance."""
        adapter.session = mock_session
        await adapter.disconnect()

        mock_session.close.assert_called_once()
        assert adapter.session is None

    @pytest.mark.asyncio
    async def test_request_success(self, adapter, mock_session):
        """Test successful API request."""
        adapter.session = mock_session

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"price": "50000.0"})
        mock_response.text = AsyncMock(return_value='{"price": "50000.0"}')

        mock_session.request = AsyncMock(return_value=mock_response)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        result = await adapter._request("GET", "/api/v3/ticker/price")

        assert result == {"price": "50000.0"}

    @pytest.mark.asyncio
    async def test_request_retry_on_network_error(self, adapter, mock_session):
        """Test request retry on network error."""
        adapter.session = mock_session

        # First two attempts fail, third succeeds
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.text = AsyncMock(return_value='{"success": true}')
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ClientError("Network error")
            return mock_response

        mock_session.request = mock_request

        result = await adapter._request("GET", "/api/v3/ping")

        assert result == {"success": True}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_request_max_retries_exceeded(self, adapter, mock_session):
        """Test request failure after max retries."""
        adapter.session = mock_session

        # All attempts fail
        mock_session.request = AsyncMock(side_effect=ClientError("Network error"))

        with pytest.raises(ExchangeAPIError, match="Request failed after"):
            await adapter._request("GET", "/api/v3/ping")

    @pytest.mark.asyncio
    async def test_request_rate_limit_error_with_retry(self, adapter, mock_session):
        """Test request retry on rate limit error."""
        adapter.session = mock_session

        # First request hits rate limit, second succeeds
        rate_limit_response = AsyncMock()
        rate_limit_response.status = 429
        rate_limit_response.json = AsyncMock(
            return_value={"code": -1003, "msg": "Too many requests"}
        )
        rate_limit_response.__aenter__ = AsyncMock(return_value=rate_limit_response)
        rate_limit_response.__aexit__ = AsyncMock(return_value=None)

        success_response = AsyncMock()
        success_response.status = 200
        success_response.json = AsyncMock(return_value={"success": True})
        success_response.text = AsyncMock(return_value='{"success": true}')
        success_response.__aenter__ = AsyncMock(return_value=success_response)
        success_response.__aexit__ = AsyncMock(return_value=None)

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return rate_limit_response
            return success_response

        mock_session.request = mock_request

        result = await adapter._request("GET", "/api/v3/ping")

        assert result == {"success": True}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_request_authentication_error(self, adapter, mock_session):
        """Test authentication error handling."""
        adapter.session = mock_session

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value={"code": -2015, "msg": "Invalid API key"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(ExchangeAuthenticationError, match="Authentication failed"):
            await adapter._request("GET", "/api/v3/account", signed=True)

    @pytest.mark.asyncio
    async def test_request_insufficient_balance_error(self, adapter, mock_session):
        """Test insufficient balance error handling."""
        adapter.session = mock_session

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"code": -2010, "msg": "Insufficient balance"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(InsufficientBalanceError):
            await adapter._request("POST", "/api/v3/order", signed=True)

    @pytest.mark.asyncio
    async def test_request_order_not_found_error(self, adapter, mock_session):
        """Test order not found error handling."""
        adapter.session = mock_session

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"code": -2011, "msg": "Order not found"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(OrderNotFoundError):
            await adapter._request("DELETE", "/api/v3/order", signed=True)

    @pytest.mark.asyncio
    async def test_request_invalid_order_error(self, adapter, mock_session):
        """Test invalid order error handling."""
        adapter.session = mock_session

        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.json = AsyncMock(return_value={"code": -1021, "msg": "Invalid timestamp"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(InvalidOrderError):
            await adapter._request("POST", "/api/v3/order", signed=True)

    @pytest.mark.asyncio
    async def test_get_account_info(self, adapter):
        """Test getting account information."""
        mock_response = {
            "accountType": "SPOT",
            "canTrade": True,
            "canWithdraw": True,
            "canDeposit": True,
            "balances": [
                {"asset": "BTC", "free": "1.5", "locked": "0.5"},
                {"asset": "USDT", "free": "10000.0", "locked": "0.0"},
                {"asset": "ETH", "free": "0.0", "locked": "0.0"},  # Should be filtered out
            ],
            "makerCommission": 10,
            "takerCommission": 10,
        }

        with patch.object(adapter, "_request", return_value=mock_response):
            account_info = await adapter.get_account_info()

            assert account_info.account_type == "SPOT"
            assert account_info.can_trade is True
            assert account_info.can_withdraw is True
            assert len(account_info.balances) == 2  # ETH filtered out
            assert account_info.balances[0].asset == "BTC"
            assert account_info.balances[0].free == Decimal("1.5")
            assert account_info.balances[0].locked == Decimal("0.5")
            assert account_info.balances[0].total == Decimal("2.0")
            assert account_info.maker_commission == Decimal("0.001")
            assert account_info.taker_commission == Decimal("0.001")

    @pytest.mark.asyncio
    async def test_get_balance_specific_asset(self, adapter):
        """Test getting balance for specific asset."""
        mock_account_info = MagicMock()
        mock_account_info.balances = [
            MagicMock(
                asset="BTC", free=Decimal("1.5"), locked=Decimal("0.5"), total=Decimal("2.0")
            ),
            MagicMock(
                asset="USDT",
                free=Decimal("10000.0"),
                locked=Decimal("0.0"),
                total=Decimal("10000.0"),
            ),
        ]

        with patch.object(adapter, "get_account_info", return_value=mock_account_info):
            balances = await adapter.get_balance("BTC")

            assert len(balances) == 1
            assert balances[0].asset == "BTC"

    @pytest.mark.asyncio
    async def test_create_market_order(self, adapter):
        """Test creating a market order."""
        mock_response = {
            "orderId": 12345,
            "clientOrderId": "test_order_1",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "MARKET",
            "status": "FILLED",
            "origQty": "0.1",
            "executedQty": "0.1",
            "cummulativeQuoteQty": "5000.0",
            "time": 1234567890000,
            "updateTime": 1234567891000,
        }

        with patch.object(adapter, "_request", return_value=mock_response):
            order = await adapter.create_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
            )

            assert order.id == "12345"
            assert order.symbol == "BTCUSDT"
            assert order.side == OrderSide.BUY
            assert order.type == OrderType.MARKET
            assert order.quantity == Decimal("0.1")
            assert order.status == OrderStatus.CLOSED

    @pytest.mark.asyncio
    async def test_create_limit_order(self, adapter):
        """Test creating a limit order."""
        mock_response = {
            "orderId": 12346,
            "symbol": "BTCUSDT",
            "side": "SELL",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "price": "55000.0",
            "status": "NEW",
            "origQty": "0.1",
            "executedQty": "0.0",
            "cummulativeQuoteQty": "0.0",
            "time": 1234567890000,
        }

        with patch.object(adapter, "_request", return_value=mock_response):
            order = await adapter.create_order(
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
                price=Decimal("55000.0"),
                time_in_force=TimeInForce.GTC,
            )

            assert order.id == "12346"
            assert order.type == OrderType.LIMIT
            assert order.price == Decimal("55000.0")
            assert order.time_in_force == TimeInForce.GTC
            assert order.status == OrderStatus.OPEN

    @pytest.mark.asyncio
    async def test_create_limit_order_without_price_raises_error(self, adapter):
        """Test that creating limit order without price raises error."""
        with pytest.raises(InvalidOrderError, match="Price required"):
            await adapter.create_order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("0.1"),
            )

    @pytest.mark.asyncio
    async def test_cancel_order(self, adapter):
        """Test canceling an order."""
        mock_response = {
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "status": "CANCELED",
            "origQty": "0.1",
            "executedQty": "0.0",
            "cummulativeQuoteQty": "0.0",
            "time": 1234567890000,
            "updateTime": 1234567891000,
        }

        with patch.object(adapter, "_request", return_value=mock_response):
            order = await adapter.cancel_order(symbol="BTCUSDT", order_id="12345")

            assert order.id == "12345"
            assert order.status == OrderStatus.CANCELED

    @pytest.mark.asyncio
    async def test_get_order(self, adapter):
        """Test getting order information."""
        mock_response = {
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "price": "50000.0",
            "status": "NEW",
            "origQty": "0.1",
            "executedQty": "0.0",
            "cummulativeQuoteQty": "0.0",
            "time": 1234567890000,
        }

        with patch.object(adapter, "_request", return_value=mock_response):
            order = await adapter.get_order(symbol="BTCUSDT", order_id="12345")

            assert order.id == "12345"
            assert order.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_open_orders(self, adapter):
        """Test getting open orders."""
        mock_response = [
            {
                "orderId": 12345,
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "status": "NEW",
                "origQty": "0.1",
                "executedQty": "0.0",
                "cummulativeQuoteQty": "0.0",
                "time": 1234567890000,
            },
            {
                "orderId": 12346,
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "status": "NEW",
                "origQty": "1.0",
                "executedQty": "0.0",
                "cummulativeQuoteQty": "0.0",
                "time": 1234567891000,
            },
        ]

        with patch.object(adapter, "_request", return_value=mock_response):
            orders = await adapter.get_open_orders()

            assert len(orders) == 2
            assert orders[0].id == "12345"
            assert orders[1].id == "12346"

    @pytest.mark.asyncio
    async def test_get_ticker_price(self, adapter):
        """Test getting ticker price."""
        mock_response = {"symbol": "BTCUSDT", "price": "50000.0"}

        with patch.object(adapter, "_request", return_value=mock_response):
            price = await adapter.get_ticker_price("BTCUSDT")

            assert price == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_test_connectivity_success(self, adapter):
        """Test connectivity check success."""
        with patch.object(adapter, "_request", return_value={}):
            result = await adapter.test_connectivity()

            assert result is True

    @pytest.mark.asyncio
    async def test_test_connectivity_failure(self, adapter):
        """Test connectivity check failure."""
        with patch.object(adapter, "_request", side_effect=ExchangeAPIError("Connection failed")):
            result = await adapter.test_connectivity()

            assert result is False

    def test_map_order_type(self, adapter):
        """Test order type mapping."""
        assert adapter._map_order_type(OrderType.MARKET) == "MARKET"
        assert adapter._map_order_type(OrderType.LIMIT) == "LIMIT"
        assert adapter._map_order_type(OrderType.STOP_LOSS) == "STOP_LOSS"

    def test_parse_order_status(self, adapter):
        """Test order status parsing."""
        assert adapter._parse_order_status("NEW") == OrderStatus.OPEN
        assert adapter._parse_order_status("FILLED") == OrderStatus.CLOSED
        assert adapter._parse_order_status("CANCELED") == OrderStatus.CANCELED
        assert adapter._parse_order_status("PARTIALLY_FILLED") == OrderStatus.PARTIALLY_FILLED
