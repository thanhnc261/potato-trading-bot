"""
Binance exchange adapter implementation.

This module provides a robust implementation of the ExchangeInterface for Binance,
with support for both testnet and mainnet environments, comprehensive error handling,
rate limiting, and request/response logging.
"""

import asyncio
import hashlib
import hmac
import time
from datetime import datetime
from decimal import Decimal
from typing import Any
from urllib.parse import urlencode

import aiohttp
import structlog

from bot.interfaces.exchange import (
    AccountInfo,
    Balance,
    ExchangeAPIError,
    ExchangeAuthenticationError,
    ExchangeConnectionError,
    ExchangeInterface,
    InsufficientBalanceError,
    InvalidOrderError,
    Order,
    OrderNotFoundError,
    OrderSide,
    OrderStatus,
    OrderType,
    RateLimitExceededError,
    TimeInForce,
    Trade,
)

logger = structlog.get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API requests.

    Implements a token bucket algorithm to respect exchange rate limits
    and prevent hitting API rate limit errors.
    """

    def __init__(self, max_requests: int, time_window: float):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.tokens: float = float(max_requests)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquire a token before making a request.

        This method will block if no tokens are available until a token becomes available.
        """
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.max_requests, self.tokens + (elapsed / self.time_window) * self.max_requests
            )
            self.last_update = now

            # Wait if no tokens available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.time_window / self.max_requests)
                logger.debug("rate_limiter_waiting", wait_time=wait_time)
                await asyncio.sleep(wait_time)
                self.tokens = 1

            self.tokens -= 1


class BinanceAdapter(ExchangeInterface):
    """
    Binance exchange adapter with robust error handling and rate limiting.

    Features:
    - Testnet/mainnet support via configuration
    - Automatic retries with exponential backoff
    - Rate limiting to respect API limits
    - Comprehensive request/response logging
    - Proper error handling and exception mapping
    """

    # API endpoints
    MAINNET_BASE_URL = "https://api.binance.com"
    TESTNET_BASE_URL = "https://testnet.binance.vision"

    # Rate limits (requests per minute)
    DEFAULT_RATE_LIMIT = 1200  # Binance allows 1200 requests per minute
    ORDER_RATE_LIMIT = 100  # Order endpoint has stricter limits

    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 30.0  # seconds
    RETRY_BACKOFF_FACTOR = 2.0

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        recv_window: int = 5000,
        timeout: int = 30,
    ):
        """
        Initialize Binance adapter.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            testnet: Whether to use testnet (True) or mainnet (False)
            recv_window: API request receive window in milliseconds
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.recv_window = recv_window
        self.timeout = timeout

        # Set base URL based on environment
        self.base_url = self.TESTNET_BASE_URL if testnet else self.MAINNET_BASE_URL

        # Initialize rate limiters
        self.general_limiter = RateLimiter(self.DEFAULT_RATE_LIMIT, 60.0)
        self.order_limiter = RateLimiter(self.ORDER_RATE_LIMIT, 60.0)

        # HTTP session
        self.session: aiohttp.ClientSession | None = None

        # Logger with context
        self.logger = logger.bind(
            exchange="binance", environment="testnet" if testnet else "mainnet"
        )

    async def connect(self) -> None:
        """
        Establish connection to Binance.

        Creates an aiohttp session and verifies connectivity and authentication.
        """
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    "X-MBX-APIKEY": self.api_key,
                    "Content-Type": "application/json",
                },
            )

            # Test connectivity
            if not await self.test_connectivity():
                raise ExchangeConnectionError("Failed to connect to Binance")

            # Verify authentication by getting account info
            await self.get_account_info()

            self.logger.info("binance_connected")

        except aiohttp.ClientError as e:
            raise ExchangeConnectionError(f"Failed to create HTTP session: {str(e)}")
        except ExchangeAuthenticationError:
            raise
        except Exception as e:
            raise ExchangeConnectionError(f"Unexpected error during connection: {str(e)}")

    async def disconnect(self) -> None:
        """Close connection to Binance."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("binance_disconnected")

    def _generate_signature(self, params: dict) -> str:
        """
        Generate HMAC SHA256 signature for authenticated requests.

        Args:
            params: Request parameters to sign

        Returns:
            str: Hexadecimal signature
        """
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return signature

    async def _request(
        self,
        method: str,
        endpoint: str,
        signed: bool = False,
        params: dict | None = None,
        use_order_limiter: bool = False,
    ) -> dict:
        """
        Make HTTP request to Binance API with retries and error handling.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            signed: Whether request requires authentication
            params: Request parameters
            use_order_limiter: Whether to use order rate limiter

        Returns:
            Dict: API response

        Raises:
            ExchangeAPIError: If request fails after retries
        """
        if not self.session:
            raise ExchangeConnectionError("Not connected to exchange")

        params = params or {}
        url = f"{self.base_url}{endpoint}"

        # Add timestamp and signature for authenticated requests
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["recvWindow"] = self.recv_window
            params["signature"] = self._generate_signature(params)

        # Apply rate limiting
        limiter = self.order_limiter if use_order_limiter else self.general_limiter
        await limiter.acquire()

        # Retry logic with exponential backoff
        retry_delay = self.INITIAL_RETRY_DELAY
        last_exception = None

        for attempt in range(self.MAX_RETRIES):
            try:
                self.logger.debug(
                    "binance_request",
                    method=method,
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    params=self._sanitize_params(params),
                )

                # Make request
                # For GET requests, use params
                # For POST/DELETE signed requests, use params (query string for signature)
                # For POST/DELETE unsigned requests, use json body
                request_params = params if (method == "GET" or signed) else None
                request_json = params if (method in ["POST", "DELETE"] and not signed) else None

                async with self.session.request(
                    method,
                    url,
                    params=request_params,
                    json=request_json,
                ) as response:
                    response_text = await response.text()

                    # Log response
                    self.logger.debug(
                        "binance_response",
                        status=response.status,
                        endpoint=endpoint,
                        response_length=len(response_text),
                    )

                    # Handle successful response
                    if response.status == 200:
                        try:
                            result: dict[Any, Any] = await response.json()
                            return result
                        except Exception as e:
                            raise ExchangeAPIError(
                                f"Failed to parse response: {str(e)}",
                                status_code=response.status,
                                response={"raw": response_text},
                            )

                    # Parse error response
                    try:
                        error_data = await response.json()
                    except Exception:
                        error_data = {"msg": response_text}

                    error_msg = error_data.get("msg", "Unknown error")
                    error_code = error_data.get("code", response.status)

                    # Map specific error codes to exceptions
                    if response.status == 429 or error_code == -1003:
                        # Rate limit exceeded
                        if attempt < self.MAX_RETRIES - 1:
                            self.logger.warning(
                                "rate_limit_exceeded_retrying",
                                attempt=attempt + 1,
                                retry_delay=retry_delay,
                            )
                            await asyncio.sleep(retry_delay)
                            retry_delay = min(
                                retry_delay * self.RETRY_BACKOFF_FACTOR, self.MAX_RETRY_DELAY
                            )
                            continue
                        raise RateLimitExceededError(f"Rate limit exceeded: {error_msg}")

                    elif response.status == 401 or error_code in [-2014, -2015]:
                        raise ExchangeAuthenticationError(f"Authentication failed: {error_msg}")

                    elif error_code == -2010:
                        raise InsufficientBalanceError(error_msg)

                    elif error_code == -2011:
                        raise OrderNotFoundError(error_msg)

                    elif error_code in [-1021, -1022]:
                        raise InvalidOrderError(error_msg)

                    # Generic API error
                    raise ExchangeAPIError(
                        f"API error: {error_msg}", status_code=response.status, response=error_data
                    )

            except (TimeoutError, aiohttp.ClientError) as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    self.logger.warning(
                        "request_failed_retrying",
                        attempt=attempt + 1,
                        error=str(e),
                        retry_delay=retry_delay,
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * self.RETRY_BACKOFF_FACTOR, self.MAX_RETRY_DELAY)
                    continue

        # All retries exhausted
        raise ExchangeAPIError(
            f"Request failed after {self.MAX_RETRIES} attempts: {str(last_exception)}"
        )

    def _sanitize_params(self, params: dict) -> dict:
        """
        Remove sensitive data from params for logging.

        Args:
            params: Request parameters

        Returns:
            Dict: Sanitized parameters
        """
        sanitized = params.copy()
        if "signature" in sanitized:
            sanitized["signature"] = "***"
        return sanitized

    async def test_connectivity(self) -> bool:
        """Test connectivity to Binance."""
        try:
            await self._request("GET", "/api/v3/ping")
            return True
        except Exception as e:
            self.logger.error("connectivity_test_failed", error=str(e))
            return False

    async def get_account_info(self) -> AccountInfo:
        """Get Binance account information."""
        try:
            response = await self._request("GET", "/api/v3/account", signed=True)

            # Parse balances
            balances = []
            for balance_data in response.get("balances", []):
                free = Decimal(balance_data["free"])
                locked = Decimal(balance_data["locked"])
                total = free + locked

                # Only include assets with non-zero balance
                if total > 0:
                    balances.append(
                        Balance(asset=balance_data["asset"], free=free, locked=locked, total=total)
                    )

            return AccountInfo(
                account_type=response.get("accountType", "SPOT"),
                can_trade=response.get("canTrade", False),
                can_withdraw=response.get("canWithdraw", False),
                can_deposit=response.get("canDeposit", False),
                balances=balances,
                maker_commission=Decimal(response.get("makerCommission", 0)) / 10000,
                taker_commission=Decimal(response.get("takerCommission", 0)) / 10000,
            )

        except ExchangeAPIError:
            raise
        except Exception as e:
            raise ExchangeAPIError(f"Failed to get account info: {str(e)}")

    async def get_balance(self, asset: str | None = None) -> list[Balance]:
        """Get account balance for specific asset or all assets."""
        account_info = await self.get_account_info()

        if asset:
            return [b for b in account_info.balances if b.asset == asset]
        return account_info.balances

    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: str | None = None,
    ) -> Order:
        """Create a new order on Binance."""
        try:
            # Build order parameters
            params = {
                "symbol": symbol,
                "side": side.value.upper(),
                "type": self._map_order_type(order_type),
                "quantity": str(quantity),
            }

            # Add optional parameters
            if order_type in [
                OrderType.LIMIT,
                OrderType.STOP_LOSS_LIMIT,
                OrderType.TAKE_PROFIT_LIMIT,
            ]:
                if price is None:
                    raise InvalidOrderError(f"Price required for {order_type} orders")
                params["price"] = str(price)
                params["timeInForce"] = time_in_force.value

            if order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT]:
                if stop_price is None:
                    raise InvalidOrderError(f"Stop price required for {order_type} orders")
                params["stopPrice"] = str(stop_price)

            if order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT]:
                if stop_price is None:
                    raise InvalidOrderError(f"Stop price required for {order_type} orders")
                params["stopPrice"] = str(stop_price)

            if client_order_id:
                params["newClientOrderId"] = client_order_id

            # Create order
            response = await self._request(
                "POST", "/api/v3/order", signed=True, params=params, use_order_limiter=True
            )

            self.logger.info(
                "order_created",
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                order_id=response["orderId"],
            )

            return self._parse_order(response)

        except (ExchangeAPIError, InvalidOrderError):
            raise
        except Exception as e:
            raise ExchangeAPIError(f"Failed to create order: {str(e)}")

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """Cancel an existing order."""
        try:
            params = {
                "symbol": symbol,
                "orderId": int(order_id),
            }

            response = await self._request(
                "DELETE", "/api/v3/order", signed=True, params=params, use_order_limiter=True
            )

            self.logger.info("order_canceled", symbol=symbol, order_id=order_id)

            return self._parse_order(response)

        except ExchangeAPIError:
            raise
        except Exception as e:
            raise ExchangeAPIError(f"Failed to cancel order: {str(e)}")

    async def get_order(self, symbol: str, order_id: str) -> Order:
        """Get information about a specific order."""
        try:
            params = {
                "symbol": symbol,
                "orderId": int(order_id),
            }

            response = await self._request("GET", "/api/v3/order", signed=True, params=params)

            return self._parse_order(response)

        except ExchangeAPIError:
            raise
        except Exception as e:
            raise ExchangeAPIError(f"Failed to get order: {str(e)}")

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """Get all open orders."""
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol

            response = await self._request("GET", "/api/v3/openOrders", signed=True, params=params)

            return [self._parse_order(order_data) for order_data in response]

        except ExchangeAPIError:
            raise
        except Exception as e:
            raise ExchangeAPIError(f"Failed to get open orders: {str(e)}")

    async def get_order_history(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> list[Order]:
        """Get historical orders."""
        try:
            params = {
                "symbol": symbol,
                "limit": min(limit, 1000),  # Binance max is 1000
            }

            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)

            response = await self._request("GET", "/api/v3/allOrders", signed=True, params=params)

            return [self._parse_order(order_data) for order_data in response]

        except ExchangeAPIError:
            raise
        except Exception as e:
            raise ExchangeAPIError(f"Failed to get order history: {str(e)}")

    async def get_trades(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> list[Trade]:
        """Get trade history."""
        try:
            params = {
                "symbol": symbol,
                "limit": min(limit, 1000),
            }

            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)

            response = await self._request("GET", "/api/v3/myTrades", signed=True, params=params)

            return [self._parse_trade(trade_data) for trade_data in response]

        except ExchangeAPIError:
            raise
        except Exception as e:
            raise ExchangeAPIError(f"Failed to get trades: {str(e)}")

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """Get current ticker price."""
        try:
            params = {"symbol": symbol}

            response = await self._request("GET", "/api/v3/ticker/price", params=params)

            return Decimal(response["price"])

        except ExchangeAPIError:
            raise
        except Exception as e:
            raise ExchangeAPIError(f"Failed to get ticker price: {str(e)}")

    def _map_order_type(self, order_type: OrderType) -> str:
        """
        Map internal order type to Binance order type.

        Args:
            order_type: Internal order type

        Returns:
            str: Binance order type
        """
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_LOSS: "STOP_LOSS",
            OrderType.STOP_LOSS_LIMIT: "STOP_LOSS_LIMIT",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT",
            OrderType.TAKE_PROFIT_LIMIT: "TAKE_PROFIT_LIMIT",
        }
        return mapping.get(order_type, "MARKET")

    def _parse_order_status(self, status: str) -> OrderStatus:
        """
        Parse Binance order status to internal status.

        Args:
            status: Binance order status

        Returns:
            OrderStatus: Internal order status
        """
        mapping = {
            "NEW": OrderStatus.OPEN,
            "FILLED": OrderStatus.CLOSED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "CANCELED": OrderStatus.CANCELED,
            "EXPIRED": OrderStatus.EXPIRED,
            "REJECTED": OrderStatus.REJECTED,
            "PENDING_CANCEL": OrderStatus.PENDING,
        }
        return mapping.get(status, OrderStatus.OPEN)

    def _parse_order(self, data: dict) -> Order:
        """
        Parse Binance order response to Order object.

        Args:
            data: Binance order data

        Returns:
            Order: Parsed order object
        """
        quantity = Decimal(data["origQty"])
        filled_quantity = Decimal(data["executedQty"])

        return Order(
            id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
            symbol=data["symbol"],
            side=OrderSide.BUY if data["side"] == "BUY" else OrderSide.SELL,
            type=self._parse_order_type(data["type"]),
            time_in_force=TimeInForce(data["timeInForce"]) if "timeInForce" in data else None,
            quantity=quantity,
            price=Decimal(data["price"]) if data.get("price") else None,
            stop_price=Decimal(data["stopPrice"]) if data.get("stopPrice") else None,
            status=self._parse_order_status(data["status"]),
            filled_quantity=filled_quantity,
            remaining_quantity=quantity - filled_quantity,
            average_price=(
                Decimal(data["cummulativeQuoteQty"]) / filled_quantity
                if filled_quantity > 0
                else None
            ),
            created_at=(
                datetime.fromtimestamp(data["time"] / 1000) if "time" in data else datetime.now()
            ),
            updated_at=(
                datetime.fromtimestamp(data["updateTime"] / 1000) if "updateTime" in data else None
            ),
            commission=None,  # Commission info not in order response, available in fills
            commission_asset=None,
        )

    def _parse_order_type(self, binance_type: str) -> OrderType:
        """Parse Binance order type to internal type."""
        mapping = {
            "MARKET": OrderType.MARKET,
            "LIMIT": OrderType.LIMIT,
            "STOP_LOSS": OrderType.STOP_LOSS,
            "STOP_LOSS_LIMIT": OrderType.STOP_LOSS_LIMIT,
            "TAKE_PROFIT": OrderType.TAKE_PROFIT,
            "TAKE_PROFIT_LIMIT": OrderType.TAKE_PROFIT_LIMIT,
        }
        return mapping.get(binance_type, OrderType.MARKET)

    def _parse_trade(self, data: dict) -> Trade:
        """
        Parse Binance trade response to Trade object.

        Args:
            data: Binance trade data

        Returns:
            Trade: Parsed trade object
        """
        return Trade(
            id=str(data["id"]),
            order_id=str(data["orderId"]),
            symbol=data["symbol"],
            side=OrderSide.BUY if data["isBuyer"] else OrderSide.SELL,
            price=Decimal(data["price"]),
            quantity=Decimal(data["qty"]),
            commission=Decimal(data["commission"]),
            commission_asset=data["commissionAsset"],
            timestamp=datetime.fromtimestamp(data["time"] / 1000),
            is_maker=data["isMaker"],
        )
