"""
Base exchange interface for trading bot.

This module defines the abstract interface that all exchange adapters must implement.
It provides a consistent API for interacting with different cryptocurrency exchanges.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LOSS_LIMIT = "stop_loss_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderStatus(str, Enum):
    """Order status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"


class TimeInForce(str, Enum):
    """Time in force enumeration."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill


class Balance(BaseModel):
    """Account balance for a specific asset."""
    asset: str = Field(..., description="Asset symbol (e.g., 'BTC', 'USDT')")
    free: Decimal = Field(..., description="Available balance for trading")
    locked: Decimal = Field(..., description="Balance locked in open orders")
    total: Decimal = Field(..., description="Total balance (free + locked)")


class Order(BaseModel):
    """Representation of an exchange order."""
    id: str = Field(..., description="Exchange order ID")
    client_order_id: Optional[str] = Field(None, description="Client-generated order ID")
    symbol: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')")
    side: OrderSide = Field(..., description="Order side (buy/sell)")
    type: OrderType = Field(..., description="Order type")
    time_in_force: Optional[TimeInForce] = Field(None, description="Time in force")
    quantity: Decimal = Field(..., description="Order quantity")
    price: Optional[Decimal] = Field(None, description="Order price (None for market orders)")
    stop_price: Optional[Decimal] = Field(None, description="Stop price for stop orders")
    status: OrderStatus = Field(..., description="Current order status")
    filled_quantity: Decimal = Field(Decimal(0), description="Filled quantity")
    remaining_quantity: Decimal = Field(..., description="Remaining quantity")
    average_price: Optional[Decimal] = Field(None, description="Average fill price")
    created_at: datetime = Field(..., description="Order creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    commission: Optional[Decimal] = Field(None, description="Trading commission")
    commission_asset: Optional[str] = Field(None, description="Commission asset")


class Trade(BaseModel):
    """Representation of an executed trade."""
    id: str = Field(..., description="Trade ID")
    order_id: str = Field(..., description="Related order ID")
    symbol: str = Field(..., description="Trading pair symbol")
    side: OrderSide = Field(..., description="Trade side")
    price: Decimal = Field(..., description="Execution price")
    quantity: Decimal = Field(..., description="Executed quantity")
    commission: Decimal = Field(..., description="Trading commission")
    commission_asset: str = Field(..., description="Commission asset")
    timestamp: datetime = Field(..., description="Trade execution timestamp")
    is_maker: bool = Field(..., description="Whether this trade was a maker trade")


class AccountInfo(BaseModel):
    """Exchange account information."""
    account_type: str = Field(..., description="Account type (e.g., 'SPOT', 'MARGIN')")
    can_trade: bool = Field(..., description="Whether trading is enabled")
    can_withdraw: bool = Field(..., description="Whether withdrawals are enabled")
    can_deposit: bool = Field(..., description="Whether deposits are enabled")
    balances: List[Balance] = Field(default_factory=list, description="Account balances")
    maker_commission: Optional[Decimal] = Field(None, description="Maker commission rate")
    taker_commission: Optional[Decimal] = Field(None, description="Taker commission rate")


class ExchangeInterface(ABC):
    """
    Abstract base interface for exchange adapters.

    All exchange implementations must inherit from this class and implement
    all abstract methods to ensure consistent behavior across different exchanges.
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        Establish connection to the exchange.

        This method should initialize any necessary connections, authenticate,
        and verify that the exchange is accessible.

        Raises:
            ExchangeConnectionError: If connection fails
            ExchangeAuthenticationError: If authentication fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Close connection to the exchange.

        This method should properly close all connections and release resources.
        """
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """
        Get account information including permissions and balances.

        Returns:
            AccountInfo: Account information with balances

        Raises:
            ExchangeAPIError: If API request fails
        """
        pass

    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> List[Balance]:
        """
        Get account balance for one or all assets.

        Args:
            asset: Specific asset to query (e.g., 'BTC'). If None, returns all balances.

        Returns:
            List[Balance]: List of balance information

        Raises:
            ExchangeAPIError: If API request fails
        """
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
    ) -> Order:
        """
        Create a new order on the exchange.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side (buy/sell)
            order_type: Type of order
            quantity: Order quantity
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Time in force policy
            client_order_id: Optional client-generated order ID

        Returns:
            Order: Created order information

        Raises:
            ExchangeAPIError: If order creation fails
            InvalidOrderError: If order parameters are invalid
        """
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """
        Cancel an existing order.

        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID to cancel

        Returns:
            Order: Canceled order information

        Raises:
            ExchangeAPIError: If cancellation fails
            OrderNotFoundError: If order doesn't exist
        """
        pass

    @abstractmethod
    async def get_order(self, symbol: str, order_id: str) -> Order:
        """
        Get information about a specific order.

        Args:
            symbol: Trading pair symbol
            order_id: Exchange order ID

        Returns:
            Order: Order information

        Raises:
            ExchangeAPIError: If request fails
            OrderNotFoundError: If order doesn't exist
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders for a symbol or all symbols.

        Args:
            symbol: Trading pair symbol. If None, returns orders for all symbols.

        Returns:
            List[Order]: List of open orders

        Raises:
            ExchangeAPIError: If request fails
        """
        pass

    @abstractmethod
    async def get_order_history(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Order]:
        """
        Get historical orders for a symbol.

        Args:
            symbol: Trading pair symbol
            start_time: Start time for query
            end_time: End time for query
            limit: Maximum number of orders to return

        Returns:
            List[Order]: List of historical orders

        Raises:
            ExchangeAPIError: If request fails
        """
        pass

    @abstractmethod
    async def get_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Trade]:
        """
        Get trade history for a symbol.

        Args:
            symbol: Trading pair symbol
            start_time: Start time for query
            end_time: End time for query
            limit: Maximum number of trades to return

        Returns:
            List[Trade]: List of executed trades

        Raises:
            ExchangeAPIError: If request fails
        """
        pass

    @abstractmethod
    async def get_ticker_price(self, symbol: str) -> Decimal:
        """
        Get current ticker price for a symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Decimal: Current price

        Raises:
            ExchangeAPIError: If request fails
        """
        pass

    @abstractmethod
    async def test_connectivity(self) -> bool:
        """
        Test connectivity to the exchange.

        Returns:
            bool: True if exchange is reachable, False otherwise
        """
        pass


# Custom exceptions for exchange operations
class ExchangeError(Exception):
    """Base exception for exchange-related errors."""
    pass


class ExchangeConnectionError(ExchangeError):
    """Raised when connection to exchange fails."""
    pass


class ExchangeAuthenticationError(ExchangeError):
    """Raised when authentication with exchange fails."""
    pass


class ExchangeAPIError(ExchangeError):
    """Raised when API request fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class InvalidOrderError(ExchangeError):
    """Raised when order parameters are invalid."""
    pass


class OrderNotFoundError(ExchangeError):
    """Raised when order is not found."""
    pass


class InsufficientBalanceError(ExchangeError):
    """Raised when account has insufficient balance."""
    pass


class RateLimitExceededError(ExchangeError):
    """Raised when rate limit is exceeded."""
    pass
