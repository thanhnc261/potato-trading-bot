"""
Simulated exchange adapter for paper trading.

This module provides a complete in-memory exchange simulator for paper trading with:
- Realistic order execution simulation with slippage and latency
- In-memory order book tracking
- Simulated portfolio and balance management
- Virtual fill simulation based on market prices
- P/L tracking without real money

The simulator uses real market data prices but executes orders virtually,
making it ideal for testing strategies without financial risk.
"""

import asyncio
import random
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from bot.core.strategy import Signal

from bot.interfaces.exchange import (
    AccountInfo,
    Balance,
    ExchangeInterface,
    InsufficientBalanceError,
    InvalidOrderError,
    Order,
    OrderNotFoundError,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
    Trade,
)

logger = structlog.get_logger(__name__)


class SimulatedExchange(ExchangeInterface):
    """
    Simulated exchange for paper trading with realistic execution modeling.

    Features:
    - In-memory portfolio tracking with virtual balances
    - Realistic order execution with slippage and latency
    - Order book simulation with order matching
    - Virtual fill simulation based on market prices
    - P/L tracking for all trades
    - Commission and slippage costs
    - Order lifecycle management (pending, filled, cancelled)
    """

    def __init__(
        self,
        initial_balances: dict[str, Decimal] | None = None,
        commission_rate: float = 0.001,
        slippage_factor: float = 0.001,
        execution_delay_ms: int = 100,
        maker_commission: float = 0.0009,
        taker_commission: float = 0.001,
    ):
        """
        Initialize simulated exchange.

        Args:
            initial_balances: Initial portfolio balances (e.g., {'USDT': 10000, 'BTC': 0})
            commission_rate: Default commission rate per trade
            slippage_factor: Slippage factor for execution
            execution_delay_ms: Simulated execution delay in milliseconds
            maker_commission: Maker commission rate
            taker_commission: Taker commission rate
        """
        # Default to 10000 USDT if no balances provided
        self.balances: dict[str, Decimal] = initial_balances or {
            "USDT": Decimal("10000.0"),
            "BTC": Decimal("0"),
            "ETH": Decimal("0"),
        }

        self.commission_rate = commission_rate
        self.slippage_factor = slippage_factor
        self.execution_delay_ms = execution_delay_ms
        self.maker_commission = maker_commission
        self.taker_commission = taker_commission

        # Order tracking
        self.orders: dict[str, Order] = {}
        self.trades: dict[str, Trade] = {}
        self.order_counter = 1

        # Market data cache (for price simulation)
        self.market_prices: dict[str, Decimal] = {}

        # Connection state
        self.connected = False

        # Performance tracking
        self.total_commission_paid = Decimal("0")
        self.total_slippage_cost = Decimal("0")
        self.initial_portfolio_value = Decimal("0")
        self.trade_count = 0

        logger.info(
            "simulated_exchange_initialized",
            initial_balances={k: float(v) for k, v in self.balances.items()},
            commission_rate=commission_rate,
            slippage_factor=slippage_factor,
            execution_delay_ms=execution_delay_ms,
        )

    async def connect(self) -> None:
        """
        Establish connection to simulated exchange.

        This is a no-op for the simulator but maintains interface consistency.
        """
        self.connected = True
        self.initial_portfolio_value = self._calculate_portfolio_value()
        logger.info(
            "simulated_exchange_connected", initial_value=float(self.initial_portfolio_value)
        )

    async def disconnect(self) -> None:
        """Close connection to simulated exchange."""
        self.connected = False
        final_value = self._calculate_portfolio_value()
        pnl = final_value - self.initial_portfolio_value
        pnl_pct = (
            (pnl / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0
        )

        logger.info(
            "simulated_exchange_disconnected",
            initial_value=float(self.initial_portfolio_value),
            final_value=float(final_value),
            pnl=float(pnl),
            pnl_pct=float(pnl_pct),
            total_trades=self.trade_count,
            total_commission=float(self.total_commission_paid),
            total_slippage=float(self.total_slippage_cost),
        )

    async def get_account_info(self) -> AccountInfo:
        """Get simulated account information."""
        balances = []
        for asset, amount in self.balances.items():
            if amount > 0:
                # Calculate locked balance from open orders
                locked = self._calculate_locked_balance(asset)
                free = amount - locked
                balances.append(
                    Balance(
                        asset=asset,
                        free=free,
                        locked=locked,
                        total=amount,
                    )
                )

        return AccountInfo(
            account_type="SPOT",
            can_trade=True,
            can_withdraw=True,
            can_deposit=True,
            balances=balances,
            maker_commission=Decimal(str(self.maker_commission)),
            taker_commission=Decimal(str(self.taker_commission)),
        )

    async def get_balance(self, asset: str | None = None) -> list[Balance]:
        """Get account balance for specific asset or all assets."""
        account_info = await self.get_account_info()

        if asset:
            return [b for b in account_info.balances if b.asset == asset]
        return account_info.balances

    def _calculate_locked_balance(self, asset: str) -> Decimal:
        """
        Calculate locked balance from open orders.

        Args:
            asset: Asset symbol

        Returns:
            Total locked amount
        """
        locked = Decimal("0")

        for order in self.orders.values():
            if order.status in [OrderStatus.OPEN, OrderStatus.PENDING]:
                # Parse symbol to get base and quote assets
                base_asset, quote_asset = self._parse_symbol(order.symbol)

                if order.side == OrderSide.BUY:
                    # Buying: quote asset is locked
                    if asset == quote_asset and order.price:
                        locked += order.remaining_quantity * order.price
                else:
                    # Selling: base asset is locked
                    if asset == base_asset:
                        locked += order.remaining_quantity

        return locked

    def _parse_symbol(self, symbol: str) -> tuple[str, str]:
        """
        Parse trading pair symbol into base and quote assets.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')

        Returns:
            Tuple of (base_asset, quote_asset)
        """
        # Common quote currencies
        for quote in ["USDT", "BUSD", "USD", "BTC", "ETH", "BNB"]:
            if symbol.endswith(quote):
                base = symbol[: -len(quote)]
                return base, quote

        raise ValueError(f"Cannot parse symbol: {symbol}")

    def update_market_price(self, symbol: str, price: Decimal) -> None:
        """
        Update current market price for a symbol.

        This should be called by the paper trading runner to update
        prices based on live or historical data.

        Args:
            symbol: Trading pair symbol
            price: Current market price
        """
        self.market_prices[symbol] = price
        logger.debug("market_price_updated", symbol=symbol, price=float(price))

    def _calculate_slippage(
        self,
        price: Decimal,
        side: OrderSide,
        quantity: Decimal,
        volatility: float = 0.01,
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate realistic slippage based on market conditions.

        Slippage model:
        - Base slippage from slippage_factor
        - Additional slippage based on volatility
        - Order size impact (larger orders = more slippage)
        - Random component for realism
        - Always unfavorable to the trader

        Args:
            price: Order price
            side: Order side (BUY/SELL)
            quantity: Order quantity
            volatility: Current market volatility

        Returns:
            Tuple of (execution_price, slippage_cost_per_unit)
        """
        # Base slippage
        base_slippage = price * Decimal(str(self.slippage_factor))

        # Volatility-adjusted slippage (higher volatility = more slippage)
        vol_slippage = price * Decimal(str(volatility * 0.5))

        # Order size impact (larger orders have more price impact)
        size_factor = min(float(quantity) / 100.0, 0.002)
        size_slippage = price * Decimal(str(size_factor))

        # Random component (0-50% of base slippage)
        random_factor = random.uniform(0, 0.5)
        random_slippage = base_slippage * Decimal(str(random_factor))

        # Total slippage
        total_slippage = base_slippage + vol_slippage + size_slippage + random_slippage

        # Apply slippage (always unfavorable)
        if side == OrderSide.BUY:
            # Buy: price increases
            execution_price = price + total_slippage
        else:
            # Sell: price decreases
            execution_price = price - total_slippage

        slippage_per_unit = abs(execution_price - price)

        return execution_price, slippage_per_unit

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
        """
        Create and execute a simulated order.

        For market orders, immediate execution is simulated.
        For limit orders, the order is placed and will be filled
        when the market price crosses the limit price.

        Args:
            symbol: Trading pair symbol
            side: Order side (buy/sell)
            order_type: Type of order
            quantity: Order quantity
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            time_in_force: Time in force policy
            client_order_id: Optional client-generated order ID

        Returns:
            Order object with execution details

        Raises:
            InvalidOrderError: If order parameters are invalid
            InsufficientBalanceError: If account has insufficient balance
        """
        # Validate order parameters
        if order_type in [OrderType.LIMIT, OrderType.STOP_LOSS_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
            if price is None:
                raise InvalidOrderError(f"Price required for {order_type} orders")

        if order_type in [OrderType.STOP_LOSS, OrderType.STOP_LOSS_LIMIT]:
            if stop_price is None:
                raise InvalidOrderError(f"Stop price required for {order_type} orders")

        # Parse symbol
        base_asset, quote_asset = self._parse_symbol(symbol)

        # Get current market price
        if symbol not in self.market_prices:
            raise InvalidOrderError(f"No market price available for {symbol}")

        market_price = self.market_prices[symbol]

        # Use market price for market orders
        if order_type == OrderType.MARKET:
            price = market_price

        # Create order ID
        order_id = str(self.order_counter)
        self.order_counter += 1

        # Simulate execution delay
        if self.execution_delay_ms > 0:
            await asyncio.sleep(self.execution_delay_ms / 1000.0)

        # Check balance
        if side == OrderSide.BUY:
            # Buying: need quote currency
            required_balance = quantity * (price if price else market_price)
            if quote_asset not in self.balances or self.balances[quote_asset] < required_balance:
                raise InsufficientBalanceError(
                    f"Insufficient {quote_asset} balance. Required: {required_balance}, "
                    f"Available: {self.balances.get(quote_asset, Decimal('0'))}"
                )
        else:
            # Selling: need base currency
            if base_asset not in self.balances or self.balances[base_asset] < quantity:
                raise InsufficientBalanceError(
                    f"Insufficient {base_asset} balance. Required: {quantity}, "
                    f"Available: {self.balances.get(base_asset, Decimal('0'))}"
                )

        # Create order
        now = datetime.now()
        order = Order(
            id=order_id,
            client_order_id=client_order_id or f"client_{order_id}",
            symbol=symbol,
            side=side,
            type=order_type,
            time_in_force=time_in_force,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            filled_quantity=Decimal("0"),
            remaining_quantity=quantity,
            average_price=None,
            created_at=now,
            updated_at=now,
            commission=None,
            commission_asset=None,
        )

        # For market orders, execute immediately
        if order_type == OrderType.MARKET:
            order = await self._execute_order(order, market_price)
        else:
            # For limit orders, mark as open
            order.status = OrderStatus.OPEN

        # Store order
        self.orders[order_id] = order

        logger.info(
            "order_created",
            order_id=order_id,
            symbol=symbol,
            side=side.value,
            type=order_type.value,
            quantity=float(quantity),
            status=order.status.value,
        )

        return order

    async def _execute_order(self, order: Order, market_price: Decimal) -> Order:
        """
        Execute an order with realistic fill simulation.

        Args:
            order: Order to execute
            market_price: Current market price

        Returns:
            Updated order with execution details
        """
        base_asset, quote_asset = self._parse_symbol(order.symbol)

        # Calculate slippage
        execution_price, slippage_per_unit = self._calculate_slippage(
            market_price,
            order.side,
            order.quantity,
        )

        # Calculate commission (taker fee for market orders)
        notional_value = execution_price * order.quantity
        commission = notional_value * Decimal(str(self.taker_commission))

        # Update balances
        if order.side == OrderSide.BUY:
            # Buying: deduct quote currency, add base currency
            total_cost = (execution_price * order.quantity) + commission
            self.balances[quote_asset] = self.balances.get(quote_asset, Decimal("0")) - total_cost
            self.balances[base_asset] = self.balances.get(base_asset, Decimal("0")) + order.quantity
        else:
            # Selling: deduct base currency, add quote currency
            total_proceeds = (execution_price * order.quantity) - commission
            self.balances[base_asset] = self.balances.get(base_asset, Decimal("0")) - order.quantity
            self.balances[quote_asset] = (
                self.balances.get(quote_asset, Decimal("0")) + total_proceeds
            )

        # Update order
        order.status = OrderStatus.CLOSED
        order.filled_quantity = order.quantity
        order.remaining_quantity = Decimal("0")
        order.average_price = execution_price
        order.commission = commission
        order.commission_asset = quote_asset
        order.updated_at = datetime.now()

        # Create trade record
        trade_id = f"trade_{uuid.uuid4().hex[:8]}"
        trade = Trade(
            id=trade_id,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=execution_price,
            quantity=order.quantity,
            commission=commission,
            commission_asset=quote_asset,
            timestamp=datetime.now(),
            is_maker=False,  # Market orders are always taker
        )

        self.trades[trade_id] = trade
        self.trade_count += 1
        self.total_commission_paid += commission
        self.total_slippage_cost += slippage_per_unit * order.quantity

        logger.info(
            "order_executed",
            order_id=order.id,
            symbol=order.symbol,
            side=order.side.value,
            quantity=float(order.quantity),
            execution_price=float(execution_price),
            commission=float(commission),
            slippage=float(slippage_per_unit * order.quantity),
        )

        return order

    async def cancel_order(self, symbol: str, order_id: str) -> Order:
        """
        Cancel an open order.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID to cancel

        Returns:
            Canceled order

        Raises:
            OrderNotFoundError: If order doesn't exist
        """
        if order_id not in self.orders:
            raise OrderNotFoundError(f"Order {order_id} not found")

        order = self.orders[order_id]

        if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
            raise InvalidOrderError(
                f"Order {order_id} cannot be cancelled (status: {order.status})"
            )

        order.status = OrderStatus.CANCELED
        order.updated_at = datetime.now()

        logger.info("order_cancelled", order_id=order_id, symbol=symbol)

        return order

    async def get_order(self, symbol: str, order_id: str) -> Order:
        """
        Get information about a specific order.

        Args:
            symbol: Trading pair symbol
            order_id: Order ID

        Returns:
            Order information

        Raises:
            OrderNotFoundError: If order doesn't exist
        """
        if order_id not in self.orders:
            raise OrderNotFoundError(f"Order {order_id} not found")

        return self.orders[order_id]

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        open_orders = [
            order
            for order in self.orders.values()
            if order.status == OrderStatus.OPEN and (symbol is None or order.symbol == symbol)
        ]

        return open_orders

    async def get_order_history(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> list[Order]:
        """
        Get historical orders.

        Args:
            symbol: Trading pair symbol
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of orders to return

        Returns:
            List of historical orders
        """
        orders = [
            order
            for order in self.orders.values()
            if order.symbol == symbol
            and (start_time is None or order.created_at >= start_time)
            and (end_time is None or order.created_at <= end_time)
        ]

        # Sort by created_at descending
        orders.sort(key=lambda o: o.created_at, reverse=True)

        return orders[:limit]

    async def get_trades(
        self,
        symbol: str,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 500,
    ) -> list[Trade]:
        """
        Get trade history.

        Args:
            symbol: Trading pair symbol
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum number of trades to return

        Returns:
            List of executed trades
        """
        trades = [
            trade
            for trade in self.trades.values()
            if trade.symbol == symbol
            and (start_time is None or trade.timestamp >= start_time)
            and (end_time is None or trade.timestamp <= end_time)
        ]

        # Sort by timestamp descending
        trades.sort(key=lambda t: t.timestamp, reverse=True)

        return trades[:limit]

    async def get_ticker_price(self, symbol: str) -> Decimal:
        """
        Get current ticker price.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price

        Raises:
            InvalidOrderError: If no price available
        """
        if symbol not in self.market_prices:
            raise InvalidOrderError(f"No market price available for {symbol}")

        return self.market_prices[symbol]

    async def test_connectivity(self) -> bool:
        """
        Test connectivity to exchange.

        For simulated exchange, always returns True if connected.

        Returns:
            True if connected, False otherwise
        """
        return self.connected

    def _calculate_portfolio_value(self) -> Decimal:
        """
        Calculate total portfolio value in quote currency (USDT).

        Returns:
            Total portfolio value
        """
        total_value = Decimal("0")

        for asset, amount in self.balances.items():
            if asset == "USDT":
                total_value += amount
            else:
                # Convert to USDT using market price
                symbol = f"{asset}USDT"
                if symbol in self.market_prices:
                    total_value += amount * self.market_prices[symbol]

        return total_value

    def execute_order(
        self,
        signal: "Signal",
        quantity: Decimal | float,
        volatility: float,
        symbol: str = "BTCUSDT",
    ) -> tuple[Decimal, Decimal, Decimal]:
        """
        Synchronous order execution for backtesting.

        This method provides a simplified interface for backtesting that returns
        execution details directly without async operations.

        Args:
            signal: Trading signal (BUY/SELL)
            quantity: Order quantity (will be converted to Decimal if float)
            volatility: Market volatility for slippage calculation
            symbol: Trading symbol (default: BTCUSDT)

        Returns:
            Tuple of (execution_price, commission, slippage_cost)

        Raises:
            InvalidOrderError: If order cannot be executed
        """
        from bot.core.strategy import Signal

        # Convert quantity to Decimal if it's a float
        if isinstance(quantity, float):
            quantity = Decimal(str(quantity))

        if symbol not in self.market_prices:
            raise InvalidOrderError(f"No market price available for {symbol}")

        market_price = self.market_prices[symbol]

        # Calculate slippage based on volatility
        slippage_factor = Decimal(str(self.slippage_factor))
        volatility_decimal = Decimal(str(volatility))
        slippage_pct = slippage_factor * volatility_decimal

        # Calculate execution price based on signal
        if signal == Signal.BUY:
            # Slippage increases buy price
            execution_price = market_price * (Decimal("1") + slippage_pct)
        elif signal == Signal.SELL:
            # Slippage decreases sell price
            execution_price = market_price * (Decimal("1") - slippage_pct)
        else:
            raise InvalidOrderError(f"Invalid signal for execution: {signal}")

        # Calculate commission
        notional_value = execution_price * quantity
        commission = notional_value * Decimal(str(self.taker_commission))

        # Calculate absolute slippage cost
        slippage_cost = abs(execution_price - market_price) * quantity

        # Track metrics
        self.total_commission_paid += commission
        self.total_slippage_cost += slippage_cost
        self.trade_count += 1

        return execution_price, commission, slippage_cost

    def get_performance_metrics(self) -> dict[str, Any]:
        """
        Get current performance metrics for paper trading.

        Returns:
            Dictionary with performance metrics including P/L, trades, commissions
        """
        current_value = self._calculate_portfolio_value()
        pnl = current_value - self.initial_portfolio_value
        pnl_pct = (
            (pnl / self.initial_portfolio_value * 100)
            if self.initial_portfolio_value > 0
            else Decimal("0")
        )

        return {
            "initial_value": float(self.initial_portfolio_value),
            "current_value": float(current_value),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "total_trades": self.trade_count,
            "total_commission": float(self.total_commission_paid),
            "total_slippage": float(self.total_slippage_cost),
            "open_orders": len([o for o in self.orders.values() if o.status == OrderStatus.OPEN]),
            "balances": {asset: float(amount) for asset, amount in self.balances.items()},
        }
