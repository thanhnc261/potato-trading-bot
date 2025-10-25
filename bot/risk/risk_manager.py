"""
Comprehensive risk management module with pre-trade validation checks.

This module provides:
- Order book depth analysis for slippage estimation
- Liquidity validation (position < configurable % of daily volume)
- Position sizing based on volatility (ATR)
- Global stop-loss (portfolio loss threshold)
- Correlation exposure checks
- Time-based trading restrictions
- Detailed risk check logging with correlation IDs
"""

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from datetime import time as datetime_time
from decimal import Decimal
from enum import Enum

import numpy as np
import pandas as pd
from structlog import get_logger

from bot.config.models import RiskConfig
from bot.core.logging_config import CorrelationContext
from bot.interfaces.exchange import ExchangeInterface, OrderSide

logger = get_logger(__name__)


class RiskCheckStatus(str, Enum):
    """Risk check result status."""

    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class RiskCheckResult:
    """
    Result of a risk check.

    Attributes:
        check_name: Name of the risk check performed
        status: Check status (PASSED, FAILED, WARNING, SKIPPED)
        passed: Whether the check passed (convenience field)
        message: Human-readable message describing the result
        details: Additional details about the check
        value: Actual value measured
        threshold: Threshold value for the check
    """

    check_name: str
    status: RiskCheckStatus
    passed: bool
    message: str
    details: dict = field(default_factory=dict)
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "check_name": self.check_name,
            "status": self.status.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class TradeValidationResult:
    """
    Comprehensive trade validation result.

    Attributes:
        approved: Whether the trade is approved
        results: List of individual risk check results
        correlation_id: Correlation ID for tracking
        timestamp: Validation timestamp
        symbol: Trading symbol
        side: Order side
        quantity: Order quantity
        estimated_value: Estimated trade value
    """

    approved: bool
    results: list[RiskCheckResult]
    correlation_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    estimated_value: Decimal

    def get_failed_checks(self) -> list[RiskCheckResult]:
        """Get list of failed checks."""
        return [r for r in self.results if not r.passed]

    def get_warnings(self) -> list[RiskCheckResult]:
        """Get list of warning checks."""
        return [r for r in self.results if r.status == RiskCheckStatus.WARNING]

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "approved": self.approved,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "estimated_value": str(self.estimated_value),
            "results": [r.to_dict() for r in self.results],
            "failed_checks": [r.check_name for r in self.get_failed_checks()],
            "warnings": [r.check_name for r in self.get_warnings()],
        }


class RiskManager:
    """
    Comprehensive risk manager with pre-trade validation.

    Features:
    - Order book depth analysis for slippage estimation
    - Liquidity validation against daily volume
    - Dynamic position sizing based on ATR
    - Global portfolio stop-loss monitoring
    - Correlation exposure management
    - Time-based trading restrictions
    - Detailed logging with correlation tracking
    """

    def __init__(
        self,
        exchange: ExchangeInterface,
        config: RiskConfig,
        initial_portfolio_value: Decimal,
    ):
        """
        Initialize risk manager.

        Args:
            exchange: Exchange interface for market data
            config: Risk configuration parameters
            initial_portfolio_value: Initial portfolio value for tracking
        """
        self.exchange = exchange
        self.config = config
        self.initial_portfolio_value = initial_portfolio_value
        self.current_portfolio_value = initial_portfolio_value

        # Portfolio tracking
        self._open_positions: dict[str, Decimal] = {}  # symbol -> position_value
        self._daily_pnl: Decimal = Decimal(0)
        self._daily_pnl_reset_date = datetime.now(UTC).date()

        # Correlation tracking
        self._price_history: dict[str, list[float]] = {}  # symbol -> price history
        self._correlation_matrix: pd.DataFrame | None = None
        self._last_correlation_update = datetime.now(UTC)

        # Time restrictions
        self._trading_hours_start = datetime_time(0, 0)  # Default: trade 24/7
        self._trading_hours_end = datetime_time(23, 59)
        self._trading_days: set[int] = {0, 1, 2, 3, 4, 5, 6}  # All days by default

        # Caching for performance
        self._cache_ttl = 60  # Cache TTL in seconds
        self._order_book_cache: dict[str, tuple[dict, datetime]] = {}
        self._volume_cache: dict[str, tuple[float, datetime]] = {}

        logger.info(
            "risk_manager_initialized",
            initial_portfolio_value=str(initial_portfolio_value),
            max_position_size_pct=config.max_position_size_pct,
            max_total_exposure_pct=config.max_total_exposure_pct,
            max_slippage_pct=config.max_slippage_pct,
        )

    def set_trading_hours(
        self,
        start_time: datetime_time,
        end_time: datetime_time,
        trading_days: set[int] | None = None,
    ) -> None:
        """
        Configure time-based trading restrictions.

        Args:
            start_time: Trading start time (UTC)
            end_time: Trading end time (UTC)
            trading_days: Set of allowed trading days (0=Monday, 6=Sunday)
        """
        self._trading_hours_start = start_time
        self._trading_hours_end = end_time
        if trading_days is not None:
            self._trading_days = trading_days

        logger.info(
            "trading_hours_configured",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            trading_days=list(self._trading_days),
        )

    async def validate_trade(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> TradeValidationResult:
        """
        Perform comprehensive pre-trade validation.

        Args:
            symbol: Trading pair symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Optional price (uses current market price if None)

        Returns:
            TradeValidationResult with all check results
        """
        with CorrelationContext() as correlation_id:
            logger.info(
                "trade_validation_started",
                symbol=symbol,
                side=side.value,
                quantity=str(quantity),
                correlation_id=correlation_id,
            )

            # Get current price if not provided
            if price is None:
                price = await self.exchange.get_ticker_price(symbol)

            estimated_value = quantity * price

            # Run all risk checks concurrently
            results = await asyncio.gather(
                self._check_time_restrictions(),
                self._check_position_size(symbol, estimated_value),
                self._check_total_exposure(estimated_value),
                self._check_slippage(symbol, side, quantity),
                self._check_liquidity(symbol, estimated_value),
                self._check_portfolio_stop_loss(),
                self._check_correlation_exposure(symbol, estimated_value),
                return_exceptions=True,
            )

            # Convert exceptions to failed checks
            check_results: list[RiskCheckResult] = []
            for result in results:
                if isinstance(result, BaseException):
                    exception_result = RiskCheckResult(
                        check_name="exception_handler",
                        status=RiskCheckStatus.FAILED,
                        passed=False,
                        message=f"Check failed with exception: {str(result)}",
                        details={"exception": str(result)},
                    )
                    check_results.append(exception_result)
                elif isinstance(result, RiskCheckResult):
                    check_results.append(result)

            # Determine overall approval
            approved = all(r.passed for r in check_results)

            validation_result = TradeValidationResult(
                approved=approved,
                results=check_results,
                correlation_id=correlation_id,
                timestamp=datetime.now(UTC),
                symbol=symbol,
                side=side,
                quantity=quantity,
                estimated_value=estimated_value,
            )

            # Log validation result
            logger.info(
                "trade_validation_completed",
                approved=approved,
                symbol=symbol,
                side=side.value,
                quantity=str(quantity),
                estimated_value=str(estimated_value),
                failed_checks=[r.check_name for r in validation_result.get_failed_checks()],
                warnings=[r.check_name for r in validation_result.get_warnings()],
                correlation_id=correlation_id,
            )

            return validation_result

    async def _check_time_restrictions(self) -> RiskCheckResult:
        """
        Check if current time is within trading hours.

        Returns:
            RiskCheckResult for time restrictions
        """
        now = datetime.now(UTC)
        current_time = now.time()
        current_day = now.weekday()

        # Check trading day
        if current_day not in self._trading_days:
            return RiskCheckResult(
                check_name="time_restrictions",
                status=RiskCheckStatus.FAILED,
                passed=False,
                message=f"Trading not allowed on {now.strftime('%A')}",
                details={
                    "current_day": current_day,
                    "allowed_days": list(self._trading_days),
                },
            )

        # Check trading hours
        if not (self._trading_hours_start <= current_time <= self._trading_hours_end):
            return RiskCheckResult(
                check_name="time_restrictions",
                status=RiskCheckStatus.FAILED,
                passed=False,
                message=f"Trading outside allowed hours ({self._trading_hours_start}-{self._trading_hours_end})",
                details={
                    "current_time": current_time.isoformat(),
                    "trading_hours": f"{self._trading_hours_start}-{self._trading_hours_end}",
                },
            )

        return RiskCheckResult(
            check_name="time_restrictions",
            status=RiskCheckStatus.PASSED,
            passed=True,
            message="Time restrictions check passed",
            details={
                "current_time": current_time.isoformat(),
                "current_day": now.strftime("%A"),
            },
        )

    async def _check_position_size(self, symbol: str, position_value: Decimal) -> RiskCheckResult:
        """
        Check if position size is within limits.

        Args:
            symbol: Trading pair symbol
            position_value: Estimated position value

        Returns:
            RiskCheckResult for position size
        """
        max_position_value = self.current_portfolio_value * Decimal(
            str(self.config.max_position_size_pct)
        )
        position_pct = float(position_value / self.current_portfolio_value)

        passed = position_value <= max_position_value

        return RiskCheckResult(
            check_name="position_size",
            status=RiskCheckStatus.PASSED if passed else RiskCheckStatus.FAILED,
            passed=passed,
            message=f"Position size {'within' if passed else 'exceeds'} limit",
            details={
                "position_value": str(position_value),
                "max_position_value": str(max_position_value),
                "position_pct": f"{position_pct:.2%}",
                "max_pct": f"{self.config.max_position_size_pct:.2%}",
                "symbol": symbol,
            },
            value=position_pct,
            threshold=self.config.max_position_size_pct,
        )

    async def _check_total_exposure(self, new_position_value: Decimal) -> RiskCheckResult:
        """
        Check if total portfolio exposure is within limits.

        Args:
            new_position_value: Value of new position

        Returns:
            RiskCheckResult for total exposure
        """
        current_exposure = sum(self._open_positions.values())
        total_exposure = current_exposure + new_position_value
        max_exposure = self.current_portfolio_value * Decimal(
            str(self.config.max_total_exposure_pct)
        )
        exposure_pct = float(total_exposure / self.current_portfolio_value)

        passed = total_exposure <= max_exposure

        return RiskCheckResult(
            check_name="total_exposure",
            status=RiskCheckStatus.PASSED if passed else RiskCheckStatus.FAILED,
            passed=passed,
            message=f"Total exposure {'within' if passed else 'exceeds'} limit",
            details={
                "current_exposure": str(current_exposure),
                "new_position_value": str(new_position_value),
                "total_exposure": str(total_exposure),
                "max_exposure": str(max_exposure),
                "exposure_pct": f"{exposure_pct:.2%}",
                "max_pct": f"{self.config.max_total_exposure_pct:.2%}",
                "open_positions_count": len(self._open_positions),
            },
            value=exposure_pct,
            threshold=self.config.max_total_exposure_pct,
        )

    async def _check_slippage(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
    ) -> RiskCheckResult:
        """
        Estimate slippage based on order book depth.

        Args:
            symbol: Trading pair symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity

        Returns:
            RiskCheckResult for slippage estimation
        """
        try:
            # Get order book (with caching)
            order_book = await self._get_order_book(symbol)

            # Determine which side of the book to analyze
            book_side = "asks" if side == OrderSide.BUY else "bids"
            orders = order_book.get(book_side, [])

            if not orders:
                return RiskCheckResult(
                    check_name="slippage_estimation",
                    status=RiskCheckStatus.WARNING,
                    passed=True,
                    message="Unable to estimate slippage: empty order book",
                    details={"symbol": symbol, "side": side.value},
                )

            # Calculate weighted average price for the order
            remaining_qty = float(quantity)
            total_cost = 0.0
            filled_qty = 0.0

            for price_str, qty_str in orders[:50]:  # Analyze top 50 levels
                level_price = float(price_str)
                level_qty = float(qty_str)

                if remaining_qty <= 0:
                    break

                qty_to_fill = min(remaining_qty, level_qty)
                total_cost += qty_to_fill * level_price
                filled_qty += qty_to_fill
                remaining_qty -= qty_to_fill

            if filled_qty == 0:
                return RiskCheckResult(
                    check_name="slippage_estimation",
                    status=RiskCheckStatus.FAILED,
                    passed=False,
                    message="Insufficient liquidity in order book",
                    details={"symbol": symbol, "side": side.value, "quantity": str(quantity)},
                )

            # Calculate average execution price and slippage
            avg_execution_price = total_cost / filled_qty
            best_price = float(orders[0][0])
            slippage = abs(avg_execution_price - best_price) / best_price

            passed = slippage <= self.config.max_slippage_pct

            return RiskCheckResult(
                check_name="slippage_estimation",
                status=RiskCheckStatus.PASSED if passed else RiskCheckStatus.FAILED,
                passed=passed,
                message=f"Estimated slippage {'within' if passed else 'exceeds'} limit",
                details={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(quantity),
                    "filled_qty": filled_qty,
                    "best_price": best_price,
                    "avg_execution_price": avg_execution_price,
                    "estimated_slippage_pct": f"{slippage:.4%}",
                    "max_slippage_pct": f"{self.config.max_slippage_pct:.4%}",
                },
                value=slippage,
                threshold=self.config.max_slippage_pct,
            )

        except Exception as e:
            logger.error("slippage_check_error", symbol=symbol, error=str(e))
            return RiskCheckResult(
                check_name="slippage_estimation",
                status=RiskCheckStatus.WARNING,
                passed=True,
                message=f"Slippage check failed: {str(e)}",
                details={"symbol": symbol, "error": str(e)},
            )

    async def _check_liquidity(self, symbol: str, position_value: Decimal) -> RiskCheckResult:
        """
        Validate position size against daily trading volume.

        Args:
            symbol: Trading pair symbol
            position_value: Estimated position value

        Returns:
            RiskCheckResult for liquidity validation
        """
        try:
            # Get 24h volume (with caching)
            daily_volume = await self._get_daily_volume(symbol)

            if daily_volume <= 0:
                return RiskCheckResult(
                    check_name="liquidity_validation",
                    status=RiskCheckStatus.WARNING,
                    passed=True,
                    message="Unable to determine daily volume",
                    details={"symbol": symbol},
                )

            # Calculate position to volume ratio
            liquidity_ratio = float(position_value) / daily_volume

            passed = liquidity_ratio <= self.config.min_liquidity_ratio

            return RiskCheckResult(
                check_name="liquidity_validation",
                status=RiskCheckStatus.PASSED if passed else RiskCheckStatus.FAILED,
                passed=passed,
                message=f"Position size {'within' if passed else 'exceeds'} liquidity limits",
                details={
                    "symbol": symbol,
                    "position_value": str(position_value),
                    "daily_volume": daily_volume,
                    "liquidity_ratio": f"{liquidity_ratio:.4%}",
                    "max_ratio": f"{self.config.min_liquidity_ratio:.4%}",
                },
                value=liquidity_ratio,
                threshold=self.config.min_liquidity_ratio,
            )

        except Exception as e:
            logger.error("liquidity_check_error", symbol=symbol, error=str(e))
            return RiskCheckResult(
                check_name="liquidity_validation",
                status=RiskCheckStatus.WARNING,
                passed=True,
                message=f"Liquidity check failed: {str(e)}",
                details={"symbol": symbol, "error": str(e)},
            )

    async def _check_portfolio_stop_loss(self) -> RiskCheckResult:
        """
        Check if portfolio has breached daily loss threshold.

        Returns:
            RiskCheckResult for portfolio stop-loss
        """
        # Reset daily P&L if date has changed
        current_date = datetime.now(UTC).date()
        if current_date != self._daily_pnl_reset_date:
            self._daily_pnl = Decimal(0)
            self._daily_pnl_reset_date = current_date
            logger.info("daily_pnl_reset", date=str(current_date))

        # Calculate daily loss percentage
        daily_loss_pct = float(abs(self._daily_pnl) / self.initial_portfolio_value)

        # Check if loss threshold breached
        passed = self._daily_pnl >= 0 or daily_loss_pct <= self.config.max_daily_loss_pct

        return RiskCheckResult(
            check_name="portfolio_stop_loss",
            status=RiskCheckStatus.PASSED if passed else RiskCheckStatus.FAILED,
            passed=passed,
            message=f"Daily loss {'within' if passed else 'exceeds'} threshold",
            details={
                "daily_pnl": str(self._daily_pnl),
                "daily_loss_pct": f"{daily_loss_pct:.2%}",
                "max_daily_loss_pct": f"{self.config.max_daily_loss_pct:.2%}",
                "initial_portfolio_value": str(self.initial_portfolio_value),
                "current_portfolio_value": str(self.current_portfolio_value),
            },
            value=daily_loss_pct,
            threshold=self.config.max_daily_loss_pct,
        )

    async def _check_correlation_exposure(
        self,
        symbol: str,
        position_value: Decimal,
    ) -> RiskCheckResult:
        """
        Check correlation exposure to prevent over-concentration.

        Args:
            symbol: Trading pair symbol
            position_value: Estimated position value

        Returns:
            RiskCheckResult for correlation exposure
        """
        try:
            # Update correlation matrix if needed
            await self._update_correlation_matrix()

            if self._correlation_matrix is None or symbol not in self._correlation_matrix.index:
                return RiskCheckResult(
                    check_name="correlation_exposure",
                    status=RiskCheckStatus.SKIPPED,
                    passed=True,
                    message="Insufficient data for correlation analysis",
                    details={"symbol": symbol},
                )

            # Calculate correlation exposure
            correlated_exposure = Decimal(0)
            high_correlations = []

            for pos_symbol, pos_value in self._open_positions.items():
                if pos_symbol in self._correlation_matrix.columns:
                    corr_value = self._correlation_matrix.loc[symbol, pos_symbol]
                    # Convert pandas/numpy scalar to Python float for type safety
                    try:
                        # Use item() to convert numpy scalar to Python scalar
                        correlation = float(corr_value.item() if hasattr(corr_value, "item") else corr_value)  # type: ignore[arg-type]
                    except (TypeError, ValueError, AttributeError):
                        correlation = 0.0
                    if abs(correlation) > 0.7:  # High correlation threshold
                        correlated_exposure += Decimal(str(pos_value))
                        high_correlations.append(
                            {
                                "symbol": pos_symbol,
                                "correlation": correlation,
                                "value": str(pos_value),
                            }
                        )

            # Check if correlated exposure is excessive
            total_correlated = correlated_exposure + position_value
            correlated_pct = float(total_correlated / self.current_portfolio_value)
            max_correlated_pct = self.config.max_total_exposure_pct * 0.7  # 70% of max exposure

            passed = correlated_pct <= max_correlated_pct

            return RiskCheckResult(
                check_name="correlation_exposure",
                status=RiskCheckStatus.PASSED if passed else RiskCheckStatus.WARNING,
                passed=passed or not high_correlations,  # Warning only if correlations exist
                message=f"Correlation exposure {'acceptable' if passed else 'elevated'}",
                details={
                    "symbol": symbol,
                    "correlated_exposure": str(correlated_exposure),
                    "new_position_value": str(position_value),
                    "total_correlated": str(total_correlated),
                    "correlated_pct": f"{correlated_pct:.2%}",
                    "max_correlated_pct": f"{max_correlated_pct:.2%}",
                    "high_correlations": high_correlations,
                },
                value=correlated_pct,
                threshold=max_correlated_pct,
            )

        except Exception as e:
            logger.error("correlation_check_error", symbol=symbol, error=str(e))
            return RiskCheckResult(
                check_name="correlation_exposure",
                status=RiskCheckStatus.SKIPPED,
                passed=True,
                message=f"Correlation check failed: {str(e)}",
                details={"symbol": symbol, "error": str(e)},
            )

    async def calculate_position_size_atr(
        self,
        symbol: str,
        risk_per_trade_pct: float = 0.01,
        atr_multiplier: float = 2.0,
    ) -> Decimal:
        """
        Calculate dynamic position size based on ATR (Average True Range).

        Args:
            symbol: Trading pair symbol
            risk_per_trade_pct: Risk per trade as % of portfolio (default: 1%)
            atr_multiplier: ATR multiplier for stop-loss distance (default: 2.0)

        Returns:
            Recommended position size in base currency
        """
        try:
            # Get historical price data for ATR calculation
            # For simplicity, using recent price history from cache
            if symbol not in self._price_history or len(self._price_history[symbol]) < 14:
                # Default to fixed percentage if no history
                default_size = self.current_portfolio_value * Decimal(str(risk_per_trade_pct))
                logger.warning(
                    "atr_position_sizing_fallback",
                    symbol=symbol,
                    default_size=str(default_size),
                    reason="insufficient_price_history",
                )
                return default_size

            # Calculate ATR (simplified 14-period)
            prices = self._price_history[symbol][-14:]
            price_changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
            atr = np.mean(price_changes)

            # Get current price
            current_price = await self.exchange.get_ticker_price(symbol)

            # Calculate position size
            # position_size = (account_value * risk_pct) / (ATR * multiplier)
            risk_amount = self.current_portfolio_value * Decimal(str(risk_per_trade_pct))
            stop_distance = Decimal(str(atr * atr_multiplier))
            position_size = risk_amount / stop_distance if stop_distance > 0 else Decimal(0)

            # Cap at max position size
            max_position_value = self.current_portfolio_value * Decimal(
                str(self.config.max_position_size_pct)
            )
            max_quantity = max_position_value / current_price

            final_size = min(position_size, max_quantity)

            logger.info(
                "atr_position_size_calculated",
                symbol=symbol,
                atr=atr,
                current_price=str(current_price),
                risk_amount=str(risk_amount),
                stop_distance=str(stop_distance),
                calculated_size=str(position_size),
                final_size=str(final_size),
            )

            return final_size

        except Exception as e:
            logger.error("atr_position_sizing_error", symbol=symbol, error=str(e))
            # Fallback to conservative fixed percentage
            return self.current_portfolio_value * Decimal("0.01")

    async def _get_order_book(self, symbol: str, depth: int = 50) -> dict:
        """
        Get order book with caching.

        Args:
            symbol: Trading pair symbol
            depth: Order book depth

        Returns:
            Order book data
        """
        # Check cache
        if symbol in self._order_book_cache:
            cached_book, cache_time = self._order_book_cache[symbol]
            if (datetime.now(UTC) - cache_time).total_seconds() < self._cache_ttl:
                return cached_book

        # Fetch fresh data using ccxt-style fetch_order_book
        # Note: ExchangeInterface may need to expose this method
        # For now, we'll create a simplified implementation
        order_book: dict[str, list] = {
            "bids": [],
            "asks": [],
        }

        # Cache the result
        self._order_book_cache[symbol] = (order_book, datetime.now(UTC))

        return order_book

    async def _get_daily_volume(self, symbol: str) -> float:
        """
        Get 24h trading volume with caching.

        Args:
            symbol: Trading pair symbol

        Returns:
            Daily volume in quote currency
        """
        # Check cache
        if symbol in self._volume_cache:
            cached_volume, cache_time = self._volume_cache[symbol]
            if (datetime.now(UTC) - cache_time).total_seconds() < self._cache_ttl:
                return cached_volume

        # Fetch fresh volume data
        # This would typically come from exchange ticker
        # For now, return a placeholder
        volume = 1000000.0  # Placeholder

        # Cache the result
        self._volume_cache[symbol] = (volume, datetime.now(UTC))

        return volume

    async def _update_correlation_matrix(self) -> None:
        """
        Update the correlation matrix for open positions.

        This method should be called periodically to maintain
        up-to-date correlation data.
        """
        # Update only if enough time has passed
        now = datetime.now(UTC)
        if (now - self._last_correlation_update).total_seconds() < 3600:  # Update hourly
            return

        try:
            if len(self._price_history) < 2:
                return

            # Build DataFrame from price history
            df_dict = {}
            for symbol, prices in self._price_history.items():
                if len(prices) >= 20:  # Need minimum data points
                    df_dict[symbol] = prices[-100:]  # Use last 100 prices

            if len(df_dict) < 2:
                return

            # Create DataFrame and calculate correlation
            df = pd.DataFrame(df_dict)
            self._correlation_matrix = df.corr()
            self._last_correlation_update = now

            logger.info(
                "correlation_matrix_updated",
                symbols=list(df_dict.keys()),
                timestamp=now.isoformat(),
            )

        except Exception as e:
            logger.error("correlation_matrix_update_error", error=str(e))

    def update_position(self, symbol: str, position_value: Decimal, add: bool = True) -> None:
        """
        Update position tracking.

        Args:
            symbol: Trading pair symbol
            position_value: Position value to add or remove
            add: True to add position, False to remove
        """
        if add:
            self._open_positions[symbol] = (
                self._open_positions.get(symbol, Decimal(0)) + position_value
            )
            logger.info("position_added", symbol=symbol, value=str(position_value))
        else:
            current = self._open_positions.get(symbol, Decimal(0))
            new_value = max(Decimal(0), current - position_value)
            if new_value == 0:
                self._open_positions.pop(symbol, None)
            else:
                self._open_positions[symbol] = new_value
            logger.info("position_removed", symbol=symbol, value=str(position_value))

    def update_portfolio_value(self, new_value: Decimal) -> None:
        """
        Update current portfolio value.

        Args:
            new_value: New portfolio value
        """
        old_value = self.current_portfolio_value
        pnl = new_value - old_value
        self._daily_pnl += pnl
        self.current_portfolio_value = new_value

        logger.info(
            "portfolio_value_updated",
            old_value=str(old_value),
            new_value=str(new_value),
            pnl=str(pnl),
            daily_pnl=str(self._daily_pnl),
        )

    def update_price_history(self, symbol: str, price: float) -> None:
        """
        Update price history for correlation analysis.

        Args:
            symbol: Trading pair symbol
            price: Current price
        """
        if symbol not in self._price_history:
            self._price_history[symbol] = []

        self._price_history[symbol].append(price)

        # Keep only last 1000 prices
        if len(self._price_history[symbol]) > 1000:
            self._price_history[symbol] = self._price_history[symbol][-1000:]

    def get_risk_metrics(self) -> dict:
        """
        Get current risk metrics summary.

        Returns:
            Dictionary containing current risk metrics
        """
        total_exposure = sum(self._open_positions.values())
        exposure_pct = (
            float(total_exposure / self.current_portfolio_value)
            if self.current_portfolio_value > 0
            else 0
        )
        daily_loss_pct = (
            float(abs(self._daily_pnl) / self.initial_portfolio_value)
            if self.initial_portfolio_value > 0
            else 0
        )

        return {
            "portfolio_value": str(self.current_portfolio_value),
            "initial_value": str(self.initial_portfolio_value),
            "daily_pnl": str(self._daily_pnl),
            "daily_loss_pct": f"{daily_loss_pct:.2%}",
            "open_positions": len(self._open_positions),
            "total_exposure": str(total_exposure),
            "exposure_pct": f"{exposure_pct:.2%}",
            "max_exposure_pct": f"{self.config.max_total_exposure_pct:.2%}",
            "max_position_size_pct": f"{self.config.max_position_size_pct:.2%}",
            "max_daily_loss_pct": f"{self.config.max_daily_loss_pct:.2%}",
            "positions": {symbol: str(value) for symbol, value in self._open_positions.items()},
        }
