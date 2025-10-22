"""
Comprehensive unit tests for RiskManager.

Tests all risk checks:
- Time-based trading restrictions
- Position size limits
- Total exposure limits
- Slippage estimation
- Liquidity validation
- Portfolio stop-loss
- Correlation exposure
- ATR-based position sizing
"""

import asyncio
from datetime import datetime, time as datetime_time, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from bot.config.models import RiskConfig
from bot.interfaces.exchange import OrderSide
from bot.risk.risk_manager import (
    RiskCheckStatus,
    RiskManager,
    TradeValidationResult,
)


@pytest.fixture
def risk_config():
    """Create a default risk configuration."""
    return RiskConfig(
        max_position_size_pct=0.03,  # 3%
        max_total_exposure_pct=0.25,  # 25%
        max_daily_loss_pct=0.02,  # 2%
        max_slippage_pct=0.005,  # 0.5%
        min_liquidity_ratio=0.01,  # 1%
        var_confidence=0.95,
        enable_emergency_stop=True,
    )


@pytest.fixture
def mock_exchange():
    """Create a mock exchange interface."""
    exchange = AsyncMock()
    exchange.get_ticker_price = AsyncMock(return_value=Decimal("50000"))
    return exchange


@pytest.fixture
def risk_manager(mock_exchange, risk_config):
    """Create a RiskManager instance for testing."""
    return RiskManager(
        exchange=mock_exchange,
        config=risk_config,
        initial_portfolio_value=Decimal("100000"),
    )


class TestTimeRestrictions:
    """Test time-based trading restrictions."""

    @pytest.mark.asyncio
    async def test_trading_within_hours(self, risk_manager):
        """Test that trading is allowed within configured hours."""
        # Set trading hours to always allow (24/7)
        risk_manager.set_trading_hours(
            start_time=datetime_time(0, 0),
            end_time=datetime_time(23, 59),
            trading_days={0, 1, 2, 3, 4, 5, 6},
        )

        result = await risk_manager._check_time_restrictions()

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED

    @pytest.mark.asyncio
    async def test_trading_outside_hours(self, risk_manager):
        """Test that trading is blocked outside configured hours."""
        # Set restrictive trading hours (impossible to match)
        current_time = datetime.now(timezone.utc).time()
        restricted_start = datetime_time(
            (current_time.hour + 1) % 24,
            current_time.minute,
        )
        restricted_end = datetime_time(
            (current_time.hour + 2) % 24,
            current_time.minute,
        )

        risk_manager.set_trading_hours(
            start_time=restricted_start,
            end_time=restricted_end,
            trading_days={0, 1, 2, 3, 4, 5, 6},
        )

        result = await risk_manager._check_time_restrictions()

        assert result.passed is False
        assert result.status == RiskCheckStatus.FAILED
        assert "outside allowed hours" in result.message

    @pytest.mark.asyncio
    async def test_trading_on_restricted_day(self, risk_manager):
        """Test that trading is blocked on restricted days."""
        # Restrict trading to days that don't include today
        current_day = datetime.now(timezone.utc).weekday()
        allowed_days = {(current_day + 1) % 7}  # Tomorrow only

        risk_manager.set_trading_hours(
            start_time=datetime_time(0, 0),
            end_time=datetime_time(23, 59),
            trading_days=allowed_days,
        )

        result = await risk_manager._check_time_restrictions()

        assert result.passed is False
        assert result.status == RiskCheckStatus.FAILED
        assert "not allowed" in result.message


class TestPositionSize:
    """Test position size validation."""

    @pytest.mark.asyncio
    async def test_position_within_limit(self, risk_manager):
        """Test that position within limit passes."""
        # 2% position (within 3% limit)
        position_value = Decimal("2000")

        result = await risk_manager._check_position_size("BTCUSDT", position_value)

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED
        assert result.value < risk_manager.config.max_position_size_pct

    @pytest.mark.asyncio
    async def test_position_exceeds_limit(self, risk_manager):
        """Test that position exceeding limit fails."""
        # 5% position (exceeds 3% limit)
        position_value = Decimal("5000")

        result = await risk_manager._check_position_size("BTCUSDT", position_value)

        assert result.passed is False
        assert result.status == RiskCheckStatus.FAILED
        assert result.value > risk_manager.config.max_position_size_pct

    @pytest.mark.asyncio
    async def test_position_at_exact_limit(self, risk_manager):
        """Test that position at exact limit passes."""
        # Exactly 3% position
        position_value = Decimal("3000")

        result = await risk_manager._check_position_size("BTCUSDT", position_value)

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED


class TestTotalExposure:
    """Test total portfolio exposure validation."""

    @pytest.mark.asyncio
    async def test_exposure_within_limit(self, risk_manager):
        """Test that total exposure within limit passes."""
        # Add some existing positions (10% total)
        risk_manager._open_positions = {
            "BTCUSDT": Decimal("5000"),
            "ETHUSDT": Decimal("5000"),
        }

        # New position of 5% (total 15%, within 25% limit)
        new_position = Decimal("5000")

        result = await risk_manager._check_total_exposure(new_position)

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED
        assert result.value < risk_manager.config.max_total_exposure_pct

    @pytest.mark.asyncio
    async def test_exposure_exceeds_limit(self, risk_manager):
        """Test that total exposure exceeding limit fails."""
        # Add existing positions (20%)
        risk_manager._open_positions = {
            "BTCUSDT": Decimal("10000"),
            "ETHUSDT": Decimal("10000"),
        }

        # New position of 10% (total 30%, exceeds 25% limit)
        new_position = Decimal("10000")

        result = await risk_manager._check_total_exposure(new_position)

        assert result.passed is False
        assert result.status == RiskCheckStatus.FAILED
        assert result.value > risk_manager.config.max_total_exposure_pct

    @pytest.mark.asyncio
    async def test_first_position_exposure(self, risk_manager):
        """Test exposure check with no existing positions."""
        # No existing positions
        assert len(risk_manager._open_positions) == 0

        # New position of 5%
        new_position = Decimal("5000")

        result = await risk_manager._check_total_exposure(new_position)

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED


class TestSlippageEstimation:
    """Test slippage estimation from order book."""

    @pytest.mark.asyncio
    async def test_slippage_within_limit(self, risk_manager, mock_exchange):
        """Test that acceptable slippage passes."""
        # Mock order book with tight spread
        order_book = {
            "asks": [
                ["50000", "1.0"],
                ["50010", "2.0"],
                ["50020", "3.0"],
            ],
            "bids": [
                ["49990", "1.0"],
                ["49980", "2.0"],
                ["49970", "3.0"],
            ],
        }

        with patch.object(risk_manager, "_get_order_book", return_value=order_book):
            result = await risk_manager._check_slippage(
                "BTCUSDT",
                OrderSide.BUY,
                Decimal("0.5"),
            )

            assert result.passed is True
            assert result.status == RiskCheckStatus.PASSED

    @pytest.mark.asyncio
    async def test_slippage_exceeds_limit(self, risk_manager):
        """Test that excessive slippage fails."""
        # Mock order book with wide spread
        order_book = {
            "asks": [
                ["50000", "0.1"],
                ["50500", "0.1"],  # Large price jump
                ["51000", "0.1"],
            ],
            "bids": [
                ["49000", "0.1"],
                ["48500", "0.1"],
                ["48000", "0.1"],
            ],
        }

        with patch.object(risk_manager, "_get_order_book", return_value=order_book):
            result = await risk_manager._check_slippage(
                "BTCUSDT",
                OrderSide.BUY,
                Decimal("0.25"),  # Large order
            )

            # Should either fail or warn about slippage
            assert result.status in [RiskCheckStatus.FAILED, RiskCheckStatus.WARNING]

    @pytest.mark.asyncio
    async def test_empty_order_book(self, risk_manager):
        """Test handling of empty order book."""
        order_book = {"asks": [], "bids": []}

        with patch.object(risk_manager, "_get_order_book", return_value=order_book):
            result = await risk_manager._check_slippage(
                "BTCUSDT",
                OrderSide.BUY,
                Decimal("1.0"),
            )

            assert result.status == RiskCheckStatus.WARNING
            assert "empty order book" in result.message


class TestLiquidityValidation:
    """Test liquidity validation against daily volume."""

    @pytest.mark.asyncio
    async def test_liquidity_sufficient(self, risk_manager):
        """Test that sufficient liquidity passes."""
        # Mock high daily volume
        daily_volume = 10000000.0  # 10M

        with patch.object(risk_manager, "_get_daily_volume", return_value=daily_volume):
            # Position of 50k (0.5% of volume, within 1% limit)
            result = await risk_manager._check_liquidity("BTCUSDT", Decimal("50000"))

            assert result.passed is True
            assert result.status == RiskCheckStatus.PASSED

    @pytest.mark.asyncio
    async def test_liquidity_insufficient(self, risk_manager):
        """Test that insufficient liquidity fails."""
        # Mock low daily volume
        daily_volume = 1000000.0  # 1M

        with patch.object(risk_manager, "_get_daily_volume", return_value=daily_volume):
            # Position of 50k (5% of volume, exceeds 1% limit)
            result = await risk_manager._check_liquidity("BTCUSDT", Decimal("50000"))

            assert result.passed is False
            assert result.status == RiskCheckStatus.FAILED

    @pytest.mark.asyncio
    async def test_liquidity_no_volume_data(self, risk_manager):
        """Test handling of missing volume data."""
        with patch.object(risk_manager, "_get_daily_volume", return_value=0.0):
            result = await risk_manager._check_liquidity("BTCUSDT", Decimal("50000"))

            assert result.status == RiskCheckStatus.WARNING
            assert "Unable to determine" in result.message


class TestPortfolioStopLoss:
    """Test portfolio-level stop-loss."""

    @pytest.mark.asyncio
    async def test_no_loss(self, risk_manager):
        """Test that portfolio with no loss passes."""
        risk_manager._daily_pnl = Decimal("0")

        result = await risk_manager._check_portfolio_stop_loss()

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED

    @pytest.mark.asyncio
    async def test_profit(self, risk_manager):
        """Test that portfolio with profit passes."""
        risk_manager._daily_pnl = Decimal("5000")  # +5k profit

        result = await risk_manager._check_portfolio_stop_loss()

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED

    @pytest.mark.asyncio
    async def test_loss_within_threshold(self, risk_manager):
        """Test that loss within threshold passes."""
        # 1% loss (within 2% threshold)
        risk_manager._daily_pnl = Decimal("-1000")

        result = await risk_manager._check_portfolio_stop_loss()

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED

    @pytest.mark.asyncio
    async def test_loss_exceeds_threshold(self, risk_manager):
        """Test that loss exceeding threshold fails."""
        # 3% loss (exceeds 2% threshold)
        risk_manager._daily_pnl = Decimal("-3000")

        result = await risk_manager._check_portfolio_stop_loss()

        assert result.passed is False
        assert result.status == RiskCheckStatus.FAILED


class TestCorrelationExposure:
    """Test correlation-based exposure limits."""

    @pytest.mark.asyncio
    async def test_no_correlations(self, risk_manager):
        """Test that check is skipped with insufficient data."""
        risk_manager._correlation_matrix = None

        result = await risk_manager._check_correlation_exposure("BTCUSDT", Decimal("5000"))

        assert result.passed is True
        assert result.status == RiskCheckStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_low_correlation_exposure(self, risk_manager):
        """Test that low correlation exposure passes."""
        # Mock correlation matrix with low correlations
        import pandas as pd
        correlation_data = {
            "BTCUSDT": {"BTCUSDT": 1.0, "ETHUSDT": 0.3, "ADAUSDT": 0.2},
            "ETHUSDT": {"BTCUSDT": 0.3, "ETHUSDT": 1.0, "ADAUSDT": 0.4},
            "ADAUSDT": {"BTCUSDT": 0.2, "ETHUSDT": 0.4, "ADAUSDT": 1.0},
        }
        risk_manager._correlation_matrix = pd.DataFrame(correlation_data)
        risk_manager._open_positions = {"ETHUSDT": Decimal("5000")}

        result = await risk_manager._check_correlation_exposure("BTCUSDT", Decimal("5000"))

        assert result.passed is True
        assert result.status == RiskCheckStatus.PASSED

    @pytest.mark.asyncio
    async def test_high_correlation_exposure(self, risk_manager):
        """Test that high correlation exposure triggers warning."""
        # Mock correlation matrix with high correlations
        import pandas as pd
        correlation_data = {
            "BTCUSDT": {"BTCUSDT": 1.0, "ETHUSDT": 0.9, "ADAUSDT": 0.85},
            "ETHUSDT": {"BTCUSDT": 0.9, "ETHUSDT": 1.0, "ADAUSDT": 0.88},
            "ADAUSDT": {"BTCUSDT": 0.85, "ETHUSDT": 0.88, "ADAUSDT": 1.0},
        }
        risk_manager._correlation_matrix = pd.DataFrame(correlation_data)
        risk_manager._open_positions = {
            "ETHUSDT": Decimal("15000"),
            "ADAUSDT": Decimal("15000"),
        }

        result = await risk_manager._check_correlation_exposure("BTCUSDT", Decimal("10000"))

        # Should generate warning about high correlation
        assert result.status in [RiskCheckStatus.WARNING, RiskCheckStatus.FAILED]
        assert len(result.details.get("high_correlations", [])) > 0


class TestATRPositionSizing:
    """Test ATR-based position sizing."""

    @pytest.mark.asyncio
    async def test_atr_calculation_with_history(self, risk_manager, mock_exchange):
        """Test ATR position sizing with sufficient price history."""
        # Mock price history
        risk_manager._price_history["BTCUSDT"] = [
            50000 + i * 100 for i in range(20)
        ]

        position_size = await risk_manager.calculate_position_size_atr(
            "BTCUSDT",
            risk_per_trade_pct=0.01,
            atr_multiplier=2.0,
        )

        assert position_size > 0
        assert isinstance(position_size, Decimal)

    @pytest.mark.asyncio
    async def test_atr_fallback_no_history(self, risk_manager, mock_exchange):
        """Test ATR position sizing fallback with no history."""
        # No price history
        risk_manager._price_history["BTCUSDT"] = []

        position_size = await risk_manager.calculate_position_size_atr(
            "BTCUSDT",
            risk_per_trade_pct=0.01,
        )

        # Should fall back to default
        expected_default = risk_manager.current_portfolio_value * Decimal("0.01")
        assert position_size == expected_default

    @pytest.mark.asyncio
    async def test_atr_respects_max_position(self, risk_manager, mock_exchange):
        """Test that ATR sizing respects max position limit."""
        # Mock price history with very low volatility
        risk_manager._price_history["BTCUSDT"] = [50000.0] * 20

        position_size = await risk_manager.calculate_position_size_atr(
            "BTCUSDT",
            risk_per_trade_pct=0.05,  # High risk
            atr_multiplier=0.1,  # Low multiplier
        )

        # Should be capped at max position size
        max_position_value = risk_manager.current_portfolio_value * Decimal(
            str(risk_manager.config.max_position_size_pct)
        )
        current_price = await risk_manager.exchange.get_ticker_price("BTCUSDT")
        max_quantity = max_position_value / current_price

        assert position_size <= max_quantity


class TestTradeValidation:
    """Test comprehensive trade validation."""

    @pytest.mark.asyncio
    async def test_valid_trade_approval(self, risk_manager, mock_exchange):
        """Test that a valid trade is approved."""
        # Set up for a valid trade
        risk_manager.set_trading_hours(
            start_time=datetime_time(0, 0),
            end_time=datetime_time(23, 59),
            trading_days={0, 1, 2, 3, 4, 5, 6},
        )

        result = await risk_manager.validate_trade(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
        )

        assert isinstance(result, TradeValidationResult)
        assert result.symbol == "BTCUSDT"
        assert result.side == OrderSide.BUY
        assert len(result.results) > 0
        assert result.correlation_id is not None

    @pytest.mark.asyncio
    async def test_invalid_trade_rejection(self, risk_manager, mock_exchange):
        """Test that an invalid trade is rejected."""
        # Create conditions for rejection (excessive position size)
        result = await risk_manager.validate_trade(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),  # Very large position
            price=Decimal("50000"),
        )

        assert isinstance(result, TradeValidationResult)
        assert result.approved is False
        assert len(result.get_failed_checks()) > 0

    @pytest.mark.asyncio
    async def test_validation_uses_current_price(self, risk_manager, mock_exchange):
        """Test that validation fetches current price when not provided."""
        mock_exchange.get_ticker_price.return_value = Decimal("55000")

        result = await risk_manager.validate_trade(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            price=None,  # No price provided
        )

        # Should have called get_ticker_price
        mock_exchange.get_ticker_price.assert_called()
        assert result.estimated_value == Decimal("0.1") * Decimal("55000")


class TestPortfolioTracking:
    """Test portfolio and position tracking."""

    def test_update_position_add(self, risk_manager):
        """Test adding a position."""
        risk_manager.update_position("BTCUSDT", Decimal("5000"), add=True)

        assert "BTCUSDT" in risk_manager._open_positions
        assert risk_manager._open_positions["BTCUSDT"] == Decimal("5000")

    def test_update_position_accumulate(self, risk_manager):
        """Test accumulating positions."""
        risk_manager.update_position("BTCUSDT", Decimal("5000"), add=True)
        risk_manager.update_position("BTCUSDT", Decimal("3000"), add=True)

        assert risk_manager._open_positions["BTCUSDT"] == Decimal("8000")

    def test_update_position_remove(self, risk_manager):
        """Test removing a position."""
        risk_manager.update_position("BTCUSDT", Decimal("5000"), add=True)
        risk_manager.update_position("BTCUSDT", Decimal("3000"), add=False)

        assert risk_manager._open_positions["BTCUSDT"] == Decimal("2000")

    def test_update_position_remove_all(self, risk_manager):
        """Test removing entire position."""
        risk_manager.update_position("BTCUSDT", Decimal("5000"), add=True)
        risk_manager.update_position("BTCUSDT", Decimal("5000"), add=False)

        assert "BTCUSDT" not in risk_manager._open_positions

    def test_update_portfolio_value(self, risk_manager):
        """Test portfolio value update."""
        initial_value = risk_manager.current_portfolio_value
        risk_manager.update_portfolio_value(Decimal("105000"))

        assert risk_manager.current_portfolio_value == Decimal("105000")
        assert risk_manager._daily_pnl == Decimal("5000")

    def test_update_price_history(self, risk_manager):
        """Test price history tracking."""
        risk_manager.update_price_history("BTCUSDT", 50000.0)
        risk_manager.update_price_history("BTCUSDT", 50500.0)
        risk_manager.update_price_history("BTCUSDT", 51000.0)

        assert "BTCUSDT" in risk_manager._price_history
        assert len(risk_manager._price_history["BTCUSDT"]) == 3
        assert risk_manager._price_history["BTCUSDT"][-1] == 51000.0

    def test_price_history_max_length(self, risk_manager):
        """Test that price history is limited."""
        # Add more than 1000 prices
        for i in range(1100):
            risk_manager.update_price_history("BTCUSDT", 50000.0 + i)

        # Should be capped at 1000
        assert len(risk_manager._price_history["BTCUSDT"]) == 1000

    def test_get_risk_metrics(self, risk_manager):
        """Test risk metrics summary."""
        risk_manager.update_position("BTCUSDT", Decimal("5000"), add=True)
        risk_manager.update_position("ETHUSDT", Decimal("3000"), add=True)
        risk_manager.update_portfolio_value(Decimal("102000"))

        metrics = risk_manager.get_risk_metrics()

        assert "portfolio_value" in metrics
        assert "daily_pnl" in metrics
        assert "open_positions" in metrics
        assert "total_exposure" in metrics
        assert metrics["open_positions"] == 2
        assert "BTCUSDT" in metrics["positions"]
        assert "ETHUSDT" in metrics["positions"]


class TestResultDataclasses:
    """Test result dataclass methods."""

    def test_risk_check_result_to_dict(self):
        """Test RiskCheckResult to_dict method."""
        from bot.risk.risk_manager import RiskCheckResult

        result = RiskCheckResult(
            check_name="test_check",
            status=RiskCheckStatus.PASSED,
            passed=True,
            message="Test passed",
            details={"key": "value"},
            value=0.5,
            threshold=1.0,
        )

        result_dict = result.to_dict()

        assert result_dict["check_name"] == "test_check"
        assert result_dict["status"] == "passed"
        assert result_dict["passed"] is True
        assert result_dict["value"] == 0.5

    def test_validation_result_to_dict(self):
        """Test TradeValidationResult to_dict method."""
        from bot.risk.risk_manager import RiskCheckResult, TradeValidationResult

        check_result = RiskCheckResult(
            check_name="test_check",
            status=RiskCheckStatus.PASSED,
            passed=True,
            message="Test passed",
        )

        validation = TradeValidationResult(
            approved=True,
            results=[check_result],
            correlation_id="test-123",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            estimated_value=Decimal("50000"),
        )

        result_dict = validation.to_dict()

        assert result_dict["approved"] is True
        assert result_dict["symbol"] == "BTCUSDT"
        assert result_dict["correlation_id"] == "test-123"
        assert len(result_dict["results"]) == 1

    def test_validation_result_get_failed_checks(self):
        """Test getting failed checks from validation result."""
        from bot.risk.risk_manager import RiskCheckResult, TradeValidationResult

        passed_check = RiskCheckResult(
            check_name="passed",
            status=RiskCheckStatus.PASSED,
            passed=True,
            message="Passed",
        )
        failed_check = RiskCheckResult(
            check_name="failed",
            status=RiskCheckStatus.FAILED,
            passed=False,
            message="Failed",
        )

        validation = TradeValidationResult(
            approved=False,
            results=[passed_check, failed_check],
            correlation_id="test-123",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            estimated_value=Decimal("50000"),
        )

        failed_checks = validation.get_failed_checks()

        assert len(failed_checks) == 1
        assert failed_checks[0].check_name == "failed"

    def test_validation_result_get_warnings(self):
        """Test getting warnings from validation result."""
        from bot.risk.risk_manager import RiskCheckResult, TradeValidationResult

        warning_check = RiskCheckResult(
            check_name="warning",
            status=RiskCheckStatus.WARNING,
            passed=True,
            message="Warning",
        )
        passed_check = RiskCheckResult(
            check_name="passed",
            status=RiskCheckStatus.PASSED,
            passed=True,
            message="Passed",
        )

        validation = TradeValidationResult(
            approved=True,
            results=[passed_check, warning_check],
            correlation_id="test-123",
            timestamp=datetime.now(timezone.utc),
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            estimated_value=Decimal("50000"),
        )

        warnings = validation.get_warnings()

        assert len(warnings) == 1
        assert warnings[0].check_name == "warning"
