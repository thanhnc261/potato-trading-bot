"""
Comprehensive unit tests for data quality monitoring module.

Tests cover:
- DataQualityMonitor initialization and configuration
- Price sanity checks (>10% spike detection)
- Volume anomaly detection (>5x average)
- Timestamp freshness validation (<60 seconds)
- NaN/missing data detection
- Quality score calculation
- Trading halt on poor data quality
- Alert notifications
- DataFrame validation
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from bot.data.quality_monitor import (
    DataQualityCheckType,
    DataQualityConfig,
    DataQualityMonitor,
    DataQualityStatus,
)


class TestDataQualityConfig:
    """Tests for DataQualityConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = DataQualityConfig()

        assert config.price_spike_threshold_pct == 0.10
        assert config.volume_anomaly_multiplier == 5.0
        assert config.freshness_threshold_seconds == 60
        assert config.min_data_points_for_checks == 10
        assert config.quality_score_fail_threshold == 0.5
        assert config.enable_trading_halt is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = DataQualityConfig(
            price_spike_threshold_pct=0.15,
            volume_anomaly_multiplier=10.0,
            freshness_threshold_seconds=120,
            enable_trading_halt=False,
        )

        assert config.price_spike_threshold_pct == 0.15
        assert config.volume_anomaly_multiplier == 10.0
        assert config.freshness_threshold_seconds == 120
        assert config.enable_trading_halt is False


class TestDataQualityMonitor:
    """Tests for DataQualityMonitor"""

    @pytest.mark.asyncio
    async def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = DataQualityMonitor()

        assert monitor.config is not None
        assert monitor.alert_callback is None
        assert monitor.halt_callback is None
        assert len(monitor._price_history) == 0
        assert len(monitor._volume_history) == 0
        assert monitor._is_halted is False

    @pytest.mark.asyncio
    async def test_monitor_with_callbacks(self):
        """Test monitor initialization with callbacks"""
        alert_callback = AsyncMock()
        halt_callback = AsyncMock()

        monitor = DataQualityMonitor(
            alert_callback=alert_callback,
            halt_callback=halt_callback,
        )

        assert monitor.alert_callback is alert_callback
        assert monitor.halt_callback is halt_callback

    @pytest.mark.asyncio
    async def test_validate_tick_valid_data(self):
        """Test validating a valid market tick"""
        monitor = DataQualityMonitor()

        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        is_valid, quality_score, results = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=45000.0,
            volume=100.0,
            timestamp=timestamp,
        )

        assert is_valid is True
        assert quality_score >= 0.5
        assert len(results) >= 2  # At least missing data and freshness checks

    @pytest.mark.asyncio
    async def test_validate_tick_nan_price(self):
        """Test validating a tick with NaN price"""
        monitor = DataQualityMonitor()

        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        is_valid, quality_score, results = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=float("nan"),
            volume=100.0,
            timestamp=timestamp,
        )

        assert is_valid is False
        assert quality_score < 0.5

        # Check that missing data check failed
        missing_data_checks = [
            r for r in results if r.check_type == DataQualityCheckType.MISSING_DATA
        ]
        assert len(missing_data_checks) > 0
        assert missing_data_checks[0].status == DataQualityStatus.FAIL

    @pytest.mark.asyncio
    async def test_validate_tick_negative_price(self):
        """Test validating a tick with negative price"""
        monitor = DataQualityMonitor()

        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        is_valid, quality_score, results = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=-45000.0,
            volume=100.0,
            timestamp=timestamp,
        )

        assert is_valid is False
        assert quality_score < 0.5

    @pytest.mark.asyncio
    async def test_validate_tick_stale_timestamp(self):
        """Test validating a tick with stale timestamp"""
        monitor = DataQualityMonitor()

        # Timestamp 2 minutes old
        stale_timestamp = int((datetime.now(UTC) - timedelta(minutes=2)).timestamp() * 1000)

        is_valid, quality_score, results = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=45000.0,
            volume=100.0,
            timestamp=stale_timestamp,
        )

        # Check that freshness check failed
        freshness_checks = [
            r for r in results if r.check_type == DataQualityCheckType.TIMESTAMP_FRESHNESS
        ]
        assert len(freshness_checks) > 0
        assert freshness_checks[0].status == DataQualityStatus.FAIL

        # Quality score should be impacted
        assert quality_score < 1.0

    @pytest.mark.asyncio
    async def test_price_spike_detection(self):
        """Test price spike detection (>10% change)"""
        config = DataQualityConfig(
            price_spike_threshold_pct=0.10,
            price_spike_window_seconds=30,  # Use shorter window for test
            min_data_points_for_checks=5,
        )
        monitor = DataQualityMonitor(config=config)

        # Feed normal prices (within 30 second window)
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        for i in range(10):
            await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0 + i * 10,  # Small increments
                volume=100.0,
                timestamp=timestamp + i * 100,  # 100ms apart
            )

        # Now feed a spike (>10% jump) still within window
        is_valid, quality_score, results = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=51000.0,  # ~12% increase from 45000
            volume=100.0,
            timestamp=timestamp + 1100,  # 1.1 seconds from start
        )

        # Check that price sanity check failed
        price_checks = [r for r in results if r.check_type == DataQualityCheckType.PRICE_SANITY]
        assert len(price_checks) > 0
        assert price_checks[0].status == DataQualityStatus.FAIL

    @pytest.mark.asyncio
    async def test_volume_anomaly_detection(self):
        """Test volume anomaly detection (>5x average)"""
        config = DataQualityConfig(
            volume_anomaly_multiplier=5.0,
            min_data_points_for_checks=5,
        )
        monitor = DataQualityMonitor(config=config)

        # Feed normal volumes
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        for i in range(15):
            await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0,
                volume=100.0,  # Consistent volume
                timestamp=timestamp + i * 1000,
            )

        # Now feed a volume spike (>5x)
        is_valid, quality_score, results = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=45000.0,
            volume=600.0,  # 6x average
            timestamp=timestamp + 16000,
        )

        # Check that volume anomaly check gave warning
        volume_checks = [r for r in results if r.check_type == DataQualityCheckType.VOLUME_ANOMALY]
        assert len(volume_checks) > 0
        assert volume_checks[0].status == DataQualityStatus.WARNING

    @pytest.mark.asyncio
    async def test_quality_score_calculation(self):
        """Test composite quality score calculation"""
        monitor = DataQualityMonitor()

        # Valid tick should have high quality score
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        is_valid, quality_score, _ = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=45000.0,
            volume=100.0,
            timestamp=timestamp,
        )

        assert quality_score >= 0.9  # Should be near perfect

        # Invalid tick should have low quality score
        is_valid, quality_score, _ = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=float("nan"),
            volume=-100.0,
            timestamp=timestamp - 120000,  # 2 minutes old
        )

        assert quality_score < 0.5  # Should be poor

    @pytest.mark.asyncio
    async def test_trading_halt_trigger(self):
        """Test that poor data quality triggers trading halt"""
        halt_callback = AsyncMock()
        config = DataQualityConfig(
            enable_trading_halt=True,
            quality_score_fail_threshold=0.5,
        )

        monitor = DataQualityMonitor(
            config=config,
            halt_callback=halt_callback,
        )

        # Feed bad data to trigger halt
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        await monitor.validate_tick(
            symbol="BTC/USDT",
            price=float("nan"),
            volume=100.0,
            timestamp=timestamp,
        )

        # Check that halt was triggered
        assert monitor.is_halted() is True
        halt_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_halt_cooldown(self):
        """Test halt cooldown prevents repeated halts"""
        halt_callback = AsyncMock()
        config = DataQualityConfig(
            enable_trading_halt=True,
            halt_cooldown_seconds=60,
        )

        monitor = DataQualityMonitor(
            config=config,
            halt_callback=halt_callback,
        )

        # First bad tick triggers halt
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        await monitor.validate_tick(
            symbol="BTC/USDT",
            price=float("nan"),
            volume=100.0,
            timestamp=timestamp,
        )

        assert halt_callback.call_count == 1

        # Second bad tick within cooldown should not trigger again
        await monitor.validate_tick(
            symbol="BTC/USDT",
            price=float("nan"),
            volume=100.0,
            timestamp=timestamp + 1000,
        )

        # Should still be 1 call (cooldown active)
        assert halt_callback.call_count == 1

    @pytest.mark.asyncio
    async def test_alert_callback_invocation(self):
        """Test that alert callback is invoked for failed checks"""
        alert_callback = AsyncMock()
        monitor = DataQualityMonitor(alert_callback=alert_callback)

        # Feed bad data
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        await monitor.validate_tick(
            symbol="BTC/USDT",
            price=float("nan"),
            volume=100.0,
            timestamp=timestamp,
        )

        # Alert should have been called
        assert alert_callback.called

    @pytest.mark.asyncio
    async def test_resume_trading(self):
        """Test resuming trading after halt"""
        monitor = DataQualityMonitor()

        # Manually trigger halt
        monitor._is_halted = True
        assert monitor.is_halted() is True

        # Resume
        await monitor.resume("test_operator")

        assert monitor.is_halted() is False

    @pytest.mark.asyncio
    async def test_get_quality_metrics_for_symbol(self):
        """Test getting quality metrics for a specific symbol"""
        monitor = DataQualityMonitor()

        # Feed some data
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        for i in range(5):
            await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0 + i,
                volume=100.0,
                timestamp=timestamp + i * 1000,
            )

        metrics = monitor.get_quality_metrics("BTC/USDT")

        assert metrics["symbol"] == "BTC/USDT"
        assert "last_update" in metrics
        assert metrics["price_history_size"] == 5
        assert metrics["volume_history_size"] == 5
        assert "recent_quality_scores" in metrics
        assert "average_quality_score" in metrics

    @pytest.mark.asyncio
    async def test_get_quality_metrics_global(self):
        """Test getting global quality metrics"""
        monitor = DataQualityMonitor()

        metrics = monitor.get_quality_metrics()

        assert "is_halted" in metrics
        assert "monitored_symbols" in metrics
        assert "total_checks_performed" in metrics
        assert "config" in metrics

    @pytest.mark.asyncio
    async def test_get_check_history(self):
        """Test retrieving check history"""
        monitor = DataQualityMonitor()

        # Feed some data
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        for i in range(3):
            await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0,
                volume=100.0,
                timestamp=timestamp + i * 1000,
            )

        # Get history
        history = monitor.get_check_history()

        assert len(history) > 0
        assert all(hasattr(r, "check_type") for r in history)
        assert all(hasattr(r, "status") for r in history)

    @pytest.mark.asyncio
    async def test_get_check_history_filtered_by_symbol(self):
        """Test retrieving check history filtered by symbol"""
        monitor = DataQualityMonitor()

        # Feed data for multiple symbols
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        await monitor.validate_tick(
            symbol="BTC/USDT",
            price=45000.0,
            volume=100.0,
            timestamp=timestamp,
        )
        await monitor.validate_tick(
            symbol="ETH/USDT",
            price=3000.0,
            volume=50.0,
            timestamp=timestamp,
        )

        # Get history for BTC only
        btc_history = monitor.get_check_history(symbol="BTC/USDT")

        assert all(r.symbol == "BTC/USDT" for r in btc_history)

    @pytest.mark.asyncio
    async def test_validate_dataframe_valid(self):
        """Test validating a valid DataFrame"""
        monitor = DataQualityMonitor()

        # Create valid DataFrame
        df = pd.DataFrame(
            {
                "timestamp": [1700000000000 + i * 1000 for i in range(10)],
                "price": [45000.0 + i for i in range(10)],
                "volume": [100.0 for _ in range(10)],
            }
        )

        is_valid, quality_score, results = await monitor.validate_dataframe(df, "BTC/USDT")

        assert is_valid is True
        assert quality_score >= 0.5

    @pytest.mark.asyncio
    async def test_validate_dataframe_missing_columns(self):
        """Test validating a DataFrame with missing columns"""
        monitor = DataQualityMonitor()

        # Create DataFrame missing volume column
        df = pd.DataFrame(
            {
                "timestamp": [1700000000000 + i * 1000 for i in range(10)],
                "price": [45000.0 + i for i in range(10)],
            }
        )

        is_valid, quality_score, results = await monitor.validate_dataframe(df, "BTC/USDT")

        assert is_valid is False
        assert quality_score == 0.0

        # Check for data completeness failure
        completeness_checks = [
            r for r in results if r.check_type == DataQualityCheckType.DATA_COMPLETENESS
        ]
        assert len(completeness_checks) > 0
        assert completeness_checks[0].status == DataQualityStatus.FAIL

    @pytest.mark.asyncio
    async def test_validate_dataframe_empty(self):
        """Test validating an empty DataFrame"""
        monitor = DataQualityMonitor()

        df = pd.DataFrame({"timestamp": [], "price": [], "volume": []})

        is_valid, quality_score, results = await monitor.validate_dataframe(df, "BTC/USDT")

        assert is_valid is False
        assert quality_score == 0.0

    @pytest.mark.asyncio
    async def test_validate_dataframe_with_nan(self):
        """Test validating a DataFrame with NaN values"""
        monitor = DataQualityMonitor()

        # Create DataFrame with NaN values
        df = pd.DataFrame(
            {
                "timestamp": [1700000000000 + i * 1000 for i in range(10)],
                "price": [45000.0 if i % 2 == 0 else float("nan") for i in range(10)],
                "volume": [100.0 for _ in range(10)],
            }
        )

        is_valid, quality_score, results = await monitor.validate_dataframe(df, "BTC/USDT")

        # Should fail due to NaN values
        missing_data_checks = [
            r for r in results if r.check_type == DataQualityCheckType.MISSING_DATA
        ]
        assert len(missing_data_checks) > 0
        assert missing_data_checks[0].status == DataQualityStatus.FAIL

    @pytest.mark.asyncio
    async def test_multiple_symbols_tracking(self):
        """Test monitoring multiple symbols simultaneously"""
        monitor = DataQualityMonitor()

        timestamp = int(datetime.now(UTC).timestamp() * 1000)

        # Feed data for multiple symbols
        symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
        for symbol in symbols:
            for i in range(5):
                await monitor.validate_tick(
                    symbol=symbol,
                    price=45000.0 + i,
                    volume=100.0,
                    timestamp=timestamp + i * 1000,
                )

        # Check that all symbols are tracked
        metrics = monitor.get_quality_metrics()
        assert len(metrics["monitored_symbols"]) == 3
        assert all(s in metrics["monitored_symbols"] for s in symbols)

    @pytest.mark.asyncio
    async def test_price_history_size_limit(self):
        """Test that price history respects maxlen"""
        monitor = DataQualityMonitor()

        # Feed more than 1000 ticks
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        for i in range(1100):
            await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0 + i,
                volume=100.0,
                timestamp=timestamp + i * 1000,
            )

        # Price history should be capped at 1000
        assert len(monitor._price_history["BTC/USDT"]) == 1000

    @pytest.mark.asyncio
    async def test_volume_history_baseline_size(self):
        """Test that volume history respects baseline periods"""
        config = DataQualityConfig(volume_baseline_periods=50)
        monitor = DataQualityMonitor(config=config)

        # Feed more than 50 ticks
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        for i in range(100):
            await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0,
                volume=100.0 + i,
                timestamp=timestamp + i * 1000,
            )

        # Volume history should be capped at 50
        assert len(monitor._volume_history["BTC/USDT"]) == 50

    @pytest.mark.asyncio
    async def test_future_timestamp_warning(self):
        """Test that future timestamps generate warnings"""
        monitor = DataQualityMonitor()

        # Future timestamp (10 seconds ahead)
        future_timestamp = int((datetime.now(UTC) + timedelta(seconds=10)).timestamp() * 1000)

        is_valid, quality_score, results = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=45000.0,
            volume=100.0,
            timestamp=future_timestamp,
        )

        # Check for warning
        freshness_checks = [
            r for r in results if r.check_type == DataQualityCheckType.TIMESTAMP_FRESHNESS
        ]
        assert len(freshness_checks) > 0
        assert freshness_checks[0].status == DataQualityStatus.WARNING

    @pytest.mark.asyncio
    async def test_check_history_limit(self):
        """Test that check history is limited to 1000 results"""
        monitor = DataQualityMonitor()

        # Feed enough data to generate >1000 check results
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        for i in range(300):  # Each tick generates multiple checks
            await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0,
                volume=100.0,
                timestamp=timestamp + i * 1000,
            )

        # Check results should be capped at 1000
        assert len(monitor._check_results) <= 1000

    @pytest.mark.asyncio
    async def test_synchronous_callback_support(self):
        """Test that synchronous callbacks are supported"""
        alert_callback = Mock()  # Synchronous callback
        monitor = DataQualityMonitor(alert_callback=alert_callback)

        # Feed bad data
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        await monitor.validate_tick(
            symbol="BTC/USDT",
            price=float("nan"),
            volume=100.0,
            timestamp=timestamp,
        )

        # Synchronous callback should have been called
        assert alert_callback.called


class TestIntegration:
    """Integration tests for complete workflow"""

    @pytest.mark.asyncio
    async def test_end_to_end_validation_workflow(self):
        """Test complete end-to-end validation workflow"""
        alert_callback = AsyncMock()
        halt_callback = AsyncMock()

        config = DataQualityConfig(
            price_spike_threshold_pct=0.10,
            volume_anomaly_multiplier=5.0,
            freshness_threshold_seconds=60,
            enable_trading_halt=True,
        )

        monitor = DataQualityMonitor(
            config=config,
            alert_callback=alert_callback,
            halt_callback=halt_callback,
        )

        # Feed good data
        timestamp = int(datetime.now(UTC).timestamp() * 1000)
        for i in range(20):
            is_valid, quality_score, _ = await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0 + i * 10,
                volume=100.0,
                timestamp=timestamp + i * 1000,
            )
            assert is_valid is True
            assert quality_score > 0.9

        # Feed bad data
        is_valid, quality_score, results = await monitor.validate_tick(
            symbol="BTC/USDT",
            price=float("nan"),
            volume=-100.0,
            timestamp=timestamp - 120000,  # Stale
        )

        # Should fail validation
        assert is_valid is False
        assert quality_score < 0.5

        # Should trigger alert
        assert alert_callback.called

        # Should trigger halt
        assert monitor.is_halted()
        assert halt_callback.called

        # Get metrics
        metrics = monitor.get_quality_metrics("BTC/USDT")
        assert metrics["symbol"] == "BTC/USDT"
        assert len(metrics["recent_quality_scores"]) > 0

        # Resume trading
        await monitor.resume("test_operator")
        assert not monitor.is_halted()

    @pytest.mark.asyncio
    async def test_multi_symbol_monitoring(self):
        """Test monitoring multiple symbols with different quality levels"""
        monitor = DataQualityMonitor()

        timestamp = int(datetime.now(UTC).timestamp() * 1000)

        # BTC - good quality
        for i in range(10):
            await monitor.validate_tick(
                symbol="BTC/USDT",
                price=45000.0 + i,
                volume=100.0,
                timestamp=timestamp + i * 1000,
            )

        # ETH - has price spike
        for i in range(10):
            await monitor.validate_tick(
                symbol="ETH/USDT",
                price=3000.0 if i < 9 else 3500.0,  # Spike at end
                volume=50.0,
                timestamp=timestamp + i * 1000,
            )

        # XRP - has stale data
        await monitor.validate_tick(
            symbol="XRP/USDT",
            price=0.5,
            volume=1000.0,
            timestamp=timestamp - 120000,  # 2 minutes old
        )

        # Check metrics
        btc_metrics = monitor.get_quality_metrics("BTC/USDT")
        xrp_metrics = monitor.get_quality_metrics("XRP/USDT")

        # BTC should have best quality
        btc_avg = btc_metrics["average_quality_score"]
        xrp_avg = xrp_metrics["average_quality_score"]

        assert btc_avg > xrp_avg  # BTC better than XRP (stale data)
