"""
Data Quality Monitor for market data validation.

This module provides comprehensive data quality monitoring with:
- Price sanity checks (>10% spikes without validation)
- Volume anomaly detection (>5x average)
- Timestamp freshness validation (<60 seconds old)
- NaN/missing data detection
- Data quality score calculation
- Trading halt on poor data quality
- Alert notifications for data issues

Architecture:
- Real-time validation of incoming market data
- Historical baseline tracking for anomaly detection
- Configurable thresholds and quality metrics
- Integration with emergency stop system
"""

import asyncio
import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import pandas as pd
import structlog

from bot.core.logging_config import CorrelationContext

logger = structlog.get_logger(__name__)


class DataQualityStatus(str, Enum):
    """Data quality check status."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class DataQualityCheckType(str, Enum):
    """Types of data quality checks."""

    PRICE_SANITY = "price_sanity"
    VOLUME_ANOMALY = "volume_anomaly"
    TIMESTAMP_FRESHNESS = "timestamp_freshness"
    MISSING_DATA = "missing_data"
    DATA_COMPLETENESS = "data_completeness"


@dataclass
class DataQualityConfig:
    """
    Configuration for data quality monitoring.

    Attributes:
        price_spike_threshold_pct: Price spike threshold (default: 0.10 = 10%)
        price_spike_window_seconds: Time window for spike detection (default: 60 seconds)
        volume_anomaly_multiplier: Volume anomaly threshold (default: 5.0 = 5x average)
        volume_baseline_periods: Number of periods for volume baseline (default: 100)
        freshness_threshold_seconds: Stale data threshold (default: 60 seconds)
        min_data_points_for_checks: Minimum data points before running checks (default: 10)
        quality_score_fail_threshold: Quality score below this triggers halt (default: 0.5)
        enable_trading_halt: Enable automatic trading halt (default: True)
        halt_cooldown_seconds: Cooldown between halt triggers (default: 300 seconds)
    """

    price_spike_threshold_pct: float = 0.10
    price_spike_window_seconds: int = 60
    volume_anomaly_multiplier: float = 5.0
    volume_baseline_periods: int = 100
    freshness_threshold_seconds: int = 60
    min_data_points_for_checks: int = 10
    quality_score_fail_threshold: float = 0.5
    enable_trading_halt: bool = True
    halt_cooldown_seconds: int = 300


@dataclass
class DataQualityResult:
    """
    Result of a data quality check.

    Attributes:
        check_type: Type of quality check performed
        status: Check status (pass/warning/fail)
        score: Quality score (0.0 to 1.0)
        message: Human-readable description
        details: Additional check details
        timestamp: Check timestamp
        symbol: Trading pair symbol
        correlation_id: Correlation ID for tracking
    """

    check_type: DataQualityCheckType
    status: DataQualityStatus
    score: float
    message: str
    details: dict[str, Any]
    timestamp: datetime
    symbol: str
    correlation_id: str | None = None


@dataclass
class PriceDataPoint:
    """Price data point for spike detection."""

    price: float
    timestamp: datetime


@dataclass
class VolumeDataPoint:
    """Volume data point for anomaly detection."""

    volume: float
    timestamp: datetime


class DataQualityMonitor:
    """
    Data quality monitor with real-time validation and alerting.

    Features:
    - Price spike detection with configurable thresholds
    - Volume anomaly detection using baseline tracking
    - Timestamp freshness validation
    - NaN/missing data detection
    - Composite quality score calculation
    - Automatic trading halt on poor quality
    - Alert notifications via callbacks
    """

    def __init__(
        self,
        config: DataQualityConfig | None = None,
        alert_callback: Callable[[DataQualityResult], None] | None = None,
        halt_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        """
        Initialize data quality monitor.

        Args:
            config: Data quality configuration
            alert_callback: Optional callback for alerts (async function)
            halt_callback: Optional callback for trading halt (async function)
        """
        self.config = config or DataQualityConfig()
        self.alert_callback = alert_callback
        self.halt_callback = halt_callback

        # Price history for spike detection (symbol -> deque of PriceDataPoint)
        self._price_history: dict[str, deque[PriceDataPoint]] = {}

        # Volume history for anomaly detection (symbol -> deque of VolumeDataPoint)
        self._volume_history: dict[str, deque[VolumeDataPoint]] = {}

        # Last data update timestamps (symbol -> datetime)
        self._last_update: dict[str, datetime] = {}

        # Quality check results history
        self._check_results: list[DataQualityResult] = []

        # Trading halt state
        self._is_halted = False
        self._halt_lock = asyncio.Lock()
        self._last_halt_time: datetime | None = None

        # Quality metrics tracking
        self._quality_scores: dict[str, deque[float]] = {}  # symbol -> recent scores

        logger.info(
            "data_quality_monitor_initialized",
            config=asdict(self.config),
        )

    async def validate_tick(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: int,
    ) -> tuple[bool, float, list[DataQualityResult]]:
        """
        Validate a market tick for data quality.

        Args:
            symbol: Trading pair symbol
            price: Current price
            volume: Current volume
            timestamp: Timestamp in milliseconds

        Returns:
            Tuple of (is_valid, quality_score, check_results)
        """
        with CorrelationContext() as correlation_id:
            results: list[DataQualityResult] = []
            now = datetime.now(UTC)

            # Convert timestamp to datetime
            tick_time = datetime.fromtimestamp(timestamp / 1000, tz=UTC)

            # Check 1: Missing data detection
            missing_check = await self._check_missing_data(
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=now,
                correlation_id=correlation_id,
            )
            results.append(missing_check)

            # Check 2: Timestamp freshness
            freshness_check = await self._check_timestamp_freshness(
                symbol=symbol,
                tick_time=tick_time,
                current_time=now,
                correlation_id=correlation_id,
            )
            results.append(freshness_check)

            # Update price history
            await self._update_price_history(symbol, price, now)

            # Update volume history
            await self._update_volume_history(symbol, volume, now)

            # Update last update timestamp
            self._last_update[symbol] = now

            # Check 3: Price sanity (only if we have enough data)
            if len(self._price_history.get(symbol, [])) >= self.config.min_data_points_for_checks:
                price_check = await self._check_price_sanity(
                    symbol=symbol,
                    correlation_id=correlation_id,
                )
                results.append(price_check)

            # Check 4: Volume anomaly (only if we have enough data)
            if len(self._volume_history.get(symbol, [])) >= self.config.min_data_points_for_checks:
                volume_check = await self._check_volume_anomaly(
                    symbol=symbol,
                    correlation_id=correlation_id,
                )
                results.append(volume_check)

            # Calculate composite quality score
            quality_score = self._calculate_quality_score(results)

            # Track quality score
            if symbol not in self._quality_scores:
                self._quality_scores[symbol] = deque(maxlen=100)
            self._quality_scores[symbol].append(quality_score)

            # Store results
            self._check_results.extend(results)

            # Trim results history to last 1000
            if len(self._check_results) > 1000:
                self._check_results = self._check_results[-1000:]

            # Check if quality is acceptable
            is_valid = quality_score >= self.config.quality_score_fail_threshold

            # Log quality check
            logger.info(
                "data_quality_check_completed",
                symbol=symbol,
                quality_score=quality_score,
                is_valid=is_valid,
                checks_performed=len(results),
                correlation_id=correlation_id,
            )

            # Send alerts for failed checks
            for result in results:
                if result.status == DataQualityStatus.FAIL:
                    await self._send_alert(result)

            # Trigger halt if quality is poor
            if not is_valid and self.config.enable_trading_halt:
                await self._trigger_halt(symbol, quality_score, results)

            return is_valid, quality_score, results

    async def _check_missing_data(
        self,
        symbol: str,
        price: float,
        volume: float,
        timestamp: datetime,
        correlation_id: str,
    ) -> DataQualityResult:
        """
        Check for NaN or missing data values.

        Args:
            symbol: Trading pair symbol
            price: Price value
            volume: Volume value
            timestamp: Check timestamp
            correlation_id: Correlation ID

        Returns:
            DataQualityResult with check outcome
        """
        issues = []

        # Check for NaN values
        if math.isnan(price):
            issues.append("price_is_nan")
        if math.isnan(volume):
            issues.append("volume_is_nan")

        # Check for None values
        if price is None:
            issues.append("price_is_none")
        if volume is None:
            issues.append("volume_is_none")

        # Check for invalid values
        if price <= 0:
            issues.append("price_is_non_positive")
        if volume < 0:
            issues.append("volume_is_negative")

        if issues:
            return DataQualityResult(
                check_type=DataQualityCheckType.MISSING_DATA,
                status=DataQualityStatus.FAIL,
                score=0.0,
                message=f"Missing or invalid data detected for {symbol}",
                details={"issues": issues, "price": price, "volume": volume},
                timestamp=timestamp,
                symbol=symbol,
                correlation_id=correlation_id,
            )

        return DataQualityResult(
            check_type=DataQualityCheckType.MISSING_DATA,
            status=DataQualityStatus.PASS,
            score=1.0,
            message=f"No missing data for {symbol}",
            details={"price": price, "volume": volume},
            timestamp=timestamp,
            symbol=symbol,
            correlation_id=correlation_id,
        )

    async def _check_timestamp_freshness(
        self,
        symbol: str,
        tick_time: datetime,
        current_time: datetime,
        correlation_id: str,
    ) -> DataQualityResult:
        """
        Check if timestamp is fresh (<60 seconds old).

        Args:
            symbol: Trading pair symbol
            tick_time: Tick timestamp
            current_time: Current timestamp
            correlation_id: Correlation ID

        Returns:
            DataQualityResult with check outcome
        """
        age_seconds = (current_time - tick_time).total_seconds()

        if age_seconds > self.config.freshness_threshold_seconds:
            # Data is stale
            score = max(0.0, 1.0 - (age_seconds / self.config.freshness_threshold_seconds))
            return DataQualityResult(
                check_type=DataQualityCheckType.TIMESTAMP_FRESHNESS,
                status=DataQualityStatus.FAIL,
                score=score,
                message=f"Stale data for {symbol}: {age_seconds:.1f}s old",
                details={
                    "age_seconds": age_seconds,
                    "threshold_seconds": self.config.freshness_threshold_seconds,
                    "tick_time": tick_time.isoformat(),
                    "current_time": current_time.isoformat(),
                },
                timestamp=current_time,
                symbol=symbol,
                correlation_id=correlation_id,
            )

        # Check for future timestamps (clock skew)
        if age_seconds < -5:  # 5 second tolerance
            return DataQualityResult(
                check_type=DataQualityCheckType.TIMESTAMP_FRESHNESS,
                status=DataQualityStatus.WARNING,
                score=0.8,
                message=f"Future timestamp detected for {symbol}: {abs(age_seconds):.1f}s ahead",
                details={
                    "age_seconds": age_seconds,
                    "tick_time": tick_time.isoformat(),
                    "current_time": current_time.isoformat(),
                },
                timestamp=current_time,
                symbol=symbol,
                correlation_id=correlation_id,
            )

        return DataQualityResult(
            check_type=DataQualityCheckType.TIMESTAMP_FRESHNESS,
            status=DataQualityStatus.PASS,
            score=1.0,
            message=f"Timestamp is fresh for {symbol}",
            details={"age_seconds": age_seconds},
            timestamp=current_time,
            symbol=symbol,
            correlation_id=correlation_id,
        )

    async def _update_price_history(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Update price history for spike detection."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=1000)

        self._price_history[symbol].append(PriceDataPoint(price=price, timestamp=timestamp))

    async def _update_volume_history(
        self,
        symbol: str,
        volume: float,
        timestamp: datetime,
    ) -> None:
        """Update volume history for anomaly detection."""
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self.config.volume_baseline_periods)

        self._volume_history[symbol].append(VolumeDataPoint(volume=volume, timestamp=timestamp))

    async def _check_price_sanity(
        self,
        symbol: str,
        correlation_id: str,
    ) -> DataQualityResult:
        """
        Check for price spikes (>10% without validation).

        Args:
            symbol: Trading pair symbol
            correlation_id: Correlation ID

        Returns:
            DataQualityResult with check outcome
        """
        now = datetime.now(UTC)
        window_start = now - timedelta(seconds=self.config.price_spike_window_seconds)

        # Get recent prices within window
        recent_prices = [
            dp.price for dp in self._price_history[symbol] if dp.timestamp >= window_start
        ]

        if len(recent_prices) < 2:
            # Not enough data
            return DataQualityResult(
                check_type=DataQualityCheckType.PRICE_SANITY,
                status=DataQualityStatus.PASS,
                score=1.0,
                message=f"Insufficient data for price sanity check on {symbol}",
                details={"data_points": len(recent_prices)},
                timestamp=now,
                symbol=symbol,
                correlation_id=correlation_id,
            )

        # Calculate price change
        min_price = min(recent_prices)
        max_price = max(recent_prices)
        current_price = recent_prices[-1]

        if max_price > 0:
            price_change_pct = (max_price - min_price) / max_price

            if price_change_pct > self.config.price_spike_threshold_pct:
                # Price spike detected
                score = max(0.0, 1.0 - (price_change_pct / self.config.price_spike_threshold_pct))
                return DataQualityResult(
                    check_type=DataQualityCheckType.PRICE_SANITY,
                    status=DataQualityStatus.FAIL,
                    score=score,
                    message=f"Price spike detected for {symbol}: {price_change_pct:.2%} in {self.config.price_spike_window_seconds}s",
                    details={
                        "price_change_pct": price_change_pct,
                        "min_price": min_price,
                        "max_price": max_price,
                        "current_price": current_price,
                        "threshold_pct": self.config.price_spike_threshold_pct,
                        "window_seconds": self.config.price_spike_window_seconds,
                    },
                    timestamp=now,
                    symbol=symbol,
                    correlation_id=correlation_id,
                )

        return DataQualityResult(
            check_type=DataQualityCheckType.PRICE_SANITY,
            status=DataQualityStatus.PASS,
            score=1.0,
            message=f"Price sanity check passed for {symbol}",
            details={
                "min_price": min_price,
                "max_price": max_price,
                "current_price": current_price,
            },
            timestamp=now,
            symbol=symbol,
            correlation_id=correlation_id,
        )

    async def _check_volume_anomaly(
        self,
        symbol: str,
        correlation_id: str,
    ) -> DataQualityResult:
        """
        Check for volume anomalies (>5x average).

        Args:
            symbol: Trading pair symbol
            correlation_id: Correlation ID

        Returns:
            DataQualityResult with check outcome
        """
        now = datetime.now(UTC)

        volumes = [dp.volume for dp in self._volume_history[symbol]]

        if len(volumes) < self.config.min_data_points_for_checks:
            # Not enough data
            return DataQualityResult(
                check_type=DataQualityCheckType.VOLUME_ANOMALY,
                status=DataQualityStatus.PASS,
                score=1.0,
                message=f"Insufficient data for volume anomaly check on {symbol}",
                details={"data_points": len(volumes)},
                timestamp=now,
                symbol=symbol,
                correlation_id=correlation_id,
            )

        # Calculate baseline (excluding current volume)
        baseline_volumes = volumes[:-1] if len(volumes) > 1 else volumes
        avg_volume = sum(baseline_volumes) / len(baseline_volumes)
        current_volume = volumes[-1]

        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume

            if volume_ratio > self.config.volume_anomaly_multiplier:
                # Volume anomaly detected
                score = max(0.0, 1.0 - (volume_ratio / self.config.volume_anomaly_multiplier) + 1.0)
                return DataQualityResult(
                    check_type=DataQualityCheckType.VOLUME_ANOMALY,
                    status=DataQualityStatus.WARNING,
                    score=score,
                    message=f"Volume anomaly detected for {symbol}: {volume_ratio:.1f}x average",
                    details={
                        "current_volume": current_volume,
                        "average_volume": avg_volume,
                        "volume_ratio": volume_ratio,
                        "threshold_multiplier": self.config.volume_anomaly_multiplier,
                    },
                    timestamp=now,
                    symbol=symbol,
                    correlation_id=correlation_id,
                )

        return DataQualityResult(
            check_type=DataQualityCheckType.VOLUME_ANOMALY,
            status=DataQualityStatus.PASS,
            score=1.0,
            message=f"Volume check passed for {symbol}",
            details={
                "current_volume": current_volume,
                "average_volume": avg_volume,
            },
            timestamp=now,
            symbol=symbol,
            correlation_id=correlation_id,
        )

    def _calculate_quality_score(
        self,
        results: list[DataQualityResult],
    ) -> float:
        """
        Calculate composite quality score from check results.

        Score calculation:
        - Each check contributes its score weighted by importance
        - FAIL status has more impact than WARNING
        - Returns score between 0.0 (worst) and 1.0 (best)

        Args:
            results: List of check results

        Returns:
            Composite quality score (0.0 to 1.0)
        """
        if not results:
            return 1.0

        # Weight each check type
        weights = {
            DataQualityCheckType.MISSING_DATA: 1.5,  # Most critical
            DataQualityCheckType.TIMESTAMP_FRESHNESS: 1.2,
            DataQualityCheckType.PRICE_SANITY: 1.0,
            DataQualityCheckType.VOLUME_ANOMALY: 0.8,  # Less critical (warning only)
        }

        weighted_sum = 0.0
        total_weight = 0.0

        for result in results:
            weight = weights.get(result.check_type, 1.0)
            weighted_sum += result.score * weight
            total_weight += weight

        if total_weight == 0:
            return 1.0

        return weighted_sum / total_weight

    async def _send_alert(self, result: DataQualityResult) -> None:
        """
        Send alert notification for failed check.

        Args:
            result: Data quality check result
        """
        try:
            if self.alert_callback:
                if asyncio.iscoroutinefunction(self.alert_callback):
                    await self.alert_callback(result)
                else:
                    self.alert_callback(result)

            logger.warning(
                "data_quality_alert",
                check_type=result.check_type.value,
                status=result.status.value,
                symbol=result.symbol,
                message=result.message,
                correlation_id=result.correlation_id,
            )

        except Exception as e:
            logger.error(
                "alert_send_failed",
                error=str(e),
                check_type=result.check_type.value,
            )

    async def _trigger_halt(
        self,
        symbol: str,
        quality_score: float,
        results: list[DataQualityResult],
    ) -> None:
        """
        Trigger trading halt due to poor data quality.

        Args:
            symbol: Trading pair symbol
            quality_score: Current quality score
            results: Check results that triggered halt
        """
        async with self._halt_lock:
            # Check cooldown
            if self._last_halt_time:
                time_since_halt = (datetime.now(UTC) - self._last_halt_time).total_seconds()
                if time_since_halt < self.config.halt_cooldown_seconds:
                    logger.debug(
                        "halt_in_cooldown",
                        symbol=symbol,
                        time_since_halt=time_since_halt,
                    )
                    return

            self._is_halted = True
            self._last_halt_time = datetime.now(UTC)

            # Collect failed checks
            failed_checks = [
                {
                    "type": r.check_type.value,
                    "message": r.message,
                    "score": r.score,
                }
                for r in results
                if r.status == DataQualityStatus.FAIL
            ]

            details = {
                "symbol": symbol,
                "quality_score": quality_score,
                "threshold": self.config.quality_score_fail_threshold,
                "failed_checks": failed_checks,
            }

            logger.critical(
                "trading_halted_due_to_poor_data_quality",
                symbol=symbol,
                quality_score=quality_score,
                failed_checks_count=len(failed_checks),
            )

            # Call halt callback
            if self.halt_callback:
                try:
                    if asyncio.iscoroutinefunction(self.halt_callback):
                        await self.halt_callback(f"Poor data quality for {symbol}", details)
                    else:
                        self.halt_callback(f"Poor data quality for {symbol}", details)
                except Exception as e:
                    logger.error("halt_callback_failed", error=str(e))

    def is_halted(self) -> bool:
        """
        Check if trading is halted due to data quality issues.

        Returns:
            bool: True if halted, False otherwise
        """
        return self._is_halted

    async def resume(self, operator: str) -> None:
        """
        Resume trading after data quality halt.

        Args:
            operator: Name/ID of operator resuming trading
        """
        async with self._halt_lock:
            if not self._is_halted:
                logger.warning("resume_called_but_not_halted", operator=operator)
                return

            self._is_halted = False
            logger.info("trading_resumed_after_data_quality_halt", operator=operator)

    def get_quality_metrics(self, symbol: str | None = None) -> dict[str, Any]:
        """
        Get current data quality metrics.

        Args:
            symbol: Optional symbol filter

        Returns:
            Dictionary containing quality metrics
        """
        if symbol:
            return {
                "symbol": symbol,
                "is_halted": self._is_halted,
                "last_update": (
                    self._last_update[symbol].isoformat() if symbol in self._last_update else None
                ),
                "price_history_size": len(self._price_history.get(symbol, [])),
                "volume_history_size": len(self._volume_history.get(symbol, [])),
                "recent_quality_scores": list(self._quality_scores.get(symbol, []))[-10:],
                "average_quality_score": (
                    sum(self._quality_scores[symbol]) / len(self._quality_scores[symbol])
                    if symbol in self._quality_scores and len(self._quality_scores[symbol]) > 0
                    else None
                ),
                "config": asdict(self.config),
            }

        return {
            "is_halted": self._is_halted,
            "last_halt_time": (self._last_halt_time.isoformat() if self._last_halt_time else None),
            "monitored_symbols": list(self._price_history.keys()),
            "total_checks_performed": len(self._check_results),
            "config": asdict(self.config),
        }

    def get_check_history(
        self,
        symbol: str | None = None,
        limit: int = 100,
    ) -> list[DataQualityResult]:
        """
        Get recent data quality check results.

        Args:
            symbol: Optional symbol filter
            limit: Maximum number of results to return

        Returns:
            List of DataQualityResult objects
        """
        results = self._check_results

        if symbol:
            results = [r for r in results if r.symbol == symbol]

        return results[-limit:]

    async def validate_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
    ) -> tuple[bool, float, list[DataQualityResult]]:
        """
        Validate a DataFrame of market data.

        Args:
            df: DataFrame with columns: timestamp, price, volume
            symbol: Trading pair symbol

        Returns:
            Tuple of (is_valid, quality_score, check_results)
        """
        with CorrelationContext() as correlation_id:
            results: list[DataQualityResult] = []
            now = datetime.now(UTC)

            # Check for required columns
            required_columns = {"timestamp", "price", "volume"}
            if not required_columns.issubset(df.columns):
                result = DataQualityResult(
                    check_type=DataQualityCheckType.DATA_COMPLETENESS,
                    status=DataQualityStatus.FAIL,
                    score=0.0,
                    message=f"Missing required columns for {symbol}",
                    details={
                        "required": list(required_columns),
                        "present": list(df.columns),
                    },
                    timestamp=now,
                    symbol=symbol,
                    correlation_id=correlation_id,
                )
                results.append(result)
                return False, 0.0, results

            # Check for empty DataFrame
            if df.empty:
                result = DataQualityResult(
                    check_type=DataQualityCheckType.DATA_COMPLETENESS,
                    status=DataQualityStatus.FAIL,
                    score=0.0,
                    message=f"Empty DataFrame for {symbol}",
                    details={"rows": 0},
                    timestamp=now,
                    symbol=symbol,
                    correlation_id=correlation_id,
                )
                results.append(result)
                return False, 0.0, results

            # Check for NaN values
            nan_counts = df[["price", "volume"]].isna().sum()
            if nan_counts.sum() > 0:
                result = DataQualityResult(
                    check_type=DataQualityCheckType.MISSING_DATA,
                    status=DataQualityStatus.FAIL,
                    score=1.0 - (nan_counts.sum() / (len(df) * 2)),
                    message=f"NaN values detected in DataFrame for {symbol}",
                    details={
                        "nan_price_count": int(nan_counts["price"]),
                        "nan_volume_count": int(nan_counts["volume"]),
                        "total_rows": len(df),
                    },
                    timestamp=now,
                    symbol=symbol,
                    correlation_id=correlation_id,
                )
                results.append(result)
            else:
                result = DataQualityResult(
                    check_type=DataQualityCheckType.MISSING_DATA,
                    status=DataQualityStatus.PASS,
                    score=1.0,
                    message=f"No NaN values in DataFrame for {symbol}",
                    details={"total_rows": len(df)},
                    timestamp=now,
                    symbol=symbol,
                    correlation_id=correlation_id,
                )
                results.append(result)

            # Calculate quality score
            quality_score = self._calculate_quality_score(results)

            return quality_score >= self.config.quality_score_fail_threshold, quality_score, results
