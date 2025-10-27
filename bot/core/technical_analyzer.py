"""
Comprehensive Technical Analysis Module

Provides multi-timeframe technical indicator calculations with efficient caching
and pattern detection capabilities.

Features:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA/EMA for 20, 50, 200 periods)
- ATR (Average True Range)
- Support/Resistance level detection
- Multi-timeframe analysis (1h, 4h, 1d)
- Efficient indicator caching
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
import structlog
import ta

log = structlog.get_logger(__name__)


class Timeframe(str, Enum):
    """Supported timeframes for multi-timeframe analysis"""

    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"

    def to_minutes(self) -> int:
        """Convert timeframe to minutes for resampling"""
        match self:
            case Timeframe.ONE_HOUR:
                return 60
            case Timeframe.FOUR_HOUR:
                return 240
            case Timeframe.ONE_DAY:
                return 1440


class TrendDirection(str, Enum):
    """Trend direction classification"""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class IndicatorValues:
    """Container for all technical indicator values at a specific point in time"""

    timestamp: int  # Unix milliseconds
    price: float

    # RSI
    rsi: float | None = None
    rsi_oversold: bool = False
    rsi_overbought: bool = False

    # MACD
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    macd_bullish_crossover: bool = False
    macd_bearish_crossover: bool = False

    # Bollinger Bands
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_bandwidth: float | None = None
    bb_percent: float | None = None  # Position within bands (0-1)

    # Moving Averages
    ma_20: float | None = None
    ma_50: float | None = None
    ma_200: float | None = None
    ma_trend: TrendDirection = TrendDirection.NEUTRAL

    # ATR (volatility)
    atr: float | None = None
    atr_percent: float | None = None  # ATR as percentage of price

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SupportResistanceLevel:
    """Detected support or resistance level"""

    price: float
    strength: int  # Number of times price tested this level
    last_test_timestamp: int  # Unix milliseconds
    level_type: str  # "support" or "resistance"
    touches: list[int] = field(default_factory=list)  # Timestamps of touches


@dataclass
class MultiTimeframeAnalysis:
    """Analysis results across multiple timeframes"""

    symbol: str
    timestamp: int  # Unix milliseconds
    timeframes: dict[Timeframe, IndicatorValues]
    support_levels: list[SupportResistanceLevel] = field(default_factory=list)
    resistance_levels: list[SupportResistanceLevel] = field(default_factory=list)
    overall_trend: TrendDirection = TrendDirection.NEUTRAL


class TechnicalAnalyzer:
    """
    Comprehensive technical analysis engine with multi-timeframe support
    and efficient indicator caching.

    This analyzer computes technical indicators using the `ta` library and
    provides pattern detection for support/resistance levels.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        ma_periods: list[int] | None = None,
        atr_period: int = 14,
        support_resistance_lookback: int = 100,
        support_resistance_tolerance: float = 0.02,  # 2% price tolerance
        cache_ttl_seconds: int = 300,  # 5 minutes
    ):
        """
        Initialize technical analyzer with configurable parameters.

        Args:
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold (default 30)
            rsi_overbought: RSI overbought threshold (default 70)
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            ma_periods: Moving average periods (default [20, 50, 200])
            atr_period: ATR calculation period
            support_resistance_lookback: Number of candles to look back for S/R
            support_resistance_tolerance: Price tolerance for clustering (0.02 = 2%)
            cache_ttl_seconds: Cache time-to-live in seconds
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.ma_periods = ma_periods or [20, 50, 200]
        self.atr_period = atr_period
        self.support_resistance_lookback = support_resistance_lookback
        self.support_resistance_tolerance = support_resistance_tolerance
        self.cache_ttl_seconds = cache_ttl_seconds

        # Cache for computed indicators: {(symbol, timeframe, cache_key): (timestamp, result)}
        self._indicator_cache: dict[tuple[str, str, str], tuple[int, Any]] = {}

        # Previous MACD values for crossover detection: {(symbol, timeframe): (macd, signal)}
        self._prev_macd: dict[tuple[str, str], tuple[float, float]] = {}

        log.info(
            "technical_analyzer_initialized",
            rsi_period=rsi_period,
            macd_periods=(macd_fast, macd_slow, macd_signal),
            bb_period=bb_period,
            ma_periods=self.ma_periods,
            atr_period=atr_period,
        )

    def analyze(
        self,
        data: pd.DataFrame,
        symbol: str = "BTC/USDT",
        timeframe: Timeframe = Timeframe.ONE_HOUR,
    ) -> IndicatorValues:
        """
        Compute all technical indicators for the given data.

        Args:
            data: DataFrame with columns [timestamp, open, high, low, close, volume]
            symbol: Trading pair symbol
            timeframe: Timeframe of the data

        Returns:
            IndicatorValues containing all computed indicators

        Raises:
            ValueError: If data is insufficient for calculations
        """
        min_required = max(
            self.rsi_period,
            self.macd_slow + self.macd_signal,
            self.bb_period,
            max(self.ma_periods),
            self.atr_period,
        )
        if len(data) < min_required:
            raise ValueError(
                f"Insufficient data for technical analysis. "
                f"Need at least {min_required} rows, got {len(data)}."
            )

        # Ensure data is sorted by timestamp
        data = data.sort_values("timestamp").copy()

        # Extract latest values
        latest = data.iloc[-1]
        timestamp = int(latest["timestamp"])
        price = float(latest["close"])

        # Compute indicators with caching
        rsi_value = self._compute_rsi(data, symbol, timeframe)
        macd_values = self._compute_macd(data, symbol, timeframe)
        bb_values = self._compute_bollinger_bands(data, symbol, timeframe)
        ma_values = self._compute_moving_averages(data, symbol, timeframe)
        atr_value = self._compute_atr(data, symbol, timeframe)

        # Determine MACD crossovers
        macd_bullish = False
        macd_bearish = False
        if macd_values["macd"] is not None and macd_values["signal"] is not None:
            cache_key = (symbol, timeframe.value)
            if cache_key in self._prev_macd:
                prev_macd, prev_signal = self._prev_macd[cache_key]
                curr_macd = macd_values["macd"]
                curr_signal = macd_values["signal"]

                # Bullish crossover: MACD crosses above signal
                if prev_macd <= prev_signal and curr_macd > curr_signal:
                    macd_bullish = True

                # Bearish crossover: MACD crosses below signal
                if prev_macd >= prev_signal and curr_macd < curr_signal:
                    macd_bearish = True

            # Update previous values
            self._prev_macd[cache_key] = (macd_values["macd"], macd_values["signal"])

        # Determine MA trend
        ma_trend = self._determine_ma_trend(ma_values, price)

        # Build result
        result = IndicatorValues(
            timestamp=timestamp,
            price=price,
            rsi=rsi_value,
            rsi_oversold=(rsi_value < self.rsi_oversold if rsi_value is not None else False),
            rsi_overbought=(rsi_value > self.rsi_overbought if rsi_value is not None else False),
            macd=macd_values["macd"],
            macd_signal=macd_values["signal"],
            macd_histogram=macd_values["histogram"],
            macd_bullish_crossover=macd_bullish,
            macd_bearish_crossover=macd_bearish,
            bb_upper=bb_values["upper"],
            bb_middle=bb_values["middle"],
            bb_lower=bb_values["lower"],
            bb_bandwidth=bb_values["bandwidth"],
            bb_percent=bb_values["percent"],
            ma_20=ma_values.get("ma_20"),
            ma_50=ma_values.get("ma_50"),
            ma_200=ma_values.get("ma_200"),
            ma_trend=ma_trend,
            atr=atr_value,
            atr_percent=(atr_value / price * 100.0 if atr_value is not None else None),
        )

        log.debug(
            "technical_analysis_complete",
            symbol=symbol,
            timeframe=timeframe.value,
            rsi=rsi_value,
            macd_histogram=macd_values["histogram"],
            bb_percent=bb_values["percent"],
            ma_trend=ma_trend.value,
        )

        return result

    def analyze_multi_timeframe(
        self,
        data_by_timeframe: dict[Timeframe, pd.DataFrame],
        symbol: str = "BTC/USDT",
    ) -> MultiTimeframeAnalysis:
        """
        Perform multi-timeframe analysis across 1h, 4h, and 1d charts.

        Args:
            data_by_timeframe: Dict mapping Timeframe to OHLCV DataFrame
            symbol: Trading pair symbol

        Returns:
            MultiTimeframeAnalysis with indicators for each timeframe
        """
        timeframe_indicators: dict[Timeframe, IndicatorValues] = {}

        for timeframe, data in data_by_timeframe.items():
            try:
                indicators = self.analyze(data, symbol, timeframe)
                timeframe_indicators[timeframe] = indicators
            except ValueError as e:
                log.warning(
                    "insufficient_data_for_timeframe",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    error=str(e),
                )
                continue

        # Detect support and resistance levels using daily data if available
        support_levels: list[SupportResistanceLevel] = []
        resistance_levels: list[SupportResistanceLevel] = []

        if Timeframe.ONE_DAY in data_by_timeframe:
            support_levels, resistance_levels = self.detect_support_resistance(
                data_by_timeframe[Timeframe.ONE_DAY], symbol
            )

        # Determine overall trend from higher timeframes
        overall_trend = TrendDirection.NEUTRAL
        if Timeframe.ONE_DAY in timeframe_indicators:
            overall_trend = timeframe_indicators[Timeframe.ONE_DAY].ma_trend
        elif Timeframe.FOUR_HOUR in timeframe_indicators:
            overall_trend = timeframe_indicators[Timeframe.FOUR_HOUR].ma_trend

        result = MultiTimeframeAnalysis(
            symbol=symbol,
            timestamp=int(datetime.now().timestamp() * 1000),
            timeframes=timeframe_indicators,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            overall_trend=overall_trend,
        )

        log.info(
            "multi_timeframe_analysis_complete",
            symbol=symbol,
            timeframes_analyzed=len(timeframe_indicators),
            support_levels=len(support_levels),
            resistance_levels=len(resistance_levels),
            overall_trend=overall_trend.value,
        )

        return result

    def detect_support_resistance(
        self, data: pd.DataFrame, symbol: str = "BTC/USDT"
    ) -> tuple[list[SupportResistanceLevel], list[SupportResistanceLevel]]:
        """
        Detect support and resistance levels using swing high/low clustering.

        Algorithm:
        1. Find local maxima (resistance) and minima (support)
        2. Cluster nearby levels within tolerance percentage
        3. Count touches to determine level strength

        Args:
            data: DataFrame with OHLCV data
            symbol: Trading pair symbol

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if len(data) < self.support_resistance_lookback:
            log.warning(
                "insufficient_data_for_support_resistance",
                symbol=symbol,
                data_length=len(data),
                required=self.support_resistance_lookback,
            )
            return [], []

        # Use recent data
        recent_data = data.tail(self.support_resistance_lookback).copy()

        # Find local minima (support)
        support_candidates: list[tuple[float, int]] = []
        for i in range(1, len(recent_data) - 1):
            if (
                recent_data.iloc[i]["low"] < recent_data.iloc[i - 1]["low"]
                and recent_data.iloc[i]["low"] < recent_data.iloc[i + 1]["low"]
            ):
                price = float(recent_data.iloc[i]["low"])
                timestamp = int(recent_data.iloc[i]["timestamp"])
                support_candidates.append((price, timestamp))

        # Find local maxima (resistance)
        resistance_candidates: list[tuple[float, int]] = []
        for i in range(1, len(recent_data) - 1):
            if (
                recent_data.iloc[i]["high"] > recent_data.iloc[i - 1]["high"]
                and recent_data.iloc[i]["high"] > recent_data.iloc[i + 1]["high"]
            ):
                price = float(recent_data.iloc[i]["high"])
                timestamp = int(recent_data.iloc[i]["timestamp"])
                resistance_candidates.append((price, timestamp))

        # Cluster and create levels
        support_levels = self._cluster_levels(support_candidates, "support")
        resistance_levels = self._cluster_levels(resistance_candidates, "resistance")

        # Sort by strength
        support_levels.sort(key=lambda x: x.strength, reverse=True)
        resistance_levels.sort(key=lambda x: x.strength, reverse=True)

        log.debug(
            "support_resistance_detected",
            symbol=symbol,
            support_count=len(support_levels),
            resistance_count=len(resistance_levels),
        )

        return support_levels, resistance_levels

    def _cluster_levels(
        self, candidates: list[tuple[float, int]], level_type: str
    ) -> list[SupportResistanceLevel]:
        """
        Cluster nearby price levels within tolerance.

        Args:
            candidates: List of (price, timestamp) tuples
            level_type: "support" or "resistance"

        Returns:
            List of clustered SupportResistanceLevel objects
        """
        if not candidates:
            return []

        # Sort by price
        candidates_sorted = sorted(candidates, key=lambda x: x[0])

        clusters: list[list[tuple[float, int]]] = []
        current_cluster: list[tuple[float, int]] = [candidates_sorted[0]]

        for i in range(1, len(candidates_sorted)):
            price, timestamp = candidates_sorted[i]
            cluster_avg = sum(p for p, _ in current_cluster) / len(current_cluster)

            # Check if within tolerance
            if abs(price - cluster_avg) / cluster_avg <= self.support_resistance_tolerance:
                current_cluster.append((price, timestamp))
            else:
                clusters.append(current_cluster)
                current_cluster = [(price, timestamp)]

        # Add last cluster
        if current_cluster:
            clusters.append(current_cluster)

        # Convert clusters to levels
        levels: list[SupportResistanceLevel] = []
        for cluster in clusters:
            if len(cluster) >= 2:  # Minimum 2 touches to be significant
                avg_price = sum(p for p, _ in cluster) / len(cluster)
                timestamps = [ts for _, ts in cluster]
                levels.append(
                    SupportResistanceLevel(
                        price=avg_price,
                        strength=len(cluster),
                        last_test_timestamp=max(timestamps),
                        level_type=level_type,
                        touches=timestamps,
                    )
                )

        return levels

    def _compute_rsi(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> float | None:
        """Compute RSI with caching"""
        cache_key = (symbol, timeframe.value, "rsi")

        # Check cache
        if cache_key in self._indicator_cache:
            cached_timestamp, cached_value = self._indicator_cache[cache_key]
            if self._is_cache_valid(cached_timestamp):
                return cached_value  # type: ignore[no-any-return]

        try:
            rsi_indicator = ta.momentum.RSIIndicator(close=data["close"], window=self.rsi_period)
            rsi_series = rsi_indicator.rsi()
            rsi_value = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else None

            # Update cache
            self._indicator_cache[cache_key] = (int(datetime.now().timestamp()), rsi_value)

            return rsi_value
        except Exception as e:
            log.error("rsi_calculation_error", symbol=symbol, error=str(e))
            return None

    def _compute_macd(
        self, data: pd.DataFrame, symbol: str, timeframe: Timeframe
    ) -> dict[str, float | None]:
        """Compute MACD with caching"""
        cache_key = (symbol, timeframe.value, "macd")

        if cache_key in self._indicator_cache:
            cached_timestamp, cached_value = self._indicator_cache[cache_key]
            if self._is_cache_valid(cached_timestamp):
                return cached_value  # type: ignore[no-any-return]

        try:
            macd_indicator = ta.trend.MACD(
                close=data["close"],
                window_slow=self.macd_slow,
                window_fast=self.macd_fast,
                window_sign=self.macd_signal,
            )

            macd_line = macd_indicator.macd()
            signal_line = macd_indicator.macd_signal()
            histogram = macd_indicator.macd_diff()

            result = {
                "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else None,
                "signal": (
                    float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else None
                ),
                "histogram": (
                    float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else None
                ),
            }

            self._indicator_cache[cache_key] = (int(datetime.now().timestamp()), result)

            return result
        except Exception as e:
            log.error("macd_calculation_error", symbol=symbol, error=str(e))
            return {"macd": None, "signal": None, "histogram": None}

    def _compute_bollinger_bands(
        self, data: pd.DataFrame, symbol: str, timeframe: Timeframe
    ) -> dict[str, float | None]:
        """Compute Bollinger Bands with caching"""
        cache_key = (symbol, timeframe.value, "bb")

        if cache_key in self._indicator_cache:
            cached_timestamp, cached_value = self._indicator_cache[cache_key]
            if self._is_cache_valid(cached_timestamp):
                return cached_value  # type: ignore[no-any-return]

        try:
            bb_indicator = ta.volatility.BollingerBands(
                close=data["close"], window=self.bb_period, window_dev=self.bb_std
            )

            upper = bb_indicator.bollinger_hband()
            middle = bb_indicator.bollinger_mavg()
            lower = bb_indicator.bollinger_lband()

            upper_val = float(upper.iloc[-1]) if not pd.isna(upper.iloc[-1]) else None
            middle_val = float(middle.iloc[-1]) if not pd.isna(middle.iloc[-1]) else None
            lower_val = float(lower.iloc[-1]) if not pd.isna(lower.iloc[-1]) else None

            # Calculate bandwidth and position
            bandwidth = None
            percent = None
            if upper_val is not None and lower_val is not None and middle_val is not None:
                bandwidth = (upper_val - lower_val) / middle_val
                price = float(data.iloc[-1]["close"])
                if upper_val != lower_val:
                    percent = (price - lower_val) / (upper_val - lower_val)

            result = {
                "upper": upper_val,
                "middle": middle_val,
                "lower": lower_val,
                "bandwidth": bandwidth,
                "percent": percent,
            }

            self._indicator_cache[cache_key] = (int(datetime.now().timestamp()), result)

            return result
        except Exception as e:
            log.error("bollinger_bands_calculation_error", symbol=symbol, error=str(e))
            return {
                "upper": None,
                "middle": None,
                "lower": None,
                "bandwidth": None,
                "percent": None,
            }

    def _compute_moving_averages(
        self, data: pd.DataFrame, symbol: str, timeframe: Timeframe
    ) -> dict[str, float | None]:
        """Compute moving averages with caching"""
        cache_key = (symbol, timeframe.value, "ma")

        if cache_key in self._indicator_cache:
            cached_timestamp, cached_value = self._indicator_cache[cache_key]
            if self._is_cache_valid(cached_timestamp):
                return cached_value  # type: ignore[no-any-return]

        result: dict[str, float | None] = {}

        try:
            for period in self.ma_periods:
                if len(data) >= period:
                    sma = ta.trend.SMAIndicator(close=data["close"], window=period)
                    ma_series = sma.sma_indicator()
                    ma_value = (
                        float(ma_series.iloc[-1]) if not pd.isna(ma_series.iloc[-1]) else None
                    )
                    result[f"ma_{period}"] = ma_value
                else:
                    result[f"ma_{period}"] = None

            self._indicator_cache[cache_key] = (int(datetime.now().timestamp()), result)

            return result
        except Exception as e:
            log.error("moving_average_calculation_error", symbol=symbol, error=str(e))
            return {f"ma_{p}": None for p in self.ma_periods}

    def _compute_atr(self, data: pd.DataFrame, symbol: str, timeframe: Timeframe) -> float | None:
        """Compute ATR with caching"""
        cache_key = (symbol, timeframe.value, "atr")

        if cache_key in self._indicator_cache:
            cached_timestamp, cached_value = self._indicator_cache[cache_key]
            if self._is_cache_valid(cached_timestamp):
                return cached_value  # type: ignore[no-any-return]

        try:
            atr_indicator = ta.volatility.AverageTrueRange(
                high=data["high"], low=data["low"], close=data["close"], window=self.atr_period
            )
            atr_series = atr_indicator.average_true_range()
            atr_value = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else None

            self._indicator_cache[cache_key] = (int(datetime.now().timestamp()), atr_value)

            return atr_value
        except Exception as e:
            log.error("atr_calculation_error", symbol=symbol, error=str(e))
            return None

    def _determine_ma_trend(
        self, ma_values: dict[str, float | None], current_price: float
    ) -> TrendDirection:
        """
        Determine trend direction based on moving averages.

        Bullish: price > MA20 > MA50 > MA200
        Bearish: price < MA20 < MA50 < MA200
        Neutral: otherwise
        """
        ma_20 = ma_values.get("ma_20")
        ma_50 = ma_values.get("ma_50")
        ma_200 = ma_values.get("ma_200")

        if ma_20 is None or ma_50 is None or ma_200 is None:
            return TrendDirection.NEUTRAL

        # Bullish alignment
        if current_price > ma_20 > ma_50 > ma_200:
            return TrendDirection.BULLISH

        # Bearish alignment
        if current_price < ma_20 < ma_50 < ma_200:
            return TrendDirection.BEARISH

        return TrendDirection.NEUTRAL

    def _is_cache_valid(self, cached_timestamp: int) -> bool:
        """Check if cached value is still valid based on TTL"""
        current_timestamp = int(datetime.now().timestamp())
        return (current_timestamp - cached_timestamp) < self.cache_ttl_seconds

    def clear_cache(self, symbol: str | None = None, timeframe: Timeframe | None = None) -> None:
        """
        Clear indicator cache.

        Args:
            symbol: If provided, only clear cache for this symbol
            timeframe: If provided, only clear cache for this timeframe
        """
        if symbol is None and timeframe is None:
            self._indicator_cache.clear()
            self._prev_macd.clear()
            log.info("cache_cleared", scope="all")
        else:
            keys_to_remove = [
                key
                for key in self._indicator_cache.keys()
                if (symbol is None or key[0] == symbol)
                and (timeframe is None or key[1] == timeframe.value)
            ]
            for key in keys_to_remove:
                del self._indicator_cache[key]

            macd_keys_to_remove: list[tuple[str, str]] = [
                key
                for key in self._prev_macd.keys()
                if (symbol is None or key[0] == symbol)
                and (timeframe is None or key[1] == timeframe.value)
            ]
            for macd_key in macd_keys_to_remove:
                del self._prev_macd[macd_key]

            log.info(
                "cache_cleared",
                scope="filtered",
                symbol=symbol,
                timeframe=timeframe.value if timeframe else None,
                items_removed=len(keys_to_remove) + len(macd_keys_to_remove),
            )


def resample_ohlcv(data: pd.DataFrame, target_timeframe: Timeframe) -> pd.DataFrame:
    """
    Resample OHLCV data to a different timeframe.

    Args:
        data: DataFrame with columns [timestamp, open, high, low, close, volume]
        target_timeframe: Target timeframe to resample to

    Returns:
        Resampled DataFrame with same columns
    """
    # Convert timestamp to datetime
    df = data.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime")

    # Resample
    minutes = target_timeframe.to_minutes()
    resampled = df.resample(f"{minutes}T").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    # Drop NaN rows and reset index
    resampled = resampled.dropna()
    resampled["timestamp"] = resampled.index.astype(int) // 10**6  # Convert to milliseconds
    resampled = resampled.reset_index(drop=True)

    return resampled[["timestamp", "open", "high", "low", "close", "volume"]]
