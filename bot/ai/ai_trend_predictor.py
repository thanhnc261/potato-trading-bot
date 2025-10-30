"""
AI Trend Predictor - LLM-powered trend prediction for cryptocurrency trading.

This module integrates with the LLM Manager to provide AI-powered trend predictions
based on technical indicator analysis. It constructs prompts incorporating indicator
summaries and enforces structured JSON output for reliable prediction parsing.

Features:
- Single timeframe trend prediction (initial implementation)
- Structured JSON output contract with direction, confidence, and reasoning
- Prediction caching to reduce repeated LLM calls
- Integration with TechnicalAnalyzer for indicator data
- Configurable confidence thresholds and temperature settings
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from bot.ai.llm_manager import LLMManager, LLMResponse
from bot.core.technical_analyzer import IndicatorValues, Timeframe

logger = structlog.get_logger(__name__)


class PredictionDirection(str, Enum):
    """Predicted trend direction for the asset"""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TrendPrediction:
    """
    AI-generated trend prediction with confidence and reasoning.

    Attributes:
        direction: Predicted trend direction (bullish, bearish, neutral)
        confidence: Confidence score between 0.0 and 1.0
        reasoning: Human-readable explanation of the prediction
        timeframe: Timeframe for which the prediction applies
        symbol: Trading pair symbol
        timestamp: Unix milliseconds when prediction was made
        indicators_used: Summary of indicators that influenced the prediction
        llm_response: Raw LLM response metadata (cost, tokens, latency)
        cached: Whether this prediction came from cache
    """

    direction: PredictionDirection
    confidence: float  # 0.0 to 1.0
    reasoning: str
    timeframe: Timeframe
    symbol: str
    timestamp: int
    indicators_used: dict[str, Any]
    llm_response: LLMResponse | None = None
    cached: bool = False

    def __post_init__(self) -> None:
        """Validate confidence is within valid range"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")


class AITrendPredictor:
    """
    AI-powered trend predictor using LLM analysis of technical indicators.

    This predictor constructs structured prompts from technical indicator data
    and uses an LLM to generate trend predictions with confidence scores and
    reasoning. Predictions are cached to minimize API costs and latency.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        cache_ttl_seconds: int = 600,  # 10 minutes default
        min_confidence_threshold: float = 0.3,
        temperature: float = 0.3,  # Lower temperature for more consistent predictions
        max_tokens: int = 500,
    ):
        """
        Initialize AI Trend Predictor.

        Args:
            llm_manager: LLM Manager instance for generating predictions
            cache_ttl_seconds: Cache time-to-live in seconds (default 10 minutes)
            min_confidence_threshold: Minimum confidence for valid predictions (0.0-1.0)
            temperature: LLM temperature for prediction generation (lower = more deterministic)
            max_tokens: Maximum tokens for LLM response
        """
        self.llm_manager = llm_manager
        self.cache_ttl_seconds = cache_ttl_seconds
        self.min_confidence_threshold = min_confidence_threshold
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Prediction cache: {cache_key: (timestamp, prediction)}
        self._prediction_cache: dict[str, tuple[int, TrendPrediction]] = {}

        logger.info(
            "ai_trend_predictor_initialized",
            cache_ttl_seconds=cache_ttl_seconds,
            min_confidence_threshold=min_confidence_threshold,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def predict_trend(
        self,
        indicators: IndicatorValues,
        symbol: str,
        timeframe: Timeframe,
    ) -> TrendPrediction:
        """
        Generate AI-powered trend prediction for a single timeframe.

        This method constructs a prompt from technical indicator data and uses
        the LLM Manager to generate a structured prediction with direction,
        confidence, and reasoning. Results are cached to reduce API calls.

        Args:
            indicators: Technical indicator values for the timeframe
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Timeframe for the prediction (e.g., Timeframe.ONE_HOUR)

        Returns:
            TrendPrediction with direction, confidence, and reasoning

        Raises:
            ValueError: If LLM returns invalid JSON or missing required fields
            Exception: If LLM Manager fails to generate prediction
        """
        # Check cache first
        cache_key = self._generate_cache_key(indicators, symbol, timeframe)
        cached_prediction = self._get_cached_prediction(cache_key)
        if cached_prediction:
            logger.info(
                "cache_hit",
                symbol=symbol,
                timeframe=timeframe.value,
                direction=cached_prediction.direction.value,
                confidence=cached_prediction.confidence,
            )
            return cached_prediction

        # Build prompt from indicators
        prompt = self._build_prompt(indicators, symbol, timeframe)
        system_prompt = self._build_system_prompt()

        logger.info(
            "requesting_trend_prediction",
            symbol=symbol,
            timeframe=timeframe.value,
            price=indicators.price,
        )

        # Generate prediction using LLM Manager
        try:
            llm_response = await self.llm_manager.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                validate_json=True,
                json_schema=self._get_json_schema(),
                check_hallucination=True,
            )

            # Parse structured response
            prediction = self._parse_llm_response(llm_response, indicators, symbol, timeframe)

            # Validate confidence threshold
            if prediction.confidence < self.min_confidence_threshold:
                logger.warning(
                    "prediction_below_confidence_threshold",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    confidence=prediction.confidence,
                    threshold=self.min_confidence_threshold,
                )

            # Cache the prediction
            self._cache_prediction(cache_key, prediction)

            logger.info(
                "trend_prediction_generated",
                symbol=symbol,
                timeframe=timeframe.value,
                direction=prediction.direction.value,
                confidence=prediction.confidence,
                cost=float(llm_response.cost),
                tokens=llm_response.tokens_used,
                latency_ms=llm_response.latency_ms,
                cached=llm_response.cached,
            )

            return prediction

        except Exception as e:
            logger.error(
                "trend_prediction_failed",
                symbol=symbol,
                timeframe=timeframe.value,
                error=str(e),
            )
            raise

    def _build_system_prompt(self) -> str:
        """
        Build system prompt that defines the AI's role and output format.

        Returns:
            System prompt string
        """
        return """You are an expert cryptocurrency trading analyst specializing in technical analysis.
Your task is to analyze technical indicators and predict the short-term price trend.

You must respond with ONLY a valid JSON object following this exact structure:
{
  "direction": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "reasoning": "Clear explanation of your prediction based on the indicators"
}

Guidelines:
- Analyze all provided indicators (RSI, MACD, Bollinger Bands, Moving Averages, ATR)
- Consider indicator interactions and confirmations
- Assign confidence based on signal strength and agreement
- Provide specific reasoning referencing indicator values
- Be conservative with confidence scores (0.7+ only for very strong signals)
- Return neutral with low confidence if signals are mixed or unclear
- Do NOT include markdown formatting, just raw JSON"""

    def _build_prompt(self, indicators: IndicatorValues, symbol: str, timeframe: Timeframe) -> str:
        """
        Build user prompt with technical indicator summary.

        Args:
            indicators: Technical indicator values
            symbol: Trading pair symbol
            timeframe: Timeframe for analysis

        Returns:
            Formatted prompt string
        """
        # Format indicator summary
        indicator_summary = f"""Analyze the following technical indicators for {symbol} on {timeframe.value} timeframe:

CURRENT PRICE: ${indicators.price:.2f}

RSI ANALYSIS:
- RSI Value: {indicators.rsi:.2f if indicators.rsi is not None else 'N/A'}
- Status: {'OVERSOLD' if indicators.rsi_oversold else 'OVERBOUGHT' if indicators.rsi_overbought else 'NEUTRAL'}
- Interpretation: {'Potential buy signal' if indicators.rsi_oversold else 'Potential sell signal' if indicators.rsi_overbought else 'No extreme reading'}

MACD ANALYSIS:
- MACD Line: {indicators.macd:.4f if indicators.macd is not None else 'N/A'}
- Signal Line: {indicators.macd_signal:.4f if indicators.macd_signal is not None else 'N/A'}
- Histogram: {indicators.macd_histogram:.4f if indicators.macd_histogram is not None else 'N/A'}
- Crossover: {'BULLISH CROSSOVER' if indicators.macd_bullish_crossover else 'BEARISH CROSSOVER' if indicators.macd_bearish_crossover else 'No crossover'}

BOLLINGER BANDS:
- Upper Band: ${indicators.bb_upper:.2f if indicators.bb_upper is not None else 'N/A'}
- Middle Band: ${indicators.bb_middle:.2f if indicators.bb_middle is not None else 'N/A'}
- Lower Band: ${indicators.bb_lower:.2f if indicators.bb_lower is not None else 'N/A'}
- Band Position: {f'{indicators.bb_percent:.2%}' if indicators.bb_percent is not None else 'N/A'} (0% = lower band, 100% = upper band)
- Bandwidth: {f'{indicators.bb_bandwidth:.4f}' if indicators.bb_bandwidth is not None else 'N/A'}
- Interpretation: {'Price near upper band - potential overbought' if indicators.bb_percent and indicators.bb_percent > 0.8 else 'Price near lower band - potential oversold' if indicators.bb_percent and indicators.bb_percent < 0.2 else 'Price within bands'}

MOVING AVERAGES:
- MA 20: ${indicators.ma_20:.2f if indicators.ma_20 is not None else 'N/A'}
- MA 50: ${indicators.ma_50:.2f if indicators.ma_50 is not None else 'N/A'}
- MA 200: ${indicators.ma_200:.2f if indicators.ma_200 is not None else 'N/A'}
- Trend: {indicators.ma_trend.value.upper()}
- Price vs MA20: {'Above' if indicators.ma_20 and indicators.price > indicators.ma_20 else 'Below' if indicators.ma_20 else 'N/A'}

VOLATILITY (ATR):
- ATR: {indicators.atr:.2f if indicators.atr is not None else 'N/A'}
- ATR Percentage: {f'{indicators.atr_percent:.2f}%' if indicators.atr_percent is not None else 'N/A'}
- Volatility: {'HIGH' if indicators.atr_percent and indicators.atr_percent > 3.0 else 'MODERATE' if indicators.atr_percent and indicators.atr_percent > 1.5 else 'LOW' if indicators.atr_percent else 'N/A'}

Based on these indicators, predict the trend direction (bullish/bearish/neutral) with a confidence score."""

        return indicator_summary

    def _get_json_schema(self) -> dict[str, Any]:
        """
        Get JSON schema for response validation.

        Returns:
            JSON schema dict
        """
        return {
            "type": "object",
            "required": ["direction", "confidence", "reasoning"],
            "properties": {
                "direction": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "reasoning": {"type": "string", "minLength": 10},
            },
        }

    def _parse_llm_response(
        self,
        llm_response: LLMResponse,
        indicators: IndicatorValues,
        symbol: str,
        timeframe: Timeframe,
    ) -> TrendPrediction:
        """
        Parse LLM response into TrendPrediction object.

        Args:
            llm_response: Raw LLM response
            indicators: Indicator values used for prediction
            symbol: Trading pair symbol
            timeframe: Timeframe for prediction

        Returns:
            TrendPrediction object

        Raises:
            ValueError: If response is invalid JSON or missing required fields
        """
        try:
            # Parse JSON response
            response_data = json.loads(llm_response.content)

            # Extract required fields
            direction_str = response_data.get("direction")
            confidence = response_data.get("confidence")
            reasoning = response_data.get("reasoning")

            # Validate required fields
            if not direction_str or confidence is None or not reasoning:
                raise ValueError(f"Missing required fields in response: {response_data}")

            # Validate direction
            try:
                direction = PredictionDirection(direction_str.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid direction value: {direction_str}. "
                    f"Must be one of: bullish, bearish, neutral"
                )

            # Validate confidence type and range
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"Confidence must be a number, got {type(confidence)}")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

            # Build indicator summary for metadata
            indicators_used = {
                "rsi": indicators.rsi,
                "rsi_signal": (
                    "oversold"
                    if indicators.rsi_oversold
                    else "overbought" if indicators.rsi_overbought else "neutral"
                ),
                "macd_histogram": indicators.macd_histogram,
                "macd_crossover": (
                    "bullish"
                    if indicators.macd_bullish_crossover
                    else "bearish" if indicators.macd_bearish_crossover else "none"
                ),
                "bb_position": indicators.bb_percent,
                "ma_trend": indicators.ma_trend.value,
                "atr_percent": indicators.atr_percent,
            }

            prediction = TrendPrediction(
                direction=direction,
                confidence=float(confidence),
                reasoning=str(reasoning),
                timeframe=timeframe,
                symbol=symbol,
                timestamp=int(datetime.now().timestamp() * 1000),
                indicators_used=indicators_used,
                llm_response=llm_response,
                cached=False,
            )

            return prediction

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}. Content: {llm_response.content}")
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")

    def _generate_cache_key(
        self, indicators: IndicatorValues, symbol: str, timeframe: Timeframe
    ) -> str:
        """
        Generate cache key from indicator values and parameters.

        The cache key is a hash of the indicator values to detect when
        indicators have changed significantly enough to warrant a new prediction.

        Args:
            indicators: Technical indicator values
            symbol: Trading pair symbol
            timeframe: Timeframe

        Returns:
            SHA256 hash as cache key
        """
        # Include key indicator values in cache key
        key_data = (
            f"{symbol}|{timeframe.value}|"
            f"{indicators.price:.2f}|"
            f"{indicators.rsi:.2f if indicators.rsi else 'NA'}|"
            f"{indicators.macd_histogram:.4f if indicators.macd_histogram else 'NA'}|"
            f"{indicators.bb_percent:.3f if indicators.bb_percent else 'NA'}|"
            f"{indicators.ma_trend.value}|"
            f"{indicators.macd_bullish_crossover}|"
            f"{indicators.macd_bearish_crossover}"
        )
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _get_cached_prediction(self, cache_key: str) -> TrendPrediction | None:
        """
        Retrieve cached prediction if available and not expired.

        Args:
            cache_key: Cache key to look up

        Returns:
            Cached TrendPrediction or None if not found/expired
        """
        if cache_key not in self._prediction_cache:
            return None

        cached_timestamp, cached_prediction = self._prediction_cache[cache_key]

        # Check if expired
        current_timestamp = int(datetime.now().timestamp())
        if (current_timestamp * 1000 - cached_timestamp) > (self.cache_ttl_seconds * 1000):
            del self._prediction_cache[cache_key]
            return None

        # Mark as cached and return
        cached_prediction.cached = True
        return cached_prediction

    def _cache_prediction(self, cache_key: str, prediction: TrendPrediction) -> None:
        """
        Cache prediction for future use.

        Args:
            cache_key: Cache key
            prediction: Prediction to cache
        """
        timestamp = int(datetime.now().timestamp() * 1000)
        self._prediction_cache[cache_key] = (timestamp, prediction)

    def clear_cache(self) -> None:
        """Clear all cached predictions."""
        cache_size = len(self._prediction_cache)
        self._prediction_cache.clear()
        logger.info("prediction_cache_cleared", items_removed=cache_size)

    def get_cache_stats(self) -> dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size
        """
        return {
            "cache_size": len(self._prediction_cache),
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }
