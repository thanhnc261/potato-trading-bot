"""
Trading-specific prompt templates for various analysis workflows.

This module provides ready-to-use prompt templates for common trading analysis
tasks including trend prediction, risk assessment, and market sentiment analysis.
Each template includes chain-of-thought reasoning guidance and output schemas.
"""

from typing import Any

from bot.ai.prompts.base import PromptComponent, PromptTemplate, PromptType, PromptVersion
from bot.ai.prompts.schemas import FieldDefinition, OutputSchema, SchemaType

# Version 1.0.0: Initial implementation
VERSION_1_0_0 = PromptVersion(
    version="1.0.0",
    created_at=1704067200000,  # 2024-01-01
    author="Trading Bot Team",
    changelog="Initial prompt engineering system implementation",
    performance_notes="",
    is_active=True,
)


# Reusable prompt components for composition
ROLE_COMPONENT = PromptComponent(
    name="expert_role",
    content="You are an expert cryptocurrency trading analyst with deep expertise in technical analysis and quantitative trading strategies.",
    required_variables=set(),
)

CHAIN_OF_THOUGHT_COMPONENT = PromptComponent(
    name="chain_of_thought",
    content="""Follow this analysis framework:
1. DATA REVIEW: Examine all provided technical indicators systematically
2. SIGNAL IDENTIFICATION: Identify bullish, bearish, and neutral signals from each indicator
3. CONFLUENCE ANALYSIS: Look for agreement or divergence between indicators
4. WEIGHT ASSESSMENT: Assign importance based on signal strength and reliability
5. SYNTHESIS: Combine weighted signals into overall trend assessment
6. CONFIDENCE CALIBRATION: Set confidence based on signal agreement and clarity
7. REASONING: Document your analysis path referencing specific indicator values""",
    required_variables=set(),
)

NO_SPECULATION_COMPONENT = PromptComponent(
    name="no_speculation",
    content="""Critical Constraints:
- Base analysis ONLY on provided technical indicator data
- Do NOT speculate about news, events, or external factors
- Do NOT make price predictions beyond trend direction
- Do NOT reference information not present in the provided data
- Acknowledge when signals are mixed or unclear
- Be conservative with confidence scores""",
    required_variables=set(),
)

JSON_FORMAT_COMPONENT = PromptComponent(
    name="json_format",
    content="""Output Format:
Respond with ONLY a valid JSON object. No markdown formatting, no explanations outside JSON.
Follow this exact structure:
{json_structure}""",
    required_variables={"json_structure"},
)


class TrendAnalysisPrompt(PromptTemplate):
    """
    Prompt template for trend direction prediction with confidence scoring.

    Analyzes technical indicators to predict short-term trend direction
    (bullish/bearish/neutral) with confidence score and reasoning.
    """

    def __init__(self, version: PromptVersion = VERSION_1_0_0):
        """Initialize trend analysis prompt template."""
        super().__init__(
            name="trend_analysis",
            description="Predict trend direction from technical indicators",
            version=version,
            prompt_type=PromptType.SYSTEM,
        )

        # Define output schema
        self.output_schema = OutputSchema(
            name="trend_prediction",
            version="1.0.0",
            description="Trend direction prediction with confidence and reasoning",
            fields=[
                FieldDefinition(
                    name="direction",
                    type=SchemaType.STRING,
                    required=True,
                    description="Predicted trend direction",
                    enum_values=["bullish", "bearish", "neutral"],
                ),
                FieldDefinition(
                    name="confidence",
                    type=SchemaType.NUMBER,
                    required=True,
                    description="Confidence score from 0.0 to 1.0",
                    min_value=0.0,
                    max_value=1.0,
                ),
                FieldDefinition(
                    name="reasoning",
                    type=SchemaType.STRING,
                    required=True,
                    description="Detailed explanation of prediction",
                    min_length=50,
                    max_length=1000,
                ),
                FieldDefinition(
                    name="key_indicators",
                    type=SchemaType.ARRAY,
                    required=False,
                    description="List of most influential indicators",
                    items_schema={"type": "string"},
                ),
            ],
        )

    def render(self, **kwargs: Any) -> str:
        """
        Render trend analysis system prompt.

        Args:
            **kwargs: Not used, template is static

        Returns:
            Rendered system prompt
        """
        self.track_usage()

        json_structure = """{
  "direction": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "reasoning": "Detailed explanation referencing specific indicator values",
  "key_indicators": ["indicator1", "indicator2", ...]
}"""

        components = [
            ROLE_COMPONENT,
            CHAIN_OF_THOUGHT_COMPONENT,
            NO_SPECULATION_COMPONENT,
            PromptComponent(
                name="json_format_with_structure",
                content=JSON_FORMAT_COMPONENT.content.format(json_structure=json_structure),
                required_variables=set(),
            ),
        ]

        return "\n\n".join(comp.render() for comp in components)

    def get_output_schema(self) -> OutputSchema:
        """Get the output schema for this prompt."""
        return self.output_schema


class RiskAnalysisPrompt(PromptTemplate):
    """
    Prompt template for risk assessment of potential trades.

    Evaluates risk factors including volatility, market conditions, and
    technical setup quality to provide risk rating and recommendations.
    """

    def __init__(self, version: PromptVersion = VERSION_1_0_0):
        """Initialize risk analysis prompt template."""
        super().__init__(
            name="risk_analysis",
            description="Assess risk factors for potential trades",
            version=version,
            prompt_type=PromptType.SYSTEM,
        )

        # Define output schema
        self.output_schema = OutputSchema(
            name="risk_assessment",
            version="1.0.0",
            description="Risk assessment with rating and recommendations",
            fields=[
                FieldDefinition(
                    name="risk_level",
                    type=SchemaType.STRING,
                    required=True,
                    description="Overall risk level",
                    enum_values=["low", "medium", "high", "extreme"],
                ),
                FieldDefinition(
                    name="risk_score",
                    type=SchemaType.NUMBER,
                    required=True,
                    description="Numeric risk score from 0.0 (low) to 1.0 (extreme)",
                    min_value=0.0,
                    max_value=1.0,
                ),
                FieldDefinition(
                    name="risk_factors",
                    type=SchemaType.ARRAY,
                    required=True,
                    description="List of identified risk factors",
                    items_schema={"type": "string"},
                ),
                FieldDefinition(
                    name="volatility_assessment",
                    type=SchemaType.STRING,
                    required=True,
                    description="Volatility level analysis",
                    min_length=20,
                ),
                FieldDefinition(
                    name="recommendations",
                    type=SchemaType.ARRAY,
                    required=True,
                    description="Risk mitigation recommendations",
                    items_schema={"type": "string"},
                ),
            ],
        )

    def render(self, **kwargs: Any) -> str:
        """
        Render risk analysis system prompt.

        Args:
            **kwargs: Not used, template is static

        Returns:
            Rendered system prompt
        """
        self.track_usage()

        json_structure = """{
  "risk_level": "low" | "medium" | "high" | "extreme",
  "risk_score": 0.0 to 1.0,
  "risk_factors": ["factor1", "factor2", ...],
  "volatility_assessment": "Analysis of current volatility levels",
  "recommendations": ["recommendation1", "recommendation2", ...]
}"""

        risk_analysis_component = PromptComponent(
            name="risk_analysis_framework",
            content="""Risk Analysis Framework:
1. VOLATILITY ANALYSIS: Evaluate ATR and Bollinger Band width for volatility level
2. TREND STRENGTH: Assess trend clarity and momentum indicators
3. SUPPORT/RESISTANCE: Identify proximity to key levels
4. INDICATOR DIVERGENCE: Flag conflicting signals as risk factors
5. MARKET STRUCTURE: Evaluate overall market condition quality
6. RISK QUANTIFICATION: Assign risk score based on identified factors
7. MITIGATION: Provide specific recommendations to manage identified risks""",
            required_variables=set(),
        )

        components = [
            ROLE_COMPONENT,
            risk_analysis_component,
            NO_SPECULATION_COMPONENT,
            PromptComponent(
                name="json_format_with_structure",
                content=JSON_FORMAT_COMPONENT.content.format(json_structure=json_structure),
                required_variables=set(),
            ),
        ]

        return "\n\n".join(comp.render() for comp in components)

    def get_output_schema(self) -> OutputSchema:
        """Get the output schema for this prompt."""
        return self.output_schema


class MarketSentimentPrompt(PromptTemplate):
    """
    Prompt template for market sentiment analysis.

    Analyzes technical indicators to gauge overall market sentiment and
    momentum, providing sentiment classification and strength assessment.
    """

    def __init__(self, version: PromptVersion = VERSION_1_0_0):
        """Initialize market sentiment prompt template."""
        super().__init__(
            name="market_sentiment",
            description="Analyze overall market sentiment from technical data",
            version=version,
            prompt_type=PromptType.SYSTEM,
        )

        # Define output schema
        self.output_schema = OutputSchema(
            name="sentiment_analysis",
            version="1.0.0",
            description="Market sentiment classification and strength",
            fields=[
                FieldDefinition(
                    name="sentiment",
                    type=SchemaType.STRING,
                    required=True,
                    description="Overall market sentiment",
                    enum_values=[
                        "extremely_bearish",
                        "bearish",
                        "neutral",
                        "bullish",
                        "extremely_bullish",
                    ],
                ),
                FieldDefinition(
                    name="strength",
                    type=SchemaType.NUMBER,
                    required=True,
                    description="Sentiment strength from 0.0 (weak) to 1.0 (strong)",
                    min_value=0.0,
                    max_value=1.0,
                ),
                FieldDefinition(
                    name="momentum",
                    type=SchemaType.STRING,
                    required=True,
                    description="Momentum assessment",
                    enum_values=["accelerating", "steady", "weakening", "reversing"],
                ),
                FieldDefinition(
                    name="analysis",
                    type=SchemaType.STRING,
                    required=True,
                    description="Detailed sentiment analysis",
                    min_length=50,
                ),
            ],
        )

    def render(self, **kwargs: Any) -> str:
        """
        Render market sentiment system prompt.

        Args:
            **kwargs: Not used, template is static

        Returns:
            Rendered system prompt
        """
        self.track_usage()

        json_structure = """{
  "sentiment": "extremely_bearish" | "bearish" | "neutral" | "bullish" | "extremely_bullish",
  "strength": 0.0 to 1.0,
  "momentum": "accelerating" | "steady" | "weakening" | "reversing",
  "analysis": "Detailed sentiment analysis with indicator references"
}"""

        sentiment_analysis_component = PromptComponent(
            name="sentiment_analysis_framework",
            content="""Sentiment Analysis Framework:
1. MOMENTUM INDICATORS: Analyze MACD histogram and crossovers for momentum direction
2. OVERBOUGHT/OVERSOLD: Evaluate RSI levels for extreme sentiment
3. TREND ALIGNMENT: Check moving average alignment and price position
4. VOLATILITY: Consider ATR and Bollinger Band position for sentiment strength
5. SENTIMENT CLASSIFICATION: Categorize overall sentiment on 5-point scale
6. MOMENTUM STATE: Determine if sentiment is accelerating, steady, weakening, or reversing
7. STRENGTH ASSESSMENT: Quantify sentiment strength based on indicator agreement""",
            required_variables=set(),
        )

        components = [
            ROLE_COMPONENT,
            sentiment_analysis_component,
            NO_SPECULATION_COMPONENT,
            PromptComponent(
                name="json_format_with_structure",
                content=JSON_FORMAT_COMPONENT.content.format(json_structure=json_structure),
                required_variables=set(),
            ),
        ]

        return "\n\n".join(comp.render() for comp in components)

    def get_output_schema(self) -> OutputSchema:
        """Get the output schema for this prompt."""
        return self.output_schema


def create_user_prompt_for_indicators(
    symbol: str,
    timeframe: str,
    indicators: dict[str, Any],
) -> str:
    """
    Create user prompt with technical indicator data.

    Args:
        symbol: Trading pair symbol (e.g., "BTC/USDT")
        timeframe: Analysis timeframe (e.g., "1h", "4h", "1d")
        indicators: Dictionary of indicator values

    Returns:
        Formatted user prompt with indicator data
    """
    prompt = f"""Analyze technical indicators for {symbol} on {timeframe} timeframe:

CURRENT PRICE: ${indicators.get('price', 'N/A')}

RSI ANALYSIS:
- RSI Value: {indicators.get('rsi', 'N/A')}
- Status: {indicators.get('rsi_signal', 'neutral').upper()}

MACD ANALYSIS:
- MACD Line: {indicators.get('macd', 'N/A')}
- Signal Line: {indicators.get('macd_signal', 'N/A')}
- Histogram: {indicators.get('macd_histogram', 'N/A')}
- Crossover: {indicators.get('macd_crossover', 'none').upper()}

BOLLINGER BANDS:
- Upper Band: ${indicators.get('bb_upper', 'N/A')}
- Middle Band: ${indicators.get('bb_middle', 'N/A')}
- Lower Band: ${indicators.get('bb_lower', 'N/A')}
- Band Position: {indicators.get('bb_position', 'N/A')}
- Bandwidth: {indicators.get('bb_bandwidth', 'N/A')}

MOVING AVERAGES:
- MA 20: ${indicators.get('ma_20', 'N/A')}
- MA 50: ${indicators.get('ma_50', 'N/A')}
- MA 200: ${indicators.get('ma_200', 'N/A')}
- Trend: {indicators.get('ma_trend', 'neutral').upper()}

VOLATILITY (ATR):
- ATR: {indicators.get('atr', 'N/A')}
- ATR Percentage: {indicators.get('atr_percent', 'N/A')}%
- Volatility Level: {indicators.get('volatility', 'N/A').upper()}

Provide your analysis following the specified framework."""

    return prompt
