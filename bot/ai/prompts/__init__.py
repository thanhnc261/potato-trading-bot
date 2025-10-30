"""
Prompt engineering system for trading analysis workflows.

This module provides a comprehensive prompt management system with:
- Reusable system prompt templates
- Versioning and A/B testing support
- JSON output schema validation
- Guardrails to prevent speculation
- Chain-of-thought reasoning guidance
- Structured prompt update workflow
"""

from bot.ai.prompts.base import PromptTemplate, PromptVersion, SystemPromptBuilder
from bot.ai.prompts.guardrails import GuardrailRule, GuardrailValidator
from bot.ai.prompts.schemas import OutputSchema, SchemaValidator
from bot.ai.prompts.trading import MarketSentimentPrompt, RiskAnalysisPrompt, TrendAnalysisPrompt

__all__ = [
    "PromptTemplate",
    "PromptVersion",
    "SystemPromptBuilder",
    "OutputSchema",
    "SchemaValidator",
    "TrendAnalysisPrompt",
    "RiskAnalysisPrompt",
    "MarketSentimentPrompt",
    "GuardrailValidator",
    "GuardrailRule",
]
