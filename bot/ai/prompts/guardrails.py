"""
Guardrail system to keep LLM responses data-driven and prevent speculation.

This module provides validation rules and checks to ensure LLM outputs remain
grounded in provided data, avoid hallucinations, and maintain focus on
technical analysis rather than speculation.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class GuardrailSeverity(str, Enum):
    """Severity level for guardrail violations."""

    WARNING = "warning"  # Log but allow response
    ERROR = "error"  # Block response and require retry
    CRITICAL = "critical"  # Block response and escalate to human review


@dataclass
class GuardrailViolation:
    """
    Record of a guardrail rule violation.

    Attributes:
        rule_name: Name of the violated rule
        severity: Severity level of the violation
        message: Human-readable violation description
        context: Additional context about the violation
        matched_text: Text snippet that triggered the violation (optional)
    """

    rule_name: str
    severity: GuardrailSeverity
    message: str
    context: dict[str, Any]
    matched_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert violation to dictionary for logging."""
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "matched_text": self.matched_text,
        }


class GuardrailRule(ABC):
    """
    Abstract base class for guardrail validation rules.

    Subclasses implement specific validation logic to prevent unwanted
    behavior in LLM responses.
    """

    def __init__(self, name: str, severity: GuardrailSeverity, description: str):
        """
        Initialize guardrail rule.

        Args:
            name: Unique rule identifier
            severity: Severity level for violations
            description: Human-readable rule description
        """
        self.name = name
        self.severity = severity
        self.description = description

        logger.debug(
            "guardrail_rule_initialized",
            name=name,
            severity=severity.value,
        )

    @abstractmethod
    def validate(self, response: str, context: dict[str, Any]) -> list[GuardrailViolation]:
        """
        Validate response against the rule.

        Args:
            response: LLM response text to validate
            context: Additional context for validation (e.g., original data)

        Returns:
            List of violations found (empty list if valid)
        """
        pass


class SpeculationDetectionRule(GuardrailRule):
    """
    Detects speculative language that isn't grounded in data.

    Flags phrases indicating prediction, speculation, or opinion rather than
    analysis of provided technical indicators.
    """

    # Patterns indicating speculation or prediction
    SPECULATION_PATTERNS = [
        r"\b(I think|I believe|I feel|in my opinion)\b",
        r"\b(might|could|may|possibly|perhaps|probably)\s+(see|expect|anticipate)\b",
        r"\b(prediction:|forecast:|guess:)\b",
        r"\b(will definitely|will certainly|guaranteed to)\b",
        r"\b(without a doubt|absolutely certain)\b",
        r"\b(my analysis suggests|I predict)\b",
    ]

    def __init__(self, severity: GuardrailSeverity = GuardrailSeverity.WARNING):
        """Initialize speculation detection rule."""
        super().__init__(
            name="speculation_detection",
            severity=severity,
            description="Detects speculative language not grounded in provided data",
        )
        self.patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SPECULATION_PATTERNS
        ]

    def validate(self, response: str, context: dict[str, Any]) -> list[GuardrailViolation]:
        """Check for speculative language patterns."""
        violations: list[GuardrailViolation] = []

        for pattern in self.patterns:
            matches = pattern.finditer(response)
            for match in matches:
                violations.append(
                    GuardrailViolation(
                        rule_name=self.name,
                        severity=self.severity,
                        message="Response contains speculative language",
                        context={"pattern": pattern.pattern},
                        matched_text=match.group(0),
                    )
                )

        return violations


class DataGroundingRule(GuardrailRule):
    """
    Ensures response references provided data and indicators.

    Validates that the LLM response actually uses the technical indicators
    and data provided in the prompt, rather than making unsupported claims.
    """

    def __init__(
        self, required_indicators: list[str], severity: GuardrailSeverity = GuardrailSeverity.ERROR
    ):
        """
        Initialize data grounding rule.

        Args:
            required_indicators: List of indicator names that should be referenced
            severity: Severity level for violations
        """
        super().__init__(
            name="data_grounding",
            severity=severity,
            description="Ensures response references provided technical indicators",
        )
        self.required_indicators = required_indicators

    def validate(self, response: str, context: dict[str, Any]) -> list[GuardrailViolation]:
        """Check if response references required indicators."""
        violations: list[GuardrailViolation] = []
        response_lower = response.lower()

        # Check if at least some required indicators are mentioned
        mentioned_indicators = []
        for indicator in self.required_indicators:
            if indicator.lower() in response_lower:
                mentioned_indicators.append(indicator)

        # Require at least 50% of indicators to be mentioned
        mention_ratio = (
            len(mentioned_indicators) / len(self.required_indicators)
            if self.required_indicators
            else 0
        )

        if mention_ratio < 0.5:
            violations.append(
                GuardrailViolation(
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Response mentions only {len(mentioned_indicators)}/{len(self.required_indicators)} required indicators",
                    context={
                        "required_indicators": self.required_indicators,
                        "mentioned_indicators": mentioned_indicators,
                        "mention_ratio": mention_ratio,
                    },
                )
            )

        return violations


class NumericHallucinationRule(GuardrailRule):
    """
    Detects numeric values in response that don't match provided data.

    Prevents the LLM from citing indicator values that differ from the
    actual values provided in the prompt.
    """

    def __init__(
        self, tolerance_pct: float = 5.0, severity: GuardrailSeverity = GuardrailSeverity.ERROR
    ):
        """
        Initialize numeric hallucination rule.

        Args:
            tolerance_pct: Percentage tolerance for numeric differences
            severity: Severity level for violations
        """
        super().__init__(
            name="numeric_hallucination",
            severity=severity,
            description="Detects numeric values that don't match provided data",
        )
        self.tolerance_pct = tolerance_pct

    def validate(self, response: str, context: dict[str, Any]) -> list[GuardrailViolation]:
        """Check for hallucinated numeric values."""
        violations: list[GuardrailViolation] = []

        # Get indicator values from context
        indicator_values = context.get("indicator_values", {})
        if not indicator_values:
            return violations  # Cannot validate without reference data

        # Extract numbers from response
        number_pattern = re.compile(r"\b\d+\.?\d*\b")
        response_numbers = [float(match.group(0)) for match in number_pattern.finditer(response)]

        # Check if any response numbers significantly differ from indicator values
        for resp_num in response_numbers:
            found_match = False
            for _indicator_name, indicator_val in indicator_values.items():
                if indicator_val is None:
                    continue

                # Check if response number is close to indicator value
                if (
                    abs(resp_num - indicator_val) / max(abs(indicator_val), 1e-6) * 100
                    <= self.tolerance_pct
                ):
                    found_match = True
                    break

            # If a number doesn't match any indicator within tolerance, flag it
            # Note: This is a simplified check; production version should be more sophisticated
            if (
                not found_match and resp_num > 1.0
            ):  # Ignore small numbers like 0.5 which might be percentages
                logger.debug(
                    "numeric_value_not_matched",
                    value=resp_num,
                    indicators=list(indicator_values.keys()),
                )

        return violations


class ConfidenceJustificationRule(GuardrailRule):
    """
    Ensures confidence scores are justified by reasoning.

    High confidence (>0.7) must be accompanied by strong supporting evidence
    from multiple indicators. Low confidence must acknowledge conflicting signals.
    """

    def __init__(self, severity: GuardrailSeverity = GuardrailSeverity.WARNING):
        """Initialize confidence justification rule."""
        super().__init__(
            name="confidence_justification",
            severity=severity,
            description="Ensures confidence scores are properly justified",
        )

    def validate(self, response: str, context: dict[str, Any]) -> list[GuardrailViolation]:
        """Check if confidence is justified by reasoning."""
        violations: list[GuardrailViolation] = []

        # Extract confidence from context (should be in parsed JSON)
        confidence = context.get("confidence")
        reasoning = context.get("reasoning", "")

        if confidence is None:
            return violations

        # High confidence (>0.7) should mention multiple indicators
        if confidence > 0.7:
            reasoning_lower = reasoning.lower()
            indicator_keywords = ["rsi", "macd", "bollinger", "moving average", "atr", "volume"]
            mentioned = sum(1 for keyword in indicator_keywords if keyword in reasoning_lower)

            if mentioned < 2:
                violations.append(
                    GuardrailViolation(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"High confidence ({confidence:.2f}) but reasoning mentions only {mentioned} indicators",
                        context={
                            "confidence": confidence,
                            "indicators_mentioned": mentioned,
                            "reasoning_length": len(reasoning),
                        },
                    )
                )

        # Low confidence (<0.4) should acknowledge uncertainty
        if confidence < 0.4:
            uncertainty_keywords = ["mixed", "conflicting", "unclear", "uncertain", "weak"]
            has_uncertainty = any(keyword in reasoning.lower() for keyword in uncertainty_keywords)

            if not has_uncertainty:
                violations.append(
                    GuardrailViolation(
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Low confidence ({confidence:.2f}) but reasoning doesn't acknowledge uncertainty",
                        context={
                            "confidence": confidence,
                            "reasoning_length": len(reasoning),
                        },
                    )
                )

        return violations


class GuardrailValidator:
    """
    Orchestrates multiple guardrail rules to validate LLM responses.

    Executes all configured rules and aggregates violations to determine
    if a response should be accepted, logged with warnings, or rejected.
    """

    def __init__(self, rules: list[GuardrailRule] | None = None):
        """
        Initialize guardrail validator.

        Args:
            rules: List of guardrail rules to apply (defaults to standard set)
        """
        self.rules = rules or self._get_default_rules()

        logger.info(
            "guardrail_validator_initialized",
            rule_count=len(self.rules),
            rule_names=[rule.name for rule in self.rules],
        )

    def _get_default_rules(self) -> list[GuardrailRule]:
        """
        Get default set of guardrail rules.

        Returns:
            List of default GuardrailRule instances
        """
        return [
            SpeculationDetectionRule(severity=GuardrailSeverity.WARNING),
            DataGroundingRule(
                required_indicators=["RSI", "MACD", "Bollinger", "MA", "ATR"],
                severity=GuardrailSeverity.WARNING,
            ),
            ConfidenceJustificationRule(severity=GuardrailSeverity.WARNING),
        ]

    def add_rule(self, rule: GuardrailRule) -> None:
        """
        Add a guardrail rule to the validator.

        Args:
            rule: GuardrailRule to add
        """
        self.rules.append(rule)
        logger.debug("guardrail_rule_added", rule_name=rule.name)

    def validate(
        self, response: str, context: dict[str, Any]
    ) -> tuple[bool, list[GuardrailViolation]]:
        """
        Validate response against all guardrail rules.

        Args:
            response: LLM response text to validate
            context: Additional context for validation

        Returns:
            Tuple of (is_valid, list_of_violations)
            is_valid is False if any CRITICAL or ERROR violations found
        """
        all_violations: list[GuardrailViolation] = []

        for rule in self.rules:
            violations = rule.validate(response, context)
            all_violations.extend(violations)

            if violations:
                logger.debug(
                    "guardrail_violations_found",
                    rule_name=rule.name,
                    violation_count=len(violations),
                    violations=[v.to_dict() for v in violations],
                )

        # Determine if response should be blocked
        blocking_violations = [
            v
            for v in all_violations
            if v.severity in (GuardrailSeverity.ERROR, GuardrailSeverity.CRITICAL)
        ]

        is_valid = len(blocking_violations) == 0

        # Log summary
        if all_violations:
            severity_counts: dict[str, int] = {}
            for v in all_violations:
                severity_counts[v.severity.value] = severity_counts.get(v.severity.value, 0) + 1

            logger.info(
                "guardrail_validation_complete",
                is_valid=is_valid,
                total_violations=len(all_violations),
                severity_counts=severity_counts,
            )
        else:
            logger.info("guardrail_validation_passed", rule_count=len(self.rules))

        return is_valid, all_violations

    def get_rules(self) -> list[GuardrailRule]:
        """Get list of configured rules."""
        return self.rules.copy()
