# Prompt Engineering System

Comprehensive prompt management system for trading analysis workflows with versioning, validation, and guardrails.

## Overview

This module provides a robust framework for creating, managing, and evolving LLM prompts for cryptocurrency trading analysis. It includes:

- **Reusable Templates**: Modular prompt components that can be composed
- **Versioning**: Semantic versioning for A/B testing and evolution tracking
- **Schema Validation**: Strict JSON output validation with type checking
- **Guardrails**: Automated checks to prevent speculation and hallucination
- **Chain-of-Thought**: Built-in reasoning frameworks for consistent analysis

## Architecture

```
bot/ai/prompts/
├── __init__.py          # Public API exports
├── base.py              # Base template classes and versioning
├── schemas.py           # JSON schema definitions and validators
├── guardrails.py        # Validation rules and guardrails
├── trading.py           # Trading-specific prompt templates
└── README.md            # This file
```

## Core Components

### 1. Prompt Templates (`base.py`)

#### PromptTemplate (Abstract Base Class)

Base class for all prompt templates with versioning and usage tracking.

```python
from bot.ai.prompts.base import PromptTemplate, PromptVersion, PromptType

class MyPrompt(PromptTemplate):
    def __init__(self):
        version = PromptVersion(
            version="1.0.0",
            created_at=1704067200000,
            author="Your Name",
            changelog="Initial implementation",
        )
        super().__init__(
            name="my_prompt",
            description="Description of what this prompt does",
            version=version,
            prompt_type=PromptType.SYSTEM,
        )

    def render(self, **kwargs):
        self.track_usage()
        return "Your prompt text here"
```

#### PromptComponent

Reusable building blocks for composing complex prompts.

```python
from bot.ai.prompts.base import PromptComponent

component = PromptComponent(
    name="expert_role",
    content="You are an expert {domain} analyst.",
    required_variables={"domain"},
)

rendered = component.render(domain="cryptocurrency trading")
```

#### SystemPromptBuilder

Compose system prompts from multiple components.

```python
from bot.ai.prompts.base import SystemPromptBuilder, PromptVersion

builder = SystemPromptBuilder(
    name="trading_system_prompt",
    version=PromptVersion(version="1.0.0", ...)
)

builder.add_component(role_component)
builder.add_component(cot_component)
builder.add_component(format_component)

system_prompt = builder.build(domain="crypto", format="JSON")
```

### 2. Output Schemas (`schemas.py`)

#### OutputSchema

Define strict JSON schemas for LLM responses.

```python
from bot.ai.prompts.schemas import OutputSchema, FieldDefinition, SchemaType

schema = OutputSchema(
    name="prediction",
    version="1.0.0",
    description="Trend prediction output",
    fields=[
        FieldDefinition(
            name="direction",
            type=SchemaType.STRING,
            required=True,
            enum_values=["bullish", "bearish", "neutral"],
        ),
        FieldDefinition(
            name="confidence",
            type=SchemaType.NUMBER,
            required=True,
            min_value=0.0,
            max_value=1.0,
        ),
    ],
)

# Convert to JSON schema for validation
json_schema = schema.to_json_schema()
```

#### SchemaValidator

Validate LLM responses against schemas.

```python
from bot.ai.prompts.schemas import SchemaValidator

validator = SchemaValidator(strict_mode=True)
is_valid = validator.validate(response_data, schema)

if not is_valid:
    errors = validator.get_errors()
    print(f"Validation failed: {errors}")
```

### 3. Guardrails (`guardrails.py`)

#### GuardrailValidator

Enforce data-driven analysis and prevent speculation.

```python
from bot.ai.prompts.guardrails import GuardrailValidator

validator = GuardrailValidator()  # Uses default rules

is_valid, violations = validator.validate(
    response=llm_response_text,
    context={
        "confidence": 0.85,
        "reasoning": "...",
        "indicator_values": {...},
    }
)

if not is_valid:
    for violation in violations:
        print(f"{violation.severity}: {violation.message}")
```

#### Built-in Rules

- **SpeculationDetectionRule**: Flags speculative language patterns
- **DataGroundingRule**: Ensures response references provided indicators
- **ConfidenceJustificationRule**: Validates confidence scores are justified
- **NumericHallucinationRule**: Detects fabricated numeric values

#### Custom Rules

```python
from bot.ai.prompts.guardrails import GuardrailRule, GuardrailViolation, GuardrailSeverity

class MyCustomRule(GuardrailRule):
    def __init__(self):
        super().__init__(
            name="my_rule",
            severity=GuardrailSeverity.WARNING,
            description="Custom validation logic"
        )

    def validate(self, response, context):
        violations = []
        # Your validation logic
        return violations

validator.add_rule(MyCustomRule())
```

### 4. Trading Templates (`trading.py`)

Pre-built templates for common trading analysis tasks.

#### TrendAnalysisPrompt

Predicts trend direction with confidence scoring.

```python
from bot.ai.prompts.trading import TrendAnalysisPrompt, create_user_prompt_for_indicators

prompt = TrendAnalysisPrompt()
system_prompt = prompt.render()
user_prompt = create_user_prompt_for_indicators(
    symbol="BTC/USDT",
    timeframe="1h",
    indicators={
        "price": 43250.50,
        "rsi": 65.3,
        "macd_histogram": 0.0042,
        # ... other indicators
    }
)

# Get output schema for validation
output_schema = prompt.get_output_schema()
```

#### RiskAnalysisPrompt

Assesses risk factors for potential trades.

```python
from bot.ai.prompts.trading import RiskAnalysisPrompt

prompt = RiskAnalysisPrompt()
system_prompt = prompt.render()
# Returns risk_level, risk_score, risk_factors, recommendations
```

#### MarketSentimentPrompt

Analyzes overall market sentiment and momentum.

```python
from bot.ai.prompts.trading import MarketSentimentPrompt

prompt = MarketSentimentPrompt()
system_prompt = prompt.render()
# Returns sentiment, strength, momentum, analysis
```

## Prompt Update Workflow

### 1. Creating a New Prompt Version

When updating an existing prompt template:

```python
# Define new version
VERSION_1_1_0 = PromptVersion(
    version="1.1.0",
    created_at=int(datetime.now().timestamp() * 1000),
    author="Your Name",
    changelog="Added support for volume analysis indicators",
    performance_notes="Improved confidence calibration by 5%",
    is_active=True,
)

# Update template with new version
class TrendAnalysisPrompt(PromptTemplate):
    def __init__(self, version: PromptVersion = VERSION_1_1_0):
        super().__init__(
            name="trend_analysis",
            description="...",
            version=version,
        )
```

### 2. A/B Testing

Test multiple prompt versions simultaneously:

```python
VERSION_A = PromptVersion(
    version="2.0.0-a",
    a_b_test_group="A",
    changelog="Conservative confidence scoring",
    is_active=True,
)

VERSION_B = PromptVersion(
    version="2.0.0-b",
    a_b_test_group="B",
    changelog="Aggressive confidence scoring",
    is_active=True,
)

# Use different versions based on user/session
if user_id % 2 == 0:
    prompt = TrendAnalysisPrompt(version=VERSION_A)
else:
    prompt = TrendAnalysisPrompt(version=VERSION_B)

# Track performance metrics for each version
metadata = prompt.get_metadata()
log_metrics(metadata["a_b_test_group"], prediction_accuracy)
```

### 3. Schema Evolution

When evolving output schemas:

```python
# Version 1.0.0 schema
schema_v1 = OutputSchema(
    name="prediction",
    version="1.0.0",
    fields=[
        FieldDefinition(name="direction", ...),
        FieldDefinition(name="confidence", ...),
    ]
)

# Version 2.0.0 schema with additional field
schema_v2 = OutputSchema(
    name="prediction",
    version="2.0.0",
    fields=[
        FieldDefinition(name="direction", ...),
        FieldDefinition(name="confidence", ...),
        FieldDefinition(
            name="timeframe_outlook",  # New field
            required=False,  # Make optional for backward compatibility
            ...
        ),
    ]
)
```

### 4. Guardrail Updates

Adding new validation rules:

```python
# Create new rule
class VolumeAnalysisRule(GuardrailRule):
    def __init__(self):
        super().__init__(
            name="volume_analysis",
            severity=GuardrailSeverity.WARNING,
            description="Ensures volume indicators are considered"
        )

    def validate(self, response, context):
        # Implementation
        pass

# Add to validator
validator = GuardrailValidator()
validator.add_rule(VolumeAnalysisRule())
```

## Best Practices

### 1. Versioning Strategy

- **Major version (X.0.0)**: Breaking changes to output schema or prompt structure
- **Minor version (1.X.0)**: New features, additional fields (backward compatible)
- **Patch version (1.0.X)**: Bug fixes, wording improvements

### 2. Chain-of-Thought Design

Always include explicit reasoning steps:

```
1. DATA REVIEW: What indicators do we have?
2. SIGNAL IDENTIFICATION: What does each indicator suggest?
3. CONFLUENCE: Do indicators agree?
4. SYNTHESIS: What's the overall picture?
5. CONFIDENCE: How certain are we?
```

### 3. Schema Design

- Use `required=True` for critical fields
- Set reasonable constraints (min/max values)
- Use enums for categorical values
- Add descriptions for all fields

### 4. Guardrail Configuration

- Start with WARNING severity for new rules
- Upgrade to ERROR after validation
- Use CRITICAL sparingly for safety-critical checks
- Test guardrails against historical responses

### 5. Testing

```python
# Unit test for prompt rendering
def test_prompt_render():
    prompt = TrendAnalysisPrompt()
    rendered = prompt.render()
    assert "expert" in rendered.lower()
    assert "JSON" in rendered

# Integration test with actual LLM
async def test_full_workflow():
    prompt = TrendAnalysisPrompt()
    response = await llm_manager.generate(
        prompt=user_prompt,
        system_prompt=prompt.render(),
        json_schema=prompt.get_output_schema().to_json_schema(),
    )

    validator = SchemaValidator()
    assert validator.validate(response, prompt.get_output_schema())

    guardrail = GuardrailValidator()
    is_valid, violations = guardrail.validate(response, context)
    assert is_valid
```

## Performance Monitoring

Track prompt performance metrics:

```python
# Log prompt usage
metadata = prompt.get_metadata()
logger.info(
    "prompt_used",
    name=metadata["name"],
    version=metadata["version"],
    usage_count=metadata["usage_count"],
)

# Track prediction accuracy by version
track_accuracy(
    prompt_version=metadata["version"],
    prediction=prediction.direction,
    actual_outcome=actual_direction,
)

# Monitor validation failure rates
track_validation(
    prompt_version=metadata["version"],
    schema_valid=schema_validator.validate(...),
    guardrail_valid=guardrail_validator.validate(...),
)
```

## Migration Guide

### Updating AITrendPredictor to Use New System

```python
from bot.ai.prompts.trading import TrendAnalysisPrompt
from bot.ai.prompts.schemas import SchemaValidator
from bot.ai.prompts.guardrails import GuardrailValidator

class AITrendPredictor:
    def __init__(self, llm_manager):
        self.llm_manager = llm_manager

        # Use new prompt system
        self.prompt_template = TrendAnalysisPrompt()
        self.schema_validator = SchemaValidator(strict_mode=True)
        self.guardrail_validator = GuardrailValidator()

    async def predict_trend(self, indicators, symbol, timeframe):
        # Render prompts
        system_prompt = self.prompt_template.render()
        user_prompt = create_user_prompt_for_indicators(
            symbol=symbol,
            timeframe=timeframe.value,
            indicators=self._format_indicators(indicators),
        )

        # Generate response
        response = await self.llm_manager.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            json_schema=self.prompt_template.get_output_schema().to_json_schema(),
        )

        # Validate schema
        parsed = json.loads(response.content)
        if not self.schema_validator.validate(parsed, self.prompt_template.get_output_schema()):
            raise ValueError(f"Schema validation failed: {self.schema_validator.get_errors()}")

        # Check guardrails
        is_valid, violations = self.guardrail_validator.validate(
            response=response.content,
            context={
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"],
                "indicator_values": self._format_indicators(indicators),
            }
        )

        if not is_valid:
            logger.warning("guardrail_violations", violations=[v.to_dict() for v in violations])

        return parsed
```

## Troubleshooting

### Schema Validation Failures

```python
validator = SchemaValidator(strict_mode=False)  # Collect warnings instead of failing
is_valid = validator.validate(data, schema)

print("Errors:", validator.get_errors())
print("Warnings:", validator.get_warnings())
```

### Guardrail Tuning

```python
# Get all violations for analysis
is_valid, violations = guardrail_validator.validate(response, context)

for v in violations:
    print(f"Rule: {v.rule_name}")
    print(f"Severity: {v.severity}")
    print(f"Message: {v.message}")
    print(f"Matched: {v.matched_text}")
```

### Template Debugging

```python
# Get template metadata
metadata = prompt.get_metadata()
print(f"Version: {metadata['version']}")
print(f"Usage count: {metadata['usage_count']}")
print(f"A/B group: {metadata['a_b_test_group']}")

# Get required context variables
builder = SystemPromptBuilder(...)
required_vars = builder.get_required_variables()
print(f"Required variables: {required_vars}")
```

## Future Enhancements

Potential improvements to consider:

1. **Prompt Registry**: Central registry for version management
2. **Automated Testing**: Generate test cases from schemas
3. **Performance Dashboard**: Real-time prompt performance monitoring
4. **Dynamic Adjustment**: Auto-tune prompts based on feedback
5. **Multi-language Support**: Templates in multiple languages
6. **Prompt Compression**: Optimize token usage while maintaining effectiveness

## References

- JSON Schema specification: https://json-schema.org/
- Semantic versioning: https://semver.org/
- Chain-of-thought prompting: Research papers on CoT reasoning
- Guardrail patterns: Best practices for LLM safety
