"""
LLM Manager - Unified interface for multiple LLM providers with fallback chain.

This module provides a comprehensive LLM management system that:
- Supports multiple providers (OpenAI, Anthropic, local models)
- Implements fallback chain with automatic retries
- Tracks usage costs per provider
- Enforces monthly budget limits
- Caches responses with configurable TTL
- Implements circuit breaker for repeated failures
- Validates responses against JSON schemas
- Handles async API calls with timeouts
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import structlog
from anthropic import AsyncAnthropic  # type: ignore[import-not-found]
from openai import AsyncOpenAI  # type: ignore[import-not-found]
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failures exceed threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class LLMResponse:
    """
    Standardized LLM response format.

    Attributes:
        content: Response text content
        provider: Provider that generated the response
        model: Model name used
        tokens_used: Total tokens consumed (prompt + completion)
        prompt_tokens: Tokens in prompt
        completion_tokens: Tokens in completion
        cost: Cost in USD for this request
        latency_ms: Request latency in milliseconds
        cached: Whether response came from cache
        metadata: Additional provider-specific metadata
    """

    content: str
    provider: ProviderType
    model: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    cost: Decimal
    latency_ms: float
    cached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UsageStats:
    """
    Usage statistics for a provider.

    Attributes:
        total_requests: Total number of requests made
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        total_tokens: Total tokens consumed
        total_cost: Total cost in USD
        average_latency_ms: Average request latency
        last_request_time: Timestamp of last request
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: Decimal = Decimal("0.0")
    average_latency_ms: float = 0.0
    last_request_time: datetime | None = None
    cache_hits: int = 0
    cache_misses: int = 0

    def update_success(self, tokens: int, cost: Decimal, latency_ms: float) -> None:
        """Update stats after successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_tokens += tokens
        self.total_cost += cost
        # Calculate running average latency
        total_latency = self.average_latency_ms * (self.successful_requests - 1)
        self.average_latency_ms = (total_latency + latency_ms) / self.successful_requests
        self.last_request_time = datetime.now()

    def update_failure(self) -> None:
        """Update stats after failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_request_time = datetime.now()

    def update_cache_hit(self) -> None:
        """Update stats after cache hit."""
        self.cache_hits += 1

    def update_cache_miss(self) -> None:
        """Update stats after cache miss."""
        self.cache_misses += 1


@dataclass
class CircuitBreaker:
    """
    Circuit breaker implementation for provider failure handling.

    The circuit breaker prevents cascading failures by temporarily blocking
    requests to a failing provider and allowing periodic test requests.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        half_open_max_requests: Max requests allowed in half-open state
        state: Current circuit state
        failure_count: Current consecutive failures
        last_failure_time: Timestamp of last failure
        half_open_requests: Number of requests made in half-open state
    """

    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_requests: int = 3
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: datetime | None = None
    half_open_requests: int = 0

    def record_success(self) -> None:
        """Record successful request, potentially closing circuit."""
        if self.state == CircuitState.HALF_OPEN:
            # Successful request in half-open state closes the circuit
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.half_open_requests = 0
            logger.info("circuit_breaker_closed", state=self.state.value)

    def record_failure(self) -> None:
        """Record failed request, potentially opening circuit."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            # Open circuit after threshold exceeded
            self.state = CircuitState.OPEN
            logger.warning(
                "circuit_breaker_opened",
                state=self.state.value,
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )
        elif self.state == CircuitState.HALF_OPEN:
            # Failed test request, re-open circuit
            self.state = CircuitState.OPEN
            self.half_open_requests = 0
            logger.warning("circuit_breaker_reopened", state=self.state.value)

    def can_request(self) -> bool:
        """Check if requests are allowed through circuit."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if (
                self.last_failure_time
                and (datetime.now() - self.last_failure_time).total_seconds()
                >= self.recovery_timeout
            ):
                # Transition to half-open for testing
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                logger.info("circuit_breaker_half_open", state=self.state.value)
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Allow limited test requests
            if self.half_open_requests < self.half_open_max_requests:
                self.half_open_requests += 1
                return True
            return False

        return False


class LLMProviderConfig(BaseModel):
    """
    Configuration for an LLM provider.

    Attributes:
        type: Provider type (openai, anthropic, local)
        api_key: API key for the provider
        model: Model name to use
        max_tokens: Maximum tokens in completion
        temperature: Sampling temperature (0.0-2.0)
        timeout_seconds: Request timeout in seconds
        cost_per_1k_prompt_tokens: Cost per 1000 prompt tokens in USD
        cost_per_1k_completion_tokens: Cost per 1000 completion tokens in USD
        enabled: Whether provider is enabled
        priority: Provider priority (lower = higher priority)
        base_url: Custom base URL for API (optional)
    """

    type: ProviderType = Field(description="Provider type")
    api_key: str = Field(description="API key")
    model: str = Field(description="Model name")
    max_tokens: int = Field(default=1000, ge=1, le=100000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    cost_per_1k_prompt_tokens: Decimal = Field(default=Decimal("0.0"))
    cost_per_1k_completion_tokens: Decimal = Field(default=Decimal("0.0"))
    enabled: bool = Field(default=True)
    priority: int = Field(default=0, ge=0)
    base_url: str | None = Field(default=None)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty."""
        if not v or v.strip() == "":
            raise ValueError("API key cannot be empty")
        return v

    class Config:
        """Pydantic config."""

        use_enum_values = True


class LLMManagerConfig(BaseModel):
    """
    Configuration for LLM Manager.

    Attributes:
        providers: List of provider configurations
        monthly_budget_usd: Monthly budget limit in USD
        cache_ttl_seconds: Cache TTL in seconds (default 1 hour)
        enable_caching: Whether to enable response caching
        fallback_enabled: Whether to enable fallback chain
        circuit_breaker_enabled: Whether to enable circuit breaker
        circuit_breaker_threshold: Failures before opening circuit
        circuit_breaker_timeout: Seconds before attempting recovery
    """

    providers: list[LLMProviderConfig] = Field(description="Provider configurations")
    monthly_budget_usd: Decimal = Field(default=Decimal("100.0"), ge=Decimal("0.0"))
    cache_ttl_seconds: int = Field(default=3600, ge=0)  # 1 hour default
    enable_caching: bool = Field(default=True)
    fallback_enabled: bool = Field(default=True)
    circuit_breaker_enabled: bool = Field(default=True)
    circuit_breaker_threshold: int = Field(default=5, ge=1)
    circuit_breaker_timeout: int = Field(default=60, ge=1)

    @field_validator("providers")
    @classmethod
    def validate_providers(cls, v: list[LLMProviderConfig]) -> list[LLMProviderConfig]:
        """Validate at least one provider is configured."""
        if not v:
            raise ValueError("At least one provider must be configured")
        return v


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides common interface for all LLM providers with standardized
    request/response handling, error management, and cost tracking.
    """

    def __init__(self, config: LLMProviderConfig):
        """
        Initialize provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self.stats = UsageStats()
        self.circuit_breaker = CircuitBreaker()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """
        Generate completion from prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            max_tokens: Override max tokens (optional)
            temperature: Override temperature (optional)

        Returns:
            LLMResponse with completion and metadata

        Raises:
            Exception: If generation fails
        """
        pass

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Decimal:
        """
        Calculate cost for token usage.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Cost in USD
        """
        prompt_cost = Decimal(prompt_tokens) / Decimal(1000) * self.config.cost_per_1k_prompt_tokens
        completion_cost = (
            Decimal(completion_tokens) / Decimal(1000) * self.config.cost_per_1k_completion_tokens
        )
        return prompt_cost + completion_cost

    def get_stats(self) -> UsageStats:
        """Get usage statistics for this provider."""
        return self.stats


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def __init__(self, config: LLMProviderConfig):
        """
        Initialize OpenAI provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate completion using OpenAI API."""
        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=temperature if temperature is not None else self.config.temperature,
            )

            latency_ms = (time.time() - start_time) * 1000
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = prompt_tokens + completion_tokens
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            content = response.choices[0].message.content or ""

            return LLMResponse(
                content=content,
                provider=ProviderType.OPENAI,
                model=self.config.model,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                latency_ms=latency_ms,
                cached=False,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "response_id": response.id,
                },
            )

        except Exception as e:
            logger.error(
                "openai_generation_failed",
                error=str(e),
                model=self.config.model,
            )
            raise


class AnthropicProvider(LLMProvider):
    """Anthropic provider implementation."""

    def __init__(self, config: LLMProviderConfig):
        """
        Initialize Anthropic provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate completion using Anthropic API."""
        start_time = time.time()

        try:
            kwargs = {
                "model": self.config.model,
                "max_tokens": max_tokens or self.config.max_tokens,
                "temperature": temperature if temperature is not None else self.config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            response = await self.client.messages.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens
            cost = self.calculate_cost(prompt_tokens, completion_tokens)

            content = response.content[0].text if response.content else ""

            return LLMResponse(
                content=content,
                provider=ProviderType.ANTHROPIC,
                model=self.config.model,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                latency_ms=latency_ms,
                cached=False,
                metadata={
                    "stop_reason": response.stop_reason,
                    "response_id": response.id,
                },
            )

        except Exception as e:
            logger.error(
                "anthropic_generation_failed",
                error=str(e),
                model=self.config.model,
            )
            raise


class LocalProvider(LLMProvider):
    """
    Local model provider implementation.

    This is a placeholder for local model integration (e.g., Ollama, llama.cpp).
    Implement actual local model inference based on specific requirements.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate completion using local model."""
        # Placeholder implementation
        # In production, integrate with Ollama, llama.cpp, or other local inference
        raise NotImplementedError("Local provider not yet implemented")


class ResponseCache:
    """
    Response cache with TTL support.

    Caches LLM responses to reduce costs and latency for repeated queries.
    Uses SHA256 hash of (prompt + system_prompt + model) as cache key.
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[LLMResponse, datetime]] = {}

    def _generate_key(
        self,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """
        Generate cache key from request parameters.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            SHA256 hash as cache key
        """
        key_data = f"{prompt}|{system_prompt or ''}|{model}|{max_tokens}|{temperature}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse | None:
        """
        Get cached response if available and not expired.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(prompt, system_prompt, model, max_tokens, temperature)

        if key not in self._cache:
            return None

        response, timestamp = self._cache[key]

        # Check if entry expired
        if (datetime.now() - timestamp).total_seconds() > self.ttl_seconds:
            del self._cache[key]
            return None

        # Mark as cached and return
        response.cached = True
        return response

    def set(
        self,
        prompt: str,
        system_prompt: str | None,
        model: str,
        max_tokens: int,
        temperature: float,
        response: LLMResponse,
    ) -> None:
        """
        Cache response.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            model: Model name
            max_tokens: Max tokens
            temperature: Temperature
            response: Response to cache
        """
        key = self._generate_key(prompt, system_prompt, model, max_tokens, temperature)
        self._cache[key] = (response, datetime.now())

    def clear_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            key
            for key, (_, timestamp) in self._cache.items()
            if (now - timestamp).total_seconds() > self.ttl_seconds
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


class ResponseValidator:
    """
    Response validation utilities.

    Validates LLM responses for JSON schema compliance and basic
    hallucination detection.
    """

    @staticmethod
    def validate_json(response: str, schema: dict[str, Any] | None = None) -> bool:
        """
        Validate response is valid JSON and optionally matches schema.

        Args:
            response: Response text to validate
            schema: Optional JSON schema to validate against

        Returns:
            True if valid, False otherwise
        """
        try:
            data = json.loads(response)

            if schema:
                # Basic schema validation
                # For production, use jsonschema library for full validation
                return ResponseValidator._validate_schema(data, schema)

            return True

        except json.JSONDecodeError:
            return False

    @staticmethod
    def _validate_schema(data: Any, schema: dict[str, Any]) -> bool:
        """
        Basic schema validation.

        Args:
            data: Data to validate
            schema: Schema to validate against

        Returns:
            True if valid
        """
        # Simplified schema validation
        # In production, use jsonschema library for comprehensive validation
        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "object" and not isinstance(data, dict):
                return False
            if expected_type == "array" and not isinstance(data, list):
                return False
            if expected_type == "string" and not isinstance(data, str):
                return False
            if expected_type == "number" and not isinstance(data, (int, float)):
                return False

        if "required" in schema and isinstance(data, dict):
            for key in schema["required"]:
                if key not in data:
                    return False

        return True

    @staticmethod
    def check_hallucination_indicators(response: str) -> tuple[bool, list[str]]:
        """
        Check for common hallucination indicators.

        This is a basic implementation that checks for suspicious patterns.
        For production use, consider more sophisticated hallucination detection.

        Args:
            response: Response text to check

        Returns:
            Tuple of (is_suspicious, list of indicators found)
        """
        indicators = []

        # Check for uncertainty phrases
        uncertainty_phrases = [
            "i'm not sure",
            "i don't know",
            "i cannot confirm",
            "may or may not",
            "possibly",
            "perhaps",
        ]
        for phrase in uncertainty_phrases:
            if phrase in response.lower():
                indicators.append(f"uncertainty_phrase: {phrase}")

        # Check for contradictions
        if "however" in response.lower() and "but" in response.lower():
            indicators.append("potential_contradiction")

        # Check for excessive hedging
        hedge_words = ["maybe", "might", "could", "possibly", "perhaps"]
        hedge_count = sum(1 for word in hedge_words if word in response.lower())
        if hedge_count > 3:
            indicators.append(f"excessive_hedging: {hedge_count} hedge words")

        is_suspicious = len(indicators) > 0

        return is_suspicious, indicators


class LLMManager:
    """
    Unified LLM manager with multi-provider support and fallback chain.

    Provides centralized management of multiple LLM providers with:
    - Automatic fallback chain
    - Usage tracking and cost calculation
    - Monthly budget enforcement
    - Response caching
    - Circuit breaker for failure handling
    - Response validation
    - Async API calls with timeout
    """

    def __init__(self, config: LLMManagerConfig):
        """
        Initialize LLM manager.

        Args:
            config: Manager configuration
        """
        self.config = config
        self.providers: dict[str, LLMProvider] = {}
        self.cache = ResponseCache(ttl_seconds=config.cache_ttl_seconds)
        self.monthly_usage: dict[str, UsageStats] = {}
        self.current_month = datetime.now().strftime("%Y-%m")

        self._initialize_providers()

        logger.info(
            "llm_manager_initialized",
            num_providers=len(self.providers),
            cache_enabled=config.enable_caching,
            fallback_enabled=config.fallback_enabled,
            circuit_breaker_enabled=config.circuit_breaker_enabled,
        )

    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        # Sort providers by priority (lower priority value = higher priority)
        sorted_providers = sorted(self.config.providers, key=lambda p: p.priority)

        for provider_config in sorted_providers:
            if not provider_config.enabled:
                continue

            provider_key = f"{provider_config.type.value}_{provider_config.model}"

            try:
                provider: LLMProvider
                if provider_config.type == ProviderType.OPENAI:
                    provider = OpenAIProvider(provider_config)
                elif provider_config.type == ProviderType.ANTHROPIC:
                    provider = AnthropicProvider(provider_config)
                elif provider_config.type == ProviderType.LOCAL:
                    provider = LocalProvider(provider_config)
                else:
                    logger.warning(
                        "unknown_provider_type",
                        provider_type=provider_config.type,
                    )
                    continue

                # Configure circuit breaker if enabled
                if self.config.circuit_breaker_enabled:
                    provider.circuit_breaker.failure_threshold = (
                        self.config.circuit_breaker_threshold
                    )
                    provider.circuit_breaker.recovery_timeout = self.config.circuit_breaker_timeout

                self.providers[provider_key] = provider

                logger.info(
                    "provider_initialized",
                    provider_key=provider_key,
                    type=provider_config.type.value,
                    model=provider_config.model,
                    priority=provider_config.priority,
                )

            except Exception as e:
                logger.error(
                    "provider_initialization_failed",
                    provider_type=provider_config.type.value,
                    error=str(e),
                )

    def _check_monthly_budget(self) -> bool:
        """
        Check if monthly budget limit has been exceeded.

        Returns:
            True if budget allows more requests, False if exceeded
        """
        current_month = datetime.now().strftime("%Y-%m")

        # Reset tracking if new month
        if current_month != self.current_month:
            self.current_month = current_month
            self.monthly_usage = {}

        # Calculate total spending this month
        total_spent = sum(stats.total_cost for stats in self.monthly_usage.values())

        if total_spent >= self.config.monthly_budget_usd:
            logger.warning(
                "monthly_budget_exceeded",
                total_spent=float(total_spent),
                budget=float(self.config.monthly_budget_usd),
                month=current_month,
            )
            return False

        return True

    def _update_monthly_usage(self, provider_key: str, response: LLMResponse) -> None:
        """
        Update monthly usage tracking.

        Args:
            provider_key: Provider identifier
            response: LLM response with usage data
        """
        if provider_key not in self.monthly_usage:
            self.monthly_usage[provider_key] = UsageStats()

        self.monthly_usage[provider_key].update_success(
            tokens=response.tokens_used,
            cost=response.cost,
            latency_ms=response.latency_ms,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        validate_json: bool = False,
        json_schema: dict[str, Any] | None = None,
        check_hallucination: bool = False,
    ) -> LLMResponse:
        """
        Generate completion with automatic fallback chain.

        Attempts to generate completion using providers in priority order.
        Falls back to next provider on failure. Implements caching, circuit
        breaking, and budget enforcement.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            max_tokens: Override max tokens (optional)
            temperature: Override temperature (optional)
            validate_json: Whether to validate response as JSON
            json_schema: JSON schema to validate against (optional)
            check_hallucination: Whether to check for hallucination indicators

        Returns:
            LLMResponse from first successful provider

        Raises:
            Exception: If all providers fail or budget exceeded
        """
        # Check budget before proceeding
        if not self._check_monthly_budget():
            raise Exception("Monthly budget exceeded")

        # Try cache first if enabled
        if self.config.enable_caching:
            for provider_key, provider in self.providers.items():
                cached_response = self.cache.get(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=provider.config.model,
                    max_tokens=max_tokens or provider.config.max_tokens,
                    temperature=(
                        temperature if temperature is not None else provider.config.temperature
                    ),
                )

                if cached_response:
                    provider.stats.update_cache_hit()
                    logger.info(
                        "cache_hit",
                        provider_key=provider_key,
                        model=provider.config.model,
                    )
                    return cached_response

            # Log cache miss
            for provider in self.providers.values():
                provider.stats.update_cache_miss()

        # Try providers in priority order
        last_error = None
        for provider_key, provider in self.providers.items():
            # Skip if circuit breaker open
            if self.config.circuit_breaker_enabled and not provider.circuit_breaker.can_request():
                logger.warning(
                    "provider_circuit_open",
                    provider_key=provider_key,
                    state=provider.circuit_breaker.state.value,
                )
                continue

            try:
                logger.info(
                    "attempting_generation",
                    provider_key=provider_key,
                    model=provider.config.model,
                )

                # Generate with timeout
                response = await asyncio.wait_for(
                    provider.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    ),
                    timeout=provider.config.timeout_seconds,
                )

                # Validate response if requested
                if validate_json:
                    if not ResponseValidator.validate_json(response.content, json_schema):
                        logger.warning(
                            "json_validation_failed",
                            provider_key=provider_key,
                        )
                        raise ValueError("Invalid JSON response")

                if check_hallucination:
                    is_suspicious, indicators = ResponseValidator.check_hallucination_indicators(
                        response.content
                    )
                    if is_suspicious:
                        logger.warning(
                            "hallucination_indicators_detected",
                            provider_key=provider_key,
                            indicators=indicators,
                        )

                # Update stats and cache
                provider.stats.update_success(
                    tokens=response.tokens_used,
                    cost=response.cost,
                    latency_ms=response.latency_ms,
                )

                if self.config.circuit_breaker_enabled:
                    provider.circuit_breaker.record_success()

                self._update_monthly_usage(provider_key, response)

                if self.config.enable_caching:
                    self.cache.set(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=provider.config.model,
                        max_tokens=max_tokens or provider.config.max_tokens,
                        temperature=(
                            temperature if temperature is not None else provider.config.temperature
                        ),
                        response=response,
                    )

                logger.info(
                    "generation_successful",
                    provider_key=provider_key,
                    model=provider.config.model,
                    tokens=response.tokens_used,
                    cost=float(response.cost),
                    latency_ms=response.latency_ms,
                    cached=response.cached,
                )

                return response

            except TimeoutError:
                last_error = Exception(f"Timeout after {provider.config.timeout_seconds}s")
                provider.stats.update_failure()
                if self.config.circuit_breaker_enabled:
                    provider.circuit_breaker.record_failure()
                logger.error(
                    "generation_timeout",
                    provider_key=provider_key,
                    timeout_seconds=provider.config.timeout_seconds,
                )

            except Exception as e:
                last_error = e
                provider.stats.update_failure()
                if self.config.circuit_breaker_enabled:
                    provider.circuit_breaker.record_failure()
                logger.error(
                    "generation_failed",
                    provider_key=provider_key,
                    error=str(e),
                )

            # Continue to next provider if fallback enabled
            if not self.config.fallback_enabled:
                break

        # All providers failed
        error_msg = f"All providers failed. Last error: {last_error}"
        logger.error("all_providers_failed", error=error_msg)
        raise Exception(error_msg)

    def get_provider_stats(self, provider_key: str | None = None) -> dict[str, UsageStats]:
        """
        Get usage statistics for provider(s).

        Args:
            provider_key: Specific provider key, or None for all providers

        Returns:
            Dictionary mapping provider keys to usage stats
        """
        if provider_key:
            if provider_key in self.providers:
                return {provider_key: self.providers[provider_key].get_stats()}
            return {}

        return {key: provider.get_stats() for key, provider in self.providers.items()}

    def get_monthly_usage(self) -> dict[str, UsageStats]:
        """
        Get monthly usage statistics.

        Returns:
            Dictionary mapping provider keys to monthly usage stats
        """
        return self.monthly_usage.copy()

    def get_total_monthly_cost(self) -> Decimal:
        """
        Get total cost for current month.

        Returns:
            Total cost in USD
        """
        return sum((stats.total_cost for stats in self.monthly_usage.values()), Decimal("0.0"))

    def get_budget_remaining(self) -> Decimal:
        """
        Get remaining budget for current month.

        Returns:
            Remaining budget in USD
        """
        return self.config.monthly_budget_usd - self.get_total_monthly_cost()

    def get_cache_stats(self) -> dict[str, int | float]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache size and hit/miss counts
        """
        total_hits = sum(p.stats.cache_hits for p in self.providers.values())
        total_misses = sum(p.stats.cache_misses for p in self.providers.values())

        return {
            "cache_size": self.cache.size(),
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": (
                total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
            ),
        }

    def clear_cache(self) -> None:
        """Clear response cache."""
        self.cache.clear()
        logger.info("cache_cleared")

    def reset_monthly_usage(self) -> None:
        """Reset monthly usage tracking."""
        self.monthly_usage = {}
        self.current_month = datetime.now().strftime("%Y-%m")
        logger.info("monthly_usage_reset", month=self.current_month)
