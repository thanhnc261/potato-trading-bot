"""
Base prompt template system with versioning support.

This module provides the foundation for creating reusable, versioned prompt
templates that can be used across different trading analysis workflows.
It supports A/B testing, prompt evolution tracking, and structured template
management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class PromptType(str, Enum):
    """Types of prompts in the system."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class PromptVersion:
    """
    Version metadata for prompt templates.

    Tracks prompt evolution for A/B testing and performance analysis.

    Attributes:
        version: Semantic version string (e.g., "1.0.0", "1.1.0")
        created_at: Unix timestamp when version was created
        author: Creator of this version
        changelog: Description of changes from previous version
        performance_notes: Notes on observed performance (optional)
        is_active: Whether this version is currently in use
        a_b_test_group: A/B test group identifier (optional, e.g., "A", "B")
    """

    version: str
    created_at: int
    author: str
    changelog: str
    performance_notes: str = ""
    is_active: bool = True
    a_b_test_group: str | None = None

    def __post_init__(self) -> None:
        """Validate version format."""
        parts = self.version.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError(
                f"Version must follow semantic versioning (x.y.z), got: {self.version}"
            )


class PromptTemplate(ABC):
    """
    Abstract base class for all prompt templates.

    Provides versioning, rendering, and metadata tracking capabilities.
    Subclasses must implement render() to generate the actual prompt text.
    """

    def __init__(
        self,
        name: str,
        description: str,
        version: PromptVersion,
        prompt_type: PromptType = PromptType.USER,
    ):
        """
        Initialize prompt template.

        Args:
            name: Unique identifier for the prompt template
            description: Human-readable description of prompt purpose
            version: Version metadata for tracking evolution
            prompt_type: Type of prompt (system, user, assistant)
        """
        self.name = name
        self.description = description
        self.version = version
        self.prompt_type = prompt_type
        self.usage_count = 0
        self.last_used: int | None = None

        logger.info(
            "prompt_template_initialized",
            name=name,
            version=version.version,
            prompt_type=prompt_type.value,
        )

    @abstractmethod
    def render(self, **kwargs: Any) -> str:
        """
        Render the prompt template with provided context variables.

        Args:
            **kwargs: Context variables to inject into the template

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If required context variables are missing
        """
        pass

    def validate_context(self, required_keys: set[str], context: dict[str, Any]) -> None:
        """
        Validate that all required context keys are present.

        Args:
            required_keys: Set of required context variable names
            context: Context dictionary to validate

        Raises:
            ValueError: If any required keys are missing
        """
        missing_keys = required_keys - set(context.keys())
        if missing_keys:
            raise ValueError(f"Missing required context variables for {self.name}: {missing_keys}")

    def track_usage(self) -> None:
        """Track usage statistics for analytics and optimization."""
        self.usage_count += 1
        self.last_used = int(datetime.now().timestamp() * 1000)

        logger.debug(
            "prompt_template_used",
            name=self.name,
            version=self.version.version,
            usage_count=self.usage_count,
        )

    def get_metadata(self) -> dict[str, Any]:
        """
        Get template metadata including version and usage statistics.

        Returns:
            Dictionary with template metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version.version,
            "prompt_type": self.prompt_type.value,
            "usage_count": self.usage_count,
            "last_used": self.last_used,
            "is_active": self.version.is_active,
            "a_b_test_group": self.version.a_b_test_group,
        }


@dataclass
class PromptComponent:
    """
    Reusable component that can be composed into larger prompts.

    Components allow modular prompt construction and easy testing of
    different combinations.

    Attributes:
        name: Component identifier
        content: Template content with placeholders
        required_variables: Set of required context variables
        optional_variables: Set of optional context variables with defaults
    """

    name: str
    content: str
    required_variables: set[str] = field(default_factory=set)
    optional_variables: dict[str, Any] = field(default_factory=dict)

    def render(self, **context: Any) -> str:
        """
        Render component with context variables.

        Args:
            **context: Context variables for rendering

        Returns:
            Rendered component text

        Raises:
            ValueError: If required variables are missing
        """
        # Check required variables
        missing = self.required_variables - set(context.keys())
        if missing:
            raise ValueError(f"Component {self.name} missing required variables: {missing}")

        # Merge optional defaults with provided context
        full_context = {**self.optional_variables, **context}

        # Render template
        try:
            return self.content.format(**full_context)
        except KeyError as e:
            raise ValueError(f"Component {self.name} template error: {e}")


class SystemPromptBuilder:
    """
    Builder for constructing complex system prompts from reusable components.

    Allows composing system prompts from modular pieces, making it easier to
    test different configurations and maintain consistency.
    """

    def __init__(self, name: str, version: PromptVersion):
        """
        Initialize system prompt builder.

        Args:
            name: Identifier for the system prompt
            version: Version metadata
        """
        self.name = name
        self.version = version
        self.components: list[PromptComponent] = []
        self.separator = "\n\n"

        logger.info(
            "system_prompt_builder_initialized",
            name=name,
            version=version.version,
        )

    def add_component(self, component: PromptComponent) -> "SystemPromptBuilder":
        """
        Add a component to the system prompt.

        Args:
            component: PromptComponent to add

        Returns:
            Self for method chaining
        """
        self.components.append(component)
        logger.debug("component_added", name=component.name, total=len(self.components))
        return self

    def set_separator(self, separator: str) -> "SystemPromptBuilder":
        """
        Set separator between components.

        Args:
            separator: String to use between components

        Returns:
            Self for method chaining
        """
        self.separator = separator
        return self

    def build(self, **context: Any) -> str:
        """
        Build the complete system prompt by rendering all components.

        Args:
            **context: Context variables for component rendering

        Returns:
            Complete system prompt string

        Raises:
            ValueError: If any component fails to render
        """
        if not self.components:
            raise ValueError(f"System prompt builder {self.name} has no components")

        rendered_components = []
        for component in self.components:
            try:
                rendered = component.render(**context)
                rendered_components.append(rendered)
            except ValueError as e:
                logger.error(
                    "component_render_failed",
                    component=component.name,
                    error=str(e),
                )
                raise

        system_prompt = self.separator.join(rendered_components)

        logger.info(
            "system_prompt_built",
            name=self.name,
            version=self.version.version,
            components=len(self.components),
            length=len(system_prompt),
        )

        return system_prompt

    def get_required_variables(self) -> set[str]:
        """
        Get all required variables across all components.

        Returns:
            Set of all required variable names
        """
        all_required = set()
        for component in self.components:
            all_required.update(component.required_variables)
        return all_required
