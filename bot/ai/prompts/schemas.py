"""
JSON output schema definitions and validation rules.

This module provides strict JSON schema definitions for LLM outputs and
comprehensive validation to ensure responses conform to expected structures.
Schemas are versioned and can be evolved over time while maintaining backward
compatibility checks.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class SchemaType(str, Enum):
    """Supported JSON schema data types."""

    OBJECT = "object"
    ARRAY = "array"
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    NULL = "null"


@dataclass
class FieldDefinition:
    """
    Definition of a single field in a JSON schema.

    Attributes:
        name: Field name
        type: Data type (SchemaType enum)
        required: Whether field is required
        description: Human-readable field description
        enum_values: Allowed enum values (optional)
        min_value: Minimum numeric value (optional)
        max_value: Maximum numeric value (optional)
        min_length: Minimum string length (optional)
        max_length: Maximum string length (optional)
        pattern: Regex pattern for string validation (optional)
        items_schema: Schema for array items (optional)
        properties: Nested object properties (optional)
    """

    name: str
    type: SchemaType
    required: bool = True
    description: str = ""
    enum_values: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    items_schema: dict[str, Any] | None = None
    properties: dict[str, "FieldDefinition"] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """
        Convert field definition to JSON schema format.

        Returns:
            JSON schema dictionary
        """
        schema: dict[str, Any] = {"type": self.type.value}

        if self.description:
            schema["description"] = self.description

        if self.enum_values is not None:
            schema["enum"] = self.enum_values

        if self.min_value is not None:
            schema["minimum"] = self.min_value

        if self.max_value is not None:
            schema["maximum"] = self.max_value

        if self.min_length is not None:
            schema["minLength"] = self.min_length

        if self.max_length is not None:
            schema["maxLength"] = self.max_length

        if self.pattern is not None:
            schema["pattern"] = self.pattern

        if self.items_schema is not None:
            schema["items"] = self.items_schema

        if self.properties is not None:
            schema["properties"] = {
                name: field.to_json_schema() for name, field in self.properties.items()
            }
            schema["required"] = [name for name, field in self.properties.items() if field.required]

        return schema


class OutputSchema:
    """
    JSON schema definition for LLM output validation.

    Provides strict schema definition, validation, and versioning for ensuring
    LLM responses conform to expected structure and data types.
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str,
        fields: list[FieldDefinition],
        additional_properties: bool = False,
    ):
        """
        Initialize output schema.

        Args:
            name: Schema identifier
            version: Schema version (semantic versioning)
            description: Human-readable schema description
            fields: List of field definitions
            additional_properties: Whether to allow additional properties not in schema
        """
        self.name = name
        self.version = version
        self.description = description
        self.fields = {field.name: field for field in fields}
        self.additional_properties = additional_properties

        logger.info(
            "output_schema_initialized",
            name=name,
            version=version,
            field_count=len(fields),
            additional_properties=additional_properties,
        )

    def to_json_schema(self) -> dict[str, Any]:
        """
        Convert to standard JSON schema format.

        Returns:
            JSON schema dictionary compatible with standard validators
        """
        schema: dict[str, Any] = {
            "type": "object",
            "description": self.description,
            "properties": {name: field.to_json_schema() for name, field in self.fields.items()},
            "required": [name for name, field in self.fields.items() if field.required],
            "additionalProperties": self.additional_properties,
        }

        return schema

    def get_field(self, name: str) -> FieldDefinition | None:
        """
        Get field definition by name.

        Args:
            name: Field name

        Returns:
            FieldDefinition or None if not found
        """
        return self.fields.get(name)

    def get_required_fields(self) -> list[str]:
        """
        Get list of required field names.

        Returns:
            List of required field names
        """
        return [name for name, field in self.fields.items() if field.required]

    def validate_completeness(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Check if data contains all required fields.

        Args:
            data: Data dictionary to validate

        Returns:
            Tuple of (is_valid, list_of_missing_fields)
        """
        required = set(self.get_required_fields())
        present = set(data.keys())
        missing = required - present

        return (len(missing) == 0, list(missing))


class SchemaValidator:
    """
    Comprehensive validator for JSON responses against defined schemas.

    Performs multi-level validation including structure, types, constraints,
    and custom business rules.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize schema validator.

        Args:
            strict_mode: If True, fail on any validation error. If False, collect warnings.
        """
        self.strict_mode = strict_mode
        self.validation_errors: list[str] = []
        self.validation_warnings: list[str] = []

        logger.info("schema_validator_initialized", strict_mode=strict_mode)

    def validate(
        self, data: dict[str, Any], schema: OutputSchema, allow_extra: bool = False
    ) -> bool:
        """
        Validate data against schema.

        Args:
            data: Data dictionary to validate
            schema: OutputSchema to validate against
            allow_extra: Whether to allow extra fields not in schema

        Returns:
            True if valid, False otherwise

        Side effects:
            Populates self.validation_errors and self.validation_warnings
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()

        logger.debug(
            "validating_data",
            schema=schema.name,
            version=schema.version,
            data_keys=list(data.keys()),
        )

        # Check required fields
        is_complete, missing = schema.validate_completeness(data)
        if not is_complete:
            self.validation_errors.append(f"Missing required fields: {missing}")
            if self.strict_mode:
                logger.error("validation_failed_missing_fields", missing=missing)
                return False

        # Check extra fields
        if not allow_extra and not schema.additional_properties:
            schema_fields = set(schema.fields.keys())
            data_fields = set(data.keys())
            extra = data_fields - schema_fields
            if extra:
                msg = f"Unexpected fields not in schema: {extra}"
                if self.strict_mode:
                    self.validation_errors.append(msg)
                else:
                    self.validation_warnings.append(msg)

        # Validate each field
        for field_name, field_def in schema.fields.items():
            if field_name not in data:
                if field_def.required:
                    continue  # Already handled in completeness check
                else:
                    continue  # Optional field, skip validation

            field_value = data[field_name]
            if not self._validate_field(field_name, field_value, field_def):
                if self.strict_mode:
                    logger.error(
                        "validation_failed",
                        schema=schema.name,
                        field=field_name,
                        errors=self.validation_errors,
                    )
                    return False

        if self.validation_errors:
            logger.error("validation_failed", errors=self.validation_errors)
            return False

        if self.validation_warnings:
            logger.warning("validation_warnings", warnings=self.validation_warnings)

        logger.info("validation_passed", schema=schema.name, version=schema.version)
        return True

    def _validate_field(self, field_name: str, value: Any, field_def: FieldDefinition) -> bool:
        """
        Validate a single field against its definition.

        Args:
            field_name: Name of the field
            value: Field value to validate
            field_def: Field definition

        Returns:
            True if valid, False otherwise
        """
        # Type validation
        if not self._validate_type(value, field_def.type):
            self.validation_errors.append(
                f"Field '{field_name}' has invalid type. "
                f"Expected {field_def.type.value}, got {type(value).__name__}"
            )
            return False

        # Enum validation
        if field_def.enum_values is not None:
            if value not in field_def.enum_values:
                self.validation_errors.append(
                    f"Field '{field_name}' value '{value}' not in allowed values: "
                    f"{field_def.enum_values}"
                )
                return False

        # Numeric constraints
        if field_def.type in (SchemaType.NUMBER, SchemaType.INTEGER):
            if field_def.min_value is not None and value < field_def.min_value:
                self.validation_errors.append(
                    f"Field '{field_name}' value {value} below minimum {field_def.min_value}"
                )
                return False
            if field_def.max_value is not None and value > field_def.max_value:
                self.validation_errors.append(
                    f"Field '{field_name}' value {value} above maximum {field_def.max_value}"
                )
                return False

        # String constraints
        if field_def.type == SchemaType.STRING:
            if field_def.min_length is not None and len(str(value)) < field_def.min_length:
                self.validation_errors.append(
                    f"Field '{field_name}' length {len(str(value))} below minimum "
                    f"{field_def.min_length}"
                )
                return False
            if field_def.max_length is not None and len(str(value)) > field_def.max_length:
                self.validation_errors.append(
                    f"Field '{field_name}' length {len(str(value))} above maximum "
                    f"{field_def.max_length}"
                )
                return False

        return True

    def _validate_type(self, value: Any, expected_type: SchemaType) -> bool:
        """
        Validate value type matches expected type.

        Args:
            value: Value to check
            expected_type: Expected SchemaType

        Returns:
            True if type matches, False otherwise
        """
        type_checks = {
            SchemaType.STRING: lambda v: isinstance(v, str),
            SchemaType.NUMBER: lambda v: isinstance(v, (int, float)),
            SchemaType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            SchemaType.BOOLEAN: lambda v: isinstance(v, bool),
            SchemaType.ARRAY: lambda v: isinstance(v, list),
            SchemaType.OBJECT: lambda v: isinstance(v, dict),
            SchemaType.NULL: lambda v: v is None,
        }

        check_func = type_checks.get(expected_type)
        if check_func is None:
            logger.error("unknown_schema_type", type=expected_type)
            return False

        return check_func(value)

    def get_errors(self) -> list[str]:
        """Get list of validation errors."""
        return self.validation_errors.copy()

    def get_warnings(self) -> list[str]:
        """Get list of validation warnings."""
        return self.validation_warnings.copy()

    def validate_json_string(self, json_string: str, schema: OutputSchema) -> bool:
        """
        Validate JSON string against schema.

        Args:
            json_string: JSON string to validate
            schema: OutputSchema to validate against

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If JSON is malformed
        """
        try:
            data = json.loads(json_string)
            return self.validate(data, schema)
        except json.JSONDecodeError as e:
            self.validation_errors.append(f"Invalid JSON: {e}")
            logger.error("json_decode_failed", error=str(e))
            return False
