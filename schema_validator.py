"""Backward compatible access to the monolith schema validator utilities."""

from __future__ import annotations

from saaaaaa.utils.validation.schema_validator import (
    MonolithSchemaValidator,
    MonolithIntegrityReport,
    SchemaInitializationError,
    validate_monolith_schema,
)

class SchemaValidator(MonolithSchemaValidator):
    """Legacy alias maintained for historical imports."""

__all__ = [
    "SchemaValidator",
    "MonolithSchemaValidator",
    "MonolithIntegrityReport",
    "SchemaInitializationError",
    "validate_monolith_schema",
]
