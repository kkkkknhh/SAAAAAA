"""Compatibility wrapper for direct schema validator imports."""
from validation.schema_validator import (  # noqa: F401
    MonolithIntegrityReport,
    MonolithSchemaValidator,
    SchemaInitializationError,
    validate_monolith_schema,
)

SchemaValidator = MonolithSchemaValidator
SchemaValidationIssue = SchemaInitializationError

__all__ = [
    "MonolithIntegrityReport",
    "MonolithSchemaValidator",
    "SchemaInitializationError",
    "SchemaValidator",
    "SchemaValidationIssue",
    "validate_monolith_schema",
]
