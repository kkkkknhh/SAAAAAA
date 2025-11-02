"""Compatibility wrapper for direct schema validator imports."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.utils.validation.schema_validator import (  # noqa: F401, E402
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
