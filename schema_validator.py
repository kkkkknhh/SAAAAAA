"""Compatibility wrapper for direct schema validator imports."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

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
