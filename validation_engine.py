"""Compatibility wrapper for the validation engine."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.utils.validation_engine import (  # noqa: F401, E402
    ValidationEngine,
    ValidationPredicates,
    ValidationReport,
    ValidationResult,
)

__all__ = [
    "ValidationEngine",
    "ValidationPredicates",
    "ValidationReport",
    "ValidationResult",
]
