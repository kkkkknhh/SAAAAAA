"""Validation module for pre-execution checks and preconditions."""

from .architecture_validator import (
    ArchitectureValidationResult,
    validate_architecture,
    write_validation_report,
)
from .golden_rule import GoldenRuleValidator, GoldenRuleViolation

__all__ = [
    "ArchitectureValidationResult",
    "GoldenRuleValidator",
    "GoldenRuleViolation",
    "validate_architecture",
    "write_validation_report",
]
