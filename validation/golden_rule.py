"""Compatibility shim for golden rule validator."""
from saaaaaa.utils.validation.golden_rule import (  # noqa: F401
    GoldenRuleValidator,
    GoldenRuleViolation,
)

__all__ = [
    "GoldenRuleValidator",
    "GoldenRuleViolation",
]
