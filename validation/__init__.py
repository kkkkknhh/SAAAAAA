"""Validation module for pre-execution checks and preconditions."""

from .golden_rule import GoldenRuleValidator, GoldenRuleViolation

__all__ = [
    "GoldenRuleValidator",
    "GoldenRuleViolation",
]
