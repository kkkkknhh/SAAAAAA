"""Compatibility wrapper for coverage gate module."""
from saaaaaa.utils.coverage_gate import (  # noqa: F401
    count_methods_in_class,
    count_producer_methods,
    enforce_coverage_gate,
    generate_audit_json,
)

__all__ = [
    "count_methods_in_class",
    "count_producer_methods",
    "enforce_coverage_gate",
    "generate_audit_json",
]
