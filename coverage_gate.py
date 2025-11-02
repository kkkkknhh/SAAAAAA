"""Compatibility wrapper for coverage gate module."""
from saaaaaa.utils.coverage_gate import (  # noqa: F401
    count_all_methods,
    count_file_methods,
    count_methods_in_class,
    main,
    validate_schema_exists,
)

__all__ = [
    "count_all_methods",
    "count_file_methods",
    "count_methods_in_class",
    "main",
    "validate_schema_exists",
]
