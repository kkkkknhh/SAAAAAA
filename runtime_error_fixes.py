"""Compatibility wrapper for runtime defensive helpers."""
from saaaaaa.utils.runtime_error_fixes import (  # noqa: F401
    ensure_list_return,
    safe_list_iteration,
    safe_text_extract,
    safe_weighted_multiply,
)

__all__ = [
    "ensure_list_return",
    "safe_list_iteration",
    "safe_text_extract",
    "safe_weighted_multiply",
]
