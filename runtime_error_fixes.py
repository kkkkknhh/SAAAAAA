"""Compatibility wrapper for runtime defensive helpers."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.utils.runtime_error_fixes import (  # noqa: F401, E402
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
