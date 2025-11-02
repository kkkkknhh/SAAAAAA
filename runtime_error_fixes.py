"""Compatibility wrapper for runtime defensive helpers."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

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
