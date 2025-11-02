"""Compatibility shim for validation predicates."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.utils.validation.predicates import (  # noqa: F401, E402
    ValidationPredicates,
    ValidationResult,
)

__all__ = [
    "ValidationPredicates",
    "ValidationResult",
]
