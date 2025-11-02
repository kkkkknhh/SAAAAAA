"""Compatibility wrapper for the validation engine."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

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
