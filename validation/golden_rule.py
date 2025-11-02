"""Compatibility shim for golden rule validator."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.utils.validation.golden_rule import (  # noqa: F401, E402
    GoldenRuleValidator,
    GoldenRuleViolation,
)

__all__ = [
    "GoldenRuleValidator",
    "GoldenRuleViolation",
]
