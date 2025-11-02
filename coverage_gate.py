"""Compatibility wrapper for coverage gate module."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.utils.coverage_gate import (  # noqa: F401, E402
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
