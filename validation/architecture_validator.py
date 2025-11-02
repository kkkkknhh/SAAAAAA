"""Compatibility wrapper for architecture validator."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.utils.validation.architecture_validator import (  # noqa: F401, E402
    ArchitectureValidationResult,
    extract_architecture_methods,
    load_architecture_spec,
    load_method_inventory,
    main,
    validate_architecture,
    write_validation_report,
)

__all__ = [
    "ArchitectureValidationResult",
    "extract_architecture_methods",
    "load_architecture_spec",
    "load_method_inventory",
    "main",
    "validate_architecture",
    "write_validation_report",
]
