"""Compatibility wrapper for signature validation utilities."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.utils.signature_validator import (  # noqa: F401, E402
    SignatureMismatch,
    validate_call_signature,
    validate_signature,
)

# Provide backward-compatible aliases
SignatureIssue = SignatureMismatch
ValidationIssue = SignatureMismatch

__all__ = [
    "SignatureIssue",
    "SignatureMismatch",
    "ValidationIssue",
    "validate_call_signature",
    "validate_signature",
]
