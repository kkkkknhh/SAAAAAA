"""Compatibility wrapper for signature validation utilities."""
from saaaaaa.utils.signature_validator import (  # noqa: F401
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
