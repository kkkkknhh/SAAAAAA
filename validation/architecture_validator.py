"""Compatibility shim for architecture validator."""
from saaaaaa.utils.validation.architecture_validator import (  # noqa: F401
    ArchitectureValidator,
    ArchitectureViolation,
)

__all__ = [
    "ArchitectureValidator",
    "ArchitectureViolation",
]
