"""Compatibility layer for optional dependencies."""

from .safe_imports import try_import, OptionalDependencyError

__all__ = ["try_import", "OptionalDependencyError"]
