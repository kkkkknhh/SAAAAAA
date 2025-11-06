"""Safe import utilities for optional dependencies.

This module provides utilities to handle optional dependencies gracefully,
with prescriptive error messages guiding users to install missing packages.

Design Principles:
- No silent failures - missing required dependencies raise clear errors
- Optional dependencies return None with warnings when missing
- Prescriptive messages tell users exactly what to install
- Lazy imports avoid import-time errors
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


class OptionalDependencyError(ImportError):
    """Raised when a required optional dependency is missing."""
    
    def __init__(self, package: str, hint: str | None = None, install_cmd: str | None = None):
        """
        Initialize error with package info and installation hint.
        
        Args:
            package: Name of the missing package
            hint: Usage hint explaining why it's needed
            install_cmd: Installation command (default: pip install {package})
        """
        self.package = package
        self.hint = hint
        self.install_cmd = install_cmd or f"pip install {package}"
        
        message = f"Missing required dependency: '{package}'"
        if hint:
            message += f"\n  Purpose: {hint}"
        message += f"\n  Install: {self.install_cmd}"
        
        super().__init__(message)


def try_import(
    package: str,
    *,
    required: bool = False,
    hint: str | None = None,
    install_cmd: str | None = None,
    min_version: str | None = None,
) -> ModuleType | None:
    """
    Safely import an optional dependency.
    
    Args:
        package: Package name to import (e.g., 'pyarrow', 'torch')
        required: If True, raise OptionalDependencyError when missing
        hint: Usage hint for error message
        install_cmd: Custom installation command
        min_version: Minimum version required (format: "1.2.3")
        
    Returns:
        Imported module if available, None if optional and missing
        
    Raises:
        OptionalDependencyError: If required=True and package is missing
    """
    try:
        module = __import__(package)
        
        # Check version if specified (skip if packaging not available)
        if min_version and hasattr(module, "__version__"):
            try:
                from packaging.version import parse
                if parse(module.__version__) < parse(min_version):
                    msg = (
                        f"Package '{package}' version {module.__version__} is too old. "
                        f"Minimum required: {min_version}"
                    )
                    if required:
                        raise OptionalDependencyError(package, hint=msg, install_cmd=install_cmd)
                    else:
                        logger.warning(msg)
                        return None
            except ImportError:
                # packaging not available, skip version check
                pass
        
        logger.debug(f"Successfully imported {package}")
        return module
        
    except ImportError as e:
        if required:
            raise OptionalDependencyError(package, hint=hint, install_cmd=install_cmd) from e
        else:
            log_msg = f"Optional dependency '{package}' not available"
            if hint:
                log_msg += f" (used for: {hint})"
            logger.info(log_msg)
            return None


def get_optional_import_status() -> dict[str, Any]:
    """
    Get status of all known optional dependencies.
    
    Returns:
        Dict with package availability and version info
    """
    optional_deps = {
        "pyarrow": {"hint": "Arrow serialization for CPPAdapter"},
        "torch": {"hint": "ML backends and neural models"},
        "structlog": {"hint": "Structured logging"},
        "tensorflow": {"hint": "TensorFlow ML models"},
    }
    
    status = {}
    for package, config in optional_deps.items():
        module = try_import(package, required=False, hint=config["hint"])
        if module is not None:
            version = getattr(module, "__version__", "unknown")
            status[package] = {"available": True, "version": version}
        else:
            status[package] = {"available": False, "version": None}
    
    return status
