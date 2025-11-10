"""Compatibility shim for argument routing utilities.

.. deprecated::
    This module is deprecated. Import directly from:
    - saaaaaa.core.orchestrator.arg_router (deprecated, use ExtendedArgRouter instead)
    - saaaaaa.core.orchestrator.arg_router_extended (recommended)
"""
import warnings

warnings.warn(
    "Importing from orchestrator.arg_router is deprecated. "
    "Use saaaaaa.core.orchestrator.arg_router_extended.ExtendedArgRouter instead.",
    DeprecationWarning,
    stacklevel=2,
)

from saaaaaa.core.orchestrator.arg_router import *  # noqa: F401,F403
