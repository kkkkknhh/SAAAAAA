"""Facade module exposing policy processor classes from the refactored package."""
from __future__ import annotations

from saaaaaa.processing.policy_processor import *  # noqa: F401,F403

__all__ = [
    name
    for name in dir()
    if not name.startswith("_") and name not in {"annotations"}
]
