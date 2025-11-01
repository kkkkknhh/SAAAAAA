"""Facade exposing the recommendation engine implementation."""
from __future__ import annotations

from saaaaaa.analysis.recommendation_engine import *  # noqa: F401,F403

__all__ = [
    name
    for name in dir()
    if not name.startswith("_") and name not in {"annotations"}
]
