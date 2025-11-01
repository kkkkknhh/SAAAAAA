"""Compatibility package for the legacy scoring import path."""

from __future__ import annotations

from .scoring import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
