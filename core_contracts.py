"""Compatibility shim that exposes ``saaaaaa.utils.core_contracts``."""

from __future__ import annotations

from saaaaaa.utils.core_contracts import *  # noqa: F401,F403

__all__ = [name for name in dir() if not name.startswith("_")]
