"""Compatibility layer exposing aggregation utilities at the repository root.

The refactor moved the aggregation implementation into ``saaaaaa.processing``.
Legacy integrations (and the provided tests) still import ``aggregation`` from
the repository root. Importing the real module here keeps backwards
compatibility without duplicating any logic.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import List

_SOURCE_MODULE_NAME = "saaaaaa.processing.aggregation"
_source: ModuleType = import_module(_SOURCE_MODULE_NAME)

if hasattr(_source, "__all__"):
    public_names: List[str] = list(getattr(_source, "__all__"))  # type: ignore[list-item]
else:
    public_names = [name for name in dir(_source) if not name.startswith("_")]

globals().update({name: getattr(_source, name) for name in public_names})

__all__ = public_names
