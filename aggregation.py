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
_source = None
public_names = None

def _load_source():
    global _source, public_names
    if _source is not None:
        return
    _source = import_module(_SOURCE_MODULE_NAME)
    if hasattr(_source, "__all__"):
        public_names = list(getattr(_source, "__all__"))  # type: ignore[list-item]
    else:
        public_names = [name for name in dir(_source) if not name.startswith("_")]
    globals().update({name: getattr(_source, name) for name in public_names})
    globals()["__all__"] = public_names

def __getattr__(name):
    _load_source()
    try:
        return globals()[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__} has no attribute {name}") from exc

def __dir__():
    _load_source()
    return sorted(list(globals().keys()))
