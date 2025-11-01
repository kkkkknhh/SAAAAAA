"""Compatibility shim for the legacy scoring module path."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import List

_SOURCE_MODULE_NAME = "saaaaaa.analysis.scoring.scoring"
_source: ModuleType = import_module(_SOURCE_MODULE_NAME)

# Import the public API defined by the source module
for name in getattr(_source, "__all__", []):
    globals()[name] = getattr(_source, name)

# Legacy helpers that were not included in __all__ but are required by tests
_LEGACY_FUNCTIONS = [
    "score_type_a",
    "score_type_b",
    "score_type_c",
    "score_type_d",
    "score_type_e",
    "score_type_f",
]
for name in _LEGACY_FUNCTIONS:
    if hasattr(_source, name):
        globals()[name] = getattr(_source, name)

if hasattr(_source, "__all__"):
    __all__: List[str] = list(getattr(_source, "__all__"))  # type: ignore[list-item]
else:
    __all__ = [name for name in dir(_source) if not name.startswith("_")]

for legacy_name in _LEGACY_FUNCTIONS:
    if legacy_name not in __all__ and hasattr(_source, legacy_name):
        __all__.append(legacy_name)
