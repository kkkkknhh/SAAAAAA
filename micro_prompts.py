
"""Compatibility wrapper that re-exports ``saaaaaa.analysis.micro_prompts`` under the legacy name.

The implementation now lives in ``saaaaaa.analysis.micro_prompts``, but several integration tests
still import the module from the project root. Importing and re-exporting the
public attributes keeps those imports working without duplicating logic.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import List

_SOURCE_MODULE_NAME = "saaaaaa.analysis.micro_prompts"
_source: ModuleType = import_module(_SOURCE_MODULE_NAME)

if hasattr(_source, "__all__"):
    public_names: List[str] = list(getattr(_source, "__all__"))  # type: ignore[list-item]
else:
    public_names = [name for name in dir(_source) if not name.startswith("_")]

globals().update({name: getattr(_source, name) for name in public_names})

__all__ = public_names
