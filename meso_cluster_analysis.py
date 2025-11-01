
"""Compatibility wrapper that re-exports ``saaaaaa.analysis.meso_cluster_analysis`` under the legacy name.

The implementation now lives in ``saaaaaa.analysis.meso_cluster_analysis``, but several integration tests
still import the module from the project root. Importing and re-exporting the
public attributes keeps those imports working without duplicating logic.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import List

_SOURCE_MODULE_NAME = "saaaaaa.analysis.meso_cluster_analysis"
try:
    _source: ModuleType = import_module(_SOURCE_MODULE_NAME)
except ImportError as e:
    raise ImportError(
        f"Could not import compatibility source module '{_SOURCE_MODULE_NAME}': {e}"
    ) from e

__all_from_source = getattr(_source, "__all__", None)
if __all_from_source is not None:
    public_names: List[str] = list(__all_from_source)  # type: ignore[list-item]
else:
    public_names = [name for name in dir(_source) if not name.startswith("_")]

globals().update({name: getattr(_source, name) for name in public_names})

__all__ = public_names
