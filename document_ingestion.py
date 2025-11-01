
"""Compatibility wrapper that re-exports ``saaaaaa.processing.document_ingestion`` under the legacy name.

The implementation now lives in ``saaaaaa.processing.document_ingestion``, but several integration tests
still import the module from the project root. Importing and re-exporting the
public attributes keeps those imports working without duplicating logic.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import List

_SOURCE_MODULE_NAME = "saaaaaa.processing.document_ingestion"
try:
    _source: ModuleType = import_module(_SOURCE_MODULE_NAME)
except Exception as e:
    raise ImportError(f"Failed to import compatibility source module '{_SOURCE_MODULE_NAME}'") from e

if hasattr(_source, "__all__"):
    public_names: List[str] = list(getattr(_source, "__all__"))  # type: ignore[list-item]
else:
    public_names = [name for name in dir(_source) if not name.startswith("_")]

globals().update({name: getattr(_source, name) for name in public_names})

_ALIASES = {
    "DocumentIndexes": "DocumentIndexesV1",
    "StructuredText": "StructuredTextV1",
}
for alias, target in _ALIASES.items():
    if alias not in globals() and hasattr(_source, target):
        globals()[alias] = getattr(_source, target)
        public_names.append(alias)

__all__ = public_names
