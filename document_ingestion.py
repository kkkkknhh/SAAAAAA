
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

_all = getattr(_source, "__all__", None)
if isinstance(_all, (list, tuple)):
    public_names = [name for name in _all if isinstance(name, str)]
else:
    public_names = [name for name in dir(_source) if not name.startswith("_")]
# Remove duplicates while preserving order
public_names = list(dict.fromkeys(public_names))

for name in public_names:
    try:
        globals()[name] = getattr(_source, name)
    except AttributeError:
        # Skip names that may be listed but not actually present on the source module
        continue

_ALIASES = {
    "DocumentIndexes": "DocumentIndexesV1",
    "StructuredText": "StructuredTextV1",
}
for alias, target in _ALIASES.items():
    if alias not in globals() and hasattr(_source, target):
        globals()[alias] = getattr(_source, target)
        public_names.append(alias)

__all__ = public_names
