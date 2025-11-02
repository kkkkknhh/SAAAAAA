"""Compatibility wrapper for ingestion data models."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.processing.document_ingestion import (  # noqa: F401, E402
    DocumentIndexes,
    PreprocessedDocument,
    RawDocument,
    StructuredText,
)

__all__ = [
    "DocumentIndexes",
    "PreprocessedDocument",
    "RawDocument",
    "StructuredText",
]
