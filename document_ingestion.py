"""Compatibility wrapper for ingestion data models."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.processing.document_ingestion import (  # noqa: F401, E402
    DocumentLoader,
    PreprocessingEngine,
    RawDocument,
    TextExtractor,
)

__all__ = [
    "DocumentLoader",
    "PreprocessingEngine",
    "RawDocument",
    "TextExtractor",
]
