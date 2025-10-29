"""
Ingestion Module - Deterministic Document Processing
=====================================================

This module provides deterministic document ingestion functionality
for PDF policy documents, including:

- PDF loading and validation
- Text extraction and normalization
- Sentence/paragraph segmentation with offsets
- Table extraction and classification
- Structural index construction
- Immutable PreprocessedDocument assembly
"""

from ingestion.document_ingestion import (
    DocumentLoader,
    TextExtractor,
    PreprocessingEngine,
    PreprocessedDocument,
    RawDocument,
    IngestionError,
    ValidationError,
)

__all__ = [
    "DocumentLoader",
    "TextExtractor",
    "PreprocessingEngine",
    "PreprocessedDocument",
    "RawDocument",
    "IngestionError",
    "ValidationError",
]
