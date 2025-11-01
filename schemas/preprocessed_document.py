"""Lightweight dataclasses describing preprocessed document payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Mapping, Tuple

_EmptyMapping: Mapping[str, Any] = MappingProxyType({})


def _freeze_mapping(data: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not data:
        return _EmptyMapping
    if isinstance(data, MappingProxyType):
        return data
    return MappingProxyType(dict(data))


@dataclass(frozen=True)
class StructuredSection:
    """Structured section extracted from a document."""

    title: str
    start_char: int
    content: str


@dataclass(frozen=True)
class StructuredTextV1:
    """Structured representation of document text."""

    full_text: str
    sections: Tuple[StructuredSection, ...] = field(default_factory=tuple)
    page_boundaries: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SentenceMetadata:
    """Metadata associated with a sentence."""

    index: int
    page_number: int | None = None
    start_char: int | None = None
    end_char: int | None = None
    extra: Mapping[str, Any] = field(default_factory=lambda: _EmptyMapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _freeze_mapping(self.extra))


@dataclass(frozen=True)
class TableAnnotation:
    """Annotation describing an extracted table."""

    table_id: str
    label: str
    attributes: Mapping[str, Any] = field(default_factory=lambda: _EmptyMapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", _freeze_mapping(self.attributes))


@dataclass(frozen=True)
class DocumentIndexesV1:
    """Inverted indexes used for fast lookups inside a document."""

    term_index: Mapping[str, Tuple[int, ...]] = field(default_factory=lambda: _EmptyMapping)
    numeric_index: Mapping[str, Tuple[int, ...]] = field(default_factory=lambda: _EmptyMapping)
    temporal_index: Mapping[str, Tuple[int, ...]] = field(default_factory=lambda: _EmptyMapping)
    entity_index: Mapping[str, Tuple[int, ...]] = field(default_factory=lambda: _EmptyMapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "term_index", _freeze_mapping(self.term_index))
        object.__setattr__(self, "numeric_index", _freeze_mapping(self.numeric_index))
        object.__setattr__(self, "temporal_index", _freeze_mapping(self.temporal_index))
        object.__setattr__(self, "entity_index", _freeze_mapping(self.entity_index))


@dataclass(frozen=True)
class PreprocessedDocument:
    """Normalized document produced by the ingestion pipeline."""

    document_id: str
    full_text: str
    sentences: Tuple[str, ...]
    language: str
    structured_text: StructuredTextV1
    sentence_metadata: Tuple[SentenceMetadata, ...]
    tables: Tuple[TableAnnotation, ...]
    indexes: DocumentIndexesV1
    metadata: Mapping[str, Any] = field(default_factory=lambda: _EmptyMapping)
    ingested_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        if self.ingested_at is None:
            object.__setattr__(self, "ingested_at", datetime.utcnow())


__all__ = [
    "StructuredSection",
    "StructuredTextV1",
    "SentenceMetadata",
    "TableAnnotation",
    "DocumentIndexesV1",
    "PreprocessedDocument",
]
