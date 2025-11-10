"""
Canon Policy Package (CPP) Data Models

Minimal implementation of CPP models for SPC ingestion compatibility.
These models represent the structured output from the SPC ingestion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ChunkResolution(Enum):
    """Chunk resolution levels."""
    MICRO = "micro"
    MESO = "meso"
    MACRO = "macro"


@dataclass
class TextSpan:
    """Text span with start and end positions."""
    start: int
    end: int


@dataclass
class Confidence:
    """Confidence scores for various aspects."""
    layout: float = 1.0
    ocr: float = 1.0
    typing: float = 1.0


@dataclass
class PolicyFacet:
    """Policy-related facets of a chunk."""
    axes: list[str] = field(default_factory=list)
    programs: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)


@dataclass
class TimeFacet:
    """Temporal facets of a chunk."""
    years: list[int] = field(default_factory=list)
    periods: list[str] = field(default_factory=list)


@dataclass
class GeoFacet:
    """Geographic facets of a chunk."""
    territories: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)


@dataclass
class BudgetInfo:
    """Budget information."""
    source: str = ""
    use: str = ""
    amount: float = 0.0
    year: Optional[int] = None
    currency: str = "COP"


@dataclass
class KPIInfo:
    """Key Performance Indicator information."""
    name: str = ""
    baseline: Optional[float] = None
    target: Optional[float] = None
    unit: str = ""


@dataclass
class EntityInfo:
    """Entity information."""
    text: str = ""
    type: str = ""
    confidence: float = 1.0


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    id: str
    bytes_hash: str
    text_span: TextSpan
    resolution: ChunkResolution
    text: str
    policy_facets: PolicyFacet = field(default_factory=PolicyFacet)
    time_facets: TimeFacet = field(default_factory=TimeFacet)
    geo_facets: GeoFacet = field(default_factory=GeoFacet)
    confidence: Confidence = field(default_factory=Confidence)
    provenance: Optional[dict[str, Any]] = None
    kpi: Optional[KPIInfo] = None
    budget: Optional[BudgetInfo] = None
    entities: list[EntityInfo] = field(default_factory=list)


@dataclass
class ChunkGraph:
    """Graph of chunks with relationships."""
    chunks: dict[str, Chunk] = field(default_factory=dict)
    edges: list[dict[str, Any]] = field(default_factory=list)
    
    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the graph."""
        self.chunks[chunk.id] = chunk


@dataclass
class PolicyManifest:
    """Manifest of policy elements found in the document."""
    axes: int = 0
    programs: int = 0
    projects: int = 0
    years: list[int] = field(default_factory=list)
    territories: list[str] = field(default_factory=list)
    indicators: int = 0
    budget_rows: int = 0


@dataclass
class ProvenanceMap:
    """Provenance mapping for chunks."""
    page_mappings: dict[str, int] = field(default_factory=dict)
    token_mappings: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for the ingested document."""
    boundary_f1: float = 0.0
    kpi_linkage_rate: float = 0.0
    budget_consistency_score: float = 0.0
    provenance_completeness: float = 0.0
    structural_consistency: float = 0.0
    temporal_robustness: float = 0.0
    chunk_context_coverage: float = 0.0


@dataclass
class IntegrityIndex:
    """Integrity index for verification."""
    blake3_root: str
    chunk_hashes: dict[str, str] = field(default_factory=dict)


@dataclass
class CanonPolicyPackage:
    """Canon Policy Package - the main container for ingested documents."""
    schema_version: str
    policy_manifest: PolicyManifest
    chunk_graph: ChunkGraph
    provenance_map: ProvenanceMap
    quality_metrics: QualityMetrics
    integrity_index: IntegrityIndex
    metadata: dict[str, Any] = field(default_factory=dict)
