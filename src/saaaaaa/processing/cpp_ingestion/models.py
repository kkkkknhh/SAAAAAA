"""
Canon Policy Package (CPP) data models.

This module defines the canonical data structures for CPP ingestion,
bridging Smart Policy Chunks (SPC) to the orchestrator pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ChunkResolution(Enum):
    """Resolution level of a chunk."""
    MICRO = "micro"  # Sentence or clause level
    MESO = "meso"    # Paragraph level
    MACRO = "macro"  # Section level


@dataclass
class TextSpan:
    """Text span in the source document."""
    start: int
    end: int


@dataclass
class ChunkConfidence:
    """Confidence scores for different aspects of chunk extraction."""
    layout: float = 1.0  # Layout analysis confidence
    ocr: float = 1.0     # OCR confidence (if applicable)
    typing: float = 1.0  # Type classification confidence


@dataclass
class BudgetData:
    """Budget information extracted from a chunk."""
    source: str
    use: str
    amount: float
    year: int
    currency: str = "COP"


@dataclass
class KPIData:
    """Key Performance Indicator data."""
    name: str
    target: Optional[str] = None
    baseline: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class ProvenanceData:
    """Provenance information for traceability."""
    source_page: int
    source_section: str
    extraction_method: str
    extraction_confidence: float


@dataclass
class Chunk:
    """
    A semantic chunk from the Canon Policy Package.
    
    This represents a unit of policy content with its metadata,
    semantic type, and relationships to other chunks.
    """
    id: str
    text: str
    text_span: TextSpan
    resolution: ChunkResolution
    chunk_type: str  # diagnostic, activity, indicator, resource, temporal, entity
    confidence: ChunkConfidence
    
    # Optional rich data
    provenance: Optional[ProvenanceData] = None
    kpi: Optional[KPIData] = None
    budget: Optional[BudgetData] = None
    entities: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkGraph:
    """
    Graph of chunks with their relationships.
    
    Represents the semantic structure preserving chunk-to-chunk
    connections (sequential, hierarchical, reference, dependency).
    """
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    edges: List[Dict[str, Any]] = field(default_factory=list)  # {source, target, type, weight}
    
    # Computed graph properties
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyManifest:
    """High-level policy document metadata."""
    axes: List[str] = field(default_factory=list)
    programs: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    territories: List[str] = field(default_factory=list)
    indicators: List[str] = field(default_factory=list)
    budget_rows: int = 0


@dataclass
class QualityMetrics:
    """Quality metrics for the ingestion process."""
    boundary_f1: float = 0.0
    kpi_linkage_rate: float = 0.0
    budget_consistency_score: float = 0.0
    provenance_completeness: float = 0.0
    structural_consistency: float = 0.0
    temporal_robustness: float = 0.0
    chunk_context_coverage: float = 0.0


@dataclass
class IntegrityIndex:
    """Cryptographic integrity information."""
    blake3_root: str
    chunk_hashes: Dict[str, str] = field(default_factory=dict)


@dataclass
class CanonPolicyPackage:
    """
    Canon Policy Package - complete ingestion result.
    
    This is the top-level structure containing all chunks,
    their relationships, and metadata from the ingestion pipeline.
    """
    schema_version: str
    chunk_graph: ChunkGraph
    policy_manifest: Optional[PolicyManifest] = None
    quality_metrics: Optional[QualityMetrics] = None
    integrity_index: Optional[IntegrityIndex] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
