"""
Data models for Canon Policy Package (CPP) system.

Defines all data structures for chunks, graphs, manifests, and outcomes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa


class ChunkResolution(Enum):
    """Multi-resolution chunk levels."""
    MICRO = "micro"  # 200-400 tokens
    MESO = "meso"    # 800-1200 tokens
    MACRO = "macro"  # Full section


class PolicyUnitType(Enum):
    """Types of policy units in Development Plans."""
    EJE = "eje"
    PILAR = "pilar"
    PROGRAMA = "programa"
    PROYECTO = "proyecto"
    META = "meta"
    INDICADOR = "indicador"


@dataclass(frozen=True)
class TextSpan:
    """Immutable text span with byte/char offsets."""
    start: int
    end: int
    
    def __len__(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class BoundingBox:
    """Document bounding box coordinates."""
    page_id: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class PolicyFacet:
    """Policy-aware facet for chunks."""
    area: Optional[str] = None
    eje: Optional[str] = None
    programa: Optional[str] = None
    proyecto: Optional[str] = None
    ods: Optional[str] = None  # Sustainable Development Goals


@dataclass(frozen=True)
class TimeFacet:
    """Temporal facet for chunks."""
    from_year: Optional[int] = None
    to_year: Optional[int] = None
    period_bucket: Optional[str] = None
    vigencia: Optional[str] = None


@dataclass(frozen=True)
class GeoFacet:
    """Geographic facet for chunks."""
    level: Optional[str] = None  # municipal, departamental, nacional
    code: Optional[str] = None
    municipio: Optional[str] = None
    departamento: Optional[str] = None


@dataclass(frozen=True)
class KPIData:
    """Key Performance Indicator data."""
    indicator: str
    baseline: Optional[float] = None
    target: Optional[float] = None
    unit: Optional[str] = None


@dataclass(frozen=True)
class BudgetData:
    """Budget allocation data."""
    source: str
    use: str
    amount: float
    year: int
    currency: str = "COP"


@dataclass(frozen=True)
class Entity:
    """Named entity with span."""
    id: str
    type: str
    span: TextSpan


@dataclass(frozen=True)
class NormRef:
    """Normative reference (law, decree, article)."""
    law: str
    article: Optional[str] = None
    anchor_span: Optional[TextSpan] = None


@dataclass(frozen=True)
class ChunkContext:
    """Context information for chunk."""
    parent_title: Optional[str] = None
    upstream_defs: List[str] = field(default_factory=list)
    crossrefs: List[str] = field(default_factory=list)
    local_window_pre: Optional[str] = None
    local_window_post: Optional[str] = None


@dataclass(frozen=True)
class Provenance:
    """Complete provenance tracking for chunk."""
    page_id: int
    bbox: Optional[BoundingBox]
    parser_id: str
    byte_range: Tuple[int, int]
    ocr_data: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Confidence:
    """Confidence scores for chunk extraction."""
    layout: float
    ocr: Optional[float] = None
    typing: float = 1.0


@dataclass
class Chunk:
    """
    Policy-aware chunk with multi-resolution support.
    
    Each chunk represents a semantically coherent unit of text with:
    - Policy facets (area, eje, programa, proyecto)
    - Temporal and geographic facets
    - Optional KPI or budget data
    - Complete provenance tracking
    - Context information for retrieval
    """
    id: str
    bytes_hash: str
    text_span: TextSpan
    resolution: ChunkResolution
    
    # Content
    text: str
    
    # Facets
    policy_facets: PolicyFacet
    time_facets: TimeFacet
    geo_facets: GeoFacet
    
    # Structured data
    kpi: Optional[KPIData] = None
    budget: Optional[BudgetData] = None
    
    # Semantic elements
    entities: List[Entity] = field(default_factory=list)
    norm_refs: List[NormRef] = field(default_factory=list)
    
    # Context and provenance
    context: ChunkContext = field(default_factory=ChunkContext)
    provenance: Optional[Provenance] = None
    confidence: Confidence = field(default_factory=lambda: Confidence(layout=1.0))


class EdgeType(Enum):
    """Types of edges in chunk graph."""
    PRECEDES = "precedes"
    CONTAINS = "contains"
    REFERS_TO = "refers_to"
    DEFINED_BY = "defined_by"
    JUSTIFIES_BUDGET = "justifies_budget"
    SATISFIES_INDICATOR = "satisfies_indicator"


@dataclass
class ChunkGraph:
    """
    Graph structure representing relationships between chunks.
    
    Uses adjacency list representation for efficient traversal.
    """
    chunks: Dict[str, Chunk] = field(default_factory=dict)
    edges: List[Tuple[str, str, EdgeType]] = field(default_factory=list)
    
    def add_chunk(self, chunk: Chunk) -> None:
        """Add chunk to graph."""
        self.chunks[chunk.id] = chunk
    
    def add_edge(self, source_id: str, target_id: str, edge_type: EdgeType) -> None:
        """Add directed edge between chunks."""
        self.edges.append((source_id, target_id, edge_type))
    
    def get_neighbors(self, chunk_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """Get neighboring chunk IDs."""
        return [
            target for source, target, etype in self.edges
            if source == chunk_id and (edge_type is None or etype == edge_type)
        ]


@dataclass
class PolicyManifest:
    """
    Manifest of policy structure extracted from document.
    """
    axes: int = 0
    programs: int = 0
    projects: int = 0
    years: List[int] = field(default_factory=list)
    territories: List[str] = field(default_factory=list)
    indicators: int = 0
    budget_rows: int = 0


@dataclass
class ProvenanceMap:
    """
    Complete provenance mapping from tokens to source.
    
    Stored as Arrow table for efficiency.
    """
    table: Optional[pa.Table] = None
    
    def validate_completeness(self) -> bool:
        """Ensure every token has provenance."""
        if self.table is None:
            return False
        # Check that no nulls in critical columns
        required_cols = ['token_id', 'page_id', 'byte_start', 'byte_end']
        for col in required_cols:
            if col in self.table.column_names:
                if self.table.column(col).null_count > 0:
                    return False
        return True


@dataclass
class IntegrityIndex:
    """
    Integrity verification index with Merkle root.
    """
    blake3_root: str = ""
    tal_chain: List[str] = field(default_factory=list)
    chunk_hashes: Dict[str, str] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for ingestion."""
    boundary_f1: float = 0.0
    kpi_linkage_rate: float = 0.0
    budget_consistency_score: float = 0.0
    provenance_completeness: float = 0.0
    structural_consistency: float = 0.0
    temporal_robustness: float = 0.0
    chunk_context_coverage: float = 0.0


@dataclass
class CanonPolicyPackage:
    """
    Complete Canon Policy Package output.
    
    Represents the final ingested and processed document with:
    - Schema version
    - Policy manifest and graph
    - Chunk graph with relationships
    - Content stream (Arrow IPC)
    - Provenance map (Arrow IPC)
    - Integrity index with Merkle root
    - Quality gates validation
    """
    schema_version: str
    policy_manifest: PolicyManifest
    chunk_graph: ChunkGraph
    content_stream: Optional[pa.Table] = None
    provenance_map: ProvenanceMap = field(default_factory=ProvenanceMap)
    integrity_index: IntegrityIndex = field(default_factory=IntegrityIndex)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)


@dataclass
class IngestionOutcome:
    """
    Final outcome of ingestion pipeline.
    """
    status: str  # "OK" or "ABORT"
    cpp_uri: Optional[str] = None
    policy_manifest: Optional[PolicyManifest] = None
    metrics: Optional[QualityMetrics] = None
    fingerprints: Dict[str, Any] = field(default_factory=dict)
    diagnostics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: Dict[str, Any] = {
            "status": self.status,
            "cpp_uri": self.cpp_uri,
        }
        if self.policy_manifest:
            result["policy_manifest"] = {
                "axes": self.policy_manifest.axes,
                "programs": self.policy_manifest.programs,
                "years": self.policy_manifest.years,
                "territories": self.policy_manifest.territories,
            }
        if self.metrics:
            result["metrics"] = {
                "boundary_f1": self.metrics.boundary_f1,
                "kpi_linkage_rate": self.metrics.kpi_linkage_rate,
                "budget_consistency_score": self.metrics.budget_consistency_score,
            }
        result["fingerprints"] = self.fingerprints
        if self.diagnostics:
            result["diagnostics"] = self.diagnostics
        return result
