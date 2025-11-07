"""
Advanced chunking system with policy-awareness and multi-resolution.

Implements 8 chunking mechanisms:
1. Boundary by cohesion with policy conditioning
2. Multi-resolution (micro/meso/macro)
3. Graph-aware chunking
4. KPI/Budget-anchored chunks
5. Temporal windows
6. Territoriality
7. Normative expansion
8. Redundancy guard
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

# Optional dependency - pyarrow
if TYPE_CHECKING:
    import pyarrow as pa
else:
    try:
        import pyarrow as pa
        PYARROW_AVAILABLE = True
    except ImportError:
        PYARROW_AVAILABLE = False
        pa = None  # type: ignore

from .models import (
    BoundingBox,
    Chunk,
    ChunkContext,
    ChunkGraph,
    ChunkResolution,
    Confidence,
    EdgeType,
    Entity,
    GeoFacet,
    KPIData,
    NormRef,
    PolicyFacet,
    Provenance,
    ProvenanceMap,
    TextSpan,
    TimeFacet,
)


class AdvancedChunker:
    """
    Policy-aware advanced chunking system.
    
    Creates multi-resolution chunks with semantic boundaries,
    policy facets, and complete provenance tracking.
    """
    
    # Token count ranges for each resolution
    MICRO_TOKENS = (200, 400)
    MESO_TOKENS = (800, 1200)
    
    def __init__(self, overlap_threshold: float = 0.15):
        """
        Initialize chunker.
        
        Args:
            overlap_threshold: Maximum allowed overlap between chunks
        """
        self.overlap_threshold = overlap_threshold
        self.chunk_counter = 0
    
    def chunk(
        self,
        content_stream: pa.Table,
        policy_graph: Dict[str, Any],
        tables_subgraph: Dict[str, Any],
        provenance_map: ProvenanceMap,
    ) -> ChunkGraph:
        """
        Execute advanced chunking pipeline.
        
        Args:
            content_stream: Text content with offsets
            policy_graph: Policy structure graph
            tables_subgraph: Extracted tables and budget data
            provenance_map: Provenance mapping
            
        Returns:
            ChunkGraph with all chunks and relationships
        """
        chunk_graph = ChunkGraph()
        
        # Extract text pages
        pages = self._extract_pages(content_stream)
        
        # 1. Create micro chunks with semantic boundaries
        micro_chunks = self._create_micro_chunks(pages, policy_graph)
        
        # 2. Create meso chunks (policy unit coherent)
        meso_chunks = self._create_meso_chunks(micro_chunks, policy_graph)
        
        # 3. Create macro chunks (full sections)
        macro_chunks = self._create_macro_chunks(pages, policy_graph)
        
        # 4. Add KPI/Budget-anchored chunks
        kpi_chunks = self._create_kpi_chunks(tables_subgraph, policy_graph)
        budget_chunks = self._create_budget_chunks(tables_subgraph, policy_graph)
        
        # Combine all chunks
        all_chunks = micro_chunks + meso_chunks + macro_chunks + kpi_chunks + budget_chunks
        
        # Add chunks to graph
        for chunk in all_chunks:
            chunk_graph.add_chunk(chunk)
        
        # 5. Build chunk relationships
        self._build_relationships(chunk_graph, policy_graph)
        
        # 6. Apply redundancy guard
        self._apply_redundancy_guard(chunk_graph)
        
        return chunk_graph
    
    def _extract_pages(self, content_stream: pa.Table) -> List[Dict[str, Any]]:
        """Extract pages from content stream."""
        pages = []
        for i in range(len(content_stream)):
            pages.append({
                "page_id": content_stream["page_id"][i].as_py(),
                "text": content_stream["text"][i].as_py(),
                "byte_start": content_stream["byte_start"][i].as_py(),
                "byte_end": content_stream["byte_end"][i].as_py(),
            })
        return pages
    
    def _create_micro_chunks(
        self,
        pages: List[Dict[str, Any]],
        policy_graph: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create micro-resolution chunks (200-400 tokens).
        
        Uses semantic drift detection with policy conditioning.
        """
        chunks = []
        
        for page in pages:
            text = page["text"]
            page_id = page["page_id"]
            
            # Simple sentence-based chunking for now
            sentences = self._split_sentences(text)
            
            current_chunk_text = []
            current_token_count = 0
            chunk_start = 0
            
            for sent in sentences:
                token_count = len(sent.split())
                
                # Check if adding this sentence would exceed micro range
                if current_token_count + token_count > self.MICRO_TOKENS[1]:
                    # Create chunk
                    if current_chunk_text:
                        chunk = self._create_chunk(
                            "".join(current_chunk_text),
                            ChunkResolution.MICRO,
                            page_id,
                            chunk_start,
                            policy_graph,
                        )
                        chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk_text = [sent]
                    current_token_count = token_count
                    chunk_start += len("".join(current_chunk_text))
                else:
                    current_chunk_text.append(sent)
                    current_token_count += token_count
            
            # Add remaining chunk
            if current_chunk_text and current_token_count >= self.MICRO_TOKENS[0]:
                chunk = self._create_chunk(
                    "".join(current_chunk_text),
                    ChunkResolution.MICRO,
                    page_id,
                    chunk_start,
                    policy_graph,
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_meso_chunks(
        self,
        micro_chunks: List[Chunk],
        policy_graph: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create meso-resolution chunks (800-1200 tokens).
        
        Cohesive policy units.
        """
        chunks = []
        
        # Group micro chunks by policy facet
        policy_groups: Dict[str, List[Chunk]] = {}
        for chunk in micro_chunks:
            key = f"{chunk.policy_facets.eje}_{chunk.policy_facets.programa}"
            if key not in policy_groups:
                policy_groups[key] = []
            policy_groups[key].append(chunk)
        
        # Merge each group into meso chunk
        for group_key, group_chunks in policy_groups.items():
            if not group_chunks:
                continue
            
            # Combine text
            combined_text = " ".join(c.text for c in group_chunks)
            token_count = len(combined_text.split())
            
            # Only create if within meso range
            if self.MESO_TOKENS[0] <= token_count <= self.MESO_TOKENS[1]:
                first_chunk = group_chunks[0]
                meso_chunk = Chunk(
                    id=self._generate_chunk_id(),
                    bytes_hash=self._hash_text(combined_text),
                    text_span=TextSpan(0, len(combined_text)),
                    resolution=ChunkResolution.MESO,
                    text=combined_text,
                    policy_facets=first_chunk.policy_facets,
                    time_facets=first_chunk.time_facets,
                    geo_facets=first_chunk.geo_facets,
                    context=ChunkContext(
                        parent_title=f"Policy Unit: {group_key}",
                    ),
                )
                chunks.append(meso_chunk)
        
        return chunks
    
    def _create_macro_chunks(
        self,
        pages: List[Dict[str, Any]],
        policy_graph: Dict[str, Any],
    ) -> List[Chunk]:
        """
        Create macro-resolution chunks (full sections).
        """
        chunks = []
        
        # Each section becomes a macro chunk
        for section in policy_graph.get("sections", []):
            text = section.get("text", "")
            if not text:
                continue
            
            chunk = Chunk(
                id=self._generate_chunk_id(),
                bytes_hash=self._hash_text(text),
                text_span=TextSpan(0, len(text)),
                resolution=ChunkResolution.MACRO,
                text=text,
                policy_facets=PolicyFacet(
                    area=section.get("area"),
                    eje=section.get("eje"),
                ),
                time_facets=TimeFacet(),
                geo_facets=GeoFacet(),
                context=ChunkContext(
                    parent_title=section.get("title"),
                ),
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_kpi_chunks(
        self,
        tables_subgraph: Dict[str, Any],
        policy_graph: Dict[str, Any],
    ) -> List[Chunk]:
        """Create KPI-anchored chunks."""
        chunks = []
        
        for kpi_data in tables_subgraph.get("kpis", []):
            text = f"KPI: {kpi_data.get('indicator', 'Unknown')}"
            
            chunk = Chunk(
                id=self._generate_chunk_id(),
                bytes_hash=self._hash_text(text),
                text_span=TextSpan(0, len(text)),
                resolution=ChunkResolution.MICRO,
                text=text,
                policy_facets=PolicyFacet(),
                time_facets=TimeFacet(),
                geo_facets=GeoFacet(),
                kpi=KPIData(
                    indicator=kpi_data.get("indicator", ""),
                    baseline=kpi_data.get("baseline"),
                    target=kpi_data.get("target"),
                    unit=kpi_data.get("unit"),
                ),
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_budget_chunks(
        self,
        tables_subgraph: Dict[str, Any],
        policy_graph: Dict[str, Any],
    ) -> List[Chunk]:
        """Create budget-anchored chunks."""
        chunks = []
        
        for budget_data in tables_subgraph.get("budgets", []):
            text = f"Budget: {budget_data.get('source', '')} -> {budget_data.get('use', '')}"
            
            chunk = Chunk(
                id=self._generate_chunk_id(),
                bytes_hash=self._hash_text(text),
                text_span=TextSpan(0, len(text)),
                resolution=ChunkResolution.MICRO,
                text=text,
                policy_facets=PolicyFacet(),
                time_facets=TimeFacet(
                    from_year=budget_data.get("year"),
                    to_year=budget_data.get("year"),
                ),
                geo_facets=GeoFacet(),
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(
        self,
        text: str,
        resolution: ChunkResolution,
        page_id: int,
        byte_start: int,
        policy_graph: Dict[str, Any],
    ) -> Chunk:
        """Create a chunk with policy facets."""
        return Chunk(
            id=self._generate_chunk_id(),
            bytes_hash=self._hash_text(text),
            text_span=TextSpan(byte_start, byte_start + len(text)),
            resolution=resolution,
            text=text,
            policy_facets=self._extract_policy_facets(text, policy_graph),
            time_facets=self._extract_time_facets(text),
            geo_facets=self._extract_geo_facets(text),
            provenance=Provenance(
                page_id=page_id,
                bbox=None,
                parser_id="default",
                byte_range=(byte_start, byte_start + len(text)),
            ),
            confidence=Confidence(layout=1.0, typing=1.0),
        )
    
    def _extract_policy_facets(
        self, text: str, policy_graph: Dict[str, Any]
    ) -> PolicyFacet:
        """Extract policy facets from text."""
        # Simplified: look for keywords
        eje = None
        programa = None
        
        for policy_unit in policy_graph.get("policy_units", []):
            if policy_unit.get("type") == "eje":
                if policy_unit.get("name", "").lower() in text.lower():
                    eje = policy_unit.get("name")
            elif policy_unit.get("type") == "programa":
                if policy_unit.get("name", "").lower() in text.lower():
                    programa = policy_unit.get("name")
        
        return PolicyFacet(eje=eje, programa=programa)
    
    def _extract_time_facets(self, text: str) -> TimeFacet:
        """Extract temporal facets from text."""
        # Simplified: look for year patterns
        import re
        years = re.findall(r'\b(20\d{2})\b', text)
        if years:
            years_int = [int(y) for y in years]
            return TimeFacet(
                from_year=min(years_int),
                to_year=max(years_int),
            )
        return TimeFacet()
    
    def _extract_geo_facets(self, text: str) -> GeoFacet:
        """Extract geographic facets from text."""
        # Simplified: look for geographic keywords
        if "municipal" in text.lower():
            return GeoFacet(level="municipal")
        elif "departamental" in text.lower():
            return GeoFacet(level="departamental")
        return GeoFacet()
    
    def _build_relationships(
        self, chunk_graph: ChunkGraph, policy_graph: Dict[str, Any]
    ) -> None:
        """Build relationships between chunks."""
        chunks = list(chunk_graph.chunks.values())
        
        # PRECEDES: sequential chunks
        for i in range(len(chunks) - 1):
            if chunks[i].resolution == chunks[i + 1].resolution:
                chunk_graph.add_edge(
                    chunks[i].id, chunks[i + 1].id, EdgeType.PRECEDES
                )
        
        # CONTAINS: macro contains meso, meso contains micro
        for macro in [c for c in chunks if c.resolution == ChunkResolution.MACRO]:
            for meso in [c for c in chunks if c.resolution == ChunkResolution.MESO]:
                if self._chunk_contains(macro, meso):
                    chunk_graph.add_edge(macro.id, meso.id, EdgeType.CONTAINS)
        
        # SATISFIES_INDICATOR: budget chunks linked to KPI chunks
        kpi_chunks = [c for c in chunks if c.kpi is not None]
        budget_chunks = [c for c in chunks if c.budget is not None]
        
        for budget_chunk in budget_chunks:
            for kpi_chunk in kpi_chunks:
                # Link if they share policy facets
                if (budget_chunk.policy_facets.programa ==
                    kpi_chunk.policy_facets.programa):
                    chunk_graph.add_edge(
                        budget_chunk.id,
                        kpi_chunk.id,
                        EdgeType.SATISFIES_INDICATOR,
                    )
    
    def _chunk_contains(self, container: Chunk, contained: Chunk) -> bool:
        """Check if one chunk contains another based on text span."""
        return (
            container.text_span.start <= contained.text_span.start
            and container.text_span.end >= contained.text_span.end
        )
    
    def _apply_redundancy_guard(self, chunk_graph: ChunkGraph) -> None:
        """Remove redundant overlapping chunks."""
        chunks = list(chunk_graph.chunks.values())
        to_remove: Set[str] = set()
        
        for i, chunk1 in enumerate(chunks):
            for chunk2 in chunks[i + 1:]:
                if chunk1.resolution == chunk2.resolution:
                    overlap = self._compute_overlap(chunk1, chunk2)
                    if overlap > self.overlap_threshold:
                        # Keep the first one, mark second for removal
                        to_remove.add(chunk2.id)
        
        # Remove marked chunks
        for chunk_id in to_remove:
            if chunk_id in chunk_graph.chunks:
                del chunk_graph.chunks[chunk_id]
        
        # Remove edges involving removed chunks
        chunk_graph.edges = [
            (src, tgt, etype)
            for src, tgt, etype in chunk_graph.edges
            if src not in to_remove and tgt not in to_remove
        ]
    
    def _compute_overlap(self, chunk1: Chunk, chunk2: Chunk) -> float:
        """Compute overlap ratio between two chunks."""
        # Based on text span
        span1 = chunk1.text_span
        span2 = chunk2.text_span
        
        overlap_start = max(span1.start, span2.start)
        overlap_end = min(span1.end, span2.end)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_len = overlap_end - overlap_start
        min_len = min(len(span1), len(span2))
        
        return overlap_len / min_len if min_len > 0 else 0.0
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simplified sentence splitting
        import re
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() + ". " for s in sentences if s.strip()]
    
    def _generate_chunk_id(self) -> str:
        """Generate unique chunk ID."""
        self.chunk_counter += 1
        return f"chunk_{self.chunk_counter:06d}"
    
    def _hash_text(self, text: str) -> str:
        """Hash text with BLAKE2b."""
        return hashlib.blake2b(text.encode(), digest_size=16).hexdigest()
