"""CPP to Orchestrator Adapter.

This adapter converts Canon Policy Package (CPP) documents from the ingestion
pipeline into the orchestrator's PreprocessedDocument format.

Design Principles:
- Preserves complete provenance information
- Orders chunks by text_span.start for deterministic ordering
- Computes provenance_completeness metric
- Provides prescriptive error messages on failure
- Supports micro, meso, and macro chunk resolutions
- Optional dependencies handled gracefully (pyarrow, structlog)
"""

from __future__ import annotations

import logging
from typing import Any

from saaaaaa.compat import try_import
from saaaaaa.core.orchestrator.core import ChunkData, PreprocessedDocument
from saaaaaa.processing.cpp_ingestion.models import (
    CanonPolicyPackage,
    Chunk,
    ChunkResolution,
)

# Optional dependencies - gracefully handle if not available
structlog = try_import("structlog", required=False, hint="Structured logging for CPPAdapter")
pyarrow = try_import("pyarrow", required=False, hint="Arrow serialization for CPPAdapter")

# Use structlog if available, otherwise fallback to standard logging
if structlog is not None:
    logger = structlog.get_logger(__name__)
else:
    logger = logging.getLogger(__name__)


class CPPAdapterError(Exception):
    """Raised when CPP adaptation fails."""
    
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class CPPAdapter:
    """
    Adapter for converting Canon Policy Package to PreprocessedDocument.
    
    Handles:
    - Chunk ordering by text_span.start
    - Provenance completeness calculation
    - Metadata preservation
    - Error handling with prescriptive messages
    """
    
    def __init__(self) -> None:
        """Initialize CPP adapter."""
        self._conversions_count = 0
        if structlog is not None:
            logger.info("cpp_adapter_initialized")
        else:
            logger.info("cpp_adapter_initialized")
    
    def to_preprocessed_document(
        self,
        cpp: CanonPolicyPackage,
        *,
        document_id: str | None = None,
        preserve_chunk_resolution: ChunkResolution | None = None,
    ) -> PreprocessedDocument:
        """
        Convert Canon Policy Package to PreprocessedDocument.
        
        Args:
            cpp: Canon Policy Package from ingestion pipeline
            document_id: Optional document ID override
            preserve_chunk_resolution: If set, only include chunks of this resolution
            
        Returns:
            PreprocessedDocument for orchestrator
            
        Raises:
            CPPAdapterError: If conversion fails with prescriptive error message
        """
        # Validate input
        if not cpp:
            raise CPPAdapterError(
                "Cannot convert None CPP to PreprocessedDocument",
                details={"input_type": type(cpp).__name__}
            )
        
        if not cpp.chunk_graph or not cpp.chunk_graph.chunks:
            raise CPPAdapterError(
                "CPP chunk graph is empty. Ensure ingestion pipeline completed successfully.",
                details={
                    "schema_version": cpp.schema_version,
                    "has_chunk_graph": cpp.chunk_graph is not None,
                    "chunk_count": len(cpp.chunk_graph.chunks) if cpp.chunk_graph else 0,
                }
            )
        
        # Extract chunks
        chunks = list(cpp.chunk_graph.chunks.values())
        
        # Filter by resolution if specified
        if preserve_chunk_resolution:
            chunks = [
                c for c in chunks
                if c.resolution == preserve_chunk_resolution
            ]
            if not chunks:
                raise CPPAdapterError(
                    f"No chunks found with resolution '{preserve_chunk_resolution.value}'",
                    details={
                        "requested_resolution": preserve_chunk_resolution.value,
                        "total_chunks": len(cpp.chunk_graph.chunks),
                        "available_resolutions": list({
                            c.resolution.value for c in cpp.chunk_graph.chunks.values()
                        }),
                    }
                )
        
        # Order chunks by text_span.start for deterministic ordering
        chunks = sorted(chunks, key=lambda c: c.text_span.start)
        
        # Derive document ID
        if document_id is None:
            # Use first chunk's ID prefix or schema version
            document_id = (
                chunks[0].id.split("_")[0] if chunks else "cpp_document"
            )
        
        # Concatenate raw text from ordered chunks
        raw_text = " ".join(chunk.text for chunk in chunks)
        
        # Extract sentences (use chunks as sentence-like units for now)
        sentences = [
            {
                "text": chunk.text,
                "span": {"start": chunk.text_span.start, "end": chunk.text_span.end},
                "chunk_id": chunk.id,
                "resolution": chunk.resolution.value,
            }
            for chunk in chunks
        ]
        
        # Extract tables from chunks with budget data
        tables = []
        for chunk in chunks:
            if chunk.budget:
                tables.append({
                    "source": chunk.budget.source,
                    "use": chunk.budget.use,
                    "amount": chunk.budget.amount,
                    "year": chunk.budget.year,
                    "currency": chunk.budget.currency,
                    "chunk_id": chunk.id,
                })
        
        # Calculate provenance completeness
        provenance_completeness = self._calculate_provenance_completeness(chunks)
        
        # Build metadata
        metadata: dict[str, Any] = {
            "adapter_source": "cpp_adapter.CPPAdapter",
            "schema_version": cpp.schema_version,
            "chunk_count": len(chunks),
            "total_chunks": len(cpp.chunk_graph.chunks),
            "provenance_completeness": provenance_completeness,
            "chunk_resolutions": [c.resolution.value for c in chunks],
        }
        
        # Add policy manifest
        if cpp.policy_manifest:
            metadata["policy_manifest"] = {
                "axes": cpp.policy_manifest.axes,
                "programs": cpp.policy_manifest.programs,
                "projects": cpp.policy_manifest.projects,
                "years": cpp.policy_manifest.years,
                "territories": cpp.policy_manifest.territories,
                "indicators": cpp.policy_manifest.indicators,
                "budget_rows": cpp.policy_manifest.budget_rows,
            }
        
        # Add quality metrics
        if cpp.quality_metrics:
            metadata["quality_metrics"] = {
                "boundary_f1": cpp.quality_metrics.boundary_f1,
                "kpi_linkage_rate": cpp.quality_metrics.kpi_linkage_rate,
                "budget_consistency_score": cpp.quality_metrics.budget_consistency_score,
                "provenance_completeness": cpp.quality_metrics.provenance_completeness,
                "structural_consistency": cpp.quality_metrics.structural_consistency,
                "temporal_robustness": cpp.quality_metrics.temporal_robustness,
                "chunk_context_coverage": cpp.quality_metrics.chunk_context_coverage,
            }
        
        # Add integrity index
        if cpp.integrity_index:
            metadata["integrity_index"] = {
                "blake3_root": cpp.integrity_index.blake3_root,
                "chunk_hashes_count": len(cpp.integrity_index.chunk_hashes),
            }
        
        # Store chunk details for traceability
        metadata["chunks"] = [
            {
                "id": chunk.id,
                "resolution": chunk.resolution.value,
                "text_span": {
                    "start": chunk.text_span.start,
                    "end": chunk.text_span.end,
                },
                "has_provenance": chunk.provenance is not None,
                "has_kpi": chunk.kpi is not None,
                "has_budget": chunk.budget is not None,
                "entity_count": len(chunk.entities),
                "confidence": {
                    "layout": chunk.confidence.layout,
                    "ocr": chunk.confidence.ocr,
                    "typing": chunk.confidence.typing,
                },
            }
            for chunk in chunks
        ]
        
        self._conversions_count += 1
        
        # Log completion (compatible with both structlog and standard logging)
        if structlog is not None:
            logger.info(
                "cpp_to_preprocessed_document_complete",
                document_id=document_id,
                chunk_count=len(chunks),
                provenance_completeness=provenance_completeness,
                conversion_number=self._conversions_count,
            )
        else:
            logger.info(
                f"cpp_to_preprocessed_document_complete: document_id={document_id}, "
                f"chunk_count={len(chunks)}, provenance_completeness={provenance_completeness}, "
                f"conversion_number={self._conversions_count}"
            )
        
        # Validate provenance completeness
        if provenance_completeness < 1.0:
            if structlog is not None:
                logger.warning(
                    "cpp_incomplete_provenance",
                    document_id=document_id,
                    provenance_completeness=provenance_completeness,
                    message="Some chunks are missing provenance information",
                )
            else:
                logger.warning(
                    f"cpp_incomplete_provenance: document_id={document_id}, "
                    f"provenance_completeness={provenance_completeness}, "
                    f"message=Some chunks are missing provenance information"
                )
        
        # NEW: Build chunk objects for SPC exploitation
        chunk_data_list = self._build_chunk_objects(chunks, cpp.chunk_graph.edges, sentences, tables)
        chunk_index = self._build_chunk_index(chunk_data_list, sentences, tables)
        
        # Expose chunk graph for downstream processing
        chunk_graph_dict = {
            "nodes": [
                {
                    "id": idx,
                    "type": cd.chunk_type,
                    "text": cd.text[:100],  # Summary for graph visualization
                    "confidence": cd.confidence,
                }
                for idx, cd in enumerate(chunk_data_list)
            ],
            "edges": cpp.chunk_graph.edges,
        }
        
        return PreprocessedDocument(
            document_id=document_id,
            raw_text=raw_text,
            sentences=sentences,
            tables=tables,
            metadata=metadata,
            # NEW: Chunk fields for SPC exploitation
            chunks=chunk_data_list,
            chunk_index=chunk_index,
            chunk_graph=chunk_graph_dict,
            processing_mode="chunked",  # Enable chunk-aware processing
        )
    
    def _calculate_provenance_completeness(self, chunks: list[Chunk]) -> float:
        """
        Calculate provenance completeness ratio.
        
        Args:
            chunks: List of chunks to analyze
            
        Returns:
            Completeness ratio [0.0, 1.0] where 1.0 means all chunks have provenance
        """
        if not chunks:
            return 0.0
        
        chunks_with_provenance = sum(
            1 for chunk in chunks if chunk.provenance is not None
        )
        
        return chunks_with_provenance / len(chunks)
    
    def _build_chunk_objects(
        self, 
        cpp_chunks: list[Chunk], 
        graph_edges: list[dict[str, Any]],
        sentences: list[dict[str, Any]],
        tables: list[dict[str, Any]]
    ) -> list[ChunkData]:
        """
        Convert CPP chunks to ChunkData objects for orchestrator.
        
        Args:
            cpp_chunks: List of CPP Chunk objects
            graph_edges: List of graph edges from chunk_graph
            sentences: List of sentence dictionaries for mapping
            tables: List of table dictionaries for mapping
            
        Returns:
            List of ChunkData objects preserving structure
        """
        # Build edge mappings
        edge_map_out: dict[str, list[int]] = {}
        edge_map_in: dict[str, list[int]] = {}
        
        # Create chunk ID to index mapping
        chunk_id_to_idx = {chunk.id: idx for idx, chunk in enumerate(cpp_chunks)}
        
        for edge in graph_edges:
            source_id = edge.get("source")
            target_id = edge.get("target")
            
            if source_id in chunk_id_to_idx and target_id in chunk_id_to_idx:
                source_idx = chunk_id_to_idx[source_id]
                target_idx = chunk_id_to_idx[target_id]
                
                edge_map_out.setdefault(source_id, []).append(target_idx)
                edge_map_in.setdefault(target_id, []).append(source_idx)
        
        # Build sentence and table mappings per chunk
        chunk_sentences: dict[int, list[int]] = {idx: [] for idx in range(len(cpp_chunks))}
        chunk_tables: dict[int, list[int]] = {idx: [] for idx in range(len(cpp_chunks))}
        
        # Map sentences to chunks by chunk_id
        for sent_idx, sentence in enumerate(sentences):
            chunk_id = sentence.get("chunk_id")
            if chunk_id is not None:
                # Convert chunk_id to integer index if needed
                if isinstance(chunk_id, str):
                    # Try to find matching chunk by ID
                    for idx, chunk in enumerate(cpp_chunks):
                        if chunk.id == chunk_id:
                            chunk_sentences[idx].append(sent_idx)
                            break
                elif isinstance(chunk_id, int) and chunk_id < len(cpp_chunks):
                    chunk_sentences[chunk_id].append(sent_idx)
        
        # Map tables to chunks by chunk_id
        for table_idx, table in enumerate(tables):
            chunk_id = table.get("chunk_id")
            if chunk_id is not None:
                # Convert chunk_id to integer index if needed
                if isinstance(chunk_id, str):
                    # Try to find matching chunk by ID
                    for idx, chunk in enumerate(cpp_chunks):
                        if chunk.id == chunk_id:
                            chunk_tables[idx].append(table_idx)
                            break
                elif isinstance(chunk_id, int) and chunk_id < len(cpp_chunks):
                    chunk_tables[chunk_id].append(table_idx)
        
        # Build ChunkData objects
        chunk_data_list = []
        for idx, chunk in enumerate(cpp_chunks):
            # Calculate average confidence
            avg_confidence = (
                chunk.confidence.layout + 
                chunk.confidence.ocr + 
                chunk.confidence.typing
            ) / 3.0
            
            chunk_data = ChunkData(
                id=idx,
                text=chunk.text,
                chunk_type=chunk.chunk_type,  # Already normalized to expected types
                sentences=chunk_sentences[idx],  # Populated from sentence mapping
                tables=chunk_tables[idx],       # Populated from table mapping
                start_pos=chunk.text_span.start,
                end_pos=chunk.text_span.end,
                confidence=avg_confidence,
                edges_out=edge_map_out.get(chunk.id, []),
                edges_in=edge_map_in.get(chunk.id, []),
            )
            chunk_data_list.append(chunk_data)
        
        return chunk_data_list
    
    def _build_chunk_index(
        self, 
        chunk_data_list: list[ChunkData],
        sentences: list[dict[str, Any]],
        tables: list[dict[str, Any]]
    ) -> dict[str, int]:
        """
        Build index for fast lookups and populate sentence/table assignments.
        
        Args:
            chunk_data_list: List of ChunkData objects
            sentences: List of sentence dictionaries
            tables: List of table dictionaries
            
        Returns:
            Dict mapping entity IDs to chunk IDs
        """
        chunk_index: dict[str, int] = {}
        
        # Assign sentences to chunks
        for sent_idx, sentence in enumerate(sentences):
            chunk_id = sentence.get("chunk_id")
            if chunk_id:
                chunk_index[f"sent_{sent_idx}"] = int(chunk_id) if isinstance(chunk_id, str) and chunk_id.isdigit() else 0
                # Note: ChunkData is frozen, so we need to work around this
                # For now, we'll skip modifying the frozen dataclass
        
        # Assign tables to chunks
        for table_idx, table in enumerate(tables):
            chunk_id = table.get("chunk_id")
            if chunk_id:
                chunk_index[f"table_{table_idx}"] = int(chunk_id) if isinstance(chunk_id, str) and chunk_id.isdigit() else 0
        
        return chunk_index
    
    def _safe_concat_text(self, chunks: list[Chunk]) -> str:
        """
        Concatenate chunk text with markers for traceability.
        
        Args:
            chunks: List of CPP Chunk objects
            
        Returns:
            Concatenated text with chunk markers
        """
        # For backward compatibility, keep simple concatenation
        # Markers can be added if needed for debugging
        return " ".join(chunk.text for chunk in chunks)
    
    def get_metrics(self) -> dict[str, Any]:
        """
        Get adapter metrics.
        
        Returns:
            Dict with conversion statistics
        """
        return {
            "conversions_count": self._conversions_count,
        }


def adapt_cpp_to_orchestrator(
    cpp: CanonPolicyPackage,
    *,
    document_id: str | None = None,
    preserve_chunk_resolution: ChunkResolution | None = None,
) -> PreprocessedDocument:
    """
    Convenience function to adapt CPP to PreprocessedDocument.
    
    Args:
        cpp: Canon Policy Package
        document_id: Optional document ID
        preserve_chunk_resolution: Optional resolution filter
        
    Returns:
        PreprocessedDocument for orchestrator
    """
    adapter = CPPAdapter()
    return adapter.to_preprocessed_document(
        cpp,
        document_id=document_id,
        preserve_chunk_resolution=preserve_chunk_resolution,
    )
