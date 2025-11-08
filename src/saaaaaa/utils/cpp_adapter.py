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
from saaaaaa.core.orchestrator.core import PreprocessedDocument
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
        
        return PreprocessedDocument(
            document_id=document_id,
            raw_text=raw_text,
            sentences=sentences,
            tables=tables,
            metadata=metadata,
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
