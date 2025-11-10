"""
CPP (Canon Policy Package) Ingestion Models

This module provides data models for the Canon Policy Package format.
These models are used by the CPP adapter to convert ingested documents
into the orchestrator's PreprocessedDocument format.
"""

from .models import (
    CanonPolicyPackage,
    Chunk,
    ChunkGraph,
    ChunkResolution,
    Confidence,
    PolicyFacet,
    PolicyManifest,
    ProvenanceMap,
    QualityMetrics,
    TextSpan,
    TimeFacet,
    GeoFacet,
    IntegrityIndex,
    BudgetInfo,
    KPIInfo,
    EntityInfo,
)

__all__ = [
    "CanonPolicyPackage",
    "Chunk",
    "ChunkGraph",
    "ChunkResolution",
    "Confidence",
    "PolicyFacet",
    "PolicyManifest",
    "ProvenanceMap",
    "QualityMetrics",
    "TextSpan",
    "TimeFacet",
    "GeoFacet",
    "IntegrityIndex",
    "BudgetInfo",
    "KPIInfo",
    "EntityInfo",
]
