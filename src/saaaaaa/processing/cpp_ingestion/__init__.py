"""
Canon Policy Package (CPP) Ingestion System - 2025

Deterministic ingestion and policy-aware advanced chunking for Development Plans.

This module implements a complete pipeline for transforming heterogeneous Development Plans
into structured Canon Policy Packages with advanced chunking, provenance tracking, and
quality validation.

Architecture:
    - Rust core for performance-critical operations (hashing, parsing)
    - Python orchestration layer for flexibility
    - Arrow IPC for efficient data serialization
    - Deterministic processing with abort-on-failure
"""

from .models import (
    CanonPolicyPackage,
    Chunk,
    ChunkGraph,
    IngestionOutcome,
    PolicyManifest,
    ProvenanceMap,
)
from .pipeline import CPPIngestionPipeline
from .quality_gates import QualityGates

__version__ = "2025.1.0"

__all__ = [
    "CPPIngestionPipeline",
    "CanonPolicyPackage",
    "Chunk",
    "ChunkGraph",
    "IngestionOutcome",
    "PolicyManifest",
    "ProvenanceMap",
    "QualityGates",
]
