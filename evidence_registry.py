"""Compatibility wrapper for orchestrator evidence registry."""
from saaaaaa.core.orchestrator.evidence_registry import (  # noqa: F401
    EvidenceRecord,
    EvidenceRegistry,
    ProvenanceDAG,
    ProvenanceNode,
    get_global_registry,
)

__all__ = [
    "EvidenceRecord",
    "EvidenceRegistry",
    "ProvenanceDAG",
    "ProvenanceNode",
    "get_global_registry",
]
