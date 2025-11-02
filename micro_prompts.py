"""Compatibility wrapper for micro-level prompt orchestrations."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.analysis.micro_prompts import (  # noqa: F401, E402
    AuditResult,
    AntiMilagroStressTester,
    AuditResult,
    BayesianPosteriorExplainer,
    CausalChain,
    PosteriorJustification,
    ProportionalityPattern,
    ProvenanceAuditor,
    ProvenanceDAG,
    ProvenanceNode,
    QMCMRecord,
    Signal,
)

__all__ = [
    "AuditResult",
    "AntiMilagroStressTester",
    "BayesianPosteriorExplainer",
    "CausalChain",
    "PosteriorJustification",
    "ProvenanceAuditor",
    "ProvenanceDAG",
    "ProvenanceNode",
    "QMCMRecord",
    "ProportionalityPattern",
    "Signal",
]

