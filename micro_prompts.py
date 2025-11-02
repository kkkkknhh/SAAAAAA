"""Compatibility wrapper for micro-level prompt orchestrations."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.analysis.micro_prompts import (  # noqa: F401, E402
    AuditResult,
    AntiMilagroStressTester,
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

