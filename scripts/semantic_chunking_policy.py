"""Compatibility wrapper for the semantic chunking policy module."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.processing.semantic_chunking_policy import (  # noqa: F401, E402
    BayesianEvidenceIntegrator,
    CausalDimension,
    PDMSection,
    PolicyDocumentAnalyzer,
    SemanticChunkingProducer,
    SemanticConfig,
    SemanticProcessor,
)

__all__ = [
    "BayesianEvidenceIntegrator",
    "CausalDimension",
    "PDMSection",
    "PolicyDocumentAnalyzer",
    "SemanticChunkingProducer",
    "SemanticConfig",
    "SemanticProcessor",
]
