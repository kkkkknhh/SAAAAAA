"""Compatibility wrapper for orchestrator module."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.core.orchestrator import (  # noqa: F401, E402
    AbortRequested,
    AbortSignal,
    Evidence,
    EvidenceRecord,
    EvidenceRegistry,
    JSONContractLoader,
    LoadError,
    LoadResult,
    MethodExecutor,
    MicroQuestionRun,
    Orchestrator,
    PhaseInstrumentation,
    PhaseResult,
    PreprocessedDocument,
    ProvenanceDAG,
    ProvenanceNode,
    ResourceLimits,
    ScoredMicroQuestion,
    get_global_registry,
    get_questionnaire_payload,
    get_questionnaire_provider,
)

__all__ = [
    "AbortRequested",
    "AbortSignal",
    "Evidence",
    "EvidenceRecord",
    "EvidenceRegistry",
    "JSONContractLoader",
    "LoadError",
    "LoadResult",
    "MethodExecutor",
    "MicroQuestionRun",
    "Orchestrator",
    "PhaseInstrumentation",
    "PhaseResult",
    "PreprocessedDocument",
    "ProvenanceDAG",
    "ProvenanceNode",
    "ResourceLimits",
    "ScoredMicroQuestion",
    "get_global_registry",
    "get_questionnaire_payload",
    "get_questionnaire_provider",
]
