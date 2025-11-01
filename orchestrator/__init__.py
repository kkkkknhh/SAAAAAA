"""Compatibility shim exposing orchestrator facilities from the refactored package."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

_SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if _SRC_PATH.exists():  # pragma: no cover - executed at import time
    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from saaaaaa.core.orchestrator.core import (
    AbortRequested,
    AbortSignal,
    Evidence,
    MethodExecutor,
    MicroQuestionRun,
    Orchestrator,
    PhaseInstrumentation,
    PhaseResult,
    PreprocessedDocument,
    ResourceLimits,
    ScoredMicroQuestion,
)
from saaaaaa.core.orchestrator.evidence_registry import (
    EvidenceRecord,
    EvidenceRegistry,
    ProvenanceDAG,
    ProvenanceNode,
    get_global_registry,
)
from saaaaaa.core.orchestrator.contract_loader import (
    JSONContractLoader,
    LoadError,
    LoadResult,
)

from .factory import build_processor
from .provider import get_questionnaire_payload, get_questionnaire_provider

core = import_module("saaaaaa.core.orchestrator.core")
executors = import_module("saaaaaa.core.orchestrator.executors")

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
    "build_processor",
    "core",
    "executors",
    "get_global_registry",
    "get_questionnaire_payload",
    "get_questionnaire_provider",
]
