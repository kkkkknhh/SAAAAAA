"""Compatibility facade exposing the refactored orchestrator package."""
from __future__ import annotations

from importlib import import_module
from typing import Any

from saaaaaa.core.orchestrator import (  # noqa: F401
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
    "core",
    "executors",
    "get_global_registry",
    "get_questionnaire_payload",
    "get_questionnaire_provider",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - delegation helper
    if name in globals():
        return globals()[name]
    return getattr(core, name)
