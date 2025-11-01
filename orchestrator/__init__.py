"""Compatibility package for the refactored orchestrator implementation.

The orchestration engine now lives under ``saaaaaa.core.orchestrator``.  The
original tests import the modules from ``orchestrator`` at the repository root,
so this package re-exports the modern implementation and registers the
submodules in ``sys.modules`` for backwards compatibility.
"""

from __future__ import annotations

import importlib
import sys
from typing import Dict

from saaaaaa.core.orchestrator import (  # noqa: F401,F403
    AbortRequested,
    AbortSignal,
    Evidence,
    EvidenceRecord,
    EvidenceRegistry,
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

_SUBMODULE_ALIASES: Dict[str, str] = {
    "orchestrator.arg_router": "saaaaaa.core.orchestrator.arg_router",
    "orchestrator.class_registry": "saaaaaa.core.orchestrator.class_registry",
    "orchestrator.contract_loader": "saaaaaa.core.orchestrator.contract_loader",
    "orchestrator.core": "saaaaaa.core.orchestrator.core",
    "orchestrator.evidence_registry": "saaaaaa.core.orchestrator.evidence_registry",
    "orchestrator.executors": "saaaaaa.core.orchestrator.executors",
    "orchestrator.factory": "saaaaaa.core.orchestrator.factory",
}

for alias, target in _SUBMODULE_ALIASES.items():
    if alias not in sys.modules:
        sys.modules[alias] = importlib.import_module(target)
