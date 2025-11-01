"""Legacy shim that exposes the refactored orchestrator module layout."""

from __future__ import annotations

import importlib
import sys
from typing import Dict

_core = importlib.import_module("saaaaaa.core.orchestrator")
_core_impl = importlib.import_module("saaaaaa.core.orchestrator.core")
_executors = importlib.import_module("saaaaaa.core.orchestrator.executors")

_PUBLIC_EXPORTS = [
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

for name in _PUBLIC_EXPORTS:
    if hasattr(_core, name):
        globals()[name] = getattr(_core, name)

_ADDITIONAL_EXPORTS = [
    "D1Q1_Executor",
]

for name in _ADDITIONAL_EXPORTS:
    if hasattr(_core_impl, name):
        globals()[name] = getattr(_core_impl, name)
    elif hasattr(_executors, name):
        globals()[name] = getattr(_executors, name)

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
    if alias in sys.modules:
        continue
    try:
        sys.modules[alias] = importlib.import_module(target)
    except ImportError:
        # If a target submodule is not available, skip creating the alias to keep the shim importable.
        continue

__all__ = sorted(set(_PUBLIC_EXPORTS + _ADDITIONAL_EXPORTS))
