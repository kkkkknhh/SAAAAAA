"""Compatibility shim exposing orchestrator facilities from the refactored package."""
from __future__ import annotations
from importlib import import_module
from pathlib import Path
import sys
from typing import Any, Dict

# Add src to path for development environments
_SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if _SRC_PATH.exists():  # pragma: no cover - executed at import time
    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

# Import from unified orchestrator package (if available) or fall back to submodules
try:
    from saaaaaa.core.orchestrator import (
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
except ImportError:
    # Fall back to granular imports
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
    from .provider import get_questionnaire_payload, get_questionnaire_provider

from .factory import build_processor

# Import submodules for backwards compatibility
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

# Register submodule aliases for backwards compatibility
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
        sys.modules[alias] = import_module(target)


def __getattr__(name: str) -> Any:  # pragma: no cover - delegation helper
    """Delegate unknown attributes to the core module."""
    return getattr(core, name)