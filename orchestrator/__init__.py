"""Orchestrator utilities with contract validation on import."""

try:
    from .canonical_registry import CANONICAL_METHODS
    _CANONICAL_AVAILABLE = True
except ImportError:
    CANONICAL_METHODS = {}
    _CANONICAL_AVAILABLE = False

from .evidence_registry import (
    EvidenceRecord,
    EvidenceRegistry,
    ProvenanceDAG,
    ProvenanceNode,
    get_global_registry,
)
from .contract_loader import (
    JSONContractLoader,
    LoadError,
    LoadResult,
)

try:
    from .d1_orchestrator import (
        D1Question,
        D1QuestionOrchestrator,
        D1OrchestrationError,
        ExecutionTrace,
        MethodContract,
        OrchestrationResult,
    )
    _D1_AVAILABLE = True
except ImportError:
    _D1_AVAILABLE = False

try:  # pragma: no cover - executed at import time
    from schema_validator import SchemaValidator

    _validator = SchemaValidator()
    _q_report, _r_report, _c_report = _validator.validate_all()
    if any(report.errors for report in (_q_report, _r_report, _c_report)):
        raise RuntimeError(
            "Data contract validation failed; see schema_validator output for details."
        )
except ImportError:  # pragma: no cover - schema_validator not available
    pass  # Schema validation is optional
except Exception as exc:  # pragma: no cover - validation failure path
    raise

__all__ = [
    "CANONICAL_METHODS",
    "EvidenceRecord",
    "EvidenceRegistry",
    "ProvenanceDAG",
    "ProvenanceNode",
    "get_global_registry",
    "JSONContractLoader",
    "LoadError",
    "LoadResult",
]

if _D1_AVAILABLE:
    __all__.extend([
        "D1Question",
        "D1QuestionOrchestrator",
        "D1OrchestrationError",
        "ExecutionTrace",
        "MethodContract",
        "OrchestrationResult",
    ])
