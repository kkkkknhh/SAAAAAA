"""Orchestrator utilities with contract validation on import."""

from .canonical_registry import CANONICAL_METHODS

# D2 Activities Design & Coherence - Method Concurrence Enforcement
from .d2_activities_orchestrator import (
    D2ActivitiesOrchestrator,
    D2MethodRegistry,
    D2Question,
    OrchestrationError,
    MethodExecutionError,
    validate_d2_orchestration,
)

from .d2_integration import (
    D2IntegrationHook,
    integrate_d2_validation,
)

try:  # pragma: no cover - executed at import time
    from schema_validator import SchemaValidator

    _validator = SchemaValidator()
    _q_report, _r_report, _c_report = _validator.validate_all()
    if any(report.errors for report in (_q_report, _r_report, _c_report)):
        raise RuntimeError(
            "Data contract validation failed; see schema_validator output for details."
        )
except Exception as exc:  # pragma: no cover - validation failure path
    raise

__all__ = [
    "CANONICAL_METHODS",
    # D2 Method Concurrence
    "D2ActivitiesOrchestrator",
    "D2MethodRegistry",
    "D2Question",
    "OrchestrationError",
    "MethodExecutionError",
    "validate_d2_orchestration",
    "D2IntegrationHook",
    "integrate_d2_validation",
]
