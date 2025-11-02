"""Compatibility wrapper for Derek Beach CDAF framework."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.analysis.dereck_beach import (  # noqa: F401, E402
    AdaptivePriorCalculator,
    AuditResult,
    BayesFactorTable,
    BayesianCounterfactualAuditor,
    BayesianMechanismInference,
    BeachEvidentialTest,
    CDAFConfigSchema,
    CDAFException,
    CDAFFramework,
    CDAFProcessingError,
    CDAFValidationError,
    CausalExtractor,
    CausalInferenceSetup,
    CausalLink,
    ConfigLoader,
    DerekBeachProducer,
    EntityActivity,
    FinancialAuditor,
    GoalClassification,
    HierarchicalGenerativeModel,
    MechanismPartExtractor,
    MetaNode,
    OperationalizationAuditor,
    PDFProcessor,
    ReportingEngine,
)

__all__ = [
    "AdaptivePriorCalculator",
    "AuditResult",
    "BayesFactorTable",
    "BayesianCounterfactualAuditor",
    "BayesianMechanismInference",
    "BeachEvidentialTest",
    "CDAFConfigSchema",
    "CDAFException",
    "CDAFFramework",
    "CDAFProcessingError",
    "CDAFValidationError",
    "CausalExtractor",
    "CausalInferenceSetup",
    "CausalLink",
    "ConfigLoader",
    "DerekBeachProducer",
    "EntityActivity",
    "FinancialAuditor",
    "GoalClassification",
    "HierarchicalGenerativeModel",
    "MechanismPartExtractor",
    "MetaNode",
    "OperationalizationAuditor",
    "PDFProcessor",
    "ReportingEngine",
]
