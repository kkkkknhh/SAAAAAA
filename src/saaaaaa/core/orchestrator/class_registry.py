"""Dynamic class registry for orchestrator method execution."""
from __future__ import annotations

from importlib import import_module
from typing import Dict, Mapping, Type


class ClassRegistryError(RuntimeError):
    """Raised when one or more classes cannot be loaded."""


# Map of orchestrator-facing class names to their import paths.
_CLASS_PATHS: Mapping[str, str] = {
    "IndustrialPolicyProcessor": "saaaaaa.processing.policy_processor.IndustrialPolicyProcessor",
    "PolicyTextProcessor": "saaaaaa.processing.policy_processor.PolicyTextProcessor",
    "BayesianEvidenceScorer": "saaaaaa.processing.policy_processor.BayesianEvidenceScorer",
    "PolicyContradictionDetector": "saaaaaa.analysis.contradiction_deteccion.PolicyContradictionDetector",
    "TemporalLogicVerifier": "saaaaaa.analysis.contradiction_deteccion.TemporalLogicVerifier",
    "BayesianConfidenceCalculator": "saaaaaa.analysis.contradiction_deteccion.BayesianConfidenceCalculator",
    "PDETMunicipalPlanAnalyzer": "saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer",
    "CDAFFramework": "saaaaaa.analysis.dereck_beach.CDAFFramework",
    "CausalExtractor": "saaaaaa.analysis.dereck_beach.CausalExtractor",
    "OperationalizationAuditor": "saaaaaa.analysis.dereck_beach.OperationalizationAuditor",
    "FinancialAuditor": "saaaaaa.analysis.dereck_beach.FinancialAuditor",
    "BayesianMechanismInference": "saaaaaa.analysis.dereck_beach.BayesianMechanismInference",
    "BayesianNumericalAnalyzer": "saaaaaa.processing.embedding_policy.BayesianNumericalAnalyzer",
    "PolicyAnalysisEmbedder": "saaaaaa.processing.embedding_policy.PolicyAnalysisEmbedder",
    "AdvancedSemanticChunker": "saaaaaa.processing.embedding_policy.AdvancedSemanticChunker",
    # SemanticChunker is an alias maintained for backwards compatibility.
    "SemanticChunker": "saaaaaa.processing.embedding_policy.AdvancedSemanticChunker",
    "SemanticAnalyzer": "saaaaaa.analysis.Analyzer_one.SemanticAnalyzer",
    "PerformanceAnalyzer": "saaaaaa.analysis.Analyzer_one.PerformanceAnalyzer",
    "TextMiningEngine": "saaaaaa.analysis.Analyzer_one.TextMiningEngine",
    "MunicipalOntology": "saaaaaa.analysis.Analyzer_one.MunicipalOntology",
    "TeoriaCambio": "saaaaaa.analysis.teoria_cambio.TeoriaCambio",
    "AdvancedDAGValidator": "saaaaaa.analysis.teoria_cambio.AdvancedDAGValidator",
}


def build_class_registry() -> Dict[str, Type[object]]:
    """Return a mapping of class names to loaded types, validating availability."""
    resolved: Dict[str, Type[object]] = {}
    missing: Dict[str, str] = {}
    for name, path in _CLASS_PATHS.items():
        module_name, _, class_name = path.rpartition(".")
        if not module_name:
            missing[name] = path
            continue
        try:
            module = import_module(module_name)
        except ImportError as exc:
            missing[name] = f"{path} (import error: {exc})"
            continue
        try:
            attr = getattr(module, class_name)
        except AttributeError:
            missing[name] = f"{path} (attribute missing)"
        else:
            if not isinstance(attr, type):
                missing[name] = f"{path} (attribute is not a class: {type(attr).__name__})"
            else:
                resolved[name] = attr
    if missing:
        formatted = ", ".join(f"{name}: {reason}" for name, reason in missing.items())
        raise ClassRegistryError(f"Failed to load orchestrator classes: {formatted}")
    return resolved


def get_class_paths() -> Mapping[str, str]:
    """Expose the raw class path mapping for diagnostics."""
    return _CLASS_PATHS
