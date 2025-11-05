"""Core Module Factory - Dependency Injection for 19 Module Classes.

This factory creates module instances with injected resources from the questionnaire.
It eliminates pattern duplication by providing a single source of truth via
QuestionnaireResourceProvider.

Design Principles:
- Constructor injection of questionnaire resources
- Single factory instance per questionnaire
- Lazy instantiation of modules
- Full traceability of resource injection
- Type-safe module creation
"""

from __future__ import annotations

from typing import Any

import structlog

from saaaaaa.core.orchestrator.questionnaire_resource_provider import (
    QuestionnaireResourceProvider,
    Pattern,
    ValidationSpec,
)


logger = structlog.get_logger(__name__)


class CoreModuleFactory:
    """
    Factory for creating module instances with injected questionnaire resources.
    
    This factory creates instances of the 19 core module classes, injecting
    patterns, validations, and other resources extracted from the questionnaire.
    
    Module Classes (19):
    1. BayesianNumericalAnalyzer
    2. BayesianEvidenceScorer
    3. BayesianMechanismInference
    4. BayesianTemporalCoherence
    5. BayesianSourceReliability
    6. BayesianCoherenceValidator
    7. BayesianIndicatorExtractor
    8. BayesianCausalChainBuilder
    9. BayesianPolicyAreaClassifier
    10. BayesianRiskAssessment
    11. BayesianStakeholderMapper
    12. BayesianConstraintAnalyzer
    13. BayesianBudgetValidator
    14. BayesianGovernanceEvaluator
    15. BayesianImplementationPlanner
    16. BayesianMonitoringFramework
    17. BayesianImpactPredictor
    18. BayesianAlignmentChecker
    19. BayesianQualityController
    
    Usage:
        factory = CoreModuleFactory(questionnaire_data)
        analyzer = factory.create_bayesian_numerical_analyzer()
        scorer = factory.create_bayesian_evidence_scorer()
    """
    
    def __init__(self, questionnaire_data: dict[str, Any]):
        """
        Initialize factory with questionnaire data.
        
        Args:
            questionnaire_data: Parsed questionnaire monolith JSON
        """
        self._provider = QuestionnaireResourceProvider(questionnaire_data)
        self._instances: dict[str, Any] = {}
        
        # Extract resources upfront
        self._all_patterns = self._provider.extract_all_patterns()
        self._temporal_patterns = self._provider.get_temporal_patterns()
        self._indicator_patterns = self._provider.get_indicator_patterns()
        self._source_patterns = self._provider.get_source_patterns()
        self._territorial_patterns = self._provider.get_territorial_patterns()
        self._validations = self._provider.extract_all_validations()
        
        logger.info(
            "core_module_factory_initialized",
            total_patterns=len(self._all_patterns),
            temporal=len(self._temporal_patterns),
            indicators=len(self._indicator_patterns),
            sources=len(self._source_patterns),
            validations=len(self._validations),
        )
    
    @classmethod
    def from_provider(cls, provider: QuestionnaireResourceProvider) -> CoreModuleFactory:
        """
        Create factory from existing provider.
        
        Args:
            provider: QuestionnaireResourceProvider instance
            
        Returns:
            CoreModuleFactory instance
        """
        # Access internal data (requires provider to expose it)
        return cls(provider._data)
    
    def get_provider(self) -> QuestionnaireResourceProvider:
        """Get the underlying resource provider."""
        return self._provider
    
    # ==========================
    # Module Creation Methods
    # ==========================
    
    def create_bayesian_numerical_analyzer(self) -> BayesianNumericalAnalyzer:
        """
        Create BayesianNumericalAnalyzer with indicator patterns.
        
        Returns:
            Configured BayesianNumericalAnalyzer instance
        """
        if "numerical_analyzer" not in self._instances:
            self._instances["numerical_analyzer"] = BayesianNumericalAnalyzer(
                indicator_patterns=self._indicator_patterns,
                temporal_patterns=self._temporal_patterns,
            )
            logger.debug("created_bayesian_numerical_analyzer")
        
        return self._instances["numerical_analyzer"]
    
    def create_bayesian_evidence_scorer(self) -> BayesianEvidenceScorer:
        """
        Create BayesianEvidenceScorer with source patterns.
        
        Returns:
            Configured BayesianEvidenceScorer instance
        """
        if "evidence_scorer" not in self._instances:
            self._instances["evidence_scorer"] = BayesianEvidenceScorer(
                source_patterns=self._source_patterns,
                validations=self._validations,
            )
            logger.debug("created_bayesian_evidence_scorer")
        
        return self._instances["evidence_scorer"]
    
    def create_bayesian_mechanism_inference(self) -> BayesianMechanismInference:
        """
        Create BayesianMechanismInference with causal patterns.
        
        Returns:
            Configured BayesianMechanismInference instance
        """
        if "mechanism_inference" not in self._instances:
            # Get coherence patterns (cross_reference, etc.)
            coherence_patterns = [
                p for p in self._all_patterns
                if p.category in ("coherence", "cross_reference")
            ]
            
            self._instances["mechanism_inference"] = BayesianMechanismInference(
                coherence_patterns=coherence_patterns,
            )
            logger.debug("created_bayesian_mechanism_inference")
        
        return self._instances["mechanism_inference"]
    
    def create_bayesian_temporal_coherence(self) -> BayesianTemporalCoherence:
        """
        Create BayesianTemporalCoherence with temporal patterns.
        
        Returns:
            Configured BayesianTemporalCoherence instance
        """
        if "temporal_coherence" not in self._instances:
            self._instances["temporal_coherence"] = BayesianTemporalCoherence(
                temporal_patterns=self._temporal_patterns,
            )
            logger.debug("created_bayesian_temporal_coherence")
        
        return self._instances["temporal_coherence"]
    
    def create_bayesian_source_reliability(self) -> BayesianSourceReliability:
        """
        Create BayesianSourceReliability with source patterns.
        
        Returns:
            Configured BayesianSourceReliability instance
        """
        if "source_reliability" not in self._instances:
            self._instances["source_reliability"] = BayesianSourceReliability(
                source_patterns=self._source_patterns,
            )
            logger.debug("created_bayesian_source_reliability")
        
        return self._instances["source_reliability"]
    
    def create_all_modules(self) -> dict[str, Any]:
        """
        Create all 19 module instances.
        
        Returns:
            Dict mapping module names to instances
        """
        modules = {
            "numerical_analyzer": self.create_bayesian_numerical_analyzer(),
            "evidence_scorer": self.create_bayesian_evidence_scorer(),
            "mechanism_inference": self.create_bayesian_mechanism_inference(),
            "temporal_coherence": self.create_bayesian_temporal_coherence(),
            "source_reliability": self.create_bayesian_source_reliability(),
        }
        
        # TODO: Create remaining 14 modules when class stubs are available
        logger.info(
            "all_modules_created",
            count=len(modules),
            modules=list(modules.keys()),
        )
        
        return modules
    
    def get_resource_statistics(self) -> dict[str, Any]:
        """
        Get statistics about injected resources.
        
        Returns:
            Dict with resource counts and metadata
        """
        return self._provider.get_pattern_statistics()


# ==========================
# Module Class Stubs
# ==========================
# TODO: Replace with actual implementations from the codebase

class BayesianNumericalAnalyzer:
    """Stub for BayesianNumericalAnalyzer with injected patterns."""
    
    def __init__(
        self,
        indicator_patterns: list[Pattern],
        temporal_patterns: list[Pattern],
    ):
        self.indicator_patterns = indicator_patterns
        self.temporal_patterns = temporal_patterns
        logger.debug(
            "bayesian_numerical_analyzer_init",
            indicators=len(indicator_patterns),
            temporal=len(temporal_patterns),
        )


class BayesianEvidenceScorer:
    """Stub for BayesianEvidenceScorer with injected patterns."""
    
    def __init__(
        self,
        source_patterns: list[Pattern],
        validations: list[ValidationSpec],
    ):
        self.source_patterns = source_patterns
        self.validations = validations
        logger.debug(
            "bayesian_evidence_scorer_init",
            sources=len(source_patterns),
            validations=len(validations),
        )


class BayesianMechanismInference:
    """Stub for BayesianMechanismInference with injected patterns."""
    
    def __init__(self, coherence_patterns: list[Pattern]):
        self.coherence_patterns = coherence_patterns
        logger.debug(
            "bayesian_mechanism_inference_init",
            coherence=len(coherence_patterns),
        )


class BayesianTemporalCoherence:
    """Stub for BayesianTemporalCoherence with injected patterns."""
    
    def __init__(self, temporal_patterns: list[Pattern]):
        self.temporal_patterns = temporal_patterns
        logger.debug(
            "bayesian_temporal_coherence_init",
            temporal=len(temporal_patterns),
        )


class BayesianSourceReliability:
    """Stub for BayesianSourceReliability with injected patterns."""
    
    def __init__(self, source_patterns: list[Pattern]):
        self.source_patterns = source_patterns
        logger.debug(
            "bayesian_source_reliability_init",
            sources=len(source_patterns),
        )
