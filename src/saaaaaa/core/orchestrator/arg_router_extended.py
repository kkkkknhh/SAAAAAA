"""Extended ArgRouter with Special Routes and Strict Validation.

This module extends the base ArgRouter with:
- 25+ special route handlers for commonly-called methods
- Strict validation (no silent parameter drops)
- **kwargs support for forward compatibility
- Full observability and metrics

Design Principles:
- Explicit route definitions for high-traffic methods
- Fail-fast on missing required arguments
- Fail-fast on unexpected arguments (unless **kwargs present)
- Full traceability of routing decisions
- Zero tolerance for silent parameter drops
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

import structlog

from saaaaaa.core.orchestrator.arg_router import (
    ArgRouter,
    ArgumentValidationError,
    MethodSpec,
    MISSING,
)


logger = structlog.get_logger(__name__)


@dataclass
class RoutingMetrics:
    """Metrics for monitoring routing behavior."""
    
    total_routes: int = 0
    special_routes_hit: int = 0
    default_routes_hit: int = 0
    validation_errors: int = 0
    silent_drops_prevented: int = 0


class ExtendedArgRouter(ArgRouter):
    """
    Extended argument router with special route handling.
    
    Extends base ArgRouter with:
    - 25+ special route definitions
    - Strict validation (no silent drops)
    - **kwargs awareness for forward compatibility
    - Comprehensive metrics
    
    Special Routes (≥25):
    1. _extract_quantitative_claims
    2. _parse_number
    3. _determine_semantic_role
    4. _compile_pattern_registry
    5. _analyze_temporal_coherence
    6. _validate_evidence_chain
    7. _calculate_confidence_score
    8. _extract_indicators
    9. _parse_temporal_reference
    10. _determine_policy_area
    11. _compile_regex_patterns
    12. _analyze_source_reliability
    13. _validate_numerical_consistency
    14. _calculate_bayesian_update
    15. _extract_entities
    16. _parse_citation
    17. _determine_validation_type
    18. _compile_indicator_patterns
    19. _analyze_coherence_score
    20. _validate_threshold_compliance
    21. _calculate_evidence_weight
    22. _extract_temporal_markers
    23. _parse_budget_allocation
    24. _determine_risk_level
    25. _compile_validation_rules
    26. _analyze_stakeholder_impact
    27. _validate_governance_structure
    28. _calculate_alignment_score
    29. _extract_constraint_declarations
    30. _parse_implementation_timeline
    """
    
    def __init__(self, class_registry: Mapping[str, type]) -> None:
        """
        Initialize extended router.
        
        Args:
            class_registry: Mapping of class names to class types
        """
        super().__init__(class_registry)
        self._special_routes = self._build_special_routes()
        self._metrics = RoutingMetrics()
        
        logger.info(
            "extended_arg_router_initialized",
            special_routes=len(self._special_routes),
            classes=len(class_registry),
        )
    
    def _build_special_routes(self) -> dict[str, dict[str, Any]]:
        """
        Build special route definitions for commonly-called methods.
        
        Each route specifies:
        - required_args: List of required parameter names
        - optional_args: List of optional parameter names
        - accepts_kwargs: Whether method accepts **kwargs
        - description: Human-readable description
        
        Returns:
            Dict mapping method names to route specs
        """
        routes = {
            "_extract_quantitative_claims": {
                "required_args": ["content"],
                "optional_args": ["context", "thresholds", "patterns"],
                "accepts_kwargs": True,
                "description": "Extract quantitative claims from content",
            },
            "_parse_number": {
                "required_args": ["text"],
                "optional_args": ["locale", "unit_system"],
                "accepts_kwargs": True,
                "description": "Parse numerical value from text",
            },
            "_determine_semantic_role": {
                "required_args": ["text", "context"],
                "optional_args": ["role_taxonomy", "confidence_threshold"],
                "accepts_kwargs": True,
                "description": "Determine semantic role of text element",
            },
            "_compile_pattern_registry": {
                "required_args": ["patterns"],
                "optional_args": ["category", "flags"],
                "accepts_kwargs": False,
                "description": "Compile patterns into regex registry",
            },
            "_analyze_temporal_coherence": {
                "required_args": ["content"],
                "optional_args": ["temporal_patterns", "baseline_date"],
                "accepts_kwargs": True,
                "description": "Analyze temporal coherence of content",
            },
            "_validate_evidence_chain": {
                "required_args": ["claims", "evidence"],
                "optional_args": ["validation_rules", "min_confidence"],
                "accepts_kwargs": True,
                "description": "Validate evidence chain for claims",
            },
            "_calculate_confidence_score": {
                "required_args": ["evidence"],
                "optional_args": ["prior", "weights"],
                "accepts_kwargs": True,
                "description": "Calculate Bayesian confidence score",
            },
            "_extract_indicators": {
                "required_args": ["content"],
                "optional_args": ["indicator_patterns", "extraction_mode"],
                "accepts_kwargs": True,
                "description": "Extract KPI indicators from content",
            },
            "_parse_temporal_reference": {
                "required_args": ["text"],
                "optional_args": ["reference_date", "format_hints"],
                "accepts_kwargs": True,
                "description": "Parse temporal reference from text",
            },
            "_determine_policy_area": {
                "required_args": ["content"],
                "optional_args": ["taxonomy", "multi_label"],
                "accepts_kwargs": True,
                "description": "Classify content into policy area",
            },
            "_compile_regex_patterns": {
                "required_args": ["pattern_list"],
                "optional_args": ["flags", "validate"],
                "accepts_kwargs": False,
                "description": "Compile list of regex patterns",
            },
            "_analyze_source_reliability": {
                "required_args": ["source"],
                "optional_args": ["source_patterns", "reliability_threshold"],
                "accepts_kwargs": True,
                "description": "Analyze reliability of information source",
            },
            "_validate_numerical_consistency": {
                "required_args": ["numbers"],
                "optional_args": ["tolerance", "consistency_rules"],
                "accepts_kwargs": True,
                "description": "Validate numerical consistency across values",
            },
            "_calculate_bayesian_update": {
                "required_args": ["prior", "likelihood", "evidence"],
                "optional_args": ["normalization"],
                "accepts_kwargs": True,
                "description": "Calculate Bayesian posterior update",
            },
            "_extract_entities": {
                "required_args": ["content"],
                "optional_args": ["entity_types", "confidence_threshold"],
                "accepts_kwargs": True,
                "description": "Extract named entities from content",
            },
            "_parse_citation": {
                "required_args": ["text"],
                "optional_args": ["citation_style", "strict_mode"],
                "accepts_kwargs": True,
                "description": "Parse citation from text",
            },
            "_determine_validation_type": {
                "required_args": ["validation_spec"],
                "optional_args": ["context"],
                "accepts_kwargs": True,
                "description": "Determine type of validation to apply",
            },
            "_compile_indicator_patterns": {
                "required_args": ["indicators"],
                "optional_args": ["category", "weights"],
                "accepts_kwargs": False,
                "description": "Compile indicator patterns for matching",
            },
            "_analyze_coherence_score": {
                "required_args": ["content"],
                "optional_args": ["coherence_patterns", "scoring_mode"],
                "accepts_kwargs": True,
                "description": "Analyze narrative coherence score",
            },
            "_validate_threshold_compliance": {
                "required_args": ["value", "thresholds"],
                "optional_args": ["strict_mode"],
                "accepts_kwargs": True,
                "description": "Validate value against thresholds",
            },
            "_calculate_evidence_weight": {
                "required_args": ["evidence"],
                "optional_args": ["weighting_scheme", "normalization"],
                "accepts_kwargs": True,
                "description": "Calculate evidence weight for scoring",
            },
            "_extract_temporal_markers": {
                "required_args": ["content"],
                "optional_args": ["temporal_patterns", "extraction_depth"],
                "accepts_kwargs": True,
                "description": "Extract temporal markers from content",
            },
            "_parse_budget_allocation": {
                "required_args": ["text"],
                "optional_args": ["currency", "fiscal_year"],
                "accepts_kwargs": True,
                "description": "Parse budget allocation from text",
            },
            "_determine_risk_level": {
                "required_args": ["indicators"],
                "optional_args": ["risk_thresholds", "aggregation_method"],
                "accepts_kwargs": True,
                "description": "Determine risk level from indicators",
            },
            "_compile_validation_rules": {
                "required_args": ["rules"],
                "optional_args": ["rule_format"],
                "accepts_kwargs": False,
                "description": "Compile validation rules for execution",
            },
            "_analyze_stakeholder_impact": {
                "required_args": ["stakeholders", "policy"],
                "optional_args": ["impact_dimensions", "time_horizon"],
                "accepts_kwargs": True,
                "description": "Analyze stakeholder impact of policy",
            },
            "_validate_governance_structure": {
                "required_args": ["structure"],
                "optional_args": ["governance_standards", "strict_mode"],
                "accepts_kwargs": True,
                "description": "Validate governance structure compliance",
            },
            "_calculate_alignment_score": {
                "required_args": ["policy_content", "reference_framework"],
                "optional_args": ["alignment_weights", "scoring_method"],
                "accepts_kwargs": True,
                "description": "Calculate alignment score with framework",
            },
            "_extract_constraint_declarations": {
                "required_args": ["content"],
                "optional_args": ["constraint_types", "extraction_mode"],
                "accepts_kwargs": True,
                "description": "Extract constraint declarations from content",
            },
            "_parse_implementation_timeline": {
                "required_args": ["text"],
                "optional_args": ["reference_date", "granularity"],
                "accepts_kwargs": True,
                "description": "Parse implementation timeline from text",
            },
        }
        
        return routes
    
    def route(
        self,
        class_name: str,
        method_name: str,
        payload: MutableMapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Route method call with special handling and strict validation.
        
        This override:
        1. Checks for special route definitions
        2. Applies strict validation
        3. Prevents silent parameter drops
        4. Tracks metrics
        
        Args:
            class_name: Target class name
            method_name: Target method name
            payload: Method parameters
            
        Returns:
            Tuple of (args, kwargs) for method invocation
            
        Raises:
            ArgumentValidationError: On validation failure
        """
        self._metrics.total_routes += 1
        
        # Check for special route
        if method_name in self._special_routes:
            return self._route_special(class_name, method_name, payload)
        
        # Use default routing with enhanced validation
        return self._route_default_strict(class_name, method_name, payload)
    
    def _route_special(
        self,
        class_name: str,
        method_name: str,
        payload: MutableMapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Route using special route definition.
        
        Args:
            class_name: Target class name
            method_name: Target method name
            payload: Method parameters
            
        Returns:
            Tuple of (args, kwargs)
        """
        self._metrics.special_routes_hit += 1
        
        route_spec = self._special_routes[method_name]
        required_args = set(route_spec["required_args"])
        optional_args = set(route_spec["optional_args"])
        accepts_kwargs = route_spec["accepts_kwargs"]
        
        provided_keys = set(payload.keys())
        
        # Check required arguments
        missing = required_args - provided_keys
        if missing:
            self._metrics.validation_errors += 1
            logger.error(
                "special_route_missing_args",
                class_name=class_name,
                method_name=method_name,
                missing=sorted(missing),
            )
            raise ArgumentValidationError(
                class_name,
                method_name,
                missing=missing,
            )
        
        # Check unexpected arguments
        expected = required_args | optional_args
        unexpected = provided_keys - expected
        
        if unexpected and not accepts_kwargs:
            # Method doesn't accept **kwargs, so unexpected args are an error
            self._metrics.validation_errors += 1
            self._metrics.silent_drops_prevented += 1
            
            logger.error(
                "special_route_unexpected_args",
                class_name=class_name,
                method_name=method_name,
                unexpected=sorted(unexpected),
                accepts_kwargs=accepts_kwargs,
            )
            raise ArgumentValidationError(
                class_name,
                method_name,
                unexpected=unexpected,
            )
        
        # Build kwargs (all parameters go to kwargs for special routes)
        kwargs = dict(payload)
        
        logger.debug(
            "special_route_applied",
            class_name=class_name,
            method_name=method_name,
            params_count=len(kwargs),
        )
        
        return (), kwargs
    
    def _route_default_strict(
        self,
        class_name: str,
        method_name: str,
        payload: MutableMapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """
        Route using default strategy with strict validation.
        
        This prevents silent parameter drops by failing when:
        - Required arguments are missing
        - Unexpected arguments are provided AND method lacks **kwargs
        
        Args:
            class_name: Target class name
            method_name: Target method name
            payload: Method parameters
            
        Returns:
            Tuple of (args, kwargs)
        """
        self._metrics.default_routes_hit += 1
        
        # Use base implementation for inspection
        spec = self.describe(class_name, method_name)
        
        # Strict validation: if unexpected args and no **kwargs, fail
        provided_keys = set(payload.keys())
        accepted = set(spec.accepted_arguments)
        unexpected = provided_keys - accepted
        
        if unexpected and not spec.has_var_keyword:
            # Method doesn't accept **kwargs - unexpected args are errors
            self._metrics.validation_errors += 1
            self._metrics.silent_drops_prevented += 1
            
            logger.error(
                "default_route_unexpected_args_strict",
                class_name=class_name,
                method_name=method_name,
                unexpected=sorted(unexpected),
                has_var_keyword=spec.has_var_keyword,
            )
            raise ArgumentValidationError(
                class_name,
                method_name,
                unexpected=unexpected,
            )
        
        # Delegate to base implementation
        try:
            result = super().route(class_name, method_name, payload)
            logger.debug(
                "default_route_applied",
                class_name=class_name,
                method_name=method_name,
            )
            return result
        except ArgumentValidationError:
            self._metrics.validation_errors += 1
            raise
    
    def get_special_route_coverage(self) -> int:
        """
        Get count of special routes defined.
        
        Returns:
            Number of special routes (target: ≥25)
        """
        return len(self._special_routes)
    
    def get_metrics(self) -> dict[str, Any]:
        """
        Get routing metrics.
        
        Returns:
            Dict with routing statistics
        """
        total = self._metrics.total_routes or 1  # Avoid division by zero
        
        return {
            "total_routes": self._metrics.total_routes,
            "special_routes_hit": self._metrics.special_routes_hit,
            "special_routes_coverage": len(self._special_routes),
            "default_routes_hit": self._metrics.default_routes_hit,
            "validation_errors": self._metrics.validation_errors,
            "silent_drops_prevented": self._metrics.silent_drops_prevented,
            "special_route_hit_rate": self._metrics.special_routes_hit / total,
            "error_rate": self._metrics.validation_errors / total,
        }
    
    def list_special_routes(self) -> list[dict[str, Any]]:
        """
        List all special routes with their specifications.
        
        Returns:
            List of route specifications
        """
        routes = []
        for method_name, spec in sorted(self._special_routes.items()):
            routes.append({
                "method_name": method_name,
                "required_args": spec["required_args"],
                "optional_args": spec["optional_args"],
                "accepts_kwargs": spec["accepts_kwargs"],
                "description": spec["description"],
            })
        return routes
