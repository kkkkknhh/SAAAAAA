# report_assembly.py - COMPLETE IMPLEMENTATION
# coding=utf-8
"""
Report Assembly - MICRO/MESO/MACRO Multi-Level Reporting System
================================================================

Fully integrated with:
- questionnaire_parser.py (question specs and scoring)
- module_adapters_COMPLETE_MERGED.py (9 adapters, 413 methods)
- rubric_scoring.json (scoring modalities)
- FARFAN_3.0_UPDATED_QUESTIONNAIRE.yaml (execution chains)
- macro_prompts.py (5 macro-level analysis prompts)

This module generates comprehensive reports at three hierarchical levels:
1. MICRO: Individual question answers with evidence
2. MESO: Cluster-level aggregations by policy areas
3. MACRO: Overall plan convergence with Decálogo framework

Author: Integration Team
Version: 3.1.0 - Complete with Macro Prompts Integration
Python: 3.10+
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import statistics
import re
from datetime import datetime

logger = logging.getLogger(__name__)

# Import macro prompts orchestrator
try:
    from macro_prompts import MacroPromptsOrchestrator
    MACRO_PROMPTS_AVAILABLE = True
except ImportError:
    logger.warning("macro_prompts module not available - macro analyses will be limited")
    MACRO_PROMPTS_AVAILABLE = False


# ============================================================================
# DATA STRUCTURES FOR MULTI-LEVEL REPORTING
# ============================================================================

@dataclass
class MicroLevelAnswer:
    """
    MICRO level: Individual question answer with full traceability
    
    Maps to single question (e.g., P1-D1-Q1) with complete evidence chain
    """
    question_id: str  # P#-D#-Q# format (e.g., "P1-D1-Q1")
    qualitative_note: str  # EXCELENTE/BUENO/ACEPTABLE/INSUFICIENTE
    quantitative_score: float  # 0.0-3.0 (question level scoring)
    evidence: List[str]  # Text extracts from plan document
    explanation: str  # 150-300 words, doctoral-level analysis
    confidence: float  # 0.0-1.0 confidence score
    
    # Detailed scoring breakdown
    scoring_modality: str  # TYPE_A, TYPE_B, TYPE_C, etc.
    elements_found: Dict[str, bool]  # Which expected elements were detected
    search_pattern_matches: Dict[str, Any]  # Pattern matching results
    
    # Module execution details
    modules_executed: List[str]  # Adapter names that executed
    module_results: Dict[str, Any]  # Results from each adapter
    execution_time: float  # Total execution time in seconds
    execution_chain: List[Dict[str, str]] = field(default_factory=list)  # Complete execution traceability
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MesoLevelCluster:
    """
    MESO level: Cluster aggregation across policy areas

    Aggregates related questions into thematic clusters for mid-level analysis
    """
    cluster_name: str  # CLUSTER_1, CLUSTER_2, etc.
    cluster_description: str  # Human-readable description
    policy_areas: List[str]  # Policy areas included (e.g., [P1, P2, P3])
    avg_score: float  # 0-100 percentage score
    dimension_scores: Dict[str, float]  # D1: 75.0, D2: 65.0, etc. (percentages)
    strengths: List[str]  # Identified strengths
    weaknesses: List[str]  # Identified weaknesses
    recommendations: List[str]  # Strategic recommendations
    question_coverage: float  # Percentage of questions answered
    total_questions: int  # Total questions in cluster
    answered_questions: int  # Number of questions answered
    policy_area_scores: Dict[str, float] = field(default_factory=dict)
    evidence_quality: Dict[str, float] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroLevelConvergence:
    """
    MACRO level: Overall convergence with Decálogo framework
    
    Provides executive-level assessment of entire plan
    """
    overall_score: float  # 0-100 percentage
    convergence_by_dimension: Dict[str, float]  # D1-D6 scores (percentages)
    convergence_by_policy_area: Dict[str, float]  # P1-P10 scores (percentages)
    gap_analysis: Dict[str, Any]  # Identified gaps and missing elements
    agenda_alignment: float  # 0.0-1.0 alignment with Decálogo agenda
    critical_gaps: List[str]  # Most critical gaps requiring attention
    strategic_recommendations: List[str]  # High-level recommendations
    plan_classification: str  # Overall classification (EXCELENTE/BUENO/etc.)
    evidence_synthesis: Dict[str, Any] = field(default_factory=dict)
    implementation_roadmap: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistical summaries
    score_distribution: Dict[str, int] = field(default_factory=dict)
    confidence_metrics: Dict[str, float] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# REPORT ASSEMBLER - MAIN CLASS
# ============================================================================

class ReportAssembler:
    """
    Assembles comprehensive reports at three hierarchical levels
    
    Integrates with:
    - ModuleAdapterRegistry (9 adapters)
    - QuestionnaireParser (300 questions)
    - ExecutionChoreographer (orchestration)
    """

    def __init__(
            self,
            dimension_descriptions: Optional[Dict[str, str]] = None,
            cluster_weights: Optional[Dict[str, float]] = None,
            cluster_policy_weights: Optional[Dict[str, Dict[str, float]]] = None,
            causal_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize report assembler with rubric definitions
        """
        # Rubric levels for overall/dimension scoring (percentage thresholds)
        self.rubric_levels = {
            "EXCELENTE": (85, 100),
            "BUENO": (70, 84),
            "SATISFACTORIO": (55, 69),
            "INSUFICIENTE": (40, 54),
            "DEFICIENTE": (0, 39)
        }

        # Question-level rubric (0-3 scale)
        self.question_rubric = {
            "EXCELENTE": (2.55, 3.00),  # 85% of 3.0
            "BUENO": (2.10, 2.54),      # 70% of 3.0
            "ACEPTABLE": (1.65, 2.09),  # 55% of 3.0
            "INSUFICIENTE": (0.00, 1.64)  # Below 55%
        }

        # Dimension descriptions (D1-D6)
        self.dimension_descriptions = dimension_descriptions or {
            "D1": "Diagnóstico y Recursos - Líneas base, magnitud del problema, recursos y capacidades",
            "D2": "Diseño de Intervención - Actividades, mecanismos causales, secuencias temporales",
            "D3": "Productos y Outputs - Entregables, verificación, presupuesto",
            "D4": "Resultados y Outcomes - Indicadores de resultado, causalidad mediano plazo",
            "D5": "Impactos y Efectos de Largo Plazo - Transformación estructural, sostenibilidad",
            "D6": "Teoría de Cambio y Coherencia Causal - Coherencia causal global, auditoría completa"
        }

        self.cluster_weights = cluster_weights or {}
        self.cluster_policy_weights = cluster_policy_weights or {}
        self.causal_thresholds = causal_thresholds or {
            "default": 0.6,
            "D4": 0.65,
            "D5": 0.7,
            "D6": 0.75
        }

        # Initialize macro prompts orchestrator if available
        if MACRO_PROMPTS_AVAILABLE:
            self.macro_prompts = MacroPromptsOrchestrator()
            logger.info("ReportAssembler initialized with rubric definitions and macro prompts")
        else:
            self.macro_prompts = None
            logger.info("ReportAssembler initialized with rubric definitions (macro prompts unavailable)")

    # ========================================================================
    # MICRO LEVEL - Question-by-Question Analysis
    # ========================================================================

    def generate_micro_answer(
            self,
            question_spec,  # QuestionSpec from questionnaire_parser
            execution_results: Dict[str, Any],  # Results from ExecutionChoreographer
            plan_text: str
    ) -> MicroLevelAnswer:
        """
        Generate MICRO-level answer for a single question
        
        Args:
            question_spec: Question specification with scoring modality
            execution_results: Dict of adapter execution results
            plan_text: Original plan document text
            
        Returns:
            MicroLevelAnswer with complete analysis
        """
        logger.info(f"Generating MICRO answer for {question_spec.canonical_id}")

        start_time = datetime.now()

        # Step 1: Apply scoring modality to calculate score
        score, elements_found, pattern_matches, causal_correction = self._apply_scoring_modality(
            question_spec,
            execution_results,
            plan_text
        )

        # Step 2: Map quantitative score to qualitative level
        qualitative = self._score_to_qualitative_question(score)

        # Step 3: Extract evidence excerpts from plan
        evidence_excerpts = self._extract_evidence_excerpts(
            question_spec,
            execution_results,
            elements_found,
            plan_text
        )

        # Step 4: Calculate overall confidence
        confidence = self._calculate_confidence(
            execution_results,
            elements_found,
            pattern_matches
        )

        # Step 5: Generate doctoral-level explanation
        explanation = self._generate_explanation(
            question_spec,
            score,
            qualitative,
            elements_found,
            execution_results,
            evidence_excerpts
        )

        # Step 6: Collect module execution details
        modules_executed = list(execution_results.keys())
        module_results = {
            module: {
                "status": result.get("status", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "data_summary": self._summarize_module_data(result)
            }
            for module, result in execution_results.items()
        }

        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Build execution chain for traceability
        execution_chain = []
        if hasattr(question_spec, 'execution_chain'):
            execution_chain = question_spec.execution_chain
        else:
            # Fallback: build from modules_executed
            for module in modules_executed:
                execution_chain.append({
                    "module": module,
                    "status": module_results[module]["status"],
                    "confidence": module_results[module]["confidence"]
                })

        return MicroLevelAnswer(
            question_id=question_spec.canonical_id,
            qualitative_note=qualitative,
            quantitative_score=score,
            evidence=evidence_excerpts,
            explanation=explanation,
            confidence=confidence,
            scoring_modality=question_spec.scoring_modality,
            elements_found=elements_found,
            search_pattern_matches=pattern_matches,
            modules_executed=modules_executed,
            module_results=module_results,
            execution_time=execution_time,
            execution_chain=execution_chain,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "policy_area": question_spec.policy_area,
                "dimension": question_spec.dimension,
                "question_number": getattr(question_spec, 'question_no', 0),
                "cluster_id": getattr(question_spec, 'cluster_id', ''),
                "cluster_name": getattr(question_spec, 'cluster_name', ''),
                "causal_correction": causal_correction
            }
        )

    def _apply_scoring_modality(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Tuple[float, Dict[str, bool], Dict[str, Any], Dict[str, Any]]:
        """
        Apply scoring modality (TYPE_A, TYPE_B, etc.) to calculate question score

        Returns:
            (score, elements_found, pattern_matches, causal_correction_metadata)
        """
        modality = question_spec.scoring_modality
        logger.debug(f"Applying scoring modality: {modality}")

        elements_found = {}
        pattern_matches = {}

        if modality == "TYPE_A":
            # Binary presence/absence of key elements
            score, elements_found = self._score_type_a(
                question_spec, execution_results, plan_text
            )
        
        elif modality == "TYPE_B":
            # Weighted sum of multiple elements
            score, elements_found = self._score_type_b(
                question_spec, execution_results, plan_text
            )
        
        elif modality == "TYPE_C":
            # Quality assessment with rubric
            score, elements_found = self._score_type_c(
                question_spec, execution_results, plan_text
            )
        
        elif modality == "TYPE_D":
            # Numerical threshold matching
            score, elements_found = self._score_type_d(
                question_spec, execution_results, plan_text
            )
        
        elif modality == "TYPE_E":
            # Logical rule-based scoring
            score, elements_found = self._score_type_e(
                question_spec, execution_results, plan_text
            )
        
        elif modality == "TYPE_F":
            # Semantic analysis with similarity matching
            score, elements_found = self._score_type_f(
                question_spec, execution_results, plan_text
            )
        
        else:
            # Default: confidence-weighted average
            score, elements_found = self._score_default(
                question_spec, execution_results
            )

        # Extract pattern matches for traceability
        pattern_matches = self._extract_pattern_matches(
            question_spec, execution_results, plan_text
        )

        score, correction_metadata = self._apply_causal_correction(
            question_spec,
            score,
            execution_results
        )

        return score, elements_found, pattern_matches, correction_metadata

    def _apply_causal_correction(
            self,
            question_spec,
            base_score: float,
            execution_results: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """Adjust base score using causal coherence signals when required."""

        dimension = getattr(question_spec, 'dimension', '') or "default"
        threshold = self.causal_thresholds.get(
            dimension,
            self.causal_thresholds.get('default', 0.6)
        )

        signals = self._extract_causal_signals(execution_results)
        flags = self._extract_causal_flags(execution_results)

        numeric_values = [value for _, value in signals if isinstance(value, (int, float))]
        coherence_estimate = statistics.mean(numeric_values) if numeric_values else None

        penalty_ratio = 0.0
        applied = False

        if coherence_estimate is not None and threshold:
            if coherence_estimate < threshold:
                gap = threshold - coherence_estimate
                penalty_ratio = min(0.4, max(0.0, gap / max(threshold, 1e-6) * 0.5))
                applied = penalty_ratio > 0

        flag_penalty = 0.0
        if any(not flag_value for _, flag_value in flags):
            flag_penalty = 0.1
            applied = True

        total_penalty = min(0.5, penalty_ratio + flag_penalty)
        adjusted_score = base_score * (1 - total_penalty)
        adjusted_score = max(0.0, min(3.0, adjusted_score))

        correction_metadata = {
            "applied": applied,
            "dimension": dimension,
            "base_score": round(base_score, 4),
            "adjusted_score": round(adjusted_score, 4),
            "coherence_estimate": round(coherence_estimate, 4) if coherence_estimate is not None else None,
            "threshold": threshold,
            "penalty_ratio": round(total_penalty, 4),
            "signal_count": len(signals),
            "flags": [
                {"key": key, "value": flag_value}
                for key, flag_value in flags
            ],
            "signals": [
                {"key": key, "value": round(value, 4)}
                for key, value in signals[:5]
            ]
        }

        return adjusted_score, correction_metadata

    def _extract_causal_signals(
            self,
            execution_results: Dict[str, Any]
    ) -> List[Tuple[str, float]]:
        """Collect numeric causal signals present in execution results."""

        signal_keys = {
            "causal_coherence",
            "coherence_score",
            "causal_confidence",
            "causal_score",
            "causal_strength",
            "causal_density",
            "mission_causal_score",
            "causal_alignment"
        }

        signals: List[Tuple[str, float]] = []

        def traverse(obj: Any):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in signal_keys and isinstance(value, (int, float)):
                        signals.append((key, float(value)))
                    elif isinstance(value, (dict, list)):
                        traverse(value)
            elif isinstance(obj, list):
                for item in obj:
                    traverse(item)

        for module_result in execution_results.values():
            traverse(module_result)

        return signals

    def _extract_causal_flags(
            self,
            execution_results: Dict[str, Any]
    ) -> List[Tuple[str, bool]]:
        """Identify boolean causal flags signalling structural issues."""

        flag_keys = {
            "meets_threshold",
            "causal_chain_valid",
            "causal_gap_detected",
            "causal_alert"
        }

        flags: List[Tuple[str, bool]] = []

        def traverse(obj: Any):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key in flag_keys and isinstance(value, bool):
                        flags.append((key, value))
                    elif isinstance(value, (dict, list)):
                        traverse(value)
            elif isinstance(obj, list):
                for item in obj:
                    traverse(item)

        for module_result in execution_results.values():
            traverse(module_result)

        return flags

    def _score_type_a(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Tuple[float, Dict[str, bool]]:
        """
        TYPE_A: Binary presence/absence scoring (from rubric_scoring.json)
        
        FORMULA: Score = (elements_found / 4) * 3
        - Expected elements: 4 (as defined in rubric TYPE_A)
        - Each element worth: 0.75 points
        - Max score: 3.0 points
        
        CONVERSION TABLE (from rubric):
        0 elements → 0.00 points
        1 element  → 0.75 points
        2 elements → 1.50 points
        3 elements → 2.25 points
        4 elements → 3.00 points
        
        PRECONDITIONS:
        - question_spec must have expected_elements list
        - execution_results must be non-empty dict
        - plan_text must be non-empty string
        """
        # Precondition checks (raise exceptions instead of assert)
        if not hasattr(question_spec, 'expected_elements'):
            raise AttributeError("question_spec must have expected_elements attribute for TYPE_A scoring")
        if not execution_results:
            raise ValueError("execution_results cannot be empty for TYPE_A scoring")
        if not (plan_text and isinstance(plan_text, str)):
            raise TypeError("plan_text must be non-empty string for TYPE_A scoring")
        
        required_elements = question_spec.expected_elements or []
        elements_found = {}

        for element in required_elements:
            # Check in execution results (adapter outputs)
            found = any(
                element.lower() in str(result.get("data", "")).lower()
                for result in execution_results.values()
            )
            
            # Fallback: check in raw plan text
            if not found:
                found = element.lower() in plan_text.lower()
            
            elements_found[element] = found

        if not required_elements:
            return 0.0, elements_found

        # Apply TYPE_A formula from rubric_scoring.json
        found_count = sum(elements_found.values())
        score = (found_count / len(required_elements)) * 3.0

        return score, elements_found

    def _score_type_b(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Tuple[float, Dict[str, bool]]:
        """
        TYPE_B: Weighted sum of multiple elements
        
        Different elements have different weights
        """
        elements_weights = question_spec.element_weights or {}
        elements_found = {}

        total_weight = sum(elements_weights.values())
        if total_weight == 0:
            return self._score_type_a(question_spec, execution_results, plan_text)

        weighted_score = 0.0

        for element, weight in elements_weights.items():
            found = any(
                element.lower() in str(result.get("data", "")).lower()
                for result in execution_results.values()
            )
            
            if not found:
                found = element.lower() in plan_text.lower()
            
            elements_found[element] = found
            
            if found:
                weighted_score += weight

        score = (weighted_score / total_weight) * 3.0

        return score, elements_found

    def _score_type_c(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Tuple[float, Dict[str, bool]]:
        """
        TYPE_C: Quality assessment with rubric
        
        Uses confidence scores from adapters to assess quality
        """
        elements_found = {}
        confidence_scores = []

        # Collect confidence scores from all execution results
        for module, result in execution_results.items():
            conf = result.get("confidence", 0.0)
            confidence_scores.append(conf)
            elements_found[module] = conf > 0.5

        if not confidence_scores:
            return 0.0, elements_found

        # Use average confidence as quality indicator
        avg_confidence = statistics.mean(confidence_scores)
        score = avg_confidence * 3.0

        return score, elements_found

    def _score_type_d(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Tuple[float, Dict[str, bool]]:
        """
        TYPE_D: Numerical threshold matching
        
        Checks if numerical values meet specified thresholds
        """
        thresholds = question_spec.numerical_thresholds or {}
        elements_found = {}

        met_count = 0
        total_thresholds = len(thresholds)

        for metric, threshold in thresholds.items():
            # Try to extract numerical value from results
            value = self._extract_numerical_value(metric, execution_results)
            
            if value is not None:
                meets_threshold = value >= threshold
                elements_found[metric] = meets_threshold
                if meets_threshold:
                    met_count += 1
            else:
                elements_found[metric] = False

        if total_thresholds == 0:
            return 0.0, elements_found

        score = (met_count / total_thresholds) * 3.0

        return score, elements_found

    def _score_type_e(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Tuple[float, Dict[str, bool]]:
        """
        TYPE_E: Logical rule-based scoring (from rubric_scoring.json)
        
        Applies if-then-else logic based on custom rules
        Uses custom logic defined per question
        
        PRECONDITIONS:
        - question_spec must have validation_rules with logical conditions
        - execution_results must be non-empty dict
        """
        if not execution_results:
            raise ValueError("execution_results cannot be empty for TYPE_E scoring")
        
        elements_found = {}
        score = 0.0
        
        # Get validation rules (custom logic per question)
        validation_rules = getattr(question_spec, 'validation_rules', {})
        
        # Check if conditional logic is defined
        if "logical_conditions" in validation_rules:
            conditions = validation_rules["logical_conditions"]
            
            # Evaluate each condition
            for condition_name, condition_def in conditions.items():
                if_clause = condition_def.get("if", {})
                then_value = condition_def.get("then", 0.0)
                else_value = condition_def.get("else", 0.0)
                
                # Evaluate if clause
                condition_met = self._evaluate_condition(if_clause, execution_results, plan_text)
                elements_found[condition_name] = condition_met
                
                # Apply then/else scoring
                if condition_met:
                    score += then_value
                else:
                    score += else_value
        else:
            # Fallback: use element presence with custom weights
            custom_logic = validation_rules.get("custom_scoring", {})
            total_weight = 0
            weighted_score = 0
            
            for element, weight in custom_logic.items():
                found = any(
                    element.lower() in str(result.get("data", "")).lower()
                    for result in execution_results.values()
                )
                if not found:
                    found = element.lower() in plan_text.lower()
                
                elements_found[element] = found
                total_weight += weight
                if found:
                    weighted_score += weight
            
            if total_weight > 0:
                score = (weighted_score / total_weight) * 3.0
        
        # Ensure score is within bounds
        score = max(0.0, min(3.0, score))
        
        return score, elements_found

    def _score_type_f(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Tuple[float, Dict[str, bool]]:
        """
        TYPE_F: Semantic analysis with similarity matching (from rubric_scoring.json)
        
        Uses semantic matching with cosine similarity
        Applies thresholds based on coverage ratio
        
        FORMULA: f(coverage_ratio) with thresholds
        - similarity_threshold: 0.6 (default from rubric)
        
        PRECONDITIONS:
        - question_spec must have expected_elements or search_patterns
        - execution_results must contain semantic analysis results
        """
        elements_found = {}
        
        # Get search patterns or expected elements for semantic matching
        search_patterns = getattr(question_spec, 'search_patterns', {})
        expected_elements = getattr(question_spec, 'expected_elements', [])
        
        if not search_patterns and not expected_elements:
            # Fallback to TYPE_A if no semantic patterns defined
            return self._score_type_a(question_spec, execution_results, plan_text)
        
        # Calculate semantic coverage
        total_patterns = len(search_patterns) if search_patterns else len(expected_elements)
        matched_patterns = 0
        
        # Check execution results for semantic matches
        for module, result in execution_results.items():
            similarity = result.get("semantic_similarity", 0.0)
            coverage = result.get("coverage_ratio", 0.0)
            
            # Apply similarity threshold (0.6 from rubric)
            if similarity >= 0.6 or coverage >= 0.6:
                matched_patterns += 1
                elements_found[module] = True
            else:
                elements_found[module] = False
        
        # If no semantic data in results, do keyword matching
        if matched_patterns == 0:
            patterns_to_check = list(search_patterns.values()) if search_patterns else expected_elements
            
            for pattern in patterns_to_check[:10]:  # Limit to avoid excessive processing
                pattern_str = str(pattern).lower()
                found = pattern_str in plan_text.lower()
                elements_found[pattern_str[:50]] = found
                if found:
                    matched_patterns += 1
            
            total_patterns = len(patterns_to_check)
        
        # Calculate score based on coverage ratio
        if total_patterns == 0:
            coverage_ratio = 0.0
        else:
            coverage_ratio = matched_patterns / total_patterns
        
        # Apply thresholds from rubric (similar to TYPE_A but with semantic weighting)
        if coverage_ratio >= 0.9:
            score = 3.0
        elif coverage_ratio >= 0.75:
            score = 2.5
        elif coverage_ratio >= 0.6:
            score = 2.0
        elif coverage_ratio >= 0.4:
            score = 1.5
        elif coverage_ratio >= 0.25:
            score = 1.0
        else:
            score = coverage_ratio * 3.0
        
        return score, elements_found

    def _evaluate_condition(
            self,
            condition: Dict[str, Any],
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> bool:
        """
        Evaluate a logical condition for TYPE_E scoring
        
        Supports:
        - "contains": check if text contains pattern
        - "threshold": check if value meets threshold
        - "all_of": all subconditions must be true
        - "any_of": at least one subcondition must be true
        """
        condition_type = condition.get("type", "contains")
        
        if condition_type == "contains":
            pattern = condition.get("pattern", "")
            # Check in execution results
            for result in execution_results.values():
                if pattern.lower() in str(result.get("data", "")).lower():
                    return True
            # Check in plan text
            return pattern.lower() in plan_text.lower()
        
        elif condition_type == "threshold":
            metric = condition.get("metric", "")
            threshold = condition.get("value", 0)
            operator = condition.get("operator", ">=")
            
            # Extract value from results
            for result in execution_results.values():
                data = result.get("data", {})
                if isinstance(data, dict) and metric in data:
                    value = data[metric]
                    if operator == ">=":
                        return value >= threshold
                    elif operator == ">":
                        return value > threshold
                    elif operator == "<=":
                        return value <= threshold
                    elif operator == "<":
                        return value < threshold
                    elif operator == "==":
                        return value == threshold
            return False
        
        elif condition_type == "all_of":
            subconditions = condition.get("conditions", [])
            return all(
                self._evaluate_condition(sub, execution_results, plan_text)
                for sub in subconditions
            )
        
        elif condition_type == "any_of":
            subconditions = condition.get("conditions", [])
            return any(
                self._evaluate_condition(sub, execution_results, plan_text)
                for sub in subconditions
            )
        
        return False

    def _score_default(
            self,
            question_spec,
            execution_results: Dict[str, Any]
    ) -> Tuple[float, Dict[str, bool]]:
        """
        Default scoring: Confidence-weighted average
        """
        elements_found = {}
        confidence_scores = []

        for module, result in execution_results.items():
            conf = result.get("confidence", 0.0)
            confidence_scores.append(conf)
            elements_found[module] = conf > 0.6

        if not confidence_scores:
            return 0.0, elements_found

        avg_confidence = statistics.mean(confidence_scores)
        score = avg_confidence * 3.0

        return score, elements_found

    def _extract_numerical_value(
            self,
            metric: str,
            execution_results: Dict[str, Any]
    ) -> Optional[float]:
        """Extract numerical value for a metric from execution results"""
        for result in execution_results.values():
            data = result.get("data", {})
            if isinstance(data, dict) and metric in data:
                try:
                    return float(data[metric])
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_pattern_matches(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Dict[str, Any]:
        """Extract pattern matches from execution results"""
        pattern_matches = {}

        for module, result in execution_results.items():
            if "evidence" in result:
                pattern_matches[module] = result["evidence"]

        return pattern_matches

    def _score_to_qualitative_question(self, score: float) -> str:
        """
        Map quantitative score (0-3) to qualitative level
        
        Uses question_rubric thresholds with >= comparisons from high to low.
        This ensures highest scores are matched first, preventing incorrect
        assignment to lower rubrics.
        """
        # Check from highest to lowest to ensure correct assignment
        # Score thresholds: 85% (2.55), 70% (2.10), 55% (1.65) of 3.0
        if score >= 2.55:  # 85% of 3.0
            return "EXCELENTE"
        elif score >= 2.10:  # 70% of 3.0
            return "BUENO"
        elif score >= 1.65:  # 55% of 3.0
            return "ACEPTABLE"
        else:  # Below 55%
            return "INSUFICIENTE"

    def _extract_evidence_excerpts(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            elements_found: Dict[str, bool],
            plan_text: str,
            max_excerpts: int = 5
    ) -> List[str]:
        """
        Extract relevant text excerpts as evidence
        
        Returns up to max_excerpts text snippets from plan
        """
        excerpts = []

        # Get excerpts from execution results
        for module, result in execution_results.items():
            evidence = result.get("evidence", [])
            if isinstance(evidence, list):
                for item in evidence[:2]:  # Max 2 per module
                    if isinstance(item, dict) and "text" in item:
                        excerpts.append(item["text"])
                    elif isinstance(item, str):
                        excerpts.append(item)

        # If not enough, search for found elements in plan text
        if len(excerpts) < max_excerpts:
            for element, found in elements_found.items():
                if found and len(excerpts) < max_excerpts:
                    excerpt = self._find_context_around(element, plan_text)
                    if excerpt:
                        excerpts.append(excerpt)

        return excerpts[:max_excerpts]

    def _find_context_around(
            self,
            keyword: str,
            text: str,
            context_chars: int = 200
    ) -> Optional[str]:
        """Find text excerpt around keyword"""
        keyword_lower = keyword.lower()
        text_lower = text.lower()

        pos = text_lower.find(keyword_lower)
        if pos == -1:
            return None

        start = max(0, pos - context_chars)
        end = min(len(text), pos + len(keyword) + context_chars)

        excerpt = text[start:end].strip()
        
        # Clean up
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(text):
            excerpt = excerpt + "..."

        return excerpt

    def _calculate_confidence(
            self,
            execution_results: Dict[str, Any],
            elements_found: Dict[str, bool],
            pattern_matches: Dict[str, Any]
    ) -> float:
        """
        Calculate overall confidence score
        
        Combines:
        - Adapter confidence scores
        - Element detection rate
        - Pattern match quality
        """
        confidence_scores = []

        # Adapter confidences
        for result in execution_results.values():
            conf = result.get("confidence", 0.0)
            confidence_scores.append(conf)

        # Element detection rate
        if elements_found:
            detection_rate = sum(elements_found.values()) / len(elements_found)
            confidence_scores.append(detection_rate)

        # Pattern match quality
        if pattern_matches:
            match_quality = len(pattern_matches) / max(len(execution_results), 1)
            confidence_scores.append(match_quality)

        if not confidence_scores:
            return 0.0

        return statistics.mean(confidence_scores)

    def _generate_explanation(
            self,
            question_spec,
            score: float,
            qualitative: str,
            elements_found: Dict[str, bool],
            execution_results: Dict[str, Any],
            evidence: List[str]
    ) -> str:
        """
        Generate doctoral-level explanation (150-300 words)
        
        Explains:
        - What was evaluated
        - What was found
        - Why it received this score
        - Implications for policy implementation
        """
        explanation_parts = []

        # Introduction
        explanation_parts.append(
            f"Esta pregunta evalúa {question_spec.question_text.lower()}. "
        )

        # Findings
        found_count = sum(elements_found.values())
        total_elements = len(elements_found)
        
        if found_count == total_elements:
            explanation_parts.append(
                f"El análisis identificó todos los elementos esperados ({found_count}/{total_elements}), "
                f"resultando en una calificación de {qualitative}. "
            )
        elif found_count > 0:
            explanation_parts.append(
                f"Se identificaron {found_count} de {total_elements} elementos esperados, "
                f"resultando en una calificación de {qualitative}. "
            )
        else:
            explanation_parts.append(
                f"No se identificaron los elementos esperados en el plan, "
                f"resultando en una calificación de {qualitative}. "
            )

        # Evidence synthesis
        if evidence:
            explanation_parts.append(
                f"La evidencia documental incluye referencias a: {', '.join(evidence[:3])}. "
            )

        # Module execution summary
        high_conf_modules = [
            module for module, result in execution_results.items()
            if result.get("confidence", 0) > 0.7
        ]
        
        if high_conf_modules:
            explanation_parts.append(
                f"Los módulos de análisis {', '.join(high_conf_modules[:3])} "
                f"reportaron alta confianza en sus hallazgos. "
            )

        # Implications
        if qualitative == "EXCELENTE":
            explanation_parts.append(
                "Este resultado indica una excelente alineación con los estándares del Decálogo, "
                "sugiriendo un diseño robusto en esta dimensión."
            )
        elif qualitative == "BUENO":
            explanation_parts.append(
                "Este resultado indica buena alineación con los estándares, "
                "con oportunidades de fortalecimiento identificadas."
            )
        elif qualitative == "ACEPTABLE":
            explanation_parts.append(
                "Este resultado indica cumplimiento básico, "
                "requiriendo mejoras significativas para optimizar la implementación."
            )
        else:
            explanation_parts.append(
                "Este resultado indica deficiencias críticas que deben ser atendidas "
                "para asegurar la viabilidad y efectividad de la intervención propuesta."
            )

        return " ".join(explanation_parts)

    def _summarize_module_data(self, result: Dict[str, Any]) -> str:
        """Create brief summary of module result data"""
        data = result.get("data", {})
        if isinstance(data, dict):
            keys = list(data.keys())[:3]
            return f"Keys: {', '.join(keys)}"
        elif isinstance(data, list):
            return f"List with {len(data)} items"
        else:
            return str(data)[:50]

    # ========================================================================
    # MESO LEVEL - Cluster Aggregation
    # ========================================================================

    def generate_meso_cluster(
            self,
            cluster_name: str,
            cluster_description: str,
            micro_answers: List[MicroLevelAnswer],
            cluster_definition: Dict[str, Any]
    ) -> MesoLevelCluster:
        """
        Generate MESO-level cluster aggregation
        
        Args:
            cluster_name: Cluster identifier (e.g., "CLUSTER_1")
            cluster_description: Human-readable description
            micro_answers: List of MICRO answers in cluster
            cluster_definition: Cluster configuration
            
        Returns:
            MesoLevelCluster with aggregated analysis
        """
        logger.info(f"Generating MESO cluster: {cluster_name}")

        # Extract policy areas and dimensions
        policy_areas = sorted(set(
            answer.metadata.get("policy_area", "")
            for answer in micro_answers
        ))

        # Calculate scores
        scores = [answer.quantitative_score for answer in micro_answers]
        avg_score_raw = statistics.mean(scores) if scores else 0.0
        avg_score_pct = (avg_score_raw / 3.0) * 100  # Convert to percentage

        policy_area_scores = self._calculate_policy_area_scores(micro_answers)
        avg_score_pct, weighting_trace = self._apply_policy_weighting(
            cluster_name,
            avg_score_pct,
            policy_area_scores,
            cluster_definition
        )

        # Calculate dimension scores
        dimension_scores = self._calculate_dimension_scores(micro_answers)

        # Calculate coverage
        total_questions = cluster_definition.get("total_questions", len(micro_answers))
        answered_questions = len(micro_answers)
        question_coverage = (answered_questions / total_questions * 100) if total_questions > 0 else 0

        # Identify strengths and weaknesses
        strengths = self._identify_strengths(micro_answers, dimension_scores)
        weaknesses = self._identify_weaknesses(micro_answers, dimension_scores)

        # Generate recommendations
        recommendations = self._generate_cluster_recommendations(
            cluster_name,
            micro_answers,
            strengths,
            weaknesses
        )

        # Calculate evidence quality
        evidence_quality = self._assess_evidence_quality(micro_answers)

        return MesoLevelCluster(
            cluster_name=cluster_name,
            cluster_description=cluster_description,
            policy_areas=policy_areas,
            avg_score=avg_score_pct,
            policy_area_scores=policy_area_scores,
            dimension_scores=dimension_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            question_coverage=question_coverage,
            total_questions=total_questions,
            answered_questions=answered_questions,
            evidence_quality=evidence_quality,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "score_distribution": self._calculate_score_distribution(micro_answers),
                "weighting_trace": weighting_trace,
                "policy_weights": cluster_definition.get("policy_weights", {}),
                "macro_weight": cluster_definition.get("macro_weight")
            }
        )

    def _calculate_dimension_scores(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, float]:
        """Calculate average scores by dimension (as percentages)"""
        dimension_scores = defaultdict(list)

        for answer in micro_answers:
            dimension = answer.metadata.get("dimension", "")
            if dimension:
                # Convert 0-3 score to percentage
                pct_score = (answer.quantitative_score / 3.0) * 100
                dimension_scores[dimension].append(pct_score)

        return {
            dim: statistics.mean(scores)
            for dim, scores in dimension_scores.items()
        }

    def _calculate_policy_area_scores(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, float]:
        """Calculate average scores per policy area in percentage scale."""

        policy_scores = defaultdict(list)

        for answer in micro_answers:
            policy_area = answer.metadata.get("policy_area", "")
            if not policy_area:
                continue
            pct_score = (answer.quantitative_score / 3.0) * 100
            policy_scores[policy_area].append(pct_score)

        return {
            policy_area: statistics.mean(scores)
            for policy_area, scores in policy_scores.items()
        }

    def _apply_policy_weighting(
            self,
            cluster_id: str,
            fallback_score: float,
            policy_area_scores: Dict[str, float],
            cluster_definition: Dict[str, Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Apply rubric-defined weights to policy area contributions."""

        weights = self.cluster_policy_weights.get(cluster_id) or cluster_definition.get("policy_weights", {})
        if not weights:
            return fallback_score, []

        weighted_sum = 0.0
        used_weight = 0.0
        trace: List[Dict[str, Any]] = []

        for policy_area, weight in weights.items():
            score = policy_area_scores.get(policy_area)
            if score is None:
                continue
            contribution = score * weight
            weighted_sum += contribution
            used_weight += weight
            trace.append({
                "policy_area": policy_area,
                "weight": weight,
                "score": score,
                "contribution": contribution
            })

        if used_weight == 0:
            return fallback_score, trace

        weighted_score = weighted_sum / used_weight
        return weighted_score, trace

    def _identify_strengths(
            self,
            micro_answers: List[MicroLevelAnswer],
            dimension_scores: Dict[str, float]
    ) -> List[str]:
        """Identify cluster strengths"""
        strengths = []

        # High-scoring dimensions
        for dim, score in dimension_scores.items():
            if score >= 85:
                strengths.append(
                    f"{dim}: Excelente desempeño ({score:.1f}%) - "
                    f"{self.dimension_descriptions.get(dim, dim)}"
                )
            elif score >= 70:
                strengths.append(
                    f"{dim}: Buen desempeño ({score:.1f}%) - "
                    f"{self.dimension_descriptions.get(dim, dim)}"
                )

        # High-confidence answers
        high_conf_count = sum(
            1 for answer in micro_answers
            if answer.confidence >= 0.8
        )
        if high_conf_count >= len(micro_answers) * 0.7:
            strengths.append(
                f"Alta confianza en los hallazgos ({high_conf_count}/{len(micro_answers)} respuestas)"
            )

        # Comprehensive evidence
        avg_evidence = statistics.mean(
            len(answer.evidence) for answer in micro_answers
        )
        if avg_evidence >= 3:
            strengths.append(
                f"Evidencia documental sólida (promedio {avg_evidence:.1f} extractos por pregunta)"
            )

        return strengths[:5]  # Top 5 strengths

    def _identify_weaknesses(
            self,
            micro_answers: List[MicroLevelAnswer],
            dimension_scores: Dict[str, float]
    ) -> List[str]:
        """Identify cluster weaknesses"""
        weaknesses = []

        # Low-scoring dimensions
        for dim, score in dimension_scores.items():
            if score < 40:
                weaknesses.append(
                    f"{dim}: Deficiencias críticas ({score:.1f}%) - "
                    f"{self.dimension_descriptions.get(dim, dim)}"
                )
            elif score < 55:
                weaknesses.append(
                    f"{dim}: Desempeño insuficiente ({score:.1f}%) - "
                    f"{self.dimension_descriptions.get(dim, dim)}"
                )

        # Low confidence
        low_conf_count = sum(
            1 for answer in micro_answers
            if answer.confidence < 0.5
        )
        if low_conf_count >= len(micro_answers) * 0.3:
            weaknesses.append(
                f"Baja confianza en hallazgos ({low_conf_count}/{len(micro_answers)} respuestas)"
            )

        # Missing elements
        avg_elements_found = statistics.mean(
            sum(answer.elements_found.values()) / len(answer.elements_found)
            if answer.elements_found else 0
            for answer in micro_answers
        )
        if avg_elements_found < 0.5:
            weaknesses.append(
                f"Elementos esperados insuficientes (promedio {avg_elements_found:.1%} detectados)"
            )

        return weaknesses[:5]  # Top 5 weaknesses

    def _generate_cluster_recommendations(
            self,
            cluster_name: str,
            micro_answers: List[MicroLevelAnswer],
            strengths: List[str],
            weaknesses: List[str]
    ) -> List[str]:
        """Generate strategic recommendations for cluster"""
        recommendations = []

        # Address weaknesses
        if len(weaknesses) >= 3:
            recommendations.append(
                f"Priorizar el fortalecimiento de las dimensiones con menor desempeño "
                f"identificadas en este cluster"
            )

        # Leverage strengths
        if len(strengths) >= 3:
            recommendations.append(
                f"Utilizar las fortalezas identificadas como modelo para mejorar otras áreas"
            )

        # Evidence gaps
        low_evidence = [
            answer for answer in micro_answers
            if len(answer.evidence) < 2
        ]
        if len(low_evidence) >= len(micro_answers) * 0.3:
            recommendations.append(
                "Fortalecer la documentación y evidencia en las áreas con menor respaldo"
            )

        # Module-specific
        module_usage = defaultdict(int)
        for answer in micro_answers:
            for module in answer.modules_executed:
                module_usage[module] += 1

        underused_modules = [
            module for module, count in module_usage.items()
            if count < len(micro_answers) * 0.5
        ]
        if underused_modules:
            recommendations.append(
                f"Considerar análisis adicional utilizando módulos subutilizados: "
                f"{', '.join(underused_modules[:3])}"
            )

        return recommendations[:5]  # Top 5 recommendations

    def _assess_evidence_quality(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, float]:
        """Assess overall evidence quality"""
        return {
            "avg_excerpts_per_question": statistics.mean(
                len(answer.evidence) for answer in micro_answers
            ),
            "avg_confidence": statistics.mean(
                answer.confidence for answer in micro_answers
            ),
            "pct_high_confidence": sum(
                1 for answer in micro_answers if answer.confidence >= 0.7
            ) / len(micro_answers) * 100 if micro_answers else 0
        }

    def _calculate_score_distribution(
            self,
            micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, int]:
        """Calculate distribution of qualitative scores"""
        distribution = defaultdict(int)
        for answer in micro_answers:
            distribution[answer.qualitative_note] += 1
        return dict(distribution)

    # ========================================================================
    # MACRO LEVEL - Overall Convergence
    # ========================================================================

    def generate_macro_convergence(
            self,
            all_micro_answers: List[MicroLevelAnswer],
            all_meso_clusters: List[MesoLevelCluster],
            plan_metadata: Dict[str, Any]
    ) -> MacroLevelConvergence:
        """
        Generate MACRO-level convergence analysis
        
        Args:
            all_micro_answers: All MICRO answers across all questions
            all_meso_clusters: All MESO clusters
            plan_metadata: Plan document metadata
            
        Returns:
            MacroLevelConvergence with executive summary
        """
        logger.info("Generating MACRO convergence analysis")

        # Calculate overall score
        all_scores = [answer.quantitative_score for answer in all_micro_answers]
        base_micro_average = (statistics.mean(all_scores) / 3.0) * 100 if all_scores else 0.0

        overall_score, weighting_trace, missing_clusters = self._aggregate_macro_score(
            all_meso_clusters,
            base_micro_average
        )

        # Calculate convergence by dimension (D1-D6)
        convergence_by_dimension = self._calculate_dimension_convergence(
            all_micro_answers
        )

        # Calculate convergence by policy area (P1-P10)
        convergence_by_policy_area = self._calculate_policy_area_convergence(
            all_micro_answers
        )

        # Perform gap analysis
        gap_analysis = self._perform_gap_analysis(
            all_micro_answers,
            convergence_by_dimension,
            convergence_by_policy_area
        )

        # Calculate agenda alignment
        agenda_alignment = self._calculate_agenda_alignment(
            all_micro_answers,
            plan_metadata
        )

        # Identify critical gaps
        critical_gaps = self._identify_critical_gaps(
            gap_analysis,
            convergence_by_dimension
        )

        # Generate strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(
            overall_score,
            convergence_by_dimension,
            critical_gaps,
            all_meso_clusters
        )

        # Classify plan overall
        plan_classification = self._classify_plan(overall_score)

        # Synthesize evidence
        evidence_synthesis = self._synthesize_evidence(all_micro_answers)

        # Generate implementation roadmap
        implementation_roadmap = self._generate_implementation_roadmap(
            critical_gaps,
            strategic_recommendations
        )

        # Calculate score distribution
        score_distribution = self._calculate_overall_distribution(all_micro_answers)

        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(all_micro_answers)

        # Apply macro prompts if available
        macro_prompts_results = None
        if self.macro_prompts:
            macro_prompts_results = self._apply_macro_prompts(
                all_micro_answers,
                all_meso_clusters,
                convergence_by_dimension,
                convergence_by_policy_area,
                missing_clusters,
                critical_gaps,
                confidence_metrics
            )

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "total_questions_analyzed": len(all_micro_answers),
            "total_clusters": len(all_meso_clusters),
            "plan_name": plan_metadata.get("name", "Unknown"),
            "base_micro_average": base_micro_average,
            "weighting_trace": weighting_trace,
            "cluster_weights": self.cluster_weights,
            "missing_clusters": missing_clusters,
            "macro_prompts_results": macro_prompts_results
        }

        return MacroLevelConvergence(
            overall_score=overall_score,
            convergence_by_dimension=convergence_by_dimension,
            convergence_by_policy_area=convergence_by_policy_area,
            gap_analysis=gap_analysis,
            agenda_alignment=agenda_alignment,
            critical_gaps=critical_gaps,
            strategic_recommendations=strategic_recommendations,
            plan_classification=plan_classification,
            evidence_synthesis=evidence_synthesis,
            implementation_roadmap=implementation_roadmap,
            score_distribution=score_distribution,
            confidence_metrics=confidence_metrics,
            metadata=metadata
        )

    def _aggregate_macro_score(
            self,
            meso_clusters: List[MesoLevelCluster],
            fallback_score: float
    ) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """Aggregate macro score using rubric-defined cluster weights."""

        if not meso_clusters:
            return fallback_score, [], []

        if not self.cluster_weights:
            return fallback_score, [], []

        weighted_sum = 0.0
        used_weight = 0.0
        trace: List[Dict[str, Any]] = []
        available_clusters = {cluster.cluster_name: cluster for cluster in meso_clusters}

        for cluster_id, weight in self.cluster_weights.items():
            cluster = available_clusters.get(cluster_id)
            if not cluster:
                trace.append({
                    "cluster_id": cluster_id,
                    "weight": weight,
                    "cluster_score": None,
                    "contribution": 0.0
                })
                continue

            cluster_score = cluster.avg_score
            contribution = cluster_score * weight
            weighted_sum += contribution
            used_weight += weight
            trace.append({
                "cluster_id": cluster_id,
                "weight": weight,
                "cluster_score": cluster_score,
                "contribution": contribution
            })

        missing_clusters = [
            entry["cluster_id"]
            for entry in trace
            if entry.get("cluster_score") is None
        ]

        if used_weight == 0:
            return fallback_score, trace, missing_clusters

        macro_score = weighted_sum / used_weight
        macro_score = max(0.0, min(100.0, macro_score))

        return macro_score, trace, missing_clusters

    def _calculate_dimension_convergence(
            self,
            all_micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, float]:
        """Calculate convergence score by dimension (percentages)"""
        dimension_scores = defaultdict(list)

        for answer in all_micro_answers:
            dimension = answer.metadata.get("dimension", "")
            if dimension:
                pct_score = (answer.quantitative_score / 3.0) * 100
                dimension_scores[dimension].append(pct_score)

        return {
            dim: statistics.mean(scores)
            for dim, scores in dimension_scores.items()
        }

    def _calculate_policy_area_convergence(
            self,
            all_micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, float]:
        """Calculate convergence score by policy area (percentages)"""
        policy_scores = defaultdict(list)

        for answer in all_micro_answers:
            policy_area = answer.metadata.get("policy_area", "")
            if policy_area:
                pct_score = (answer.quantitative_score / 3.0) * 100
                policy_scores[policy_area].append(pct_score)

        return {
            policy: statistics.mean(scores)
            for policy, scores in policy_scores.items()
        }

    def _perform_gap_analysis(
            self,
            all_micro_answers: List[MicroLevelAnswer],
            dim_convergence: Dict[str, float],
            policy_convergence: Dict[str, float]
    ) -> Dict[str, Any]:
        """Perform comprehensive gap analysis"""
        gaps = {
            "dimensional_gaps": [],
            "policy_gaps": [],
            "evidence_gaps": [],
            "confidence_gaps": []
        }

        # Dimensional gaps
        for dim, score in dim_convergence.items():
            if score < 55:
                gaps["dimensional_gaps"].append({
                    "dimension": dim,
                    "score": score,
                    "gap": 55 - score,
                    "description": self.dimension_descriptions.get(dim, dim)
                })

        # Policy area gaps
        for policy, score in policy_convergence.items():
            if score < 55:
                gaps["policy_gaps"].append({
                    "policy_area": policy,
                    "score": score,
                    "gap": 55 - score
                })

        # Evidence gaps (questions with insufficient evidence)
        gaps["evidence_gaps"] = [
            {
                "question_id": answer.question_id,
                "evidence_count": len(answer.evidence),
                "confidence": answer.confidence
            }
            for answer in all_micro_answers
            if len(answer.evidence) < 2
        ]

        # Confidence gaps (low-confidence answers)
        gaps["confidence_gaps"] = [
            {
                "question_id": answer.question_id,
                "confidence": answer.confidence,
                "score": answer.quantitative_score
            }
            for answer in all_micro_answers
            if answer.confidence < 0.5
        ]

        return gaps

    def _calculate_agenda_alignment(
            self,
            all_micro_answers: List[MicroLevelAnswer],
            plan_metadata: Dict[str, Any]
    ) -> float:
        """
        Calculate alignment with Decálogo agenda (0.0-1.0)
        
        Based on:
        - Coverage of all dimensions
        - Quality of evidence
        - Overall confidence
        """
        factors = []

        # Dimension coverage
        dimensions_covered = set(
            answer.metadata.get("dimension", "")
            for answer in all_micro_answers
        )
        dimension_coverage = len(dimensions_covered) / 6.0  # 6 dimensions
        factors.append(dimension_coverage)

        # Average confidence
        avg_confidence = statistics.mean(
            answer.confidence for answer in all_micro_answers
        )
        factors.append(avg_confidence)

        # Evidence quality
        avg_evidence = statistics.mean(
            len(answer.evidence) for answer in all_micro_answers
        )
        evidence_quality = min(avg_evidence / 3.0, 1.0)  # Normalize to 1.0
        factors.append(evidence_quality)

        return statistics.mean(factors)

    def _identify_critical_gaps(
            self,
            gap_analysis: Dict[str, Any],
            dim_convergence: Dict[str, float]
    ) -> List[str]:
        """Identify the most critical gaps requiring attention"""
        critical_gaps = []

        # Critical dimensional gaps (score < 40)
        for gap_info in gap_analysis["dimensional_gaps"]:
            if gap_info["score"] < 40:
                critical_gaps.append(
                    f"CRÍTICO - {gap_info['dimension']}: {gap_info['description']} "
                    f"({gap_info['score']:.1f}%)"
                )

        # High evidence gaps
        evidence_gaps = len(gap_analysis["evidence_gaps"])
        if evidence_gaps > 10:
            critical_gaps.append(
                f"CRÍTICO - Evidencia insuficiente en {evidence_gaps} preguntas"
            )

        # High confidence gaps
        confidence_gaps = len(gap_analysis["confidence_gaps"])
        if confidence_gaps > 10:
            critical_gaps.append(
                f"CRÍTICO - Baja confianza en {confidence_gaps} respuestas"
            )

        return critical_gaps[:10]  # Top 10 critical gaps

    def _generate_strategic_recommendations(
            self,
            overall_score: float,
            dim_convergence: Dict[str, float],
            critical_gaps: List[str],
            meso_clusters: List[MesoLevelCluster]
    ) -> List[str]:
        """Generate executive-level strategic recommendations"""
        recommendations = []

        # Overall performance
        if overall_score >= 85:
            recommendations.append(
                "Mantener los altos estándares de calidad y documentación evidenciados"
            )
        elif overall_score >= 70:
            recommendations.append(
                "Fortalecer las áreas identificadas para alcanzar nivel de excelencia"
            )
        elif overall_score >= 55:
            recommendations.append(
                "Priorizar mejoras en dimensiones con desempeño insuficiente"
            )
        else:
            recommendations.append(
                "Requiere intervención urgente - múltiples deficiencias críticas identificadas"
            )

        # Dimension-specific
        weak_dimensions = [
            dim for dim, score in dim_convergence.items()
            if score < 70
        ]
        if weak_dimensions:
            recommendations.append(
                f"Fortalecer específicamente: {', '.join(weak_dimensions)}"
            )

        # Critical gaps
        if critical_gaps:
            recommendations.append(
                "Atender de inmediato las brechas críticas identificadas en el análisis"
            )

        # Cluster insights
        weak_clusters = [
            cluster for cluster in meso_clusters
            if cluster.avg_score < 70
        ]
        if weak_clusters:
            recommendations.append(
                f"Revisar y mejorar los clusters: {', '.join(c.cluster_name for c in weak_clusters[:3])}"
            )

        # Evidence strengthening
        recommendations.append(
            "Fortalecer la documentación y evidencia en todas las áreas del plan"
        )

        return recommendations[:10]  # Top 10 recommendations

    def _classify_plan(self, overall_score: float) -> str:
        """
        Classify plan using rubric levels (percentage scale 0-100)
        
        Uses >= comparisons from high to low to ensure correct assignment.
        """
        # Check from highest to lowest percentage
        if overall_score >= 85:
            return "EXCELENTE"
        elif overall_score >= 70:
            return "BUENO"
        elif overall_score >= 55:
            return "SATISFACTORIO"
        elif overall_score >= 40:
            return "INSUFICIENTE"
        else:
            return "DEFICIENTE"

    def _synthesize_evidence(
            self,
            all_micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, Any]:
        """Synthesize evidence across all questions"""
        return {
            "total_evidence_excerpts": sum(
                len(answer.evidence) for answer in all_micro_answers
            ),
            "avg_excerpts_per_question": statistics.mean(
                len(answer.evidence) for answer in all_micro_answers
            ),
            "questions_with_strong_evidence": sum(
                1 for answer in all_micro_answers
                if len(answer.evidence) >= 3
            ),
            "questions_with_weak_evidence": sum(
                1 for answer in all_micro_answers
                if len(answer.evidence) < 2
            )
        }

    def _generate_implementation_roadmap(
            self,
            critical_gaps: List[str],
            strategic_recommendations: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate prioritized implementation roadmap"""
        roadmap = []

        # Phase 1: Address critical gaps
        if critical_gaps:
            roadmap.append({
                "phase": "Fase 1 - Urgente",
                "priority": "CRÍTICA",
                "actions": critical_gaps[:5],
                "timeline": "0-3 meses"
            })

        # Phase 2: Strategic improvements
        roadmap.append({
            "phase": "Fase 2 - Estratégica",
            "priority": "ALTA",
            "actions": strategic_recommendations[:5],
            "timeline": "3-6 meses"
        })

        # Phase 3: Optimization
        roadmap.append({
            "phase": "Fase 3 - Optimización",
            "priority": "MEDIA",
            "actions": [
                "Optimizar procesos de documentación",
                "Fortalecer mecanismos de monitoreo",
                "Mejorar sistemas de evidencia"
            ],
            "timeline": "6-12 meses"
        })

        return roadmap

    def _calculate_overall_distribution(
            self,
            all_micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, int]:
        """Calculate overall score distribution"""
        distribution = defaultdict(int)
        for answer in all_micro_answers:
            distribution[answer.qualitative_note] += 1
        return dict(distribution)

    def _calculate_confidence_metrics(
            self,
            all_micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, float]:
        """Calculate confidence metrics"""
        confidences = [answer.confidence for answer in all_micro_answers]
        return {
            "avg_confidence": statistics.mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "std_confidence": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "pct_high_confidence": sum(1 for c in confidences if c >= 0.7) / len(confidences) * 100
        }

    def _apply_macro_prompts(
            self,
            all_micro_answers: List[MicroLevelAnswer],
            all_meso_clusters: List[MesoLevelCluster],
            convergence_by_dimension: Dict[str, float],
            convergence_by_policy_area: Dict[str, float],
            missing_clusters: List[str],
            critical_gaps: List[str],
            confidence_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Apply all 5 macro-level analysis prompts
        
        Returns enriched analysis including:
        - Coverage gap assessment
        - Inter-level contradiction detection
        - Bayesian portfolio composition
        - Optimized implementation roadmap
        - Peer-normalized confidence
        """
        logger.info("Applying macro prompts for enhanced analysis")
        
        # Calculate dimension and policy area coverage
        dimension_coverage = self._calculate_dimension_coverage(all_micro_answers)
        policy_area_coverage = self._calculate_policy_area_coverage(all_micro_answers)
        
        # Extract micro claims for contradiction scanning
        micro_claims = self._extract_micro_claims(all_micro_answers)
        
        # Extract meso signals
        meso_summary_signals = self._extract_meso_signals(all_meso_clusters)
        
        # Build macro narratives from convergence
        macro_narratives = {
            dim: {"score": score / 100.0} 
            for dim, score in convergence_by_dimension.items()
        }
        
        # Extract meso posteriors (using cluster scores)
        meso_posteriors = {
            cluster.cluster_name: cluster.avg_score / 100.0 
            for cluster in all_meso_clusters
        }
        
        # Use cluster weights (or equal weights if not set)
        cluster_weights = self.cluster_weights or {
            cluster.cluster_name: 1.0 / len(all_meso_clusters)
            for cluster in all_meso_clusters
        }
        
        # Build critical gaps with effort and impact estimates
        critical_gaps_structured = self._structure_critical_gaps(critical_gaps)
        
        # Build dependency graph (simplified)
        dependency_graph = {gap["id"]: [] for gap in critical_gaps_structured}
        
        # Estimate efforts and impacts
        effort_estimates = {gap["id"]: gap.get("effort", 2.0) for gap in critical_gaps_structured}
        impact_scores = {gap["id"]: gap.get("impact", 0.7) for gap in critical_gaps_structured}
        
        # Mock peer distributions (in production, would come from database)
        peer_distributions = self._get_peer_distributions(convergence_by_policy_area)
        
        # Prepare macro data for prompts
        macro_data = {
            "convergence_by_dimension": {k: v/100.0 for k, v in convergence_by_dimension.items()},
            "convergence_by_policy_area": {k: v/100.0 for k, v in convergence_by_policy_area.items()},
            "missing_clusters": missing_clusters,
            "dimension_coverage": dimension_coverage,
            "policy_area_coverage": policy_area_coverage,
            "micro_claims": micro_claims,
            "meso_summary_signals": meso_summary_signals,
            "macro_narratives": macro_narratives,
            "meso_posteriors": meso_posteriors,
            "cluster_weights": cluster_weights,
            "critical_gaps": critical_gaps_structured,
            "dependency_graph": dependency_graph,
            "effort_estimates": effort_estimates,
            "impact_scores": impact_scores,
            "peer_distributions": peer_distributions,
            "baseline_confidence": confidence_metrics.get("avg_confidence", 0.85)
        }
        
        # Execute all macro prompts
        results = self.macro_prompts.execute_all(macro_data)
        
        logger.info("Macro prompts analysis complete")
        return results
    
    def _calculate_dimension_coverage(
            self,
            all_micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, float]:
        """Calculate coverage percentage by dimension"""
        total_by_dim = defaultdict(int)
        answered_by_dim = defaultdict(int)
        
        for answer in all_micro_answers:
            # Extract dimension from question_id (e.g., "P1-D1-Q1" -> "D1")
            parts = answer.question_id.split("-")
            if len(parts) >= 2:
                dim = parts[1]
                total_by_dim[dim] += 1
                if answer.quantitative_score > 0:
                    answered_by_dim[dim] += 1
        
        coverage = {}
        for dim in total_by_dim:
            coverage[dim] = answered_by_dim[dim] / total_by_dim[dim] if total_by_dim[dim] > 0 else 0.0
        
        return coverage
    
    def _calculate_policy_area_coverage(
            self,
            all_micro_answers: List[MicroLevelAnswer]
    ) -> Dict[str, float]:
        """Calculate coverage percentage by policy area"""
        total_by_policy = defaultdict(int)
        answered_by_policy = defaultdict(int)
        
        for answer in all_micro_answers:
            # Extract policy area from question_id (e.g., "P1-D1-Q1" -> "P1")
            parts = answer.question_id.split("-")
            if len(parts) >= 1:
                policy = parts[0]
                total_by_policy[policy] += 1
                if answer.quantitative_score > 0:
                    answered_by_policy[policy] += 1
        
        coverage = {}
        for policy in total_by_policy:
            coverage[policy] = answered_by_policy[policy] / total_by_policy[policy] if total_by_policy[policy] > 0 else 0.0
        
        return coverage
    
    def _extract_micro_claims(
            self,
            all_micro_answers: List[MicroLevelAnswer]
    ) -> List[Dict[str, Any]]:
        """Extract micro-level claims from answers"""
        claims = []
        for answer in all_micro_answers:
            parts = answer.question_id.split("-")
            dimension = parts[1] if len(parts) >= 2 else "unknown"
            
            claims.append({
                "dimension": dimension,
                "score": answer.quantitative_score / 3.0,  # Normalize to 0-1
                "posterior": answer.confidence,
                "question_id": answer.question_id
            })
        
        return claims
    
    def _extract_meso_signals(
            self,
            all_meso_clusters: List[MesoLevelCluster]
    ) -> Dict[str, Any]:
        """Extract meso-level summary signals"""
        signals = {}
        for cluster in all_meso_clusters:
            for dim, score in cluster.dimension_scores.items():
                signals[dim] = {"score": score / 100.0}  # Normalize to 0-1
        
        return signals
    
    def _structure_critical_gaps(
            self,
            critical_gaps: List[str]
    ) -> List[Dict[str, Any]]:
        """Structure critical gaps with effort and impact estimates"""
        structured = []
        for i, gap in enumerate(critical_gaps):
            # Estimate effort based on gap description
            effort = 2.0  # Default 2 person-months
            if "causal" in gap.lower() or "teoría" in gap.lower():
                effort = 4.0
            elif "baseline" in gap.lower() or "línea base" in gap.lower():
                effort = 3.0
            
            # Estimate impact (higher for earlier dimensions)
            impact = 0.8 - (i * 0.05)  # Decreasing importance
            
            structured.append({
                "id": f"GAP_{i+1}",
                "name": gap,
                "effort": effort,
                "impact": max(0.5, impact)
            })
        
        return structured
    
    def _get_peer_distributions(
            self,
            convergence_by_policy_area: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Get peer distributions for normalization (mock data)"""
        # In production, this would query a database of peer plans
        # For now, use mock distributions
        distributions = {}
        for policy in convergence_by_policy_area:
            distributions[policy] = {
                "mean": 0.75,  # 75% average peer performance
                "std": 0.10    # 10% standard deviation
            }
        
        return distributions

    # ========================================================================
    # EXPORT UTILITIES
    # ========================================================================

    def export_report(
            self,
            micro_answers: List[MicroLevelAnswer],
            meso_clusters: List[MesoLevelCluster],
            macro_convergence: MacroLevelConvergence,
            output_path: Path
    ):
        """
        Export complete report to JSON
        
        Args:
            micro_answers: All MICRO-level answers
            meso_clusters: All MESO-level clusters
            macro_convergence: MACRO-level convergence
            output_path: Path to save JSON report
        """
        logger.info(f"Exporting complete report to {output_path}")

        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "3.0.0",
                "total_micro_answers": len(micro_answers),
                "total_meso_clusters": len(meso_clusters)
            },
            "macro_level": asdict(macro_convergence),
            "meso_level": [asdict(cluster) for cluster in meso_clusters],
            "micro_level": [asdict(answer) for answer in micro_answers]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Report exported successfully: {output_path}")

    # ========================================================================
    # REGISTRY EXPOSURE - Public API Methods
    # ========================================================================

    def validate_micro_answer_schema(self, answer_data: Dict[str, Any]) -> bool:
        """Validate MICRO answer against JSON schema"""
        from pathlib import Path
        import jsonschema
        
        schema_path = Path(__file__).parent / "schemas" / "report_assembly" / "micro_answer.schema.json"
        if not schema_path.exists():
            logger.warning(f"Schema not found: {schema_path}")
            return False
        
        schema = json.loads(schema_path.read_text())
        try:
            jsonschema.validate(instance=answer_data, schema=schema)
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def validate_meso_cluster_schema(self, cluster_data: Dict[str, Any]) -> bool:
        """Validate MESO cluster against JSON schema"""
        from pathlib import Path
        import jsonschema
        
        schema_path = Path(__file__).parent / "schemas" / "report_assembly" / "meso_cluster.schema.json"
        if not schema_path.exists():
            logger.warning(f"Schema not found: {schema_path}")
            return False
        
        schema = json.loads(schema_path.read_text())
        try:
            jsonschema.validate(instance=cluster_data, schema=schema)
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            return False

    def validate_macro_convergence_schema(self, convergence_data: Dict[str, Any]) -> bool:
        """Validate MACRO convergence against JSON schema"""
        from pathlib import Path
        import jsonschema
        
        schema_path = Path(__file__).parent / "schemas" / "report_assembly" / "macro_convergence.schema.json"
        if not schema_path.exists():
            logger.warning(f"Schema not found: {schema_path}")
            return False
        
        schema = json.loads(schema_path.read_text())
        try:
            jsonschema.validate(instance=convergence_data, schema=schema)
            return True
        except jsonschema.ValidationError as e:
            logger.error(f"Schema validation failed: {e}")
            return False


# ============================================================================
# PRODUCER CLASS - Registry Exposure
# ============================================================================

class ReportAssemblyProducer:
    """
    Producer wrapper for ReportAssembler with public API for registry exposure
    
    Provides 40+ public methods for orchestrator integration without
    exposing internal implementation details or summarization logic.
    
    Version: 1.0.0
    Producer Type: Report Assembly / Aggregation
    """
    
    def __init__(
            self,
            dimension_descriptions: Optional[Dict[str, str]] = None,
            cluster_weights: Optional[Dict[str, float]] = None,
            cluster_policy_weights: Optional[Dict[str, Dict[str, float]]] = None,
            causal_thresholds: Optional[Dict[str, float]] = None
    ):
        """Initialize producer with optional configuration"""
        self.assembler = ReportAssembler(
            dimension_descriptions=dimension_descriptions,
            cluster_weights=cluster_weights,
            cluster_policy_weights=cluster_policy_weights,
            causal_thresholds=causal_thresholds
        )
        logger.info("ReportAssemblyProducer initialized")
    
    # ========================================================================
    # MICRO LEVEL API - Question Analysis
    # ========================================================================
    
    def produce_micro_answer(
            self,
            question_spec: Any,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Dict[str, Any]:
        """
        Produce MICRO-level answer for a single question
        
        Returns: Serializable dictionary with complete answer
        """
        answer = self.assembler.generate_micro_answer(
            question_spec, execution_results, plan_text
        )
        return asdict(answer)
    
    def get_micro_answer_score(self, answer: MicroLevelAnswer) -> float:
        """Extract quantitative score from MICRO answer"""
        return answer.quantitative_score
    
    def get_micro_answer_qualitative(self, answer: MicroLevelAnswer) -> str:
        """Extract qualitative classification from MICRO answer"""
        return answer.qualitative_note
    
    def get_micro_answer_evidence(self, answer: MicroLevelAnswer) -> List[str]:
        """Extract evidence excerpts from MICRO answer"""
        return answer.evidence
    
    def get_micro_answer_confidence(self, answer: MicroLevelAnswer) -> float:
        """Extract confidence score from MICRO answer"""
        return answer.confidence
    
    def get_micro_answer_modules(self, answer: MicroLevelAnswer) -> List[str]:
        """Extract list of executed modules from MICRO answer"""
        return answer.modules_executed
    
    def get_micro_answer_execution_time(self, answer: MicroLevelAnswer) -> float:
        """Extract execution time from MICRO answer"""
        return answer.execution_time
    
    def get_micro_answer_elements_found(self, answer: MicroLevelAnswer) -> Dict[str, bool]:
        """Extract detected elements from MICRO answer"""
        return answer.elements_found
    
    def count_micro_evidence_excerpts(self, answer: MicroLevelAnswer) -> int:
        """Count evidence excerpts in MICRO answer"""
        return len(answer.evidence)
    
    def is_micro_answer_excellent(self, answer: MicroLevelAnswer) -> bool:
        """Check if MICRO answer is classified as EXCELENTE"""
        return answer.qualitative_note == "EXCELENTE"
    
    def is_micro_answer_passing(self, answer: MicroLevelAnswer) -> bool:
        """Check if MICRO answer meets minimum passing threshold"""
        return answer.quantitative_score >= 1.65  # ACEPTABLE threshold
    
    # ========================================================================
    # MESO LEVEL API - Cluster Aggregation
    # ========================================================================
    
    def produce_meso_cluster(
            self,
            cluster_name: str,
            cluster_description: str,
            micro_answers: List[Dict[str, Any]],
            cluster_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Produce MESO-level cluster aggregation
        
        Returns: Serializable dictionary with cluster analysis
        """
        # Convert dicts to MicroLevelAnswer objects
        answer_objects = [
            MicroLevelAnswer(**answer) for answer in micro_answers
        ]
        
        cluster = self.assembler.generate_meso_cluster(
            cluster_name, cluster_description, answer_objects, cluster_definition
        )
        return asdict(cluster)
    
    def get_meso_cluster_score(self, cluster: MesoLevelCluster) -> float:
        """Extract average score from MESO cluster"""
        return cluster.avg_score
    
    def get_meso_cluster_policy_areas(self, cluster: MesoLevelCluster) -> List[str]:
        """Extract policy areas from MESO cluster"""
        return cluster.policy_areas
    
    def get_meso_cluster_dimension_scores(self, cluster: MesoLevelCluster) -> Dict[str, float]:
        """Extract dimension scores from MESO cluster"""
        return cluster.dimension_scores
    
    def get_meso_cluster_strengths(self, cluster: MesoLevelCluster) -> List[str]:
        """Extract identified strengths from MESO cluster"""
        return cluster.strengths
    
    def get_meso_cluster_weaknesses(self, cluster: MesoLevelCluster) -> List[str]:
        """Extract identified weaknesses from MESO cluster"""
        return cluster.weaknesses
    
    def get_meso_cluster_recommendations(self, cluster: MesoLevelCluster) -> List[str]:
        """Extract recommendations from MESO cluster"""
        return cluster.recommendations
    
    def get_meso_cluster_coverage(self, cluster: MesoLevelCluster) -> float:
        """Extract question coverage percentage from MESO cluster"""
        return cluster.question_coverage
    
    def get_meso_cluster_question_counts(self, cluster: MesoLevelCluster) -> Tuple[int, int]:
        """Extract total and answered question counts from MESO cluster"""
        return cluster.total_questions, cluster.answered_questions
    
    def count_meso_strengths(self, cluster: MesoLevelCluster) -> int:
        """Count strengths identified in MESO cluster"""
        return len(cluster.strengths)
    
    def count_meso_weaknesses(self, cluster: MesoLevelCluster) -> int:
        """Count weaknesses identified in MESO cluster"""
        return len(cluster.weaknesses)
    
    def is_meso_cluster_excellent(self, cluster: MesoLevelCluster) -> bool:
        """Check if MESO cluster score is in EXCELENTE range"""
        return cluster.avg_score >= 85
    
    def is_meso_cluster_passing(self, cluster: MesoLevelCluster) -> bool:
        """Check if MESO cluster meets minimum passing threshold"""
        return cluster.avg_score >= 55  # SATISFACTORIO threshold
    
    # ========================================================================
    # MACRO LEVEL API - Convergence Analysis
    # ========================================================================
    
    def produce_macro_convergence(
            self,
            all_micro_answers: List[Dict[str, Any]],
            all_meso_clusters: List[Dict[str, Any]],
            plan_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Produce MACRO-level convergence analysis
        
        Returns: Serializable dictionary with executive summary
        """
        # Convert dicts to objects
        answer_objects = [
            MicroLevelAnswer(**answer) for answer in all_micro_answers
        ]
        cluster_objects = [
            MesoLevelCluster(**cluster) for cluster in all_meso_clusters
        ]
        
        convergence = self.assembler.generate_macro_convergence(
            answer_objects, cluster_objects, plan_metadata
        )
        return asdict(convergence)
    
    def get_macro_overall_score(self, convergence: MacroLevelConvergence) -> float:
        """Extract overall score from MACRO convergence"""
        return convergence.overall_score
    
    def get_macro_dimension_convergence(self, convergence: MacroLevelConvergence) -> Dict[str, float]:
        """Extract dimension convergence scores from MACRO"""
        return convergence.convergence_by_dimension
    
    def get_macro_policy_convergence(self, convergence: MacroLevelConvergence) -> Dict[str, float]:
        """Extract policy area convergence scores from MACRO"""
        return convergence.convergence_by_policy_area
    
    def get_macro_gap_analysis(self, convergence: MacroLevelConvergence) -> Dict[str, Any]:
        """Extract gap analysis from MACRO convergence"""
        return convergence.gap_analysis
    
    def get_macro_agenda_alignment(self, convergence: MacroLevelConvergence) -> float:
        """Extract agenda alignment score from MACRO"""
        return convergence.agenda_alignment
    
    def get_macro_critical_gaps(self, convergence: MacroLevelConvergence) -> List[str]:
        """Extract critical gaps list from MACRO"""
        return convergence.critical_gaps
    
    def get_macro_strategic_recommendations(self, convergence: MacroLevelConvergence) -> List[str]:
        """Extract strategic recommendations from MACRO"""
        return convergence.strategic_recommendations
    
    def get_macro_classification(self, convergence: MacroLevelConvergence) -> str:
        """Extract plan classification from MACRO"""
        return convergence.plan_classification
    
    def get_macro_evidence_synthesis(self, convergence: MacroLevelConvergence) -> Dict[str, Any]:
        """Extract evidence synthesis from MACRO"""
        return convergence.evidence_synthesis
    
    def get_macro_implementation_roadmap(self, convergence: MacroLevelConvergence) -> List[Dict[str, Any]]:
        """Extract implementation roadmap from MACRO"""
        return convergence.implementation_roadmap
    
    def get_macro_score_distribution(self, convergence: MacroLevelConvergence) -> Dict[str, int]:
        """Extract score distribution from MACRO"""
        return convergence.score_distribution
    
    def get_macro_confidence_metrics(self, convergence: MacroLevelConvergence) -> Dict[str, float]:
        """Extract confidence metrics from MACRO"""
        return convergence.confidence_metrics
    
    def count_macro_critical_gaps(self, convergence: MacroLevelConvergence) -> int:
        """Count critical gaps in MACRO convergence"""
        return len(convergence.critical_gaps)
    
    def count_macro_strategic_recommendations(self, convergence: MacroLevelConvergence) -> int:
        """Count strategic recommendations in MACRO"""
        return len(convergence.strategic_recommendations)
    
    def is_macro_excellent(self, convergence: MacroLevelConvergence) -> bool:
        """Check if MACRO overall score is in EXCELENTE range"""
        return convergence.overall_score >= 85
    
    def is_macro_passing(self, convergence: MacroLevelConvergence) -> bool:
        """Check if MACRO meets minimum passing threshold"""
        return convergence.overall_score >= 55
    
    # ========================================================================
    # SCORING UTILITIES API
    # ========================================================================
    
    def convert_score_to_percentage(self, score: float) -> float:
        """Convert 0-3 score to 0-100 percentage"""
        return (score / 3.0) * 100
    
    def convert_percentage_to_score(self, percentage: float) -> float:
        """Convert 0-100 percentage to 0-3 score"""
        return (percentage / 100.0) * 3.0
    
    def classify_score(self, score: float) -> str:
        """Classify a 0-3 score into qualitative level"""
        return self.assembler._score_to_qualitative_question(score)
    
    def classify_percentage(self, percentage: float) -> str:
        """Classify a 0-100 percentage into qualitative level"""
        for level, (min_pct, max_pct) in self.assembler.rubric_levels.items():
            if min_pct <= percentage <= max_pct:
                return level
        return "DEFICIENTE"
    
    def get_rubric_threshold(self, level: str) -> Tuple[float, float]:
        """Get percentage threshold range for a rubric level"""
        return self.assembler.rubric_levels.get(level, (0, 0))
    
    def get_question_rubric_threshold(self, level: str) -> Tuple[float, float]:
        """Get 0-3 score threshold range for question-level rubric"""
        return self.assembler.question_rubric.get(level, (0, 0))
    
    # ========================================================================
    # CONFIGURATION API
    # ========================================================================
    
    def get_dimension_description(self, dimension: str) -> str:
        """Get description for a dimension (D1-D6)"""
        return self.assembler.dimension_descriptions.get(dimension, "")
    
    def list_dimensions(self) -> List[str]:
        """List all dimensions"""
        return list(self.assembler.dimension_descriptions.keys())
    
    def list_rubric_levels(self) -> List[str]:
        """List all rubric levels"""
        return list(self.assembler.rubric_levels.keys())
    
    def get_causal_threshold(self, dimension: str) -> float:
        """Get causal coherence threshold for a dimension"""
        return self.assembler.causal_thresholds.get(
            dimension,
            self.assembler.causal_thresholds.get('default', 0.6)
        )
    
    def get_cluster_weight(self, cluster_id: str) -> Optional[float]:
        """Get weight for a cluster in macro aggregation"""
        return self.assembler.cluster_weights.get(cluster_id)
    
    def get_cluster_policy_weights(self, cluster_id: str) -> Optional[Dict[str, float]]:
        """Get policy area weights for a cluster"""
        return self.assembler.cluster_policy_weights.get(cluster_id)
    
    # ========================================================================
    # EXPORT API
    # ========================================================================
    
    def export_complete_report(
            self,
            micro_answers: List[Dict[str, Any]],
            meso_clusters: List[Dict[str, Any]],
            macro_convergence: Dict[str, Any],
            output_path: str
    ) -> None:
        """Export complete report to JSON file"""
        # Convert dicts to objects
        answer_objects = [MicroLevelAnswer(**answer) for answer in micro_answers]
        cluster_objects = [MesoLevelCluster(**cluster) for cluster in meso_clusters]
        convergence_object = MacroLevelConvergence(**macro_convergence)
        
        self.assembler.export_report(
            answer_objects,
            cluster_objects,
            convergence_object,
            Path(output_path)
        )
    
    def serialize_micro_answer(self, answer: MicroLevelAnswer) -> Dict[str, Any]:
        """Serialize MICRO answer to dictionary"""
        return asdict(answer)
    
    def serialize_meso_cluster(self, cluster: MesoLevelCluster) -> Dict[str, Any]:
        """Serialize MESO cluster to dictionary"""
        return asdict(cluster)
    
    def serialize_macro_convergence(self, convergence: MacroLevelConvergence) -> Dict[str, Any]:
        """Serialize MACRO convergence to dictionary"""
        return asdict(convergence)
    
    def deserialize_micro_answer(self, data: Dict[str, Any]) -> MicroLevelAnswer:
        """Deserialize dictionary to MICRO answer"""
        return MicroLevelAnswer(**data)
    
    def deserialize_meso_cluster(self, data: Dict[str, Any]) -> MesoLevelCluster:
        """Deserialize dictionary to MESO cluster"""
        return MesoLevelCluster(**data)
    
    def deserialize_macro_convergence(self, data: Dict[str, Any]) -> MacroLevelConvergence:
        """Deserialize dictionary to MACRO convergence"""
        return MacroLevelConvergence(**data)
    
    # ========================================================================
    # SCHEMA VALIDATION API
    # ========================================================================
    
    def validate_micro_answer(self, answer_data: Dict[str, Any]) -> bool:
        """Validate MICRO answer against JSON schema"""
        return self.assembler.validate_micro_answer_schema(answer_data)
    
    def validate_meso_cluster(self, cluster_data: Dict[str, Any]) -> bool:
        """Validate MESO cluster against JSON schema"""
        return self.assembler.validate_meso_cluster_schema(cluster_data)
    
    def validate_macro_convergence(self, convergence_data: Dict[str, Any]) -> bool:
        """Validate MACRO convergence against JSON schema"""
        return self.assembler.validate_macro_convergence_schema(convergence_data)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    producer = ReportAssemblyProducer()
    
    print("=" * 80)
    print("REPORT ASSEMBLY PRODUCER - REGISTRY EXPOSURE")
    print("=" * 80)
    print("\nCapabilities:")
    print("  - MICRO level: Question-by-question analysis")
    print("  - MESO level: Cluster aggregation")
    print("  - MACRO level: Overall convergence")
    print("\nRubric Levels:")
    for level in producer.list_rubric_levels():
        min_s, max_s = producer.get_rubric_threshold(level)
        print(f"  {level}: {min_s}-{max_s}%")
    print("\nDimensions:")
    for dim in producer.list_dimensions():
        desc = producer.get_dimension_description(dim)
        print(f"  {dim}: {desc[:60]}...")
    print("=" * 80)
    print("=" * 80)