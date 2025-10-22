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

This module generates comprehensive reports at three hierarchical levels:
1. MICRO: Individual question answers with evidence
2. MESO: Cluster-level aggregations by policy areas
3. MACRO: Overall plan convergence with Decálogo framework

Author: Integration Team
Version: 3.0.0 - Complete with 9 Adapters
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

    def __init__(self, dimension_descriptions: Optional[Dict[str, str]] = None):
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

        logger.info("ReportAssembler initialized with rubric definitions")

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
        score, elements_found, pattern_matches = self._apply_scoring_modality(
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
                "question_number": getattr(question_spec, 'question_no', 0)
            }
        )

    def _apply_scoring_modality(
            self,
            question_spec,
            execution_results: Dict[str, Any],
            plan_text: str
    ) -> Tuple[float, Dict[str, bool], Dict[str, Any]]:
        """
        Apply scoring modality (TYPE_A, TYPE_B, etc.) to calculate question score
        
        Returns:
            (score, elements_found, pattern_matches)
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

        return score, elements_found, pattern_matches

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
        # Precondition assertions
        assert hasattr(question_spec, 'expected_elements'), \
            "question_spec must have expected_elements attribute for TYPE_A scoring"
        assert execution_results, \
            "execution_results cannot be empty for TYPE_A scoring"
        assert plan_text and isinstance(plan_text, str), \
            "plan_text must be non-empty string for TYPE_A scoring"
        
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
        assert execution_results, \
            "execution_results cannot be empty for TYPE_E scoring"
        
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
        
        Uses question_rubric thresholds
        """
        for level, (min_score, max_score) in self.question_rubric.items():
            if min_score <= score <= max_score:
                return level
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
                "score_distribution": self._calculate_score_distribution(micro_answers)
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
        overall_score = (statistics.mean(all_scores) / 3.0) * 100 if all_scores else 0

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
            metadata={
                "timestamp": datetime.now().isoformat(),
                "total_questions_analyzed": len(all_micro_answers),
                "total_clusters": len(all_meso_clusters),
                "plan_name": plan_metadata.get("name", "Unknown")
            }
        )

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
        """Classify plan using rubric levels"""
        for level, (min_score, max_score) in self.rubric_levels.items():
            if min_score <= overall_score <= max_score:
                return level
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


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    assembler = ReportAssembler()
    
    print("=" * 80)
    print("REPORT ASSEMBLER - COMPLETE IMPLEMENTATION")
    print("=" * 80)
    print("\nCapabilities:")
    print("  - MICRO level: Question-by-question analysis")
    print("  - MESO level: Cluster aggregation")
    print("  - MACRO level: Overall convergence")
    print("\nRubric Levels:")
    for level, (min_s, max_s) in assembler.rubric_levels.items():
        print(f"  {level}: {min_s}-{max_s}%")
    print("=" * 80)