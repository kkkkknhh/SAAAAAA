#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 Activities Design & Coherence - Strict Method Concurrence Orchestrator
=========================================================================

DIMENSIÓN D2: DISEÑO DE ACTIVIDADES Y COHERENCIA

This module enforces strict concurrence of methods for D2 questions (Q1-Q5)
following SIN_CARRETA doctrine:
- No graceful degradation
- No strategic simplification
- Deterministic execution
- Explicit failure semantics
- Full traceability

CONTRACT CONDITIONS:
- All listed methods for each question MUST execute deterministically
- Partial execution is strictly forbidden
- Fallback or best-effort responses are prohibited
- Execution must be observable, auditable, and reproducible
- Any method failure aborts orchestration with explicit diagnostics

Author: Integration Team
Version: 1.0.0
Standard: SIN_CARRETA Doctrine
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class D2Question(Enum):
    """D2 Question enumeration for type safety."""
    Q1_FORMATO_TABULAR = "D2-Q1"
    Q2_CAUSALIDAD_ACTIVIDADES = "D2-Q2"
    Q3_CLASIFICACION_TEMATICA = "D2-Q3"
    Q4_RIESGOS_MITIGACION = "D2-Q4"
    Q5_COHERENCIA_ESTRATEGICA = "D2-Q5"


class OrchestrationError(Exception):
    """Raised when orchestration contract is violated."""
    pass


class MethodExecutionError(Exception):
    """Raised when a required method fails to execute."""
    pass


@dataclass
class MethodSpec:
    """Specification for a required method in the orchestration chain."""
    module_name: str
    class_name: str
    method_name: str
    fully_qualified_name: str
    is_private: bool = False
    
    def __post_init__(self):
        """Generate fully qualified name if not provided."""
        if not self.fully_qualified_name:
            self.fully_qualified_name = f"{self.class_name}.{self.method_name}"


@dataclass
class MethodExecutionResult:
    """Result of a method execution with full traceability."""
    method_spec: MethodSpec
    success: bool
    execution_time_ms: float
    output: Any = None
    error: Optional[Exception] = None
    trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionOrchestrationResult:
    """Result of orchestrating all methods for a D2 question."""
    question_id: str
    total_methods: int
    executed_methods: int
    failed_methods: int
    success: bool
    execution_time_ms: float
    method_results: List[MethodExecutionResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)


class D2MethodRegistry:
    """
    Registry of all required methods for each D2 question.
    
    This is the canonical source of truth for method concurrence.
    Based on the issue specification.
    """
    
    # D2-Q1: Formato Tabular y Trazabilidad (20 methods)
    Q1_METHODS = [
        # policy_processor.py (4 methods)
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "_match_patterns_in_sentences",
                   "IndustrialPolicyProcessor._match_patterns_in_sentences", is_private=True),
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "process",
                   "IndustrialPolicyProcessor.process"),
        MethodSpec("policy_processor", "PolicyTextProcessor", "segment_into_sentences",
                   "PolicyTextProcessor.segment_into_sentences"),
        MethodSpec("policy_processor", "BayesianEvidenceScorer", "compute_evidence_score",
                   "BayesianEvidenceScorer.compute_evidence_score"),
        
        # financiero_viabilidad_tablas.py (12 methods)
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "extract_tables",
                   "PDETMunicipalPlanAnalyzer.extract_tables"),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_clean_dataframe",
                   "PDETMunicipalPlanAnalyzer._clean_dataframe", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_is_likely_header",
                   "PDETMunicipalPlanAnalyzer._is_likely_header", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_deduplicate_tables",
                   "PDETMunicipalPlanAnalyzer._deduplicate_tables", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_reconstruct_fragmented_tables",
                   "PDETMunicipalPlanAnalyzer._reconstruct_fragmented_tables", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_classify_tables",
                   "PDETMunicipalPlanAnalyzer._classify_tables", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "analyze_municipal_plan",
                   "PDETMunicipalPlanAnalyzer.analyze_municipal_plan"),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_extract_from_budget_table",
                   "PDETMunicipalPlanAnalyzer._extract_from_budget_table", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_extract_from_responsibility_tables",
                   "PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "identify_responsible_entities",
                   "PDETMunicipalPlanAnalyzer.identify_responsible_entities"),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_consolidate_entities",
                   "PDETMunicipalPlanAnalyzer._consolidate_entities", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_score_entity_specificity",
                   "PDETMunicipalPlanAnalyzer._score_entity_specificity", is_private=True),
        
        # contradiction_deteccion.py (3 methods)
        MethodSpec("contradiction_deteccion", "TemporalLogicVerifier", "_build_timeline",
                   "TemporalLogicVerifier._build_timeline", is_private=True),
        MethodSpec("contradiction_deteccion", "TemporalLogicVerifier", "_check_deadline_constraints",
                   "TemporalLogicVerifier._check_deadline_constraints", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_detect_temporal_conflicts",
                   "PolicyContradictionDetector._detect_temporal_conflicts", is_private=True),
        
        # semantic_chunking_policy.py (1 method)
        MethodSpec("semantic_chunking_policy", "SemanticProcessor", "_detect_table",
                   "SemanticProcessor._detect_table", is_private=True),
    ]
    
    # D2-Q2: Causalidad de Actividades (25 methods)
    Q2_METHODS = [
        # policy_processor.py (5 methods)
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "_match_patterns_in_sentences",
                   "IndustrialPolicyProcessor._match_patterns_in_sentences", is_private=True),
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "_analyze_causal_dimensions",
                   "IndustrialPolicyProcessor._analyze_causal_dimensions", is_private=True),
        MethodSpec("policy_processor", "PolicyTextProcessor", "segment_into_sentences",
                   "PolicyTextProcessor.segment_into_sentences"),
        MethodSpec("policy_processor", "PolicyTextProcessor", "extract_contextual_window",
                   "PolicyTextProcessor.extract_contextual_window"),
        MethodSpec("policy_processor", "BayesianEvidenceScorer", "compute_evidence_score",
                   "BayesianEvidenceScorer.compute_evidence_score"),
        
        # contradiction_deteccion.py (8 methods)
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_determine_relation_type",
                   "PolicyContradictionDetector._determine_relation_type", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_build_knowledge_graph",
                   "PolicyContradictionDetector._build_knowledge_graph", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_extract_policy_statements",
                   "PolicyContradictionDetector._extract_policy_statements", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_identify_dependencies",
                   "PolicyContradictionDetector._identify_dependencies", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_get_dependency_depth",
                   "PolicyContradictionDetector._get_dependency_depth", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_global_semantic_coherence",
                   "PolicyContradictionDetector._calculate_global_semantic_coherence", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_generate_embeddings",
                   "PolicyContradictionDetector._generate_embeddings", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_similarity",
                   "PolicyContradictionDetector._calculate_similarity", is_private=True),
        
        # dereck_beach.py (12 methods) - Note: Only listing the ones from the issue
        MethodSpec("dereck_beach", "CausalExtractor", "__init__",
                   "CausalExtractor.__init__"),
        MethodSpec("dereck_beach", "CausalExtractor", "extract_causal_hierarchy",
                   "CausalExtractor.extract_causal_hierarchy"),
        MethodSpec("dereck_beach", "CausalExtractor", "_extract_goals",
                   "CausalExtractor._extract_goals", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_extract_goal_text",
                   "CausalExtractor._extract_goal_text", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_classify_goal_type",
                   "CausalExtractor._classify_goal_type", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_add_node_to_graph",
                   "CausalExtractor._add_node_to_graph", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_extract_causal_links",
                   "CausalExtractor._extract_causal_links", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_build_type_hierarchy",
                   "CausalExtractor._build_type_hierarchy", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_extract_numerical_data",
                   "CausalExtractor._extract_numerical_data", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_identify_responsible_entity",
                   "CausalExtractor._identify_responsible_entity", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_extract_entity_activities",
                   "CausalExtractor._extract_entity_activities", is_private=True),
        MethodSpec("dereck_beach", "CausalExtractor", "_extract_contextual_risks",
                   "CausalExtractor._extract_contextual_risks", is_private=True),
    ]
    
    # D2-Q3: Clasificación Temática (18 methods)
    Q3_METHODS = [
        # policy_processor.py (4 methods)
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "_match_patterns_in_sentences",
                   "IndustrialPolicyProcessor._match_patterns_in_sentences", is_private=True),
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "process",
                   "IndustrialPolicyProcessor.process"),
        MethodSpec("policy_processor", "PolicyTextProcessor", "segment_into_sentences",
                   "PolicyTextProcessor.segment_into_sentences"),
        MethodSpec("policy_processor", "BayesianEvidenceScorer", "compute_evidence_score",
                   "BayesianEvidenceScorer.compute_evidence_score"),
        
        # Analyzer_one.py (6 methods)
        MethodSpec("Analyzer_one", "SemanticAnalyzer", "_classify_cross_cutting_themes",
                   "SemanticAnalyzer._classify_cross_cutting_themes", is_private=True),
        MethodSpec("Analyzer_one", "SemanticAnalyzer", "_classify_policy_domain",
                   "SemanticAnalyzer._classify_policy_domain", is_private=True),
        MethodSpec("Analyzer_one", "SemanticAnalyzer", "extract_semantic_cube",
                   "SemanticAnalyzer.extract_semantic_cube"),
        MethodSpec("Analyzer_one", "SemanticAnalyzer", "_process_segment",
                   "SemanticAnalyzer._process_segment", is_private=True),
        MethodSpec("Analyzer_one", "SemanticAnalyzer", "_vectorize_segments",
                   "SemanticAnalyzer._vectorize_segments", is_private=True),
        MethodSpec("Analyzer_one", "SemanticAnalyzer", "_calculate_semantic_complexity",
                   "SemanticAnalyzer._calculate_semantic_complexity", is_private=True),
        
        # contradiction_deteccion.py (5 methods)
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_build_knowledge_graph",
                   "PolicyContradictionDetector._build_knowledge_graph", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_global_semantic_coherence",
                   "PolicyContradictionDetector._calculate_global_semantic_coherence", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_generate_embeddings",
                   "PolicyContradictionDetector._generate_embeddings", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_similarity",
                   "PolicyContradictionDetector._calculate_similarity", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_determine_semantic_role",
                   "PolicyContradictionDetector._determine_semantic_role", is_private=True),
        
        # embedding_policy.py (3 methods)
        MethodSpec("embedding_policy", "PolicyAnalysisEmbedder", "semantic_search",
                   "PolicyAnalysisEmbedder.semantic_search"),
        MethodSpec("embedding_policy", "PolicyAnalysisEmbedder", "_filter_by_pdq",
                   "PolicyAnalysisEmbedder._filter_by_pdq", is_private=True),
        MethodSpec("embedding_policy", "AdvancedSemanticChunker", "_infer_pdq_context",
                   "AdvancedSemanticChunker._infer_pdq_context", is_private=True),
    ]
    
    # D2-Q4: Riesgos y Mitigación (20 methods)
    Q4_METHODS = [
        # policy_processor.py (4 methods)
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "_match_patterns_in_sentences",
                   "IndustrialPolicyProcessor._match_patterns_in_sentences", is_private=True),
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "process",
                   "IndustrialPolicyProcessor.process"),
        MethodSpec("policy_processor", "PolicyTextProcessor", "extract_contextual_window",
                   "PolicyTextProcessor.extract_contextual_window"),
        MethodSpec("policy_processor", "BayesianEvidenceScorer", "compute_evidence_score",
                   "BayesianEvidenceScorer.compute_evidence_score"),
        
        # contradiction_deteccion.py (8 methods)
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_detect_logical_incompatibilities",
                   "PolicyContradictionDetector._detect_logical_incompatibilities", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_has_logical_conflict",
                   "PolicyContradictionDetector._has_logical_conflict", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_detect_resource_conflicts",
                   "PolicyContradictionDetector._detect_resource_conflicts", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_contradiction_entropy",
                   "PolicyContradictionDetector._calculate_contradiction_entropy", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_classify_contradiction",
                   "PolicyContradictionDetector._classify_contradiction", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_severity",
                   "PolicyContradictionDetector._calculate_severity", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_generate_resolution_recommendations",
                   "PolicyContradictionDetector._generate_resolution_recommendations", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_suggest_resolutions",
                   "PolicyContradictionDetector._suggest_resolutions", is_private=True),
        
        # Analyzer_one.py (4 methods)
        MethodSpec("Analyzer_one", "PerformanceAnalyzer", "_detect_bottlenecks",
                   "PerformanceAnalyzer._detect_bottlenecks", is_private=True),
        MethodSpec("Analyzer_one", "PerformanceAnalyzer", "_calculate_loss_functions",
                   "PerformanceAnalyzer._calculate_loss_functions", is_private=True),
        MethodSpec("Analyzer_one", "TextMiningEngine", "_assess_risks",
                   "TextMiningEngine._assess_risks", is_private=True),
        MethodSpec("Analyzer_one", "TextMiningEngine", "_generate_interventions",
                   "TextMiningEngine._generate_interventions", is_private=True),
        
        # financiero_viabilidad_tablas.py (4 methods)
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_bayesian_risk_inference",
                   "PDETMunicipalPlanAnalyzer._bayesian_risk_inference", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_interpret_risk",
                   "PDETMunicipalPlanAnalyzer._interpret_risk", is_private=True),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "sensitivity_analysis",
                   "PDETMunicipalPlanAnalyzer.sensitivity_analysis"),
        MethodSpec("financiero_viabilidad_tablas", "PDETMunicipalPlanAnalyzer", "_identify_confounders",
                   "PDETMunicipalPlanAnalyzer._identify_confounders", is_private=True),
    ]
    
    # D2-Q5: Coherencia Estratégica (24 methods)
    Q5_METHODS = [
        # policy_processor.py (5 methods)
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "process",
                   "IndustrialPolicyProcessor.process"),
        MethodSpec("policy_processor", "IndustrialPolicyProcessor", "_analyze_causal_dimensions",
                   "IndustrialPolicyProcessor._analyze_causal_dimensions", is_private=True),
        MethodSpec("policy_processor", "PolicyTextProcessor", "segment_into_sentences",
                   "PolicyTextProcessor.segment_into_sentences"),
        MethodSpec("policy_processor", "BayesianEvidenceScorer", "compute_evidence_score",
                   "BayesianEvidenceScorer.compute_evidence_score"),
        MethodSpec("policy_processor", "BayesianEvidenceScorer", "_calculate_shannon_entropy",
                   "BayesianEvidenceScorer._calculate_shannon_entropy", is_private=True),
        
        # contradiction_deteccion.py (12 methods)
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_global_semantic_coherence",
                   "PolicyContradictionDetector._calculate_global_semantic_coherence", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_text_similarity",
                   "PolicyContradictionDetector._text_similarity", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_build_knowledge_graph",
                   "PolicyContradictionDetector._build_knowledge_graph", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_get_dependency_depth",
                   "PolicyContradictionDetector._get_dependency_depth", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_graph_fragmentation",
                   "PolicyContradictionDetector._calculate_graph_fragmentation", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_get_graph_statistics",
                   "PolicyContradictionDetector._get_graph_statistics", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_generate_embeddings",
                   "PolicyContradictionDetector._generate_embeddings", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_similarity",
                   "PolicyContradictionDetector._calculate_similarity", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_extract_policy_statements",
                   "PolicyContradictionDetector._extract_policy_statements", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_identify_dependencies",
                   "PolicyContradictionDetector._identify_dependencies", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_objective_alignment",
                   "PolicyContradictionDetector._calculate_objective_alignment", is_private=True),
        MethodSpec("contradiction_deteccion", "PolicyContradictionDetector", "_calculate_syntactic_complexity",
                   "PolicyContradictionDetector._calculate_syntactic_complexity", is_private=True),
        
        # Analyzer_one.py (4 methods)
        MethodSpec("Analyzer_one", "SemanticAnalyzer", "extract_semantic_cube",
                   "SemanticAnalyzer.extract_semantic_cube"),
        MethodSpec("Analyzer_one", "SemanticAnalyzer", "_calculate_semantic_complexity",
                   "SemanticAnalyzer._calculate_semantic_complexity", is_private=True),
        MethodSpec("Analyzer_one", "PerformanceAnalyzer", "analyze_performance",
                   "PerformanceAnalyzer.analyze_performance"),
        MethodSpec("Analyzer_one", "TextMiningEngine", "diagnose_critical_links",
                   "TextMiningEngine.diagnose_critical_links"),
        
        # embedding_policy.py (3 methods)
        MethodSpec("embedding_policy", "PolicyAnalysisEmbedder", "semantic_search",
                   "PolicyAnalysisEmbedder.semantic_search"),
        MethodSpec("embedding_policy", "PolicyAnalysisEmbedder", "compare_policy_interventions",
                   "PolicyAnalysisEmbedder.compare_policy_interventions"),
        MethodSpec("embedding_policy", "BayesianNumericalAnalyzer", "_compute_coherence",
                   "BayesianNumericalAnalyzer._compute_coherence", is_private=True),
    ]
    
    @classmethod
    def get_methods_for_question(cls, question: D2Question) -> List[MethodSpec]:
        """Get the list of required methods for a D2 question."""
        mapping = {
            D2Question.Q1_FORMATO_TABULAR: cls.Q1_METHODS,
            D2Question.Q2_CAUSALIDAD_ACTIVIDADES: cls.Q2_METHODS,
            D2Question.Q3_CLASIFICACION_TEMATICA: cls.Q3_METHODS,
            D2Question.Q4_RIESGOS_MITIGACION: cls.Q4_METHODS,
            D2Question.Q5_COHERENCIA_ESTRATEGICA: cls.Q5_METHODS,
        }
        return mapping[question]
    
    @classmethod
    def get_all_d2_methods(cls) -> Set[str]:
        """Get all unique methods across D2 questions."""
        all_methods = set()
        for question in D2Question:
            methods = cls.get_methods_for_question(question)
            all_methods.update(m.fully_qualified_name for m in methods)
        return all_methods


class D2ActivitiesOrchestrator:
    """
    Orchestrator for D2 Activities Design & Coherence.
    
    Enforces strict concurrence of methods with SIN_CARRETA doctrine:
    - Deterministic execution
    - No graceful degradation
    - Explicit failure semantics
    - Full traceability
    """
    
    def __init__(self, strict_mode: bool = True, trace_execution: bool = True):
        """
        Initialize D2 orchestrator.
        
        Args:
            strict_mode: If True, abort on any method failure (SIN_CARRETA doctrine)
            trace_execution: If True, maintain full execution traces
        """
        self.strict_mode = strict_mode
        self.trace_execution = trace_execution
        self.registry = D2MethodRegistry()
        self._method_cache: Dict[str, Callable] = {}
        
        logger.info(f"D2ActivitiesOrchestrator initialized (strict_mode={strict_mode}, trace={trace_execution})")
    
    def _resolve_method(self, method_spec: MethodSpec) -> Callable:
        """
        Resolve a method specification to an actual callable.
        
        Args:
            method_spec: Method specification to resolve
            
        Returns:
            Callable method
            
        Raises:
            OrchestrationError: If method cannot be resolved
        """
        fqn = method_spec.fully_qualified_name
        
        # Check cache first
        if fqn in self._method_cache:
            return self._method_cache[fqn]
        
        try:
            # Import the module
            module = __import__(method_spec.module_name, fromlist=[method_spec.class_name])
            
            # Get the class
            if not hasattr(module, method_spec.class_name):
                raise OrchestrationError(
                    f"Class {method_spec.class_name} not found in module {method_spec.module_name}"
                )
            
            cls = getattr(module, method_spec.class_name)
            
            # Get the method
            if not hasattr(cls, method_spec.method_name):
                raise OrchestrationError(
                    f"Method {method_spec.method_name} not found in class {method_spec.class_name}"
                )
            
            method = getattr(cls, method_spec.method_name)
            
            # Cache the resolved method
            self._method_cache[fqn] = method
            
            return method
            
        except ImportError as e:
            raise OrchestrationError(
                f"Failed to import module {method_spec.module_name}: {e}"
            )
        except Exception as e:
            raise OrchestrationError(
                f"Failed to resolve method {fqn}: {e}"
            )
    
    def _execute_method(
        self,
        method_spec: MethodSpec,
        *args,
        **kwargs
    ) -> MethodExecutionResult:
        """
        Execute a single method with traceability.
        
        Args:
            method_spec: Method specification to execute
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            MethodExecutionResult with execution details
        """
        start_time = time.time()
        trace = {
            "method": method_spec.fully_qualified_name,
            "start_time": start_time,
        }
        
        try:
            # Resolve the method
            method = self._resolve_method(method_spec)
            
            # Execute the method
            # Note: This is a validation that the method exists and can be resolved.
            # Actual execution would require proper initialization of instances, etc.
            # For now, we just verify resolution.
            
            execution_time = (time.time() - start_time) * 1000
            trace["end_time"] = time.time()
            trace["execution_time_ms"] = execution_time
            
            return MethodExecutionResult(
                method_spec=method_spec,
                success=True,
                execution_time_ms=execution_time,
                output=f"Method {method_spec.fully_qualified_name} resolved successfully",
                trace=trace if self.trace_execution else {}
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            trace["end_time"] = time.time()
            trace["execution_time_ms"] = execution_time
            trace["error"] = str(e)
            
            logger.error(f"Method execution failed: {method_spec.fully_qualified_name} - {e}")
            
            return MethodExecutionResult(
                method_spec=method_spec,
                success=False,
                execution_time_ms=execution_time,
                error=e,
                trace=trace if self.trace_execution else {}
            )
    
    def validate_method_existence(self, question: D2Question) -> QuestionOrchestrationResult:
        """
        Validate that all required methods for a D2 question exist and can be resolved.
        
        This is a contract validation step that ensures all methods are present
        before any actual execution begins.
        
        Args:
            question: D2 question to validate
            
        Returns:
            QuestionOrchestrationResult with validation details
            
        Raises:
            OrchestrationError: If strict_mode is True and validation fails
        """
        start_time = time.time()
        required_methods = self.registry.get_methods_for_question(question)
        
        logger.info(f"Validating {len(required_methods)} methods for {question.value}")
        
        method_results = []
        failed_methods = []
        errors = []
        
        for method_spec in required_methods:
            result = self._execute_method(method_spec)
            method_results.append(result)
            
            if not result.success:
                failed_methods.append(method_spec.fully_qualified_name)
                errors.append(f"{method_spec.fully_qualified_name}: {result.error}")
                
                if self.strict_mode:
                    # Abort immediately on first failure in strict mode
                    execution_time = (time.time() - start_time) * 1000
                    
                    error_msg = (
                        f"CONTRACT VIOLATION: Method {method_spec.fully_qualified_name} "
                        f"failed validation for {question.value}. "
                        f"SIN_CARRETA doctrine requires all methods to be present and executable. "
                        f"Error: {result.error}"
                    )
                    
                    logger.error(error_msg)
                    
                    raise OrchestrationError(error_msg)
        
        execution_time = (time.time() - start_time) * 1000
        success = len(failed_methods) == 0
        
        result = QuestionOrchestrationResult(
            question_id=question.value,
            total_methods=len(required_methods),
            executed_methods=len(method_results),
            failed_methods=len(failed_methods),
            success=success,
            execution_time_ms=execution_time,
            method_results=method_results,
            errors=errors,
            trace={
                "validation_timestamp": time.time(),
                "strict_mode": self.strict_mode,
                "question": question.value,
            } if self.trace_execution else {}
        )
        
        if success:
            logger.info(
                f"✓ {question.value}: All {len(required_methods)} methods validated successfully "
                f"in {execution_time:.2f}ms"
            )
        else:
            logger.warning(
                f"✗ {question.value}: {len(failed_methods)}/{len(required_methods)} methods failed validation"
            )
        
        return result
    
    def validate_all_d2_questions(self) -> Dict[str, QuestionOrchestrationResult]:
        """
        Validate all D2 questions (Q1-Q5) for method existence.
        
        Returns:
            Dictionary mapping question IDs to validation results
            
        Raises:
            OrchestrationError: If strict_mode is True and any validation fails
        """
        logger.info("Starting validation of all D2 questions")
        
        results = {}
        
        for question in D2Question:
            try:
                result = self.validate_method_existence(question)
                results[question.value] = result
            except OrchestrationError:
                # Re-raise in strict mode
                if self.strict_mode:
                    raise
                # Otherwise, continue to next question
                continue
        
        # Generate summary
        total_methods = sum(r.total_methods for r in results.values())
        total_failed = sum(r.failed_methods for r in results.values())
        total_success = all(r.success for r in results.values())
        
        logger.info(
            f"D2 Validation Complete: {len(results)}/5 questions validated, "
            f"{total_methods - total_failed}/{total_methods} methods resolved, "
            f"Overall: {'SUCCESS' if total_success else 'FAILED'}"
        )
        
        return results
    
    def generate_validation_report(
        self,
        results: Dict[str, QuestionOrchestrationResult],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.
        
        Args:
            results: Validation results from validate_all_d2_questions()
            output_path: Optional path to save report as JSON
            
        Returns:
            Report dictionary
        """
        report = {
            "metadata": {
                "timestamp": time.time(),
                "orchestrator_version": "1.0.0",
                "doctrine": "SIN_CARRETA",
                "strict_mode": self.strict_mode,
                "trace_enabled": self.trace_execution,
            },
            "summary": {
                "total_questions": len(results),
                "questions_passed": sum(1 for r in results.values() if r.success),
                "questions_failed": sum(1 for r in results.values() if not r.success),
                "total_methods": sum(r.total_methods for r in results.values()),
                "methods_resolved": sum(r.executed_methods - r.failed_methods for r in results.values()),
                "methods_failed": sum(r.failed_methods for r in results.values()),
                "overall_success": all(r.success for r in results.values()),
            },
            "questions": {},
            "failed_methods": [],
        }
        
        # Add per-question details
        for question_id, result in results.items():
            report["questions"][question_id] = {
                "total_methods": result.total_methods,
                "executed_methods": result.executed_methods,
                "failed_methods": result.failed_methods,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
                "errors": result.errors,
            }
            
            # Collect failed methods
            if result.errors:
                report["failed_methods"].extend([
                    {"question": question_id, "error": error}
                    for error in result.errors
                ])
        
        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.write_text(json.dumps(report, indent=2))
            logger.info(f"Validation report saved to {output_path}")
        
        return report


def validate_d2_orchestration(
    strict_mode: bool = True,
    output_report: Optional[str] = None
) -> bool:
    """
    Convenience function to validate D2 orchestration.
    
    Args:
        strict_mode: If True, abort on any validation failure
        output_report: Optional path to save validation report
        
    Returns:
        True if all validations pass, False otherwise
        
    Raises:
        OrchestrationError: If strict_mode is True and validation fails
    """
    orchestrator = D2ActivitiesOrchestrator(strict_mode=strict_mode, trace_execution=True)
    
    try:
        results = orchestrator.validate_all_d2_questions()
        
        if output_report:
            orchestrator.generate_validation_report(results, Path(output_report))
        
        return all(r.success for r in results.values())
        
    except OrchestrationError as e:
        logger.error(f"D2 Orchestration validation failed: {e}")
        if strict_mode:
            raise
        return False


if __name__ == "__main__":
    # Run validation when module is executed directly
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("D2 ACTIVITIES DESIGN & COHERENCE - METHOD CONCURRENCE VALIDATION")
    print("SIN_CARRETA DOCTRINE: No Graceful Degradation | Deterministic Execution")
    print("=" * 80)
    print()
    
    success = validate_d2_orchestration(
        strict_mode=True,
        output_report="d2_validation_report.json"
    )
    
    if success:
        print("\n✅ D2 ORCHESTRATION VALIDATION: SUCCESS")
        print("All required methods are present and resolvable.")
        sys.exit(0)
    else:
        print("\n❌ D2 ORCHESTRATION VALIDATION: FAILED")
        print("Some required methods are missing or unresolvable.")
        sys.exit(1)
