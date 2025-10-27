#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator - CHESS Strategy Control Plane (MACRO Level)
=========================================================================

ARCHITECTURE ALIGNMENT WITH YAML SPECIFICATION v1.0 (2025-10-23)
-----------------------------------------------------------------
This orchestrator implements the Control Plane (MACRO Level) coordination engine
complementing the policy_analysis_architecture.yaml specification.

ROLE: PolicyAnalysisOrchestrator (Control Plane - MACRO Level)
- Manages Choreographer instances for all 300 questions
- Implements CHESS strategy: Opening (7 producers parallel) → Middle Game (6 modalities) → Endgame (synthesis)
- Aggregates MICRO → MESO → MACRO results
- Generates comprehensive final report
- Does NOT contain YAML specialized components (delegates to Choreographer)

YAML ALIGNMENT & ARCHITECTURE DISTINCTION:
------------------------------------------

*** CRITICAL ARCHITECTURAL CLARIFICATION ***

The YAML specification documents the DATA PLANE (MICRO level) components that
are executed by the ExecutionChoreographer (policy_analysis_pipeline.py).

This Orchestrator is the CONTROL PLANE (MACRO level) that:
✓ Coordinates 300 questions across clusters
✓ Implements CHESS strategy (Opening/Middle/Endgame)
✓ Aggregates MICRO → MESO → MACRO
✓ Generates comprehensive final report
✓ Does NOT directly use YAML components (IndustrialPolicyProcessor, etc.)
✓ DELEGATES to Choreographer which contains all YAML components

YAML COMPONENTS (in Choreographer, not Orchestrator):
- IndustrialPolicyProcessor
- PolicyContradictionDetector  
- TemporalLogicVerifier
- BayesianConfidenceCalculator
- TeoriaCambio
- AdvancedDAGValidator
- MunicipalAnalyzer
- MunicipalOntology
- SemanticAnalyzer
- PerformanceAnalyzer
- PDETMunicipalPlanAnalyzer

ORCHESTRATOR COMPONENTS:
- PolicyAnalysisOrchestrator (this class)
- ExecutionChoreographer (imported from policy_analysis_pipeline)
- ReportAssembler (imported from report_assembly)

ORCHESTRATOR (Control Plane - MACRO Level):
- Manages Choreographer instances for all 300 questions
- Implements CHESS strategy: Opening (7 producers parallel) → Middle Game (6 modalities) → Endgame (synthesis)
- Aggregates MICRO → MESO → MACRO results
- Generates comprehensive final report
- Enforces Two-Core Data Flow: Generation Pipeline + Collection & Assembly Pipeline

CHESS STRATEGY IMPLEMENTATION:
✓ Opening: Parallel execution of 7 producers (dereck_beach, policy_processor, etc.)
✓ Middle Game: 6 scoring modalities (TYPE_A through TYPE_F)
✓ Endgame: Convergence synthesis with Decálogo framework alignment

GOLDEN RULES COMPLIANCE:
✓ Rule 1: Immutable Declarative Configuration (cuestionario_FIXED.json as canonical truth)
✓ Rule 2: Atomic Context Hydration (load all 300 questions before execution)
✓ Rule 3: Deterministic Pipeline Execution (sequential processing with consistent ordering)
✓ Rule 5: Absolute Processing Homogeneity (same pipeline for all questions)
✓ Rule 6: Data Provenance and Lineage (complete traceability)
✓ Rule 10: SOTA Architectural Principles

TWO-CORE DATA FLOW:
1. Generation Pipeline (Producers): 7 producers generate evidence
2. Collection & Assembly Pipeline (Assembler): Single assembler creates answer_bundles

RELATIONSHIP TO YAML:
---------------------
The YAML documents the 95% utilization architecture at the MICRO level.
This Orchestrator operates at the MACRO level and delegates MICRO execution
to the Choreographer, which implements all YAML components and dimensions.

Think of it as:
┌─────────────────────────────────────┐
│  Orchestrator (MACRO - this file)  │ ← Strategy, aggregation, reporting
│  - PolicyAnalysisOrchestrator       │
│  - CHESS strategy execution         │
│  - MESO/MACRO convergence           │
└─────────────────────────────────────┘
              ↓ delegates to
┌─────────────────────────────────────┐
│  Choreographer (MICRO - pipeline)   │ ← YAML components implementation
│  - ExecutionChoreographer           │
│  - 11 YAML specialized components   │
│  - D1-D6 dimensional execution      │
└─────────────────────────────────────┘

Author: Integration Team
Version: 1.1.0 - YAML v1.0 Aligned (2025-10-23)
Python: 3.10+
"""

import json
import hashlib
import logging
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import statistics

# Import Choreographer and Report Assembly
# NOTE: ExecutionChoreographer contains ALL YAML components
from choreographer import ExecutionChoreographer, ExecutionResult, ExecutionContext
from determinism.seeds import DeterministicContext, SeedFactory
from report_assembly import (
    ReportAssembler, MicroLevelAnswer, MesoLevelCluster, MacroLevelConvergence
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QuestionSpec:
    """Question specification from cuestionario_FIXED.json (canonical truth)"""
    canonical_id: str  # P#-D#-Q# format
    policy_area: str  # P1-P10
    dimension: str  # D1-D6
    question_number: int  # Q#
    question_text: str
    scoring_modality: str  # TYPE_A through TYPE_F
    cluster_id: str = ""
    cluster_name: str = ""
    expected_elements: List[str] = field(default_factory=list)
    search_patterns: Dict[str, Any] = field(default_factory=dict)
    element_weights: Dict[str, float] = field(default_factory=dict)
    numerical_thresholds: Dict[str, float] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterDefinition:
    """Cluster definition for MESO-level aggregation"""
    cluster_id: str  # CLUSTER_1, CLUSTER_2, etc.
    cluster_name: str
    description: str
    policy_areas: List[str]  # P1, P2, etc.
    dimensions: List[str]  # D1, D2, etc.
    question_ids: List[str]  # List of P#-D#-Q# IDs
    policy_weights: Dict[str, float] = field(default_factory=dict)
    macro_weight: float = 0.0


@dataclass
class OrchestratorConfig:
    """Configuration for Orchestrator execution"""
    questionnaire_path: str = "cuestionario_FIXED.json"
    rubric_path: str = "rubric_scoring_FIXED.json"
    plan_document_path: str = ""
    execution_mapping_path: str = "execution_mapping.yaml"
    method_class_map_path: str = "COMPLETE_METHOD_CLASS_MAP.json"
    output_directory: str = "output"
    enable_parallel_execution: bool = False
    max_workers: int = 4
    enable_meso_clustering: bool = True
    enable_macro_convergence: bool = True
    run_id: str = "run-default"


@dataclass
class ExecutionPlan:
    """Complete execution plan for CHESS strategy"""
    opening_questions: List[QuestionSpec]  # All 300 questions
    middle_game_modalities: Dict[str, List[QuestionSpec]]  # Grouped by scoring modality
    endgame_clusters: List[ClusterDefinition]  # MESO clusters
    total_questions: int
    estimated_duration: float


@dataclass
class OrchestratorResult:
    """Final orchestrator result with complete CHESS execution"""
    execution_id: str
    timestamp: str
    micro_results: Dict[str, MicroLevelAnswer]  # question_id -> answer
    meso_results: Dict[str, MesoLevelCluster]  # cluster_id -> cluster
    macro_result: MacroLevelConvergence
    execution_statistics: Dict[str, Any]
    performance_metrics: Dict[str, float]
    provenance: Dict[str, Any]


# ============================================================================
# ORCHESTRATOR ENGINE
# ============================================================================

class PolicyAnalysisOrchestrator:
    """
    Control Plane orchestrator implementing CHESS strategy for complete PDM analysis.

    *** MACRO LEVEL: Does NOT contain YAML specialized components ***

    This orchestrator operates at the MACRO level and delegates MICRO-level
    execution to ExecutionChoreographer, which contains ALL YAML components:
    - IndustrialPolicyProcessor
    - PolicyContradictionDetector
    - TemporalLogicVerifier
    - BayesianConfidenceCalculator
    - TeoriaCambio + AdvancedDAGValidator
    - MunicipalAnalyzer + Ontology + SemanticAnalyzer + PerformanceAnalyzer
    - PDETMunicipalPlanAnalyzer

    CHESS STRATEGY:
    - Opening: Parallel execution of 7 producers across all 300 questions
    - Middle Game: Scoring with 6 modalities (TYPE_A through TYPE_F)
    - Endgame: MICRO → MESO → MACRO convergence synthesis

    TWO-CORE DATA FLOW:
    - Generation Pipeline: 7 producers (dereck_beach, policy_processor, etc.)
    - Collection & Assembly Pipeline: report_assembly (single assembler)

    ARCHITECTURAL LAYER:
    This is the CONTROL PLANE that coordinates strategy.
    The DATA PLANE (with YAML components) is in ExecutionChoreographer.
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize Orchestrator with canonical truth model (Golden Rule 1)
        
        Args:
            config: Orchestrator configuration
        """
        logger.info("=" * 80)
        logger.info("ORCHESTRATOR INITIALIZATION - CHESS STRATEGY")
        logger.info("YAML v1.0 Alignment (2025-10-23) - MACRO Level")
        logger.info("=" * 80)
        
        self.config = config
        self.start_time = time.time()

        # Golden Rule 1: Load canonical truth model (cuestionario_FIXED.json)
        self.questionnaire = self._load_questionnaire()
        self.questionnaire_hash = self._compute_questionnaire_hash()

        # Load rubric configuration for weighting rules
        self.rubric = self._load_rubric()

        # Build canonical cluster catalog (CL01-CL04)
        (
            self.cluster_catalog,
            self.cluster_weights,
            self.cluster_policy_weights,
        ) = self._build_cluster_catalog()
        self.policy_area_to_cluster = {
            policy_area: cluster_id
            for cluster_id, info in self.cluster_catalog.items()
            for policy_area in info.get("policy_areas", [])
        }

        # Configure causal thresholds (dimension-specific per YAML)
        self.causal_thresholds = self._build_causal_thresholds()

        # Deterministic context configuration
        self.seed_factory = SeedFactory()
        run_identifier = config.run_id or "run-default"
        self.deterministic_context = DeterministicContext.from_factory(
            self.seed_factory,
            self.questionnaire_hash,
            run_identifier
        )
        self.run_seed = self.deterministic_context.seed
        logger.info(f"✓ Deterministic seed configured: {self.run_seed}")

        # Initialize Choreographer (Data Plane)
        # *** THIS IS WHERE ALL YAML COMPONENTS ARE INITIALIZED ***
        # The Choreographer contains:
        # - IndustrialPolicyProcessor
        # - PolicyContradictionDetector
        # - TemporalLogicVerifier
        # - BayesianConfidenceCalculator
        # - TeoriaCambio + AdvancedDAGValidator
        # - MunicipalAnalyzer + Ontology + SemanticAnalyzer + PerformanceAnalyzer
        # - PDETMunicipalPlanAnalyzer
        # - All other YAML components
        self.choreographer = ExecutionChoreographer(
            execution_mapping_path=config.execution_mapping_path,
            method_class_map_path=config.method_class_map_path,
            questionnaire_hash=self.questionnaire_hash,
            deterministic_context=self.deterministic_context
        )
        logger.info("✓ Choreographer initialized with ALL YAML components")

        # Initialize Report Assembler (Collection & Assembly Pipeline)
        self.report_assembler = ReportAssembler(
            cluster_weights=self.cluster_weights,
            cluster_policy_weights=self.cluster_policy_weights,
            causal_thresholds=self.causal_thresholds
        )

        # Parse all questions into structured specs (Golden Rule 2: Atomic Context Hydration)
        self.all_questions = self._parse_all_questions()

        # Define MESO clusters
        self.clusters = self._define_clusters()
        
        # Execution statistics
        self.stats = {
            "total_questions": len(self.all_questions),
            "questions_executed": 0,
            "questions_succeeded": 0,
            "questions_failed": 0,
            "meso_clusters": len(self.clusters),
            "execution_time": 0.0
        }
        
        logger.info(f"✓ Orchestrator initialized in {time.time() - self.start_time:.2f}s")
        logger.info(f"✓ Loaded {len(self.all_questions)} questions from canonical truth model")
        logger.info(f"✓ Defined {len(self.clusters)} MESO clusters")
        logger.info(f"✓ Choreographer contains 11/11 YAML components")
        logger.info("=" * 80)

    # ========================================================================
    # GOLDEN RULE 1: CANONICAL TRUTH MODEL LOADING
    # ========================================================================

    def _load_questionnaire(self) -> Dict[str, Any]:
        """
        Load cuestionario_FIXED.json (canonical truth model)

        This is the authoritative source for all questions, scoring logic, and evidence definitions.
        """
        logger.info(f"Loading canonical truth model: {self.config.questionnaire_path}")

        try:
            with open(self.config.questionnaire_path, 'r', encoding='utf-8') as f:
                questionnaire = json.load(f)

            question_key = "questions" if "questions" in questionnaire else "preguntas_base"
            total_questions = len(questionnaire.get(question_key, []))
            logger.info(f"✓ Loaded {total_questions} questions from canonical truth model")

            return questionnaire

        except FileNotFoundError:
            logger.error(f"Canonical truth model not found: {self.config.questionnaire_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load questionnaire: {e}")
            raise

    def _load_rubric(self) -> Dict[str, Any]:
        """
        Load rubric_scoring configuration for weighting and aggregation.

        Includes YAML causal_thresholds and dimension-specific scoring rules.
        """
        rubric_path = self.config.rubric_path
        logger.info(f"Loading rubric configuration: {rubric_path}")

        try:
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric = json.load(f)

            return rubric

        except FileNotFoundError:
            logger.warning(f"Rubric configuration not found at {rubric_path}; using defaults")
            return {}
        except Exception as e:
            logger.error(f"Failed to load rubric configuration: {e}")
            raise

    def _build_cluster_catalog(self) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Combine questionnaire metadata and rubric rules into canonical cluster catalog.

        Creates MESO-level clusters (CL01-CL04) as documented in YAML.
        """

        metadata_clusters = (
            self.questionnaire.get("metadata", {}).get("clusters", [])
            if isinstance(self.questionnaire.get("metadata"), dict)
            else []
        )
        rubric_meso = self.rubric.get("meso_clusters", {}) if isinstance(self.rubric, dict) else {}
        rubric_macro_weights = (
            self.rubric.get("aggregation_levels", {})
            .get("level_4", {})
            .get("cluster_weights", {})
            if isinstance(self.rubric, dict)
            else {}
        )

        cluster_catalog: Dict[str, Dict[str, Any]] = {}

        # Prefer questionnaire metadata for cluster definitions, fallback to rubric
        source_clusters = metadata_clusters or [
            {
                "cluster_id": cluster_id,
                "name": data.get("name", cluster_id),
                "rationale": data.get("description", ""),
                "policy_area_ids": data.get("policy_areas", []),
            }
            for cluster_id, data in rubric_meso.items()
        ]

        for cluster_data in source_clusters:
            cluster_id = cluster_data.get("cluster_id")
            if not cluster_id:
                continue

            rubric_entry = rubric_meso.get(cluster_id, {})
            cluster_catalog[cluster_id] = {
                "cluster_id": cluster_id,
                "name": cluster_data.get("name", cluster_id),
                "description": cluster_data.get("rationale", rubric_entry.get("description", "")),
                "policy_areas": cluster_data.get("policy_area_ids", rubric_entry.get("policy_areas", [])),
                "policy_weights": rubric_entry.get("weights", {}),
                "macro_weight": rubric_macro_weights.get(cluster_id, 0.0),
                "imbalance_threshold": rubric_entry.get("imbalance_threshold"),
            }

        if not cluster_catalog:
            logger.warning("No cluster definitions found; defaulting to empty catalog")

        # Ensure canonical ordering CL01-CL04 if available
        ordered_catalog = {
            cluster_id: cluster_catalog[cluster_id]
            for cluster_id in sorted(cluster_catalog.keys())
        }

        cluster_weights = {
            cluster_id: info.get("macro_weight", 0.0)
            for cluster_id, info in ordered_catalog.items()
        }

        cluster_policy_weights = {
            cluster_id: info.get("policy_weights", {})
            for cluster_id, info in ordered_catalog.items()
        }

        logger.info(
            "✓ Loaded cluster catalog: %s",
            ", ".join(f"{cid}({len(info.get('policy_areas', []))} PAs)" for cid, info in ordered_catalog.items())
        )

        return ordered_catalog, cluster_weights, cluster_policy_weights

    def _build_causal_thresholds(self) -> Dict[str, float]:
        """
        Create dimension-specific causal thresholds for correction logic.

        *** YAML ALIGNMENT: causal_thresholds section ***

        Default thresholds per dimension:
        - D1: 0.50 (Diagnóstico)
        - D2: 0.60 (Actividades)
        - D3: 0.60 (Productos)
        - D4: 0.65 (Resultados)
        - D5: 0.70 (Impactos)
        - D6: 0.75 (Causalidad - highest threshold due to complexity)
        """

        defaults = {
            "default": 0.6,
            "D2": 0.6,
            "D3": 0.6,
            "D4": 0.65,
            "D5": 0.7,
            "D6": 0.75,
        }

        rubric_thresholds = {}
        if isinstance(self.rubric, dict):
            rubric_thresholds = self.rubric.get("causal_thresholds", {}) or {}

        defaults.update({k: float(v) for k, v in rubric_thresholds.items() if isinstance(v, (int, float))})

        return defaults

    def _compute_questionnaire_hash(self) -> str:
        """Compute stable hash of the loaded questionnaire metadata."""

        canonical = json.dumps(self.questionnaire, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _parse_all_questions(self) -> List[QuestionSpec]:
        """
        Parse all questions into structured QuestionSpec objects (Golden Rule 2)

        *** Atomic Context Hydration: loads ALL question metadata before execution ***

        Each question maps to a specific dimension (D1-D6) and will be executed
        by the Choreographer using the appropriate YAML component chain.
        """
        logger.info("Parsing all questions into structured specs...")

        questions = []

        question_entries = []

        if "questions" in self.questionnaire:
            question_entries = self.questionnaire.get("questions", [])
        elif "preguntas_base" in self.questionnaire:
            question_entries = self.questionnaire.get("preguntas_base", [])

        for q_data in question_entries:
            try:
                # Extract P#-D#-Q# components
                canonical_id = q_data.get("question_unique_id") or q_data.get("id", "")
                parts = canonical_id.split("-")

                policy_area = (
                    q_data.get("policy_area")
                    or q_data.get("metadata", {}).get("policy_area")
                    or (parts[0] if len(parts) > 0 else "P0")
                )
                dimension = q_data.get("dimension") or (parts[1] if len(parts) > 1 else "D0")
                question_num = (
                    q_data.get("question_number")
                    or q_data.get("numero")
                    or (int(parts[2].replace("Q", "")) if len(parts) > 2 and parts[2].startswith("Q") else 0)
                )

                cluster_id = self.policy_area_to_cluster.get(policy_area, "")
                cluster_name = self.cluster_catalog.get(cluster_id, {}).get("name", cluster_id)

                question_text = q_data.get("question_text") or q_data.get("texto_template", "")

                scoring_modality = (
                    q_data.get("scoring_modality")
                    or q_data.get("metadata", {}).get("scoring_modality")
                    or self.questionnaire.get("scoring_system", {}).get("default_modality")
                    or "TYPE_F"
                )

                expected_elements = q_data.get("expected_elements")
                if expected_elements is None:
                    patterns = q_data.get("patrones_verificacion", [])
                    if isinstance(patterns, list):
                        expected_elements = patterns
                    else:
                        expected_elements = []

                search_patterns = q_data.get("search_patterns")
                if search_patterns is None:
                    patterns = q_data.get("patrones_verificacion", [])
                    if isinstance(patterns, list):
                        search_patterns = {
                            f"pattern_{idx:03d}": pattern
                            for idx, pattern in enumerate(patterns)
                        }
                    else:
                        search_patterns = {}

                # Create QuestionSpec
                spec = QuestionSpec(
                    canonical_id=canonical_id,
                    policy_area=policy_area,
                    dimension=dimension,
                    question_number=question_num,
                    question_text=question_text,
                    scoring_modality=scoring_modality,
                    cluster_id=cluster_id,
                    cluster_name=cluster_name,
                    expected_elements=expected_elements or [],
                    search_patterns=search_patterns,
                    element_weights=q_data.get("element_weights", {}),
                    numerical_thresholds=q_data.get("numerical_thresholds", {}),
                    validation_rules=q_data.get("validation_rules", {}),
                    metadata={
                        "hints": q_data.get("hints", []),
                        "point_code": q_data.get("point_code", q_data.get("metadata", {}).get("original_id", "")),
                        "point_title": q_data.get("point_title", ""),
                        "cluster_id": cluster_id,
                        "cluster_name": cluster_name,
                    }
                )

                questions.append(spec)

            except Exception as e:
                logger.warning(f"Failed to parse question {q_data.get('question_unique_id', 'UNKNOWN')}: {e}")
                continue
        
        logger.info(f"✓ Parsed {len(questions)} question specifications")
        return questions

    def _define_clusters(self) -> List[ClusterDefinition]:
        """
        Define MESO-level clusters for aggregation

        *** YAML MESO-Level Clustering ***

        Clusters group related questions across policy areas for mid-level analysis.
        Each cluster spans multiple dimensions (D1-D6) and policy areas (P1-P10).
        """
        clusters: List[ClusterDefinition] = []

        for cluster_id, info in self.cluster_catalog.items():
            question_ids = [
                q.canonical_id
                for q in self.all_questions
                if q.cluster_id == cluster_id
            ]

            dimensions = sorted({
                q.dimension
                for q in self.all_questions
                if q.cluster_id == cluster_id
            })

            cluster = ClusterDefinition(
                cluster_id=cluster_id,
                cluster_name=info.get("name", cluster_id),
                description=info.get("description", ""),
                policy_areas=info.get("policy_areas", []),
                dimensions=dimensions or ["D1", "D2", "D3", "D4", "D5", "D6"],
                question_ids=question_ids,
                policy_weights=info.get("policy_weights", {}),
                macro_weight=info.get("macro_weight", 0.0)
            )

            clusters.append(cluster)

        logger.info(f"✓ Defined {len(clusters)} canonical MESO clusters (expected 4)")
        return clusters

    # ========================================================================
    # CHESS STRATEGY EXECUTION
    # ========================================================================

    def execute_chess_strategy(
        self,
        plan_document: str,
        plan_metadata: Dict[str, Any]
    ) -> OrchestratorResult:
        """
        Execute complete CHESS strategy: Opening → Middle Game → Endgame

        *** YAML EXECUTION FLOW ***

        CHESS PHASES:
        1. Opening: Execute all 300 questions via Choreographer
           - Choreographer uses ALL YAML components:
             * IndustrialPolicyProcessor for pattern matching
             * PolicyContradictionDetector for contradictions
             * TemporalLogicVerifier for temporal consistency
             * BayesianConfidenceCalculator for confidence
             * TeoriaCambio + AdvancedDAGValidator for causal validation
             * MunicipalAnalyzer + SemanticAnalyzer for semantic analysis
             * PerformanceAnalyzer for loss function
             * PDETMunicipalPlanAnalyzer for tabular analysis
           - Each question executes dimension-specific chain (D1-D6)

        2. Middle Game: Scoring with 6 modalities (TYPE_A-F)
           - Aggregate results by scoring modality

        3. Endgame: MICRO → MESO → MACRO convergence
           - MESO: Cluster-level aggregation
           - MACRO: Overall convergence with Decálogo framework

        Args:
            plan_document: Full plan document text
            plan_metadata: Document metadata

        Returns:
            OrchestratorResult with complete CHESS execution
        """
        execution_start = time.time()
        execution_id = self._generate_execution_id()
        
        logger.info("=" * 80)
        logger.info("CHESS STRATEGY EXECUTION - OPENING → MIDDLE GAME → ENDGAME")
        logger.info("=" * 80)
        
        # ====================================================================
        # OPENING: Execute all questions with Choreographer (MICRO level)
        # *** This is where YAML components are invoked ***
        # ====================================================================
        logger.info("\n🎯 CHESS OPENING: Executing all questions (MICRO level)")
        logger.info("-" * 80)
        logger.info("NOTE: Choreographer executing with ALL YAML components:")
        logger.info("  ✓ IndustrialPolicyProcessor")
        logger.info("  ✓ PolicyContradictionDetector")
        logger.info("  ✓ TemporalLogicVerifier")
        logger.info("  ✓ BayesianConfidenceCalculator")
        logger.info("  ✓ TeoriaCambio + AdvancedDAGValidator")
        logger.info("  ✓ MunicipalAnalyzer + Ontology + SemanticAnalyzer + PerformanceAnalyzer")
        logger.info("  ✓ PDETMunicipalPlanAnalyzer")
        logger.info("-" * 80)
        
        micro_results = self._execute_opening(plan_document, plan_metadata)
        
        logger.info(f"✓ Opening completed: {len(micro_results)} questions executed")
        logger.info(f"✓ Success rate: {self.stats['questions_succeeded']}/{self.stats['total_questions']}")
        
        # ====================================================================
        # MIDDLE GAME: Aggregate by scoring modalities
        # ====================================================================
        logger.info("\n♟️  CHESS MIDDLE GAME: Analyzing scoring modalities")
        logger.info("-" * 80)
        
        modality_analysis = self._execute_middle_game(micro_results)
        
        logger.info(f"✓ Middle game completed: {len(modality_analysis)} modalities analyzed")
        
        # ====================================================================
        # ENDGAME: MICRO → MESO → MACRO convergence
        # ====================================================================
        logger.info("\n👑 CHESS ENDGAME: MICRO → MESO → MACRO convergence")
        logger.info("-" * 80)
        
        # MESO: Cluster-level aggregation
        meso_results = self._generate_meso_clusters(micro_results)
        
        logger.info(f"✓ MESO aggregation: {len(meso_results)} clusters generated")
        
        # MACRO: Overall convergence with Decálogo framework
        macro_result = self._generate_macro_convergence(micro_results, meso_results)
        
        logger.info(f"✓ MACRO convergence: Overall score {macro_result.overall_score:.1f}/100")
        
        # ====================================================================
        # Compile final result
        # ====================================================================
        self.stats["execution_time"] = time.time() - execution_start
        
        timestamp_iso = datetime.now().isoformat()

        result = OrchestratorResult(
            execution_id=execution_id,
            timestamp=timestamp_iso,
            micro_results=micro_results,
            meso_results=meso_results,
            macro_result=macro_result,
            execution_statistics=self.stats.copy(),
            performance_metrics={
                "total_execution_time": self.stats["execution_time"],
                "average_time_per_question": self.stats["execution_time"] / max(len(micro_results), 1),
                # NOTE: "avg_question_time" is retained for backward compatibility with legacy consumers (e.g., PolicyDashboard v2, DataSyncService).
                #       Remove this field after all downstream systems migrate to "average_time_per_question" (target: Q1 2026).
                "avg_question_time": self.stats["execution_time"] / max(len(micro_results), 1),
                "questions_per_second": len(micro_results) / max(self.stats["execution_time"], 0.001),
                "choreographer_stats": self.choreographer.get_statistics()
            },
            provenance={
                "questionnaire_hash": self.questionnaire_hash,
                "run_seed": self.run_seed,
                "execution_id": execution_id,
                "timestamp": timestamp_iso,
                "yaml_version": "1.0",
                "yaml_date": "2025-10-23",
                "questionnaire_path": self.config.questionnaire_path,
                "execution_mapping_path": self.config.execution_mapping_path,
                "method_class_map_path": self.config.method_class_map_path,
                "plan_metadata": plan_metadata
            }
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("CHESS EXECUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Execution time: {self.stats['execution_time']:.2f}s")
        logger.info(f"Overall score: {macro_result.overall_score:.1f}/100")
        logger.info(f"Classification: {macro_result.plan_classification}")
        logger.info("=" * 80)

        return result

    def _execute_opening(
        self,
        plan_document: str,
        plan_metadata: Dict[str, Any]
    ) -> Dict[str, MicroLevelAnswer]:
        """
        Execute Opening phase: All questions via Choreographer

        *** YAML COMPONENT EXECUTION HAPPENS HERE ***

        For each question (P#-D#-Q#):
        1. Create ExecutionContext with dimension info
        2. Delegate to choreographer.execute_question()
        3. Choreographer routes to dimension-specific chain (D1-D6)
        4. YAML components execute as per their chains
        5. Return MicroLevelAnswer

        Example D1-Q3 execution chain (via Choreographer):
        - IndustrialPolicyProcessor.process
        - PolicyContradictionDetector._extract_resource_mentions
        - PolicyContradictionDetector._detect_numerical_inconsistencies
        - PolicyContradictionDetector._detect_resource_conflicts
        - BayesianConfidenceCalculator.calculate_posterior

        Returns:
            Dict mapping question_id to MicroLevelAnswer
        """
        micro_results = {}
        
        total_questions = len(self.all_questions)
        
        for idx, question_spec in enumerate(self.all_questions, 1):
            logger.info(f"  [{idx}/{total_questions}] Executing {question_spec.canonical_id}")
            
            try:
                # Convert QuestionSpec to dict for Choreographer
                question_dict = {
                    "canonical_id": question_spec.canonical_id,
                    "policy_area": question_spec.policy_area,
                    "dimension": question_spec.dimension,
                    "question_text": question_spec.question_text,
                    "scoring_modality": question_spec.scoring_modality,
                    "expected_elements": question_spec.expected_elements,
                    "search_patterns": question_spec.search_patterns,
                    "element_weights": question_spec.element_weights,
                    "numerical_thresholds": question_spec.numerical_thresholds,
                    "validation_rules": question_spec.validation_rules
                }
                
                # Execute via Choreographer
                result = self.choreographer.execute_question(
                    question_dict,
                    plan_document,
                    plan_metadata
                )
                
                self.stats["questions_executed"] += 1
                
                if result.status == "success" and result.micro_answer:
                    micro_results[question_spec.canonical_id] = result.micro_answer
                    self.stats["questions_succeeded"] += 1
                    
                    logger.info(f"    ✓ Score: {result.micro_answer.quantitative_score:.2f}, "
                               f"Note: {result.micro_answer.qualitative_note}")
                else:
                    self.stats["questions_failed"] += 1
                    logger.warning(f"    ✗ Execution failed: {result.errors}")
                
            except Exception as e:
                self.stats["questions_failed"] += 1
                logger.error(f"    ✗ Exception: {e}")
                continue
        
        return micro_results

    def _execute_middle_game(
        self,
        micro_results: Dict[str, MicroLevelAnswer]
    ) -> Dict[str, Any]:
        """
        Execute Middle Game: Aggregate by scoring modalities

        Groups results by TYPE_A through TYPE_F for modality-specific analysis
        """
        modality_groups = defaultdict(list)
        
        # Group by modality
        for question_id, answer in micro_results.items():
            modality_groups[answer.scoring_modality].append(answer)
        
        # Analyze each modality
        modality_analysis = {}
        
        for modality, answers in modality_groups.items():
            scores = [a.quantitative_score for a in answers]
            confidences = [a.confidence for a in answers]
            
            analysis = {
                "modality": modality,
                "question_count": len(answers),
                "avg_score": statistics.mean(scores) if scores else 0.0,
                "std_score": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "avg_confidence": statistics.mean(confidences) if confidences else 0.0,
                "score_distribution": {
                    "EXCELENTE": sum(1 for a in answers if a.qualitative_note == "EXCELENTE"),
                    "BUENO": sum(1 for a in answers if a.qualitative_note == "BUENO"),
                    "ACEPTABLE": sum(1 for a in answers if a.qualitative_note == "ACEPTABLE"),
                    "INSUFICIENTE": sum(1 for a in answers if a.qualitative_note == "INSUFICIENTE")
                }
            }
            
            modality_analysis[modality] = analysis
            
            logger.info(f"  {modality}: {len(answers)} questions, avg score {analysis['avg_score']:.2f}")
        
        return modality_analysis

    def _generate_meso_clusters(
        self,
        micro_results: Dict[str, MicroLevelAnswer]
    ) -> Dict[str, MesoLevelCluster]:
        """
        Generate MESO-level clusters

        Aggregates MICRO answers into cluster-level results (CL01-CL04)
        """
        meso_results = {}
        
        for cluster_def in self.clusters:
            logger.info(f"  Generating cluster: {cluster_def.cluster_id}")

            # Get answers for this cluster
            cluster_answers = [
                micro_results[qid] 
                for qid in cluster_def.question_ids 
                if qid in micro_results
            ]
            
            if not cluster_answers:
                logger.warning(f"    No answers found for cluster {cluster_def.cluster_id}")
                continue
            
            # Use ReportAssembler to generate MESO cluster
            meso_cluster = self.report_assembler.generate_meso_cluster(
                cluster_name=cluster_def.cluster_id,
                cluster_description=cluster_def.description or cluster_def.cluster_name,
                micro_answers=cluster_answers,
                cluster_definition={
                    "policy_areas": cluster_def.policy_areas,
                    "dimensions": cluster_def.dimensions,
                    "total_questions": len(cluster_def.question_ids),
                    "policy_weights": cluster_def.policy_weights,
                    "macro_weight": cluster_def.macro_weight
                }
            )

            meso_results[cluster_def.cluster_id] = meso_cluster
            
            logger.info(f"    ✓ Avg score: {meso_cluster.avg_score:.1f}/100, "
                       f"Coverage: {meso_cluster.question_coverage:.1f}%")
        
        return meso_results

    def _generate_macro_convergence(
        self,
        micro_results: Dict[str, MicroLevelAnswer],
        meso_results: Dict[str, MesoLevelCluster]
    ) -> MacroLevelConvergence:
        """
        Generate MACRO-level convergence

        Creates overall assessment with Decálogo framework alignment
        """
        logger.info("  Generating MACRO convergence analysis...")
        
        # Aggregate all MICRO answers
        all_answers = list(micro_results.values())
        
        # Use ReportAssembler to generate MACRO convergence
        macro_convergence = self.report_assembler.generate_macro_convergence(
            micro_answers=all_answers,
            meso_clusters=list(meso_results.values()),
            plan_metadata={
                "total_questions": len(self.all_questions),
                "answered_questions": len(micro_results)
            }
        )
        
        logger.info(f"    ✓ Overall score: {macro_convergence.overall_score:.1f}/100")
        logger.info(f"    ✓ Classification: {macro_convergence.plan_classification}")
        logger.info(f"    ✓ Critical gaps: {len(macro_convergence.critical_gaps)}")
        
        return macro_convergence

    # ========================================================================
    # OUTPUT GENERATION
    # ========================================================================

    def save_results(self, result: OrchestratorResult):
        """
        Save orchestration results to disk

        Generates:
        - MICRO answers JSON (from Choreographer with YAML components)
        - MESO clusters JSON
        - MACRO convergence JSON
        - Executive summary report
        - Provenance trace
        """
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info("=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        
        # 1. MICRO results
        micro_path = output_dir / f"micro_answers_{timestamp}.json"
        with open(micro_path, 'w', encoding='utf-8') as f:
            micro_data = {
                qid: asdict(answer) 
                for qid, answer in result.micro_results.items()
            }
            json.dump(micro_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved MICRO answers: {micro_path}")
        
        # 2. MESO results
        meso_path = output_dir / f"meso_clusters_{timestamp}.json"
        with open(meso_path, 'w', encoding='utf-8') as f:
            meso_data = {
                cid: asdict(cluster)
                for cid, cluster in result.meso_results.items()
            }
            json.dump(meso_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved MESO clusters: {meso_path}")
        
        # 3. MACRO convergence
        macro_path = output_dir / f"macro_convergence_{timestamp}.json"
        with open(macro_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result.macro_result), f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved MACRO convergence: {macro_path}")
        
        # 4. Executive summary
        summary_path = output_dir / f"executive_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_executive_summary(result))
        logger.info(f"✓ Saved executive summary: {summary_path}")
        
        # 5. Complete result
        complete_path = output_dir / f"complete_result_{timestamp}.json"
        with open(complete_path, 'w', encoding='utf-8') as f:
            complete_data = {
                "execution_id": result.execution_id,
                "timestamp": result.timestamp,
                "micro_count": len(result.micro_results),
                "meso_count": len(result.meso_results),
                "macro_score": result.macro_result.overall_score,
                "statistics": result.execution_statistics,
                "performance": result.performance_metrics,
                "provenance": result.provenance
            }
            json.dump(complete_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved complete result: {complete_path}")
        
        logger.info("=" * 80)

    def _generate_executive_summary(self, result: OrchestratorResult) -> str:
        """Generate executive summary report"""
        lines = []
        
        lines.append("=" * 80)
        lines.append("EXECUTIVE SUMMARY - MUNICIPAL DEVELOPMENT PLAN ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        
        lines.append(f"Execution ID: {result.execution_id}")
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append(f"Execution Time: {result.execution_statistics['execution_time']:.2f}s")
        lines.append("")
        
        lines.append("OVERALL ASSESSMENT")
        lines.append("-" * 80)
        lines.append(f"Overall Score: {result.macro_result.overall_score:.1f}/100")
        lines.append(f"Classification: {result.macro_result.plan_classification}")
        lines.append(f"Agenda Alignment: {result.macro_result.agenda_alignment * 100:.1f}%")
        lines.append("")
        
        lines.append("COVERAGE STATISTICS")
        lines.append("-" * 80)
        lines.append(f"Total Questions: {result.execution_statistics['total_questions']}")
        lines.append(f"Questions Executed: {result.execution_statistics['questions_executed']}")
        lines.append(f"Successful: {result.execution_statistics['questions_succeeded']}")
        lines.append(f"Failed: {result.execution_statistics['questions_failed']}")
        lines.append("")
        
        lines.append("DIMENSIONAL ANALYSIS")
        lines.append("-" * 80)
        for dim, score in result.macro_result.convergence_by_dimension.items():
            lines.append(f"{dim}: {score:.1f}/100")
        lines.append("")
        
        lines.append("POLICY AREA ANALYSIS")
        lines.append("-" * 80)
        for policy, score in result.macro_result.convergence_by_policy_area.items():
            lines.append(f"{policy}: {score:.1f}/100")
        lines.append("")
        
        lines.append("CRITICAL GAPS")
        lines.append("-" * 80)
        for idx, gap in enumerate(result.macro_result.critical_gaps[:10], 1):
            lines.append(f"{idx}. {gap}")
        lines.append("")
        
        lines.append("STRATEGIC RECOMMENDATIONS")
        lines.append("-" * 80)
        for idx, rec in enumerate(result.macro_result.strategic_recommendations[:10], 1):
            lines.append(f"{idx}. {rec}")
        lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        import hashlib
        timestamp = datetime.now().isoformat()
        unique_string = f"orchestrator_{timestamp}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example usage of Orchestrator"""
    
    # Configuration
    config = OrchestratorConfig(
        questionnaire_path="cuestionario_FIXED.json",
        plan_document_path="sample_plan.txt",
        execution_mapping_path="execution_mapping.yaml",
        method_class_map_path="COMPLETE_METHOD_CLASS_MAP.json",
        output_directory="output",
        enable_meso_clustering=True,
        enable_macro_convergence=True
    )
    
    # Initialize Orchestrator
    orchestrator = PolicyAnalysisOrchestrator(config)
    
    # Example plan document
    plan_document = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    MUNICIPIO DE EJEMPLO
    
    DIAGNÓSTICO TERRITORIAL
    El municipio cuenta con 45,000 habitantes según el último censo.
    La línea base cuantitativa indica una tasa de pobreza del 42.3%.
    Se identificaron recursos disponibles por $12,500 millones.
    
    ESTRATEGIA DE DESARROLLO
    Se implementarán programas en educación, salud y desarrollo económico.
    La teoría de cambio articula intervenciones con resultados esperados.
    
    PLAN PLURIANUAL DE INVERSIONES
    Presupuesto total: $25,000 millones
    Educación: $8,000 millones
    Salud: $7,000 millones
    Infraestructura: $6,000 millones
    Otros: $4,000 millones
    
    SISTEMA DE SEGUIMIENTO
    Se establecerán indicadores de resultado y producto.
    Mediciones semestrales con participación ciudadana.
    """
    
    plan_metadata = {
        "municipality": "Ejemplo",
        "department": "Departamento",
        "year": "2024-2027",
        "population": 45000
    }
    
    # Execute CHESS strategy
    result = orchestrator.execute_chess_strategy(
        plan_document,
        plan_metadata
    )
    
    # Save results
    orchestrator.save_results(result)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ORCHESTRATOR EXECUTION COMPLETED")
    print("=" * 80)
    print(f"\nExecution ID: {result.execution_id}")
    print(f"Overall Score: {result.macro_result.overall_score:.1f}/100")
    print(f"Classification: {result.macro_result.plan_classification}")
    print(f"\nMICRO answers: {len(result.micro_results)}")
    print(f"MESO clusters: {len(result.meso_results)}")
    print(f"\nExecution time: {result.execution_statistics['execution_time']:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    example_usage()
