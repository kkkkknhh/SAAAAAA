#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator - CHESS Strategy Control Plane (MACRO Level)
=========================================================

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

Author: Integration Team
Version: 1.0.0 - Complete CHESS Implementation
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


@dataclass
class OrchestratorConfig:
    """Configuration for Orchestrator execution"""
    questionnaire_path: str = "cuestionario_FIXED.json"
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
    
    CHESS STRATEGY:
    - Opening: Parallel execution of 7 producers across all 300 questions
    - Middle Game: Scoring with 6 modalities (TYPE_A through TYPE_F)
    - Endgame: MICRO → MESO → MACRO convergence synthesis
    
    TWO-CORE DATA FLOW:
    - Generation Pipeline: 7 producers (dereck_beach, policy_processor, etc.)
    - Collection & Assembly Pipeline: report_assembly (single assembler)
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize Orchestrator with canonical truth model (Golden Rule 1)
        
        Args:
            config: Orchestrator configuration
        """
        logger.info("=" * 80)
        logger.info("ORCHESTRATOR INITIALIZATION - CHESS STRATEGY")
        logger.info("=" * 80)
        
        self.config = config
        self.start_time = time.time()

        # Golden Rule 1: Load canonical truth model (cuestionario_FIXED.json)
        self.questionnaire = self._load_questionnaire()
        self.questionnaire_hash = self._compute_questionnaire_hash()

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
        self.choreographer = ExecutionChoreographer(
            execution_mapping_path=config.execution_mapping_path,
            method_class_map_path=config.method_class_map_path,
            questionnaire_hash=self.questionnaire_hash,
            deterministic_context=self.deterministic_context
        )
        
        # Initialize Report Assembler (Collection & Assembly Pipeline)
        self.report_assembler = ReportAssembler()
        
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

            total_questions = len(questionnaire.get("questions", []))
            logger.info(f"✓ Loaded {total_questions} questions from canonical truth model")

            return questionnaire

        except FileNotFoundError:
            logger.error(f"Canonical truth model not found: {self.config.questionnaire_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load questionnaire: {e}")
            raise

    def _compute_questionnaire_hash(self) -> str:
        """Compute stable hash of the loaded questionnaire metadata."""

        canonical = json.dumps(self.questionnaire, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _parse_all_questions(self) -> List[QuestionSpec]:
        """
        Parse all questions into structured QuestionSpec objects (Golden Rule 2)
        
        Performs atomic context hydration - loads ALL question metadata before execution
        """
        logger.info("Parsing all questions into structured specs...")
        
        questions = []
        
        for q_data in self.questionnaire.get("questions", []):
            try:
                # Extract P#-D#-Q# components
                canonical_id = q_data.get("question_unique_id", "")
                parts = canonical_id.split("-")
                
                policy_area = parts[0] if len(parts) > 0 else "P0"
                dimension = parts[1] if len(parts) > 1 else "D0"
                question_num = int(parts[2].replace("Q", "")) if len(parts) > 2 else 0
                
                # Create QuestionSpec
                spec = QuestionSpec(
                    canonical_id=canonical_id,
                    policy_area=policy_area,
                    dimension=dimension,
                    question_number=question_num,
                    question_text=q_data.get("question_text", ""),
                    scoring_modality=q_data.get("scoring_modality", "TYPE_A"),
                    expected_elements=q_data.get("expected_elements", []),
                    search_patterns=q_data.get("search_patterns", {}),
                    element_weights=q_data.get("element_weights", {}),
                    numerical_thresholds=q_data.get("numerical_thresholds", {}),
                    validation_rules=q_data.get("validation_rules", {}),
                    metadata={
                        "hints": q_data.get("hints", []),
                        "point_code": q_data.get("point_code", ""),
                        "point_title": q_data.get("point_title", "")
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
        
        Clusters group related questions across policy areas for mid-level analysis
        """
        # Define clusters based on policy areas (P1-P10)
        clusters = []
        
        # Cluster by policy area
        policy_areas_map = defaultdict(list)
        for q in self.all_questions:
            policy_areas_map[q.policy_area].append(q.canonical_id)
        
        for policy_area, question_ids in policy_areas_map.items():
            cluster = ClusterDefinition(
                cluster_id=f"CLUSTER_{policy_area}",
                cluster_name=f"Política {policy_area}",
                description=f"Cluster para área de política {policy_area}",
                policy_areas=[policy_area],
                dimensions=["D1", "D2", "D3", "D4", "D5", "D6"],
                question_ids=question_ids
            )
            clusters.append(cluster)
        
        # Additional cross-cutting clusters
        # Cluster by dimension (D1-D6)
        dimension_map = defaultdict(list)
        for q in self.all_questions:
            dimension_map[q.dimension].append(q.canonical_id)
        
        for dimension, question_ids in dimension_map.items():
            if len(question_ids) >= 10:  # Only create if sufficient questions
                cluster = ClusterDefinition(
                    cluster_id=f"CLUSTER_{dimension}",
                    cluster_name=f"Dimensión {dimension}",
                    description=f"Cluster para dimensión {dimension}",
                    policy_areas=["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
                    dimensions=[dimension],
                    question_ids=question_ids
                )
                clusters.append(cluster)
        
        logger.info(f"✓ Defined {len(clusters)} MESO clusters")
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
        
        CHESS PHASES:
        1. Opening: Parallel execution of all questions (7 producers)
        2. Middle Game: Scoring with 6 modalities (TYPE_A-F)
        3. Endgame: MICRO → MESO → MACRO convergence
        
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
        # ====================================================================
        logger.info("\n🎯 CHESS OPENING: Executing all questions (MICRO level)")
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
        
        result = OrchestratorResult(
            execution_id=execution_id,
            timestamp=datetime.now().isoformat(),
            micro_results=micro_results,
            meso_results=meso_results,
            macro_result=macro_result,
            execution_statistics=self.stats.copy(),
            performance_metrics={
                "total_execution_time": self.stats["execution_time"],
                "avg_question_time": self.stats["execution_time"] / max(1, len(micro_results)),
                "choreographer_stats": self.choreographer.get_statistics()
            },
            provenance={
                "questionnaire_path": self.config.questionnaire_path,
                "execution_mapping_path": self.config.execution_mapping_path,
                "method_class_map_path": self.config.method_class_map_path,
                "plan_metadata": plan_metadata
            }
        )
        
        logger.info("=" * 80)
        logger.info(f"🏆 CHESS STRATEGY COMPLETED in {self.stats['execution_time']:.2f}s")
        logger.info("=" * 80)
        
        return result

    def _execute_opening(
        self,
        plan_document: str,
        plan_metadata: Dict[str, Any]
    ) -> Dict[str, MicroLevelAnswer]:
        """
        CHESS OPENING: Execute all questions in parallel (or sequential)
        
        Implements Golden Rule 3: Deterministic Pipeline Execution
        Implements Golden Rule 5: Absolute Processing Homogeneity
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
        CHESS MIDDLE GAME: Analyze scoring modalities (TYPE_A through TYPE_F)
        
        Groups questions by scoring modality and analyzes patterns
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
        Generate MESO-level cluster aggregations
        
        Aggregates MICRO answers into thematic clusters for mid-level analysis
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
                cluster_description=cluster_def.description,
                micro_answers=cluster_answers,
                cluster_definition={
                    "policy_areas": cluster_def.policy_areas,
                    "dimensions": cluster_def.dimensions
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
        Generate MACRO-level convergence with Decálogo framework
        
        Provides executive-level assessment of entire plan
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
        Save complete results to output directory
        
        Generates:
        - MICRO answers JSON
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
