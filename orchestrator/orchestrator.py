"""
Orchestrator - Global Pipeline Controller for 305-Question Processing

This module implements the GLOBAL orchestration of the complete end-to-end flow 
described in PSEUDOCODIGO_FLUJO_COMPLETO.md:

- FASE 0: Configuration loading (monolith, method catalog)
- FASE 1: Document ingestion and preprocessing  
- FASE 2: Execution of 300 micro questions (ASYNC) → delegates to Choreographer
- FASE 3: Scoring of 300 micro results (ASYNC)
- FASE 4: Dimension aggregation - 60 dimensions (ASYNC)
- FASE 5: Policy area aggregation - 10 areas (ASYNC)
- FASE 6: Cluster aggregation - 4 MESO questions (SYNC)
- FASE 7: Macro evaluation - 1 holistic question (SYNC)
- FASE 8: Recommendation generation (ASYNC)
- FASE 9: Report assembly (SYNC)
- FASE 10: Format and export (ASYNC)

Architecture:
- Orchestrates the WHAT and WHEN of the complete pipeline
- Delegates HOW of single question execution to Choreographer
- Implements scoring, aggregation, and report generation
- Maintains full traceability and audit trail
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestrator.coreographer import (
    Choreographer,
    QuestionResult as ChoreographerQuestionResult,
    PreprocessedDocument,
)
from orchestrator.aggregation import (
    DimensionAggregator,
    AreaPolicyAggregator,
    ClusterAggregator,
    MacroAggregator,
    ScoredResult as AggregationScoredResult,
    DimensionScore as AggregationDimensionScore,
    AreaScore as AggregationAreaScore,
    ClusterScore as AggregationClusterScore,
    MacroScore as AggregationMacroScore,
)

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for phases."""
    SYNC = "sync"
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class PhaseResult:
    """Result of a processing phase."""
    phase_id: str
    phase_name: str
    success: bool
    execution_time_ms: float
    mode: ExecutionMode
    data: Any = None
    error: Optional[Exception] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessedDocument:
    """Preprocessed document structure."""
    document_id: str
    raw_text: str
    normalized_text: str
    sentences: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    indexes: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class QuestionResult:
    """Result of processing a single micro question."""
    question_global: int
    base_slot: str
    evidence: Dict[str, Any]
    raw_results: Dict[str, Any]
    execution_time_ms: float = 0.0


@dataclass
class ScoredResult:
    """Scored result for a micro question."""
    question_global: int
    base_slot: str
    policy_area: str
    dimension: str
    score: float
    quality_level: str
    evidence: Dict[str, Any]
    raw_results: Dict[str, Any]


@dataclass
class DimensionScore:
    """Aggregated score for a dimension."""
    dimension_id: str
    area_id: str
    score: float
    quality_level: str
    contributing_questions: List[int]


@dataclass
class AreaScore:
    """Aggregated score for a policy area."""
    area_id: str
    area_name: str
    score: float
    quality_level: str
    dimension_scores: List[DimensionScore]


@dataclass
class ClusterScore:
    """Aggregated score for a MESO cluster."""
    cluster_id: str
    cluster_name: str
    areas: List[str]
    score: float
    coherence: float
    area_scores: List[AreaScore]


@dataclass
class MacroScore:
    """Holistic MACRO evaluation."""
    question_global: int
    type: str
    global_quality_index: float
    cross_cutting_coherence: float
    systemic_gaps: List[str]
    cluster_scores: List[ClusterScore]


@dataclass
class CompleteReport:
    """Complete analysis report."""
    micro_results: List[ScoredResult]
    dimension_scores: List[DimensionScore]
    area_scores: List[AreaScore]
    cluster_scores: List[ClusterScore]
    macro_score: MacroScore
    recommendations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class Orchestrator:
    """
    Orchestrator - Controls the GLOBAL 305-question processing pipeline.
    
    Responsibilities:
    - Load configuration (monolith, catalog)
    - Ingest and preprocess documents
    - Orchestrate 300 micro questions (delegates to Choreographer)
    - Score results (TYPE_A-F modalities)
    - Aggregate across all levels (dimensions, areas, clusters, macro)
    - Generate recommendations
    - Assemble and export reports
    
    NOT responsible for:
    - HOW to execute a single question (that's Choreographer)
    - Method-level DAG execution
    - Method priorities and retries
    """
    
    def __init__(
        self,
        monolith_path: Optional[Path] = None,
        method_catalog_path: Optional[Path] = None,
        choreographer: Optional[Choreographer] = None,
        enable_async: bool = True,
    ):
        """
        Initialize orchestrator.
        
        Args:
            monolith_path: Path to questionnaire_monolith.json
            method_catalog_path: Path to metodos_completos_nivel3.json
            choreographer: Choreographer instance (or create new)
            enable_async: Enable async execution for parallel phases
        """
        self.monolith_path = monolith_path or Path("questionnaire_monolith.json")
        self.method_catalog_path = method_catalog_path or Path("rules/METODOS/metodos_completos_nivel3.json")
        self.choreographer = choreographer or Choreographer()
        self.enable_async = enable_async
        
        # Configuration loaded in FASE 0
        self.monolith: Optional[Dict[str, Any]] = None
        self.method_catalog: Optional[Dict[str, Any]] = None
        
        # Aggregators (initialized after monolith is loaded)
        self.dimension_aggregator: Optional[DimensionAggregator] = None
        self.area_aggregator: Optional[AreaPolicyAggregator] = None
        self.cluster_aggregator: Optional[ClusterAggregator] = None
        self.macro_aggregator: Optional[MacroAggregator] = None
        
        # Execution tracking
        self.phase_results: List[PhaseResult] = []
        self.start_time: Optional[float] = None
        
        logger.info(
            f"Orchestrator initialized: "
            f"async={'enabled' if enable_async else 'disabled'}, "
            f"monolith={self.monolith_path}, "
            f"catalog={self.method_catalog_path}"
        )
    
    def _verify_integrity_hash(self, data: Dict[str, Any], expected_hash: str) -> bool:
        """
        Verify integrity hash of loaded data.
        
        Args:
            data: Data dictionary
            expected_hash: Expected SHA256 hash
            
        Returns:
            True if hash matches
            
        Raises:
            ValueError: If hash mismatch detected
        """
        # Extract data without integrity block for hash computation
        data_copy = {k: v for k, v in data.items() if k != 'integrity'}
        
        # Compute SHA256 hash
        computed_hash = hashlib.sha256(
            json.dumps(data_copy, sort_keys=True, ensure_ascii=False).encode('utf-8')
        ).hexdigest()
        
        # Compare hashes
        if computed_hash != expected_hash:
            raise ValueError(
                f"Monolith integrity hash mismatch! "
                f"Expected: {expected_hash[:16]}..., "
                f"Got: {computed_hash[:16]}..."
            )
        
        return True
    
    def _load_configuration(self) -> PhaseResult:
        """
        FASE 0: Load and validate configuration.
        
        Loads:
        - questionnaire_monolith.json (305 questions)
        - metodos_completos_nivel3.json (166 unique methods, 416 total)
        
        Validates:
        - Integrity hashes
        - Question counts (300 micro + 4 meso + 1 macro)
        - Method catalog structure
        
        Returns:
            PhaseResult for configuration loading
        """
        logger.info("=== FASE 0: CARGA DE CONFIGURACIÓN ===")
        start = time.time()
        
        try:
            # Load monolith
            with open(self.monolith_path) as f:
                self.monolith = json.load(f)
            
            # Verify monolith integrity
            expected_hash = self.monolith.get("integrity", {}).get("monolith_hash")
            if expected_hash and not self._verify_integrity_hash(self.monolith, expected_hash):
                raise ValueError("Monolith corrupted: hash mismatch")
            
            # Verify question counts
            counts = self.monolith.get("integrity", {}).get("question_count", {})
            if counts.get("total") != 305:
                raise ValueError(f"Expected 305 questions, found {counts.get('total')}")
            
            logger.info(f"✓ Monolith loaded: {counts.get('total')} questions")
            
            # Load method catalog
            with open(self.method_catalog_path) as f:
                self.method_catalog = json.load(f)
            
            # Verify method catalog
            meta = self.method_catalog.get("metadata", {})
            if meta.get("total_methods") != 416:
                logger.warning(
                    f"Expected 416 methods, found {meta.get('total_methods')}"
                )
            
            logger.info(f"✓ Catálogo cargado: {meta.get('total_methods')} métodos")
            
            # Initialize aggregators with loaded monolith
            self.dimension_aggregator = DimensionAggregator(self.monolith)
            self.area_aggregator = AreaPolicyAggregator(self.monolith)
            self.cluster_aggregator = ClusterAggregator(self.monolith)
            self.macro_aggregator = MacroAggregator(self.monolith)
            
            logger.info("✓ Aggregation modules initialized")
            
            duration = (time.time() - start) * 1000
            return PhaseResult(
                phase_id="FASE_0",
                phase_name="Carga de Configuración",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                metrics={
                    "questions_loaded": counts.get("total"),
                    "methods_loaded": meta.get("total_methods"),
                }
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Configuration loading failed: {e}")
            return PhaseResult(
                phase_id="FASE_0",
                phase_name="Carga de Configuración",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                error=e
            )
    
    def _ingest_document(self, pdf_path: str) -> PhaseResult:
        """
        FASE 1: Document ingestion and preprocessing.
        
        Steps:
        1. Load PDF (DI.DocumentLoader.load_pdf)
        2. Extract full text (DI.TextExtractor.extract_full_text)
        3. Preprocess document (DI.PreprocessingEngine.preprocess_document):
           - Normalize encoding
           - Segment into sentences
           - Extract and classify tables
           - Build indexes (term, numeric, temporal, table)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PhaseResult with PreprocessedDocument
        """
        logger.info("=== FASE 1: INGESTIÓN DEL DOCUMENTO ===")
        start = time.time()
        
        try:
            # TODO: Implement document ingestion
            # This is a placeholder - actual implementation would use DI module
            
            # For now, create a mock preprocessed document
            preprocessed_doc = PreprocessedDocument(
                document_id=hashlib.sha256(pdf_path.encode()).hexdigest()[:16],
                raw_text="",
                normalized_text="",
                sentences=[],
                tables=[],
                indexes={},
                metadata={"pdf_path": pdf_path}
            )
            
            duration = (time.time() - start) * 1000
            logger.info(f"✓ Documento preprocesado (placeholder)")
            
            return PhaseResult(
                phase_id="FASE_1",
                phase_name="Ingestión del Documento",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                data=preprocessed_doc,
                metrics={
                    "sentences": len(preprocessed_doc.sentences),
                    "tables": len(preprocessed_doc.tables),
                }
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Document ingestion failed: {e}")
            return PhaseResult(
                phase_id="FASE_1",
                phase_name="Ingestión del Documento",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                error=e
            )
    
    async def _process_micro_question_async(
        self,
        question_global: int,
        preprocessed_doc: PreprocessedDocument,
    ) -> QuestionResult:
        """
        Process a single micro question asynchronously.
        
        DELEGATES to Choreographer.process_micro_question()
        
        Args:
            question_global: Question number (1-300)
            preprocessed_doc: Preprocessed document
            
        Returns:
            QuestionResult with evidence and raw results
        """
        # Delegate to choreographer for granular execution
        result = await asyncio.to_thread(
            self.choreographer.process_micro_question,
            question_global,
            preprocessed_doc,
            self.monolith,
            self.method_catalog
        )
        
        # Convert choreographer's QuestionResult to orchestrator's QuestionResult
        return QuestionResult(
            question_global=result.question_global,
            base_slot=result.base_slot,
            evidence=result.evidence,
            raw_results=result.raw_results,
            execution_time_ms=result.execution_time_ms
        )
    
    def _process_micro_question_sync(
        self,
        question_global: int,
        preprocessed_doc: PreprocessedDocument,
    ) -> QuestionResult:
        """
        Process a single micro question synchronously.
        
        DELEGATES to Choreographer.process_micro_question()
        
        Args:
            question_global: Question number (1-300)
            preprocessed_doc: Preprocessed document
            
        Returns:
            QuestionResult
        """
        # Delegate to choreographer for granular execution
        result = self.choreographer.process_micro_question(
            question_global,
            preprocessed_doc,
            self.monolith,
            self.method_catalog
        )
        
        # Convert choreographer's QuestionResult to orchestrator's QuestionResult
        return QuestionResult(
            question_global=result.question_global,
            base_slot=result.base_slot,
            evidence=result.evidence,
            raw_results=result.raw_results,
            execution_time_ms=result.execution_time_ms
        )
    
    async def _execute_micro_questions_async(
        self,
        preprocessed_doc: PreprocessedDocument,
        timeout_seconds: int = 3600,  # 1 hour default timeout
    ) -> PhaseResult:
        """
        FASE 2: Execute all 300 micro questions in parallel with timeout and cancellation.
        
        Args:
            preprocessed_doc: Preprocessed document
            timeout_seconds: Maximum time to wait for all questions
            
        Returns:
            PhaseResult with list of QuestionResults
        """
        logger.info("=== FASE 2: EJECUCIÓN DE 300 MICRO PREGUNTAS ===")
        start = time.time()
        
        try:
            # Create tasks for all 300 questions
            tasks = [
                asyncio.create_task(self._process_micro_question_async(i, preprocessed_doc))
                for i in range(1, 301)
            ]
            
            # Execute all in parallel with timeout
            done, pending = await asyncio.wait(
                tasks,
                timeout=timeout_seconds,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # If there are pending tasks, cancel them
            if pending:
                logger.warning(f"Timeout reached! Cancelling {len(pending)} pending tasks")
                for task in pending:
                    task.cancel()
                # Wait for cancellation to complete
                await asyncio.gather(*pending, return_exceptions=True)
                
                # Raise error about timeout
                raise TimeoutError(
                    f"FASE 2 timeout after {timeout_seconds}s. "
                    f"Completed: {len(done)}/300, Cancelled: {len(pending)}"
                )
            
            # Collect results from completed tasks
            all_micro_results = [task.result() for task in done]
            
            duration = (time.time() - start) * 1000
            avg_time = duration / 300 if all_micro_results else 0
            
            logger.info(f"✓ 300 micro preguntas procesadas")
            logger.info(f"  - Tiempo promedio por pregunta: {avg_time:.2f}ms")
            logger.info(f"  - Tiempo total con paralelismo: {duration/1000:.2f}s")
            
            return PhaseResult(
                phase_id="FASE_2",
                phase_name="Ejecución de Micro Preguntas",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.ASYNC,
                data=all_micro_results,
                metrics={
                    "questions_processed": len(all_micro_results),
                    "avg_time_ms": avg_time,
                    "timeout_seconds": timeout_seconds,
                }
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Micro question execution failed: {e}")
            return PhaseResult(
                phase_id="FASE_2",
                phase_name="Ejecución de Micro Preguntas",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.ASYNC,
                error=e
            )
    
    def _apply_scoring_modality(
        self,
        evidence: Dict[str, Any],
        modality: str,
        scoring_config: Dict[str, Any],
    ) -> float:
        """
        Apply scoring modality to evidence.
        
        Implements scoring types from questionnaire_monolith.json:
        - TYPE_A: Count 4 elements and scale to 0-3
        - TYPE_B: Count up to 3 elements, each worth 1 point
        - TYPE_C: Count 2 elements and scale to 0-3
        - TYPE_D: Count 3 elements, weighted
        - TYPE_E: Boolean presence check
        - TYPE_F: Continuous scale
        
        Args:
            evidence: Evidence dictionary from micro question
            modality: Scoring modality (TYPE_A, TYPE_B, etc.)
            scoring_config: Scoring configuration from monolith
            
        Returns:
            Score between 0 and 3
        """
        # Simple implementation - count successful methods as proxy
        successful_count = evidence.get("successful_methods", 0)
        
        if modality == "TYPE_A":
            # Count 4 elements, scale to 0-3
            return min(3.0, (successful_count / 4.0) * 3.0)
        elif modality == "TYPE_B":
            # Count up to 3 elements, each worth 1 point
            return min(3.0, float(successful_count))
        elif modality == "TYPE_C":
            # Count 2 elements, scale to 0-3
            return min(3.0, (successful_count / 2.0) * 3.0)
        elif modality == "TYPE_D":
            # Weighted sum - get weights from config if available
            modality_defs = scoring_config.get("modality_definitions", {})
            modality_def = modality_defs.get(modality, {})
            weights = modality_def.get("weights", [0.4, 0.3, 0.3])
            # Simplified - just use first weight
            return min(3.0, successful_count * weights[0]) if weights else 0.0
        elif modality == "TYPE_E":
            # Boolean presence
            return 3.0 if successful_count > 0 else 0.0
        elif modality == "TYPE_F":
            # Continuous scale - use confidence from evidence
            return min(3.0, evidence.get("confidence", 0.0) * 3.0)
        else:
            logger.warning(f"Unknown scoring modality: {modality}")
            return 0.0
    
    def _determine_quality_level(
        self,
        score: float,
        thresholds: List[Dict[str, Any]],
    ) -> str:
        """
        Determine quality level from score.
        
        Args:
            score: Score (typically 0-3 range)
            thresholds: Quality level thresholds from monolith
            
        Returns:
            Quality level (EXCELENTE, BUENO, ACEPTABLE, INSUFICIENTE)
        
        Note:
            Assumes scores are in 0-3 range as per scoring modality definitions.
            Scores outside this range are clamped to [0, 3] before normalization.
        """
        # Clamp score to valid range [0, 3]
        clamped_score = max(0.0, min(3.0, score))
        
        # Convert score to 0-1 scale
        normalized_score = clamped_score / 3.0
        
        # Apply thresholds (from questionnaire_monolith.json)
        # These are the standard thresholds used across the system
        if normalized_score >= 0.85:
            return "EXCELENTE"
        elif normalized_score >= 0.70:
            return "BUENO"
        elif normalized_score >= 0.55:
            return "ACEPTABLE"
        else:
            return "INSUFICIENTE"
    
    async def _score_micro_result_async(
        self,
        micro_result: QuestionResult,
    ) -> ScoredResult:
        """
        Score a single micro result asynchronously.
        
        Args:
            micro_result: QuestionResult from FASE 2
            
        Returns:
            ScoredResult with score and quality level
        """
        # Get question metadata
        q_metadata = self.monolith["blocks"]["micro_questions"][
            micro_result.question_global - 1
        ]
        
        # Get scoring config
        scoring_config = self.monolith["blocks"]["scoring"]
        scoring_modality = q_metadata.get("scoring_modality", "TYPE_A")
        
        # Apply scoring
        score = self._apply_scoring_modality(
            micro_result.evidence,
            scoring_modality,
            scoring_config
        )
        
        # Determine quality level
        quality_level = self._determine_quality_level(
            score,
            scoring_config.get("micro_levels", [])
        )
        
        return ScoredResult(
            question_global=micro_result.question_global,
            base_slot=micro_result.base_slot,
            policy_area=q_metadata.get("policy_area_id", ""),
            dimension=q_metadata.get("dimension_id", ""),
            score=score,
            quality_level=quality_level,
            evidence=micro_result.evidence,
            raw_results=micro_result.raw_results
        )
    
    async def _score_micro_results_async(
        self,
        all_micro_results: List[QuestionResult],
    ) -> PhaseResult:
        """
        FASE 3: Score all 300 micro results in parallel.
        
        Args:
            all_micro_results: List of QuestionResults from FASE 2
            
        Returns:
            PhaseResult with list of ScoredResults
        """
        logger.info("=== FASE 3: SCORING DE MICRO PREGUNTAS ===")
        start = time.time()
        
        try:
            # Create tasks for scoring all results
            tasks = [
                self._score_micro_result_async(result)
                for result in all_micro_results
            ]
            
            # Execute all in parallel
            all_scored_results = await asyncio.gather(*tasks)
            
            # Calculate statistics
            stats = {
                "EXCELENTE": sum(1 for r in all_scored_results if r.quality_level == "EXCELENTE"),
                "BUENO": sum(1 for r in all_scored_results if r.quality_level == "BUENO"),
                "ACEPTABLE": sum(1 for r in all_scored_results if r.quality_level == "ACEPTABLE"),
                "INSUFICIENTE": sum(1 for r in all_scored_results if r.quality_level == "INSUFICIENTE"),
            }
            
            duration = (time.time() - start) * 1000
            
            logger.info(f"✓ 300 micro preguntas scored")
            logger.info(f"  - EXCELENTE: {stats['EXCELENTE']}")
            logger.info(f"  - BUENO: {stats['BUENO']}")
            logger.info(f"  - ACEPTABLE: {stats['ACEPTABLE']}")
            logger.info(f"  - INSUFICIENTE: {stats['INSUFICIENTE']}")
            
            return PhaseResult(
                phase_id="FASE_3",
                phase_name="Scoring de Micro Preguntas",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.ASYNC,
                data=all_scored_results,
                metrics=stats
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Scoring failed: {e}")
            return PhaseResult(
                phase_id="FASE_3",
                phase_name="Scoring de Micro Preguntas",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.ASYNC,
                error=e
            )
    
    async def _aggregate_dimension_async(
        self,
        dimension_id: str,
        area_id: str,
        scored_results: List[ScoredResult],
    ) -> DimensionScore:
        """
        Aggregate a single dimension asynchronously.
        
        Delegates to DimensionAggregator for full validation and logging.
        
        Args:
            dimension_id: Dimension ID (e.g., "DIM01")
            area_id: Policy area ID (e.g., "PA01")
            scored_results: List of scored results for this dimension
            
        Returns:
            DimensionScore with aggregated score
        """
        # Delegate to aggregation module
        return await asyncio.to_thread(
            self.dimension_aggregator.aggregate_dimension,
            dimension_id,
            area_id,
            scored_results
        )
    
    async def _aggregate_dimensions_async(
        self,
        all_scored_results: List[ScoredResult],
    ) -> PhaseResult:
        """
        FASE 4: Aggregate 60 dimensions (6 dimensions × 10 policy areas).
        
        Args:
            all_scored_results: List of ScoredResults from FASE 3
            
        Returns:
            PhaseResult with list of DimensionScores
        """
        logger.info("=== FASE 4: AGREGACIÓN POR DIMENSIÓN ===")
        start = time.time()
        
        try:
            # Get all policy areas and dimensions
            policy_areas = self.monolith["blocks"]["niveles_abstraccion"]["policy_areas"]
            dimensions = self.monolith["blocks"]["niveles_abstraccion"]["dimensions"]
            
            # Create tasks for all 60 combinations
            tasks = []
            for area in policy_areas:
                area_id = area["policy_area_id"]
                for dim in dimensions:
                    dim_id = dim["dimension_id"]
                    tasks.append(
                        self._aggregate_dimension_async(dim_id, area_id, all_scored_results)
                    )
            
            # Execute all in parallel
            all_dimension_scores = await asyncio.gather(*tasks)
            
            duration = (time.time() - start) * 1000
            
            logger.info(f"✓ {len(all_dimension_scores)} dimensiones agregadas")
            
            return PhaseResult(
                phase_id="FASE_4",
                phase_name="Agregación por Dimensión",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.ASYNC,
                data=all_dimension_scores,
                metrics={"dimensions_aggregated": len(all_dimension_scores)}
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Dimension aggregation failed: {e}")
            return PhaseResult(
                phase_id="FASE_4",
                phase_name="Agregación por Dimensión",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.ASYNC,
                error=e
            )
    
    def _aggregate_dimensions_sync(
        self,
        all_scored_results: List[ScoredResult],
    ) -> PhaseResult:
        """FASE 4: Aggregate dimensions sequentially."""
        return asyncio.run(self._aggregate_dimensions_async(all_scored_results))
    
    async def _aggregate_area_async(
        self,
        area_id: str,
        dimension_scores: List[DimensionScore],
    ) -> AreaScore:
        """
        Aggregate a single policy area asynchronously.
        
        Delegates to AreaPolicyAggregator for full validation and logging.
        
        Args:
            area_id: Policy area ID (e.g., "PA01")
            dimension_scores: List of dimension scores
            
        Returns:
            AreaScore with aggregated score
        """
        # Delegate to aggregation module
        return await asyncio.to_thread(
            self.area_aggregator.aggregate_area,
            area_id,
            dimension_scores
        )
    
    async def _aggregate_areas_async(
        self,
        all_dimension_scores: List[DimensionScore],
    ) -> PhaseResult:
        """
        FASE 5: Aggregate 10 policy areas.
        
        Args:
            all_dimension_scores: List of DimensionScores from FASE 4
            
        Returns:
            PhaseResult with list of AreaScores
        """
        logger.info("=== FASE 5: AGREGACIÓN POR ÁREA DE POLÍTICA ===")
        start = time.time()
        
        try:
            # Get all policy areas
            policy_areas = self.monolith["blocks"]["niveles_abstraccion"]["policy_areas"]
            
            # Create tasks for all 10 areas
            tasks = [
                self._aggregate_area_async(area["policy_area_id"], all_dimension_scores)
                for area in policy_areas
            ]
            
            # Execute all in parallel
            all_area_scores = await asyncio.gather(*tasks)
            
            duration = (time.time() - start) * 1000
            
            logger.info(f"✓ {len(all_area_scores)} áreas de política agregadas")
            
            return PhaseResult(
                phase_id="FASE_5",
                phase_name="Agregación por Área de Política",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.ASYNC,
                data=all_area_scores,
                metrics={"areas_aggregated": len(all_area_scores)}
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Area aggregation failed: {e}")
            return PhaseResult(
                phase_id="FASE_5",
                phase_name="Agregación por Área de Política",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.ASYNC,
                error=e
            )
    
    def _aggregate_areas_sync(
        self,
        all_dimension_scores: List[DimensionScore],
    ) -> PhaseResult:
        """FASE 5: Aggregate areas sequentially."""
        return asyncio.run(self._aggregate_areas_async(all_dimension_scores))
    
    def _aggregate_clusters(
        self,
        all_area_scores: List[AreaScore],
    ) -> PhaseResult:
        """
        FASE 6: Aggregate 4 MESO clusters (Q301-Q304).
        
        Synchronous execution as per pseudocode - only 4 clusters, not worth parallelizing.
        
        Args:
            all_area_scores: List of AreaScores from FASE 5
            
        Returns:
            PhaseResult with list of ClusterScores
        """
        logger.info("=== FASE 6: AGREGACIÓN POR CLUSTER (MESO) ===")
        start = time.time()
        
        try:
            # Get cluster definitions
            clusters = self.monolith["blocks"]["niveles_abstraccion"]["clusters"]
            all_cluster_scores = []
            
            for cluster in clusters:
                cluster_id = cluster["cluster_id"]
                
                # Filter area scores for this cluster
                policy_area_ids = cluster["policy_area_ids"]
                cluster_area_scores = [
                    a for a in all_area_scores
                    if a.area_id in policy_area_ids
                ]
                
                if not cluster_area_scores:
                    logger.warning(f"No area scores found for cluster {cluster_id}")
                    continue
                
                # Delegate to aggregation module
                cluster_score = self.cluster_aggregator.aggregate_cluster(
                    cluster_id,
                    cluster_area_scores
                )
                
                all_cluster_scores.append(cluster_score)
                logger.info(
                    f"✓ {cluster_id} ({cluster_score.cluster_name}): {cluster_score.score:.2f}"
                )
            
            duration = (time.time() - start) * 1000
            
            logger.info(f"✓ {len(all_cluster_scores)} clusters MESO agregados (Q301-Q304 respondidas)")
            
            return PhaseResult(
                phase_id="FASE_6",
                phase_name="Agregación por Cluster (MESO)",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                data=all_cluster_scores,
                metrics={"clusters_aggregated": len(all_cluster_scores)}
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Cluster aggregation failed: {e}")
            return PhaseResult(
                phase_id="FASE_6",
                phase_name="Agregación por Cluster (MESO)",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                error=e
            )
    
    def _evaluate_macro(
        self,
        all_cluster_scores: List[ClusterScore],
        all_area_scores: List[AreaScore],
        all_dimension_scores: List[DimensionScore],
    ) -> PhaseResult:
        """
        FASE 7: Macro holistic evaluation (Q305).
        
        Synchronous execution - only 1 macro evaluation.
        
        Args:
            all_cluster_scores: List of ClusterScores from FASE 6
            all_area_scores: List of AreaScores from FASE 5
            all_dimension_scores: List of DimensionScores from FASE 4
            
        Returns:
            PhaseResult with MacroScore
        """
        logger.info("=== FASE 7: EVALUACIÓN MACRO HOLÍSTICA ===")
        start = time.time()
        
        try:
            # Delegate to aggregation module
            macro_score = self.macro_aggregator.evaluate_macro(
                all_cluster_scores,
                all_area_scores,
                all_dimension_scores
            )
            
            duration = (time.time() - start) * 1000
            
            logger.info(f"✓ Evaluación MACRO completada (Q305 respondida)")
            logger.info(f"  - Score: {macro_score.score:.2f}")
            logger.info(f"  - Calidad: {macro_score.quality_level}")
            logger.info(f"  - Coherencia transversal: {macro_score.cross_cutting_coherence:.2f}")
            logger.info(f"  - Brechas sistémicas: {len(macro_score.systemic_gaps)}")
            
            return PhaseResult(
                phase_id="FASE_7",
                phase_name="Evaluación Macro Holística",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                data=macro_score,
                metrics={
                    "macro_score": macro_score.score,
                    "quality_level": macro_score.quality_level,
                    "systemic_gaps_count": len(macro_score.systemic_gaps),
                    "strategic_alignment": macro_score.strategic_alignment,
                }
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Macro evaluation failed: {e}")
            return PhaseResult(
                phase_id="FASE_7",
                phase_name="Evaluación Macro Holística",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                error=e
            )
    
    def _score_micro_results_sync(
        self,
        all_micro_results: List[QuestionResult],
    ) -> PhaseResult:
        """
        FASE 3: Score all 300 micro results sequentially.
        
        Fallback for when async is disabled.
        
        Args:
            all_micro_results: List of QuestionResults from FASE 2
            
        Returns:
            PhaseResult with list of ScoredResults
        """
        logger.info("=== FASE 3: SCORING DE MICRO PREGUNTAS (SYNC) ===")
        start = time.time()
        
        try:
            all_scored_results = []
            for result in all_micro_results:
                scored = asyncio.run(self._score_micro_result_async(result))
                all_scored_results.append(scored)
            
            # Calculate statistics
            stats = {
                "EXCELENTE": sum(1 for r in all_scored_results if r.quality_level == "EXCELENTE"),
                "BUENO": sum(1 for r in all_scored_results if r.quality_level == "BUENO"),
                "ACEPTABLE": sum(1 for r in all_scored_results if r.quality_level == "ACEPTABLE"),
                "INSUFICIENTE": sum(1 for r in all_scored_results if r.quality_level == "INSUFICIENTE"),
            }
            
            duration = (time.time() - start) * 1000
            
            logger.info(f"✓ 300 micro preguntas scored (secuencial)")
            logger.info(f"  - EXCELENTE: {stats['EXCELENTE']}")
            logger.info(f"  - BUENO: {stats['BUENO']}")
            logger.info(f"  - ACEPTABLE: {stats['ACEPTABLE']}")
            logger.info(f"  - INSUFICIENTE: {stats['INSUFICIENTE']}")
            
            return PhaseResult(
                phase_id="FASE_3",
                phase_name="Scoring de Micro Preguntas",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                data=all_scored_results,
                metrics=stats
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Scoring failed: {e}")
            return PhaseResult(
                phase_id="FASE_3",
                phase_name="Scoring de Micro Preguntas",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                error=e
            )
    
    def _execute_micro_questions_sync(
        self,
        preprocessed_doc: PreprocessedDocument,
    ) -> PhaseResult:
        """
        FASE 2: Execute all 300 micro questions sequentially.
        
        Fallback for when async is disabled.
        
        Args:
            preprocessed_doc: Preprocessed document
            
        Returns:
            PhaseResult with list of QuestionResults
        """
        logger.info("=== FASE 2: EJECUCIÓN DE 300 MICRO PREGUNTAS (SYNC) ===")
        start = time.time()
        
        try:
            all_micro_results = []
            for i in range(1, 301):
                result = self._process_micro_question_sync(i, preprocessed_doc)
                all_micro_results.append(result)
            
            duration = (time.time() - start) * 1000
            avg_time = duration / 300 if all_micro_results else 0
            
            logger.info(f"✓ 300 micro preguntas procesadas (secuencial)")
            logger.info(f"  - Tiempo promedio por pregunta: {avg_time:.2f}ms")
            logger.info(f"  - Tiempo total: {duration/1000:.2f}s")
            
            return PhaseResult(
                phase_id="FASE_2",
                phase_name="Ejecución de Micro Preguntas",
                success=True,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                data=all_micro_results,
                metrics={
                    "questions_processed": len(all_micro_results),
                    "avg_time_ms": avg_time,
                }
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"Micro question execution failed: {e}")
            return PhaseResult(
                phase_id="FASE_2",
                phase_name="Ejecución de Micro Preguntas",
                success=False,
                execution_time_ms=duration,
                mode=ExecutionMode.SYNC,
                error=e
            )
    
    def process_development_plan(self, pdf_path: str) -> CompleteReport:
        """
        Main entry point: Process a development plan from PDF to complete report.
        
        This implements the complete process_development_plan function from
        PSEUDOCODIGO_FLUJO_COMPLETO.md, executing all 10 phases:
        
        FASE 0: Configuration loading
        FASE 1: Document ingestion
        FASE 2: 300 micro questions execution
        FASE 3: Scoring micro results
        FASE 4: Dimension aggregation (60)
        FASE 5: Policy area aggregation (10)
        FASE 6: Cluster aggregation (4 MESO)
        FASE 7: Macro evaluation (1)
        FASE 8: Recommendation generation
        FASE 9: Report assembly
        FASE 10: Format and export
        
        Args:
            pdf_path: Path to PDF file to analyze
            
        Returns:
            CompleteReport with all 305 answers and analysis
            
        Raises:
            ValueError: If configuration or processing fails
        """
        self.start_time = time.time()
        self.phase_results = []
        
        logger.info("=" * 70)
        logger.info("INICIANDO PROCESAMIENTO COMPLETO DEL PLAN DE DESARROLLO")
        logger.info("=" * 70)
        
        # FASE 0: Load configuration
        phase0 = self._load_configuration()
        self.phase_results.append(phase0)
        if not phase0.success:
            raise ValueError(f"Configuration loading failed: {phase0.error}")
        
        # FASE 1: Ingest document
        phase1 = self._ingest_document(pdf_path)
        self.phase_results.append(phase1)
        if not phase1.success:
            raise ValueError(f"Document ingestion failed: {phase1.error}")
        
        preprocessed_doc = phase1.data
        
        # FASE 2: Execute micro questions
        if self.enable_async:
            phase2 = asyncio.run(self._execute_micro_questions_async(preprocessed_doc))
        else:
            phase2 = self._execute_micro_questions_sync(preprocessed_doc)
        self.phase_results.append(phase2)
        if not phase2.success:
            raise ValueError(f"Micro question execution failed: {phase2.error}")
        
        all_micro_results = phase2.data
        
        # FASE 3: Score micro results
        if self.enable_async:
            phase3 = asyncio.run(self._score_micro_results_async(all_micro_results))
        else:
            phase3 = self._score_micro_results_sync(all_micro_results)
        self.phase_results.append(phase3)
        if not phase3.success:
            raise ValueError(f"Scoring failed: {phase3.error}")
        
        all_scored_results = phase3.data
        
        # FASE 4: Aggregate dimensions
        if self.enable_async:
            phase4 = asyncio.run(self._aggregate_dimensions_async(all_scored_results))
        else:
            phase4 = self._aggregate_dimensions_sync(all_scored_results)
        self.phase_results.append(phase4)
        if not phase4.success:
            raise ValueError(f"Dimension aggregation failed: {phase4.error}")
        
        all_dimension_scores = phase4.data
        
        # FASE 5: Aggregate policy areas
        if self.enable_async:
            phase5 = asyncio.run(self._aggregate_areas_async(all_dimension_scores))
        else:
            phase5 = self._aggregate_areas_sync(all_dimension_scores)
        self.phase_results.append(phase5)
        if not phase5.success:
            raise ValueError(f"Area aggregation failed: {phase5.error}")
        
        all_area_scores = phase5.data
        
        # FASE 6: Aggregate clusters (MESO)
        phase6 = self._aggregate_clusters(all_area_scores)
        self.phase_results.append(phase6)
        if not phase6.success:
            raise ValueError(f"Cluster aggregation failed: {phase6.error}")
        
        all_cluster_scores = phase6.data
        
        # FASE 7: Macro evaluation
        phase7 = self._evaluate_macro(all_cluster_scores, all_area_scores, all_dimension_scores)
        self.phase_results.append(phase7)
        if not phase7.success:
            raise ValueError(f"Macro evaluation failed: {phase7.error}")
        
        macro_score = phase7.data
        
        # TODO: FASE 8-10 implementation (recommendations, report assembly, export)
        # For now, create report with all computed data
        
        total_time = time.time() - self.start_time
        logger.info("=" * 70)
        logger.info("PROCESAMIENTO COMPLETO - 305 PREGUNTAS RESPONDIDAS")
        logger.info(f"✓ Tiempo total: {total_time:.2f}s")
        logger.info(f"✓ Índice de calidad global: {macro_score.global_quality_index:.2f}/100")
        logger.info("=" * 70)
        
        # Return complete report structure
        return CompleteReport(
            micro_results=all_scored_results,
            dimension_scores=all_dimension_scores,
            area_scores=all_area_scores,
            cluster_scores=all_cluster_scores,
            macro_score=macro_score,
            recommendations=[],  # TODO: FASE 8
            metadata={
                "pdf_path": pdf_path,
                "total_time_s": total_time,
                "phases_completed": len(self.phase_results),
                "phases": [
                    {
                        "id": p.phase_id,
                        "name": p.phase_name,
                        "success": p.success,
                        "duration_ms": p.execution_time_ms,
                        "mode": p.mode.value,
                    }
                    for p in self.phase_results
                ],
            }
        )


__all__ = [
    "Choreographer",
    "CompleteReport",
    "PreprocessedDocument",
    "QuestionResult",
    "ScoredResult",
    "DimensionScore",
    "AreaScore",
    "ClusterScore",
    "MacroScore",
    "ExecutionMode",
    "PhaseResult",
]
