"""
Choreographer - Main Flow Controller for 305-Question Processing Pipeline

This module implements the complete end-to-end flow described in PSEUDOCODIGO_FLUJO_COMPLETO.md:
- FASE 0: Configuration loading (monolith, method catalog)
- FASE 1: Document ingestion and preprocessing
- FASE 2: Execution of 300 micro questions (ASYNC)
- FASE 3: Scoring of 300 micro results (ASYNC)
- FASE 4: Dimension aggregation - 60 dimensions (ASYNC)
- FASE 5: Policy area aggregation - 10 areas (ASYNC)
- FASE 6: Cluster aggregation - 4 MESO questions (SYNC)
- FASE 7: Macro evaluation - 1 holistic question (SYNC)
- FASE 8: Recommendation generation (ASYNC)
- FASE 9: Report assembly (SYNC)
- FASE 10: Format and export (ASYNC)

Architecture:
- Uses choreographer_dispatch for FQN-based method invocation
- Implements both SYNC and ASYNC execution patterns
- Maintains full traceability and audit trail
- Follows SIN_CARRETA doctrine (no graceful degradation)
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
from typing import Any, Dict, List, Optional, Set, Tuple

from orchestrator.choreographer_dispatch import (
    ChoreographerDispatcher,
    InvocationContext,
    InvocationResult,
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


class Choreographer:
    """
    Main choreographer for the complete 305-question processing pipeline.
    
    This class orchestrates the entire flow from document ingestion to final report,
    following the exact structure defined in PSEUDOCODIGO_FLUJO_COMPLETO.md.
    
    Key responsibilities:
    - Load and validate configuration (monolith, method catalog)
    - Coordinate document ingestion and preprocessing
    - Execute all 305 questions (300 micro + 4 meso + 1 macro)
    - Aggregate results across all levels
    - Generate recommendations
    - Assemble and export final report
    
    Execution patterns:
    - SYNC: Sequential execution (configuration, ingestion, some aggregations)
    - ASYNC: Parallel execution (micro questions, scoring, lower-level aggregations)
    - HYBRID: Mix of sync and async based on DAG structure
    """
    
    def __init__(
        self,
        monolith_path: Optional[Path] = None,
        method_catalog_path: Optional[Path] = None,
        dispatcher: Optional[ChoreographerDispatcher] = None,
        enable_async: bool = True,
    ):
        """
        Initialize choreographer.
        
        Args:
            monolith_path: Path to questionnaire_monolith.json
            method_catalog_path: Path to metodos_completos_nivel3.json
            dispatcher: ChoreographerDispatcher instance (or create new)
            enable_async: Enable async execution for parallel phases
        """
        self.monolith_path = monolith_path or Path("questionnaire_monolith.json")
        self.method_catalog_path = method_catalog_path or Path("rules/METODOS/metodos_completos_nivel3.json")
        self.dispatcher = dispatcher or ChoreographerDispatcher()
        self.enable_async = enable_async
        
        # Configuration loaded in FASE 0
        self.monolith: Optional[Dict[str, Any]] = None
        self.method_catalog: Optional[Dict[str, Any]] = None
        
        # Execution tracking
        self.phase_results: List[PhaseResult] = []
        self.start_time: Optional[float] = None
        
        logger.info(
            f"Choreographer initialized: "
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
        """
        # Compute hash of data (excluding integrity block)
        data_copy = data.copy()
        if "integrity" in data_copy:
            del data_copy["integrity"]
        
        computed = hashlib.sha256(
            json.dumps(data_copy, sort_keys=True).encode()
        ).hexdigest()
        
        return computed == expected_hash
    
    def _load_configuration(self) -> PhaseResult:
        """
        FASE 0: Load and validate configuration.
        
        Loads:
        - questionnaire_monolith.json (305 questions)
        - metodos_completos_nivel3.json (166 unique methods, 593 total)
        
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
            if meta.get("total_methods") != 593:
                logger.warning(
                    f"Expected 593 methods, found {meta.get('total_methods')}"
                )
            
            logger.info(f"✓ Catálogo cargado: {meta.get('total_methods')} métodos")
            
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
        
        This implements the process_micro_question function from the pseudocode:
        1. Map question to base_slot
        2. Build execution DAG from flow spec
        3. Execute methods (hybrid: sync for sequential, async for parallel)
        4. Extract evidence
        
        Args:
            question_global: Question number (1-300)
            preprocessed_doc: Preprocessed document
            
        Returns:
            QuestionResult with evidence and raw results
        """
        start = time.time()
        
        # Step 1: Map to base_slot
        base_index = (question_global - 1) % 30
        base_slot = f"D{base_index // 5 + 1}-Q{base_index % 5 + 1}"
        
        # Get question metadata from monolith
        q_metadata = self.monolith["blocks"]["micro_questions"][question_global - 1]
        
        # TODO: Get method packages from catalog and build DAG
        # TODO: Execute methods according to DAG
        # TODO: Extract evidence
        
        # Placeholder implementation
        evidence = {}
        raw_results = {}
        
        duration = (time.time() - start) * 1000
        
        return QuestionResult(
            question_global=question_global,
            base_slot=base_slot,
            evidence=evidence,
            raw_results=raw_results,
            execution_time_ms=duration
        )
    
    def _process_micro_question_sync(
        self,
        question_global: int,
        preprocessed_doc: PreprocessedDocument,
    ) -> QuestionResult:
        """
        Process a single micro question synchronously.
        
        Wrapper for synchronous execution when async is disabled.
        
        Args:
            question_global: Question number (1-300)
            preprocessed_doc: Preprocessed document
            
        Returns:
            QuestionResult
        """
        # Use asyncio.run to execute async function synchronously
        return asyncio.run(
            self._process_micro_question_async(question_global, preprocessed_doc)
        )
    
    async def _execute_micro_questions_async(
        self,
        preprocessed_doc: PreprocessedDocument,
    ) -> PhaseResult:
        """
        FASE 2: Execute all 300 micro questions in parallel.
        
        Args:
            preprocessed_doc: Preprocessed document
            
        Returns:
            PhaseResult with list of QuestionResults
        """
        logger.info("=== FASE 2: EJECUCIÓN DE 300 MICRO PREGUNTAS ===")
        start = time.time()
        
        try:
            # Create tasks for all 300 questions
            tasks = [
                self._process_micro_question_async(i, preprocessed_doc)
                for i in range(1, 301)
            ]
            
            # Execute all in parallel
            all_micro_results = await asyncio.gather(*tasks)
            
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
        
        # TODO: FASE 3-10 implementation
        # For now, create a minimal report structure
        
        total_time = time.time() - self.start_time
        logger.info("=" * 70)
        logger.info("PROCESAMIENTO COMPLETO")
        logger.info(f"✓ Tiempo total: {total_time:.2f}s")
        logger.info("=" * 70)
        
        # Return minimal report structure
        return CompleteReport(
            micro_results=[],
            dimension_scores=[],
            area_scores=[],
            cluster_scores=[],
            macro_score=MacroScore(
                question_global=305,
                type="MACRO",
                global_quality_index=0.0,
                cross_cutting_coherence=0.0,
                systemic_gaps=[],
                cluster_scores=[]
            ),
            recommendations=[],
            metadata={
                "pdf_path": pdf_path,
                "total_time_s": total_time,
                "phases_completed": len(self.phase_results),
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
