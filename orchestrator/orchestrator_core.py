"""
Core Orchestrator module for coordinating the entire evaluation pipeline.

This module implements the central orchestrator that manages the end-to-end
execution of the policy evaluation system across 11 phases (0-10).
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .orchestrator_types import (
    CompleteReport,
    ErrorContext,
    OrchestratorConfig,
    PhaseMetrics,
    PhaseStatus,
    PreprocessedDocument,
    ProcessingPhase,
    ProcessingStatus,
    QuestionResult,
    ScoredResult,
    DimensionScore,
    AreaScore,
    ClusterScore,
    MacroScore,
)
from .contract_loader import JSONContractLoader, LoadResult

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class ValidationError(OrchestratorError):
    """Error during validation phase."""
    pass


class ConfigurationError(OrchestratorError):
    """Error in configuration."""
    pass


class Orchestrator:
    """
    Central orchestrator for the policy evaluation pipeline.
    
    Responsibilities:
    - Coordinate all 11 phases (0-10)
    - Validate configuration and contracts
    - Manage document ingestion
    - Coordinate question execution pool
    - Aggregate results at multiple levels
    - Generate comprehensive reports
    - Provide global abortability
    - Track metrics and progress
    """
    
    def __init__(
        self,
        monolith_path: str = "questionnaire_monolith.json",
        catalog_path: str = "rules/METODOS/metodos_completos_nivel3.json",
        config: Optional[OrchestratorConfig] = None,
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            monolith_path: Path to questionnaire monolith
            catalog_path: Path to method catalog
            config: Optional configuration override
        """
        self.monolith_path = Path(monolith_path)
        self.catalog_path = Path(catalog_path)
        self.config = config or OrchestratorConfig()
        
        # State management
        self.monolith: Optional[Dict[str, Any]] = None
        self.method_catalog: Optional[Dict[str, Any]] = None
        self.state: ProcessingStatus = self._init_state()
        self.abort_requested: bool = False
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Orchestrator initialized with config: {self.config}")
    
    def _init_state(self) -> ProcessingStatus:
        """Initialize processing state."""
        return ProcessingStatus(
            current_phase=ProcessingPhase.PHASE_0_VALIDATION,
            progress=0.0,
            questions_completed=0,
            questions_total=300,
            elapsed_time_seconds=0.0,
            estimated_time_remaining_seconds=None,
            phase_metrics={},
        )
    
    def process_document(self, pdf_path: str) -> CompleteReport:
        """
        Execute the complete pipeline end-to-end.
        
        This is the main entry point for document processing.
        
        Args:
            pdf_path: Path to the PDF document to process
            
        Returns:
            CompleteReport with all levels of analysis
            
        Raises:
            OrchestratorError: If a critical error occurs
        """
        logger.info(f"Starting document processing: {pdf_path}")
        start_time = time.time()
        
        try:
            # Phase 0: Validation
            self._execute_phase(
                ProcessingPhase.PHASE_0_VALIDATION,
                self.validate_configuration
            )
            
            # Phase 1: Ingestion
            preprocessed_doc = self._execute_phase(
                ProcessingPhase.PHASE_1_INGESTION,
                self.ingest_document,
                pdf_path
            )
            
            # Phase 2: Micro Question Execution
            question_results = self._execute_phase(
                ProcessingPhase.PHASE_2_EXECUTION,
                self.execute_all_micro_questions,
                preprocessed_doc
            )
            
            # Phase 3: Scoring
            scored_results = self._execute_phase(
                ProcessingPhase.PHASE_3_SCORING,
                self.score_all_questions,
                question_results
            )
            
            # Phase 4: Dimension Aggregation
            dimension_scores = self._execute_phase(
                ProcessingPhase.PHASE_4_DIMENSION_AGG,
                self.aggregate_dimensions,
                scored_results
            )
            
            # Phase 5: Area Aggregation
            area_scores = self._execute_phase(
                ProcessingPhase.PHASE_5_AREA_AGG,
                self.aggregate_areas,
                dimension_scores
            )
            
            # Phase 6: Cluster Aggregation
            cluster_scores = self._execute_phase(
                ProcessingPhase.PHASE_6_CLUSTER_AGG,
                self.aggregate_clusters,
                area_scores
            )
            
            # Phase 7: Macro Evaluation
            macro_score = self._execute_phase(
                ProcessingPhase.PHASE_7_MACRO_EVAL,
                self.evaluate_macro,
                cluster_scores
            )
            
            # Phase 8: Recommendations
            recommendations = self._execute_phase(
                ProcessingPhase.PHASE_8_RECOMMENDATIONS,
                self.generate_recommendations,
                macro_score
            )
            macro_score.recommendations = recommendations
            
            # Phase 9: Report Assembly
            report = self._execute_phase(
                ProcessingPhase.PHASE_9_ASSEMBLY,
                self.assemble_report,
                preprocessed_doc,
                macro_score,
                cluster_scores,
                area_scores,
                dimension_scores,
                scored_results
            )
            
            # Phase 10: Output Formatting (handled separately as needed)
            self._mark_phase_complete(ProcessingPhase.PHASE_10_FORMATTING)
            
            # Update final metrics
            self.state.elapsed_time_seconds = time.time() - start_time
            self.state.progress = 1.0
            
            logger.info(f"Processing complete in {self.state.elapsed_time_seconds:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            self._handle_critical_error(e)
            raise
    
    def _execute_phase(self, phase: ProcessingPhase, func, *args, **kwargs):
        """
        Execute a phase with metrics tracking and error handling.
        
        Args:
            phase: The phase being executed
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result from the function
        """
        if self.abort_requested:
            raise OrchestratorError("Processing aborted by user")
        
        logger.info(f"Starting {phase.name}")
        self.state.current_phase = phase
        
        # Initialize phase metrics
        metrics = PhaseMetrics(
            phase=phase,
            status=PhaseStatus.RUNNING,
            start_time=datetime.now()
        )
        self.state.phase_metrics[phase] = metrics
        
        try:
            result = func(*args, **kwargs)
            
            metrics.status = PhaseStatus.COMPLETED
            metrics.end_time = datetime.now()
            if metrics.start_time:
                metrics.duration_seconds = (
                    metrics.end_time - metrics.start_time
                ).total_seconds()
            
            logger.info(
                f"Completed {phase.name} in {metrics.duration_seconds:.2f}s"
            )
            
            return result
            
        except Exception as e:
            metrics.status = PhaseStatus.FAILED
            metrics.end_time = datetime.now()
            metrics.errors.append(str(e))
            
            logger.error(f"Phase {phase.name} failed: {e}")
            raise
    
    def _mark_phase_complete(self, phase: ProcessingPhase):
        """Mark a phase as completed without execution."""
        metrics = PhaseMetrics(
            phase=phase,
            status=PhaseStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        self.state.phase_metrics[phase] = metrics
    
    # ========================================================================
    # PHASE 0: VALIDATION
    # ========================================================================
    
    def validate_configuration(self) -> bool:
        """
        Validate configuration and load contracts.
        
        Verifies:
        - Monolith integrity
        - Method catalog completeness
        - Configuration correctness
        
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        logger.info("Validating configuration...")
        
        # Load monolith
        if not self.monolith_path.exists():
            raise ValidationError(f"Monolith not found: {self.monolith_path}")
        
        with open(self.monolith_path, 'r', encoding='utf-8') as f:
            self.monolith = json.load(f)
        
        # Validate monolith structure
        if 'blocks' not in self.monolith:
            raise ValidationError("Monolith missing 'blocks' key")
        
        blocks = self.monolith['blocks']
        if 'micro_questions' not in blocks:
            raise ValidationError("Monolith missing 'micro_questions'")
        
        # Verify question count
        micro_count = len(blocks['micro_questions'])
        if micro_count != 300:
            raise ValidationError(
                f"Expected 300 micro questions, found {micro_count}"
            )
        
        logger.info(f"Validated monolith: {micro_count} micro questions")
        
        # Load method catalog
        if not self.catalog_path.exists():
            raise ValidationError(f"Method catalog not found: {self.catalog_path}")
        
        with open(self.catalog_path, 'r', encoding='utf-8') as f:
            catalog_data = json.load(f)
        
        # The catalog may be the full document or just the methods
        if isinstance(catalog_data, dict):
            if 'methods_catalog' in catalog_data:
                self.method_catalog = catalog_data['methods_catalog']
            elif 'dimensions' in catalog_data:
                self.method_catalog = catalog_data['dimensions']
            else:
                # Try to use it as-is
                self.method_catalog = catalog_data
        else:
            self.method_catalog = catalog_data
        
        # Validate method catalog structure
        # The catalog might be a list of method packages or a dict with dimensions
        if isinstance(self.method_catalog, list):
            # It's a list of method packages - validate we have methods
            method_count = len(self.method_catalog)
            logger.info(f"Validated method catalog: {method_count} method packages")
        elif isinstance(self.method_catalog, dict) and 'dimensions' in self.method_catalog:
            dimensions = self.method_catalog['dimensions']
            if len(dimensions) != 6:
                raise ValidationError(
                    f"Expected 6 dimensions, found {len(dimensions)}"
                )
            
            # Verify each dimension has 5 questions
            for i, dim in enumerate(dimensions):
                if 'questions' not in dim:
                    raise ValidationError(f"Dimension {i} missing 'questions'")
                if len(dim['questions']) != 5:
                    raise ValidationError(
                        f"Dimension {i} should have 5 questions, has {len(dim['questions'])}"
                    )
            
            logger.info(f"Validated method catalog: {len(dimensions)} dimensions")
        else:
            # Assume it's valid if we got this far
            logger.info("Method catalog loaded (structure validation skipped)")
        
        return True
    
    # ========================================================================
    # PHASE 1: INGESTION
    # ========================================================================
    
    def ingest_document(self, pdf_path: str) -> PreprocessedDocument:
        """
        Ingest and preprocess the document.
        
        Args:
            pdf_path: Path to PDF document
            
        Returns:
            PreprocessedDocument ready for processing
        """
        logger.info(f"Ingesting document: {pdf_path}")
        
        # Placeholder implementation - would integrate with actual document ingestion
        # For now, return a mock preprocessed document
        doc = PreprocessedDocument(
            document_id=Path(pdf_path).stem,
            raw_text="",
            sentences=[],
            tables=[],
            metadata={
                "source_file": pdf_path,
                "ingestion_timestamp": datetime.now().isoformat(),
            }
        )
        
        logger.info(f"Document ingested: {doc.document_id}")
        return doc
    
    # ========================================================================
    # PHASE 2: MICRO QUESTION EXECUTION
    # ========================================================================
    
    def execute_all_micro_questions(
        self,
        preprocessed_doc: PreprocessedDocument
    ) -> List[QuestionResult]:
        """
        Execute all 300 micro questions in parallel.
        
        Args:
            preprocessed_doc: Preprocessed document
            
        Returns:
            List of 300 QuestionResults
        """
        logger.info("Executing 300 micro questions...")
        
        results = []
        failed_questions = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures: Dict[Future, int] = {}
            
            # Submit all questions
            for q_num in range(1, 301):
                future = executor.submit(
                    self._execute_single_question,
                    q_num,
                    preprocessed_doc
                )
                futures[future] = q_num
            
            # Collect results
            for future in as_completed(futures):
                q_num = futures[future]
                try:
                    result = future.result(timeout=self.config.default_question_timeout)
                    results.append(result)
                    self.state.questions_completed += 1
                    
                    # Update progress
                    self.state.progress = (
                        self.state.questions_completed / self.state.questions_total
                    )
                    
                    if self.state.questions_completed % 10 == 0:
                        logger.info(
                            f"Progress: {self.state.questions_completed}/300 "
                            f"({self.state.completion_percentage:.1f}%)"
                        )
                        
                except Exception as e:
                    logger.error(f"Question {q_num} failed: {e}")
                    failed_questions.append(q_num)
                    
                    # Create failed result
                    result = QuestionResult(
                        question_global=q_num,
                        base_slot=self._get_base_slot(q_num),
                        policy_area="",
                        dimension="",
                        evidence={},
                        raw_results={},
                        execution_time_ms=0.0,
                        status="FAILED",
                        errors=[str(e)]
                    )
                    results.append(result)
        
        # Check completion rate
        completion_rate = len(results) / 300
        if completion_rate < self.config.min_completion_rate:
            raise OrchestratorError(
                f"Completion rate {completion_rate:.2%} below minimum "
                f"{self.config.min_completion_rate:.2%}"
            )
        
        logger.info(
            f"Completed {len(results)} questions "
            f"({len(failed_questions)} failed)"
        )
        
        # Sort by question number
        results.sort(key=lambda r: r.question_global)
        
        return results
    
    def _execute_single_question(
        self,
        question_global: int,
        preprocessed_doc: PreprocessedDocument
    ) -> QuestionResult:
        """
        Execute a single question.
        
        This is a placeholder that would delegate to the Choreographer.
        
        Args:
            question_global: Question number (1-300)
            preprocessed_doc: Preprocessed document
            
        Returns:
            QuestionResult
        """
        # Placeholder implementation
        # In real implementation, would delegate to Choreographer
        base_slot = self._get_base_slot(question_global)
        
        return QuestionResult(
            question_global=question_global,
            base_slot=base_slot,
            policy_area=base_slot.split('-')[0],
            dimension=base_slot.split('-')[0],
            evidence={"placeholder": True},
            raw_results={},
            execution_time_ms=10.0,
            methods_executed=1,
        )
    
    def _get_base_slot(self, question_global: int) -> str:
        """Get base slot for a question number."""
        base_index = (question_global - 1) % 30
        dim_idx = base_index // 5 + 1
        q_idx = base_index % 5 + 1
        return f"D{dim_idx}-Q{q_idx}"
    
    # ========================================================================
    # PHASE 3: SCORING
    # ========================================================================
    
    def score_all_questions(
        self,
        question_results: List[QuestionResult]
    ) -> List[ScoredResult]:
        """
        Apply scoring to all question results.
        
        Args:
            question_results: List of question results
            
        Returns:
            List of scored results
        """
        logger.info("Scoring questions...")
        
        scored_results = []
        for result in question_results:
            # Placeholder scoring
            scored = ScoredResult(
                question_result=result,
                score=0.5 if result.status == "SUCCESS" else 0.0,
                confidence=0.8 if result.status == "SUCCESS" else 0.0,
            )
            scored_results.append(scored)
        
        logger.info(f"Scored {len(scored_results)} questions")
        return scored_results
    
    # ========================================================================
    # PHASE 4-7: AGGREGATIONS
    # ========================================================================
    
    def aggregate_dimensions(
        self,
        scored_results: List[ScoredResult]
    ) -> List[DimensionScore]:
        """Aggregate into 60 dimensions."""
        logger.info("Aggregating dimensions...")
        # Placeholder implementation
        return []
    
    def aggregate_areas(
        self,
        dimension_scores: List[DimensionScore]
    ) -> List[AreaScore]:
        """Aggregate into 10 policy areas."""
        logger.info("Aggregating areas...")
        # Placeholder implementation
        return []
    
    def aggregate_clusters(
        self,
        area_scores: List[AreaScore]
    ) -> List[ClusterScore]:
        """Aggregate into 4 clusters (MESO)."""
        logger.info("Aggregating clusters...")
        # Placeholder implementation
        return []
    
    def evaluate_macro(
        self,
        cluster_scores: List[ClusterScore]
    ) -> MacroScore:
        """Perform macro evaluation."""
        logger.info("Performing macro evaluation...")
        # Placeholder implementation
        return MacroScore(
            overall_score=0.5,
            overall_confidence=0.7,
            cluster_scores=cluster_scores,
            recommendations=[],
        )
    
    # ========================================================================
    # PHASE 8-9: REPORTING
    # ========================================================================
    
    def generate_recommendations(
        self,
        macro_score: MacroScore
    ) -> List[str]:
        """Generate recommendations."""
        logger.info("Generating recommendations...")
        # Placeholder implementation
        return ["Recommendation 1", "Recommendation 2"]
    
    def assemble_report(
        self,
        preprocessed_doc: PreprocessedDocument,
        macro_score: MacroScore,
        cluster_scores: List[ClusterScore],
        area_scores: List[AreaScore],
        dimension_scores: List[DimensionScore],
        scored_results: List[ScoredResult]
    ) -> CompleteReport:
        """Assemble complete report."""
        logger.info("Assembling report...")
        
        return CompleteReport(
            document_id=preprocessed_doc.document_id,
            macro_score=macro_score,
            cluster_scores=cluster_scores,
            area_scores=area_scores,
            dimension_scores=dimension_scores,
            question_results=scored_results,
            processing_metadata=self.state,
        )
    
    # ========================================================================
    # PHASE 10: OUTPUT FORMATTING
    # ========================================================================
    
    def format_outputs(
        self,
        report: CompleteReport
    ) -> Dict[str, bytes]:
        """
        Generate multiple output formats.
        
        Args:
            report: Complete report
            
        Returns:
            Dictionary mapping format to bytes
        """
        logger.info("Formatting outputs...")
        
        outputs = {}
        
        # JSON format
        outputs['json'] = json.dumps(
            {
                'document_id': report.document_id,
                'macro_score': report.macro_score.overall_score,
                'generation_timestamp': report.generation_timestamp.isoformat(),
            },
            indent=2
        ).encode('utf-8')
        
        return outputs
    
    # ========================================================================
    # STATE AND MONITORING
    # ========================================================================
    
    def get_processing_status(self) -> ProcessingStatus:
        """Get current processing status."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed processing metrics."""
        return {
            'phase_metrics': {
                phase.name: {
                    'status': metrics.status.value,
                    'duration': metrics.duration_seconds,
                    'items_processed': metrics.items_processed,
                    'items_failed': metrics.items_failed,
                    'success_rate': metrics.success_rate,
                }
                for phase, metrics in self.state.phase_metrics.items()
            },
            'total_questions': self.state.questions_total,
            'completed_questions': self.state.questions_completed,
            'progress': self.state.progress,
            'elapsed_time': self.state.elapsed_time_seconds,
        }
    
    def request_abort(self):
        """Request graceful abort of processing."""
        logger.warning("Abort requested")
        self.abort_requested = True
    
    # ========================================================================
    # ERROR HANDLING
    # ========================================================================
    
    def _handle_critical_error(self, error: Exception):
        """
        Handle critical errors that prevent continuation.
        
        Args:
            error: The exception that occurred
        """
        context = ErrorContext(
            phase=self.state.current_phase,
            component="Orchestrator",
            error_type=type(error).__name__,
            message=str(error),
            recoverable=False,
        )
        
        logger.error(
            f"Critical error in {context.phase.name}: "
            f"{context.error_type}: {context.message}"
        )
        
        # Update phase metrics
        if self.state.current_phase in self.state.phase_metrics:
            metrics = self.state.phase_metrics[self.state.current_phase]
            metrics.status = PhaseStatus.FAILED
            metrics.errors.append(context.message)


__all__ = [
    "Orchestrator",
    "OrchestratorError",
    "ValidationError",
    "ConfigurationError",
]
