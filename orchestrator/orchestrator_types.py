"""
Data types and enums for the Orchestrator module.

This module defines all the data structures used by the Orchestrator
for managing the end-to-end pipeline execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class PhaseStatus(Enum):
    """Status of a pipeline phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingPhase(Enum):
    """Pipeline processing phases."""
    PHASE_0_VALIDATION = 0
    PHASE_1_INGESTION = 1
    PHASE_2_EXECUTION = 2
    PHASE_3_SCORING = 3
    PHASE_4_DIMENSION_AGG = 4
    PHASE_5_AREA_AGG = 5
    PHASE_6_CLUSTER_AGG = 6
    PHASE_7_MACRO_EVAL = 7
    PHASE_8_RECOMMENDATIONS = 8
    PHASE_9_ASSEMBLY = 9
    PHASE_10_FORMATTING = 10


@dataclass
class PhaseMetrics:
    """Metrics for a single phase execution."""
    phase: ProcessingPhase
    status: PhaseStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate for this phase."""
        total = self.items_processed + self.items_failed
        if total == 0:
            return 1.0
        return self.items_processed / total


@dataclass
class ProcessingStatus:
    """Current status of the orchestrator processing."""
    current_phase: ProcessingPhase
    progress: float  # 0.0 to 1.0
    questions_completed: int
    questions_total: int
    elapsed_time_seconds: float
    estimated_time_remaining_seconds: Optional[float]
    phase_metrics: Dict[ProcessingPhase, PhaseMetrics] = field(default_factory=dict)
    
    @property
    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.progress >= 1.0
    
    @property
    def completion_percentage(self) -> float:
        """Get completion percentage."""
        return self.progress * 100.0


@dataclass
class PreprocessedDocument:
    """Document after preprocessing phase."""
    document_id: str
    raw_text: str
    sentences: List[str]
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    preprocessing_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionResult:
    """Result from executing a single question."""
    question_global: int
    base_slot: str
    policy_area: str
    dimension: str
    evidence: Dict[str, Any]
    raw_results: Dict[str, Any]
    execution_time_ms: float
    methods_executed: int = 0
    status: str = "SUCCESS"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ScoredResult:
    """Result after scoring a question."""
    question_result: QuestionResult
    score: float
    confidence: float
    scoring_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionScore:
    """Aggregated score for a dimension."""
    dimension_id: str
    dimension_name: str
    score: float
    confidence: float
    question_scores: List[ScoredResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AreaScore:
    """Aggregated score for a policy area."""
    area_id: str
    area_name: str
    score: float
    confidence: float
    dimension_scores: List[DimensionScore]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterScore:
    """Aggregated score for a cluster (MESO question)."""
    cluster_id: str
    cluster_name: str
    score: float
    confidence: float
    area_scores: List[AreaScore]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroScore:
    """Holistic macro evaluation."""
    overall_score: float
    overall_confidence: float
    cluster_scores: List[ClusterScore]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompleteReport:
    """Complete evaluation report with all levels."""
    document_id: str
    macro_score: MacroScore
    cluster_scores: List[ClusterScore]
    area_scores: List[AreaScore]
    dimension_scores: List[DimensionScore]
    question_results: List[ScoredResult]
    processing_metadata: ProcessingStatus
    generation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrchestratorConfig:
    """Configuration for the Orchestrator."""
    # Pool configuration
    max_workers: int = 50
    min_workers: int = 10
    
    # Timeouts (in seconds)
    default_question_timeout: float = 180.0
    complex_question_timeout: float = 300.0
    global_timeout: float = 3600.0
    
    # Retry configuration
    max_question_retries: int = 3
    retry_backoff_factor: float = 2.0
    
    # Tolerance settings
    min_completion_rate: float = 0.9
    allow_partial_report: bool = True
    
    # Resource limits
    memory_limit_per_worker: str = "2GB"
    cpu_cores_per_worker: int = 1
    
    # Monitoring
    progress_report_interval: int = 30
    enable_metrics: bool = True
    log_level: str = "INFO"
    
    # Phase control
    enable_phase_validation: bool = True
    abort_on_critical_failure: bool = True


@dataclass
class ErrorContext:
    """Context information for an error."""
    phase: ProcessingPhase
    component: str
    error_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    stacktrace: Optional[str] = None
    recoverable: bool = True


__all__ = [
    "PhaseStatus",
    "ProcessingPhase",
    "PhaseMetrics",
    "ProcessingStatus",
    "PreprocessedDocument",
    "QuestionResult",
    "ScoredResult",
    "DimensionScore",
    "AreaScore",
    "ClusterScore",
    "MacroScore",
    "CompleteReport",
    "OrchestratorConfig",
    "ErrorContext",
]
