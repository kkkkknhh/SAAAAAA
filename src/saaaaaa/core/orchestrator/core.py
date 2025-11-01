"""Core orchestrator classes, data models, and execution engine.

This module contains the fundamental building blocks for orchestration:
- Data models (PreprocessedDocument, Evidence, PhaseResult, etc.)
- Abort signaling (AbortSignal, AbortRequested)
- Resource management (ResourceLimits, PhaseInstrumentation)
- Method execution (MethodExecutor)
- Orchestrator (the main 11-phase orchestration engine)

The Orchestrator is the sole owner of the provider; processors and executors
receive pre-prepared data.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import statistics
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from saaaaaa.analysis.recommendation_engine import RecommendationEngine

from .arg_router import ArgRouter, ArgRouterError, ArgumentValidationError
from .class_registry import build_class_registry, ClassRegistryError

if TYPE_CHECKING:
    from document_ingestion import PreprocessedDocument as IngestionPreprocessedDocument

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedDocument:
    """Orchestrator representation of a processed document.
    
    This is the normalized document format used internally by the orchestrator.
    It can be constructed from ingestion payloads or created directly.
    """
    document_id: str
    raw_text: str
    sentences: List[Any]
    tables: List[Any]
    metadata: Dict[str, Any]

    @staticmethod
    def _dataclass_to_dict(value: Any) -> Any:
        """Convert a dataclass to a dictionary if applicable."""
        if is_dataclass(value):
            return asdict(value)
        return value

    @classmethod
    def ensure(
        cls, document: Any, *, document_id: Optional[str] = None
    ) -> PreprocessedDocument:
        """Normalize arbitrary ingestion payloads into orchestrator documents."""
        if isinstance(document, cls):
            return document

        if hasattr(document, "raw_document") and hasattr(document, "full_text"):
            return cls._from_ingestion(document, document_id=document_id)

        raise TypeError(
            "Unsupported preprocessed document payload: "
            f"expected orchestrator or document_ingestion schema, got {type(document)!r}"
        )

    @classmethod
    def _from_ingestion(
        cls,
        document: Union[IngestionPreprocessedDocument, Any],
        *,
        document_id: Optional[str] = None,
    ) -> PreprocessedDocument:
        """Build an orchestrator document from the ingestion schema."""
        raw_doc = getattr(document, "raw_document", None)
        derived_id: Optional[str] = document_id or getattr(document, "document_id", None)

        if not derived_id and raw_doc is not None:
            derived_id = getattr(raw_doc, "file_name", None)

        if not derived_id and hasattr(document, "preprocessing_metadata"):
            derived_id = getattr(
                document.preprocessing_metadata, "get", lambda _key, _default=None: None
            )("document_id")

        if not derived_id and hasattr(document, "metadata"):
            derived_id = getattr(
                document.metadata, "get", lambda _key, _default=None: None
            )("document_id")

        if not derived_id and raw_doc is not None:
            source_path = getattr(raw_doc, "file_path", "")
            if source_path:
                derived_id = os.path.splitext(os.path.basename(str(source_path)))[0]

        if not derived_id:
            derived_id = "document_1"

        metadata: Dict[str, Any] = {}
        preprocessing_block: Optional[Dict[str, Any]] = None
        if hasattr(document, "preprocessing_metadata"):
            preprocessing_metadata = document.preprocessing_metadata
            if isinstance(preprocessing_metadata, dict):
                preprocessing_block = preprocessing_metadata
            else:
                maybe_dict = cls._dataclass_to_dict(preprocessing_metadata)
                if isinstance(maybe_dict, dict):
                    preprocessing_block = maybe_dict
        if preprocessing_block:
            metadata["preprocessing_metadata"] = preprocessing_block

        sentence_metadata = getattr(document, "sentence_metadata", None)
        if sentence_metadata is not None:
            metadata["sentence_metadata"] = list(sentence_metadata)

        indexes = getattr(document, "indexes", None)
        if indexes is not None:
            metadata["indexes"] = cls._dataclass_to_dict(indexes)

        structured_text = getattr(document, "structured_text", None)
        if structured_text is not None:
            metadata["structured_text"] = cls._dataclass_to_dict(structured_text)

        language = getattr(document, "language", None)
        if language:
            metadata["language"] = language

        raw_doc_dict = cls._dataclass_to_dict(raw_doc) if raw_doc is not None else None
        if isinstance(raw_doc_dict, dict):
            metadata.setdefault("raw_document", raw_doc_dict)
            source_path = raw_doc_dict.get("file_path")
            if source_path:
                metadata.setdefault("source_path", source_path)

        metadata.setdefault("document_id", str(derived_id))
        metadata.setdefault("adapter_source", "document_ingestion.PreprocessedDocument")

        return cls(
            document_id=str(derived_id),
            raw_text=getattr(document, "full_text", "") or "",
            sentences=list(getattr(document, "sentences", [])),
            tables=list(getattr(document, "tables", [])),
            metadata=metadata,
        )


@dataclass
class Evidence:
    """Evidence container for orchestrator results."""
    modality: str
    elements: List[Any] = field(default_factory=list)
    raw_results: Dict[str, Any] = field(default_factory=dict)


class AbortRequested(RuntimeError):
    """Raised when an abort signal is triggered during orchestration."""


class AbortSignal:
    """Thread-safe abort signal shared across orchestration phases."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: Optional[str] = None
        self._timestamp: Optional[datetime] = None

    def abort(self, reason: str) -> None:
        """Trigger an abort with a reason and timestamp."""
        if not reason:
            reason = "Abort requested"
        with self._lock:
            if not self._event.is_set():
                self._event.set()
                self._reason = reason
                self._timestamp = datetime.utcnow()

    def is_aborted(self) -> bool:
        """Check whether abort has been triggered."""
        return self._event.is_set()

    def get_reason(self) -> Optional[str]:
        """Return the abort reason if set."""
        with self._lock:
            return self._reason

    def get_timestamp(self) -> Optional[datetime]:
        """Return the abort timestamp if set."""
        with self._lock:
            return self._timestamp

    def reset(self) -> None:
        """Clear the abort signal."""
        with self._lock:
            self._event.clear()
            self._reason = None
            self._timestamp = None


class ResourceLimits:
    """Runtime resource guard with adaptive worker prediction."""

    def __init__(
        self,
        max_memory_mb: Optional[float] = 4096.0,
        max_cpu_percent: float = 85.0,
        max_workers: int = 32,
        min_workers: int = 4,
        hard_max_workers: int = 64,
        history: int = 120,
    ) -> None:
        self.max_memory_mb = max_memory_mb
        self.max_cpu_percent = max_cpu_percent
        self.min_workers = max(1, min_workers)
        self.hard_max_workers = max(self.min_workers, hard_max_workers)
        self._max_workers = max(self.min_workers, min(max_workers, self.hard_max_workers))
        self._usage_history: deque[Dict[str, float]] = deque(maxlen=history)
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_limit = self._max_workers
        self._async_lock: Optional[asyncio.Lock] = None
        self._psutil = None
        self._psutil_process = None
        try:  # pragma: no cover - optional dependency
            import psutil  # type: ignore[import-untyped]

            self._psutil = psutil
            self._psutil_process = psutil.Process(os.getpid())
        except Exception:  # pragma: no cover - psutil missing
            self._psutil = None
            self._psutil_process = None

    @property
    def max_workers(self) -> int:
        """Return the current worker budget."""
        return self._max_workers

    def attach_semaphore(self, semaphore: asyncio.Semaphore) -> None:
        """Attach an asyncio semaphore for budget control."""
        self._semaphore = semaphore
        self._semaphore_limit = self._max_workers

    async def apply_worker_budget(self) -> int:
        """Apply the current worker budget to the semaphore."""
        if self._semaphore is None:
            return self._max_workers

        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            desired = self._max_workers
            current = self._semaphore_limit
            if desired > current:
                for _ in range(desired - current):
                    self._semaphore.release()
            elif desired < current:
                reduction = current - desired
                for _ in range(reduction):
                    await self._semaphore.acquire()
            self._semaphore_limit = desired
            return self._max_workers

    def _record_usage(self, usage: Dict[str, float]) -> None:
        """Record resource usage and predict worker budget."""
        self._usage_history.append(usage)
        self._predict_worker_budget()

    def _predict_worker_budget(self) -> None:
        """Adjust worker budget based on recent resource usage."""
        if len(self._usage_history) < 5:
            return

        cpu_vals = [entry["cpu_percent"] for entry in self._usage_history]
        mem_vals = [entry["memory_percent"] for entry in self._usage_history]
        recent_cpu = cpu_vals[-5:]
        recent_mem = mem_vals[-5:]
        avg_cpu = statistics.mean(recent_cpu)
        avg_mem = statistics.mean(recent_mem)

        new_budget = self._max_workers
        if self.max_cpu_percent and avg_cpu > self.max_cpu_percent * 0.95:
            new_budget = max(self.min_workers, self._max_workers - 1)
        elif self.max_memory_mb and avg_mem > 90.0:
            new_budget = max(self.min_workers, self._max_workers - 1)
        elif avg_cpu < self.max_cpu_percent * 0.6 and avg_mem < 70.0:
            new_budget = min(self.hard_max_workers, self._max_workers + 1)

        self._max_workers = max(self.min_workers, min(new_budget, self.hard_max_workers))

    def check_memory_exceeded(
        self, usage: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if memory limit has been exceeded."""
        usage = usage or self.get_resource_usage()
        exceeded = False
        if self.max_memory_mb is not None:
            exceeded = usage.get("rss_mb", 0.0) > self.max_memory_mb
        return exceeded, usage

    def check_cpu_exceeded(
        self, usage: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """Check if CPU limit has been exceeded."""
        usage = usage or self.get_resource_usage()
        exceeded = False
        if self.max_cpu_percent:
            exceeded = usage.get("cpu_percent", 0.0) > self.max_cpu_percent
        return exceeded, usage

    def get_resource_usage(self) -> Dict[str, float]:
        """Capture current resource usage metrics."""
        timestamp = datetime.utcnow().isoformat()
        cpu_percent = 0.0
        memory_percent = 0.0
        rss_mb = 0.0

        if self._psutil:
            try:  # pragma: no cover - psutil branch
                cpu_percent = float(self._psutil.cpu_percent(interval=None))
                virtual_memory = self._psutil.virtual_memory()
                memory_percent = float(virtual_memory.percent)
                if self._psutil_process is not None:
                    rss_mb = float(self._psutil_process.memory_info().rss / (1024 * 1024))
            except Exception:
                cpu_percent = 0.0
        else:
            try:
                load1, _, _ = os.getloadavg()
                cpu_percent = float(min(100.0, load1 * 100))
            except OSError:
                cpu_percent = 0.0
            try:
                import resource

                usage_info = resource.getrusage(resource.RUSAGE_SELF)
                rss_mb = float(usage_info.ru_maxrss / 1024)
            except Exception:
                rss_mb = 0.0

        usage = {
            "timestamp": timestamp,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "rss_mb": rss_mb,
            "worker_budget": float(self._max_workers),
        }
        self._record_usage(usage)
        return usage

    def get_usage_history(self) -> List[Dict[str, float]]:
        """Return the recorded usage history."""
        return list(self._usage_history)


class PhaseInstrumentation:
    """Collects granular telemetry for each orchestration phase."""

    def __init__(
        self,
        phase_id: int,
        name: str,
        items_total: Optional[int] = None,
        snapshot_interval: int = 10,
        resource_limits: Optional[ResourceLimits] = None,
    ) -> None:
        self.phase_id = phase_id
        self.name = name
        self.items_total = items_total or 0
        self.snapshot_interval = max(1, snapshot_interval)
        self.resource_limits = resource_limits
        self.items_processed = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.warnings: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
        self.resource_snapshots: List[Dict[str, Any]] = []
        self.latencies: List[float] = []
        self.anomalies: List[Dict[str, Any]] = []

    def start(self, items_total: Optional[int] = None) -> None:
        """Mark the start of phase execution."""
        if items_total is not None:
            self.items_total = items_total
        self.start_time = time.perf_counter()

    def increment(self, count: int = 1, latency: Optional[float] = None) -> None:
        """Increment processed item count and optionally record latency."""
        self.items_processed += count
        if latency is not None:
            self.latencies.append(latency)
            self._detect_latency_anomaly(latency)
        if self.resource_limits and self.should_snapshot():
            self.capture_resource_snapshot()

    def should_snapshot(self) -> bool:
        """Determine if a resource snapshot should be captured."""
        if self.items_total == 0:
            return False
        if self.items_processed == 0:
            return False
        return self.items_processed % self.snapshot_interval == 0

    def capture_resource_snapshot(self) -> None:
        """Capture a resource usage snapshot."""
        if not self.resource_limits:
            return
        snapshot = self.resource_limits.get_resource_usage()
        snapshot["items_processed"] = self.items_processed
        self.resource_snapshots.append(snapshot)

    def record_warning(self, category: str, message: str, **extra: Any) -> None:
        """Record a warning during phase execution."""
        entry = {
            "category": category,
            "message": message,
            **extra,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.warnings.append(entry)

    def record_error(self, category: str, message: str, **extra: Any) -> None:
        """Record an error during phase execution."""
        entry = {
            "category": category,
            "message": message,
            **extra,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.errors.append(entry)

    def _detect_latency_anomaly(self, latency: float) -> None:
        """Detect latency anomalies using statistical thresholds."""
        if len(self.latencies) < 5:
            return
        mean_latency = statistics.mean(self.latencies)
        std_latency = statistics.pstdev(self.latencies) or 0.0
        threshold = mean_latency + (3 * std_latency)
        if std_latency and latency > threshold:
            self.anomalies.append(
                {
                    "type": "latency_spike",
                    "latency": latency,
                    "mean": mean_latency,
                    "std": std_latency,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    def complete(self) -> None:
        """Mark the end of phase execution."""
        self.end_time = time.perf_counter()

    def duration_ms(self) -> Optional[float]:
        """Return the phase duration in milliseconds."""
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def progress(self) -> Optional[float]:
        """Return the progress fraction (0.0 to 1.0)."""
        if not self.items_total:
            return None
        return min(1.0, self.items_processed / float(self.items_total))

    def throughput(self) -> Optional[float]:
        """Return items processed per second."""
        if self.start_time is None:
            return None
        elapsed = (
            (time.perf_counter() - self.start_time)
            if self.end_time is None
            else (self.end_time - self.start_time)
        )
        if not elapsed:
            return None
        return self.items_processed / elapsed

    def latency_histogram(self) -> Dict[str, Optional[float]]:
        """Return latency percentiles."""
        if not self.latencies:
            return {"p50": None, "p95": None, "p99": None}
        sorted_latencies = sorted(self.latencies)

        def percentile(p: float) -> float:
            if not sorted_latencies:
                return 0.0
            k = (len(sorted_latencies) - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, len(sorted_latencies) - 1)
            if f == c:
                return sorted_latencies[int(k)]
            d0 = sorted_latencies[f] * (c - k)
            d1 = sorted_latencies[c] * (k - f)
            return d0 + d1

        return {
            "p50": percentile(50.0),
            "p95": percentile(95.0),
            "p99": percentile(99.0),
        }

    def build_metrics(self) -> Dict[str, Any]:
        """Build a metrics summary dictionary."""
        return {
            "phase_id": self.phase_id,
            "name": self.name,
            "duration_ms": self.duration_ms(),
            "items_processed": self.items_processed,
            "items_total": self.items_total,
            "progress": self.progress(),
            "throughput": self.throughput(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "resource_snapshots": list(self.resource_snapshots),
            "latency_histogram": self.latency_histogram(),
            "anomalies": list(self.anomalies),
        }


@dataclass
class PhaseResult:
    """Result of a single orchestration phase."""
    success: bool
    phase_id: str
    data: Any
    error: Optional[Exception]
    duration_ms: float
    mode: str
    aborted: bool = False


@dataclass
class MicroQuestionRun:
    """Result of executing a single micro-question."""
    question_id: str
    question_global: int
    base_slot: str
    metadata: Dict[str, Any]
    evidence: Optional[Evidence]
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    aborted: bool = False


@dataclass
class ScoredMicroQuestion:
    """Scored micro-question result."""
    question_id: str
    question_global: int
    base_slot: str
    score: Optional[float]
    normalized_score: Optional[float]
    quality_level: Optional[str]
    evidence: Optional[Evidence]
    scoring_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class MethodExecutor:
    """Execute catalog methods using ArgRouter and class registry.
    
    This executor builds the class registry, instantiates all required classes,
    and delegates signature/kwargs handling to ArgRouter. No hardcoded logic.
    """

    def __init__(self) -> None:
        # Build the class registry
        try:
            registry = build_class_registry()
        except (ClassRegistryError, ModuleNotFoundError, ImportError) as exc:
            logger.warning("Some modules unavailable - operating in limited mode: %s", exc)
            registry = {}

        # Instantiate all classes
        self.instances: Dict[str, Any] = {}
        ontology_instance = None

        for class_name, cls in registry.items():
            try:
                # Special handling for MunicipalOntology - shared instance
                if class_name == "MunicipalOntology":
                    ontology_instance = cls()
                    self.instances[class_name] = ontology_instance
                # Classes that need ontology
                elif class_name in ("SemanticAnalyzer", "PerformanceAnalyzer", "TextMiningEngine"):
                    if ontology_instance is not None:
                        self.instances[class_name] = cls(ontology_instance)
                    else:
                        logger.warning(
                            "Cannot instantiate %s: MunicipalOntology not available", class_name
                        )
                # PolicyTextProcessor needs ProcessorConfig
                elif class_name == "PolicyTextProcessor":
                    try:
                        from saaaaaa.processing.policy_processor import ProcessorConfig
                        self.instances[class_name] = cls(ProcessorConfig())
                    except ImportError:
                        logger.warning("Cannot instantiate PolicyTextProcessor: ProcessorConfig unavailable")
                else:
                    # Standard instantiation
                    self.instances[class_name] = cls()
            except Exception as exc:
                logger.error("Failed to instantiate %s: %s", class_name, exc)

        # Create ArgRouter with the registry
        self._router = ArgRouter(registry)

    def execute(self, class_name: str, method_name: str, **kwargs: Any) -> Any:
        """Execute a method from the catalog.
        
        Args:
            class_name: Name of the class
            method_name: Name of the method to execute
            **kwargs: Keyword arguments to pass to the method
            
        Returns:
            The method's return value
            
        Raises:
            ArgRouterError: If routing fails
            AttributeError: If method doesn't exist
        """
        instance = self.instances.get(class_name)
        if not instance:
            logger.warning("No instance available for class %s", class_name)
            return None

        try:
            method = getattr(instance, method_name)
        except AttributeError:
            logger.error("Class %s has no method %s", class_name, method_name)
            raise

        try:
            args, routed_kwargs = self._router.route(class_name, method_name, dict(kwargs))
            return method(*args, **routed_kwargs)
        except (ArgRouterError, ArgumentValidationError) as exc:
            logger.exception("Argument routing failed for %s.%s", class_name, method_name)
            raise
        except Exception as exc:
            logger.exception("Method execution failed for %s.%s", class_name, method_name)
            raise


class Orchestrator:
    """Robust 11-phase orchestrator with abort support and resource control.
    
    The Orchestrator owns the provider and prepares all data for processors
    and executors. It executes 11 phases synchronously or asynchronously,
    with full instrumentation and abort capability.
    """

    FASES: List[Tuple[int, str, str, str]] = [
        (0, "sync", "_load_configuration", "FASE 0 - Validación de Configuración"),
        (1, "sync", "_ingest_document", "FASE 1 - Ingestión de Documento"),
        (2, "async", "_execute_micro_questions_async", "FASE 2 - Micro Preguntas"),
        (3, "async", "_score_micro_results_async", "FASE 3 - Scoring Micro"),
        (4, "async", "_aggregate_dimensions_async", "FASE 4 - Agregación Dimensiones"),
        (5, "async", "_aggregate_policy_areas_async", "FASE 5 - Agregación Áreas"),
        (6, "sync", "_aggregate_clusters", "FASE 6 - Agregación Clústeres"),
        (7, "sync", "_evaluate_macro", "FASE 7 - Evaluación Macro"),
        (8, "async", "_generate_recommendations", "FASE 8 - Recomendaciones"),
        (9, "sync", "_assemble_report", "FASE 9 - Ensamblado de Reporte"),
        (10, "async", "_format_and_export", "FASE 10 - Formateo y Exportación"),
    ]

    PHASE_ITEM_TARGETS: Dict[int, int] = {
        0: 1,
        1: 1,
        2: 300,
        3: 300,
        4: 60,
        5: 10,
        6: 4,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
    }

    PHASE_OUTPUT_KEYS: Dict[int, str] = {
        0: "config",
        1: "document",
        2: "micro_results",
        3: "scored_results",
        4: "dimension_scores",
        5: "policy_area_scores",
        6: "cluster_scores",
        7: "macro_result",
        8: "recommendations",
        9: "report",
        10: "export_payload",
    }

    PHASE_ARGUMENT_KEYS: Dict[int, List[str]] = {
        1: ["pdf_path", "config"],
        2: ["document", "config"],
        3: ["micro_results", "config"],
        4: ["scored_results", "config"],
        5: ["dimension_scores", "config"],
        6: ["policy_area_scores", "config"],
        7: ["cluster_scores", "config"],
        8: ["macro_result", "config"],
        9: ["recommendations", "config"],
        10: ["report", "config"],
    }

    def __init__(
        self,
        catalog_path: str = "rules/METODOS/metodos_completos_nivel3.json",
        monolith_path: str = "questionnaire_monolith.json",
        method_map_path: str = "COMPLETE_METHOD_CLASS_MAP.json",
        schema_path: Optional[str] = "schemas/questionnaire.schema.json",
        resource_limits: Optional[ResourceLimits] = None,
        resource_snapshot_interval: int = 10,
    ) -> None:
        """Initialize the orchestrator.
        
        Args:
            catalog_path: Path to method catalog JSON
            monolith_path: Path to questionnaire monolith JSON  
            method_map_path: Path to method-class mapping JSON
            schema_path: Optional path to questionnaire schema
            resource_limits: Resource limit configuration
            resource_snapshot_interval: Interval for resource snapshots
        """
        self.catalog_path = self._resolve_path(catalog_path)
        self.monolith_path = self._resolve_path(monolith_path)
        self.method_map_path = self._resolve_path(method_map_path)
        self.schema_path = self._resolve_path(schema_path) if schema_path else None
        self.resource_limits = resource_limits or ResourceLimits()
        self.resource_snapshot_interval = max(1, resource_snapshot_interval)

        with open(self.catalog_path) as f:
            self.catalog = json.load(f)

        self.executor = MethodExecutor()
        
        # Import executors from the executors module
        from . import executors
        
        self.executors = {
            "D1-Q1": executors.D1Q1_Executor,
            "D1-Q2": executors.D1Q2_Executor,
            "D1-Q3": executors.D1Q3_Executor,
            "D1-Q4": executors.D1Q4_Executor,
            "D1-Q5": executors.D1Q5_Executor,
            "D2-Q1": executors.D2Q1_Executor,
            "D2-Q2": executors.D2Q2_Executor,
            "D2-Q3": executors.D2Q3_Executor,
            "D2-Q4": executors.D2Q4_Executor,
            "D2-Q5": executors.D2Q5_Executor,
            "D3-Q1": executors.D3Q1_Executor,
            "D3-Q2": executors.D3Q2_Executor,
            "D3-Q3": executors.D3Q3_Executor,
            "D3-Q4": executors.D3Q4_Executor,
            "D3-Q5": executors.D3Q5_Executor,
            "D4-Q1": executors.D4Q1_Executor,
            "D4-Q2": executors.D4Q2_Executor,
            "D4-Q3": executors.D4Q3_Executor,
            "D4-Q4": executors.D4Q4_Executor,
            "D4-Q5": executors.D4Q5_Executor,
            "D5-Q1": executors.D5Q1_Executor,
            "D5-Q2": executors.D5Q2_Executor,
            "D5-Q3": executors.D5Q3_Executor,
            "D5-Q4": executors.D5Q4_Executor,
            "D5-Q5": executors.D5Q5_Executor,
            "D6-Q1": executors.D6Q1_Executor,
            "D6-Q2": executors.D6Q2_Executor,
            "D6-Q3": executors.D6Q3_Executor,
            "D6-Q4": executors.D6Q4_Executor,
            "D6-Q5": executors.D6Q5_Executor,
        }

        self.abort_signal = AbortSignal()
        self.phase_results: List[PhaseResult] = []
        self._phase_instrumentation: Dict[int, PhaseInstrumentation] = {}
        self._phase_status: Dict[int, str] = {
            phase_id: "not_started" for phase_id, *_ in self.FASES
        }
        self._phase_outputs: Dict[int, Any] = {}
        self._context: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

        # Initialize RecommendationEngine for 3-level recommendations
        try:
            # Try to load enhanced rules first (v2.0), fallback to v1.0
            try:
                self.recommendation_engine = RecommendationEngine(
                    rules_path="config/recommendation_rules_enhanced.json",
                    schema_path="rules/recommendation_rules_enhanced.schema.json"
                )
                logger.info("RecommendationEngine initialized with enhanced v2.0 rules")
            except Exception as e_enhanced:
                logger.info(f"Enhanced rules not available ({e_enhanced}), falling back to v1.0")
                self.recommendation_engine = RecommendationEngine(
                    rules_path="config/recommendation_rules.json",
                    schema_path="rules/recommendation_rules.schema.json"
                )
                logger.info("RecommendationEngine initialized with v1.0 rules")
        except Exception as e:
            logger.warning(f"Failed to initialize RecommendationEngine: {e}")
            self.recommendation_engine = None

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        """Resolve a relative or absolute path, searching multiple candidate locations."""
        if path is None:
            return None

        candidates = [path]
        if not os.path.isabs(path):
            base_dir = os.path.dirname(__file__)
            candidates.append(os.path.join(base_dir, path))
            candidates.append(os.path.join(os.getcwd(), path))
            if not path.startswith("rules"):
                candidates.append(os.path.join(os.getcwd(), "rules", "METODOS", path))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        return path

    def process_development_plan(
            self, pdf_path: str, preprocessed_document: Optional[Any] = None
    ) -> List[PhaseResult]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("process_development_plan() debe ejecutarse fuera de un loop asyncio activo")
        return asyncio.run(
            self.process_development_plan_async(
                pdf_path, preprocessed_document=preprocessed_document
            )
        )

    async def process_development_plan_async(
            self, pdf_path: str, preprocessed_document: Optional[Any] = None
    ) -> List[PhaseResult]:
        self.reset_abort()
        self.phase_results = []
        self._phase_instrumentation = {}
        self._phase_outputs = {}
        self._context = {"pdf_path": pdf_path}
        if preprocessed_document is not None:
            self._context["preprocessed_override"] = preprocessed_document
        self._phase_status = {phase_id: "not_started" for phase_id, *_ in self.FASES}
        self._start_time = time.perf_counter()

        for phase_id, mode, handler_name, phase_label in self.FASES:
            self._ensure_not_aborted()
            handler = getattr(self, handler_name)
            instrumentation = PhaseInstrumentation(
                phase_id=phase_id,
                name=phase_label,
                items_total=self.PHASE_ITEM_TARGETS.get(phase_id),
                snapshot_interval=self.resource_snapshot_interval,
                resource_limits=self.resource_limits,
            )
            instrumentation.start(items_total=self.PHASE_ITEM_TARGETS.get(phase_id))
            self._phase_instrumentation[phase_id] = instrumentation
            self._phase_status[phase_id] = "running"

            args = [self._context[key] for key in self.PHASE_ARGUMENT_KEYS.get(phase_id, [])]

            success = False
            data: Any = None
            error: Optional[Exception] = None
            try:
                if mode == "sync":
                    data = handler(*args)
                else:
                    data = await handler(*args)
                success = True
            except AbortRequested as exc:
                error = exc
                success = False
                instrumentation.record_warning("abort", str(exc))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Fase %s falló", phase_label)
                error = exc
                success = False
                instrumentation.record_error("exception", str(exc))
                self.request_abort(f"Fase {phase_id} falló: {exc}")
            finally:
                instrumentation.complete()

            aborted = self.abort_signal.is_aborted()
            duration_ms = instrumentation.duration_ms() or 0.0
            phase_result = PhaseResult(
                success=success and not aborted,
                phase_id=str(phase_id),
                data=data,
                error=error,
                duration_ms=duration_ms,
                mode=mode,
                aborted=aborted,
            )
            self.phase_results.append(phase_result)

            if success and not aborted:
                self._phase_outputs[phase_id] = data
                out_key = self.PHASE_OUTPUT_KEYS.get(phase_id)
                if out_key:
                    self._context[out_key] = data
                self._phase_status[phase_id] = "completed"
            elif aborted:
                self._phase_status[phase_id] = "aborted"
                break
            else:
                self._phase_status[phase_id] = "failed"
                break

        return self.phase_results

    def get_processing_status(self) -> Dict[str, Any]:
        if self._start_time is None:
            status = "not_started"
            elapsed = 0.0
            completed_flag = False
        else:
            aborted = self.abort_signal.is_aborted()
            status = "aborted" if aborted else "running"
            elapsed = time.perf_counter() - self._start_time
            completed_flag = all(state == "completed" for state in self._phase_status.values()) and not aborted

        completed = sum(1 for state in self._phase_status.values() if state == "completed")
        total = len(self.FASES)
        overall_progress = completed / total if total else 0.0

        phase_progress = {
            str(phase_id): instr.progress()
            for phase_id, instr in self._phase_instrumentation.items()
        }

        resource_usage = self.resource_limits.get_resource_usage() if self._start_time else {}

        return {
            "status": status,
            "overall_progress": overall_progress,
            "phase_progress": phase_progress,
            "elapsed_time_s": elapsed,
            "resource_usage": resource_usage,
            "abort_status": self.abort_signal.is_aborted(),
            "abort_reason": self.abort_signal.get_reason(),
            "completed": completed_flag,
        }

    def get_phase_metrics(self) -> Dict[str, Any]:
        return {
            str(phase_id): instr.build_metrics()
            for phase_id, instr in self._phase_instrumentation.items()
        }

    async def monitor_progress_async(self, poll_interval: float = 2.0):
        while True:
            status = self.get_processing_status()
            yield status
            if status["status"] != "running":
                break
            await asyncio.sleep(poll_interval)

    def abort_handler(self, reason: str) -> None:
        self.request_abort(reason)

    def health_check(self) -> Dict[str, Any]:
        usage = self.resource_limits.get_resource_usage()
        cpu_headroom = max(0.0, self.resource_limits.max_cpu_percent - usage.get("cpu_percent", 0.0))
        mem_headroom = max(0.0, (self.resource_limits.max_memory_mb or 0.0) - usage.get("rss_mb", 0.0))
        score = max(0.0, min(100.0, (cpu_headroom / max(1.0, self.resource_limits.max_cpu_percent)) * 50.0))
        if self.resource_limits.max_memory_mb:
            score += max(0.0, min(50.0, (mem_headroom / max(1.0, self.resource_limits.max_memory_mb)) * 50.0))
        score = min(100.0, score)
        if self.abort_signal.is_aborted():
            score = min(score, 20.0)
        return {"score": score, "resource_usage": usage, "abort": self.abort_signal.is_aborted()}

    def _load_configuration(self) -> Dict[str, Any]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[0]
        start = time.perf_counter()

        with open(self.monolith_path) as f:
            monolith = json.load(f)

        sha256 = hashlib.sha256()
        with open(self.monolith_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        monolith_hash = sha256.hexdigest()

        micro_questions: List[Dict[str, Any]] = monolith["blocks"].get("micro_questions", [])
        meso_questions: List[Dict[str, Any]] = monolith["blocks"].get("meso_questions", [])
        macro_question: Dict[str, Any] = monolith["blocks"].get("macro_question", {})

        question_total = len(micro_questions) + len(meso_questions) + (1 if macro_question else 0)
        if question_total != 305:
            message = f"Conteo de preguntas inesperado: {question_total}"
            instrumentation.record_error("integrity", message, expected=305, found=question_total)
            raise ValueError(message)

        structure_report = self._validate_contract_structure(monolith, instrumentation)

        method_summary: Dict[str, Any] = {}
        if self.method_map_path and os.path.exists(self.method_map_path):
            with open(self.method_map_path) as f:
                method_map = json.load(f)
            summary = method_map.get("summary", {})
            total_methods = summary.get("total_methods")
            if total_methods != 416:
                instrumentation.record_error(
                    "catalog",
                    "Total de métodos inesperado",
                    expected=416,
                    found=total_methods,
                )
            method_summary = {
                "total_methods": total_methods,
                "metadata": summary,
            }

        schema_report: Dict[str, Any] = {"errors": []}
        if self.schema_path and os.path.exists(self.schema_path):
            try:  # pragma: no cover - optional dependency
                import jsonschema

                with open(self.schema_path) as f:
                    schema = json.load(f)

                validator = jsonschema.Draft202012Validator(schema)
                schema_errors = [
                    {
                        "path": list(error.path),
                        "message": error.message,
                    }
                    for error in validator.iter_errors(monolith)
                ]
                if schema_errors:
                    for error in schema_errors:
                        instrumentation.record_error("schema", error["message"], path=error["path"])
                    schema_report["errors"] = schema_errors
            except ImportError:
                instrumentation.record_warning("schema", "jsonschema no disponible")

        duration = time.perf_counter() - start
        instrumentation.increment(latency=duration)

        config = {
            "catalog": self.catalog,
            "monolith": monolith,
            "monolith_sha256": monolith_hash,
            "micro_questions": micro_questions,
            "meso_questions": meso_questions,
            "macro_question": macro_question,
            "structure_report": structure_report,
            "method_summary": method_summary,
            "schema_report": schema_report,
        }

        return config

    def _validate_contract_structure(self, monolith: Dict[str, Any], instrumentation: PhaseInstrumentation) -> Dict[
        str, Any]:
        micro_questions = monolith["blocks"].get("micro_questions", [])
        base_slots = {question.get("base_slot") for question in micro_questions}
        modalities = {question.get("scoring_modality") for question in micro_questions}
        expected_modalities = {"TYPE_A", "TYPE_B", "TYPE_C", "TYPE_D", "TYPE_E", "TYPE_F"}

        if len(base_slots) != 30:
            instrumentation.record_error(
                "structure",
                "Cantidad de slots base inválida",
                expected=30,
                found=len(base_slots),
            )

        missing_modalities = expected_modalities - modalities
        if missing_modalities:
            instrumentation.record_error(
                "structure",
                "Modalidades faltantes",
                missing=sorted(missing_modalities),
            )

        slot_area_map: Dict[str, str] = {}
        area_cluster_map: Dict[str, str] = {}
        for question in micro_questions:
            slot = question.get("base_slot")
            area = question.get("policy_area_id")
            cluster = question.get("cluster_id")
            if slot and area:
                previous = slot_area_map.setdefault(slot, area)
                if previous != area:
                    instrumentation.record_error(
                        "structure",
                        "Asignación de área inconsistente",
                        base_slot=slot,
                        previous=previous,
                        current=area,
                    )
            if area and cluster:
                previous_cluster = area_cluster_map.setdefault(area, cluster)
                if previous_cluster != cluster:
                    instrumentation.record_error(
                        "structure",
                        "Área asignada a múltiples clústeres",
                        area=area,
                        previous=previous_cluster,
                        current=cluster,
                    )

        return {
            "base_slots": sorted(base_slots),
            "modalities": sorted(modalities),
            "slot_area_map": slot_area_map,
            "area_cluster_map": area_cluster_map,
        }

    def _ingest_document(self, pdf_path: str, config: Dict[str, Any]) -> PreprocessedDocument:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[1]
        start = time.perf_counter()

        document_id = os.path.splitext(os.path.basename(pdf_path))[0] or "doc_1"
        override_payload = self._context.get("preprocessed_override")
        if override_payload is not None:
            try:
                preprocessed = PreprocessedDocument.ensure(
                    override_payload, document_id=document_id
                )
            except TypeError as exc:
                instrumentation.record_error(
                    "ingestion", "Documento preprocesado incompatible", reason=str(exc)
                )
                raise
        else:
            preprocessed = PreprocessedDocument(
                document_id=document_id,
                raw_text="",
                sentences=[],
                tables=[],
                metadata={
                    "source_path": pdf_path,
                    "ingested_at": datetime.utcnow().isoformat(),
                },
            )

        duration = time.perf_counter() - start
        instrumentation.increment(latency=duration)
        return preprocessed

    async def _execute_micro_questions_async(
            self,
            document: PreprocessedDocument,
            config: Dict[str, Any],
    ) -> List[MicroQuestionRun]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[2]
        micro_questions = config.get("micro_questions", [])
        instrumentation.items_total = len(micro_questions)
        ordered_questions: List[Dict[str, Any]] = []

        questions_by_slot: Dict[str, deque] = {}
        for question in micro_questions:
            slot = question.get("base_slot")
            questions_by_slot.setdefault(slot, deque()).append(question)

        slots = sorted(questions_by_slot.keys())
        while True:
            added = False
            for slot in slots:
                queue = questions_by_slot.get(slot)
                if queue:
                    ordered_questions.append(queue.popleft())
                    added = True
            if not added:
                break

        semaphore = asyncio.Semaphore(self.resource_limits.max_workers)
        self.resource_limits.attach_semaphore(semaphore)

        circuit_breakers: Dict[str, Dict[str, Any]] = {
            slot: {"failures": 0, "open": False}
            for slot in self.executors
        }

        results: List[MicroQuestionRun] = []

        async def process_question(question: Dict[str, Any]) -> MicroQuestionRun:
            await self.resource_limits.apply_worker_budget()
            async with semaphore:
                self._ensure_not_aborted()
                question_id = question.get("question_id", "")
                question_global = int(question.get("question_global", 0))
                base_slot = question.get("base_slot", "")
                metadata = {
                    key: question.get(key)
                    for key in (
                        "question_id",
                        "question_global",
                        "base_slot",
                        "dimension_id",
                        "policy_area_id",
                        "cluster_id",
                        "scoring_modality",
                        "expected_elements",
                    )
                }

                circuit = circuit_breakers.setdefault(base_slot, {"failures": 0, "open": False})
                if circuit.get("open"):
                    instrumentation.record_warning(
                        "circuit_breaker",
                        "Circuit breaker abierto, pregunta omitida",
                        base_slot=base_slot,
                        question_id=question_id,
                    )
                    instrumentation.increment()
                    return MicroQuestionRun(
                        question_id=question_id,
                        question_global=question_global,
                        base_slot=base_slot,
                        metadata=metadata,
                        evidence=None,
                        error="circuit_breaker_open",
                        aborted=False,
                    )

                usage = self.resource_limits.get_resource_usage()
                mem_exceeded, usage = self.resource_limits.check_memory_exceeded(usage)
                cpu_exceeded, usage = self.resource_limits.check_cpu_exceeded(usage)
                if mem_exceeded:
                    instrumentation.record_warning("resource", "Límite de memoria excedido", usage=usage)
                if cpu_exceeded:
                    instrumentation.record_warning("resource", "Límite de CPU excedido", usage=usage)

                executor_class = self.executors.get(base_slot)
                start_time = time.perf_counter()
                evidence: Optional[Evidence] = None
                error_message: Optional[str] = None

                if not executor_class:
                    error_message = f"Ejecutor no definido para {base_slot}"
                    instrumentation.record_error("executor", error_message, base_slot=base_slot)
                else:
                    try:
                        executor_instance = executor_class(self.executor)
                        evidence = await asyncio.to_thread(executor_instance.execute, document, self.executor)
                        circuit["failures"] = 0
                    except Exception as exc:  # pragma: no cover - dependencias externas
                        circuit["failures"] += 1
                        error_message = str(exc)
                        instrumentation.record_error(
                            "micro_question",
                            error_message,
                            base_slot=base_slot,
                            question_id=question_id,
                        )
                        if circuit["failures"] >= 3:
                            circuit["open"] = True
                            instrumentation.record_warning(
                                "circuit_breaker",
                                "Circuit breaker activado",
                                base_slot=base_slot,
                            )

                duration = time.perf_counter() - start_time
                instrumentation.increment(latency=duration)
                if instrumentation.items_processed % 10 == 0:
                    instrumentation.record_warning(
                        "progress",
                        "Progreso de micro preguntas",
                        processed=instrumentation.items_processed,
                        total=instrumentation.items_total,
                    )

                return MicroQuestionRun(
                    question_id=question_id,
                    question_global=question_global,
                    base_slot=base_slot,
                    metadata=metadata,
                    evidence=evidence,
                    error=error_message,
                    duration_ms=duration * 1000.0,
                    aborted=self.abort_signal.is_aborted(),
                )

        tasks = [asyncio.create_task(process_question(question)) for question in ordered_questions]

        try:
            for task in asyncio.as_completed(tasks):
                result = await task
                results.append(result)
                if self.abort_signal.is_aborted():
                    raise AbortRequested(self.abort_signal.get_reason() or "Abort requested")
        except AbortRequested:
            for task in tasks:
                task.cancel()
            raise

        return results

    async def _score_micro_results_async(
            self,
            micro_results: List[MicroQuestionRun],
            config: Dict[str, Any],
    ) -> List[ScoredMicroQuestion]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[3]
        instrumentation.items_total = len(micro_results)

        from scoring import MicroQuestionScorer, ScoringModality, Evidence as ScoringEvidence

        scorer = MicroQuestionScorer()
        results: List[ScoredMicroQuestion] = []
        semaphore = asyncio.Semaphore(self.resource_limits.max_workers)
        self.resource_limits.attach_semaphore(semaphore)

        async def score_item(item: MicroQuestionRun) -> ScoredMicroQuestion:
            async with semaphore:
                await self.resource_limits.apply_worker_budget()
                self._ensure_not_aborted()
                start = time.perf_counter()

                modality_value = item.metadata.get("scoring_modality", "TYPE_A")
                try:
                    modality = ScoringModality(modality_value)
                except Exception:
                    modality = ScoringModality.TYPE_A

                if item.error or not item.evidence:
                    instrumentation.record_warning(
                        "scoring",
                        "Evidencia ausente para scoring",
                        question_id=item.question_id,
                        error=item.error,
                    )
                    instrumentation.increment(latency=time.perf_counter() - start)
                    return ScoredMicroQuestion(
                        question_id=item.question_id,
                        question_global=item.question_global,
                        base_slot=item.base_slot,
                        score=None,
                        normalized_score=None,
                        quality_level=None,
                        evidence=item.evidence,
                        metadata=item.metadata,
                        error=item.error or "missing_evidence",
                    )

                scoring_evidence = ScoringEvidence(
                    elements_found=item.evidence.elements,
                    confidence_scores=item.evidence.raw_results.get("confidence_scores", []),
                    semantic_similarity=item.evidence.raw_results.get("semantic_similarity"),
                    pattern_matches=item.evidence.raw_results.get("pattern_matches", {}),
                    metadata=item.evidence.raw_results,
                )

                try:
                    scored = await asyncio.to_thread(
                        scorer.apply_scoring_modality,
                        item.question_id,
                        item.question_global,
                        modality,
                        scoring_evidence,
                    )
                    duration = time.perf_counter() - start
                    instrumentation.increment(latency=duration)
                    return ScoredMicroQuestion(
                        question_id=scored.question_id,
                        question_global=scored.question_global,
                        base_slot=item.base_slot,
                        score=scored.raw_score,
                        normalized_score=scored.normalized_score,
                        quality_level=scored.quality_level.value,
                        evidence=item.evidence,
                        scoring_details=scored.scoring_details,
                        metadata=item.metadata,
                    )
                except Exception as exc:  # pragma: no cover - dependencia externa
                    instrumentation.record_error(
                        "scoring",
                        str(exc),
                        question_id=item.question_id,
                    )
                    duration = time.perf_counter() - start
                    instrumentation.increment(latency=duration)
                    return ScoredMicroQuestion(
                        question_id=item.question_id,
                        question_global=item.question_global,
                        base_slot=item.base_slot,
                        score=None,
                        normalized_score=None,
                        quality_level=None,
                        evidence=item.evidence,
                        metadata=item.metadata,
                        error=str(exc),
                    )

        tasks = [asyncio.create_task(score_item(item)) for item in micro_results]
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            if self.abort_signal.is_aborted():
                raise AbortRequested(self.abort_signal.get_reason() or "Abort requested")

        return results

    async def _aggregate_dimensions_async(
            self,
            scored_results: List[ScoredMicroQuestion],
            config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[4]

        dimension_map: Dict[str, List[ScoredMicroQuestion]] = {}
        for result in scored_results:
            dimension_id = result.metadata.get("dimension_id") if result.metadata else None
            if dimension_id:
                dimension_map.setdefault(dimension_id, []).append(result)

        instrumentation.items_total = len(dimension_map)
        dimension_scores: List[Dict[str, Any]] = []

        for dimension_id, items in dimension_map.items():
            self._ensure_not_aborted()
            await asyncio.sleep(0)
            start = time.perf_counter()
            valid_scores = [item.normalized_score for item in items if item.normalized_score is not None]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            dimension_scores.append(
                {
                    "dimension_id": dimension_id,
                    "policy_area_id": items[0].metadata.get("policy_area_id") if items else None,
                    "score": average_score,
                    "questions": [item.question_id for item in items],
                }
            )
            instrumentation.increment(latency=time.perf_counter() - start)

        return dimension_scores

    async def _aggregate_policy_areas_async(
            self,
            dimension_scores: List[Dict[str, Any]],
            config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[5]

        areas: Dict[str, List[Dict[str, Any]]] = {}
        for score in dimension_scores:
            area_id = score.get("policy_area_id")
            if area_id:
                areas.setdefault(area_id, []).append(score)

        instrumentation.items_total = len(areas)
        area_scores: List[Dict[str, Any]] = []

        for area_id, scores in areas.items():
            self._ensure_not_aborted()
            await asyncio.sleep(0)
            start = time.perf_counter()
            valid_scores = [entry.get("score") for entry in scores if entry.get("score") is not None]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            area_scores.append(
                {
                    "area_id": area_id,
                    "score": average_score,
                    "dimensions": scores,
                }
            )
            instrumentation.increment(latency=time.perf_counter() - start)

        return area_scores

    def _aggregate_clusters(
            self,
            policy_area_scores: List[Dict[str, Any]],
            config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[6]

        cluster_map: Dict[str, List[Dict[str, Any]]] = {}
        area_cluster_map = config.get("structure_report", {}).get("area_cluster_map", {})

        for area_score in policy_area_scores:
            area_id = area_score.get("area_id")
            cluster_id = area_cluster_map.get(area_id, "cluster_unknown")
            cluster_map.setdefault(cluster_id, []).append(area_score)

        instrumentation.items_total = len(cluster_map)
        cluster_scores: List[Dict[str, Any]] = []

        for cluster_id, scores in cluster_map.items():
            self._ensure_not_aborted()
            start = time.perf_counter()
            valid_scores = [entry.get("score") for entry in scores if entry.get("score") is not None]
            average_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            cluster_scores.append(
                {
                    "cluster_id": cluster_id,
                    "score": average_score,
                    "areas": scores,
                }
            )
            instrumentation.increment(latency=time.perf_counter() - start)

        return cluster_scores

    def _evaluate_macro(self, cluster_scores: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[7]
        start = time.perf_counter()

        valid_scores = [entry.get("score") for entry in cluster_scores if entry.get("score") is not None]
        macro_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

        instrumentation.increment(latency=time.perf_counter() - start)
        return {
            "macro_score": macro_score,
            "cluster_scores": cluster_scores,
        }

    async def _generate_recommendations(
            self,
            macro_result: Dict[str, Any],
            config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate recommendations at MICRO, MESO, and MACRO levels using RecommendationEngine.
        
        This phase connects to the orchestrator's 3-level flux:
        - MICRO: Uses scored question results from phase 3
        - MESO: Uses cluster aggregations from phase 6
        - MACRO: Uses macro evaluation from phase 7
        
        Args:
            macro_result: Macro evaluation results from phase 7
            config: Configuration dictionary
            
        Returns:
            Dictionary with MICRO, MESO, and MACRO recommendations
        """
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[8]
        start = time.perf_counter()

        await asyncio.sleep(0)

        # If RecommendationEngine is not available, return empty recommendations
        if self.recommendation_engine is None:
            logger.warning("RecommendationEngine not available, returning empty recommendations")
            recommendations = {
                "MICRO": {"level": "MICRO", "recommendations": [], "generated_at": datetime.utcnow().isoformat()},
                "MESO": {"level": "MESO", "recommendations": [], "generated_at": datetime.utcnow().isoformat()},
                "MACRO": {"level": "MACRO", "recommendations": [], "generated_at": datetime.utcnow().isoformat()},
                "macro_score": macro_result.get("macro_score"),
            }
            instrumentation.increment(latency=time.perf_counter() - start)
            return recommendations

        try:
            # ========================================================================
            # MICRO LEVEL: Transform scored results to PA-DIM scores
            # ========================================================================
            micro_scores: Dict[str, float] = {}
            scored_results = self._context.get('scored_results', [])

            # Group by policy area and dimension to calculate average scores
            pa_dim_groups: Dict[str, List[float]] = {}
            for result in scored_results:
                if hasattr(result, 'metadata') and result.metadata:
                    pa_id = result.metadata.get('policy_area_id')
                    dim_id = result.metadata.get('dimension_id')
                    score = result.normalized_score

                    if pa_id and dim_id and score is not None:
                        key = f"{pa_id}-{dim_id}"
                        if key not in pa_dim_groups:
                            pa_dim_groups[key] = []
                        pa_dim_groups[key].append(score)

            # Calculate average for each PA-DIM combination
            for key, scores in pa_dim_groups.items():
                if scores:
                    micro_scores[key] = sum(scores) / len(scores)

            logger.info(f"Extracted {len(micro_scores)} MICRO PA-DIM scores for recommendations")

            # ========================================================================
            # MESO LEVEL: Transform cluster scores
            # ========================================================================
            cluster_data: Dict[str, Any] = {}
            cluster_scores = self._context.get('cluster_scores', [])

            for cluster in cluster_scores:
                cluster_id = cluster.get('cluster_id')
                cluster_score = cluster.get('score')
                areas = cluster.get('areas', [])

                if cluster_id and cluster_score is not None:
                    # Calculate variance across areas in this cluster
                    area_scores = [area.get('score', 0) for area in areas if area.get('score') is not None]
                    variance = statistics.variance(area_scores) if len(area_scores) > 1 else 0.0

                    # Find weakest policy area in cluster
                    weak_pa = None
                    if area_scores:
                        min_score = min(area_scores)
                        for area in areas:
                            if area.get('score') == min_score:
                                weak_pa = area.get('area_id')
                                break

                    cluster_data[cluster_id] = {
                        'score': cluster_score * 100,  # Convert to 0-100 scale
                        'variance': variance,
                        'weak_pa': weak_pa
                    }

            logger.info(f"Extracted {len(cluster_data)} MESO cluster metrics for recommendations")

            # ========================================================================
            # MACRO LEVEL: Transform macro evaluation
            # ========================================================================
            macro_score = macro_result.get('macro_score')

            # Determine macro band based on score
            macro_band = 'INSUFICIENTE'
            if macro_score is not None:
                scaled_score = macro_score * 100
                if scaled_score >= 75:
                    macro_band = 'SATISFACTORIO'
                elif scaled_score >= 55:
                    macro_band = 'ACEPTABLE'
                elif scaled_score >= 35:
                    macro_band = 'DEFICIENTE'

            # Find clusters below target (< 55%)
            clusters_below_target = []
            for cluster in cluster_scores:
                cluster_id = cluster.get('cluster_id')
                cluster_score = cluster.get('score', 0)
                if cluster_score * 100 < 55:
                    clusters_below_target.append(cluster_id)

            # Calculate overall variance
            all_cluster_scores = [c.get('score', 0) for c in cluster_scores if c.get('score') is not None]
            overall_variance = statistics.variance(all_cluster_scores) if len(all_cluster_scores) > 1 else 0.0

            variance_alert = 'BAJA'
            if overall_variance >= 0.18:
                variance_alert = 'ALTA'
            elif overall_variance >= 0.08:
                variance_alert = 'MODERADA'

            # Find priority micro gaps (lowest scoring PA-DIM combinations)
            sorted_micro = sorted(micro_scores.items(), key=lambda x: x[1])
            priority_micro_gaps = [k for k, v in sorted_micro[:5] if v < 1.65]

            macro_data = {
                'macro_band': macro_band,
                'clusters_below_target': clusters_below_target,
                'variance_alert': variance_alert,
                'priority_micro_gaps': priority_micro_gaps
            }

            logger.info(f"Macro band: {macro_band}, Clusters below target: {len(clusters_below_target)}")

            # ========================================================================
            # GENERATE RECOMMENDATIONS AT ALL 3 LEVELS
            # ========================================================================
            context = {
                'generated_at': datetime.utcnow().isoformat(),
                'macro_score': macro_score
            }

            recommendation_sets = self.recommendation_engine.generate_all_recommendations(
                micro_scores=micro_scores,
                cluster_data=cluster_data,
                macro_data=macro_data,
                context=context
            )

            # Convert RecommendationSet objects to dictionaries
            recommendations = {
                level: rec_set.to_dict() for level, rec_set in recommendation_sets.items()
            }
            recommendations['macro_score'] = macro_score

            logger.info(
                f"Generated recommendations: "
                f"MICRO={len(recommendation_sets['MICRO'].recommendations)}, "
                f"MESO={len(recommendation_sets['MESO'].recommendations)}, "
                f"MACRO={len(recommendation_sets['MACRO'].recommendations)}"
            )

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}", exc_info=True)
            recommendations = {
                "MICRO": {"level": "MICRO", "recommendations": [], "generated_at": datetime.utcnow().isoformat()},
                "MESO": {"level": "MESO", "recommendations": [], "generated_at": datetime.utcnow().isoformat()},
                "MACRO": {"level": "MACRO", "recommendations": [], "generated_at": datetime.utcnow().isoformat()},
                "macro_score": macro_result.get("macro_score"),
                "error": str(e)
            }

        instrumentation.increment(latency=time.perf_counter() - start)
        return recommendations

    def _assemble_report(self, recommendations: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[9]
        start = time.perf_counter()

        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "recommendations": recommendations,
            "metadata": {
                "monolith_sha256": config.get("monolith_sha256"),
                "method_summary": config.get("method_summary"),
            },
        }

        instrumentation.increment(latency=time.perf_counter() - start)
        return report

    async def _format_and_export(self, report: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[10]
        start = time.perf_counter()

        await asyncio.sleep(0)
        export_payload = {
            "report": report,
            "phase_metrics": self.get_phase_metrics(),
            "completed_at": datetime.utcnow().isoformat(),
        }

        instrumentation.increment(latency=time.perf_counter() - start)
        return export_payload


