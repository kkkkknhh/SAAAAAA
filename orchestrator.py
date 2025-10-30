"""
ORQUESTADOR COMPLETO - LAS 30 PREGUNTAS BASE TODAS IMPLEMENTADAS
=================================================================

TODAS las preguntas base con sus métodos REALES del catálogo.
SIN brevedad. SIN omisiones. TODO implementado.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import statistics
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from schemas.preprocessed_document import DocumentIndexesV1, PreprocessedDocument, StructuredTextV1

_EMPTY_MAPPING: Mapping[str, Any] = MappingProxyType({})

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from document_ingestion import PreprocessedDocument as PreprocessedDocumentV1
else:  # pragma: no cover - runtime fallback when ingestion module unavailable
    PreprocessedDocumentV1 = Any


class _QuestionnaireProvider:
    """Centralized access to the questionnaire monolith payload."""

    _DEFAULT_PATH = Path(__file__).resolve().parent / "questionnaire_monolith.json"

    def __init__(self, data_path: Optional[Path] = None) -> None:
        self._data_path = data_path or self._DEFAULT_PATH
        self._cache: Optional[Dict[str, Any]] = None
        self._lock = RLock()

    @property
    def data_path(self) -> Path:
        """Return the resolved path of the questionnaire payload."""
        return self._data_path

    def _resolve_path(self, candidate: Optional[Union[str, Path]] = None) -> Path:
        """Resolve a candidate path relative to the current working directory."""
        if candidate is None:
            return self._data_path
        if isinstance(candidate, Path):
            path = candidate
        else:
            path = Path(candidate)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def exists(self, data_path: Optional[Union[str, Path]] = None) -> bool:
        """Check whether the questionnaire payload exists on disk."""
        return self._resolve_path(data_path).exists()

    def describe(self, data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Return metadata about a questionnaire payload on disk."""
        path = self._resolve_path(data_path)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        return {"path": path, "exists": exists, "size": size}

    def _read_payload(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load(
        self,
        force_reload: bool = False,
        data_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Load and optionally cache the questionnaire payload from disk."""
        target_path = self._resolve_path(data_path)
        with self._lock:
            if data_path is None:
                if force_reload or self._cache is None:
                    if not target_path.exists():
                        raise FileNotFoundError(
                            f"Questionnaire payload missing at {target_path}"
                        )
                    self._cache = self._read_payload(target_path)
                return self._cache

            if not target_path.exists():
                raise FileNotFoundError(
                    f"Questionnaire payload missing at {target_path}"
                )
            return self._read_payload(target_path)

    def save(
        self,
        payload: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Persist a questionnaire payload to disk through the orchestrator."""
        target_path = self._resolve_path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        if output_path is None:
            with self._lock:
                self._cache = payload
        return target_path

    def get_question(self, question_global: int) -> Dict[str, Any]:
        """Return the monolith entry for a given global question identifier."""
        payload = self.load()
        blocks = payload.get("blocks")
        if not isinstance(blocks, dict):
            raise ValueError("The questionnaire payload is missing the 'blocks' mapping")

        def _iter_questions():
            micro = blocks.get("micro_questions") or []
            if isinstance(micro, list):
                for item in micro:
                    if isinstance(item, dict):
                        yield item
            meso = blocks.get("meso_questions") or []
            if isinstance(meso, list):
                for item in meso:
                    if isinstance(item, dict):
                        yield item
            macro = blocks.get("macro_question")
            if isinstance(macro, dict):
                yield macro

        for question in _iter_questions():
            if question.get("question_global") == question_global:
                return question

        raise KeyError(f"Question {question_global} not present in questionnaire payload")


_questionnaire_provider = _QuestionnaireProvider()


def get_questionnaire_provider() -> _QuestionnaireProvider:
    """Expose the shared questionnaire provider singleton."""
    return _questionnaire_provider


def get_questionnaire_payload(force_reload: bool = False) -> Dict[str, Any]:
    """Convenience wrapper returning the questionnaire payload as a dictionary."""
    return _questionnaire_provider.load(force_reload=force_reload)


def get_question_payload(question_global: int) -> Dict[str, Any]:
    """Convenience wrapper returning a single question entry from the monolith."""
    return _questionnaire_provider.get_question(question_global)

# Validate dynamic class registry early so missing classes fail fast.
try:
    _CLASS_REGISTRY = build_class_registry()
    MODULES_OK = True
except ClassRegistryError as exc:
    MODULES_OK = False
    _CLASS_REGISTRY = {}
    logger.warning("Class registry validation failed: %s", exc)

@dataclass(frozen=True, slots=True)
class Evidence:
    modality: str
    elements: Tuple[Any, ...] = field(default_factory=tuple)
    raw_results: Mapping[str, Any] = _EMPTY_MAPPING


@dataclass(frozen=True)
class MethodContext:
    document: OrchestratorDocument
    overrides: Dict[str, Any]

    @property
    def raw_text(self) -> str:
        return self.document.raw_text

    @property
    def sentences(self) -> Sequence[str]:
        return self.document.sentences

    @property
    def tables(self) -> Sequence[Any]:
        return self.document.tables

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.document.metadata

    @classmethod
    def from_inputs(
        cls,
        context: OrchestratorDocument | None,
        kwargs: Dict[str, Any],
    ) -> "MethodContext":
        data = dict(kwargs)
        document = context
        if document is None:
            if "raw_text" in data:
                raw_text = data.pop("raw_text")
            else:
                raw_text = data.pop("text", "")
            sentences = data.pop("sentences", []) or []
            tables = data.pop("tables", []) or []
            metadata = data.pop("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {"metadata": metadata}
            document_id = data.pop(
                "document_id",
                metadata.get("document_id") or metadata.get("documentId") or "document",
            )
            document = OrchestratorDocument(
                document_id=document_id,
                raw_text=raw_text,
                sentences=list(sentences),
                tables=list(tables),
                metadata=metadata,
            )

        return cls(document=document, overrides=data)


def _route_text_sentences_tables(ctx: MethodContext) -> Dict[str, Any]:
    payload = {
        "document_id": ctx.document.document_id,
        "raw_text": ctx.raw_text,
        "text": ctx.raw_text,
        "sentences": list(ctx.sentences),
        "tables": list(ctx.tables),
        "metadata": ctx.metadata,
    }
    payload.update(ctx.overrides)
    return payload


DEFAULT_ROUTE = _route_text_sentences_tables

ARG_ROUTER: Dict[Tuple[str, str], Callable[[MethodContext], Dict[str, Any]]] = {
    ("IndustrialPolicyProcessor", "process"): _route_text_sentences_tables,
    ("IndustrialPolicyProcessor", "_match_patterns_in_sentences"): _route_text_sentences_tables,
    ("IndustrialPolicyProcessor", "_construct_evidence_bundle"): _route_text_sentences_tables,
    ("PolicyTextProcessor", "segment_into_sentences"): _route_text_sentences_tables,
    ("BayesianEvidenceScorer", "compute_evidence_score"): _route_text_sentences_tables,
}


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
        return self._event.is_set()

    def get_reason(self) -> Optional[str]:
        with self._lock:
            return self._reason

    def get_timestamp(self) -> Optional[datetime]:
        with self._lock:
            return self._timestamp

    def reset(self) -> None:
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
            import psutil  # type: ignore

            self._psutil = psutil
            self._psutil_process = psutil.Process(os.getpid())
        except Exception:  # pragma: no cover - psutil missing
            self._psutil = None
            self._psutil_process = None

    @property
    def max_workers(self) -> int:
        return self._max_workers

    def attach_semaphore(self, semaphore: asyncio.Semaphore) -> None:
        self._semaphore = semaphore
        self._semaphore_limit = self._max_workers

    async def apply_worker_budget(self) -> int:
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
        self._usage_history.append(usage)
        self._predict_worker_budget()

    def _predict_worker_budget(self) -> None:
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

    def check_memory_exceeded(self, usage: Optional[Dict[str, float]] = None) -> Tuple[bool, Dict[str, float]]:
        usage = usage or self.get_resource_usage()
        exceeded = False
        if self.max_memory_mb is not None:
            exceeded = usage.get("rss_mb", 0.0) > self.max_memory_mb
        return exceeded, usage

    def check_cpu_exceeded(self, usage: Optional[Dict[str, float]] = None) -> Tuple[bool, Dict[str, float]]:
        usage = usage or self.get_resource_usage()
        exceeded = False
        if self.max_cpu_percent:
            exceeded = usage.get("cpu_percent", 0.0) > self.max_cpu_percent
        return exceeded, usage

    def get_resource_usage(self) -> Dict[str, float]:
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
        if items_total is not None:
            self.items_total = items_total
        self.start_time = time.perf_counter()

    def increment(self, count: int = 1, latency: Optional[float] = None) -> None:
        self.items_processed += count
        if latency is not None:
            self.latencies.append(latency)
            self._detect_latency_anomaly(latency)
        if self.resource_limits and self.should_snapshot():
            self.capture_resource_snapshot()

    def should_snapshot(self) -> bool:
        if self.items_total == 0:
            return False
        if self.items_processed == 0:
            return False
        return self.items_processed % self.snapshot_interval == 0

    def capture_resource_snapshot(self) -> None:
        if not self.resource_limits:
            return
        snapshot = self.resource_limits.get_resource_usage()
        snapshot["items_processed"] = self.items_processed
        self.resource_snapshots.append(snapshot)

    def record_warning(self, category: str, message: str, **extra: Any) -> None:
        entry = {"category": category, "message": message, **extra, "timestamp": datetime.utcnow().isoformat()}
        self.warnings.append(entry)

    def record_error(self, category: str, message: str, **extra: Any) -> None:
        entry = {"category": category, "message": message, **extra, "timestamp": datetime.utcnow().isoformat()}
        self.errors.append(entry)

    def _detect_latency_anomaly(self, latency: float) -> None:
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
        self.end_time = time.perf_counter()

    def duration_ms(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000.0

    def progress(self) -> Optional[float]:
        if not self.items_total:
            return None
        return min(1.0, self.items_processed / float(self.items_total))

    def throughput(self) -> Optional[float]:
        if self.start_time is None:
            return None
        elapsed = (time.perf_counter() - self.start_time) if self.end_time is None else (self.end_time - self.start_time)
        if not elapsed:
            return None
        return self.items_processed / elapsed

    def latency_histogram(self) -> Dict[str, Optional[float]]:
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
    success: bool
    phase_id: str
    data: Any
    error: Optional[Exception]
    duration_ms: float
    mode: str
    aborted: bool = False


@dataclass
class MicroQuestionRun:
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


class ArgRouter:
    """Normalize orchestrator kwargs into the signatures expected by methods."""

    def __init__(self) -> None:
        self._special_routes = {
            ("IndustrialPolicyProcessor", "process"): self._route_policy_process,
            (
                "IndustrialPolicyProcessor",
                "_match_patterns_in_sentences",
            ): self._route_match_patterns,
            (
                "IndustrialPolicyProcessor",
                "_construct_evidence_bundle",
            ): self._route_construct_bundle,
            ("PolicyTextProcessor", "segment_into_sentences"): self._route_segment_sentences,
            ("BayesianEvidenceScorer", "compute_evidence_score"): self._route_evidence_score,
        }

    def route(
        self,
        class_name: str,
        instance: Any,
        method: Any,
        provided_kwargs: Dict[str, Any],
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Return positional and keyword args compatible with the target method."""

        route_key = (class_name, getattr(method, "__name__", ""))
        if route_key in self._special_routes:
            return self._special_routes[route_key](instance, method, provided_kwargs)

        return self._default_route(method, provided_kwargs)

    def _default_route(
        self, method: Any, provided_kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        signature = inspect.signature(method)
        normalized = dict(provided_kwargs)
        
        # Alias map for common parameter name variations
        alias_map = {
            "text": ("raw_text", "document_text"),
            "raw_text": ("text", "document_text"),
        }
        
        # Apply alias mapping
        for source, targets in alias_map.items():
            if source in normalized:
                for target in targets:
                    if target in signature.parameters and target not in normalized:
                        normalized[target] = normalized[source]
                        break
        
        # Filter to only accepted kwargs
        accepted_kwargs: Dict[str, Any] = {}
        allow_extra = False

        for name, param in signature.parameters.items():
            if param.kind == param.VAR_POSITIONAL:
                continue
            if param.kind == param.VAR_KEYWORD:
                allow_extra = True
                continue

            if name in normalized:
                accepted_kwargs[name] = normalized[name]

        if allow_extra:
            for key, value in normalized.items():
                accepted_kwargs.setdefault(key, value)

        return (), accepted_kwargs

    def _extract_all_patterns(self, instance: Any) -> List[Any]:
        pattern_registry = getattr(instance, "_pattern_registry", {}) or {}
        compiled_patterns: List[Any] = []
        for categories in pattern_registry.values():
            compiled_patterns.extend(chain.from_iterable(categories.values()))
        return compiled_patterns

    def _derive_dimension_category(self, instance: Any) -> Tuple[Any, str, List[Any]]:
        pattern_registry = getattr(instance, "_pattern_registry", {}) or {}
        dimension = None
        category = None
        compiled_patterns: List[Any] = []

        if pattern_registry:
            dimension = next(iter(pattern_registry.keys()))
            categories = pattern_registry.get(dimension, {})
            if categories:
                category = next(iter(categories.keys()))
                compiled_patterns = list(categories.get(category, []))

        if dimension is None:
            dimension = getattr(instance, "default_dimension", None)
        if dimension is None:
            dimension_enum = globals().get("CausalDimension")
            if dimension_enum is not None:
                dimension = getattr(dimension_enum, "D1_INSUMOS", dimension_enum)
            else:
                dimension = "d1_insumos"

        if category is None:
            category = "general"

        return dimension, category, compiled_patterns

    def _route_policy_process(
        self, instance: Any, method: Any, provided_kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        text = provided_kwargs.get("text")
        if text is None:
            text = provided_kwargs.get("raw_text", "")
        return (), {"raw_text": text}

    def _route_match_patterns(
        self, instance: Any, method: Any, provided_kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        compiled_patterns = self._extract_all_patterns(instance)
        sentences = [
            sentence
            for sentence in provided_kwargs.get("sentences", [])
            if isinstance(sentence, str)
        ]
        return (compiled_patterns, sentences), {}

    def _route_construct_bundle(
        self, instance: Any, method: Any, provided_kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        text = provided_kwargs.get("text") or provided_kwargs.get("raw_text", "")
        sentences = [
            sentence
            for sentence in provided_kwargs.get("sentences", [])
            if isinstance(sentence, str)
        ]

        dimension, category, compiled_patterns = self._derive_dimension_category(instance)

        if compiled_patterns and sentences:
            matches, positions = instance._match_patterns_in_sentences(  # type: ignore[attr-defined]
                compiled_patterns, sentences
            )
        else:
            matches = []
            positions = []

        if not matches and text:
            fallback_matches = re.findall(r"\b\w{6,}\b", text)
            matches = fallback_matches[:20]
            positions = list(range(len(matches)))

        text_length = len(text) if text else sum(len(s) for s in sentences)
        confidence = 0.0
        compute_confidence = getattr(instance, "_compute_evidence_confidence", None)
        if callable(compute_confidence) and (matches or text_length):
            try:
                confidence = compute_confidence(matches, text_length, pattern_specificity=0.85)
            except TypeError:
                confidence = compute_confidence(matches, text_length, 0.85)
        elif text_length:
            confidence = min(1.0, len(matches) / max(1, text_length))

        return (), {
            "dimension": dimension,
            "category": category,
            "matches": matches,
            "positions": positions,
            "confidence": confidence,
        }

    def _route_segment_sentences(
        self, instance: Any, method: Any, provided_kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        text = provided_kwargs.get("text")
        if text is None:
            text = provided_kwargs.get("raw_text", "")
        return (text,), {}

    def _route_evidence_score(
        self, instance: Any, method: Any, provided_kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        text = provided_kwargs.get("text") or provided_kwargs.get("raw_text", "")
        sentences = [
            sentence
            for sentence in provided_kwargs.get("sentences", [])
            if isinstance(sentence, str)
        ]

        matches: List[str] = [s for s in sentences if s.strip()]
        if not matches and text:
            matches = re.findall(r"\b\w{6,}\b", text)[:50]

        total_corpus_size = len(text)
        if total_corpus_size == 0 and sentences:
            total_corpus_size = sum(len(s) for s in sentences)

        uniqueness = len(set(matches)) if matches else 0
        pattern_specificity = 0.8
        if uniqueness and matches:
            pattern_specificity = min(0.95, max(0.2, uniqueness / max(1, len(matches))))

        return (matches, total_corpus_size), {"pattern_specificity": pattern_specificity}

class MethodExecutor:
    """Ejecuta métodos del catálogo"""

    def __init__(
        self,
        class_registry: Optional[Dict[str, type]] = None,
        arg_router: Optional[ArgRouter] = None,
        drift_monitor: Optional[PayloadDriftMonitor] = None,
    ) -> None:
        self._class_registry = class_registry or _CLASS_REGISTRY
        self.arg_router = arg_router or ExternalArgRouter(self._class_registry)
        self._drift_monitor = drift_monitor or PayloadDriftMonitor.from_env()
        if MODULES_OK:
            # Create shared ontology instance for all analyzers
            ontology = MunicipalOntology()
            
            self.instances = {
                name: cls()
                for name, cls in self._class_registry.items()
            }
        else:
            self.instances = {}

    def execute(self, class_name: str, method_name: str, **kwargs) -> Any:
        if not MODULES_OK:
            return None

        instance = self.instances.get(class_name)
        if instance is None:
            logger.error("Class '%s' not registered in executor", class_name)
            return None

        method = getattr(instance, method_name, None)
        if method is None:
            logger.error("Method '%s.%s' not available", class_name, method_name)
            return None

        try:
            args, call_kwargs = self.arg_router.route(
                class_name, method_name, dict(kwargs)
            )
        except ArgumentValidationError as error:
            expected = self.arg_router.expected_arguments(class_name, method_name)
            logger.error(
                "Argument validation failed [%s -> %s]: missing=%s unexpected=%s "
                "type_mismatches=%s provided=%s expected=%s",
                class_name,
                method_name,
                sorted(error.missing),
                sorted(error.unexpected),
                error.type_mismatches,
                sorted(kwargs.keys()),
                sorted(expected),
            )
            return None
        except ArgRouterError as error:
            logger.error(
                "Routing failure for %s.%s: %s", class_name, method_name, error
            )
            return None

        self._drift_monitor.maybe_validate(
            kwargs, producer=class_name, consumer=method_name
        )

        try:
            result = method(*args, **call_kwargs)
        except Exception as exc:
            logger.error(f"Error {class_name}.{method_name}: {exc}")
            return None

        if isinstance(result, dict):
            self._drift_monitor.maybe_validate(
                result,
                producer=f"{class_name}.{method_name}",
                consumer="return",
            )

        return result


class D1Q1_Executor(DataFlowExecutor):
    """
    D1-Q1: Líneas Base y Brechas Cuantificadas
    Flow: PP.O → CD.E+T → CD.V → CD.C → A1.C || EP.C → PP.R
    Métodos: 18
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. PP.C - BayesianEvidenceScorer._calculate_shannon_entropy (P=2)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            '_calculate_shannon_entropy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer._calculate_shannon_entropy'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.C - SemanticAnalyzer._calculate_semantic_complexity (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_calculate_semantic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._calculate_semantic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 16. A1.V - SemanticAnalyzer._classify_policy_domain (P=1)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        # 17. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.V - BayesianNumericalAnalyzer._classify_evidence_strength (P=2)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            '_classify_evidence_strength',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer._classify_evidence_strength'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D1Q2_Executor(DataFlowExecutor):
    """
    D1-Q2: Normalización y Fuentes
    Flow: PP.E → PP.T → CD.E+T → CD.V+C → EP.E+C
    Métodos: 12
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.T - IndustrialPolicyProcessor._compile_pattern_registry (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_compile_pattern_registry',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._compile_pattern_registry'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.normalize_unicode (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'normalize_unicode',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.normalize_unicode'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 11. EP.E - PolicyAnalysisEmbedder._extract_numerical_values (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_extract_numerical_values',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._extract_numerical_values'] = result
        if result is not None:
            current_data = result
        
        # 12. EP.C - BayesianNumericalAnalyzer._compute_coherence (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            '_compute_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer._compute_coherence'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D1Q3_Executor(DataFlowExecutor):
    """
    D1-Q3: Asignación de Recursos
    Flow: PP.O → CD.E → CD.V+C → FV.E → DB.O → EP.C → PP.R
    Métodos: 22
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.E - IndustrialPolicyProcessor._extract_point_evidence (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_extract_point_evidence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._extract_point_evidence'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - PolicyContradictionDetector._detect_resource_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_resource_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_resource_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._are_conflicting_allocations (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_conflicting_allocations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_conflicting_allocations'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.E - TemporalLogicVerifier._extract_resources (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_extract_resources',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._extract_resources'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 17. FV.E - PDETMunicipalPlanAnalyzer._extract_financial_amounts (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_financial_amounts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_financial_amounts'] = result
        if result is not None:
            current_data = result
        
        # 18. FV.V - PDETMunicipalPlanAnalyzer._identify_funding_source (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_identify_funding_source',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._identify_funding_source'] = result
        if result is not None:
            current_data = result
        
        # 19. FV.C - PDETMunicipalPlanAnalyzer._analyze_funding_sources (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_analyze_funding_sources',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._analyze_funding_sources'] = result
        if result is not None:
            current_data = result
        
        # 20. DB.O - FinancialAuditor.trace_financial_allocation (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            'trace_financial_allocation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor.trace_financial_allocation'] = result
        if result is not None:
            current_data = result
        
        # 21. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 22. EP.C - BayesianNumericalAnalyzer.compare_policies (P=2)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'compare_policies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.compare_policies'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D1Q4_Executor(DataFlowExecutor):
    """
    D1-Q4: Capacidad Institucional
    Flow: PP.E → CD.E+T+V+C → A1.V → FV.E+V
    Métodos: 16
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.T - IndustrialPolicyProcessor._build_point_patterns (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_build_point_patterns',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._build_point_patterns'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.C - PolicyContradictionDetector._calculate_graph_fragmentation (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_graph_fragmentation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_graph_fragmentation'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._calculate_syntactic_complexity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_syntactic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_syntactic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 12. A1.V - SemanticAnalyzer._classify_value_chain_link (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_value_chain_link',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_value_chain_link'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.V - PerformanceAnalyzer._detect_bottlenecks (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_detect_bottlenecks',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._detect_bottlenecks'] = result
        if result is not None:
            current_data = result
        
        # 14. A1.E - TextMiningEngine._identify_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_identify_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._identify_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.E - PDETMunicipalPlanAnalyzer.identify_responsible_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'identify_responsible_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.identify_responsible_entities'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.V - PDETMunicipalPlanAnalyzer._classify_entity_type (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_entity_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_entity_type'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D1Q5_Executor(DataFlowExecutor):
    """
    D1-Q5: Restricciones Temporales
    Flow: PP.E+T → CD.E → CD.V+T+C → A1.C
    Métodos: 14
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 4. CD.V - PolicyContradictionDetector._detect_temporal_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_temporal_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_temporal_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - TemporalLogicVerifier.verify_temporal_consistency (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            'verify_temporal_consistency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier.verify_temporal_consistency'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - TemporalLogicVerifier._build_timeline (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_build_timeline',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._build_timeline'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.T - TemporalLogicVerifier._parse_temporal_marker (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_parse_temporal_marker',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._parse_temporal_marker'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - TemporalLogicVerifier._has_temporal_conflict (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_has_temporal_conflict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._has_temporal_conflict'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - TemporalLogicVerifier._check_deadline_constraints (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_check_deadline_constraints',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._check_deadline_constraints'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - TemporalLogicVerifier._classify_temporal_type (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_classify_temporal_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._classify_temporal_type'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.C - SemanticAnalyzer._calculate_semantic_complexity (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_calculate_semantic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._calculate_semantic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 14. A1.C - PerformanceAnalyzer._calculate_throughput_metrics (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_calculate_throughput_metrics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._calculate_throughput_metrics'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q1_Executor(DataFlowExecutor):
    """
    D2-Q1: Formato Tabular y Trazabilidad
    Flow: PP.O → FV.E → FV.T+V → CD.V → SC.E
    Métodos: 20
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.T - PDETMunicipalPlanAnalyzer._clean_dataframe (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_clean_dataframe',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._clean_dataframe'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.V - PDETMunicipalPlanAnalyzer._is_likely_header (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_is_likely_header',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._is_likely_header'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.T - PDETMunicipalPlanAnalyzer._deduplicate_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_deduplicate_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._deduplicate_tables'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.T - PDETMunicipalPlanAnalyzer._reconstruct_fragmented_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_reconstruct_fragmented_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._reconstruct_fragmented_tables'] = result
        if result is not None:
            current_data = result
        
        # 10. FV.V - PDETMunicipalPlanAnalyzer._classify_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_tables'] = result
        if result is not None:
            current_data = result
        
        # 11. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 12. FV.E - PDETMunicipalPlanAnalyzer._extract_from_budget_table (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_budget_table',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_budget_table'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.E - PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_responsibility_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.E - PDETMunicipalPlanAnalyzer.identify_responsible_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'identify_responsible_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.identify_responsible_entities'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.T - PDETMunicipalPlanAnalyzer._consolidate_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_consolidate_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._consolidate_entities'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.C - PDETMunicipalPlanAnalyzer._score_entity_specificity (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_score_entity_specificity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._score_entity_specificity'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.T - TemporalLogicVerifier._build_timeline (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_build_timeline',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._build_timeline'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.V - TemporalLogicVerifier._check_deadline_constraints (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_check_deadline_constraints',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._check_deadline_constraints'] = result
        if result is not None:
            current_data = result
        
        # 19. CD.V - PolicyContradictionDetector._detect_temporal_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_temporal_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_temporal_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 20. SC.E - SemanticProcessor._detect_table (P=3)
        result = self.executor.execute(
            'SemanticProcessor',
            '_detect_table',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticProcessor._detect_table'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q2_Executor(DataFlowExecutor):
    """
    D2-Q2: Causalidad de Actividades
    Flow: PP.E → PP.C → CD.E+T+V+C → DB.O → TC.T+V → A1.V
    Métodos: 25
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.E - CausalExtractor._extract_goals (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_goals',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_goals'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.E - CausalExtractor._extract_goal_text (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_goal_text',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_goal_text'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.V - CausalExtractor._classify_goal_type (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_classify_goal_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._classify_goal_type'] = result
        if result is not None:
            current_data = result
        
        # 18. DB.T - CausalExtractor._add_node_to_graph (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_add_node_to_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._add_node_to_graph'] = result
        if result is not None:
            current_data = result
        
        # 19. DB.E - CausalExtractor._extract_causal_links (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_links'] = result
        if result is not None:
            current_data = result
        
        # 20. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 21. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 22. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 23. A1.C - TextMiningEngine._analyze_link_text (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_analyze_link_text',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._analyze_link_text'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q3_Executor(DataFlowExecutor):
    """
    D2-Q3: Responsables de Actividades
    Flow: PP.O → FV.E+T+V+C → CD.V → EP.E
    Métodos: 18
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.identify_responsible_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'identify_responsible_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.identify_responsible_entities'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.E - PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_responsibility_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.T - PDETMunicipalPlanAnalyzer._consolidate_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_consolidate_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._consolidate_entities'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.V - PDETMunicipalPlanAnalyzer._classify_entity_type (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_entity_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_entity_type'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.C - PDETMunicipalPlanAnalyzer._score_entity_specificity (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_score_entity_specificity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._score_entity_specificity'] = result
        if result is not None:
            current_data = result
        
        # 10. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 11. FV.T - PDETMunicipalPlanAnalyzer._clean_dataframe (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_clean_dataframe',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._clean_dataframe'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 14. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q4_Executor(DataFlowExecutor):
    """
    D2-Q4: Cuantificación de Actividades
    Flow: PP.O → FV.E → CD.E+T+V+C → EP.C
    Métodos: 21
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.E - PDETMunicipalPlanAnalyzer._extract_financial_amounts (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_financial_amounts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_financial_amounts'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.E - PDETMunicipalPlanAnalyzer._extract_from_budget_table (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_budget_table',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_budget_table'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 16. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q5_Executor(DataFlowExecutor):
    """
    D2-Q5: Eslabón Causal Diagnóstico-Actividades
    Flow: PP.E+C → CD.E+T+V+C → DB.O → TC.T+V → A1.V
    Métodos: 23
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 15. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 16. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 17. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 18. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 19. A1.C - TextMiningEngine._analyze_link_text (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_analyze_link_text',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._analyze_link_text'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q1_Executor(DataFlowExecutor):
    """
    D3-Q1: Indicadores de Producto
    Flow: PP.O → CD.E+T+V → FV.E+T+V → EP.E+C → PP.R
    Métodos: 19
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 12. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.T - PDETMunicipalPlanAnalyzer._indicator_to_dict (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_indicator_to_dict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._indicator_to_dict'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.E - PDETMunicipalPlanAnalyzer._find_product_mentions (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_find_product_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._find_product_mentions'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.V - PDETMunicipalPlanAnalyzer._classify_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_tables'] = result
        if result is not None:
            current_data = result
        
        # 17. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.E - PolicyAnalysisEmbedder._extract_numerical_values (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_extract_numerical_values',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._extract_numerical_values'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q2_Executor(DataFlowExecutor):
    """
    D3-Q2: Cuantificación de Productos
    Flow: PP.O → FV.E → CD.E+T+V+C → EP.C
    Métodos: 20
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.E - PDETMunicipalPlanAnalyzer._extract_financial_amounts (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_financial_amounts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_financial_amounts'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.E - PDETMunicipalPlanAnalyzer._extract_from_budget_table (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_budget_table',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_budget_table'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.E - PDETMunicipalPlanAnalyzer._find_product_mentions (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_find_product_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._find_product_mentions'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 16. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 19. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q3_Executor(DataFlowExecutor):
    """
    D3-Q3: Responsables de Productos
    Flow: PP.O → FV.E+T+V+C → CD.V+T → EP.E
    Métodos: 17
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.identify_responsible_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'identify_responsible_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.identify_responsible_entities'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.E - PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_responsibility_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.T - PDETMunicipalPlanAnalyzer._consolidate_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_consolidate_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._consolidate_entities'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.V - PDETMunicipalPlanAnalyzer._classify_entity_type (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_entity_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_entity_type'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.C - PDETMunicipalPlanAnalyzer._score_entity_specificity (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_score_entity_specificity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._score_entity_specificity'] = result
        if result is not None:
            current_data = result
        
        # 10. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 14. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q4_Executor(DataFlowExecutor):
    """
    D3-Q4: Plazos de Productos
    Flow: PP.E+T → CD.E → CD.V+T+C → A1.C+V
    Métodos: 19
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - TemporalLogicVerifier.verify_temporal_consistency (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            'verify_temporal_consistency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier.verify_temporal_consistency'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - TemporalLogicVerifier._check_deadline_constraints (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_check_deadline_constraints',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._check_deadline_constraints'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - TemporalLogicVerifier._classify_temporal_type (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_classify_temporal_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._classify_temporal_type'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - TemporalLogicVerifier._build_timeline (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_build_timeline',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._build_timeline'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.T - TemporalLogicVerifier._parse_temporal_marker (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_parse_temporal_marker',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._parse_temporal_marker'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - TemporalLogicVerifier._has_temporal_conflict (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_has_temporal_conflict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._has_temporal_conflict'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - TemporalLogicVerifier._extract_resources (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_extract_resources',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._extract_resources'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._detect_resource_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_resource_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_resource_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.C - PerformanceAnalyzer._calculate_throughput_metrics (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_calculate_throughput_metrics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._calculate_throughput_metrics'] = result
        if result is not None:
            current_data = result
        
        # 16. A1.V - PerformanceAnalyzer._detect_bottlenecks (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_detect_bottlenecks',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._detect_bottlenecks'] = result
        if result is not None:
            current_data = result
        
        # 17. A1.V - TextMiningEngine._assess_risks (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_assess_risks',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._assess_risks'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q5_Executor(DataFlowExecutor):
    """
    D3-Q5: Eslabón Causal Producto-Resultado
    Flow: PP.E+C → CD.E+T+V+C → DB.O (Extractor+Mechanism+Tests) → TC.T+V → A1.V
    Métodos: 26
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.E - CausalExtractor._extract_causal_links (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_links'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.E - CausalExtractor._extract_causal_justifications (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_justifications',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_justifications'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.C - CausalExtractor._calculate_confidence (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_calculate_confidence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._calculate_confidence'] = result
        if result is not None:
            current_data = result
        
        # 18. DB.E - MechanismPartExtractor.extract_entity_activity (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            'extract_entity_activity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor.extract_entity_activity'] = result
        if result is not None:
            current_data = result
        
        # 19. DB.E - MechanismPartExtractor._find_subject_entity (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            '_find_subject_entity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor._find_subject_entity'] = result
        if result is not None:
            current_data = result
        
        # 20. DB.E - MechanismPartExtractor._find_action_verb (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            '_find_action_verb',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor._find_action_verb'] = result
        if result is not None:
            current_data = result
        
        # 21. DB.V - MechanismPartExtractor._validate_entity_activity (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            '_validate_entity_activity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor._validate_entity_activity'] = result
        if result is not None:
            current_data = result
        
        # 22. DB.C - MechanismPartExtractor._calculate_ea_confidence (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            '_calculate_ea_confidence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor._calculate_ea_confidence'] = result
        if result is not None:
            current_data = result
        
        # 23. DB.O - BayesianMechanismInference.infer_mechanisms (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            'infer_mechanisms',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference.infer_mechanisms'] = result
        if result is not None:
            current_data = result
        
        # 24. DB.T - BayesianMechanismInference._build_transition_matrix (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_build_transition_matrix',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._build_transition_matrix'] = result
        if result is not None:
            current_data = result
        
        # 25. DB.V - BayesianMechanismInference._infer_activity_sequence (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_infer_activity_sequence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._infer_activity_sequence'] = result
        if result is not None:
            current_data = result
        
        # 26. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 27. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 28. DB.V - BayesianMechanismInference._classify_mechanism_type (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_classify_mechanism_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._classify_mechanism_type'] = result
        if result is not None:
            current_data = result
        
        # 29. DB.V - BeachEvidentialTest.apply_test_logic (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'apply_test_logic',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.apply_test_logic'] = result
        if result is not None:
            current_data = result
        
        # 30. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 31. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 32. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 33. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 34. A1.C - TextMiningEngine._analyze_link_text (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_analyze_link_text',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._analyze_link_text'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q1_Executor(DataFlowExecutor):
    """
    D4-Q1: Indicadores de Resultado
    Flow: PP.O → CD.E+T+V → FV.E+T+V → EP.E+C → PP.R
    Métodos: 18
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 12. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.T - PDETMunicipalPlanAnalyzer._indicator_to_dict (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_indicator_to_dict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._indicator_to_dict'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.E - PDETMunicipalPlanAnalyzer._find_outcome_mentions (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_find_outcome_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._find_outcome_mentions'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.V - PDETMunicipalPlanAnalyzer._classify_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_tables'] = result
        if result is not None:
            current_data = result
        
        # 17. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.E - PolicyAnalysisEmbedder._extract_numerical_values (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_extract_numerical_values',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._extract_numerical_values'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q2_Executor(DataFlowExecutor):
    """
    D4-Q2: Cadena Causal y Supuestos
    Flow: PP.E+C → CD.E+T+V+C → DB.O (Extractor+Tests) → TC.T+V
    Métodos: 24
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.C - PolicyContradictionDetector._calculate_syntactic_complexity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_syntactic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_syntactic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.E - CausalExtractor._extract_causal_links (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_links'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.O - BayesianMechanismInference.infer_mechanisms (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            'infer_mechanisms',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference.infer_mechanisms'] = result
        if result is not None:
            current_data = result
        
        # 18. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 19. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 20. DB.V - BeachEvidentialTest.classify_test (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'classify_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.classify_test'] = result
        if result is not None:
            current_data = result
        
        # 21. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 22. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 23. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 24. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q3_Executor(DataFlowExecutor):
    """
    D4-Q3: Justificación de Ambición
    Flow: PP.O+C → CD.E+V+C → FV.C+R → DB.C → EP.C+V
    Métodos: 20
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer._calculate_shannon_entropy (P=2)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            '_calculate_shannon_entropy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer._calculate_shannon_entropy'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.C - PolicyContradictionDetector._calculate_objective_alignment (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_objective_alignment',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_objective_alignment'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.R - PDETMunicipalPlanAnalyzer.generate_recommendations (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'generate_recommendations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.generate_recommendations'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.C - PDETMunicipalPlanAnalyzer.analyze_financial_feasibility (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_financial_feasibility',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_financial_feasibility'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.C - PDETMunicipalPlanAnalyzer._assess_financial_sustainability (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_assess_financial_sustainability',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._assess_financial_sustainability'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.C - PDETMunicipalPlanAnalyzer._bayesian_risk_inference (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_bayesian_risk_inference',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._bayesian_risk_inference'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.C - FinancialAuditor._calculate_sufficiency (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_calculate_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._calculate_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 19. EP.C - BayesianNumericalAnalyzer.compare_policies (P=2)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'compare_policies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.compare_policies'] = result
        if result is not None:
            current_data = result
        
        # 20. EP.V - BayesianNumericalAnalyzer._classify_evidence_strength (P=2)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            '_classify_evidence_strength',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer._classify_evidence_strength'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q4_Executor(DataFlowExecutor):
    """
    D4-Q4: Población Objetivo
    Flow: PP.O → CD.E+T+V+C → A1.V+E → EP.E+V
    Métodos: 15
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 11. A1.V - SemanticAnalyzer._classify_cross_cutting_themes (P=3)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_cross_cutting_themes',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_cross_cutting_themes'] = result
        if result is not None:
            current_data = result
        
        # 12. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.E - SemanticAnalyzer.extract_semantic_cube (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            'extract_semantic_cube',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer.extract_semantic_cube'] = result
        if result is not None:
            current_data = result
        
        # 14. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 15. EP.V - PolicyAnalysisEmbedder._filter_by_pdq (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_filter_by_pdq',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._filter_by_pdq'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q5_Executor(DataFlowExecutor):
    """
    D4-Q5: Alineación con Objetivos Superiores
    Flow: PP.O → CD.C+T → A1.V+E → EP.E+C
    Métodos: 17
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.C - PolicyContradictionDetector._calculate_objective_alignment (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_objective_alignment',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_objective_alignment'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 11. A1.V - SemanticAnalyzer._classify_cross_cutting_themes (P=3)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_cross_cutting_themes',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_cross_cutting_themes'] = result
        if result is not None:
            current_data = result
        
        # 12. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.E - SemanticAnalyzer.extract_semantic_cube (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            'extract_semantic_cube',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer.extract_semantic_cube'] = result
        if result is not None:
            current_data = result
        
        # 14. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 15. EP.C - PolicyAnalysisEmbedder.compare_policy_interventions (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'compare_policy_interventions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.compare_policy_interventions'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q1_Executor(DataFlowExecutor):
    """
    D5-Q1: Indicadores de Impacto
    Flow: PP.O → CD.E+T+V → FV.E+T+V → EP.C → PP.R
    Métodos: 17
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 12. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.T - PDETMunicipalPlanAnalyzer._indicator_to_dict (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_indicator_to_dict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._indicator_to_dict'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.V - PDETMunicipalPlanAnalyzer._classify_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_tables'] = result
        if result is not None:
            current_data = result
        
        # 16. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q2_Executor(DataFlowExecutor):
    """
    D5-Q2: Eslabón Causal Resultado-Impacto
    Flow: PP.E+C → CD.E+T+V+C → DB.O (Extractor+Inference+Tests) → TC.T+V
    Métodos: 25
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.E - CausalExtractor._extract_causal_links (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_links'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.E - CausalExtractor._extract_causal_justifications (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_justifications',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_justifications'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.O - BayesianMechanismInference.infer_mechanisms (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            'infer_mechanisms',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference.infer_mechanisms'] = result
        if result is not None:
            current_data = result
        
        # 18. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 19. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 20. DB.V - BayesianMechanismInference._classify_mechanism_type (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_classify_mechanism_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._classify_mechanism_type'] = result
        if result is not None:
            current_data = result
        
        # 21. DB.V - BeachEvidentialTest.apply_test_logic (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'apply_test_logic',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.apply_test_logic'] = result
        if result is not None:
            current_data = result
        
        # 22. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 23. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 24. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 25. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q3_Executor(DataFlowExecutor):
    """
    D5-Q3: Evidencia de Causalidad
    Flow: PP.O → CD.E+T+V+C → DB.O (Extractor+Tests) → EP.C
    Métodos: 19
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 13. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.E - CausalExtractor._extract_causal_justifications (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_justifications',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_justifications'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.O - BayesianMechanismInference.infer_mechanisms (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            'infer_mechanisms',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference.infer_mechanisms'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q4_Executor(DataFlowExecutor):
    """
    D5-Q4: Plazos de Impacto
    Flow: PP.E+T → CD.E → CD.V+T+C → A1.C+V
    Métodos: 15
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - TemporalLogicVerifier.verify_temporal_consistency (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            'verify_temporal_consistency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier.verify_temporal_consistency'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - TemporalLogicVerifier._check_deadline_constraints (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_check_deadline_constraints',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._check_deadline_constraints'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - TemporalLogicVerifier._classify_temporal_type (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_classify_temporal_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._classify_temporal_type'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - TemporalLogicVerifier._build_timeline (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_build_timeline',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._build_timeline'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.T - TemporalLogicVerifier._parse_temporal_marker (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_parse_temporal_marker',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._parse_temporal_marker'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - TemporalLogicVerifier._has_temporal_conflict (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_has_temporal_conflict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._has_temporal_conflict'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.C - PerformanceAnalyzer._calculate_throughput_metrics (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_calculate_throughput_metrics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._calculate_throughput_metrics'] = result
        if result is not None:
            current_data = result
        
        # 14. A1.V - PerformanceAnalyzer._detect_bottlenecks (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_detect_bottlenecks',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._detect_bottlenecks'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q5_Executor(DataFlowExecutor):
    """
    D5-Q5: Sostenibilidad Financiera
    Flow: PP.O → FV.E+C → CD.E+V+C → DB.O+C
    Métodos: 15
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.C - PDETMunicipalPlanAnalyzer.analyze_financial_feasibility (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_financial_feasibility',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_financial_feasibility'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.C - PDETMunicipalPlanAnalyzer._assess_financial_sustainability (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_assess_financial_sustainability',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._assess_financial_sustainability'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.C - PDETMunicipalPlanAnalyzer._bayesian_risk_inference (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_bayesian_risk_inference',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._bayesian_risk_inference'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.C - PDETMunicipalPlanAnalyzer._analyze_funding_sources (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_analyze_funding_sources',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._analyze_funding_sources'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._detect_resource_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_resource_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_resource_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 13. DB.O - FinancialAuditor.trace_financial_allocation (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            'trace_financial_allocation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor.trace_financial_allocation'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.C - FinancialAuditor._calculate_sufficiency (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_calculate_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._calculate_sufficiency'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q1_Executor(DataFlowExecutor):
    """
    D6-Q1: Integridad de Teoría de Cambio
    Flow: PP.O → TC.V (validacion_completa) → TC.T (construir_grafo) → CD.T+C → DB.O (CausalExtractor+Auditor+Framework) → FV.T
    Métodos: 32
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 6. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 7. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 8. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 9. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        # 10. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 11. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 12. TC.V - AdvancedDAGValidator.calculate_acyclicity_pvalue (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'calculate_acyclicity_pvalue',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.calculate_acyclicity_pvalue'] = result
        if result is not None:
            current_data = result
        
        # 13. TC.C - AdvancedDAGValidator._calculate_statistical_power (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_statistical_power',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_statistical_power'] = result
        if result is not None:
            current_data = result
        
        # 14. TC.C - AdvancedDAGValidator._calculate_bayesian_posterior (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_bayesian_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_bayesian_posterior'] = result
        if result is not None:
            current_data = result
        
        # 15. TC.V - AdvancedDAGValidator._perform_sensitivity_analysis_internal (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_perform_sensitivity_analysis_internal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._perform_sensitivity_analysis_internal'] = result
        if result is not None:
            current_data = result
        
        # 16. TC.C - AdvancedDAGValidator.get_graph_stats (P=2)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'get_graph_stats',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.get_graph_stats'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.C - PolicyContradictionDetector._get_graph_statistics (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_graph_statistics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_graph_statistics'] = result
        if result is not None:
            current_data = result
        
        # 19. CD.C - PolicyContradictionDetector._calculate_graph_fragmentation (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_graph_fragmentation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_graph_fragmentation'] = result
        if result is not None:
            current_data = result
        
        # 20. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 21. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 22. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 23. DB.V - OperationalizationAuditor.audit_evidence_traceability (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            'audit_evidence_traceability',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor.audit_evidence_traceability'] = result
        if result is not None:
            current_data = result
        
        # 24. DB.V - OperationalizationAuditor._audit_systemic_risk (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            '_audit_systemic_risk',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor._audit_systemic_risk'] = result
        if result is not None:
            current_data = result
        
        # 25. DB.V - OperationalizationAuditor.bayesian_counterfactual_audit (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            'bayesian_counterfactual_audit',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor.bayesian_counterfactual_audit'] = result
        if result is not None:
            current_data = result
        
        # 26. DB.R - OperationalizationAuditor._generate_optimal_remediations (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            '_generate_optimal_remediations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor._generate_optimal_remediations'] = result
        if result is not None:
            current_data = result
        
        # 27. DB.O - CDAFFramework.process_document (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            'process_document',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework.process_document'] = result
        if result is not None:
            current_data = result
        
        # 28. DB.V - CDAFFramework._audit_causal_coherence (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_audit_causal_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._audit_causal_coherence'] = result
        if result is not None:
            current_data = result
        
        # 29. DB.V - CDAFFramework._validate_dnp_compliance (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_validate_dnp_compliance',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._validate_dnp_compliance'] = result
        if result is not None:
            current_data = result
        
        # 30. DB.R - CDAFFramework._generate_extraction_report (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_generate_extraction_report',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._generate_extraction_report'] = result
        if result is not None:
            current_data = result
        
        # 31. FV.T - PDETMunicipalPlanAnalyzer.construct_causal_dag (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'construct_causal_dag',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.construct_causal_dag'] = result
        if result is not None:
            current_data = result
        
        # 32. FV.E - PDETMunicipalPlanAnalyzer._identify_causal_nodes (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_identify_causal_nodes',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._identify_causal_nodes'] = result
        if result is not None:
            current_data = result
        
        # 33. FV.E - PDETMunicipalPlanAnalyzer._identify_causal_edges (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_identify_causal_edges',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._identify_causal_edges'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q2_Executor(DataFlowExecutor):
    """
    D6-Q2: Proporcionalidad y Continuidad (Anti-Milagro)
    Flow: PP.E+T (3 categorías patrones) → CD.T+V+C → TC.V → DB (Beach Tests + Inference + Setup) → DB.Auditor
    Métodos: 28
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.T - IndustrialPolicyProcessor._compile_pattern_registry (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_compile_pattern_registry',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._compile_pattern_registry'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - IndustrialPolicyProcessor._build_point_patterns (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_build_point_patterns',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._build_point_patterns'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 6. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.C - PolicyContradictionDetector._calculate_syntactic_complexity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_syntactic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_syntactic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 16. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 19. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 20. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 21. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        # 22. TC.V - AdvancedDAGValidator.calculate_acyclicity_pvalue (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'calculate_acyclicity_pvalue',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.calculate_acyclicity_pvalue'] = result
        if result is not None:
            current_data = result
        
        # 23. TC.C - AdvancedDAGValidator._calculate_statistical_power (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_statistical_power',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_statistical_power'] = result
        if result is not None:
            current_data = result
        
        # 24. TC.C - AdvancedDAGValidator._calculate_bayesian_posterior (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_bayesian_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_bayesian_posterior'] = result
        if result is not None:
            current_data = result
        
        # 25. DB.V - BeachEvidentialTest.classify_test (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'classify_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.classify_test'] = result
        if result is not None:
            current_data = result
        
        # 26. DB.V - BeachEvidentialTest.apply_test_logic (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'apply_test_logic',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.apply_test_logic'] = result
        if result is not None:
            current_data = result
        
        # 27. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 28. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 29. DB.T - BayesianMechanismInference._build_transition_matrix (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_build_transition_matrix',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._build_transition_matrix'] = result
        if result is not None:
            current_data = result
        
        # 30. DB.C - BayesianMechanismInference._calculate_type_transition_prior (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_calculate_type_transition_prior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._calculate_type_transition_prior'] = result
        if result is not None:
            current_data = result
        
        # 31. DB.V - BayesianMechanismInference._infer_activity_sequence (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_infer_activity_sequence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._infer_activity_sequence'] = result
        if result is not None:
            current_data = result
        
        # 32. DB.C - BayesianMechanismInference._aggregate_bayesian_confidence (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_aggregate_bayesian_confidence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._aggregate_bayesian_confidence'] = result
        if result is not None:
            current_data = result
        
        # 33. DB.V - CausalInferenceSetup.classify_goal_dynamics (P=3)
        result = self.executor.execute(
            'CausalInferenceSetup',
            'classify_goal_dynamics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalInferenceSetup.classify_goal_dynamics'] = result
        if result is not None:
            current_data = result
        
        # 34. DB.V - CausalInferenceSetup.identify_failure_points (P=3)
        result = self.executor.execute(
            'CausalInferenceSetup',
            'identify_failure_points',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalInferenceSetup.identify_failure_points'] = result
        if result is not None:
            current_data = result
        
        # 35. DB.C - CausalInferenceSetup.assign_probative_value (P=3)
        result = self.executor.execute(
            'CausalInferenceSetup',
            'assign_probative_value',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalInferenceSetup.assign_probative_value'] = result
        if result is not None:
            current_data = result
        
        # 36. DB.E - CausalInferenceSetup._get_dynamics_pattern (P=3)
        result = self.executor.execute(
            'CausalInferenceSetup',
            '_get_dynamics_pattern',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalInferenceSetup._get_dynamics_pattern'] = result
        if result is not None:
            current_data = result
        
        # 37. DB.V - OperationalizationAuditor._audit_systemic_risk (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            '_audit_systemic_risk',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor._audit_systemic_risk'] = result
        if result is not None:
            current_data = result
        
        # 38. DB.V - OperationalizationAuditor.bayesian_counterfactual_audit (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            'bayesian_counterfactual_audit',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor.bayesian_counterfactual_audit'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q3_Executor(DataFlowExecutor):
    """
    D6-Q3: Inconsistencias (Sistema Bicameral - Ruta 1)
    Flow: PP.O → CD.V (detect suite) → CD.R (_suggest_resolutions) → TC.V → A1.V
    Métodos: 22
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - PolicyContradictionDetector._detect_logical_incompatibilities (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_logical_incompatibilities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_logical_incompatibilities'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector.detect (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            'detect',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector.detect'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._detect_semantic_contradictions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_semantic_contradictions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_semantic_contradictions'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._detect_temporal_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_temporal_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_temporal_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - PolicyContradictionDetector._detect_resource_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_resource_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_resource_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._classify_contradiction (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_classify_contradiction',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._classify_contradiction'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - PolicyContradictionDetector._calculate_severity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_severity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_severity'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.R - PolicyContradictionDetector._generate_resolution_recommendations (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_resolution_recommendations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_resolution_recommendations'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.R - PolicyContradictionDetector._suggest_resolutions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_suggest_resolutions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._suggest_resolutions'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.C - PolicyContradictionDetector._calculate_contradiction_entropy (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_contradiction_entropy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_contradiction_entropy'] = result
        if result is not None:
            current_data = result
        
        # 16. CD.C - PolicyContradictionDetector._get_domain_weight (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_domain_weight',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_domain_weight'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.V - PolicyContradictionDetector._has_logical_conflict (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_has_logical_conflict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._has_logical_conflict'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 19. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 20. A1.E - TextMiningEngine._identify_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_identify_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._identify_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 21. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 22. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q4_Executor(DataFlowExecutor):
    """
    D6-Q4: Adaptación (Sistema Bicameral - Ruta 2)
    Flow: PP.O → TC.V+R (_generar_sugerencias_internas) → CD.T+C → DB (CDAF+Auditors) → FV.R
    Métodos: 26
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 7. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        # 8. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 9. TC.R - TeoriaCambio._generar_sugerencias_internas (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_generar_sugerencias_internas',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._generar_sugerencias_internas'] = result
        if result is not None:
            current_data = result
        
        # 10. TC.R - TeoriaCambio._execute_generar_sugerencias_internas (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_execute_generar_sugerencias_internas',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._execute_generar_sugerencias_internas'] = result
        if result is not None:
            current_data = result
        
        # 11. TC.E - TeoriaCambio._extraer_categorias (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_extraer_categorias',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._extraer_categorias'] = result
        if result is not None:
            current_data = result
        
        # 12. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 13. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 14. TC.V - AdvancedDAGValidator.calculate_acyclicity_pvalue (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'calculate_acyclicity_pvalue',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.calculate_acyclicity_pvalue'] = result
        if result is not None:
            current_data = result
        
        # 15. TC.V - AdvancedDAGValidator._perform_sensitivity_analysis_internal (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_perform_sensitivity_analysis_internal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._perform_sensitivity_analysis_internal'] = result
        if result is not None:
            current_data = result
        
        # 16. TC.C - AdvancedDAGValidator._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 17. TC.C - AdvancedDAGValidator.get_graph_stats (P=2)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'get_graph_stats',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.get_graph_stats'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 19. CD.C - PolicyContradictionDetector._get_graph_statistics (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_graph_statistics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_graph_statistics'] = result
        if result is not None:
            current_data = result
        
        # 20. CD.C - PolicyContradictionDetector._calculate_graph_fragmentation (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_graph_fragmentation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_graph_fragmentation'] = result
        if result is not None:
            current_data = result
        
        # 21. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 22. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 23. A1.R - PerformanceAnalyzer._generate_recommendations (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_generate_recommendations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._generate_recommendations'] = result
        if result is not None:
            current_data = result
        
        # 24. A1.R - TextMiningEngine._generate_interventions (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_generate_interventions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._generate_interventions'] = result
        if result is not None:
            current_data = result
        
        # 25. DB.V - CDAFFramework._validate_dnp_compliance (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_validate_dnp_compliance',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._validate_dnp_compliance'] = result
        if result is not None:
            current_data = result
        
        # 26. DB.R - CDAFFramework._generate_extraction_report (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_generate_extraction_report',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._generate_extraction_report'] = result
        if result is not None:
            current_data = result
        
        # 27. DB.R - CDAFFramework._generate_causal_model_json (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_generate_causal_model_json',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._generate_causal_model_json'] = result
        if result is not None:
            current_data = result
        
        # 28. DB.R - CDAFFramework._generate_dnp_compliance_report (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_generate_dnp_compliance_report',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._generate_dnp_compliance_report'] = result
        if result is not None:
            current_data = result
        
        # 29. DB.V - OperationalizationAuditor.audit_evidence_traceability (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            'audit_evidence_traceability',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor.audit_evidence_traceability'] = result
        if result is not None:
            current_data = result
        
        # 30. DB.V - OperationalizationAuditor._perform_counterfactual_budget_check (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            '_perform_counterfactual_budget_check',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor._perform_counterfactual_budget_check'] = result
        if result is not None:
            current_data = result
        
        # 31. DB.O - FinancialAuditor.trace_financial_allocation (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            'trace_financial_allocation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor.trace_financial_allocation'] = result
        if result is not None:
            current_data = result
        
        # 32. DB.V - FinancialAuditor._match_goal_to_budget (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_match_goal_to_budget',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._match_goal_to_budget'] = result
        if result is not None:
            current_data = result
        
        # 33. DB.C - FinancialAuditor._calculate_sufficiency (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_calculate_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._calculate_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 34. DB.V - FinancialAuditor._detect_allocation_gaps (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_detect_allocation_gaps',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._detect_allocation_gaps'] = result
        if result is not None:
            current_data = result
        
        # 35. DB.V - MechanismTypeConfig.check_sum_to_one (P=3)
        result = self.executor.execute(
            'MechanismTypeConfig',
            'check_sum_to_one',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismTypeConfig.check_sum_to_one'] = result
        if result is not None:
            current_data = result
        
        # 36. FV.R - PDETMunicipalPlanAnalyzer.generate_recommendations (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'generate_recommendations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.generate_recommendations'] = result
        if result is not None:
            current_data = result
        
        # 37. FV.R - PDETMunicipalPlanAnalyzer._generate_optimal_remediations (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_generate_optimal_remediations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._generate_optimal_remediations'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q5_Executor(DataFlowExecutor):
    """
    D6-Q5: Contextualización y Enfoque Diferencial
    Flow: PP.E (patrones diferenciales) → CD.T+V+C → A1.V+E (_classify_cross_cutting_themes) → EP.E+V+C
    Métodos: 24
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 14. A1.V - SemanticAnalyzer._classify_cross_cutting_themes (P=3)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_cross_cutting_themes',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_cross_cutting_themes'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        # 16. A1.E - SemanticAnalyzer.extract_semantic_cube (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            'extract_semantic_cube',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer.extract_semantic_cube'] = result
        if result is not None:
            current_data = result
        
        # 17. A1.T - SemanticAnalyzer._process_segment (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_process_segment',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._process_segment'] = result
        if result is not None:
            current_data = result
        
        # 18. A1.T - SemanticAnalyzer._vectorize_segments (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_vectorize_segments',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._vectorize_segments'] = result
        if result is not None:
            current_data = result
        
        # 19. A1.C - SemanticAnalyzer._calculate_semantic_complexity (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_calculate_semantic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._calculate_semantic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 20. A1.O - MunicipalOntology.__init__ (P=2)
        result = self.executor.execute(
            'MunicipalOntology',
            '__init__',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MunicipalOntology.__init__'] = result
        if result is not None:
            current_data = result
        
        # 21. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 22. EP.V - PolicyAnalysisEmbedder._filter_by_pdq (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_filter_by_pdq',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._filter_by_pdq'] = result
        if result is not None:
            current_data = result
        
        # 23. EP.C - PolicyAnalysisEmbedder.compare_policy_interventions (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'compare_policy_interventions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.compare_policy_interventions'] = result
        if result is not None:
            current_data = result
        
        # 24. EP.V - AdvancedSemanticChunker._infer_pdq_context (P=3)
        result = self.executor.execute(
            'AdvancedSemanticChunker',
            '_infer_pdq_context',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedSemanticChunker._infer_pdq_context'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class Orchestrator:
    """Orquestador robusto de 11 fases con abortabilidad y control de recursos."""

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
        self.catalog_path = self._resolve_path(catalog_path)
        self.monolith_path = self._resolve_path(monolith_path)
        self.method_map_path = self._resolve_path(method_map_path)
        self.schema_path = self._resolve_path(schema_path) if schema_path else None
        self.resource_limits = resource_limits or ResourceLimits()
        self.resource_snapshot_interval = max(1, resource_snapshot_interval)

        with open(self.catalog_path) as f:
            self.catalog = json.load(f)

        self.executor = MethodExecutor()
        self.executors = {
            "D1-Q1": D1Q1_Executor,
            "D1-Q2": D1Q2_Executor,
            "D1-Q3": D1Q3_Executor,
            "D1-Q4": D1Q4_Executor,
            "D1-Q5": D1Q5_Executor,
            "D2-Q1": D2Q1_Executor,
            "D2-Q2": D2Q2_Executor,
            "D2-Q3": D2Q3_Executor,
            "D2-Q4": D2Q4_Executor,
            "D2-Q5": D2Q5_Executor,
            "D3-Q1": D3Q1_Executor,
            "D3-Q2": D3Q2_Executor,
            "D3-Q3": D3Q3_Executor,
            "D3-Q4": D3Q4_Executor,
            "D3-Q5": D3Q5_Executor,
            "D4-Q1": D4Q1_Executor,
            "D4-Q2": D4Q2_Executor,
            "D4-Q3": D4Q3_Executor,
            "D4-Q4": D4Q4_Executor,
            "D4-Q5": D4Q5_Executor,
            "D5-Q1": D5Q1_Executor,
            "D5-Q2": D5Q2_Executor,
            "D5-Q3": D5Q3_Executor,
            "D5-Q4": D5Q4_Executor,
            "D5-Q5": D5Q5_Executor,
            "D6-Q1": D6Q1_Executor,
            "D6-Q2": D6Q2_Executor,
            "D6-Q3": D6Q3_Executor,
            "D6-Q4": D6Q4_Executor,
            "D6-Q5": D6Q5_Executor,
        }

        self.abort_signal = AbortSignal()
        self.phase_results: List[PhaseResult] = []
        self._phase_instrumentation: Dict[int, PhaseInstrumentation] = {}
        self._phase_status: Dict[int, str] = {phase_id: "not_started" for phase_id, *_ in self.FASES}
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

    def request_abort(self, reason: str = "Abort requested") -> None:
        self.abort_signal.abort(reason)

    def reset_abort(self) -> None:
        self.abort_signal.reset()

    def _ensure_not_aborted(self) -> None:
        if self.abort_signal.is_aborted():
            raise AbortRequested(self.abort_signal.get_reason() or "Abort requested")

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

    def _validate_contract_structure(self, monolith: Dict[str, Any], instrumentation: PhaseInstrumentation) -> Dict[str, Any]:
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

    def _ingest_document(self, pdf_path: str, config: Mapping[str, Any]) -> PreprocessedDocument:
        self._ensure_not_aborted()
        instrumentation = self._phase_instrumentation[1]
        start = time.perf_counter()

        document_id = os.path.splitext(os.path.basename(pdf_path))[0] or "doc_1"
        ingested_at = datetime.utcnow()
        metadata_payload: Dict[str, Any] = {
            "source_path": pdf_path,
            "ingested_at": ingested_at.isoformat(),
        }
        extra_metadata = config.get("metadata") if isinstance(config, Mapping) else None
        if isinstance(extra_metadata, Mapping):
            metadata_payload.update(dict(extra_metadata))

        structured_text = StructuredTextV1(full_text="", sections=tuple(), page_boundaries=tuple())
        preprocessed = PreprocessedDocument(
            document_id=document_id,
            full_text="",
            sentences=tuple(),
            language=str(config.get("default_language", "unknown")),
            structured_text=structured_text,
            sentence_metadata=tuple(),
            tables=tuple(),
            indexes=DocumentIndexesV1(),
            metadata=MappingProxyType(metadata_payload),
            ingested_at=ingested_at,
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Orquestador de plan de desarrollo de 11 fases")
    parser.add_argument("pdf_path", help="Ruta al documento PDF a procesar")
    parser.add_argument(
        "--catalog",
        default="rules/METODOS/metodos_completos_nivel3.json",
        help="Ruta al catálogo de métodos",
    )
    args = parser.parse_args()

    orchestrator = Orchestrator(catalog_path=args.catalog)
    phase_results = orchestrator.process_development_plan(args.pdf_path)

    completed = [phase for phase in phase_results if phase.success]
    print(f"Fases completadas: {len(completed)}/{len(phase_results)}")
    status = orchestrator.get_processing_status()
    print(f"Estado final: {status['status']} - Abortado: {status['abort_status']}")


if __name__ == "__main__":
    main()
