"""Orchestrator utilities with contract validation on import."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional, Union

class _QuestionnaireProvider:
    """Centralized access to the questionnaire monolith payload.

    The provider is responsible for wiring the orchestrator to the canonical
    ``questionnaire_monolith.json`` file.  It offers a small utility surface
    that mirrors the historical API (``load``/``exists``/``describe``/``save``)
    so the legacy orchestrator package and external tooling (scripts/tests)
    continue to function while newer code paths inject pre-loaded data via
    :meth:`set_data`.
    """

    _DEFAULT_RELATIVE_PATH = Path("data") / "questionnaire_monolith.json"

    def __init__(
        self,
        initial_data: Optional[Dict[str, Any]] = None,
        data_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialise provider with optional pre-loaded data and path hint."""

        self._lock = RLock()
        self._cache: Optional[Dict[str, Any]] = initial_data
        if data_path is None:
            self._data_path = self._default_repo_root() / self._DEFAULT_RELATIVE_PATH
        else:
            self._data_path = self._coerce_path(data_path)

    @staticmethod
    def _default_repo_root() -> Path:
        """Return repository root used to resolve the questionnaire path."""

        return Path(__file__).resolve().parents[4]

    def _coerce_path(self, candidate: Union[str, Path]) -> Path:
        path = Path(candidate)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _resolve_path(self, candidate: Optional[Union[str, Path]]) -> Path:
        """Resolve *candidate* to an absolute path, defaulting to ``data_path``."""

        if candidate is None:
            return self._data_path
        return self._coerce_path(candidate)

    @property
    def data_path(self) -> Path:
        """Return the resolved path of the canonical questionnaire payload."""

        return self._data_path

    def set_data(self, data: Dict[str, Any]) -> None:
        """Inject pre-loaded questionnaire data (typically provided by factory).

        The provider caches the payload so subsequent :meth:`load` calls can
        re-use the in-memory copy without hitting the filesystem.
        """

        with self._lock:
            self._cache = data

    def get_data(self) -> Dict[str, Any]:
        """Return the cached questionnaire data.

        Raises:
            RuntimeError: If the questionnaire has not been loaded or injected
                yet.
        """

        with self._lock:
            if self._cache is None:
                raise RuntimeError(
                    "Questionnaire data not loaded. Use factory.py to load data first."
                )
            return self._cache

    def has_data(self) -> bool:
        """Return ``True`` when an in-memory questionnaire payload is cached."""

        with self._lock:
            return self._cache is not None

    # ------------------------------------------------------------------
    # Legacy compatibility helpers
    # ------------------------------------------------------------------

    def exists(self, data_path: Optional[Union[str, Path]] = None) -> bool:
        """Check whether a questionnaire payload exists on disk."""

        return self._resolve_path(data_path).exists()

    def describe(self, data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Return filesystem metadata for the questionnaire payload."""

        path = self._resolve_path(data_path)
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        return {"path": path, "exists": exists, "size": size}

    def _read_payload(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def load(
        self,
        *,
        force_reload: bool = False,
        data_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Load and optionally cache the questionnaire payload from disk.

        When ``data_path`` is ``None`` the canonical monolith is used.  The
        result is cached unless ``force_reload`` is ``True``.  Supplying a
        custom ``data_path`` bypasses the cache so callers can inspect arbitrary
        payloads without mutating the shared state.
        """

        target_path = self._resolve_path(data_path)
        with self._lock:
            if data_path is None:
                if self._cache is not None and not force_reload:
                    return self._cache
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
        *,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Persist a questionnaire payload to disk and refresh the cache."""

        target_path = self._resolve_path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)

        with self._lock:
            self._cache = payload
            if output_path is not None:
                self._data_path = target_path
        return target_path

    def get_question(self, question_global: int) -> Dict[str, Any]:
        """Return a single question entry from the cached payload."""

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
    """Get the global questionnaire provider instance."""
    return _questionnaire_provider

def get_questionnaire_payload() -> Dict[str, Any]:
    """Get questionnaire payload with caller boundary enforcement.
    
    Note: Data must be pre-loaded via factory.py before calling this function.
    
    Returns:
        Questionnaire payload dictionary
        
    Raises:
        RuntimeError: If called from outside orchestrator package or if data not loaded
    """
    caller_frame = inspect.currentframe().f_back
    caller_module = caller_frame.f_globals.get('__name__', '')
    if not caller_module.startswith('orchestrator'):
        raise RuntimeError("Questionnaire provider access restricted to orchestrator package")
    return _questionnaire_provider.get_data()

# Import utilities from submodules
from .evidence_registry import (
    EvidenceRecord,
    EvidenceRegistry,
    ProvenanceDAG,
    ProvenanceNode,
    get_global_registry,
)

from .contract_loader import (
    JSONContractLoader,
    LoadError,
    LoadResult,
)

# Import core classes from the refactored package
from .core import (
    Orchestrator,
    MethodExecutor,
    PreprocessedDocument,
    Evidence,
    AbortSignal,
    AbortRequested,
    ResourceLimits,
    PhaseInstrumentation,
    PhaseResult,
    MicroQuestionRun,
    ScoredMicroQuestion,
)

__all__ = [
    "EvidenceRecord",
    "EvidenceRegistry",
    "ProvenanceDAG",
    "ProvenanceNode",
    "get_global_registry",
    "JSONContractLoader",
    "LoadError",
    "LoadResult",
    "get_questionnaire_provider",
    "get_questionnaire_payload",
    "Orchestrator",
    "MethodExecutor",
    "PreprocessedDocument",
    "Evidence",
    "AbortSignal",
    "AbortRequested",
    "ResourceLimits",
    "PhaseInstrumentation",
    "PhaseResult",
    "MicroQuestionRun",
    "ScoredMicroQuestion",
]
