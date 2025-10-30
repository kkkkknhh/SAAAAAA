"""Orchestrator utilities with contract validation on import."""

import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional, Union

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
    """Return the monolith entry for a given global question identifier."""
    return _questionnaire_provider.get_question(question_global)


try:
    from .canonical_registry import CANONICAL_METHODS
    _CANONICAL_AVAILABLE = True
except ImportError:
    CANONICAL_METHODS = {}
    _CANONICAL_AVAILABLE = False

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
try:
    from .d1_orchestrator import (
        D1Question,
        D1QuestionOrchestrator,
        D1OrchestrationError,
        ExecutionTrace,
        MethodContract,
        OrchestrationResult,
    )
    _D1_AVAILABLE = True
except ImportError:
    _D1_AVAILABLE = False

try:  # pragma: no cover - executed at import time
    from schema_validator import SchemaValidator

    _validator = SchemaValidator()
    _q_report, _r_report, _c_report = _validator.validate_all()
    if any(report.errors for report in (_q_report, _r_report, _c_report)):
        raise RuntimeError(
            "Data contract validation failed; see schema_validator output for details."
        )
except ImportError:  # pragma: no cover - schema_validator not available
    pass  # Schema validation is optional
except Exception as exc:  # pragma: no cover - validation failure path
    raise

__all__ = [
    "CANONICAL_METHODS",
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
    "get_question_payload",
]

if _D1_AVAILABLE:
    __all__.extend([
        "D1Question",
        "D1QuestionOrchestrator",
        "D1OrchestrationError",
        "ExecutionTrace",
        "MethodContract",
        "OrchestrationResult",
    ])


