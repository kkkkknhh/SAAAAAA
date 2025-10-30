"""Orchestrator utilities with contract validation on import."""
import json
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional, Union

class _QuestionnaireProvider:
    """Centralized access to the questionnaire monolith payload."""
    _DEFAULT_PATH = Path(__file__).resolve().parent.parent / "questionnaire_monolith.json"
    
    def __init__(self, data_path: Optional[Path] = None) -> None:
        self._data_path = data_path or self._DEFAULT_PATH
        self._cache: Optional[Dict[str, Any]] = None
        self._lock = RLock()
    
    @property
    def data_path(self) -> Path:
        return self._data_path
    
    def _resolve_path(self, candidate: Optional[Union[str, Path]] = None) -> Path:
        if candidate is None:
            return self._data_path
        path = Path(candidate) if not isinstance(candidate, Path) else candidate
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path
    
    def exists(self, data_path: Optional[Union[str, Path]] = None) -> bool:
        return self._resolve_path(data_path).exists()
    
    def load(self, force_reload: bool = False, data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        target_path = self._resolve_path(data_path)
        with self._lock:
            if data_path is None:
                if force_reload or self._cache is None:
                    if not target_path.exists():
                        raise FileNotFoundError(f"Questionnaire payload missing at {target_path}")
                    with target_path.open("r", encoding="utf-8") as f:
                        self._cache = json.load(f)
                return self._cache
            if not target_path.exists():
                raise FileNotFoundError(f"Questionnaire payload missing at {target_path}")
            with target_path.open("r", encoding="utf-8") as f:
                return json.load(f)

_questionnaire_provider = _QuestionnaireProvider()

def get_questionnaire_provider() -> _QuestionnaireProvider:
    return _questionnaire_provider

def get_questionnaire_payload(force_reload: bool = False) -> Dict[str, Any]:
    return _questionnaire_provider.load(force_reload=force_reload)

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

# Import main classes from root orchestrator.py
import sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Import from orchestrator.py at root
import importlib.util as _util
_spec = _util.spec_from_file_location("_orch_main", _root / "orchestrator.py")
_orch_main = _util.module_from_spec(_spec)
_spec.loader.exec_module(_orch_main)

Orchestrator = _orch_main.Orchestrator
MethodExecutor = _orch_main.MethodExecutor

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
]
