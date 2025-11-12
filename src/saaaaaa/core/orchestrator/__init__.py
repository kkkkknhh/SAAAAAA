"""Orchestrator utilities with contract validation on import."""
import inspect
import logging
from threading import RLock
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .factory import CanonicalQuestionnaire

logger = logging.getLogger(__name__)


class _QuestionnaireProvider:
    """Centralized access to the questionnaire monolith payload.

    This provider now supports both CanonicalQuestionnaire (preferred) and
    legacy dict format for backward compatibility. New code should use
    CanonicalQuestionnaire for immutability and hash verification.
    """

    def __init__(
        self, 
        initial_data: "dict[str, Any] | CanonicalQuestionnaire | None" = None
    ) -> None:
        """Initialize provider with optional pre-loaded data.

        Args:
            initial_data: Pre-loaded questionnaire data. Can be CanonicalQuestionnaire
                         (preferred) or dict (legacy). If None, data must be set via
                         set_data() before calling get_data().
        """
        self._cache: "dict[str, Any] | CanonicalQuestionnaire | None" = initial_data
        self._lock = RLock()

    def set_data(self, data: "dict[str, Any] | CanonicalQuestionnaire") -> None:
        """Set questionnaire data (typically called by factory).

        Args:
            data: Questionnaire payload - CanonicalQuestionnaire (preferred) or dict (legacy)
        """
        with self._lock:
            # Import here to avoid circular dependency
            from .factory import CanonicalQuestionnaire
            
            if isinstance(data, CanonicalQuestionnaire):
                logger.info(
                    f"Provider loaded CanonicalQuestionnaire: "
                    f"hash={data.sha256[:16]}..., questions={data.question_count}"
                )
            elif isinstance(data, dict):
                logger.warning(
                    "Provider loaded mutable dict (deprecated). "
                    "Use CanonicalQuestionnaire for immutability and hash verification."
                )
            
            self._cache = data

    def get_data(self) -> "dict[str, Any] | CanonicalQuestionnaire":
        """Get cached questionnaire data.

        Returns:
            Questionnaire data - CanonicalQuestionnaire or dict

        Raises:
            RuntimeError: If no data has been loaded yet
        """
        with self._lock:
            if self._cache is None:
                raise RuntimeError(
                    "Questionnaire data not loaded. Use factory.load_questionnaire() first."
                )
            return self._cache

    def get_canonical(self) -> "CanonicalQuestionnaire":
        """Get questionnaire as CanonicalQuestionnaire.
        
        If the cached data is a dict, this will load it as CanonicalQuestionnaire.
        
        Returns:
            CanonicalQuestionnaire: Immutable, hash-verified questionnaire
            
        Raises:
            RuntimeError: If no data has been loaded yet
        """
        with self._lock:
            from .factory import CanonicalQuestionnaire, load_questionnaire
            
            if self._cache is None:
                # Load canonical questionnaire
                logger.info("Auto-loading canonical questionnaire")
                self._cache = load_questionnaire()
            elif isinstance(self._cache, dict):
                # Convert dict to CanonicalQuestionnaire
                logger.warning(
                    "Converting cached dict to CanonicalQuestionnaire. "
                    "This should only happen during migration."
                )
                self._cache = load_questionnaire()
            
            return self._cache

    def has_data(self) -> bool:
        """Check if data is loaded.

        Returns:
            True if data is available, False otherwise
        """
        with self._lock:
            return self._cache is not None

    def exists(self) -> bool:
        """Alias for has_data() for backward compatibility.

        Returns:
            True if data is available, False otherwise
        """
        return self.has_data()

_questionnaire_provider = _QuestionnaireProvider()

def get_questionnaire_provider() -> _QuestionnaireProvider:
    """Get the global questionnaire provider instance."""
    return _questionnaire_provider

def get_questionnaire_payload() -> dict[str, Any]:
    """Get questionnaire payload with caller boundary enforcement.

    Note: Data must be pre-loaded via factory.py before calling this function.

    Returns:
        Questionnaire payload dictionary

    Raises:
        RuntimeError: If called from outside orchestrator package or if data not loaded
    """
    caller_frame = inspect.currentframe().f_back
    caller_module = caller_frame.f_globals.get('__name__', '')
    if not caller_module.startswith('saaaaaa.core.orchestrator'):
        raise RuntimeError("Questionnaire provider access restricted to orchestrator package")
    return _questionnaire_provider.get_data()

# Import utilities from submodules
from .contract_loader import (
    JSONContractLoader,
    LoadError,
    LoadResult,
)

# Import factory types for better type hints
from .factory import (
    CanonicalQuestionnaire,
    load_questionnaire,
    load_questionnaire_monolith,
)

# Import core classes from the refactored package
from .core import (
    AbortRequested,
    AbortSignal,
    Evidence,
    MethodExecutor,
    MicroQuestionRun,
    Orchestrator,
    PhaseInstrumentation,
    PhaseResult,
    PreprocessedDocument,
    ResourceLimits,
    ScoredMicroQuestion,
)
from .evidence_registry import (
    EvidenceRecord,
    EvidenceRegistry,
    ProvenanceDAG,
    ProvenanceNode,
    get_global_registry,
)

__all__ = [
    "CanonicalQuestionnaire",
    "load_questionnaire",
    "load_questionnaire_monolith",
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
