"""
Factory module for core module initialization with dependency injection.

This module is responsible for:
1. Reading data from disk (questionnaire_monolith.json, etc.)
2. Constructing InputContracts for core modules
3. Initializing core modules with injected dependencies
4. Managing I/O operations so core modules remain pure

Architectural Pattern:
- Factory reads from disk
- Factory constructs contracts
- Factory injects dependencies into core modules
- Core modules remain I/O-free and testable

QUESTIONNAIRE INTEGRITY PROTOCOL:
This is the ONLY module that should load questionnaire_monolith.json.
All consumers MUST use load_questionnaire() which returns CanonicalQuestionnaire.

Version: 2.0.0
Status: Questionnaire determinism enforcement implemented
"""

import copy
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Optional

from ..contracts import (
    CDAFFrameworkInputContract,
    ContradictionDetectorInputContract,
    DocumentData,
    EmbeddingPolicyInputContract,
    PDETAnalyzerInputContract,
    PolicyProcessorInputContract,
    SemanticAnalyzerInputContract,
    SemanticChunkingInputContract,
    TeoriaCambioInputContract,
)
from . import get_questionnaire_provider
from .core import MethodExecutor

logger = logging.getLogger(__name__)

# Canonical repository root - single source of truth for all file paths
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"

# QUESTIONNAIRE INTEGRITY CONSTANTS
# The ONLY valid questionnaire path
QUESTIONNAIRE_PATH: Final[Path] = _DEFAULT_DATA_DIR / "questionnaire_monolith.json"

# Expected structure hash - MUST match actual file or load fails
# This hash is computed using: json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
# Update this when questionnaire is legitimately modified
EXPECTED_HASH: Final[str] = "27f7f784583d637158cb70ee236f1a98f77c1a08366612b5ae11f3be24062658"

# Expected question counts (enforcement levels)
EXPECTED_MICRO_QUESTION_COUNT: Final[int] = 300
EXPECTED_MESO_QUESTION_COUNT: Final[int] = 4
EXPECTED_TOTAL_QUESTION_COUNT: Final[int] = 305  # 300 micro + 4 meso + 1 macro

@dataclass(frozen=True)
class CanonicalQuestionnaire:
    """Immutable, validated, hash-verified questionnaire.

    This is the ONLY valid representation of questionnaire data in the system.
    All questionnaire consumers MUST accept this type, not raw dicts.

    Attributes:
        data: Immutable view of questionnaire structure
        sha256: Computed SHA-256 hash (must match EXPECTED_HASH)
        micro_questions: Immutable tuple of micro questions
        meso_questions: Immutable tuple of meso questions
        macro_question: Immutable macro question or None
        micro_question_count: Number of micro questions (must be 300)
        total_question_count: Total questions including meso + macro (must be 305)
        version: Questionnaire version string
        schema_version: Schema version string

    Invariants enforced at construction:
        - sha256 == EXPECTED_HASH
        - micro_question_count == 300
        - total_question_count == 305
        - All data structures are immutable (MappingProxyType/tuple)
        - No None values in required fields
        - No duplicate question_id or question_global values
    """

    data: MappingProxyType[str, Any]
    sha256: str
    micro_questions: tuple[MappingProxyType, ...]
    meso_questions: tuple[MappingProxyType, ...]
    macro_question: MappingProxyType | None
    micro_question_count: int
    total_question_count: int
    version: str
    schema_version: str

    def __post_init__(self) -> None:
        """Validate all invariants on construction.

        Raises:
            ValueError: If any invariant is violated
        """
        # Hash verification
        if self.sha256 != EXPECTED_HASH:
            raise ValueError(
                f"QUESTIONNAIRE INTEGRITY VIOLATION: Hash mismatch!\n"
                f"Expected: {EXPECTED_HASH}\n"
                f"Got:      {self.sha256}\n"
                f"The questionnaire file has been modified. If this was intentional, "
                f"update EXPECTED_HASH in factory.py"
            )

        # Count verification
        if self.micro_question_count != EXPECTED_MICRO_QUESTION_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_MICRO_QUESTION_COUNT} micro questions, "
                f"got {self.micro_question_count}"
            )

        if self.total_question_count != EXPECTED_TOTAL_QUESTION_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_TOTAL_QUESTION_COUNT} total questions, "
                f"got {self.total_question_count}"
            )

        # Immutability verification
        if not isinstance(self.data, MappingProxyType):
            raise TypeError(
                f"data must be MappingProxyType, got {type(self.data).__name__}"
            )

        if not isinstance(self.micro_questions, tuple):
            raise TypeError(
                f"micro_questions must be tuple, got {type(self.micro_questions).__name__}"
            )

        # Validate all micro questions are immutable
        for i, q in enumerate(self.micro_questions):
            if not isinstance(q, MappingProxyType):
                raise TypeError(
                    f"micro_questions[{i}] must be MappingProxyType, "
                    f"got {type(q).__name__}"
                )

        logger.info(
            "canonical_questionnaire_validated",
            sha256=self.sha256[:16] + "...",
            micro_count=self.micro_question_count,
            total_count=self.total_question_count,
            version=self.version,
        )

@dataclass(frozen=True)
class ProcessorBundle:
    """Aggregated orchestrator dependencies built by the factory.

    Attributes:
        method_executor: Preconfigured :class:`MethodExecutor` instance ready for
            execution.  This object encapsulates dynamic class loading via the
            orchestrator registry.
        questionnaire: Read-only view of the questionnaire monolith payload.
            Consumers must treat this mapping as immutable.
        factory: The :class:`CoreModuleFactory` used to construct ancillary
            input contracts for downstream processors.
    """

    method_executor: MethodExecutor
    questionnaire: Mapping[str, Any]
    factory: "CoreModuleFactory"

# ============================================================================
# FILE I/O OPERATIONS
# ============================================================================

def validate_questionnaire_structure(data: dict[str, object]) -> None:
    """Validate questionnaire structure for required fields and types.
    
    Args:
        data: Questionnaire data to validate
        
    Raises:
        ValueError: If required fields are missing or invalid
        TypeError: If data is not a dictionary
    """
    if not isinstance(data, dict):
        raise ValueError("Questionnaire must be a dictionary")
    
    # Check top-level keys
    required_keys = ["version", "blocks", "schema_version"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Questionnaire missing keys: {missing}")
    
    # Validate blocks structure
    blocks = data["blocks"]  # type: ignore[index]
    if not isinstance(blocks, dict):
        raise ValueError("blocks must be a dict")
    
    if "micro_questions" not in blocks:
        raise ValueError("blocks.micro_questions is required")
    
    micro_questions = blocks["micro_questions"]
    if not isinstance(micro_questions, list):
        raise ValueError("blocks.micro_questions must be a list")
    
    # Track for duplicate detection
    seen_question_ids = set()
    seen_question_globals = set()
    
    # Validate each question
    required_q_keys = ["question_id", "question_global", "base_slot"]
    
    for i, q in enumerate(micro_questions):
        if not isinstance(q, dict):
            raise ValueError(f"Question {i} must be a dict, got {type(q).__name__}")
        
        # Check required keys
        missing_q = [k for k in required_q_keys if k not in q]
        if missing_q:
            raise ValueError(f"Question {i} missing keys: {missing_q}")
        
        # Check for None values
        for key in required_q_keys:
            if q[key] is None:
                raise ValueError(f"Question {i}: {key} cannot be None")
        
        # Type validation
        question_id = q["question_id"]
        if not isinstance(question_id, str):
            raise ValueError(
                f"Question {i}: question_id must be string, got {type(question_id).__name__}"
            )
        
        question_global = q["question_global"]
        if not isinstance(question_global, int):
            raise ValueError(
                f"Question {i}: question_global must be an integer, got {type(question_global).__name__}"
            )
        
        base_slot = q["base_slot"]
        if not isinstance(base_slot, str):
            raise ValueError(
                f"Question {i}: base_slot must be string, got {type(base_slot).__name__}"
            )
        
        # Duplicate detection
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate question_id: {question_id} at index {i}")
        seen_question_ids.add(question_id)
        
        if question_global in seen_question_globals:
            raise ValueError(
                f"Duplicate question_global: {question_global} at index {i}"
            )
        seen_question_globals.add(question_global)
    
    logger.info(
        f"questionnaire_validation_passed: {len(micro_questions)} questions validated"
    )


def load_questionnaire(path: Path | None = None) -> CanonicalQuestionnaire:
    """Load and validate questionnaire with full integrity checking.

    This is the ONLY function that should load questionnaire_monolith.json.
    It enforces:
    - File existence and readability
    - JSON validity
    - Structure validation (300 micro, 4 meso, 1 macro)
    - SHA-256 hash verification
    - Immutability (all data wrapped in MappingProxyType/tuple)
    - No duplicate question IDs

    Args:
        path: Optional path to questionnaire file.
              Defaults to data/questionnaire_monolith.json

    Returns:
        CanonicalQuestionnaire with validated, immutable data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If structure/hash validation fails
        TypeError: If data types are incorrect
    """
    if path is None:
        path = QUESTIONNAIRE_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Questionnaire file not found: {path}\n"
            f"Expected location: {QUESTIONNAIRE_PATH}"
        )

    logger.info(f"Loading questionnaire from {path}")

    # Read file content
    content = path.read_text(encoding='utf-8')

    # Parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in questionnaire file: {e.msg}",
            e.doc,
            e.pos
        ) from e

    if not isinstance(data, dict):
        raise TypeError(
            "questionnaire_monolith.json must contain a JSON object at the top level"
        )

    # Validate structure (raises on failure)
    validate_questionnaire_structure(data)

    # Compute SHA-256 hash for integrity verification
    canonical_json = json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=True,
        separators=(',', ':'),
    )
    sha256 = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    # Extract blocks
    blocks = data['blocks']
    micro_questions = blocks['micro_questions']
    meso_questions = blocks.get('meso_questions', [])
    macro_question = blocks.get('macro_question')

    # Convert to immutable structures
    micro_immutable = tuple(MappingProxyType(q) for q in micro_questions)
    meso_immutable = tuple(MappingProxyType(q) for q in meso_questions)
    macro_immutable = MappingProxyType(macro_question) if macro_question else None

    # Count total questions
    total_count = len(micro_questions) + len(meso_questions)
    if macro_question:
        total_count += 1

    # Construct CanonicalQuestionnaire (validates invariants in __post_init__)
    return CanonicalQuestionnaire(
        data=MappingProxyType(data),
        sha256=sha256,
        micro_questions=micro_immutable,
        meso_questions=meso_immutable,
        macro_question=macro_immutable,
        micro_question_count=len(micro_questions),
        total_question_count=total_count,
        version=data.get('version', 'unknown'),
        schema_version=data.get('schema_version', 'unknown'),
    )


def load_questionnaire_monolith(path: Path | None = None) -> dict[str, Any]:
    """DEPRECATED: Use load_questionnaire() instead.

    This function is maintained for backward compatibility only.
    It loads the questionnaire but returns mutable dict instead of
    CanonicalQuestionnaire.

    Args:
        path: Optional path to questionnaire file

    Returns:
        Mutable questionnaire dict (DEPRECATED)
    """
    import warnings
    warnings.warn(
        "load_questionnaire_monolith() is deprecated. "
        "Use load_questionnaire() which returns CanonicalQuestionnaire.",
        DeprecationWarning,
        stacklevel=2
    )

    canonical = load_questionnaire(path)
    # Return mutable copy for backward compatibility
    return dict(canonical.data)

def load_catalog(path: Path | None = None) -> dict[str, Any]:
    """Load method catalog JSON file.

    Args:
        path: Path to catalog file. Defaults to config/rules/METODOS/catalogo_completo_canonico.json
              relative to repository root.

    Returns:
        Loaded catalog data
    
    Raises:
        FileNotFoundError: If catalog file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if path is None:
        path = _REPO_ROOT / "config" / "rules" / "METODOS" / "catalogo_completo_canonico.json"

    logger.info(f"Loading catalog from {path}")

    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_method_map(path: Path | None = None) -> dict[str, Any]:
    """Load method-class mapping JSON file.

    Args:
        path: Path to method map file. Defaults to COMPLETE_METHOD_CLASS_MAP.json
              relative to repository root.

    Returns:
        Loaded method map data
    
    Raises:
        FileNotFoundError: If method map file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if path is None:
        path = _REPO_ROOT / "COMPLETE_METHOD_CLASS_MAP.json"

    logger.info(f"Loading method map from {path}")

    with open(path, encoding='utf-8') as f:
        return json.load(f)

def get_canonical_dimensions(questionnaire_path: Path | None = None) -> dict[str, dict[str, str]]:
    """
    Get canonical dimension definitions from questionnaire monolith.
    
    This function loads the canonical notation from questionnaire_monolith.json
    and returns the dimension definitions.
    
    Args:
        questionnaire_path: Optional path to questionnaire file
        
    Returns:
        Dictionary mapping dimension keys (D1-D6) to dimension info with code, name, label
        
    Example:
        >>> dims = get_canonical_dimensions()
        >>> dims['D1']
        {'code': 'DIM01', 'name': 'INSUMOS', 'label': 'Diagnóstico y Recursos'}
    """
    monolith = load_questionnaire_monolith(questionnaire_path)
    
    if 'canonical_notation' not in monolith:
        raise KeyError("canonical_notation section missing from questionnaire")
    
    if 'dimensions' not in monolith['canonical_notation']:
        raise KeyError("dimensions section missing from canonical_notation")
    
    return monolith['canonical_notation']['dimensions']

def get_canonical_policy_areas(questionnaire_path: Path | None = None) -> dict[str, dict[str, str]]:
    """
    Get canonical policy area definitions from questionnaire monolith.
    
    This function loads the canonical notation from questionnaire_monolith.json
    and returns the policy area definitions.
    
    Args:
        questionnaire_path: Optional path to questionnaire file
        
    Returns:
        Dictionary mapping policy area codes (PA01-PA10) to policy area info with name, legacy_id
        
    Example:
        >>> areas = get_canonical_policy_areas()
        >>> areas['PA01']
        {'name': 'Derechos de las mujeres e igualdad de género', 'legacy_id': 'P1'}
    """
    monolith = load_questionnaire_monolith(questionnaire_path)
    
    if 'canonical_notation' not in monolith:
        raise KeyError("canonical_notation section missing from questionnaire")
    
    if 'policy_areas' not in monolith['canonical_notation']:
        raise KeyError("policy_areas section missing from canonical_notation")
    
    return monolith['canonical_notation']['policy_areas']

def load_schema(path: Path | None = None) -> dict[str, Any]:
    """Load questionnaire schema JSON file.

    Args:
        path: Path to schema file. Defaults to schemas/questionnaire_monolith.schema.json
              relative to repository root.

    Returns:
        Loaded schema data
    
    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    if path is None:
        path = _REPO_ROOT / "schemas" / "questionnaire_monolith.schema.json"

    logger.info(f"Loading schema from {path}")

    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_document(file_path: Path) -> DocumentData:
    """Load a document and construct DocumentData contract.

    This handles file I/O and parsing, providing structured data to core modules.

    Args:
        file_path: Path to document file

    Returns:
        DocumentData contract with parsed content
    """
    logger.info(f"Loading document from {file_path}")

    # Read file
    with open(file_path, encoding='utf-8') as f:
        raw_text = f.read()

    # Basic parsing (to be enhanced)
    sentences = raw_text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    return DocumentData(
        raw_text=raw_text,
        sentences=sentences,
        tables=[],  # Table extraction to be implemented
        metadata={
            'file_path': str(file_path),
            'file_name': file_path.name,
            'num_sentences': len(sentences),
        }
    )

def save_results(results: dict[str, Any], output_path: Path) -> None:
    """Save analysis results to file.

    This is the ONLY place that should write analysis results.
    Core modules return data via contracts; the factory handles persistence.

    Args:
        results: Analysis results to save
        output_path: Path to output file
    """
    logger.info(f"Saving results to {output_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ============================================================================
# CONTRACT CONSTRUCTORS
# ============================================================================

def construct_semantic_analyzer_input(
    document: DocumentData,
    **kwargs: Any
) -> SemanticAnalyzerInputContract:
    """Construct input contract for SemanticAnalyzer.

    Args:
        document: Loaded document data
        **kwargs: Additional parameters

    Returns:
        Typed input contract
    """
    return SemanticAnalyzerInputContract(
        text=document['raw_text'],
        segments=kwargs.get('segments', document['sentences']),
        ontology_params=kwargs.get('ontology_params', {}),
    )

def construct_cdaf_input(
    document: DocumentData,
    plan_name: str,
    **kwargs: Any
) -> CDAFFrameworkInputContract:
    """Construct input contract for CDAFFramework.

    Args:
        document: Loaded document data
        plan_name: Name of the development plan
        **kwargs: Additional parameters

    Returns:
        Typed input contract
    """
    return CDAFFrameworkInputContract(
        document_text=document['raw_text'],
        plan_metadata={
            'plan_name': plan_name,
            **document['metadata'],
            **kwargs.get('plan_metadata', {}),
        },
        config=kwargs.get('config', {}),
    )

def construct_pdet_input(
    document: DocumentData,
    **kwargs: Any
) -> PDETAnalyzerInputContract:
    """Construct input contract for PDETMunicipalPlanAnalyzer.

    Args:
        document: Loaded document data
        **kwargs: Additional parameters

    Returns:
        Typed input contract
    """
    return PDETAnalyzerInputContract(
        document_content=document['raw_text'],
        extract_tables=kwargs.get('extract_tables', True),
        config=kwargs.get('config', {}),
    )

def construct_teoria_cambio_input(
    document: DocumentData,
    **kwargs: Any
) -> TeoriaCambioInputContract:
    """Construct input contract for TeoriaCambio.

    Args:
        document: Loaded document data
        **kwargs: Additional parameters

    Returns:
        Typed input contract
    """
    return TeoriaCambioInputContract(
        document_text=document['raw_text'],
        strategic_goals=kwargs.get('strategic_goals', []),
        config=kwargs.get('config', {}),
    )

def construct_contradiction_detector_input(
    document: DocumentData,
    plan_name: str,
    **kwargs: Any
) -> ContradictionDetectorInputContract:
    """Construct input contract for PolicyContradictionDetector.

    Args:
        document: Loaded document data
        plan_name: Name of the development plan
        **kwargs: Additional parameters

    Returns:
        Typed input contract
    """
    return ContradictionDetectorInputContract(
        text=document['raw_text'],
        plan_name=plan_name,
        dimension=kwargs.get('dimension'),
        config=kwargs.get('config', {}),
    )

def construct_embedding_policy_input(
    document: DocumentData,
    **kwargs: Any
) -> EmbeddingPolicyInputContract:
    """Construct input contract for embedding policy analysis.

    Args:
        document: Loaded document data
        **kwargs: Additional parameters

    Returns:
        Typed input contract
    """
    return EmbeddingPolicyInputContract(
        text=document['raw_text'],
        dimensions=kwargs.get('dimensions', []),
        model_config=kwargs.get('model_config', {}),
    )

def construct_semantic_chunking_input(
    document: DocumentData,
    **kwargs: Any
) -> SemanticChunkingInputContract:
    """Construct input contract for semantic chunking.

    Args:
        document: Loaded document data
        **kwargs: Additional parameters

    Returns:
        Typed input contract
    """
    return SemanticChunkingInputContract(
        text=document['raw_text'],
        preserve_structure=kwargs.get('preserve_structure', True),
        config=kwargs.get('config', {}),
    )

def construct_policy_processor_input(
    document: DocumentData,
    **kwargs: Any
) -> PolicyProcessorInputContract:
    """Construct input contract for IndustrialPolicyProcessor.

    Args:
        document: Loaded document data
        **kwargs: Additional parameters

    Returns:
        Typed input contract
    """
    return PolicyProcessorInputContract(
        data=kwargs.get('data', document['raw_text']),
        text=document['raw_text'],
        sentences=document['sentences'],
        tables=document['tables'],
        config=kwargs.get('config', {}),
    )

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

class CoreModuleFactory:
    """Factory for constructing core modules with injected dependencies.

    This factory:
    1. Loads data from disk
    2. Constructs contracts
    3. Initializes core modules
    4. Manages all I/O operations

    Usage:
        factory = CoreModuleFactory()
        document = factory.load_document(Path("plan.txt"))

        # Construct input contract
        input_contract = factory.construct_semantic_analyzer_input(document)

        # Use with core module (once modules are refactored)
        # analyzer = SemanticAnalyzer()
        # result = analyzer.analyze(input_contract)
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize factory.

        Args:
            data_dir: Optional directory for data files
        """
        self.data_dir = data_dir or _DEFAULT_DATA_DIR
        self.questionnaire_cache: dict[str, Any] | None = None
        self.catalog_cache: dict[str, Any] | None = None

    def get_questionnaire(self) -> dict[str, Any]:
        """Get questionnaire monolith data (cached).

        Uses canonical loader for hash verification.

        Returns:
            Questionnaire data (dict for backward compatibility)
        """
        if self.questionnaire_cache is None:
            questionnaire_path = self.data_dir / "questionnaire_monolith.json"
            # Use canonical loader for hash verification
            canonical_q = load_questionnaire(questionnaire_path)
            self.questionnaire_cache = dict(canonical_q.data)
            # Also set it in the global provider for backward compatibility
            get_questionnaire_provider().set_data(self.questionnaire_cache)
            logger.info(
                "factory_loaded_questionnaire",
                sha256=canonical_q.sha256[:16] + "...",
                question_count=canonical_q.total_question_count,
            )
        return self.questionnaire_cache

    @property
    def catalog(self) -> dict[str, Any]:
        """Get method catalog data (cached).

        Returns:
            Method catalog data
        """
        if self.catalog_cache is None:
            self.catalog_cache = load_catalog()
        return self.catalog_cache

    def load_document(self, file_path: Path) -> DocumentData:
        """Load document and return structured data.

        Args:
            file_path: Path to document

        Returns:
            Parsed document data
        """
        return load_document(file_path)

    def save_results(self, results: dict[str, Any], output_path: Path) -> None:
        """Save analysis results.

        Args:
            results: Results to save
            output_path: Output file path
        """
        save_results(results, output_path)

    def load_catalog(self, path: Path | None = None) -> dict[str, Any]:
        """Load method catalog JSON file.

        Args:
            path: Path to catalog file. Defaults to config/rules/METODOS/catalogo_completo_canonico.json
                  relative to repository root.

        Returns:
            Loaded catalog data
        """
        return load_catalog(path)

    # Contract constructor methods
    construct_semantic_analyzer_input = construct_semantic_analyzer_input
    construct_cdaf_input = construct_cdaf_input
    construct_pdet_input = construct_pdet_input
    construct_teoria_cambio_input = construct_teoria_cambio_input
    construct_contradiction_detector_input = construct_contradiction_detector_input
    construct_embedding_policy_input = construct_embedding_policy_input
    construct_semantic_chunking_input = construct_semantic_chunking_input
    construct_policy_processor_input = construct_policy_processor_input

def build_processor(
    *,
    questionnaire_path: Path | None = None,
    data_dir: Path | None = None,
    factory: Optional["CoreModuleFactory"] = None,
    enable_signals: bool = True,
) -> ProcessorBundle:
    """Create a processor bundle with orchestrator dependencies wired together.

    Args:
        questionnaire_path: Optional path to the questionnaire monolith. When
            provided, it overrides the factory's default resolution logic.
        data_dir: Optional directory for ancillary data files such as the
            questionnaire. Useful for tests that operate inside temporary
            directories.
        factory: Pre-existing :class:`CoreModuleFactory` instance. When omitted
            the function creates a new factory configured with ``data_dir``.
        enable_signals: Enable signal infrastructure (default: True)

    Returns:
        A :class:`ProcessorBundle` containing a ready-to-use method executor,
        the questionnaire payload (as an immutable mapping) and the factory.

    Note:
        Uses load_questionnaire() for hash verification and immutability.
    """

    core_factory = factory or CoreModuleFactory(data_dir=data_dir)

    if questionnaire_path is not None:
        # Use canonical loader for hash verification
        canonical_q = load_questionnaire(questionnaire_path)
        questionnaire_data = dict(canonical_q.data)  # Convert for backward compat
        core_factory.questionnaire_cache = copy.deepcopy(questionnaire_data)
        # Initialize the global provider with this data
        get_questionnaire_provider().set_data(questionnaire_data)
        logger.info(
            "build_processor_using_canonical_loader",
            path=str(questionnaire_path),
            sha256=canonical_q.sha256[:16] + "...",
            question_count=canonical_q.total_question_count,
        )
    else:
        questionnaire_data = core_factory.get_questionnaire()

    questionnaire_snapshot = MappingProxyType(copy.deepcopy(questionnaire_data))

    # Build signal infrastructure if enabled
    signal_registry = None
    if enable_signals:
        try:
            from .core_module_factory import CoreModuleFactory as SignalFactory
            
            # Create signal-enabled factory
            signal_factory = SignalFactory(
                questionnaire_data=questionnaire_data,
                enable_signals=True,
            )
            signal_registry = signal_factory._signal_registry
            
            logger.info(
                "signals_enabled_in_processor",
                enabled=True,
                registry_size=len(signal_registry._cache) if signal_registry else 0,
            )
        except Exception as e:
            logger.warning(
                "signal_initialization_failed",
                error=str(e),
                fallback="continuing without signals"
            )
            signal_registry = None

    executor = MethodExecutor(signal_registry=signal_registry)

    return ProcessorBundle(
        method_executor=executor,
        questionnaire=questionnaire_snapshot,
        factory=core_factory,
    )

# ============================================================================
# HASH AND VALIDATION UTILITIES
# ============================================================================

def compute_monolith_hash(monolith: dict[str, Any]) -> str:
    """
    Compute deterministic SHA-256 hash of questionnaire monolith.
    
    This function ensures:
    - Key order independence via sort_keys=True
    - Consistent unicode handling via ensure_ascii=True
    - No whitespace variation via separators
    
    Args:
        monolith: Questionnaire monolith dictionary
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    import hashlib
    
    serialized = json.dumps(
        monolith,
        sort_keys=True,
        ensure_ascii=True,  # Consistent unicode handling
        separators=(',', ':'),  # No whitespace
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def validate_questionnaire_structure(data: dict[str, Any]) -> None:
    """
    Validate questionnaire has required structure and types.
    
    Performs comprehensive validation including:
    - Top-level structure (version, blocks, schema_version)
    - Block structure (micro_questions must be list)
    - Question fields (question_id, question_global, base_slot)
    - Type validation for all fields
    - Duplicate detection (question_id, question_global)
    - Null value checking
    
    Args:
        data: Questionnaire data to validate
        
    Raises:
        ValueError: If validation fails with specific error message
        TypeError: If top-level structure is invalid
    """
    if not isinstance(data, dict):
        raise TypeError("Questionnaire must be a dictionary")

    # Check top-level keys
    required_keys = ['version', 'blocks', 'schema_version']
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Questionnaire missing keys: {missing}")

    # Validate blocks structure
    blocks = data['blocks']
    if not isinstance(blocks, dict):
        raise ValueError("blocks must be a dict")

    if 'micro_questions' not in blocks:
        raise ValueError("blocks.micro_questions is required")

    micro_questions = blocks['micro_questions']
    if not isinstance(micro_questions, list):
        raise ValueError("blocks.micro_questions must be a list")
    
    # Enforce minimum: at least 1 question required
    if len(micro_questions) < 1:
        raise ValueError(
            "Questionnaire must have at least 1 micro question, got 0. "
            "Cannot proceed with empty questionnaire."
        )

    # Track for duplicate detection
    seen_question_ids = set()
    seen_question_globals = set()

    # Validate each question
    required_q_keys = ['question_id', 'question_global', 'base_slot']

    for i, q in enumerate(micro_questions):
        if not isinstance(q, dict):
            raise ValueError(f"Question {i} must be a dict, got {type(q).__name__}")
        
        # Check required keys
        missing_q = [k for k in required_q_keys if k not in q]
        if missing_q:
            raise ValueError(f"Question {i} missing keys: {missing_q}")
        
        # Check for None values
        for key in required_q_keys:
            if q[key] is None:
                raise ValueError(f"Question {i}: {key} cannot be None")
        
        # Type validation
        question_id = q['question_id']
        if not isinstance(question_id, str):
            raise ValueError(
                f"Question {i}: question_id must be string, got {type(question_id).__name__}"
            )
        
        question_global = q['question_global']
        if not isinstance(question_global, int):
            raise ValueError(
                f"Question {i}: question_global must be an integer, got {type(question_global).__name__}"
            )
        
        base_slot = q['base_slot']
        if not isinstance(base_slot, str):
            raise ValueError(
                f"Question {i}: base_slot must be string, got {type(base_slot).__name__}"
            )
        
        # Duplicate detection
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate question_id: {question_id} at index {i}")
        seen_question_ids.add(question_id)
        
        if question_global in seen_question_globals:
            raise ValueError(
                f"Duplicate question_global: {question_global} at index {i}"
            )
        seen_question_globals.add(question_global)

    logger.info(
        "questionnaire_validation_passed",
        extra={
            "question_count": len(micro_questions),
            "unique_question_ids": len(seen_question_ids),
        }
    )

# ============================================================================
# MIGRATION HELPERS
# ============================================================================

def migrate_io_from_module(module_name: str, line_numbers: list[int]) -> None:
    """Helper to track I/O migration progress.

    This is a placeholder function to document which I/O operations
    have been migrated from core modules to the factory.

    Args:
        module_name: Name of the module being migrated
        line_numbers: Line numbers of I/O operations migrated
    """
    logger.info(
        f"Migrating {len(line_numbers)} I/O operations from {module_name}: "
        f"lines {line_numbers}"
    )

# TODO: Migrate I/O operations from core modules
# Track progress:
# - Analyzer_one.py: 72 I/O operations to migrate
# - dereck_beach.py: 40 I/O operations to migrate
# - financiero_viabilidad_tablas.py: Multiple operations to migrate
# - teoria_cambio.py: Some operations to migrate
# Others are clean

__all__ = [
    # Questionnaire integrity types and constants
    'CanonicalQuestionnaire',
    'EXPECTED_HASH',
    'EXPECTED_MICRO_QUESTION_COUNT',
    'EXPECTED_TOTAL_QUESTION_COUNT',
    'QUESTIONNAIRE_PATH',
    # Canonical loader (use this!)
    'load_questionnaire',
    # Factory classes
    'CoreModuleFactory',
    'ProcessorBundle',
    # Legacy/deprecated (use load_questionnaire instead)
    'load_questionnaire_monolith',
    # Validation
    'validate_questionnaire_structure',
    'compute_monolith_hash',
    # Other loaders
    'load_catalog',
    'load_method_map',
    'get_canonical_dimensions',
    'get_canonical_policy_areas',
    'load_schema',
    'load_document',
    'save_results',
    # Contract constructors
    'construct_semantic_analyzer_input',
    'construct_cdaf_input',
    'construct_pdet_input',
    'construct_teoria_cambio_input',
    'construct_contradiction_detector_input',
    'construct_embedding_policy_input',
    'construct_semantic_chunking_input',
    'construct_policy_processor_input',
    # Builder
    'build_processor',
]
