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

Version: 1.0.0
Status: Skeleton implementation (to be expanded with I/O migration)
"""

import copy
import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Optional

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


@dataclass(frozen=True)
class ContractManifest:
    """Cryptographic proof of factory-executor contract alignment.

    This manifest provides bidirectional hash-based verification that ensures
    factory-produced components are compatible with executors. Uses BLAKE3
    for high-performance, cryptographically-secure fingerprinting.

    Attributes:
        factory_version: Semantic version of the factory
        questionnaire_hash: BLAKE3 hash of questionnaire structure
        catalog_hash: BLAKE3 hash of method catalog
        method_map_hash: BLAKE3 hash of method-class mapping
        contract_schemas_hash: BLAKE3 hash of all contract type definitions
        manifest_hash: Self-referential hash of this manifest (computed)
        created_at: ISO timestamp of manifest creation
    """

    factory_version: str
    questionnaire_hash: str
    catalog_hash: str
    method_map_hash: str
    contract_schemas_hash: str
    created_at: str
    manifest_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute self-referential manifest hash."""
        # Use object.__setattr__ since dataclass is frozen
        manifest_data = {
            'factory_version': self.factory_version,
            'questionnaire_hash': self.questionnaire_hash,
            'catalog_hash': self.catalog_hash,
            'method_map_hash': self.method_map_hash,
            'contract_schemas_hash': self.contract_schemas_hash,
            'created_at': self.created_at,
        }
        manifest_json = json.dumps(manifest_data, sort_keys=True, separators=(',', ':'))
        manifest_hash = hashlib.blake2b(manifest_json.encode('utf-8'), digest_size=32).hexdigest()
        object.__setattr__(self, 'manifest_hash', manifest_hash)

    def verify_compatibility(self, executor_manifest: 'ContractManifest') -> tuple[bool, str]:
        """Verify bidirectional compatibility with executor manifest.

        Args:
            executor_manifest: Manifest from executor requiring compatibility

        Returns:
            Tuple of (is_compatible, reason). If incompatible, reason explains mismatch.
        """
        if self.questionnaire_hash != executor_manifest.questionnaire_hash:
            return False, f"Questionnaire mismatch: factory={self.questionnaire_hash[:8]}... vs executor={executor_manifest.questionnaire_hash[:8]}..."

        if self.catalog_hash != executor_manifest.catalog_hash:
            return False, f"Catalog mismatch: factory={self.catalog_hash[:8]}... vs executor={executor_manifest.catalog_hash[:8]}..."

        if self.method_map_hash != executor_manifest.method_map_hash:
            return False, f"Method map mismatch: factory={self.method_map_hash[:8]}... vs executor={executor_manifest.method_map_hash[:8]}..."

        if self.contract_schemas_hash != executor_manifest.contract_schemas_hash:
            return False, f"Contract schemas mismatch: factory={self.contract_schemas_hash[:8]}... vs executor={executor_manifest.contract_schemas_hash[:8]}..."

        return True, "All contract hashes match - compatible"


@dataclass(frozen=True)
class ImmutableExecutionContext:
    """Immutable execution context with copy-on-write semantics.

    This is Intervention #4: Immutable Execution Context.
    Prevents accidental mutation bugs by using frozen dataclass + structural sharing.
    Each modification creates a new context version, enabling perfect auditability
    and thread-safety by construction.

    Attributes:
        phase_id: Current execution phase identifier
        phase_name: Human-readable phase name
        document_id: Identifier for the document being processed
        method_sequence: Tuple of (class_name, method_name) pairs (immutable)
        arguments: Frozen mapping of execution arguments
        metadata: Frozen mapping of execution metadata
        parent_context_hash: Hash of parent context (for audit trail)
        context_version: Monotonically increasing version number
        created_at: ISO timestamp of context creation

    Example:
        >>> ctx1 = ImmutableExecutionContext.create(phase_id=1, phase_name="init")
        >>> ctx2 = ctx1.with_arguments({'doc': 'test'})  # Creates new context
        >>> ctx1.arguments  # Original unchanged
        MappingProxyType({})
        >>> ctx2.arguments  # New context has arguments
        MappingProxyType({'doc': 'test'})
    """

    phase_id: int
    phase_name: str
    document_id: str
    method_sequence: tuple[tuple[str, str], ...]
    arguments: Mapping[str, Any]
    metadata: Mapping[str, Any]
    parent_context_hash: str
    context_version: int
    created_at: str

    @classmethod
    def create(
        cls,
        phase_id: int,
        phase_name: str,
        document_id: str = "",
        method_sequence: list[tuple[str, str]] | None = None,
        arguments: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> 'ImmutableExecutionContext':
        """Create a new root execution context.

        Args:
            phase_id: Phase identifier
            phase_name: Phase name
            document_id: Document identifier
            method_sequence: List of (class, method) tuples
            arguments: Execution arguments
            metadata: Execution metadata

        Returns:
            New immutable context
        """
        from datetime import datetime

        method_seq = tuple(method_sequence) if method_sequence else ()
        args = MappingProxyType(arguments or {})
        meta = MappingProxyType(metadata or {})

        # Compute hash of empty parent
        parent_hash = "root"

        return cls(
            phase_id=phase_id,
            phase_name=phase_name,
            document_id=document_id,
            method_sequence=method_seq,
            arguments=args,
            metadata=meta,
            parent_context_hash=parent_hash,
            context_version=1,
            created_at=datetime.utcnow().isoformat(),
        )

    def _compute_hash(self) -> str:
        """Compute hash of current context for audit trail."""
        ctx_data = {
            'phase_id': self.phase_id,
            'phase_name': self.phase_name,
            'document_id': self.document_id,
            'method_sequence': self.method_sequence,
            'arguments': dict(self.arguments),
            'metadata': dict(self.metadata),
            'context_version': self.context_version,
        }
        return compute_blake3_hash(ctx_data)

    def with_arguments(self, new_arguments: dict[str, Any]) -> 'ImmutableExecutionContext':
        """Create new context with updated arguments (copy-on-write).

        Args:
            new_arguments: New arguments to add/update

        Returns:
            New context with merged arguments
        """
        from datetime import datetime

        # Merge arguments (structural sharing - original dict unchanged)
        merged_args = {**dict(self.arguments), **new_arguments}

        return ImmutableExecutionContext(
            phase_id=self.phase_id,
            phase_name=self.phase_name,
            document_id=self.document_id,
            method_sequence=self.method_sequence,
            arguments=MappingProxyType(merged_args),
            metadata=self.metadata,
            parent_context_hash=self._compute_hash(),
            context_version=self.context_version + 1,
            created_at=datetime.utcnow().isoformat(),
        )

    def with_metadata(self, new_metadata: dict[str, Any]) -> 'ImmutableExecutionContext':
        """Create new context with updated metadata (copy-on-write).

        Args:
            new_metadata: New metadata to add/update

        Returns:
            New context with merged metadata
        """
        from datetime import datetime

        merged_meta = {**dict(self.metadata), **new_metadata}

        return ImmutableExecutionContext(
            phase_id=self.phase_id,
            phase_name=self.phase_name,
            document_id=self.document_id,
            method_sequence=self.method_sequence,
            arguments=self.arguments,
            metadata=MappingProxyType(merged_meta),
            parent_context_hash=self._compute_hash(),
            context_version=self.context_version + 1,
            created_at=datetime.utcnow().isoformat(),
        )

    def with_phase(self, phase_id: int, phase_name: str) -> 'ImmutableExecutionContext':
        """Create new context for different phase (copy-on-write).

        Args:
            phase_id: New phase ID
            phase_name: New phase name

        Returns:
            New context for next phase
        """
        from datetime import datetime

        return ImmutableExecutionContext(
            phase_id=phase_id,
            phase_name=phase_name,
            document_id=self.document_id,
            method_sequence=self.method_sequence,
            arguments=self.arguments,
            metadata=self.metadata,
            parent_context_hash=self._compute_hash(),
            context_version=self.context_version + 1,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization.

        Returns:
            Dictionary representation
        """
        return {
            'phase_id': self.phase_id,
            'phase_name': self.phase_name,
            'document_id': self.document_id,
            'method_sequence': list(self.method_sequence),
            'arguments': dict(self.arguments),
            'metadata': dict(self.metadata),
            'parent_context_hash': self.parent_context_hash,
            'context_version': self.context_version,
            'created_at': self.created_at,
        }


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
        contract_manifest: Cryptographic proof of contract alignment between
            factory and executors. Executors MUST verify compatibility before use.
    """

    method_executor: MethodExecutor
    questionnaire: Mapping[str, Any]
    factory: "CoreModuleFactory"
    contract_manifest: ContractManifest

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


def load_questionnaire_monolith(path: Path | None = None) -> dict[str, Any]:
    """Load questionnaire monolith JSON file.

    This is the ONLY place in the system that should read questionnaire_monolith.json.
    Core modules receive the data via contracts.

    Args:
        path: Optional path to questionnaire file. Defaults to ./questionnaire_monolith.json

    Returns:
        Loaded questionnaire data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If questionnaire structure is invalid
    """
    if path is None:
        path = _DEFAULT_DATA_DIR / "questionnaire_monolith.json"

    logger.info(f"Loading questionnaire from {path}")

    with open(path, encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(
            "questionnaire_monolith.json must contain a JSON object at the top level"
        )
    
    # Validate structure before returning
    validate_questionnaire_structure(payload)

    return payload

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
    5. Creates executors with pre-wired dependencies (Intervention #3)

    Usage:
        factory = CoreModuleFactory()
        document = factory.load_document(Path("plan.txt"))

        # Construct input contract
        input_contract = factory.construct_semantic_analyzer_input(document)

        # Create executor with all dependencies (NEW)
        executor = factory.create_executor('D1Q1_Executor', processor_bundle)
    """

    # Class-level executor registry for automatic discovery
    _executor_registry: dict[str, type] = {}

    def __init__(self, data_dir: Path | None = None) -> None:
        """Initialize factory.

        Args:
            data_dir: Optional directory for data files
        """
        self.data_dir = data_dir or _DEFAULT_DATA_DIR
        self.questionnaire_cache: dict[str, Any] | None = None
        self.catalog_cache: dict[str, Any] | None = None

    @classmethod
    def register_executor(cls, name: str, executor_class: type) -> None:
        """Register an executor class for lazy loading.

        Args:
            name: Name of the executor (e.g., 'D1Q1_Executor')
            executor_class: The executor class to register
        """
        cls._executor_registry[name] = executor_class
        logger.debug(f"Registered executor: {name}")

    @classmethod
    def get_registered_executors(cls) -> list[str]:
        """Get list of all registered executor names.

        Returns:
            List of executor names
        """
        return list(cls._executor_registry.keys())

    def create_executor(
        self,
        executor_name: str,
        processor_bundle: ProcessorBundle,
        config: Any | None = None,
        calibration_orchestrator: Any | None = None,
    ) -> Any:
        """Create an executor with pre-wired dependencies.

        This is Intervention #3: Lazy-Loading Executor Factory with Fail-Fast Constructor.
        All dependencies are validated and wired at construction time, not execution time.

        Args:
            executor_name: Name of executor class (e.g., 'D1Q1_Executor')
            processor_bundle: Bundle with method_executor and signal_registry
            config: Optional ExecutorConfig instance
            calibration_orchestrator: Optional calibration orchestrator

        Returns:
            Initialized executor instance with all dependencies wired

        Raises:
            ValueError: If executor_name not registered
            TypeError: If processor_bundle lacks required dependencies
            RuntimeError: If executor construction fails validation

        Example:
            >>> bundle = build_processor()
            >>> executor = factory.create_executor('D1Q1_Executor', bundle)
            >>> # Executor is guaranteed to have all dependencies and be ready to use
        """
        # ====================================================================
        # INTERVENTION #3: Fail-Fast Validation
        # ====================================================================

        # Validate executor exists in registry
        if executor_name not in self._executor_registry:
            # Lazy load from executors module if not registered
            try:
                from .executors import (
                    D1Q1_Executor, D1Q2_Executor, D1Q3_Executor, D1Q4_Executor, D1Q5_Executor,
                    D2Q1_Executor, D2Q2_Executor, D2Q3_Executor, D2Q4_Executor, D2Q5_Executor,
                    D3Q1_Executor, D3Q2_Executor, D3Q3_Executor, D3Q4_Executor, D3Q5_Executor,
                    D4Q1_Executor, D4Q2_Executor, D4Q3_Executor, D4Q4_Executor, D4Q5_Executor,
                    D5Q1_Executor, D5Q2_Executor, D5Q3_Executor, D5Q4_Executor, D5Q5_Executor,
                    D6Q1_Executor, D6Q2_Executor, D6Q3_Executor, D6Q4_Executor, D6Q5_Executor,
                )

                # Auto-register all executors
                executor_classes = {
                    'D1Q1_Executor': D1Q1_Executor, 'D1Q2_Executor': D1Q2_Executor,
                    'D1Q3_Executor': D1Q3_Executor, 'D1Q4_Executor': D1Q4_Executor,
                    'D1Q5_Executor': D1Q5_Executor, 'D2Q1_Executor': D2Q1_Executor,
                    'D2Q2_Executor': D2Q2_Executor, 'D2Q3_Executor': D2Q3_Executor,
                    'D2Q4_Executor': D2Q4_Executor, 'D2Q5_Executor': D2Q5_Executor,
                    'D3Q1_Executor': D3Q1_Executor, 'D3Q2_Executor': D3Q2_Executor,
                    'D3Q3_Executor': D3Q3_Executor, 'D3Q4_Executor': D3Q4_Executor,
                    'D3Q5_Executor': D3Q5_Executor, 'D4Q1_Executor': D4Q1_Executor,
                    'D4Q2_Executor': D4Q2_Executor, 'D4Q3_Executor': D4Q3_Executor,
                    'D4Q4_Executor': D4Q4_Executor, 'D4Q5_Executor': D4Q5_Executor,
                    'D5Q1_Executor': D5Q1_Executor, 'D5Q2_Executor': D5Q2_Executor,
                    'D5Q3_Executor': D5Q3_Executor, 'D5Q4_Executor': D5Q4_Executor,
                    'D5Q5_Executor': D5Q5_Executor, 'D6Q1_Executor': D6Q1_Executor,
                    'D6Q2_Executor': D6Q2_Executor, 'D6Q3_Executor': D6Q3_Executor,
                    'D6Q4_Executor': D6Q4_Executor, 'D6Q5_Executor': D6Q5_Executor,
                }

                for name, cls in executor_classes.items():
                    self.register_executor(name, cls)

                logger.info(f"Auto-registered {len(executor_classes)} executors")

            except ImportError as e:
                raise ValueError(
                    f"Executor '{executor_name}' not registered and auto-registration failed: {e}"
                ) from e

        if executor_name not in self._executor_registry:
            available = ', '.join(self.get_registered_executors())
            raise ValueError(
                f"Executor '{executor_name}' not found in registry. "
                f"Available: {available}"
            )

        # Validate ProcessorBundle has required attributes
        if not hasattr(processor_bundle, 'method_executor'):
            raise TypeError(
                "ProcessorBundle must have 'method_executor' attribute. "
                "Did you pass a valid ProcessorBundle from build_processor()?"
            )

        # Extract dependencies from bundle
        method_executor = processor_bundle.method_executor
        signal_registry = getattr(method_executor, 'signal_registry', None) if hasattr(method_executor, 'signal_registry') else None

        # Get executor class
        executor_class = self._executor_registry[executor_name]

        # Construct executor with fail-fast validation
        try:
            executor = executor_class(
                method_executor=method_executor,
                signal_registry=signal_registry,
                config=config,
                calibration_orchestrator=calibration_orchestrator,
            )

            logger.info(
                "executor_created_successfully",
                extra={
                    "executor_name": executor_name,
                    "has_signal_registry": signal_registry is not None,
                    "has_config": config is not None,
                    "has_calibration": calibration_orchestrator is not None,
                }
            )

            return executor

        except Exception as e:
            logger.error(
                "executor_construction_failed",
                extra={
                    "executor_name": executor_name,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                exc_info=True
            )
            raise RuntimeError(
                f"Failed to construct executor '{executor_name}': {e}"
            ) from e

    def get_questionnaire(self) -> dict[str, Any]:
        """Get questionnaire monolith data (cached).

        Returns:
            Questionnaire data
        """
        if self.questionnaire_cache is None:
            questionnaire_path = self.data_dir / "questionnaire_monolith.json"
            self.questionnaire_cache = load_questionnaire_monolith(questionnaire_path)
            # Also set it in the global provider for backward compatibility
            get_questionnaire_provider().set_data(self.questionnaire_cache)
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
        the questionnaire payload (as an immutable mapping), the factory,
        and a cryptographic contract manifest for executor verification.
    """
    from datetime import datetime

    core_factory = factory or CoreModuleFactory(data_dir=data_dir)

    if questionnaire_path is not None:
        questionnaire_data = load_questionnaire_monolith(questionnaire_path)
        core_factory.questionnaire_cache = copy.deepcopy(questionnaire_data)
        # Initialize the global provider with this data
        get_questionnaire_provider().set_data(questionnaire_data)
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

    # ========================================================================
    # INTERVENTION #2: Bidirectional Contract Hash Protocol
    # ========================================================================
    # Create cryptographic manifest for factory-executor alignment verification

    catalog_data = core_factory.catalog

    # Load method map for hashing
    try:
        method_map_data = load_method_map()
    except Exception as e:
        logger.warning(f"Could not load method map for manifest: {e}")
        method_map_data = {}

    # Compute component hashes
    questionnaire_hash = compute_blake3_hash(questionnaire_data)
    catalog_hash = compute_blake3_hash(catalog_data)
    method_map_hash = compute_blake3_hash(method_map_data)
    contract_schemas_hash = compute_contract_schemas_hash()

    # Create contract manifest
    manifest = ContractManifest(
        factory_version="1.0.0",  # TODO: Extract from package metadata
        questionnaire_hash=questionnaire_hash,
        catalog_hash=catalog_hash,
        method_map_hash=method_map_hash,
        contract_schemas_hash=contract_schemas_hash,
        created_at=datetime.utcnow().isoformat(),
    )

    logger.info(
        "contract_manifest_created",
        extra={
            "manifest_hash": manifest.manifest_hash,
            "questionnaire_hash": questionnaire_hash[:16] + "...",
            "catalog_hash": catalog_hash[:16] + "...",
            "contract_schemas_hash": contract_schemas_hash[:16] + "...",
        }
    )

    return ProcessorBundle(
        method_executor=executor,
        questionnaire=questionnaire_snapshot,
        factory=core_factory,
        contract_manifest=manifest,
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
    serialized = json.dumps(
        monolith,
        sort_keys=True,
        ensure_ascii=True,  # Consistent unicode handling
        separators=(',', ':'),  # No whitespace
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def compute_blake3_hash(data: dict[str, Any] | str) -> str:
    """
    Compute BLAKE3 hash for contract verification.

    BLAKE3 is faster than SHA-256 and provides better security guarantees.
    Used for contract manifest fingerprinting.

    Args:
        data: Dictionary or string to hash

    Returns:
        Hexadecimal BLAKE3 hash string (64 chars, 32 bytes)
    """
    if isinstance(data, dict):
        serialized = json.dumps(data, sort_keys=True, separators=(',', ':'))
    else:
        serialized = data

    # Use blake2b as BLAKE3 equivalent (Python stdlib doesn't have blake3)
    # digest_size=32 for 256-bit output matching BLAKE3 default
    return hashlib.blake2b(serialized.encode('utf-8'), digest_size=32).hexdigest()


def compute_contract_schemas_hash() -> str:
    """
    Compute hash of all contract type definitions.

    This creates a fingerprint of the contract interface that both factory
    and executors must agree on. Changes to contract structure will change
    this hash, preventing incompatible components from being used together.

    Returns:
        BLAKE3 hash of contract schemas
    """
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

    # Get contract class names and their annotations as proxy for schema
    contracts = [
        CDAFFrameworkInputContract,
        ContradictionDetectorInputContract,
        DocumentData,
        EmbeddingPolicyInputContract,
        PDETAnalyzerInputContract,
        PolicyProcessorInputContract,
        SemanticAnalyzerInputContract,
        SemanticChunkingInputContract,
        TeoriaCambioInputContract,
    ]

    schema_repr = {}
    for contract in contracts:
        if hasattr(contract, '__annotations__'):
            schema_repr[contract.__name__] = {
                k: str(v) for k, v in contract.__annotations__.items()
            }
        else:
            schema_repr[contract.__name__] = "TypedDict"

    return compute_blake3_hash(schema_repr)


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
    'CoreModuleFactory',
    'ProcessorBundle',
    'ContractManifest',
    'ImmutableExecutionContext',
    'load_questionnaire_monolith',
    'validate_questionnaire_structure',
    'load_catalog',
    'load_method_map',
    'get_canonical_dimensions',
    'get_canonical_policy_areas',
    'load_schema',
    'load_document',
    'save_results',
    'construct_semantic_analyzer_input',
    'construct_cdaf_input',
    'construct_pdet_input',
    'construct_teoria_cambio_input',
    'construct_contradiction_detector_input',
    'construct_embedding_policy_input',
    'construct_semantic_chunking_input',
    'construct_policy_processor_input',
    'build_processor',
    'compute_monolith_hash',
    'compute_blake3_hash',
    'compute_contract_schemas_hash',
    'validate_questionnaire_structure',
]
