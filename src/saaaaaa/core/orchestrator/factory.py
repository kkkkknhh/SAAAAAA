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

from dataclasses import dataclass
import copy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional
import json
import logging

from ..contracts import (
    DocumentData,
    SemanticAnalyzerInputContract,
    CDAFFrameworkInputContract,
    PDETAnalyzerInputContract,
    TeoriaCambioInputContract,
    ContradictionDetectorInputContract,
    EmbeddingPolicyInputContract,
    SemanticChunkingInputContract,
    PolicyProcessorInputContract,
)

from .core import MethodExecutor

logger = logging.getLogger(__name__)


_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[4] / "data"


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

def load_questionnaire_monolith(path: Optional[Path] = None) -> Dict[str, Any]:
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
    """
    if path is None:
        path = _DEFAULT_DATA_DIR / "questionnaire_monolith.json"
    
    logger.info(f"Loading questionnaire from {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise TypeError(
            "questionnaire_monolith.json must contain a JSON object at the top level"
        )

    return payload


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
    with open(file_path, 'r', encoding='utf-8') as f:
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


def save_results(results: Dict[str, Any], output_path: Path) -> None:
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
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize factory.
        
        Args:
            data_dir: Optional directory for data files
        """
        self.data_dir = data_dir or _DEFAULT_DATA_DIR
        self.questionnaire_cache: Optional[Dict[str, Any]] = None
    
    def get_questionnaire(self) -> Dict[str, Any]:
        """Get questionnaire monolith data (cached).
        
        Returns:
            Questionnaire data
        """
        if self.questionnaire_cache is None:
            questionnaire_path = self.data_dir / "questionnaire_monolith.json"
            self.questionnaire_cache = load_questionnaire_monolith(questionnaire_path)
        return self.questionnaire_cache
    
    def load_document(self, file_path: Path) -> DocumentData:
        """Load document and return structured data.
        
        Args:
            file_path: Path to document
            
        Returns:
            Parsed document data
        """
        return load_document(file_path)

    def save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save analysis results.
        
        Args:
            results: Results to save
            output_path: Output file path
        """
        save_results(results, output_path)
    
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
    questionnaire_path: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    factory: Optional["CoreModuleFactory"] = None,
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

    Returns:
        A :class:`ProcessorBundle` containing a ready-to-use method executor,
        the questionnaire payload (as an immutable mapping) and the factory.
    """

    core_factory = factory or CoreModuleFactory(data_dir=data_dir)

    if questionnaire_path is not None:
        questionnaire_data = load_questionnaire_monolith(questionnaire_path)
        core_factory.questionnaire_cache = copy.deepcopy(questionnaire_data)
    else:
        questionnaire_data = core_factory.get_questionnaire()

    questionnaire_snapshot = MappingProxyType(copy.deepcopy(questionnaire_data))

    executor = MethodExecutor()

    return ProcessorBundle(
        method_executor=executor,
        questionnaire=questionnaire_snapshot,
        factory=core_factory,
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
    'load_questionnaire_monolith',
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
]
