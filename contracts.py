"""Compatibility wrapper for the refactored contracts module."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.utils.contracts import (  # noqa: F401, E402
    AnalysisInputV1,
    AnalysisInputV1Optional,
    AnalysisOutputV1,
    AnalysisOutputV1Optional,
    AnalyzerProtocol,
    ContractMismatchError,
    DocumentLoaderProtocol,
    DocumentMetadataV1,
    DocumentMetadataV1Optional,
    ExecutionContextV1,
    ExecutionContextV1Optional,
    MISSING,
    ProcessedTextV1,
    ProcessedTextV1Optional,
    SentenceCollection,
    TextDocument,
    TextProcessorProtocol,
    ensure_hashable,
    ensure_iterable_not_string,
    validate_contract,
    validate_mapping_keys,
)
from saaaaaa.utils.seed_factory import SeedFactory  # noqa: F401, E402

__all__ = [
    "AnalysisInputV1",
    "AnalysisInputV1Optional",
    "AnalysisOutputV1",
    "AnalysisOutputV1Optional",
    "AnalyzerProtocol",
    "ContractMismatchError",
    "DocumentLoaderProtocol",
    "DocumentMetadataV1",
    "DocumentMetadataV1Optional",
    "ExecutionContextV1",
    "ExecutionContextV1Optional",
    "MISSING",
    "ProcessedTextV1",
    "ProcessedTextV1Optional",
    "SeedFactory",
    "SentenceCollection",
    "TextDocument",
    "TextProcessorProtocol",
    "ensure_hashable",
    "ensure_iterable_not_string",
    "validate_contract",
    "validate_mapping_keys",
]
