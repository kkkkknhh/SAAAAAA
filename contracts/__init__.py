"""
Contracts package - defines API contracts and interfaces.

This package provides backward compatibility by re-exporting
from saaaaaa.utils.contracts.
"""

# Direct imports from the source modules
from saaaaaa.utils.contracts import (  # noqa: F401
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
from saaaaaa.utils.seed_factory import SeedFactory  # noqa: F401

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
