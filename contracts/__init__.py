"""
Contracts package - defines API contracts and interfaces.
This package contains contract definitions and validation logic.

This __init__.py re-exports from the compatibility shim at ../contracts.py
to maintain backward compatibility.
"""

import sys
from pathlib import Path

# Import from the contracts.py file (not this directory)
# We need to be careful here since this directory shadows contracts.py
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Now import from the actual contracts compatibility shim
# We have to use importlib to avoid importing ourselves
import importlib.util
_contracts_file = Path(__file__).parent.parent / "contracts.py"
_spec = importlib.util.spec_from_file_location("_contracts_shim", _contracts_file)
if _spec and _spec.loader:
    _contracts_shim = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_contracts_shim)
    
    # Re-export everything from the shim
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
else:
    __all__ = []
