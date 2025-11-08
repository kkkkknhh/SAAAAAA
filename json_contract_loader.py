"""Compatibility wrapper for JSON contract loader utilities."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.utils.json_contract_loader import (  # noqa: F401, E402
    ContractDocument,
    ContractLoadReport,
    JSONContractLoader,
)

__all__ = [
    "ContractDocument",
    "ContractLoadReport",
    "JSONContractLoader",
]
