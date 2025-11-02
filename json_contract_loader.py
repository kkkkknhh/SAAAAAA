"""Compatibility wrapper for JSON contract loader utilities."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

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
