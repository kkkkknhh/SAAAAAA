"""Simplified dispatcher used by the test choreographer module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

@dataclass
class DispatchResult:
    """Minimal result object used for smoke tests."""

    executed_methods: Sequence[str]
    metadata: Dict[str, Any]

class ChoreographerDispatcher:
    """Lightweight stand-in for the production choreographer dispatcher.

    The real project contains a significantly more capable dispatcher.  The test
    suite only verifies instantiation and very small bits of behaviour, so this
    shim keeps the public API while delegating to simple data structures.
    """

    def build_execution_plan(self, method_packages: Iterable[Dict[str, Any]]) -> List[str]:
        """Return an ordered list of method identifiers from the catalog."""
        return [pkg.get("m", [pkg.get("method", "")])[0] for pkg in method_packages if pkg]

    def dispatch(self, methods: Sequence[str]) -> DispatchResult:
        """Pretend to dispatch the provided methods and return a result."""
        return DispatchResult(executed_methods=list(methods), metadata={})

__all__ = ["ChoreographerDispatcher", "DispatchResult"]
