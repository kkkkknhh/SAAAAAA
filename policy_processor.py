"""Compatibility wrapper for policy processor components."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.processing.policy_processor import (  # noqa: E402
    BayesianEvidenceScorer,
    IndustrialPolicyProcessor,
    PolicyTextProcessor,
)

__all__ = [
    "BayesianEvidenceScorer",
    "IndustrialPolicyProcessor",
    "PolicyTextProcessor",
]
