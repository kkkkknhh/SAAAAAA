"""Compatibility wrapper for policy processor components."""
from __future__ import annotations

from pathlib import Path

# Ensure src/ is in path for imports
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
