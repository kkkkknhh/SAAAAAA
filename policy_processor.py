"""Compatibility wrapper for policy processor components."""
from saaaaaa.processing.policy_processor import (  # noqa: F401
    BayesianEvidenceScorer,
    IndustrialPolicyProcessor,
    PolicyTextProcessor,
)

__all__ = [
    "BayesianEvidenceScorer",
    "IndustrialPolicyProcessor",
    "PolicyTextProcessor",
]
