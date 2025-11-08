"""Compatibility wrapper for the recommendation engine."""
from __future__ import annotations

from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.analysis.recommendation_engine import (  # noqa: E402
    Recommendation,
    RecommendationEngine,
    RecommendationSet,
    load_recommendation_engine,
)

# Backwards compatible aliases for legacy names
RecommendationResultSet = RecommendationSet
RecommendationRule = Recommendation
RecommendationRuleSet = RecommendationSet

__all__ = [
    "Recommendation",
    "RecommendationEngine",
    "RecommendationResultSet",
    "RecommendationRule",
    "RecommendationRuleSet",
    "RecommendationSet",
    "load_recommendation_engine",
]
