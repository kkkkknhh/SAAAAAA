"""Compatibility wrapper for the recommendation engine."""
from saaaaaa.analysis.recommendation_engine import (  # noqa: F401
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
