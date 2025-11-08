"""Compatibility wrapper for the refactored aggregation module."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.processing.aggregation import (  # noqa: F401, E402
    AggregationError,
    AreaPolicyAggregator,
    AreaScore,
    ClusterAggregator,
    ClusterScore,
    CoverageError,
    DimensionAggregator,
    DimensionScore,
    HermeticityValidationError,
    MacroAggregator,
    MacroScore,
    ScoredResult,
    ThresholdValidationError,
    ValidationError,
    WeightValidationError,
)

__all__ = [
    "AggregationError",
    "AreaPolicyAggregator",
    "AreaScore",
    "ClusterAggregator",
    "ClusterScore",
    "CoverageError",
    "DimensionAggregator",
    "DimensionScore",
    "HermeticityValidationError",
    "MacroAggregator",
    "MacroScore",
    "ScoredResult",
    "ThresholdValidationError",
    "ValidationError",
    "WeightValidationError",
]
