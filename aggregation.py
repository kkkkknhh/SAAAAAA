"""Compatibility wrapper for the refactored aggregation module."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

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
