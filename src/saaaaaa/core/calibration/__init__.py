"""
SAAAAAA Calibration System.

This package implements the 7-layer method calibration framework:
- @b (Base): Intrinsic method quality
- @u (Unit): PDT quality
- @q, @d, @p (Contextual): Method-context compatibility
- @C (Congruence): Method ensemble validation
- @chain (Chain): Data flow integrity
- @m (Meta): Governance and observability

Final scores are produced via Choquet 2-Additive aggregation.
"""

from .data_structures import (
    LayerID,
    LayerScore,
    ContextTuple,
    CalibrationSubject,
    CompatibilityMapping,
    InteractionTerm,
    CalibrationResult,
)

from .config import (
    UnitLayerConfig,
    MetaLayerConfig,
    ChoquetAggregationConfig,
    CalibrationSystemConfig,
    DEFAULT_CALIBRATION_CONFIG,
)

__all__ = [
    # Data structures
    "LayerID",
    "LayerScore",
    "ContextTuple",
    "CalibrationSubject",
    "CompatibilityMapping",
    "InteractionTerm",
    "CalibrationResult",
    # Configuration
    "UnitLayerConfig",
    "MetaLayerConfig",
    "ChoquetAggregationConfig",
    "CalibrationSystemConfig",
    "DEFAULT_CALIBRATION_CONFIG",
]
