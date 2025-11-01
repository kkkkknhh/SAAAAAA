"""
Scoring Module

Implements TYPE_A through TYPE_F scoring modalities with strict validation
and reproducible results.
"""

from .scoring import (
    ScoringModality,
    QualityLevel,
    ScoringError,
    ModalityValidationError,
    EvidenceStructureError,
    ScoredResult,
    ModalityConfig,
    ScoringValidator,
    apply_scoring,
    determine_quality_level,
)

__all__ = [
    "ScoringModality",
    "QualityLevel",
    "ScoringError",
    "ModalityValidationError",
    "EvidenceStructureError",
    "ScoredResult",
    "ModalityConfig",
    "ScoringValidator",
    "apply_scoring",
    "determine_quality_level",
]
