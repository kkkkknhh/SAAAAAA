"""Compatibility wrapper for scoring module."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.analysis.scoring import (  # noqa: F401, E402
    Evidence,
    MicroQuestionScorer,
    QualityLevel,
    ScoredResult,
    ScoringConfig,
    ScoringModality,
    score_question,
)

__all__ = [
    "Evidence",
    "MicroQuestionScorer",
    "QualityLevel",
    "ScoredResult",
    "ScoringConfig",
    "ScoringModality",
    "score_question",
]
