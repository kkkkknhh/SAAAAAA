"""Compatibility wrapper for macro-level prompt builders."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.analysis.macro_prompts import (  # noqa: F401, E402
    BayesianPortfolio,
    BayesianPortfolioComposer,
    ContradictionReport,
    ContradictionScanner,
    CoverageAnalysis,
    CoverageGapStressor,
    ImplementationRoadmap,
    MacroPromptsOrchestrator,
    PeerNormalization,
    PeerNormalizer,
    RoadmapOptimizer,
)

__all__ = [
    "BayesianPortfolio",
    "BayesianPortfolioComposer",
    "CoverageAnalysis",
    "CoverageGapStressor",
    "ContradictionReport",
    "ContradictionScanner",
    "ImplementationRoadmap",
    "MacroPromptsOrchestrator",
    "PeerNormalization",
    "PeerNormalizer",
    "RoadmapOptimizer",
]
