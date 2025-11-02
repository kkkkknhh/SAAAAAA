"""Compatibility wrapper for macro-level prompt builders."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

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
