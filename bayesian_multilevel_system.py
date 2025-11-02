"""Compatibility wrapper for Bayesian Multi-Level Analysis System."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.analysis.bayesian_multilevel_system import (  # noqa: F401, E402
    BayesianPortfolioComposer,
    BayesianRollUp,
    BayesianUpdate,
    BayesianUpdater,
    ContradictionDetection,
    ContradictionScanner,
    DispersionEngine,
    MacroLevelAnalysis,
    MesoLevelAnalysis,
    MicroLevelAnalysis,
    MultiLevelBayesianOrchestrator,
    PeerCalibrator,
    PeerComparison,
    PeerContext,
    PenaltyCategory,
    ProbativeTest,
    ProbativeTestType,
    ReconciliationValidator,
    ValidationResult,
    ValidationRule,
    ValidatorType,
)

__all__ = [
    "BayesianPortfolioComposer",
    "BayesianRollUp",
    "BayesianUpdate",
    "BayesianUpdater",
    "ContradictionDetection",
    "ContradictionScanner",
    "DispersionEngine",
    "MacroLevelAnalysis",
    "MesoLevelAnalysis",
    "MicroLevelAnalysis",
    "MultiLevelBayesianOrchestrator",
    "PeerCalibrator",
    "PeerComparison",
    "PeerContext",
    "PenaltyCategory",
    "ProbativeTest",
    "ProbativeTestType",
    "ReconciliationValidator",
    "ValidationResult",
    "ValidationRule",
    "ValidatorType",
]
