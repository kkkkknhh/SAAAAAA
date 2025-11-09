"""Compatibility wrapper for demo macro prompts - example/demo version of macro prompts."""
import importlib.util
from pathlib import Path

from saaaaaa.analysis.macro_prompts import (  # noqa: F401
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

# Ensure src/ is in path for imports within the example
# Load the actual module from examples/
_root = Path(__file__).parent
_module_path = _root / "examples" / "demo_macro_prompts.py"
_spec = importlib.util.spec_from_file_location("_demo_macro_prompts_impl", _module_path)
if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    
    # Re-export everything from the module
    for _name in dir(_module):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_module, _name)
