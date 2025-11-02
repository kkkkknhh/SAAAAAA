"""Compatibility wrapper for meso-level cluster analysis utilities."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.analysis.meso_cluster_analysis import (  # noqa: F401, E402
    analyze_policy_dispersion,
    calibrate_against_peers,
    compose_cluster_posterior,
    reconcile_cross_metrics,
)

__all__ = [
    "analyze_policy_dispersion",
    "calibrate_against_peers",
    "compose_cluster_posterior",
    "reconcile_cross_metrics",
]
