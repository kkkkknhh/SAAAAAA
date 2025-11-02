"""Compatibility wrapper for embedding policy analyzers."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.processing.embedding_policy import (  # noqa: F401, E402
    PolicyAnalysisEmbedder,
)

__all__ = [
    "PolicyAnalysisEmbedder",
]
