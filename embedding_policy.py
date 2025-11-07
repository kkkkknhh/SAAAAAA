"""Compatibility wrapper for embedding policy analyzers."""
from pathlib import Path

# Ensure src/ is in path for imports
from saaaaaa.processing.embedding_policy import (  # noqa: F401, E402
    PolicyAnalysisEmbedder,
)

__all__ = [
    "PolicyAnalysisEmbedder",
]
