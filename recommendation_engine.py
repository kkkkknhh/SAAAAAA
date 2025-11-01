"""Facade exposing the recommendation engine implementation."""
from __future__ import annotations

from pathlib import Path
import sys

_SRC_PATH = Path(__file__).resolve().parent / "src"
if _SRC_PATH.exists():  # pragma: no cover - executed at import time
    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from saaaaaa.analysis.recommendation_engine import *  # noqa: F401,F403

__all__ = [
    name
    for name in dir()
    if not name.startswith("_") and name not in {"annotations"}
]
