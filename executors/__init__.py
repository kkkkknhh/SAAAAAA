"""Compatibility wrapper exposing orchestrator executors."""
from pathlib import Path
import sys

_SRC_PATH = Path(__file__).resolve().parent.parent / "src"
if _SRC_PATH.exists():  # pragma: no cover - executed at import time
    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from saaaaaa.core.orchestrator import executors as _executors

__all__ = getattr(_executors, "__all__", [])

for name in dir(_executors):
    if name.startswith("_"):
        continue
    globals()[name] = getattr(_executors, name)
