"""Compatibility shim for :mod:`saaaaaa.concurrency.concurrency`."""
from __future__ import annotations
from pathlib import Path
import sys

# Add src to path for development environments
_SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if _SRC_PATH.exists():  # pragma: no cover - executed at import time
    src_str = str(_SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from saaaaaa.concurrency.concurrency import *  # noqa: F401,F403,E402
