"""Test harness bootstrap for legacy import paths.

This module ensures the project source tree is discoverable without
requiring ``pip install -e .``.  Python automatically imports
``sitecustomize`` during interpreter start-up, so we use it to insert the
``src`` directory at the front of ``sys.path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"
if SRC_PATH.is_dir():
    src_str = str(SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
