"""Pytest configuration for ensuring src is on sys.path."""
import sys
from pathlib import Path

# Add src directory to sys.path for legacy import compatibility
src_path = Path(__file__).parent / "src"
if src_path.is_dir() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
