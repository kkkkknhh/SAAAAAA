"""Compatibility wrapper for deterministic seed helpers."""
import sys
from pathlib import Path

# Ensure src/ is in path for imports
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from saaaaaa.utils.seed_factory import (  # noqa: F401, E402
    DeterministicContext,
    SeedFactory,
    create_deterministic_seed,
)

__all__ = [
    "DeterministicContext",
    "SeedFactory",
    "create_deterministic_seed",
]
