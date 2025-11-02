"""Compatibility wrapper for demo_macro_prompts example."""
# Re-export from examples directory
import sys
from pathlib import Path

# Add examples directory to path
examples_dir = Path(__file__).parent / "examples"
sys.path.insert(0, str(examples_dir))

# Import everything from the actual implementation
from demo_macro_prompts import *  # noqa: F401, F403
