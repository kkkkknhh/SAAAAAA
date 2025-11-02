"""Compatibility wrapper for verify_complete_implementation script."""
# Re-export from scripts directory
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import everything from the actual implementation
from verify_complete_implementation import *  # noqa: F401, F403
