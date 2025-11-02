"""Compatibility wrapper for validate_system script."""
# Re-export from scripts directory
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import everything from the actual implementation
from validate_system import *  # noqa: F401, F403
