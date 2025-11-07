"""Compatibility wrapper for verify_complete_implementation script."""
# Import from the actual implementation in scripts/
import importlib.util
from pathlib import Path

# Ensure src/ is in path for imports within the script
# Load the actual module from scripts/
_module_path = _root / "scripts" / "verify_complete_implementation.py"
_spec = importlib.util.spec_from_file_location("_verify_complete_impl", _module_path)
if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)

    # Re-export everything from the module
    for _name in dir(_module):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_module, _name)
