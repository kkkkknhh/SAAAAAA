"""Compatibility wrapper for demo_macro_prompts example."""
# Import from the actual implementation in examples/
import importlib.util
import sys
from pathlib import Path

# Ensure src/ is in path for imports within the example
_root = Path(__file__).parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

# Load the actual module from examples/
_module_path = _root / "examples" / "demo_macro_prompts.py"
_spec = importlib.util.spec_from_file_location("_demo_macro_prompts_impl", _module_path)
if _spec and _spec.loader:
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    
    # Re-export everything from the module
    for _name in dir(_module):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_module, _name)

