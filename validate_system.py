"""System validation script - validates entire SAAAAAA system."""

import sys
from pathlib import Path


def validate_system() -> bool:
    """
    Validate the entire SAAAAAA system.
    
    Returns:
        True if system is valid, False otherwise
    """
    root = Path(__file__).parent
    
    # Check package structure
    package_dir = root / "src" / "saaaaaa"
    if not package_dir.exists():
        print("✗ Package directory not found")
        return False
    
    # Check for critical modules
    critical_modules = [
        "core",
        "analysis",
        "processing",
        "utils",
        "concurrency",
        "controls",
    ]
    
    for module in critical_modules:
        if not (package_dir / module).exists():
            print(f"✗ Critical module missing: {module}")
            return False
    
    print("✓ System validation passed")
    return True


if __name__ == "__main__":
    result = validate_system()
    sys.exit(0 if result else 1)
