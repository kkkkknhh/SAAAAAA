#!/usr/bin/env python3
"""
Verification script for import path fixes.

This script verifies that all class registry paths use absolute imports
with the saaaaaa.* namespace, resolving the Import Failure issue.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from saaaaaa.core.orchestrator.class_registry import get_class_paths


def main():
    """Verify all import paths are absolute."""
    print("=" * 80)
    print("IMPORT PATH VERIFICATION")
    print("=" * 80)
    
    paths = get_class_paths()
    
    print(f"\nTotal classes registered: {len(paths)}")
    print("\nVerifying all paths use absolute imports with saaaaaa namespace...")
    
    issues = []
    analysis_count = 0
    processing_count = 0
    
    for class_name, import_path in sorted(paths.items()):
        # Check if path starts with saaaaaa
        if not import_path.startswith("saaaaaa."):
            issues.append(f"  ❌ {class_name}: {import_path} (not using saaaaaa namespace)")
        else:
            if "saaaaaa.analysis." in import_path:
                analysis_count += 1
            elif "saaaaaa.processing." in import_path:
                processing_count += 1
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(issue)
        return 1
    else:
        print("\n✅ All paths verified!")
        print(f"\n   • {processing_count} classes from saaaaaa.processing.*")
        print(f"   • {analysis_count} classes from saaaaaa.analysis.*")
        print(f"   • Total: {len(paths)} classes\n")
        
        print("Sample paths:")
        for class_name, import_path in list(sorted(paths.items()))[:5]:
            print(f"  ✓ {class_name:35} → {import_path}")
        
        print("\n" + "=" * 80)
        print("✅ Import path fix verified successfully!")
        print("=" * 80)
        return 0


if __name__ == "__main__":
    sys.exit(main())
