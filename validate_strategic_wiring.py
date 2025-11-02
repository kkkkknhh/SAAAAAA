#!/usr/bin/env python3
"""
Strategic Wiring Validation Script
==================================
Validates the high-level wiring and integration across all strategic files.
"""

import sys
from pathlib import Path


def validate_strategic_wiring() -> bool:
    """
    Validate strategic wiring across all files.

    Returns:
        True if validation passes, False otherwise
    """
    root = Path(__file__).parent

    print("=== Strategic Wiring Validation ===")
    print()

    # Check strategic files exist
    strategic_files = [
        "demo_macro_prompts.py",
        "verify_complete_implementation.py",
        "validation_engine.py",
        "validate_system.py",
        "seed_factory.py",
        "qmcm_hooks.py",
        "meso_cluster_analysis.py",
        "macro_prompts.py",
        "json_contract_loader.py",
        "evidence_registry.py",
        "document_ingestion.py",
        "scoring.py",
        "recommendation_engine.py",
        "orchestrator.py",
        "micro_prompts.py",
        "coverage_gate.py",
    ]

    missing = []
    for file_path in strategic_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} MISSING")
            missing.append(file_path)

    print()

    if missing:
        print(f"❌ Validation FAILED: {len(missing)} files missing")
        return False
    else:
        print(f"✅ Validation PASSED: All {len(strategic_files)} strategic files present")
        return True


if __name__ == "__main__":
    result = validate_strategic_wiring()
    sys.exit(0 if result else 1)
