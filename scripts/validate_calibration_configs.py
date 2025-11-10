#!/usr/bin/env python3
"""
CI Validation Script for Three-Pillar Calibration System

This script validates all calibration config files and should be run in CI/CD.
Exits with non-zero status if any validation fails.

Spec compliance: Section 8 (CI / QA Rules)
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration.validators import validate_config_files


def main():
    """Run all calibration config validations"""
    print("=" * 70)
    print("Three-Pillar Calibration System - CI Validation")
    print("=" * 70)
    print()
    
    print("Validating calibration configuration files...")
    print()
    
    # Run validation
    is_valid, errors = validate_config_files()
    
    if is_valid:
        print("✅ All calibration configs are valid!")
        print()
        print("Validated:")
        print("  - config/intrinsic_calibration.json")
        print("  - config/contextual_parametrization.json")
        print("  - config/fusion_specification.json")
        print()
        print("All constraints satisfied:")
        print("  ✓ Base weights sum to 1.0")
        print("  ✓ All intrinsic scores in [0,1]")
        print("  ✓ All fusion weights sum to 1.0 per role")
        print("  ✓ All weights are non-negative")
        print()
        return 0
    else:
        print("❌ Calibration config validation FAILED")
        print()
        print("Errors found:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        print()
        print("Please fix these errors before merging.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
