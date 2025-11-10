#!/usr/bin/env python3
"""
SIN_CARRETA Calibration Coverage Validator

This script enforces calibration coverage requirements:
- Minimum 25% coverage of methods requiring calibration
- Fail loudly if coverage is below threshold
- Compute coverage from canonical_method_catalog.json against calibration_registry.py

Exit codes:
- 0: Coverage meets or exceeds 25% threshold
- 1: Coverage below 25% threshold (FAIL)
- 2: Script error (misconfiguration, missing files, etc.)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set, Tuple


class CalibrationCoverageError(Exception):
    """Raised when calibration coverage is below threshold"""
    pass


def load_canonical_catalog(catalog_path: Path) -> Dict:
    """Load canonical method catalog"""
    try:
        with open(catalog_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Canonical catalog not found: {catalog_path}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in catalog: {e}")
        sys.exit(2)


def extract_methods_requiring_calibration(catalog: Dict) -> Set[Tuple[str, str]]:
    """
    Extract set of methods requiring calibration from catalog.
    
    Returns:
        Set of (class_name, method_name) tuples for methods requiring calibration
    """
    methods_requiring_calibration = set()
    
    # Iterate through all layers
    for layer_name, methods in catalog.get("layers", {}).items():
        for method_info in methods:
            if method_info.get("requires_calibration", False):
                class_name = method_info.get("class_name", "")
                method_name = method_info.get("method_name", "")
                if class_name and method_name:
                    methods_requiring_calibration.add((class_name, method_name))
    
    return methods_requiring_calibration


def load_calibration_registry() -> Set[Tuple[str, str]]:
    """
    Load calibrations from calibration_registry.py.
    
    Returns:
        Set of (class_name, method_name) tuples that have calibrations
    """
    # Import the calibration registry
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    
    try:
        from saaaaaa.core.orchestrator.calibration_registry import CALIBRATIONS
    except ImportError as e:
        print(f"ERROR: Could not import calibration_registry: {e}")
        print(f"Tried path: {repo_root / 'src'}")
        sys.exit(2)
    
    # Extract calibrated methods
    calibrated_methods = set()
    for (class_name, method_name) in CALIBRATIONS.keys():
        calibrated_methods.add((class_name, method_name))
    
    return calibrated_methods


def compute_coverage(
    required_methods: Set[Tuple[str, str]],
    calibrated_methods: Set[Tuple[str, str]]
) -> Tuple[float, int, int, Set[Tuple[str, str]]]:
    """
    Compute calibration coverage.
    
    Returns:
        (coverage_percentage, calibrated_count, required_count, uncalibrated_methods)
    """
    total_required = len(required_methods)
    if total_required == 0:
        return 100.0, 0, 0, set()
    
    # Find intersection: methods that are both required and calibrated
    calibrated_required = required_methods.intersection(calibrated_methods)
    calibrated_count = len(calibrated_required)
    
    # Find uncalibrated methods
    uncalibrated = required_methods - calibrated_methods
    
    coverage = (calibrated_count / total_required) * 100.0
    
    return coverage, calibrated_count, total_required, uncalibrated


def print_coverage_report(
    coverage: float,
    calibrated_count: int,
    required_count: int,
    uncalibrated: Set[Tuple[str, str]],
    threshold: float
):
    """Print detailed coverage report"""
    print("=" * 80)
    print("SIN_CARRETA CALIBRATION COVERAGE REPORT")
    print("=" * 80)
    print(f"\nTotal methods requiring calibration: {required_count}")
    print(f"Methods with calibration: {calibrated_count}")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Threshold: {threshold}%")
    print()
    
    if coverage >= threshold:
        print(f"✓ PASS: Coverage meets threshold ({coverage:.2f}% >= {threshold}%)")
    else:
        print(f"✗ FAIL: Coverage below threshold ({coverage:.2f}% < {threshold}%)")
        deficit = threshold - coverage
        methods_needed = int((deficit / 100.0) * required_count) + 1
        print(f"\nDeficit: {deficit:.2f}%")
        print(f"Additional calibrations needed: ~{methods_needed}")
    
    # Show sample of uncalibrated methods
    if uncalibrated:
        print(f"\nUncalibrated methods: {len(uncalibrated)}")
        print("\nSample of uncalibrated methods (first 10):")
        for i, (class_name, method_name) in enumerate(sorted(uncalibrated)[:10]):
            print(f"  {i+1}. {class_name}.{method_name}")
        if len(uncalibrated) > 10:
            print(f"  ... and {len(uncalibrated) - 10} more")
    
    print("\n" + "=" * 80)


def validate_coverage(threshold: float = 25.0) -> int:
    """
    Main validation function.
    
    Args:
        threshold: Minimum coverage percentage required
        
    Returns:
        0 if coverage >= threshold, 1 otherwise
    """
    # Paths
    repo_root = Path(__file__).parent.parent
    catalog_path = repo_root / "config" / "canonical_method_catalog.json"
    
    print(f"Repository root: {repo_root}")
    print(f"Catalog path: {catalog_path}")
    print()
    
    # Load data
    catalog = load_canonical_catalog(catalog_path)
    required_methods = extract_methods_requiring_calibration(catalog)
    calibrated_methods = load_calibration_registry()
    
    # Compute coverage
    coverage, calibrated_count, required_count, uncalibrated = compute_coverage(
        required_methods, calibrated_methods
    )
    
    # Print report
    print_coverage_report(coverage, calibrated_count, required_count, uncalibrated, threshold)
    
    # Determine pass/fail
    if coverage < threshold:
        return 1  # FAIL
    else:
        return 0  # PASS


def main():
    """CLI entry point"""
    # Parse arguments (simple threshold override)
    threshold = 25.0
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            print(f"ERROR: Invalid threshold: {sys.argv[1]}")
            print("Usage: validate_calibration_coverage.py [threshold_percentage]")
            sys.exit(2)
    
    # Run validation
    exit_code = validate_coverage(threshold)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
