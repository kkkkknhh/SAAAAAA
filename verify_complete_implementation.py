#!/usr/bin/env python3
"""
Complete Implementation Verification Script
Verifies all components of the Bayesian Multi-Level Analysis System
"""

from pathlib import Path
import sys

def verify_implementation():
    """Verify complete implementation"""
    print("=" * 80)
    print("BAYESIAN MULTI-LEVEL ANALYSIS SYSTEM - VERIFICATION")
    print("=" * 80)
    print()
    
    checks = {
        'passed': 0,
        'failed': 0
    }
    
    # Check 1: Core module exists
    print("[1] Core Module (bayesian_multilevel_system.py)")
    if Path('bayesian_multilevel_system.py').exists():
        size = Path('bayesian_multilevel_system.py').stat().st_size
        lines = len(Path('bayesian_multilevel_system.py').read_text().splitlines())
        print(f"    ✓ File exists: {size:,} bytes, {lines:,} lines")
        checks['passed'] += 1
    else:
        print("    ✗ File not found")
        checks['failed'] += 1
    
    # Check 2: Test suite exists and passes
    print("[2] Test Suite (tests/test_bayesian_multilevel_system.py)")
    if Path('tests/test_bayesian_multilevel_system.py').exists():
        lines = len(Path('tests/test_bayesian_multilevel_system.py').read_text().splitlines())
        print(f"    ✓ File exists: {lines:,} lines")
        
        # Run tests
        import unittest
        from tests import test_bayesian_multilevel_system
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(test_bayesian_multilevel_system)
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print(f"    ✓ All {result.testsRun} tests passed")
            checks['passed'] += 1
        else:
            print(f"    ✗ {len(result.failures)} failures, {len(result.errors)} errors")
            checks['failed'] += 1
    else:
        print("    ✗ File not found")
        checks['failed'] += 1
    
    # Check 3: Demo script exists
    print("[3] Demonstration (demo_bayesian_multilevel.py)")
    if Path('demo_bayesian_multilevel.py').exists():
        lines = len(Path('demo_bayesian_multilevel.py').read_text().splitlines())
        print(f"    ✓ File exists: {lines:,} lines")
        checks['passed'] += 1
    else:
        print("    ✗ File not found")
        checks['failed'] += 1
    
    # Check 4: Integration guide exists
    print("[4] Integration Guide (integration_guide_bayesian.py)")
    if Path('integration_guide_bayesian.py').exists():
        lines = len(Path('integration_guide_bayesian.py').read_text().splitlines())
        print(f"    ✓ File exists: {lines:,} lines")
        checks['passed'] += 1
    else:
        print("    ✗ File not found")
        checks['failed'] += 1
    
    # Check 5: Documentation exists
    print("[5] Documentation (BAYESIAN_MULTILEVEL_README.md)")
    if Path('BAYESIAN_MULTILEVEL_README.md').exists():
        lines = len(Path('BAYESIAN_MULTILEVEL_README.md').read_text().splitlines())
        print(f"    ✓ File exists: {lines:,} lines")
        checks['passed'] += 1
    else:
        print("    ✗ File not found")
        checks['failed'] += 1
    
    # Check 6: CSV outputs exist
    print("[6] CSV Outputs")
    csv_files = [
        'data/bayesian_outputs/posterior_table_micro.csv',
        'data/bayesian_outputs/posterior_table_meso.csv',
        'data/bayesian_outputs/posterior_table_macro.csv'
    ]
    all_exist = True
    for f in csv_files:
        if Path(f).exists():
            size = Path(f).stat().st_size
            print(f"    ✓ {Path(f).name}: {size} bytes")
        else:
            print(f"    ✗ {Path(f).name}: not found")
            all_exist = False
    
    if all_exist:
        checks['passed'] += 1
    else:
        checks['failed'] += 1
    
    # Check 7: All classes importable
    print("[7] Module Imports")
    try:
        from bayesian_multilevel_system import (
            ValidationRule, ValidatorType, ReconciliationValidator,
            ProbativeTest, ProbativeTestType, BayesianUpdater,
            DispersionEngine, PeerCalibrator, PeerContext,
            BayesianRollUp, ContradictionScanner, BayesianPortfolioComposer,
            MultiLevelBayesianOrchestrator
        )
        print("    ✓ All 13 classes imported successfully")
        checks['passed'] += 1
    except Exception as e:
        print(f"    ✗ Import failed: {e}")
        checks['failed'] += 1
    
    # Summary
    print()
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    total = checks['passed'] + checks['failed']
    print(f"Checks passed: {checks['passed']}/{total}")
    print(f"Checks failed: {checks['failed']}/{total}")
    
    if checks['failed'] == 0:
        print()
        print("✅ IMPLEMENTATION COMPLETE - All checks passed")
        print()
        print("The Bayesian Multi-Level Analysis System is fully operational:")
        print("  • Reconciliation Layer (micro)")
        print("  • Bayesian Updater (micro)")
        print("  • Dispersion Engine (meso)")
        print("  • Peer Calibration (meso)")
        print("  • Bayesian Roll-Up (meso)")
        print("  • Contradiction Scanner (macro)")
        print("  • Bayesian Portfolio Composer (macro)")
        print()
        print("Ready for integration with report_assembly.py")
        return 0
    else:
        print()
        print("⚠️ IMPLEMENTATION INCOMPLETE - Some checks failed")
        return 1

if __name__ == '__main__':
    sys.exit(verify_implementation())
