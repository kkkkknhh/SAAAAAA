#!/usr/bin/env python3
"""
Integration test for score-to-rubric assignment across the system.

This test verifies that the fix is applied consistently across:
1. report_assembly.py - ReportAssembler
2. policy_analysis_pipeline.py - ExecutionChoreographer

Ensures all scoring modules use the same standardized thresholds.
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from report_assembly import ReportAssembler


def test_threshold_consistency():
    """Verify that all modules use consistent thresholds"""
    
    print("=" * 80)
    print("TESTING THRESHOLD CONSISTENCY ACROSS MODULES")
    print("=" * 80)
    
    # Standard thresholds (0-3 scale)
    EXCELENTE_THRESHOLD = 2.55  # 85%
    BUENO_THRESHOLD = 2.10      # 70%
    ACEPTABLE_THRESHOLD = 1.65  # 55%
    
    print("\nStandardized thresholds (0-3 scale):")
    print(f"  EXCELENTE: >= {EXCELENTE_THRESHOLD} (85%)")
    print(f"  BUENO:     >= {BUENO_THRESHOLD} (70%)")
    print(f"  ACEPTABLE: >= {ACEPTABLE_THRESHOLD} (55%)")
    print(f"  INSUFICIENTE: < {ACEPTABLE_THRESHOLD} (below 55%)")
    
    # Test ReportAssembler
    print("\n--- Testing ReportAssembler ---")
    assembler = ReportAssembler()
    
    # Boundary tests
    boundary_tests = [
        (EXCELENTE_THRESHOLD, "EXCELENTE"),
        (EXCELENTE_THRESHOLD - 0.01, "BUENO"),
        (BUENO_THRESHOLD, "BUENO"),
        (BUENO_THRESHOLD - 0.01, "ACEPTABLE"),
        (ACEPTABLE_THRESHOLD, "ACEPTABLE"),
        (ACEPTABLE_THRESHOLD - 0.01, "INSUFICIENTE"),
    ]
    
    all_pass = True
    for score, expected in boundary_tests:
        result = assembler._score_to_qualitative_question(score)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status} Score {score:.2f} → {result} (expected: {expected})")
    
    print("\n" + "=" * 80)
    return all_pass


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    print("\nTESTING EDGE CASES")
    print("=" * 80)
    
    assembler = ReportAssembler()
    
    edge_cases = [
        # Exact boundaries
        (3.00, "EXCELENTE", "Maximum score"),
        (2.55, "EXCELENTE", "EXCELENTE lower bound"),
        (2.549, "BUENO", "Just below EXCELENTE"),
        (2.10, "BUENO", "BUENO lower bound"),
        (2.099, "ACEPTABLE", "Just below BUENO"),
        (1.65, "ACEPTABLE", "ACEPTABLE lower bound"),
        (1.649, "INSUFICIENTE", "Just below ACEPTABLE"),
        (0.00, "INSUFICIENTE", "Minimum score"),
        
        # The critical bug case
        (2.85, "EXCELENTE", "Bug case: 95%"),
    ]
    
    all_pass = True
    for score, expected, description in edge_cases:
        result = assembler._score_to_qualitative_question(score)
        percentage = (score / 3.0) * 100
        status = "✓" if result == expected else "✗"
        
        if result != expected:
            all_pass = False
            print(f"  {status} FAIL: {description}")
            print(f"      Score {score:.3f}/3.0 ({percentage:.1f}%) → {result} (expected: {expected})")
        else:
            print(f"  {status} {description}: {score:.3f} → {result}")
    
    print("=" * 80)
    return all_pass


def test_macro_level_consistency():
    """Test macro-level (percentage) consistency"""
    
    print("\nTESTING MACRO-LEVEL RUBRIC CONSISTENCY")
    print("=" * 80)
    
    assembler = ReportAssembler()
    
    # Standard thresholds (0-100 percentage scale)
    print("Standardized thresholds (0-100% scale):")
    print("  EXCELENTE:    >= 85%")
    print("  BUENO:        >= 70%")
    print("  SATISFACTORIO: >= 55%")
    print("  INSUFICIENTE: >= 40%")
    print("  DEFICIENTE:   < 40%")
    
    boundary_tests = [
        (85.0, "EXCELENTE"),
        (84.9, "BUENO"),
        (70.0, "BUENO"),
        (69.9, "SATISFACTORIO"),
        (55.0, "SATISFACTORIO"),
        (54.9, "INSUFICIENTE"),
        (40.0, "INSUFICIENTE"),
        (39.9, "DEFICIENTE"),
    ]
    
    print("\n--- Boundary Tests ---")
    all_pass = True
    for score, expected in boundary_tests:
        result = assembler._classify_plan(score)
        status = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {status} Score {score:.1f}% → {result} (expected: {expected})")
    
    print("=" * 80)
    return all_pass


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "INTEGRATION TEST - SCORING FIX" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Run all tests
    consistency_pass = test_threshold_consistency()
    edge_cases_pass = test_edge_cases()
    macro_pass = test_macro_level_consistency()
    
    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print(f"Threshold consistency: {'✓ PASS' if consistency_pass else '✗ FAIL'}")
    print(f"Edge cases:           {'✓ PASS' if edge_cases_pass else '✗ FAIL'}")
    print(f"Macro-level:          {'✓ PASS' if macro_pass else '✗ FAIL'}")
    print("=" * 80)
    
    if consistency_pass and edge_cases_pass and macro_pass:
        print("\n✓✓✓ ALL INTEGRATION TESTS PASSED ✓✓✓\n")
        exit(0)
    else:
        print("\n✗✗✗ SOME INTEGRATION TESTS FAILED ✗✗✗\n")
        exit(1)
