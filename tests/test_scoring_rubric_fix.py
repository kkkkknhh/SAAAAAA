#!/usr/bin/env python3
"""
Test for Score-to-Rubric Assignment Fix

Verifies that the bug described in the issue is fixed:
- High scores (e.g., 2.85/3.0 = 95%) should map to high rubrics (EXCELENTE)
- Not to the lowest rubric (INSUFICIENTE)
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from report_assembly import ReportAssembler


def test_question_score_to_rubric():
    """Test question-level score to rubric mapping (0-3 scale)"""
    assembler = ReportAssembler()
    
    print("=" * 80)
    print("TESTING QUESTION-LEVEL SCORE TO RUBRIC MAPPING (0-3 scale)")
    print("=" * 80)
    
    # Test cases: (score, expected_rubric)
    test_cases = [
        # High scores should map to high rubrics
        (3.00, "EXCELENTE"),  # 100%
        (2.85, "EXCELENTE"),  # 95% - THE BUG CASE from problem statement
        (2.70, "EXCELENTE"),  # 90%
        (2.55, "EXCELENTE"),  # 85% - boundary
        
        # Medium-high scores
        (2.54, "BUENO"),      # Just below EXCELENTE
        (2.40, "BUENO"),      # 80%
        (2.10, "BUENO"),      # 70% - boundary
        
        # Medium scores
        (2.09, "ACEPTABLE"),  # Just below BUENO
        (1.80, "ACEPTABLE"),  # 60%
        (1.65, "ACEPTABLE"),  # 55% - boundary
        
        # Low scores
        (1.64, "INSUFICIENTE"),  # Just below ACEPTABLE
        (1.50, "INSUFICIENTE"),  # 50%
        (1.00, "INSUFICIENTE"),  # 33%
        (0.50, "INSUFICIENTE"),  # 17%
        (0.00, "INSUFICIENTE"),  # 0%
    ]
    
    all_pass = True
    
    for score, expected in test_cases:
        result = assembler._score_to_qualitative_question(score)
        percentage = (score / 3.0) * 100
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result != expected:
            all_pass = False
            print(f"{status}: Score {score:.2f}/3.0 ({percentage:.1f}%) → {result} (expected: {expected})")
        else:
            print(f"{status}: Score {score:.2f}/3.0 ({percentage:.1f}%) → {result}")
    
    print("=" * 80)
    
    # Highlight the critical bug case
    print("\n" + "=" * 80)
    print("CRITICAL BUG CASE FROM PROBLEM STATEMENT:")
    print("=" * 80)
    bug_score = 2.85
    bug_result = assembler._score_to_qualitative_question(bug_score)
    bug_percentage = (bug_score / 3.0) * 100
    print(f"Score: {bug_score}/3.0 (raw: {bug_score/3.0:.2f} = {bug_percentage:.1f}%)")
    print(f"Result: {bug_result}")
    print(f"Expected: EXCELENTE")
    print(f"Status: {'✓ FIXED' if bug_result == 'EXCELENTE' else '✗ STILL BROKEN'}")
    print("=" * 80)
    
    return all_pass


def test_macro_score_to_rubric():
    """Test macro-level score to rubric mapping (0-100 percentage scale)"""
    assembler = ReportAssembler()
    
    print("\n" + "=" * 80)
    print("TESTING MACRO-LEVEL SCORE TO RUBRIC MAPPING (0-100 percentage scale)")
    print("=" * 80)
    
    # Test cases: (score_percentage, expected_rubric)
    test_cases = [
        # Excellent
        (100.0, "EXCELENTE"),
        (95.0, "EXCELENTE"),
        (85.0, "EXCELENTE"),  # boundary
        
        # Good
        (84.9, "BUENO"),
        (75.0, "BUENO"),
        (70.0, "BUENO"),      # boundary
        
        # Satisfactory
        (69.9, "SATISFACTORIO"),
        (60.0, "SATISFACTORIO"),
        (55.0, "SATISFACTORIO"),  # boundary
        
        # Insufficient
        (54.9, "INSUFICIENTE"),
        (45.0, "INSUFICIENTE"),
        (40.0, "INSUFICIENTE"),   # boundary
        
        # Deficient
        (39.9, "DEFICIENTE"),
        (20.0, "DEFICIENTE"),
        (0.0, "DEFICIENTE"),
    ]
    
    all_pass = True
    
    for score, expected in test_cases:
        result = assembler._classify_plan(score)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result != expected:
            all_pass = False
            print(f"{status}: Score {score:.1f}% → {result} (expected: {expected})")
        else:
            print(f"{status}: Score {score:.1f}% → {result}")
    
    print("=" * 80)
    
    return all_pass


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "SCORE-TO-RUBRIC ASSIGNMENT FIX TEST" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    # Run tests
    question_pass = test_question_score_to_rubric()
    macro_pass = test_macro_score_to_rubric()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Question-level mapping: {'✓ ALL TESTS PASSED' if question_pass else '✗ SOME TESTS FAILED'}")
    print(f"Macro-level mapping:    {'✓ ALL TESTS PASSED' if macro_pass else '✗ SOME TESTS FAILED'}")
    print("=" * 80)
    
    if question_pass and macro_pass:
        print("\n✓✓✓ ALL TESTS PASSED - BUG IS FIXED! ✓✓✓\n")
        exit(0)
    else:
        print("\n✗✗✗ SOME TESTS FAILED - BUG STILL EXISTS ✗✗✗\n")
        exit(1)
