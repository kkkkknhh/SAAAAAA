#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI Validation for D2 Method Concurrence.

This script validates that all required methods for D2 questions are present
and can be resolved, following SIN_CARRETA doctrine.

Exit codes:
  0 - All validations passed
  1 - Validation failed (in strict mode)
  2 - Configuration or setup error
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.d2_activities_orchestrator import (
    D2ActivitiesOrchestrator,
    D2Question,
    OrchestrationError,
)


def main():
    """Main entry point for CI validation."""
    parser = argparse.ArgumentParser(
        description="Validate D2 method concurrence for CI/CD pipelines"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Enable strict mode (fail on any missing method)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save validation report (JSON)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="Print summary to stdout"
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.95,
        help="Minimum success rate (0.0-1.0) to pass in non-strict mode"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("D2 METHOD CONCURRENCE VALIDATION")
    print("SIN_CARRETA Doctrine: No Graceful Degradation | Deterministic Execution")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  - Strict mode: {args.strict}")
    print(f"  - Fail threshold: {args.fail_threshold * 100:.1f}%")
    if args.output:
        print(f"  - Output report: {args.output}")
    print()
    
    # Initialize orchestrator
    orchestrator = D2ActivitiesOrchestrator(
        strict_mode=args.strict,
        trace_execution=True
    )
    
    try:
        # Run validation
        results = orchestrator.validate_all_d2_questions()
        
        # Generate report
        report = orchestrator.generate_validation_report(
            results,
            output_path=Path(args.output) if args.output else None
        )
        
        # Calculate success metrics
        total_methods = report["summary"]["total_methods"]
        methods_resolved = report["summary"]["methods_resolved"]
        success_rate = methods_resolved / total_methods if total_methods > 0 else 0.0
        
        # Print summary if requested
        if args.summary or not args.output:
            print_summary(report, success_rate)
        
        # Determine pass/fail
        if args.strict:
            # In strict mode, all methods must be present
            if not report["summary"]["overall_success"]:
                print("\n❌ VALIDATION FAILED (Strict Mode)")
                print("One or more required methods are missing or unresolvable.")
                return 1
        else:
            # In non-strict mode, use threshold
            if success_rate < args.fail_threshold:
                print(f"\n❌ VALIDATION FAILED (Below Threshold)")
                print(f"Success rate: {success_rate * 100:.1f}% < {args.fail_threshold * 100:.1f}%")
                return 1
        
        print("\n✅ VALIDATION PASSED")
        print(f"Success rate: {success_rate * 100:.1f}%")
        return 0
        
    except OrchestrationError as e:
        print(f"\n❌ ORCHESTRATION ERROR: {e}")
        return 1
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


def print_summary(report: dict, success_rate: float):
    """Print a summary of the validation results."""
    print("VALIDATION RESULTS")
    print("-" * 80)
    print(f"Overall Success Rate: {success_rate * 100:.1f}%")
    print(f"Total Questions: {report['summary']['total_questions']}")
    print(f"Questions Passed: {report['summary']['questions_passed']}")
    print(f"Questions Failed: {report['summary']['questions_failed']}")
    print(f"Total Methods: {report['summary']['total_methods']}")
    print(f"Methods Resolved: {report['summary']['methods_resolved']}")
    print(f"Methods Failed: {report['summary']['methods_failed']}")
    print()
    
    print("Per-Question Results:")
    print("-" * 80)
    for question_id, question_data in report["questions"].items():
        status = "✓" if question_data["success"] else "✗"
        resolved = question_data["executed_methods"] - question_data["failed_methods"]
        total = question_data["total_methods"]
        pct = (resolved / total * 100) if total > 0 else 0.0
        
        print(f"{status} {question_id}: {resolved}/{total} methods ({pct:.1f}%)")
    
    if report["failed_methods"]:
        print()
        print("Failed Methods (Sample):")
        print("-" * 80)
        # Show up to 10 failed methods
        for failed in report["failed_methods"][:10]:
            print(f"  - {failed['question']}: {failed['error'][:80]}...")
        
        if len(report["failed_methods"]) > 10:
            print(f"  ... and {len(report['failed_methods']) - 10} more")


if __name__ == "__main__":
    sys.exit(main())
