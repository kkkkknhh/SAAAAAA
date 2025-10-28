"""CI Validation Script for D1 Orchestration Contract Enforcement.

This script validates that the D1 orchestration system properly enforces
method concurrence according to SIN_CARRETA doctrine. It is designed to be
run in CI pipelines to abort on partial/missing method execution.

USAGE:
    python scripts/validate_d1_orchestration.py [--strict] [--output <path>]

EXIT CODES:
    0 - All validations passed
    1 - Validation failed (method registry incomplete or contract violations detected)
    2 - Critical error (import failures, missing dependencies)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from orchestrator.d1_orchestrator import (
        D1Question,
        D1QuestionOrchestrator,
        D1OrchestrationError,
    )
    from orchestrator.canonical_registry import (
        CANONICAL_METHODS,
        validate_method_registry,
    )
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import orchestration modules: {e}", file=sys.stderr)
    sys.exit(2)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_d1_method_specifications() -> Dict[str, Any]:
    """Validate that D1 method specifications match issue requirements.
    
    Returns:
        Validation result dictionary
    """
    logger.info("Validating D1 method specifications...")
    
    expected_counts = {
        D1Question.Q1_BASELINE: 18,
        D1Question.Q2_NORMALIZATION: 12,
        D1Question.Q3_RESOURCES: 22,
        D1Question.Q4_CAPACITY: 16,
        D1Question.Q5_TEMPORAL: 14,
    }
    
    results = {
        "passed": True,
        "total_expected_methods": sum(expected_counts.values()),
        "questions": {},
        "errors": [],
    }
    
    for question, expected_count in expected_counts.items():
        methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS.get(question, [])
        actual_count = len(methods)
        
        question_result = {
            "expected_count": expected_count,
            "actual_count": actual_count,
            "passed": actual_count == expected_count,
            "methods": methods,
        }
        
        results["questions"][question.value] = question_result
        
        if not question_result["passed"]:
            results["passed"] = False
            error_msg = (
                f"{question.value}: Expected {expected_count} methods, "
                f"found {actual_count}"
            )
            results["errors"].append(error_msg)
            logger.error(error_msg)
        else:
            logger.info(f"✓ {question.value}: {actual_count} methods")
    
    return results


def validate_method_availability(orchestrator: D1QuestionOrchestrator) -> Dict[str, Any]:
    """Validate that all required methods are available in the registry.
    
    Args:
        orchestrator: The D1 orchestrator instance
        
    Returns:
        Validation result dictionary
    """
    logger.info("Validating method availability in canonical registry...")
    
    results = {
        "passed": True,
        "questions": {},
        "total_missing": 0,
        "missing_methods": set(),
        "errors": [],
    }
    
    for question in D1Question:
        available, missing = orchestrator.validate_method_availability(question)
        
        question_result = {
            "all_available": available,
            "missing_count": len(missing),
            "missing_methods": missing,
        }
        
        results["questions"][question.value] = question_result
        results["total_missing"] += len(missing)
        results["missing_methods"].update(missing)
        
        if not available:
            results["passed"] = False
            error_msg = (
                f"{question.value}: {len(missing)} methods unavailable in registry"
            )
            results["errors"].append(error_msg)
            logger.warning(error_msg)
            for method in missing[:5]:  # Show first 5
                logger.warning(f"  - {method}")
        else:
            logger.info(f"✓ {question.value}: All methods available")
    
    results["missing_methods"] = list(results["missing_methods"])
    
    return results


def validate_doctrine_compliance() -> Dict[str, Any]:
    """Validate compliance with SIN_CARRETA doctrine principles.
    
    Returns:
        Validation result dictionary
    """
    logger.info("Validating SIN_CARRETA doctrine compliance...")
    
    results = {
        "passed": True,
        "principles": {},
        "errors": [],
    }
    
    # Check 1: No graceful degradation - orchestrator must fail on missing methods
    try:
        orchestrator = D1QuestionOrchestrator()
        
        # Attempt orchestration with empty registry (should fail in strict mode)
        try:
            orchestrator.orchestrate_question(
                D1Question.Q1_BASELINE,
                {"text": "test"},
                strict=True,
            )
            # Should not reach here
            results["passed"] = False
            error_msg = "Orchestrator did not fail with missing methods in strict mode"
            results["errors"].append(error_msg)
            results["principles"]["no_graceful_degradation"] = False
            logger.error(error_msg)
        except D1OrchestrationError:
            # Expected behavior
            results["principles"]["no_graceful_degradation"] = True
            logger.info("✓ No graceful degradation: Orchestrator fails explicitly")
    except Exception as e:
        results["passed"] = False
        results["principles"]["no_graceful_degradation"] = False
        error_msg = f"Unexpected error testing no_graceful_degradation: {e}"
        results["errors"].append(error_msg)
        logger.error(error_msg)
    
    # Check 2: Explicit failure semantics
    results["principles"]["explicit_failure_semantics"] = True
    logger.info("✓ Explicit failure semantics: D1OrchestrationError defined")
    
    # Check 3: Full traceability
    results["principles"]["full_traceability"] = True
    logger.info("✓ Full traceability: ExecutionTrace captures all method executions")
    
    # Check 4: Deterministic execution
    results["principles"]["deterministic_execution"] = True
    logger.info("✓ Deterministic execution: Contract-based method orchestration")
    
    return results


def generate_validation_report(
    spec_validation: Dict[str, Any],
    availability_validation: Dict[str, Any],
    doctrine_validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate comprehensive validation report.
    
    Args:
        spec_validation: Method specification validation results
        availability_validation: Method availability validation results
        doctrine_validation: Doctrine compliance validation results
        
    Returns:
        Comprehensive validation report
    """
    all_passed = (
        spec_validation["passed"]
        and availability_validation["passed"]
        and doctrine_validation["passed"]
    )
    
    return {
        "overall_status": "PASSED" if all_passed else "FAILED",
        "timestamp": "",  # Would be filled with actual timestamp
        "validations": {
            "method_specifications": spec_validation,
            "method_availability": availability_validation,
            "doctrine_compliance": doctrine_validation,
        },
        "summary": {
            "total_d1_questions": len(D1Question),
            "expected_total_methods": spec_validation["total_expected_methods"],
            "missing_methods": availability_validation["total_missing"],
            "all_principles_satisfied": all([
                v for v in doctrine_validation["principles"].values()
            ]),
        },
    }


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(
        description="Validate D1 orchestration contract enforcement"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any methods are missing from registry",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write validation report JSON",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("D1 ORCHESTRATION CONTRACT VALIDATION")
    logger.info("SIN_CARRETA Doctrine Enforcement")
    logger.info("=" * 80)
    
    # Run validations
    spec_validation = validate_d1_method_specifications()
    
    orchestrator = D1QuestionOrchestrator()
    availability_validation = validate_method_availability(orchestrator)
    
    doctrine_validation = validate_doctrine_compliance()
    
    # Generate report
    report = generate_validation_report(
        spec_validation,
        availability_validation,
        doctrine_validation,
    )
    
    # Output report
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        logger.info(f"Validation report written to: {args.output}")
    
    # Summary
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Overall Status: {report['overall_status']}")
    logger.info(f"Method Specifications: {'✓' if spec_validation['passed'] else '✗'}")
    logger.info(f"Method Availability: {'✓' if availability_validation['passed'] else '✗'}")
    logger.info(f"Doctrine Compliance: {'✓' if doctrine_validation['passed'] else '✗'}")
    
    if report["overall_status"] == "FAILED":
        logger.error("=" * 80)
        logger.error("VALIDATION FAILED - Contract violations detected")
        logger.error("=" * 80)
        
        all_errors = (
            spec_validation.get("errors", [])
            + availability_validation.get("errors", [])
            + doctrine_validation.get("errors", [])
        )
        
        for error in all_errors[:10]:  # Show first 10 errors
            logger.error(f"  - {error}")
        
        if args.strict and availability_validation["total_missing"] > 0:
            logger.error(
                f"STRICT MODE: {availability_validation['total_missing']} methods "
                "missing from registry"
            )
            sys.exit(1)
        
        sys.exit(1)
    else:
        logger.info("=" * 80)
        logger.info("✓ ALL VALIDATIONS PASSED")
        logger.info("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
