#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D2 Method Concurrence - Example Usage

This script demonstrates how to use the D2 Activities Orchestrator
for method concurrence validation following SIN_CARRETA doctrine.
"""

import sys
from pathlib import Path

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import (
    D2ActivitiesOrchestrator,
    D2Question,
    validate_d2_orchestration,
    D2IntegrationHook,
)


def example_basic_validation():
    """Example 1: Basic validation using convenience function."""
    print("=" * 80)
    print("EXAMPLE 1: Basic Validation")
    print("=" * 80)
    print()
    
    # Use the convenience function
    success = validate_d2_orchestration(
        strict_mode=False,  # Use False for development
        output_report="/tmp/d2_validation.json"
    )
    
    if success:
        print("\n✅ All D2 methods validated successfully!")
    else:
        print("\n⚠ Some D2 methods failed validation (check report)")
    
    print()


def example_orchestrator_direct():
    """Example 2: Using orchestrator directly for fine-grained control."""
    print("=" * 80)
    print("EXAMPLE 2: Direct Orchestrator Usage")
    print("=" * 80)
    print()
    
    # Initialize orchestrator
    orchestrator = D2ActivitiesOrchestrator(
        strict_mode=False,
        trace_execution=True
    )
    
    # Validate a single question
    print("Validating D2-Q1 (Formato Tabular)...")
    result = orchestrator.validate_method_existence(D2Question.Q1_FORMATO_TABULAR)
    
    print(f"  Total methods: {result.total_methods}")
    print(f"  Resolved: {result.executed_methods - result.failed_methods}")
    print(f"  Failed: {result.failed_methods}")
    print(f"  Success: {result.success}")
    print(f"  Time: {result.execution_time_ms:.2f}ms")
    
    # Validate all questions
    print("\nValidating all D2 questions...")
    results = orchestrator.validate_all_d2_questions()
    
    print(f"\nResults for {len(results)} questions:")
    for question_id, res in results.items():
        status = "✓" if res.success else "✗"
        resolved = res.executed_methods - res.failed_methods
        print(f"  {status} {question_id}: {resolved}/{res.total_methods} methods")
    
    # Generate report
    report = orchestrator.generate_validation_report(results)
    print(f"\nOverall success: {report['summary']['overall_success']}")
    print(f"Success rate: {report['summary']['methods_resolved']}/{report['summary']['total_methods']}")
    
    print()


def example_integration_hook():
    """Example 3: Using integration hook with existing orchestrator."""
    print("=" * 80)
    print("EXAMPLE 3: Integration Hook")
    print("=" * 80)
    print()
    
    # Create integration hook
    hook = D2IntegrationHook(
        strict_mode=False,
        trace_execution=True,
        validate_on_init=True  # Validate immediately
    )
    
    # Get validation summary
    summary = hook.get_validation_summary()
    
    print("Validation Summary:")
    print(f"  Validated: {summary['validated']}")
    print(f"  Overall success: {summary.get('overall_success', 'N/A')}")
    print(f"  Total methods: {summary.get('total_methods', 0)}")
    print(f"  Methods resolved: {summary.get('methods_resolved', 0)}")
    print(f"  Success rate: {summary.get('success_rate', 0) * 100:.1f}%")
    
    # Check individual questions
    print("\nPer-Question Status:")
    for question_id, q_data in summary.get('questions', {}).items():
        status = "✓" if q_data['success'] else "✗"
        print(f"  {status} {question_id}: {q_data['methods_resolved']}/{q_data['total_methods']}")
    
    # Pre-execution check example
    print("\nPre-execution check for D2-Q1:")
    can_execute = hook.pre_execution_check("D2-Q1", abort_on_failure=False)
    print(f"  Can execute: {can_execute}")
    
    # Save report
    hook.save_validation_report(Path("/tmp/d2_integration_report.json"))
    print("\n✓ Validation report saved to /tmp/d2_integration_report.json")
    
    print()


def example_strict_mode():
    """Example 4: Strict mode enforcement (SIN_CARRETA doctrine)."""
    print("=" * 80)
    print("EXAMPLE 4: Strict Mode (SIN_CARRETA Doctrine)")
    print("=" * 80)
    print()
    
    # Note: This will likely fail if dependencies aren't installed
    # But it demonstrates the strict contract enforcement
    
    orchestrator = D2ActivitiesOrchestrator(
        strict_mode=True,  # Strict enforcement
        trace_execution=True
    )
    
    print("Attempting strict validation...")
    print("(This will abort on first missing method)\n")
    
    try:
        # This will raise OrchestrationError if any method is missing
        result = orchestrator.validate_method_existence(D2Question.Q1_FORMATO_TABULAR)
        
        print("✅ Strict validation PASSED!")
        print(f"   All {result.total_methods} methods for D2-Q1 are present")
        
    except Exception as e:
        print("❌ Strict validation FAILED!")
        print(f"   Error: {str(e)[:100]}...")
        print("\nThis is expected behavior in SIN_CARRETA strict mode:")
        print("  - No graceful degradation")
        print("  - Explicit failure on contract violation")
        print("  - Deterministic abort behavior")
    
    print()


def example_method_registry():
    """Example 5: Exploring the method registry."""
    print("=" * 80)
    print("EXAMPLE 5: Method Registry Exploration")
    print("=" * 80)
    print()
    
    from orchestrator import D2MethodRegistry
    
    print("D2 Questions and Method Counts:")
    for question in D2Question:
        methods = D2MethodRegistry.get_methods_for_question(question)
        print(f"\n{question.value}: {len(methods)} methods")
        
        # Group by module
        by_module = {}
        for method in methods:
            if method.module_name not in by_module:
                by_module[method.module_name] = []
            by_module[method.module_name].append(method)
        
        for module, module_methods in sorted(by_module.items()):
            print(f"  - {module}: {len(module_methods)} methods")
    
    # All unique methods
    all_methods = D2MethodRegistry.get_all_d2_methods()
    print(f"\nTotal unique methods across D2: {len(all_methods)}")
    
    # Total method calls (with duplicates)
    total_calls = sum(
        len(D2MethodRegistry.get_methods_for_question(q))
        for q in D2Question
    )
    print(f"Total method calls (with reuse): {total_calls}")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("D2 METHOD CONCURRENCE - USAGE EXAMPLES")
    print("SIN_CARRETA Doctrine Implementation")
    print("=" * 80)
    print("\n")
    
    examples = [
        ("Basic Validation", example_basic_validation),
        ("Direct Orchestrator", example_orchestrator_direct),
        ("Integration Hook", example_integration_hook),
        ("Strict Mode", example_strict_mode),
        ("Method Registry", example_method_registry),
    ]
    
    for i, (name, example_func) in enumerate(examples, 1):
        print(f"\n{'=' * 80}")
        print(f"Running Example {i}/{len(examples)}: {name}")
        print(f"{'=' * 80}\n")
        
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠ Example raised exception: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Review the output above")
    print("  2. Check generated reports in /tmp/")
    print("  3. Read full documentation: docs/D2_METHOD_CONCURRENCE.md")
    print("  4. Integrate into your orchestrator using D2IntegrationHook")
    print()


if __name__ == "__main__":
    main()
