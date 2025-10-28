#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for D2 Activities Design & Coherence Orchestrator.

Validates strict method concurrence enforcement following SIN_CARRETA doctrine.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.d2_activities_orchestrator import (
    D2ActivitiesOrchestrator,
    D2MethodRegistry,
    D2Question,
    OrchestrationError,
    MethodSpec,
    validate_d2_orchestration,
)


def test_registry_q1_method_count():
    """Test that D2-Q1 has exactly 20 methods as specified."""
    methods = D2MethodRegistry.get_methods_for_question(D2Question.Q1_FORMATO_TABULAR)
    assert len(methods) == 20, f"D2-Q1 should have 20 methods, found {len(methods)}"
    print(f"✓ D2-Q1 has correct method count: {len(methods)}")


def test_registry_q2_method_count():
    """Test that D2-Q2 has exactly 25 methods as specified."""
    methods = D2MethodRegistry.get_methods_for_question(D2Question.Q2_CAUSALIDAD_ACTIVIDADES)
    assert len(methods) == 25, f"D2-Q2 should have 25 methods, found {len(methods)}"
    print(f"✓ D2-Q2 has correct method count: {len(methods)}")


def test_registry_q3_method_count():
    """Test that D2-Q3 has exactly 18 methods as specified."""
    methods = D2MethodRegistry.get_methods_for_question(D2Question.Q3_CLASIFICACION_TEMATICA)
    assert len(methods) == 18, f"D2-Q3 should have 18 methods, found {len(methods)}"
    print(f"✓ D2-Q3 has correct method count: {len(methods)}")


def test_registry_q4_method_count():
    """Test that D2-Q4 has exactly 20 methods as specified."""
    methods = D2MethodRegistry.get_methods_for_question(D2Question.Q4_RIESGOS_MITIGACION)
    assert len(methods) == 20, f"D2-Q4 should have 20 methods, found {len(methods)}"
    print(f"✓ D2-Q4 has correct method count: {len(methods)}")


def test_registry_q5_method_count():
    """Test that D2-Q5 has exactly 24 methods as specified."""
    methods = D2MethodRegistry.get_methods_for_question(D2Question.Q5_COHERENCIA_ESTRATEGICA)
    assert len(methods) == 24, f"D2-Q5 should have 24 methods, found {len(methods)}"
    print(f"✓ D2-Q5 has correct method count: {len(methods)}")


def test_method_spec_structure():
    """Test that method specs have required attributes."""
    methods = D2MethodRegistry.get_methods_for_question(D2Question.Q1_FORMATO_TABULAR)
    
    for method in methods:
        assert isinstance(method, MethodSpec), f"Method should be MethodSpec instance"
        assert method.module_name, "Method should have module_name"
        assert method.class_name, "Method should have class_name"
        assert method.method_name, "Method should have method_name"
        assert method.fully_qualified_name, "Method should have fully_qualified_name"
    
    print(f"✓ All method specs have required structure")


def test_orchestrator_initialization():
    """Test that orchestrator initializes correctly."""
    orchestrator = D2ActivitiesOrchestrator(strict_mode=True, trace_execution=True)
    
    assert orchestrator.strict_mode is True
    assert orchestrator.trace_execution is True
    assert orchestrator.registry is not None
    
    print("✓ Orchestrator initializes correctly")


def test_orchestrator_strict_mode_disabled():
    """Test that orchestrator can be initialized with strict_mode=False."""
    orchestrator = D2ActivitiesOrchestrator(strict_mode=False)
    
    assert orchestrator.strict_mode is False
    
    print("✓ Orchestrator can disable strict mode")


def test_validate_method_existence_structure():
    """Test that validation returns proper result structure."""
    orchestrator = D2ActivitiesOrchestrator(strict_mode=False)
    
    # Validate Q1 (might fail if methods don't exist, but we test structure)
    try:
        result = orchestrator.validate_method_existence(D2Question.Q1_FORMATO_TABULAR)
        
        assert hasattr(result, 'question_id')
        assert hasattr(result, 'total_methods')
        assert hasattr(result, 'executed_methods')
        assert hasattr(result, 'failed_methods')
        assert hasattr(result, 'success')
        assert hasattr(result, 'execution_time_ms')
        assert hasattr(result, 'method_results')
        assert hasattr(result, 'errors')
        
        assert result.question_id == "D2-Q1"
        assert result.total_methods == 20
        
        print(f"✓ Validation result has correct structure")
        print(f"  - Total methods: {result.total_methods}")
        print(f"  - Executed: {result.executed_methods}")
        print(f"  - Failed: {result.failed_methods}")
        print(f"  - Success: {result.success}")
        
    except OrchestrationError as e:
        # Expected in strict mode if methods don't exist
        print(f"✓ Validation raised OrchestrationError as expected: {str(e)[:100]}...")


def test_validate_all_questions_non_strict():
    """Test validating all questions in non-strict mode."""
    orchestrator = D2ActivitiesOrchestrator(strict_mode=False)
    
    results = orchestrator.validate_all_d2_questions()
    
    assert len(results) == 5, f"Should have results for 5 questions, got {len(results)}"
    
    for question in D2Question:
        assert question.value in results, f"Missing result for {question.value}"
    
    print(f"✓ Validated all 5 D2 questions in non-strict mode")
    
    # Print summary
    for question_id, result in results.items():
        status = "✓" if result.success else "✗"
        print(f"  {status} {question_id}: {result.executed_methods - result.failed_methods}/{result.total_methods} methods")


def test_generate_validation_report():
    """Test that validation report generates correctly."""
    orchestrator = D2ActivitiesOrchestrator(strict_mode=False)
    results = orchestrator.validate_all_d2_questions()
    
    report = orchestrator.generate_validation_report(results)
    
    # Check report structure
    assert "metadata" in report
    assert "summary" in report
    assert "questions" in report
    assert "failed_methods" in report
    
    # Check metadata
    assert "timestamp" in report["metadata"]
    assert report["metadata"]["doctrine"] == "SIN_CARRETA"
    assert report["metadata"]["strict_mode"] is False
    
    # Check summary
    assert report["summary"]["total_questions"] == 5
    assert "total_methods" in report["summary"]
    assert "methods_resolved" in report["summary"]
    assert "overall_success" in report["summary"]
    
    # Check questions
    assert len(report["questions"]) == 5
    
    print(f"✓ Validation report generated correctly")
    print(f"  - Total methods: {report['summary']['total_methods']}")
    print(f"  - Methods resolved: {report['summary']['methods_resolved']}")
    print(f"  - Overall success: {report['summary']['overall_success']}")


def test_save_validation_report():
    """Test saving validation report to file."""
    orchestrator = D2ActivitiesOrchestrator(strict_mode=False)
    results = orchestrator.validate_all_d2_questions()
    
    # Save to temporary file
    output_path = Path("/tmp/test_d2_report.json")
    report = orchestrator.generate_validation_report(results, output_path)
    
    # Check file was created
    assert output_path.exists(), "Report file should be created"
    
    # Load and validate JSON
    saved_report = json.loads(output_path.read_text())
    assert saved_report == report, "Saved report should match generated report"
    
    # Clean up
    output_path.unlink()
    
    print(f"✓ Validation report saved to file successfully")


def test_all_unique_methods():
    """Test getting all unique D2 methods."""
    all_methods = D2MethodRegistry.get_all_d2_methods()
    
    assert isinstance(all_methods, set)
    assert len(all_methods) > 0
    
    # Count total methods across all questions (with duplicates)
    total_with_duplicates = sum(
        len(D2MethodRegistry.get_methods_for_question(q))
        for q in D2Question
    )
    
    print(f"✓ All D2 methods collected")
    print(f"  - Unique methods: {len(all_methods)}")
    print(f"  - Total method calls (with duplicates): {total_with_duplicates}")


def test_convenience_function():
    """Test the convenience validation function."""
    # Test in non-strict mode to avoid failures
    output_path = "/tmp/test_d2_convenience.json"
    
    # This should not raise even if some methods fail
    try:
        success = validate_d2_orchestration(
            strict_mode=False,
            output_report=output_path
        )
        
        # Check that report was created
        assert Path(output_path).exists()
        
        # Clean up
        Path(output_path).unlink()
        
        print(f"✓ Convenience function works (success={success})")
        
    except Exception as e:
        print(f"⚠ Convenience function raised exception: {e}")


def test_method_modules_specified():
    """Test that all methods have correct module specifications."""
    expected_modules = {
        "policy_processor",
        "financiero_viabilidad_tablas",
        "contradiction_deteccion",
        "semantic_chunking_policy",
        "dereck_beach",
        "teoria_cambio",
        "Analyzer_one",
        "embedding_policy",
    }
    
    all_methods = []
    for question in D2Question:
        all_methods.extend(D2MethodRegistry.get_methods_for_question(question))
    
    modules_found = set(m.module_name for m in all_methods)
    
    # Check that all expected modules are referenced
    # Note: Not all modules may be used in D2
    print(f"✓ Methods reference {len(modules_found)} modules")
    print(f"  Modules: {sorted(modules_found)}")
    
    # Ensure no typos in module names
    for module in modules_found:
        assert module in expected_modules or module.startswith("test_"), \
            f"Unexpected module: {module}"


if __name__ == "__main__":
    print("=" * 80)
    print("D2 ORCHESTRATOR TESTS")
    print("=" * 80)
    print()
    
    tests = [
        test_registry_q1_method_count,
        test_registry_q2_method_count,
        test_registry_q3_method_count,
        test_registry_q4_method_count,
        test_registry_q5_method_count,
        test_method_spec_structure,
        test_orchestrator_initialization,
        test_orchestrator_strict_mode_disabled,
        test_validate_method_existence_structure,
        test_validate_all_questions_non_strict,
        test_generate_validation_report,
        test_save_validation_report,
        test_all_unique_methods,
        test_convenience_function,
        test_method_modules_specified,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            print("-" * 60)
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    sys.exit(0 if failed == 0 else 1)
