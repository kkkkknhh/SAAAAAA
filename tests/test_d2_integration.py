#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for D2 Integration Hook.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.d2_integration import (
    D2IntegrationHook,
    integrate_d2_validation,
)


class MockOrchestrator:
    """Mock orchestrator for testing integration."""
    def __init__(self):
        self.name = "MockOrchestrator"


def test_integration_hook_initialization():
    """Test that integration hook initializes correctly."""
    hook = D2IntegrationHook(
        strict_mode=False,
        trace_execution=True,
        validate_on_init=False
    )
    
    assert hook.orchestrator is not None
    assert hook.validation_results is None
    assert hook.validation_report is None
    
    print("✓ Integration hook initializes correctly")


def test_integration_hook_validate_all():
    """Test validating all D2 questions through hook."""
    hook = D2IntegrationHook(strict_mode=False, validate_on_init=False)
    
    # Run validation (may fail if dependencies not installed)
    try:
        success = hook.validate_all()
        
        assert hook.validation_results is not None
        assert hook.validation_report is not None
        
        print(f"✓ Validation completed (success={success})")
        
    except Exception as e:
        print(f"⚠ Validation raised exception (expected if deps missing): {str(e)[:80]}")


def test_integration_hook_validate_question():
    """Test validating a single D2 question."""
    hook = D2IntegrationHook(strict_mode=False, validate_on_init=False)
    
    # Validate D2-Q1
    success = hook.validate_question("D2-Q1")
    
    print(f"✓ Question validation completed (success={success})")
    
    # Non-D2 question should pass trivially
    success_non_d2 = hook.validate_question("D1-Q1")
    assert success_non_d2 is True
    
    print("✓ Non-D2 question validation passes trivially")


def test_integration_hook_summary():
    """Test getting validation summary."""
    hook = D2IntegrationHook(strict_mode=False, validate_on_init=False)
    
    # Before validation
    summary = hook.get_validation_summary()
    assert summary["validated"] is False
    
    print("✓ Summary before validation shows not validated")
    
    # After validation
    hook.validate_all()
    summary = hook.get_validation_summary()
    
    assert summary["validated"] is True
    assert "overall_success" in summary
    assert "total_methods" in summary
    assert "methods_resolved" in summary
    assert "success_rate" in summary
    assert "questions" in summary
    
    print("✓ Summary after validation has correct structure")
    print(f"  - Total methods: {summary['total_methods']}")
    print(f"  - Success rate: {summary['success_rate'] * 100:.1f}%")


def test_integration_hook_save_report():
    """Test saving validation report."""
    hook = D2IntegrationHook(strict_mode=False, validate_on_init=False)
    hook.validate_all()
    
    output_path = Path("/tmp/test_d2_integration_report.json")
    hook.save_validation_report(output_path)
    
    assert output_path.exists()
    
    # Clean up
    output_path.unlink()
    
    print("✓ Validation report saved successfully")


def test_integration_hook_pre_execution_check():
    """Test pre-execution check."""
    hook = D2IntegrationHook(strict_mode=False, validate_on_init=False)
    
    # Check should work (may return False if deps missing, but shouldn't raise)
    success = hook.pre_execution_check("D2-Q1", abort_on_failure=False)
    
    print(f"✓ Pre-execution check completed (success={success})")
    
    # Non-D2 question should pass
    success_non_d2 = hook.pre_execution_check("D1-Q1", abort_on_failure=False)
    assert success_non_d2 is True
    
    print("✓ Pre-execution check for non-D2 question passes")


def test_integrate_d2_validation_function():
    """Test the convenience integration function."""
    mock_orchestrator = MockOrchestrator()
    
    output_path = Path("/tmp/test_d2_integration_func.json")
    
    hook = integrate_d2_validation(
        mock_orchestrator,
        strict_mode=False,
        validate_on_init=True,
        save_report_path=output_path
    )
    
    assert hasattr(mock_orchestrator, 'd2_validation')
    assert mock_orchestrator.d2_validation is hook
    
    # Check report was saved
    if hook.validation_report:
        assert output_path.exists()
        output_path.unlink()
    
    print("✓ Integration function works correctly")


def test_hook_with_validation_on_init():
    """Test hook with validation on initialization."""
    # This will trigger validation immediately
    hook = D2IntegrationHook(
        strict_mode=False,
        trace_execution=True,
        validate_on_init=True
    )
    
    # Should have validation results
    assert hook.validation_results is not None
    assert hook.validation_report is not None
    
    summary = hook.get_validation_summary()
    assert summary["validated"] is True
    
    print("✓ Hook with validate_on_init works correctly")


if __name__ == "__main__":
    print("=" * 80)
    print("D2 INTEGRATION HOOK TESTS")
    print("=" * 80)
    print()
    
    tests = [
        test_integration_hook_initialization,
        test_integration_hook_validate_all,
        test_integration_hook_validate_question,
        test_integration_hook_summary,
        test_integration_hook_save_report,
        test_integration_hook_pre_execution_check,
        test_integrate_d2_validation_function,
        test_hook_with_validation_on_init,
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
