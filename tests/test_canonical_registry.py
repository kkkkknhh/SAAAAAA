"""Tests for canonical registry validation and audit generation."""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.canonical_registry import (
    _count_all_methods,
    _validate_method_registry,
    generate_audit_report,
    validate_method_registry,
    RegistryValidationError,
    _PROVISIONAL_METHOD_THRESHOLD,
    _MINIMUM_METHOD_THRESHOLD,
)


def test_count_all_methods():
    """Test that we can count methods from the class map."""
    total = _count_all_methods()
    assert total > 0, "Should count at least some methods"
    print(f"✓ Total methods counted: {total}")


def test_validate_provisional_threshold():
    """Test validation with provisional threshold (≥400)."""
    total_methods = 416  # Current value from COMPLETE_METHOD_CLASS_MAP.json
    resolved_methods = 10
    
    # Should pass with provisional=True since 416 >= 400
    result = _validate_method_registry(total_methods, resolved_methods, provisional=True)
    assert result["passed"], "Should pass provisional threshold"
    assert result["threshold"] == _PROVISIONAL_METHOD_THRESHOLD
    print(f"✓ Provisional validation passed: {total_methods} >= {_PROVISIONAL_METHOD_THRESHOLD}")


def test_validate_strict_threshold():
    """Test validation with strict threshold (≥555)."""
    total_methods = 416  # Current value
    resolved_methods = 10
    
    # Should fail with provisional=False since 416 < 555
    try:
        result = _validate_method_registry(total_methods, resolved_methods, provisional=False)
        assert False, "Should have raised RegistryValidationError"
    except RegistryValidationError as e:
        assert "555" in str(e)
        print(f"✓ Strict validation correctly failed: {total_methods} < {_MINIMUM_METHOD_THRESHOLD}")


def test_audit_report_structure():
    """Test that audit report has the expected structure."""
    # Generate audit with empty registry for testing
    audit_path = Path(__file__).parent.parent / "test_audit.json"
    audit = generate_audit_report({}, audit_path)
    
    # Check required keys
    assert "metadata" in audit
    assert "coverage" in audit
    assert "validation" in audit
    assert "missing" in audit
    assert "extras" in audit
    assert "class_coverage" in audit
    
    # Check validation has both modes
    assert "provisional" in audit["validation"]
    assert "strict" in audit["validation"]
    
    # Check coverage metrics
    assert "total_methods_in_codebase" in audit["coverage"]
    assert "declared_in_metadata" in audit["coverage"]
    assert "successfully_resolved" in audit["coverage"]
    
    # Clean up
    if audit_path.exists():
        audit_path.unlink()
    
    print("✓ Audit report has correct structure")


def test_yaml_loader():
    """Test that YAML loader works if file exists."""
    from orchestrator.canonical_registry import _load_class_module_map
    
    root = Path(__file__).parent.parent
    yaml_path = root / "method_class_map.yaml"
    
    if yaml_path.exists():
        mapping = _load_class_module_map(root / "COMPLETE_METHOD_CLASS_MAP.json")
        assert len(mapping) > 0, "Should load class-to-module mappings"
        
        # Check that we have some expected classes
        assert "BayesianEvidenceScorer" in mapping or len(mapping) > 50
        print(f"✓ Loaded {len(mapping)} class-to-module mappings")
    else:
        print("⊘ Skipping YAML loader test (file not found)")


if __name__ == "__main__":
    print("Running canonical registry tests...\n")
    
    try:
        test_count_all_methods()
        test_validate_provisional_threshold()
        test_validate_strict_threshold()
        test_audit_report_structure()
        test_yaml_loader()
        
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
