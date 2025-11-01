"""
Test suite for import validation
=================================

This test verifies that all imports work correctly across the system.
"""

import sys
import importlib
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def test_core_compatibility_shims():
    """Test that all core compatibility shims can be imported"""
    shims = [
        "aggregation",
        "contracts",
        "evidence_registry",
        "json_contract_loader",
        "macro_prompts",
        "meso_cluster_analysis",
        "orchestrator",
        "qmcm_hooks",
        "recommendation_engine",
        "runtime_error_fixes",
        "seed_factory",
        "signature_validator",
    ]
    
    for shim in shims:
        try:
            importlib.import_module(shim)
        except Exception as e:
            raise AssertionError(f"Failed to import {shim}: {e}")


def test_core_packages():
    """Test that all core packages can be imported"""
    packages = [
        "saaaaaa",
        "saaaaaa.core",
        "saaaaaa.processing",
        "saaaaaa.analysis",
        "saaaaaa.utils",
        "saaaaaa.concurrency",
        "saaaaaa.api",
        "saaaaaa.infrastructure",
        "saaaaaa.controls",
    ]
    
    for package in packages:
        try:
            importlib.import_module(package)
        except Exception as e:
            raise AssertionError(f"Failed to import {package}: {e}")


def test_qmcm_hooks_backward_compatibility():
    """Test that qmcm_hooks has backward-compatible aliases"""
    import qmcm_hooks
    
    # Check that both old and new names work
    assert hasattr(qmcm_hooks, 'qmcm_record')
    assert hasattr(qmcm_hooks, 'record_qmcm_call')
    assert hasattr(qmcm_hooks, 'QMCMRecorder')
    assert hasattr(qmcm_hooks, 'get_global_recorder')
    
    # Verify the alias works
    assert qmcm_hooks.record_qmcm_call is qmcm_hooks.qmcm_record


def test_signature_validator_backward_compatibility():
    """Test that signature_validator has backward-compatible aliases"""
    import signature_validator
    
    # Check that both old and new names work
    assert hasattr(signature_validator, 'SignatureMismatch')
    assert hasattr(signature_validator, 'SignatureIssue')
    assert hasattr(signature_validator, 'ValidationIssue')
    assert hasattr(signature_validator, 'validate_signature')
    assert hasattr(signature_validator, 'validate_call_signature')
    
    # Verify the aliases work
    assert signature_validator.SignatureIssue is signature_validator.SignatureMismatch
    assert signature_validator.ValidationIssue is signature_validator.SignatureMismatch


def test_contracts_exports():
    """Test that contracts module exports expected symbols"""
    import contracts
    
    expected_exports = [
        "AnalysisInputV1",
        "AnalysisOutputV1",
        "ContractMismatchError",
        "validate_contract",
        "SeedFactory",
    ]
    
    for export in expected_exports:
        assert hasattr(contracts, export), f"Missing export: {export}"


def test_aggregation_exports():
    """Test that aggregation module exports expected symbols"""
    import aggregation
    
    expected_exports = [
        "MacroAggregator",
        "ClusterAggregator",
        "DimensionAggregator",
        "AreaPolicyAggregator",
    ]
    
    for export in expected_exports:
        assert hasattr(aggregation, export), f"Missing export: {export}"


if __name__ == "__main__":
    # Run tests manually if pytest is not available
    print("Running import tests...")
    
    tests = [
        ("Core compatibility shims", test_core_compatibility_shims),
        ("Core packages", test_core_packages),
        ("QMCM hooks backward compatibility", test_qmcm_hooks_backward_compatibility),
        ("Signature validator backward compatibility", test_signature_validator_backward_compatibility),
        ("Contracts exports", test_contracts_exports),
        ("Aggregation exports", test_aggregation_exports),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)
