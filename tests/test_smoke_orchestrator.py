"""Smoke tests for the orchestrator package migration.

This test validates that:
1. All modules can be imported without errors
2. No import cycles exist
3. The provider boundary guard works
4. Basic orchestrator instantiation works
5. Health check and basic methods function
"""

import pytest


def test_import_orchestrator_core():
    """Test that orchestrator.core can be imported."""
    from orchestrator import core
    assert hasattr(core, 'Orchestrator')
    assert hasattr(core, 'MethodExecutor')
    assert hasattr(core, 'PreprocessedDocument')
    assert hasattr(core, 'Evidence')
    assert hasattr(core, 'AbortSignal')
    assert hasattr(core, 'AbortRequested')
    assert hasattr(core, 'ResourceLimits')
    assert hasattr(core, 'PhaseInstrumentation')
    assert hasattr(core, 'PhaseResult')
    assert hasattr(core, 'MicroQuestionRun')
    assert hasattr(core, 'ScoredMicroQuestion')


def test_import_orchestrator_executors():
    """Test that orchestrator.executors can be imported."""
    from orchestrator import executors
    assert hasattr(executors, 'DataFlowExecutor')
    assert hasattr(executors, 'D1Q1_Executor')
    assert hasattr(executors, 'D6Q5_Executor')
    # Verify all 30 executors exist
    for dim in range(1, 7):
        for q in range(1, 6):
            executor_name = f'D{dim}Q{q}_Executor'
            assert hasattr(executors, executor_name), f"Missing {executor_name}"


def test_import_orchestrator_package():
    """Test that orchestrator package exports work."""
    import orchestrator
    assert hasattr(orchestrator, 'Orchestrator')
    assert hasattr(orchestrator, 'MethodExecutor')
    assert hasattr(orchestrator, 'PreprocessedDocument')
    assert hasattr(orchestrator, 'get_questionnaire_provider')
    assert hasattr(orchestrator, 'get_questionnaire_payload')
    assert hasattr(orchestrator, 'EvidenceRegistry')
    assert hasattr(orchestrator, 'JSONContractLoader')


def test_no_import_cycles():
    """Test that importing the package doesn't create cycles."""
    import sys
    import importlib
    
    # Clear any cached imports
    modules_to_clear = [m for m in sys.modules if m.startswith('orchestrator')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    # Import fresh
    import orchestrator
    from orchestrator import core, executors
    
    # Should complete without errors
    assert orchestrator is not None
    assert core is not None
    assert executors is not None


def test_provider_boundary_guard():
    """Test that provider boundary guard restricts access."""
    from orchestrator import get_questionnaire_payload
    
    # This should raise RuntimeError when called from tests (outside orchestrator package)
    with pytest.raises(RuntimeError, match="Questionnaire provider access restricted"):
        get_questionnaire_payload()


def test_provider_boundary_guard_allows_orchestrator():
    """Test that provider allows access from orchestrator package."""
    # Create a test module within orchestrator namespace
    import sys
    from types import ModuleType
    
    # Mock a module inside orchestrator package
    test_module = ModuleType('orchestrator.test_internal')
    test_module.__name__ = 'orchestrator.test_internal'
    sys.modules['orchestrator.test_internal'] = test_module
    
    # Define a function in that module
    code = """
from orchestrator import get_questionnaire_payload

def internal_access():
    try:
        return get_questionnaire_payload()
    except FileNotFoundError:
        # File might not exist in test environment
        return None
"""
    exec(code, test_module.__dict__)
    
    # This should NOT raise RuntimeError (but might raise FileNotFoundError)
    try:
        result = test_module.internal_access()
        # Either succeeds or raises FileNotFoundError, both are OK
        assert True
    except RuntimeError as e:
        if "restricted" in str(e):
            pytest.fail(f"Provider should allow access from orchestrator package: {e}")
        raise
    except FileNotFoundError:
        # Expected if questionnaire_monolith.json doesn't exist
        pass
    finally:
        # Cleanup
        del sys.modules['orchestrator.test_internal']


def test_orchestrator_instantiation():
    """Test that Orchestrator can be instantiated."""
    from orchestrator import Orchestrator
    
    # Should be able to create instance (might fail on file access, that's OK)
    try:
        orch = Orchestrator()
        assert orch is not None
        assert hasattr(orch, 'health_check')
        assert hasattr(orch, 'request_abort')
        assert hasattr(orch, 'reset_abort')
    except (FileNotFoundError, TypeError):
        # Expected if catalog or monolith files don't exist or paths resolve to None
        pytest.skip("Orchestrator files not available")


def test_orchestrator_health_check():
    """Test that health_check method works."""
    from orchestrator import Orchestrator
    
    try:
        orch = Orchestrator()
        health = orch.health_check()
        assert isinstance(health, dict)
        assert 'score' in health
        assert 'resource_usage' in health
        assert 'abort' in health
    except (FileNotFoundError, TypeError):
        # Expected if catalog or monolith files don't exist or paths resolve to None
        pytest.skip("Orchestrator files not available")


def test_abort_signal():
    """Test abort signal functionality."""
    from orchestrator import AbortSignal, AbortRequested
    
    signal = AbortSignal()
    assert not signal.is_aborted()
    
    signal.abort("Test abort")
    assert signal.is_aborted()
    assert signal.get_reason() == "Test abort"
    
    signal.reset()
    assert not signal.is_aborted()


def test_preprocessed_document():
    """Test PreprocessedDocument dataclass."""
    from orchestrator import PreprocessedDocument
    
    doc = PreprocessedDocument(
        document_id="test_doc",
        raw_text="Test content",
        sentences=[],
        tables=[],
        metadata={"source": "test"}
    )
    
    assert doc.document_id == "test_doc"
    assert doc.raw_text == "Test content"
    assert doc.metadata["source"] == "test"


def test_evidence_dataclass():
    """Test Evidence dataclass."""
    from orchestrator import Evidence
    
    evidence = Evidence(
        modality="TYPE_A",
        elements=["element1", "element2"],
        raw_results={"key": "value"}
    )
    
    assert evidence.modality == "TYPE_A"
    assert len(evidence.elements) == 2
    assert evidence.raw_results["key"] == "value"


def test_resource_limits():
    """Test ResourceLimits class."""
    from orchestrator import ResourceLimits
    
    limits = ResourceLimits(
        max_memory_mb=1024.0,
        max_cpu_percent=80.0
    )
    
    assert limits.max_memory_mb == 1024.0
    assert limits.max_cpu_percent == 80.0
    
    # Test resource usage check
    usage = limits.get_resource_usage()
    assert isinstance(usage, dict)


def test_method_executor():
    """Test MethodExecutor instantiation."""
    from orchestrator import MethodExecutor
    
    # MethodExecutor may fail if dependencies are missing
    try:
        executor = MethodExecutor()
        assert executor is not None
        assert hasattr(executor, 'execute')
    except (ModuleNotFoundError, SystemExit, ImportError):
        # Expected if dependencies like 'fitz' are missing
        pytest.skip("MethodExecutor dependencies not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
