"""Tests for choreographer dispatch system with QMCM integration."""
import sys
from pathlib import Path
import tempfile

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.choreographer_dispatch import (
    ChoreographerDispatcher,
    InvocationContext,
    InvocationResult,
    invoke_method,
    get_global_dispatcher,
)
from qmcm_hooks import QMCMRecorder


def test_invocation_context_creation():
    """Test creating invocation context."""
    ctx = InvocationContext(
        text="sample policy text",
        question_id="D1-Q1",
    )
    
    assert ctx.text == "sample policy text"
    assert ctx.question_id == "D1-Q1"
    assert ctx.data is None
    
    # Test to_dict
    ctx_dict = ctx.to_dict()
    assert ctx_dict["has_text"] is True
    assert ctx_dict["has_data"] is False
    assert ctx_dict["question_id"] == "D1-Q1"
    
    print("✓ Invocation context creation works")


def test_instance_pool():
    """Test instance pool in context."""
    ctx = InvocationContext()
    
    # Create mock instance
    class MockProcessor:
        def process(self):
            return "processed"
    
    instance = MockProcessor()
    ctx.set_instance("MockProcessor", instance)
    
    retrieved = ctx.get_instance("MockProcessor")
    assert retrieved is instance
    assert retrieved.process() == "processed"
    
    print("✓ Instance pool works")


def test_dispatcher_initialization():
    """Test dispatcher initialization."""
    recorder = QMCMRecorder()
    dispatcher = ChoreographerDispatcher(
        registry={},
        qmcm_recorder=recorder,
        enable_evidence_recording=True,
    )
    
    assert dispatcher.registry == {}
    assert dispatcher.qmcm_recorder is recorder
    assert dispatcher.enable_evidence_recording is True
    assert len(dispatcher.evidence_records) == 0
    
    print("✓ Dispatcher initialization works")


def test_fqn_resolution():
    """Test FQN resolution."""
    # Create mock callable
    def mock_method():
        return "result"
    
    registry = {
        "MockClass.mock_method": mock_method
    }
    
    dispatcher = ChoreographerDispatcher(registry=registry)
    
    # Test resolution
    resolved = dispatcher._resolve_fqn("MockClass.mock_method")
    assert resolved is mock_method
    
    # Test missing method
    try:
        dispatcher._resolve_fqn("NonExistent.method")
        assert False, "Should raise ValueError"
    except ValueError as e:
        assert "not found" in str(e)
    
    print("✓ FQN resolution works")


def test_method_signature_inspection():
    """Test method signature inspection."""
    dispatcher = ChoreographerDispatcher(registry={})
    
    # Test bound method
    class TestClass:
        def instance_method(self, arg1, arg2):
            pass
    
    obj = TestClass()
    bound_method = obj.instance_method
    
    sig_info = dispatcher._inspect_method_signature(bound_method)
    assert "params" in sig_info
    assert sig_info["is_bound"] is True
    
    # Test unbound method
    unbound_method = TestClass.instance_method
    sig_info = dispatcher._inspect_method_signature(unbound_method)
    assert "self" in sig_info["params"]
    
    print("✓ Method signature inspection works")


def test_simple_method_invocation():
    """Test invoking a simple method."""
    # Create mock method
    def simple_method():
        return {"result": "success"}
    
    registry = {"SimpleClass.simple_method": simple_method}
    recorder = QMCMRecorder()
    dispatcher = ChoreographerDispatcher(
        registry=registry,
        qmcm_recorder=recorder,
    )
    
    # Invoke method
    result = dispatcher.invoke_method("SimpleClass.simple_method")
    
    assert result.success is True
    assert result.result == {"result": "success"}
    assert result.fqn == "SimpleClass.simple_method"
    assert result.execution_time_ms > 0
    assert result.qmcm_recorded is True
    
    # Check QMCM recorded the call
    stats = recorder.get_statistics()
    assert stats["total_calls"] == 1
    assert "SimpleClass.simple_method" in stats["method_frequency"]
    
    print("✓ Simple method invocation works")


def test_method_invocation_with_context():
    """Test invoking method with context parameters."""
    # Create mock method that takes text parameter
    def process_text(text: str) -> dict:
        return {"processed": text.upper()}
    
    registry = {"TextProcessor.process_text": process_text}
    dispatcher = ChoreographerDispatcher(registry=registry)
    
    # Create context with text
    ctx = InvocationContext(text="hello world")
    
    # Invoke method
    result = dispatcher.invoke_method("TextProcessor.process_text", ctx)
    
    assert result.success is True
    assert result.result == {"processed": "HELLO WORLD"}
    
    print("✓ Method invocation with context works")


def test_evidence_recording():
    """Test evidence recording during invocation."""
    def mock_method():
        return "evidence_data"
    
    registry = {"EvidenceClass.mock_method": mock_method}
    dispatcher = ChoreographerDispatcher(
        registry=registry,
        enable_evidence_recording=True,
    )
    
    # Invoke with evidence recording
    ctx = InvocationContext(question_id="TEST-Q1")
    result = dispatcher.invoke_method("EvidenceClass.mock_method", ctx)
    
    assert result.success is True
    assert len(dispatcher.evidence_records) == 1
    
    evidence = dispatcher.evidence_records[0]
    assert evidence["fqn"] == "EvidenceClass.mock_method"
    assert "timestamp" in evidence
    
    print("✓ Evidence recording works")


def test_invocation_error_handling():
    """Test error handling during invocation."""
    def failing_method():
        raise ValueError("Test error")
    
    registry = {"FailingClass.failing_method": failing_method}
    recorder = QMCMRecorder()
    dispatcher = ChoreographerDispatcher(
        registry=registry,
        qmcm_recorder=recorder,
    )
    
    # Invoke failing method
    result = dispatcher.invoke_method("FailingClass.failing_method")
    
    assert result.success is False
    assert result.error is not None
    assert "Test error" in str(result.error)
    assert result.qmcm_recorded is True
    
    # Check QMCM recorded the error
    stats = recorder.get_statistics()
    assert stats["total_calls"] == 1
    assert stats["success_rate"] == 0.0
    
    print("✓ Error handling works")


def test_invocation_stats():
    """Test getting invocation statistics."""
    def method1():
        return "result1"
    
    def method2():
        return "result2"
    
    registry = {
        "Class1.method1": method1,
        "Class2.method2": method2,
    }
    
    # Use a dedicated recorder for this test
    recorder = QMCMRecorder()
    
    dispatcher = ChoreographerDispatcher(
        registry=registry,
        qmcm_recorder=recorder,
        enable_evidence_recording=True,
    )
    
    # Make some invocations
    dispatcher.invoke_method("Class1.method1")
    dispatcher.invoke_method("Class2.method2")
    dispatcher.invoke_method("Class1.method1")
    
    # Get stats
    stats = dispatcher.get_invocation_stats()
    
    assert stats["registry_size"] == 2
    assert stats["evidence_records"] == 3
    assert stats["qmcm_stats"]["total_calls"] == 3
    assert stats["qmcm_stats"]["method_frequency"]["Class1.method1"] == 2
    
    print("✓ Invocation statistics work")


def test_global_dispatcher():
    """Test global dispatcher singleton."""
    dispatcher1 = get_global_dispatcher()
    dispatcher2 = get_global_dispatcher()
    
    assert dispatcher1 is dispatcher2
    
    print("✓ Global dispatcher singleton works")


def test_convenience_invoke_method():
    """Test convenience invoke_method function."""
    def mock_method():
        return "convenience_result"
    
    # Set up global dispatcher with mock registry
    from orchestrator import choreographer_dispatch
    choreographer_dispatch._global_dispatcher = ChoreographerDispatcher(
        registry={"Mock.method": mock_method}
    )
    
    # Use convenience function
    result = invoke_method("Mock.method")
    
    assert result.success is True
    assert result.result == "convenience_result"
    
    print("✓ Convenience invoke_method works")


def test_invocation_result_to_dict():
    """Test InvocationResult serialization."""
    result = InvocationResult(
        fqn="Test.method",
        success=True,
        result={"data": "value"},
        execution_time_ms=123.45,
        context_summary={"has_text": True},
        qmcm_recorded=True,
    )
    
    result_dict = result.to_dict()
    
    assert result_dict["fqn"] == "Test.method"
    assert result_dict["success"] is True
    assert result_dict["result_type"] == "dict"
    assert result_dict["execution_time_ms"] == 123.45
    assert result_dict["qmcm_recorded"] is True
    
    print("✓ InvocationResult serialization works")


if __name__ == "__main__":
    print("Running choreographer dispatch tests...\n")
    
    try:
        test_invocation_context_creation()
        test_instance_pool()
        test_dispatcher_initialization()
        test_fqn_resolution()
        test_method_signature_inspection()
        test_simple_method_invocation()
        test_method_invocation_with_context()
        test_evidence_recording()
        test_invocation_error_handling()
        test_invocation_stats()
        test_global_dispatcher()
        test_convenience_invoke_method()
        test_invocation_result_to_dict()
        
        print("\n✅ All choreographer dispatch tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
