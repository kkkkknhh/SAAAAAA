"""Tests for D1 Orchestrator - Method Concurrence Enforcement.

This test suite validates that the D1 orchestrator correctly enforces
SIN_CARRETA doctrine for method concurrence and contract compliance.
"""

import unittest
from unittest.mock import Mock, patch
from typing import Dict, Any

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.d1_orchestrator import (
    D1Question,
    D1QuestionOrchestrator,
    D1OrchestrationError,
    ExecutionTrace,
    MethodContract,
    OrchestrationResult,
)


class TestD1QuestionOrchestrator(unittest.TestCase):
    """Test cases for D1QuestionOrchestrator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_registry = {}
        self.orchestrator = D1QuestionOrchestrator(canonical_registry=self.mock_registry)
    
    def test_d1_method_specifications_complete(self):
        """Test that all D1 questions have method specifications."""
        for question in D1Question:
            self.assertIn(
                question,
                D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS,
                f"Missing method specification for {question.value}",
            )
            methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[question]
            self.assertGreater(
                len(methods),
                0,
                f"No methods specified for {question.value}",
            )
    
    def test_d1_q1_has_correct_method_count(self):
        """Test that D1-Q1 has exactly 18 methods as specified in issue."""
        methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[D1Question.Q1_BASELINE]
        self.assertEqual(
            len(methods),
            18,
            f"D1-Q1 should have 18 methods, found {len(methods)}",
        )
    
    def test_d1_q2_has_correct_method_count(self):
        """Test that D1-Q2 has exactly 12 methods as specified in issue."""
        methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[D1Question.Q2_NORMALIZATION]
        self.assertEqual(
            len(methods),
            12,
            f"D1-Q2 should have 12 methods, found {len(methods)}",
        )
    
    def test_d1_q3_has_correct_method_count(self):
        """Test that D1-Q3 has exactly 22 methods as specified in issue."""
        methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[D1Question.Q3_RESOURCES]
        self.assertEqual(
            len(methods),
            22,
            f"D1-Q3 should have 22 methods, found {len(methods)}",
        )
    
    def test_d1_q4_has_correct_method_count(self):
        """Test that D1-Q4 has exactly 16 methods as specified in issue."""
        methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[D1Question.Q4_CAPACITY]
        self.assertEqual(
            len(methods),
            16,
            f"D1-Q4 should have 16 methods, found {len(methods)}",
        )
    
    def test_d1_q5_has_correct_method_count(self):
        """Test that D1-Q5 has exactly 14 methods as specified in issue."""
        methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[D1Question.Q5_TEMPORAL]
        self.assertEqual(
            len(methods),
            14,
            f"D1-Q5 should have 14 methods, found {len(methods)}",
        )
    
    def test_method_contracts_built(self):
        """Test that method contracts are built from specifications."""
        self.assertGreater(len(self.orchestrator.method_contracts), 0)
        
        # Check a sample contract
        sample_method = "IndustrialPolicyProcessor.process"
        self.assertIn(sample_method, self.orchestrator.method_contracts)
        contract = self.orchestrator.method_contracts[sample_method]
        self.assertEqual(contract.canonical_name, sample_method)
        self.assertEqual(contract.class_name, "IndustrialPolicyProcessor")
        self.assertEqual(contract.method_name, "process")
    
    def test_validate_method_availability_with_missing_methods(self):
        """Test validation detects missing methods."""
        # No methods in registry, so all should be missing
        available, missing = self.orchestrator.validate_method_availability(
            D1Question.Q1_BASELINE
        )
        
        self.assertFalse(available)
        self.assertEqual(len(missing), 18)  # All 18 methods should be missing
    
    def test_validate_method_availability_with_all_methods(self):
        """Test validation passes when all methods are available."""
        # Mock all methods for Q1
        for method_name in D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[D1Question.Q1_BASELINE]:
            mock_callable = Mock(return_value="success")
            self.orchestrator.registry[method_name] = mock_callable
            self.orchestrator.method_contracts[method_name].callable_ref = mock_callable
        
        available, missing = self.orchestrator.validate_method_availability(
            D1Question.Q1_BASELINE
        )
        
        self.assertTrue(available)
        self.assertEqual(len(missing), 0)
    
    def test_orchestrate_question_fails_with_missing_methods_strict(self):
        """Test that orchestration fails in strict mode with missing methods."""
        context = {"text": "sample text"}
        
        with self.assertRaises(D1OrchestrationError) as cm:
            self.orchestrator.orchestrate_question(
                D1Question.Q1_BASELINE,
                context,
                strict=True,
            )
        
        error = cm.exception
        self.assertEqual(error.question_id, "D1-Q1")
        self.assertGreater(len(error.failed_methods), 0)
        self.assertIn("unavailable", str(error))
    
    def test_orchestrate_question_continues_with_missing_methods_non_strict(self):
        """Test that orchestration continues in non-strict mode with missing methods."""
        context = {"text": "sample text"}
        
        # Should not raise in non-strict mode
        result = self.orchestrator.orchestrate_question(
            D1Question.Q1_BASELINE,
            context,
            strict=False,
        )
        
        self.assertFalse(result.success)
        self.assertGreater(len(result.failed_methods), 0)
        self.assertIsNotNone(result.error_summary)
    
    def test_orchestrate_question_success_with_all_methods_available(self):
        """Test successful orchestration when all methods are available."""
        # Mock all methods for Q5 (smallest set with 14 methods)
        methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[D1Question.Q5_TEMPORAL]
        for method_name in methods:
            mock_callable = Mock(return_value={"result": "success"})
            self.orchestrator.registry[method_name] = mock_callable
            self.orchestrator.method_contracts[method_name].callable_ref = mock_callable
        
        context = {"text": "sample text", "data": {}}
        
        result = self.orchestrator.orchestrate_question(
            D1Question.Q5_TEMPORAL,
            context,
            strict=True,
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.executed_methods), 14)
        self.assertEqual(len(result.failed_methods), 0)
        self.assertEqual(len(result.execution_traces), 14)
        self.assertIsNone(result.error_summary)
    
    def test_orchestrate_question_with_partial_failure_strict(self):
        """Test that partial failures cause strict orchestration to fail."""
        # Mock all methods for Q2, but make one fail
        methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[D1Question.Q2_NORMALIZATION]
        for i, method_name in enumerate(methods):
            if i == 5:  # Make 6th method fail
                mock_callable = Mock(side_effect=Exception("Simulated failure"))
            else:
                mock_callable = Mock(return_value={"result": "success"})
            self.orchestrator.registry[method_name] = mock_callable
            self.orchestrator.method_contracts[method_name].callable_ref = mock_callable
        
        context = {"text": "sample text"}
        
        with self.assertRaises(D1OrchestrationError) as cm:
            self.orchestrator.orchestrate_question(
                D1Question.Q2_NORMALIZATION,
                context,
                strict=True,
            )
        
        error = cm.exception
        self.assertEqual(error.question_id, "D1-Q2")
        self.assertEqual(len(error.failed_methods), 1)
        self.assertIn("partial execution", str(error).lower())
    
    def test_execution_trace_captures_success(self):
        """Test that execution traces correctly capture successful execution."""
        mock_method = Mock(return_value={"data": "result"})
        self.orchestrator.registry["TestClass.test_method"] = mock_method
        contract = MethodContract(
            canonical_name="TestClass.test_method",
            module_name="test",
            class_name="TestClass",
            method_name="test_method",
            callable_ref=mock_method,
        )
        self.orchestrator.method_contracts["TestClass.test_method"] = contract
        
        trace = self.orchestrator._execute_method(
            D1Question.Q1_BASELINE,
            "TestClass.test_method",
            {"text": "test"},
        )
        
        self.assertTrue(trace.success)
        self.assertIsNone(trace.error)
        self.assertIsNotNone(trace.result)
        self.assertGreater(trace.duration_ms, 0)
    
    def test_execution_trace_captures_failure(self):
        """Test that execution traces correctly capture failures."""
        mock_method = Mock(side_effect=ValueError("Test error"))
        self.orchestrator.registry["TestClass.failing_method"] = mock_method
        contract = MethodContract(
            canonical_name="TestClass.failing_method",
            module_name="test",
            class_name="TestClass",
            method_name="failing_method",
            callable_ref=mock_method,
        )
        self.orchestrator.method_contracts["TestClass.failing_method"] = contract
        
        trace = self.orchestrator._execute_method(
            D1Question.Q1_BASELINE,
            "TestClass.failing_method",
            {"text": "test"},
        )
        
        self.assertFalse(trace.success)
        self.assertIsNotNone(trace.error)
        self.assertIn("Test error", trace.error)
        self.assertIsNotNone(trace.stack_trace)
    
    def test_generate_audit_report_structure(self):
        """Test that audit report has correct structure."""
        # Create mock results
        results = [
            OrchestrationResult(
                question_id="D1-Q1",
                success=True,
                executed_methods=["method1", "method2"],
                failed_methods=[],
                execution_traces=[],
                total_duration_ms=100.0,
            ),
            OrchestrationResult(
                question_id="D1-Q2",
                success=False,
                executed_methods=["method1"],
                failed_methods=["method2"],
                execution_traces=[],
                total_duration_ms=50.0,
                error_summary="Test error",
            ),
        ]
        
        report = self.orchestrator.generate_audit_report(results)
        
        # Verify structure
        self.assertIn("summary", report)
        self.assertIn("question_results", report)
        self.assertIn("execution_traces", report)
        self.assertIn("failed_question_details", report)
        self.assertIn("doctrine_compliance", report)
        
        # Verify summary calculations
        self.assertEqual(report["summary"]["total_questions"], 2)
        self.assertEqual(report["summary"]["successful_questions"], 1)
        self.assertEqual(report["summary"]["failed_questions"], 1)
        
        # Verify doctrine compliance flags
        self.assertFalse(report["doctrine_compliance"]["no_graceful_degradation"])
        self.assertTrue(report["doctrine_compliance"]["explicit_failure_semantics"])
        self.assertTrue(report["doctrine_compliance"]["full_traceability"])
    
    def test_all_d1_questions_have_unique_methods(self):
        """Test that method lists are properly specified (no typos in duplicates)."""
        for question in D1Question:
            methods = D1QuestionOrchestrator.D1_METHOD_SPECIFICATIONS[question]
            # Methods can be duplicated across questions, but within a question
            # each should be unique (no accidental copy-paste errors)
            self.assertEqual(
                len(methods),
                len(set(methods)),
                f"Duplicate methods found within {question.value}",
            )


class TestMethodContract(unittest.TestCase):
    """Test cases for MethodContract dataclass."""
    
    def test_method_contract_creation(self):
        """Test creating a method contract."""
        contract = MethodContract(
            canonical_name="SomeClass.some_method",
            module_name="some_module",
            class_name="SomeClass",
            method_name="some_method",
        )
        
        self.assertEqual(contract.canonical_name, "SomeClass.some_method")
        self.assertEqual(contract.class_name, "SomeClass")
        self.assertEqual(contract.method_name, "some_method")
        self.assertEqual(contract.preconditions, [])
        self.assertEqual(contract.postconditions, [])
        self.assertEqual(len(contract.dependencies), 0)


class TestExecutionTrace(unittest.TestCase):
    """Test cases for ExecutionTrace dataclass."""
    
    def test_execution_trace_finalize_success(self):
        """Test finalizing execution trace with success."""
        trace = ExecutionTrace(
            question_id="D1-Q1",
            method_name="TestMethod",
            start_time=1000.0,
        )
        
        trace.finalize(success=True, result={"data": "value"})
        
        self.assertTrue(trace.success)
        self.assertIsNotNone(trace.end_time)
        self.assertGreater(trace.duration_ms, 0)
        self.assertEqual(trace.result, {"data": "value"})
        self.assertIsNone(trace.error)
    
    def test_execution_trace_finalize_failure(self):
        """Test finalizing execution trace with failure."""
        trace = ExecutionTrace(
            question_id="D1-Q1",
            method_name="TestMethod",
            start_time=1000.0,
        )
        
        error = ValueError("Test error message")
        trace.finalize(success=False, error=error)
        
        self.assertFalse(trace.success)
        self.assertIsNotNone(trace.end_time)
        self.assertGreater(trace.duration_ms, 0)
        self.assertEqual(trace.error, "Test error message")
        self.assertIsNotNone(trace.stack_trace)


if __name__ == "__main__":
    unittest.main()
