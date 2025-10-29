"""
Tests for the Choreographer (single micro-question execution).

These tests verify the granular execution of a SINGLE micro question.
"""

import json
import unittest
from pathlib import Path

from orchestrator import get_questionnaire_provider
from orchestrator.coreographer import (
    Choreographer,
    QuestionResult,
    PreprocessedDocument,
    DAGNode,
    ExecutionPlan,
    MethodResult,
    NodeResult,
    MethodPriority,
    FlowController,
    MethodExecutor,
)


QUESTIONNAIRE_PROVIDER = get_questionnaire_provider()


class TestChoreographer(unittest.TestCase):
    """Test the Choreographer class (single question execution)."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monolith_provider = QUESTIONNAIRE_PROVIDER
        self.catalog_path = Path("rules/METODOS/metodos_completos_nivel3.json")

        # Check if files exist
        self.files_exist = (
            self.monolith_provider.exists() and
            self.catalog_path.exists()
        )
    
    def test_choreographer_init(self):
        """Test choreographer initialization."""
        choreographer = Choreographer()
        
        self.assertIsNotNone(choreographer)
        self.assertIsNotNone(choreographer.dispatcher)
        self.assertIsNotNone(choreographer.method_executor)
        self.assertEqual(choreographer.default_timeout_ms, 30000)
        self.assertEqual(choreographer.default_max_retries, 2)
    
    def test_method_priority_enum(self):
        """Test method priority enum values."""
        self.assertEqual(MethodPriority.CRITICO.value, 3)
        self.assertEqual(MethodPriority.IMPORTANTE.value, 2)
        self.assertEqual(MethodPriority.COMPLEMENTARIO.value, 1)
    
    def test_dag_node_creation(self):
        """Test DAG node creation."""
        node = DAGNode(
            node_id="node_1",
            file_name="policy_processor.py",
            class_name="IndustrialPolicyProcessor",
            method_names=["process", "extract"],
            method_types=["extraction", "validation"],
            priorities=[3, 2],
            timeout_ms=30000,
            max_retries=2
        )
        
        self.assertEqual(node.node_id, "node_1")
        self.assertEqual(node.file_name, "policy_processor.py")
        self.assertEqual(len(node.method_names), 2)
    
    def test_build_execution_dag_simple(self):
        """Test building simple execution DAG."""
        method_packages = [
            {
                'f': 'policy_processor.py',
                'c': 'PolicyTextProcessor',
                'm': ['normalize_unicode', 'segment_into_sentences'],
                't': ['normalization', 'segmentation'],
                'pr': [2, 2]
            }
        ]
        
        execution_plan = FlowController.build_execution_dag(None, method_packages)
        
        self.assertIsInstance(execution_plan, ExecutionPlan)
        self.assertEqual(len(execution_plan.nodes), 1)
        self.assertEqual(execution_plan.nodes[0].file_name, 'policy_processor.py')
        self.assertEqual(len(execution_plan.nodes[0].method_names), 2)
    
    @unittest.skipUnless(
        QUESTIONNAIRE_PROVIDER.exists(),
        "Requires orchestrator questionnaire payload"
    )
    def test_map_question_to_slot(self):
        """Test mapping question to base slot."""
        choreographer = Choreographer()

        # Load config
        monolith = QUESTIONNAIRE_PROVIDER.load()
        with open("rules/METODOS/metodos_completos_nivel3.json") as f:
            method_catalog = json.load(f)
        
        # Test mapping for question 1
        base_slot, q_metadata, method_packages, flow_spec = choreographer._map_question_to_slot(
            1, monolith, method_catalog
        )
        
        self.assertEqual(base_slot, "D1-Q1")
        self.assertIsNotNone(q_metadata)
        self.assertIsInstance(method_packages, list)
    
    def test_method_result_creation(self):
        """Test method result creation."""
        result = MethodResult(
            method_name="normalize_unicode",
            success=True,
            result="normalized text",
            execution_time_ms=10.5,
            retries_used=0
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.method_name, "normalize_unicode")
        self.assertEqual(result.retries_used, 0)
    
    def test_node_result_creation(self):
        """Test node result creation."""
        method_results = [
            MethodResult("method1", True, "result1", execution_time_ms=5.0),
            MethodResult("method2", True, "result2", execution_time_ms=3.0)
        ]
        
        node_result = NodeResult(
            node_id="node_1",
            success=True,
            method_results=method_results,
            execution_time_ms=8.0
        )
        
        self.assertTrue(node_result.success)
        self.assertEqual(len(node_result.method_results), 2)
        self.assertEqual(node_result.node_id, "node_1")
    
    def test_question_result_creation(self):
        """Test question result creation."""
        result = QuestionResult(
            question_global=1,
            base_slot="D1-Q1",
            evidence={'successful_methods': 5},
            raw_results={'method1': 'result1'},
            execution_time_ms=100.0
        )
        
        self.assertEqual(result.question_global, 1)
        self.assertEqual(result.base_slot, "D1-Q1")
        self.assertIn('successful_methods', result.evidence)


class TestMethodExecutor(unittest.TestCase):
    """Test the MethodExecutor class."""
    
    def test_method_executor_init(self):
        """Test method executor initialization."""
        from orchestrator.choreographer_dispatch import ChoreographerDispatcher
        
        dispatcher = ChoreographerDispatcher()
        executor = MethodExecutor(dispatcher)
        
        self.assertIsNotNone(executor)
        self.assertEqual(executor.dispatcher, dispatcher)


class TestFlowController(unittest.TestCase):
    """Test the FlowController class."""
    
    def test_identify_parallel_branches_empty(self):
        """Test identifying parallel branches with empty plan."""
        plan = ExecutionPlan(
            nodes=[],
            parallel_groups=[],
            execution_order=[]
        )
        
        branches = FlowController.identify_parallel_branches(plan)
        self.assertEqual(len(branches), 0)


if __name__ == "__main__":
    unittest.main()
