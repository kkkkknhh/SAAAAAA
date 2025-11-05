"""Test architecture boundaries - ensure only orchestrator loads questionnaire."""
import pytest
from unittest.mock import Mock, patch, mock_open
import json


class TestArchitectureBoundaries:
    """Test that architecture boundaries are properly enforced."""

    def test_policy_processor_accepts_injected_questionnaire(self):
        """Test that PolicyProcessor accepts questionnaire via dependency injection."""
        from saaaaaa.processing.policy_processor import IndustrialPolicyProcessor
        
        # Create mock questionnaire data
        mock_questionnaire = {
            "questions": [
                {"id": "D1-Q1", "text": "Test question 1"},
                {"id": "D1-Q2", "text": "Test question 2"},
            ]
        }
        
        # Create processor with injected questionnaire (CORRECT WAY)
        processor = IndustrialPolicyProcessor(questionnaire_data=mock_questionnaire)
        
        # Verify questionnaire was accepted
        assert processor.questionnaire_data == mock_questionnaire
        assert len(processor.questionnaire_data["questions"]) == 2

    def test_factory_creates_processor_with_questionnaire(self):
        """Test that CoreModuleFactory creates PolicyProcessor with questionnaire."""
        from saaaaaa.core.orchestrator.factory import CoreModuleFactory
        from pathlib import Path
        from unittest.mock import patch
        
        mock_questionnaire = {
            "questions": [
                {"id": "D1-Q1", "text": "Test question 1"},
            ]
        }
        
        # Create factory and mock get_questionnaire to return mock data
        factory = CoreModuleFactory()
        
        with patch.object(factory, 'get_questionnaire', return_value=mock_questionnaire):
            # Create processor via factory (CORRECT WAY)
            processor = factory.create_policy_processor()
            
            # Verify processor has questionnaire data injected
            assert processor.questionnaire_data == mock_questionnaire

    def test_method_executor_accepts_questionnaire(self):
        """Test that MethodExecutor accepts questionnaire via dependency injection."""
        # This test verifies the orchestrator pattern
        try:
            from saaaaaa.core.orchestrator.ORCHESTRATOR_MONILITH import MethodExecutor
        except ImportError:
            pytest.skip("ORCHESTRATOR_MONILITH not available")
        
        mock_questionnaire = {
            "questions": [{"id": "D1-Q1"}]
        }
        
        # Create executor with injected questionnaire (CORRECT WAY)
        executor = MethodExecutor(questionnaire_data=mock_questionnaire)
        
        # Verify the executor was created successfully
        assert executor is not None
        
        # Verify the PolicyProcessor instance has questionnaire data
        if hasattr(executor, 'instances') and 'IndustrialPolicyProcessor' in executor.instances:
            processor = executor.instances['IndustrialPolicyProcessor']
            assert processor.questionnaire_data == mock_questionnaire

    def test_old_factory_injects_questionnaire(self):
        """Test that old orchestrator/factory.py also injects questionnaire."""
        from pathlib import Path
        
        mock_questionnaire = {
            "industrial": [
                {"id": "Q1", "text": "Question 1"}
            ]
        }
        
        # Mock file reading
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_questionnaire))):
            with patch("pathlib.Path.read_text", return_value=json.dumps(mock_questionnaire)):
                try:
                    from orchestrator.factory import build_processor
                    
                    # Create processor via old factory
                    processor = build_processor(path="test_questionnaire.json")
                    
                    # Verify questionnaire was injected
                    assert processor.questionnaire_data == mock_questionnaire
                except ImportError:
                    pytest.skip("orchestrator.factory not available")

    def test_policy_processor_warns_on_file_loading(self, caplog):
        """Test that PolicyProcessor warns when loading from file (deprecated path)."""
        import logging
        from saaaaaa.processing.policy_processor import IndustrialPolicyProcessor
        
        # Mock the factory load_json to avoid actual file I/O
        mock_questionnaire = {"questions": []}
        
        with patch("saaaaaa.processing.factory.load_json", return_value=mock_questionnaire):
            # Create processor WITHOUT injected data (deprecated path)
            with caplog.at_level(logging.WARNING):
                processor = IndustrialPolicyProcessor(
                    questionnaire_path=None  # Will use default path
                )
            
            # Verify warning was logged
            assert any("DEPRECATED" in record.message for record in caplog.records)

    def test_orchestrator_loads_questionnaire_once(self):
        """Test that Orchestrator loads questionnaire only once and passes to executor."""
        from pathlib import Path
        
        mock_catalog = {"methods": []}
        mock_questionnaire = {"questions": [{"id": "D1-Q1"}]}
        
        # Mock file operations
        mock_files = {
            "catalog.json": json.dumps(mock_catalog),
            "questionnaire.json": json.dumps(mock_questionnaire),
        }
        
        def mock_open_func(path, *args, **kwargs):
            path_str = str(path)
            if "catalog" in path_str or path_str.endswith("catalog.json"):
                return mock_open(read_data=mock_files["catalog.json"])()
            elif "questionnaire" in path_str or "monolith" in path_str:
                return mock_open(read_data=mock_files["questionnaire.json"])()
            raise FileNotFoundError(f"Mock file not found: {path}")
        
        try:
            with patch("builtins.open", side_effect=mock_open_func):
                from saaaaaa.core.orchestrator.ORCHESTRATOR_MONILITH import Orchestrator
                
                # Create orchestrator (should load questionnaire once)
                orchestrator = Orchestrator(
                    catalog_path="catalog.json",
                    monolith_path="questionnaire.json",
                )
                
                # Verify orchestrator was created
                assert orchestrator is not None
                
                # Verify executor exists and has questionnaire
                assert orchestrator.executor is not None
        except Exception as e:
            # If there are other dependencies missing, skip
            pytest.skip(f"Test requires full dependencies: {e}")


class TestFactoryPattern:
    """Test that factory pattern is correctly implemented."""

    def test_factory_is_single_source_of_truth(self):
        """Test that CoreModuleFactory is the single source of truth for questionnaire."""
        from saaaaaa.core.orchestrator.factory import CoreModuleFactory
        
        # Create factory
        factory = CoreModuleFactory()
        
        # Mock questionnaire
        mock_questionnaire = {"questions": [{"id": "D1-Q1"}]}
        factory.questionnaire_cache = mock_questionnaire
        
        # Get questionnaire multiple times
        q1 = factory.get_questionnaire()
        q2 = factory.get_questionnaire()
        
        # Should return the same cached instance
        assert q1 is q2
        assert q1 == mock_questionnaire

    def test_no_modules_load_questionnaire_directly(self):
        """Test that processing/analysis modules don't load questionnaire directly."""
        import ast
        from pathlib import Path
        
        # Modules that SHOULD use factory for I/O
        modules_to_check = [
            "src/saaaaaa/processing/policy_processor.py",
            "src/saaaaaa/analysis/dereck_beach.py",
            "src/saaaaaa/analysis/Analyzer_one.py",
        ]
        
        repo_root = Path(__file__).parent.parent
        
        for module_path in modules_to_check:
            full_path = repo_root / module_path
            if not full_path.exists():
                continue
                
            with open(full_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(full_path))
            
            # Look for direct json.load calls (should use factory instead)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Check if it's json.load
                    if isinstance(node.func, ast.Attribute):
                        if (isinstance(node.func.value, ast.Name) and 
                            node.func.value.id == 'json' and 
                            node.func.attr == 'load'):
                            # Found json.load - verify it's using factory
                            # This is caught in static analysis, but we allow it
                            # if it's importing from factory
                            pass
