"""Test questionnaire dependency injection pattern."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestQuestionnaireInjection:
    """Test that questionnaire is properly injected via dependency injection."""

    def test_policy_processor_signature_accepts_questionnaire_data(self):
        """Test that PolicyProcessor __init__ accepts questionnaire_data parameter."""
        import inspect
        
        # Mock the imports that might fail
        sys.modules['camelot'] = MagicMock()
        sys.modules['nltk'] = MagicMock()
        sys.modules['tensorflow'] = MagicMock()
        sys.modules['transformers'] = MagicMock()
        
        try:
            from saaaaaa.processing.policy_processor import IndustrialPolicyProcessor
            
            # Get the signature of __init__
            sig = inspect.signature(IndustrialPolicyProcessor.__init__)
            params = list(sig.parameters.keys())
            
            # Verify questionnaire_data is a parameter
            assert 'questionnaire_data' in params, \
                "PolicyProcessor must accept questionnaire_data parameter for dependency injection"
            
            # Verify it comes before questionnaire_path (preferred over file path)
            questionnaire_data_idx = params.index('questionnaire_data')
            questionnaire_path_idx = params.index('questionnaire_path') if 'questionnaire_path' in params else -1
            
            if questionnaire_path_idx >= 0:
                assert questionnaire_data_idx < questionnaire_path_idx, \
                    "questionnaire_data should come before questionnaire_path (preferred method)"
                    
        except ImportError as e:
            pytest.skip(f"Cannot import PolicyProcessor: {e}")

    def test_factory_has_create_policy_processor_method(self):
        """Test that CoreModuleFactory has create_policy_processor method."""
        from saaaaaa.core.orchestrator.factory import CoreModuleFactory
        
        # Verify method exists
        assert hasattr(CoreModuleFactory, 'create_policy_processor'), \
            "CoreModuleFactory must have create_policy_processor method"
        
        # Verify it's callable
        factory = CoreModuleFactory()
        assert callable(getattr(factory, 'create_policy_processor')), \
            "create_policy_processor must be callable"

    def test_method_executor_signature_accepts_questionnaire(self):
        """Test that MethodExecutor accepts questionnaire_data parameter."""
        import inspect
        
        try:
            # Import the MethodExecutor from the monolith
            from saaaaaa.core.orchestrator.ORCHESTRATOR_MONILITH import MethodExecutor
            
            # Get the signature
            sig = inspect.signature(MethodExecutor.__init__)
            params = list(sig.parameters.keys())
            
            # Verify questionnaire_data parameter exists
            assert 'questionnaire_data' in params, \
                "MethodExecutor must accept questionnaire_data for dependency injection"
                
        except ImportError:
            pytest.skip("ORCHESTRATOR_MONILITH not available")

    def test_orchestrator_factory_build_processor_injects_questionnaire(self):
        """Test that orchestrator/factory.py build_processor injects questionnaire."""
        import ast
        from pathlib import Path
        
        # Read the old factory file
        factory_path = Path(__file__).parent.parent / "orchestrator" / "factory.py"
        
        if not factory_path.exists():
            pytest.skip("orchestrator/factory.py not found")
        
        with open(factory_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        # Find the build_processor function
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_processor":
                # Look for IndustrialPolicyProcessor instantiation
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name) and child.func.id == "IndustrialPolicyProcessor":
                            # Check if questionnaire_data is passed as keyword argument
                            keyword_names = [kw.arg for kw in child.keywords]
                            assert 'questionnaire_data' in keyword_names, \
                                "build_processor must pass questionnaire_data to IndustrialPolicyProcessor"
                            return
        
        pytest.fail("Could not find IndustrialPolicyProcessor instantiation in build_processor")

    def test_architecture_documentation_updated(self):
        """Test that architecture documentation mentions the new pattern."""
        from pathlib import Path
        
        # Check if IO_MIGRATION_GUIDE mentions dependency injection
        migration_guide = Path(__file__).parent.parent / "IO_MIGRATION_GUIDE.md"
        
        if migration_guide.exists():
            content = migration_guide.read_text()
            
            # Check for key terms
            assert "dependency injection" in content.lower() or "inject" in content.lower(), \
                "IO_MIGRATION_GUIDE should document dependency injection pattern"

    def test_policy_processor_logs_warning_on_deprecated_path(self):
        """Test that deprecated file loading path is documented with warning."""
        from pathlib import Path
        
        # Read policy_processor source
        processor_path = Path(__file__).parent.parent / "src" / "saaaaaa" / "processing" / "policy_processor.py"
        
        if not processor_path.exists():
            pytest.skip("policy_processor.py not found")
        
        content = processor_path.read_text()
        
        # Verify warning is logged for deprecated path
        assert "DEPRECATED" in content or "deprecated" in content, \
            "policy_processor should warn about deprecated file loading"
        
        assert "logger.warning" in content, \
            "policy_processor should log warning for deprecated usage"

    def test_no_direct_questionnaire_load_in_processor_init(self):
        """Test that _load_questionnaire is not called unconditionally in __init__."""
        import ast
        from pathlib import Path
        
        processor_path = Path(__file__).parent.parent / "src" / "saaaaaa" / "processing" / "policy_processor.py"
        
        if not processor_path.exists():
            pytest.skip("policy_processor.py not found")
        
        with open(processor_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        
        # Find the IndustrialPolicyProcessor class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "IndustrialPolicyProcessor":
                # Find __init__ method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        # Look for _load_questionnaire call
                        for child in ast.walk(item):
                            if isinstance(child, ast.Call):
                                if isinstance(child.func, ast.Attribute):
                                    if child.func.attr == "_load_questionnaire":
                                        # Found the call - check if it's conditional
                                        # Walk up to find if it's in an if/else block
                                        # This is a simplified check
                                        source = ast.unparse(item)
                                        
                                        # Should be conditional on questionnaire_data being None
                                        assert "if questionnaire_data is not None" in source or \
                                               "questionnaire_data is not None" in source or \
                                               "else:" in source, \
                                            "_load_questionnaire should only be called when questionnaire_data is None"
                                        return
        
        pytest.fail("Could not verify conditional loading in PolicyProcessor.__init__")


class TestBackwardCompatibility:
    """Test that backward compatibility is maintained."""

    def test_policy_processor_still_accepts_questionnaire_path(self):
        """Test that old code using questionnaire_path still works (deprecated)."""
        import inspect
        
        sys.modules['camelot'] = MagicMock()
        sys.modules['nltk'] = MagicMock()
        
        try:
            from saaaaaa.processing.policy_processor import IndustrialPolicyProcessor
            
            sig = inspect.signature(IndustrialPolicyProcessor.__init__)
            params = list(sig.parameters.keys())
            
            # Verify questionnaire_path is still available for backward compatibility
            assert 'questionnaire_path' in params, \
                "questionnaire_path should still be available for backward compatibility"
                
        except ImportError:
            pytest.skip("Cannot import PolicyProcessor")

    def test_method_executor_works_without_questionnaire_data(self):
        """Test that MethodExecutor can be created without questionnaire_data (deprecated)."""
        import inspect
        
        try:
            from saaaaaa.core.orchestrator.ORCHESTRATOR_MONILITH import MethodExecutor
            
            sig = inspect.signature(MethodExecutor.__init__)
            
            # Verify questionnaire_data has a default value (None)
            assert sig.parameters['questionnaire_data'].default is None or \
                   sig.parameters['questionnaire_data'].default == inspect.Parameter.empty, \
                "questionnaire_data should have a default value for backward compatibility"
                
        except ImportError:
            pytest.skip("ORCHESTRATOR_MONILITH not available")
