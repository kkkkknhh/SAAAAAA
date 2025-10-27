#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests - Orchestrator and Choreographer Initialization
==================================================================

Tests the correct initialization of ExecutionChoreographer from PolicyAnalysisOrchestrator,
specifically validating that the signature mismatch issue is resolved.

Author: Copilot Agent
Version: 1.0.0
Python: 3.10+
"""

import pytest
import ast


# ============================================================================
# TEST: Choreographer Initialization Signature
# ============================================================================

class TestChoreographerInitialization:
    """Test that ExecutionChoreographer can be instantiated with the correct parameters."""

    def test_choreographer_accepts_five_parameters(self):
        """Test that ExecutionChoreographer.__init__ accepts all 5 parameters including config_path."""
        # Read the source file and parse it
        with open('/home/runner/work/SAAAAAA/SAAAAAA/policy_analysis_pipeline.py', 'r') as f:
            source = f.read()
        
        # Parse the file
        tree = ast.parse(source)
        
        # Find ExecutionChoreographer class
        choreographer_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'ExecutionChoreographer':
                choreographer_class = node
                break
        
        assert choreographer_class is not None, "ExecutionChoreographer class not found"
        
        # Find __init__ method
        init_method = None
        for item in choreographer_class.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                init_method = item
                break
        
        assert init_method is not None, "__init__ method not found in ExecutionChoreographer"
        
        # Extract parameter names
        params = [arg.arg for arg in init_method.args.args]
        
        # Verify that all expected parameters are present
        assert 'self' in params
        assert 'execution_mapping_path' in params
        assert 'method_class_map_path' in params
        assert 'questionnaire_hash' in params
        assert 'deterministic_context' in params
        assert 'config_path' in params
        
        # Verify config_path has a default value
        defaults = init_method.args.defaults
        assert len(defaults) > 0, "Expected default value for config_path"
        # config_path is the last parameter with a default of None
        assert isinstance(defaults[-1], ast.Constant) and defaults[-1].value is None

    def test_orchestrator_passes_config_path_to_choreographer(self):
        """Test that Orchestrator passes config_path parameter to ExecutionChoreographer."""
        # Read the Orchestrator source file
        with open('/home/runner/work/SAAAAAA/SAAAAAA/Industrialpolicyprocessor.py', 'r') as f:
            source = f.read()
        
        # Parse the file
        tree = ast.parse(source)
        
        # Find PolicyAnalysisOrchestrator class
        orchestrator_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == 'PolicyAnalysisOrchestrator':
                orchestrator_class = node
                break
        
        assert orchestrator_class is not None, "PolicyAnalysisOrchestrator class not found"
        
        # Find __init__ method
        init_method = None
        for item in orchestrator_class.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                init_method = item
                break
        
        assert init_method is not None, "__init__ method not found in PolicyAnalysisOrchestrator"
        
        # Search for ExecutionChoreographer instantiation
        found_choreographer_call = False
        found_config_path_param = False
        
        for node in ast.walk(init_method):
            if isinstance(node, ast.Call):
                # Check if this is a call to ExecutionChoreographer
                if isinstance(node.func, ast.Name) and node.func.id == 'ExecutionChoreographer':
                    found_choreographer_call = True
                    # Check if config_path is in the keyword arguments
                    for keyword in node.keywords:
                        if keyword.arg == 'config_path':
                            found_config_path_param = True
                            break
        
        assert found_choreographer_call, "ExecutionChoreographer instantiation not found in Orchestrator.__init__"
        assert found_config_path_param, "config_path parameter not passed to ExecutionChoreographer"

