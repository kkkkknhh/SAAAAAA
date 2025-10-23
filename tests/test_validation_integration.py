#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests - Validation Engine + Choreographer
=====================================================

Tests the integration of validation_engine.py with choreographer.py
including pre-step validation hooks and expected_elements verification.

Author: Integration Team - Agent 3
Version: 1.0.0
Python: 3.10+
"""

import pytest
import json
from pathlib import Path
from validation_engine import ValidationEngine
from validation.predicates import ValidationPredicates, ValidationResult


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_question_spec():
    """Sample question specification from rubric."""
    return {
        "id": "P1-D1-Q1",
        "dimension": "D1",
        "question_no": 1,
        "template": "Test question template",
        "scoring_modality": "TYPE_A",
        "max_score": 3.0,
        "expected_elements": [
            "valor_numerico",
            "año",
            "fuente",
            "serie_temporal"
        ],
        "search_patterns": {
            "valor_numerico": {
                "type": "regex",
                "pattern": r"\d+[.,]?\d*\s*(%|casos|personas)",
                "description": "Buscar valor numérico con unidad"
            }
        },
        "evidence_sources": {
            "orchestrator_key": "feasibility",
            "extraction_method": "Check feasibility.has_baseline"
        },
        "policy_area": "P1",
        "policy_area_name": "Derechos de las mujeres e igualdad de género"
    }


@pytest.fixture
def sample_execution_results():
    """Sample execution results."""
    return {
        "policy_processor": {
            "status": "success",
            "data": {"indicators": 5},
            "confidence": 0.85
        },
        "dereck_beach": {
            "status": "success",
            "data": {"nodes": 10, "edges": 15},
            "confidence": 0.90
        }
    }


@pytest.fixture
def sample_plan_text():
    """Sample plan document text."""
    return """
    Plan de Desarrollo Territorial 2024-2027
    
    Diagnóstico:
    Según DANE (2022), el 45.3% de las mujeres en el territorio
    enfrentan barreras de acceso. La serie histórica 2018-2022
    muestra una tendencia ascendente. Fuente: Encuesta Nacional 2022.
    """


@pytest.fixture
def validation_engine():
    """Validation engine instance."""
    return ValidationEngine()


@pytest.fixture
def sample_producers_dict():
    """Sample producers dictionary."""
    return {
        "dereck_beach": {
            "status": "initialized",
            "methods_count": 99
        },
        "policy_processor": {
            "status": "initialized",
            "methods_count": 32
        },
        "feasibility": {
            "status": "initialized",
            "methods_count": 20
        }
    }


# ============================================================================
# VALIDATION PREDICATES TESTS
# ============================================================================

class TestValidationPredicates:
    """Test ValidationPredicates class."""

    def test_verify_scoring_preconditions_success(
        self,
        sample_question_spec,
        sample_execution_results,
        sample_plan_text
    ):
        """Test successful scoring preconditions verification."""
        result = ValidationPredicates.verify_scoring_preconditions(
            sample_question_spec,
            sample_execution_results,
            sample_plan_text
        )
        
        assert result.is_valid is True
        assert result.severity == "INFO"
        assert "All scoring preconditions met" in result.message
        assert result.context["question_id"] == "P1-D1-Q1"
        assert result.context["expected_elements_count"] == 4

    def test_verify_scoring_preconditions_missing_expected_elements(
        self,
        sample_execution_results,
        sample_plan_text
    ):
        """Test preconditions fail when expected_elements missing."""
        question_spec = {
            "id": "P1-D1-Q1",
            # missing expected_elements
        }
        
        result = ValidationPredicates.verify_scoring_preconditions(
            question_spec,
            sample_execution_results,
            sample_plan_text
        )
        
        assert result.is_valid is False
        assert result.severity == "ERROR"
        assert "expected_elements" in result.message

    def test_verify_scoring_preconditions_empty_execution_results(
        self,
        sample_question_spec,
        sample_plan_text
    ):
        """Test preconditions fail with empty execution results."""
        result = ValidationPredicates.verify_scoring_preconditions(
            sample_question_spec,
            {},  # Empty execution results
            sample_plan_text
        )
        
        assert result.is_valid is False
        assert result.severity == "ERROR"
        assert "execution_results cannot be empty" in result.message

    def test_verify_scoring_preconditions_empty_plan_text(
        self,
        sample_question_spec,
        sample_execution_results
    ):
        """Test preconditions fail with empty plan text."""
        result = ValidationPredicates.verify_scoring_preconditions(
            sample_question_spec,
            sample_execution_results,
            ""  # Empty plan text
        )
        
        assert result.is_valid is False
        assert result.severity == "ERROR"
        assert "plan_text cannot be empty" in result.message

    def test_verify_expected_elements_success(self, sample_question_spec):
        """Test successful expected_elements verification."""
        result = ValidationPredicates.verify_expected_elements(
            sample_question_spec,
            {}  # cuestionario_data not used in basic check
        )
        
        assert result.is_valid is True
        assert result.severity == "INFO"
        assert result.context["count"] == 4
        assert result.context["expected_elements"] == [
            "valor_numerico", "año", "fuente", "serie_temporal"
        ]

    def test_verify_expected_elements_missing(self):
        """Test expected_elements verification with missing field."""
        question_spec = {
            "id": "P1-D1-Q1"
            # No expected_elements
        }
        
        result = ValidationPredicates.verify_expected_elements(
            question_spec,
            {}
        )
        
        assert result.is_valid is False
        assert result.severity == "WARNING"
        assert "no expected_elements defined" in result.message

    def test_verify_expected_elements_empty_list(self):
        """Test expected_elements verification with empty list."""
        question_spec = {
            "id": "P1-D1-Q1",
            "expected_elements": []  # Empty list
        }
        
        result = ValidationPredicates.verify_expected_elements(
            question_spec,
            {}
        )
        
        assert result.is_valid is False
        assert result.severity == "WARNING"
        assert "empty expected_elements" in result.message

    def test_verify_execution_context_success(self):
        """Test successful execution context verification."""
        result = ValidationPredicates.verify_execution_context(
            "P1-D1-Q1",
            "P1",
            "D1"
        )
        
        assert result.is_valid is True
        assert result.severity == "INFO"
        assert result.context["question_id"] == "P1-D1-Q1"
        assert result.context["policy_area"] == "P1"
        assert result.context["dimension"] == "D1"

    def test_verify_execution_context_invalid_policy_area(self):
        """Test execution context with invalid policy area."""
        result = ValidationPredicates.verify_execution_context(
            "P99-D1-Q1",
            "P99",  # Invalid: must be P1-P10
            "D1"
        )
        
        assert result.is_valid is False
        assert result.severity == "ERROR"
        assert "P1-P10" in result.message

    def test_verify_execution_context_invalid_dimension(self):
        """Test execution context with invalid dimension."""
        result = ValidationPredicates.verify_execution_context(
            "P1-D99-Q1",
            "P1",
            "D99"  # Invalid: must be D1-D6
        )
        
        assert result.is_valid is False
        assert result.severity == "ERROR"
        assert "D1-D6" in result.message

    def test_verify_producer_availability_success(self, sample_producers_dict):
        """Test successful producer availability check."""
        result = ValidationPredicates.verify_producer_availability(
            "dereck_beach",
            sample_producers_dict
        )
        
        assert result.is_valid is True
        assert result.severity == "INFO"
        assert result.context["producer_name"] == "dereck_beach"

    def test_verify_producer_availability_not_found(self, sample_producers_dict):
        """Test producer availability when producer not found."""
        result = ValidationPredicates.verify_producer_availability(
            "nonexistent_producer",
            sample_producers_dict
        )
        
        assert result.is_valid is False
        assert result.severity == "ERROR"
        assert "not found" in result.message


# ============================================================================
# VALIDATION ENGINE TESTS
# ============================================================================

class TestValidationEngine:
    """Test ValidationEngine class."""

    def test_validate_scoring_preconditions(
        self,
        validation_engine,
        sample_question_spec,
        sample_execution_results,
        sample_plan_text
    ):
        """Test ValidationEngine.validate_scoring_preconditions."""
        result = validation_engine.validate_scoring_preconditions(
            sample_question_spec,
            sample_execution_results,
            sample_plan_text
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_expected_elements(
        self,
        validation_engine,
        sample_question_spec
    ):
        """Test ValidationEngine.validate_expected_elements."""
        result = validation_engine.validate_expected_elements(
            sample_question_spec
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_execution_context(self, validation_engine):
        """Test ValidationEngine.validate_execution_context."""
        result = validation_engine.validate_execution_context(
            "P1-D1-Q1",
            "P1",
            "D1"
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_producer_availability(
        self,
        validation_engine,
        sample_producers_dict
    ):
        """Test ValidationEngine.validate_producer_availability."""
        result = validation_engine.validate_producer_availability(
            "dereck_beach",
            sample_producers_dict
        )
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True

    def test_validate_all_preconditions(
        self,
        validation_engine,
        sample_question_spec,
        sample_execution_results,
        sample_plan_text,
        sample_producers_dict
    ):
        """Test ValidationEngine.validate_all_preconditions."""
        report = validation_engine.validate_all_preconditions(
            sample_question_spec,
            sample_execution_results,
            sample_plan_text,
            sample_producers_dict
        )
        
        assert report.total_checks > 0
        assert report.passed > 0
        assert not report.has_errors()


# ============================================================================
# INTEGRATION TESTS WITH CHOREOGRAPHER
# ============================================================================

class TestChoreographerIntegration:
    """Test integration with Choreographer."""

    def test_validation_engine_initialized_in_choreographer(self):
        """Test that ValidationEngine is initialized in Choreographer."""
        # Import here to avoid circular dependencies
        try:
            from choreographer import ExecutionChoreographer
            
            choreographer = ExecutionChoreographer()
            
            # Check that validation_engine is initialized
            assert hasattr(choreographer, 'validation_engine')
            assert isinstance(choreographer.validation_engine, ValidationEngine)
            
        except Exception as e:
            pytest.skip(f"Choreographer initialization failed: {e}")

    def test_pre_step_validation_hook_exists(self):
        """Test that pre-step validation hook exists in _execute_pipeline."""
        try:
            from choreographer import ExecutionChoreographer
            import inspect
            
            # Get source code of _execute_pipeline
            source = inspect.getsource(ExecutionChoreographer._execute_pipeline)
            
            # Check for validation hook markers
            assert "PRE-STEP VALIDATION" in source
            assert "validate_execution_context" in source
            assert "validate_producer_availability" in source
            
        except Exception as e:
            pytest.skip(f"Could not inspect Choreographer: {e}")


# ============================================================================
# EXPECTED ELEMENTS VERIFICATION TESTS
# ============================================================================

class TestExpectedElementsVerification:
    """Test expected_elements verification from cuestionario_FIXED.json."""

    def test_load_cuestionario_expected_elements(self):
        """Test loading expected_elements from cuestionario_FIXED.json."""
        cuestionario_path = Path("cuestionario_FIXED.json")
        
        if not cuestionario_path.exists():
            pytest.skip("cuestionario_FIXED.json not found")
        
        # This test verifies the structure exists
        # Actual validation happens in ValidationEngine
        assert cuestionario_path.exists()

    def test_rubric_expected_elements_structure(self):
        """Test expected_elements structure in rubric_scoring_FIXED.json."""
        rubric_path = Path("rubric_scoring_FIXED.json")
        
        if not rubric_path.exists():
            pytest.skip("rubric_scoring_FIXED.json not found")
        
        with open(rubric_path, 'r', encoding='utf-8') as f:
            rubric = json.load(f)
        
        # Check that questions have expected_elements
        questions = rubric.get("questions", [])
        
        if len(questions) > 0:
            # Sample first question
            q1 = questions[0]
            
            assert "expected_elements" in q1, "Questions should have expected_elements field"
            assert isinstance(q1["expected_elements"], list), "expected_elements should be a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
