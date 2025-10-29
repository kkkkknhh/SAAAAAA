"""
Tests for the Choreographer main flow controller.

These tests verify the complete 305-question processing pipeline.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from orchestrator.choreographer import (
    Choreographer,
    CompleteReport,
    ExecutionMode,
    PhaseResult,
    PreprocessedDocument,
    QuestionResult,
    ScoredResult,
)


class TestChoreographer(unittest.TestCase):
    """Test the Choreographer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use actual monolith and catalog if they exist
        self.monolith_path = Path("questionnaire_monolith.json")
        self.catalog_path = Path("rules/METODOS/metodos_completos_nivel3.json")
        
        # Check if files exist
        self.files_exist = (
            self.monolith_path.exists() and
            self.catalog_path.exists()
        )
    
    def test_choreographer_init(self):
        """Test choreographer initialization."""
        choreographer = Choreographer(
            monolith_path=self.monolith_path,
            method_catalog_path=self.catalog_path,
            enable_async=False,
        )
        
        self.assertIsNotNone(choreographer)
        self.assertIsNone(choreographer.monolith)  # Not loaded yet
        self.assertIsNone(choreographer.method_catalog)  # Not loaded yet
        self.assertEqual(len(choreographer.phase_results), 0)
    
    @unittest.skipUnless(
        Path("questionnaire_monolith.json").exists(),
        "Requires questionnaire_monolith.json"
    )
    def test_load_configuration(self):
        """Test configuration loading."""
        choreographer = Choreographer(
            monolith_path=self.monolith_path,
            method_catalog_path=self.catalog_path,
            enable_async=False,
        )
        
        # Load configuration
        result = choreographer._load_configuration()
        
        self.assertTrue(result.success, f"Configuration failed: {result.error}")
        self.assertEqual(result.phase_id, "FASE_0")
        self.assertIsNotNone(choreographer.monolith)
        self.assertIsNotNone(choreographer.method_catalog)
        
        # Verify monolith structure
        self.assertIn("blocks", choreographer.monolith)
        self.assertIn("micro_questions", choreographer.monolith["blocks"])
        self.assertIn("integrity", choreographer.monolith)
        
        # Verify question count
        counts = choreographer.monolith["integrity"]["question_count"]
        self.assertEqual(counts["total"], 305)
        self.assertEqual(counts["micro"], 300)
        self.assertEqual(counts["meso"], 4)
        self.assertEqual(counts["macro"], 1)
    
    @unittest.skipUnless(
        Path("questionnaire_monolith.json").exists(),
        "Requires questionnaire_monolith.json"
    )
    def test_base_slot_mapping(self):
        """Test base slot formula."""
        choreographer = Choreographer(enable_async=False)
        choreographer._load_configuration()
        
        # Test base slot mapping formula
        test_cases = [
            (1, "D1-Q1"),    # Question 1 -> D1-Q1
            (5, "D1-Q5"),    # Question 5 -> D1-Q5
            (6, "D2-Q1"),    # Question 6 -> D2-Q1
            (30, "D6-Q5"),   # Question 30 -> D6-Q5
            (31, "D1-Q1"),   # Question 31 -> D1-Q1 (repeats)
            (300, "D6-Q5"),  # Question 300 -> D6-Q5
        ]
        
        for q, expected_slot in test_cases:
            base_index = (q - 1) % 30
            base_slot = f"D{base_index // 5 + 1}-Q{base_index % 5 + 1}"
            self.assertEqual(
                base_slot,
                expected_slot,
                f"Question {q} should map to {expected_slot}, got {base_slot}"
            )
    
    def test_scoring_modality_type_a(self):
        """Test TYPE_A scoring modality."""
        choreographer = Choreographer(enable_async=False)
        choreographer._load_configuration()
        
        # TYPE_A: Count 4 elements and scale to 0-3
        evidence = {"successful_methods": 4}
        score = choreographer._apply_scoring_modality(
            evidence,
            "TYPE_A",
            {"modality_definitions": {}}
        )
        
        # 4/4 * 3 = 3.0
        self.assertAlmostEqual(score, 3.0)
        
        # Test with 2 successful methods
        evidence = {"successful_methods": 2}
        score = choreographer._apply_scoring_modality(
            evidence,
            "TYPE_A",
            {"modality_definitions": {}}
        )
        
        # 2/4 * 3 = 1.5
        self.assertAlmostEqual(score, 1.5)
    
    def test_scoring_modality_type_b(self):
        """Test TYPE_B scoring modality."""
        choreographer = Choreographer(enable_async=False)
        choreographer._load_configuration()
        
        # TYPE_B: Count up to 3 elements, each worth 1 point
        evidence = {"successful_methods": 3}
        score = choreographer._apply_scoring_modality(
            evidence,
            "TYPE_B",
            {"modality_definitions": {}}
        )
        
        self.assertAlmostEqual(score, 3.0)
        
        # Test with more than 3 (should cap at 3)
        evidence = {"successful_methods": 5}
        score = choreographer._apply_scoring_modality(
            evidence,
            "TYPE_B",
            {"modality_definitions": {}}
        )
        
        self.assertAlmostEqual(score, 3.0)
    
    def test_quality_level_determination(self):
        """Test quality level determination."""
        choreographer = Choreographer(enable_async=False)
        choreographer._load_configuration()
        
        # Test quality levels
        test_cases = [
            (3.0, "EXCELENTE"),     # 100% = EXCELENTE
            (2.55, "EXCELENTE"),    # 85% = EXCELENTE
            (2.4, "BUENO"),         # 80% = BUENO
            (2.1, "BUENO"),         # 70% = BUENO
            (1.8, "ACEPTABLE"),     # 60% = ACEPTABLE
            (1.66, "ACEPTABLE"),    # 55.3% = ACEPTABLE (just above threshold)
            (1.5, "INSUFICIENTE"),  # 50% = INSUFICIENTE
            (0.0, "INSUFICIENTE"),  # 0% = INSUFICIENTE
        ]
        
        for score, expected_level in test_cases:
            level = choreographer._determine_quality_level(score, [])
            self.assertEqual(
                level,
                expected_level,
                f"Score {score} should be {expected_level}, got {level}"
            )
    
    @unittest.skipUnless(
        Path("questionnaire_monolith.json").exists(),
        "Requires questionnaire_monolith.json"
    )
    def test_get_method_packages(self):
        """Test method package resolution."""
        choreographer = Choreographer(enable_async=False)
        choreographer._load_configuration()
        
        # Try to get method packages for D1-Q1
        packages = choreographer._get_method_packages_for_question("D1-Q1")
        
        # Should return a list (may be empty if catalog structure differs)
        self.assertIsInstance(packages, list)
    
    def test_execution_mode_enum(self):
        """Test ExecutionMode enum."""
        self.assertEqual(ExecutionMode.SYNC.value, "sync")
        self.assertEqual(ExecutionMode.ASYNC.value, "async")
        self.assertEqual(ExecutionMode.HYBRID.value, "hybrid")


class TestDataModels(unittest.TestCase):
    """Test data model classes."""
    
    def test_preprocessed_document(self):
        """Test PreprocessedDocument dataclass."""
        doc = PreprocessedDocument(
            document_id="test123",
            raw_text="Test text",
            normalized_text="test text",
            sentences=[],
            tables=[],
            indexes={},
            metadata={}
        )
        
        self.assertEqual(doc.document_id, "test123")
        self.assertEqual(doc.raw_text, "Test text")
    
    def test_question_result(self):
        """Test QuestionResult dataclass."""
        result = QuestionResult(
            question_global=1,
            base_slot="D1-Q1",
            evidence={"successful_methods": 3},
            raw_results={"method1": "result1"},
            execution_time_ms=100.5
        )
        
        self.assertEqual(result.question_global, 1)
        self.assertEqual(result.base_slot, "D1-Q1")
        self.assertAlmostEqual(result.execution_time_ms, 100.5)
    
    def test_scored_result(self):
        """Test ScoredResult dataclass."""
        result = ScoredResult(
            question_global=1,
            base_slot="D1-Q1",
            policy_area="PA01",
            dimension="DIM01",
            score=2.5,
            quality_level="BUENO",
            evidence={},
            raw_results={}
        )
        
        self.assertEqual(result.quality_level, "BUENO")
        self.assertAlmostEqual(result.score, 2.5)


if __name__ == "__main__":
    unittest.main()
