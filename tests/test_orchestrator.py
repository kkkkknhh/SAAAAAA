"""
Tests for the Orchestrator module.

These tests verify the core orchestrator functionality including
phase coordination, validation, and lifecycle management.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from orchestrator.orchestrator_core import (
    Orchestrator,
    OrchestratorError,
    ValidationError,
)
from orchestrator.orchestrator_types import (
    PhaseStatus,
    ProcessingPhase,
    OrchestratorConfig,
    PreprocessedDocument,
    QuestionResult,
)


class TestOrchestratorInit(unittest.TestCase):
    """Test orchestrator initialization."""
    
    def test_orchestrator_creates_with_defaults(self):
        """Test that orchestrator can be created with default config."""
        orchestrator = Orchestrator()
        
        self.assertIsNotNone(orchestrator)
        self.assertIsInstance(orchestrator.config, OrchestratorConfig)
        self.assertEqual(orchestrator.config.max_workers, 50)
        self.assertFalse(orchestrator.abort_requested)
    
    def test_orchestrator_creates_with_custom_config(self):
        """Test orchestrator with custom configuration."""
        config = OrchestratorConfig(
            max_workers=10,
            default_question_timeout=60.0,
        )
        orchestrator = Orchestrator(config=config)
        
        self.assertEqual(orchestrator.config.max_workers, 10)
        self.assertEqual(orchestrator.config.default_question_timeout, 60.0)
    
    def test_orchestrator_initializes_state(self):
        """Test that orchestrator initializes processing state."""
        orchestrator = Orchestrator()
        
        self.assertEqual(
            orchestrator.state.current_phase,
            ProcessingPhase.PHASE_0_VALIDATION
        )
        self.assertEqual(orchestrator.state.progress, 0.0)
        self.assertEqual(orchestrator.state.questions_completed, 0)
        self.assertEqual(orchestrator.state.questions_total, 300)


class TestOrchestratorValidation(unittest.TestCase):
    """Test orchestrator validation phase."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monolith_path = Path("questionnaire_monolith.json")
        self.catalog_path = Path("rules/METODOS/metodos_completos_nivel3.json")
        
        # Check if files exist
        self.files_exist = (
            self.monolith_path.exists() and
            self.catalog_path.exists()
        )
    
    @unittest.skipUnless(
        Path("questionnaire_monolith.json").exists(),
        "Requires questionnaire_monolith.json"
    )
    def test_validate_configuration_succeeds_with_valid_files(self):
        """Test validation succeeds with valid configuration files."""
        orchestrator = Orchestrator()
        
        result = orchestrator.validate_configuration()
        
        self.assertTrue(result)
        self.assertIsNotNone(orchestrator.monolith)
        self.assertIsNotNone(orchestrator.method_catalog)
    
    def test_validate_configuration_fails_missing_monolith(self):
        """Test validation fails when monolith is missing."""
        orchestrator = Orchestrator(monolith_path="nonexistent.json")
        
        with self.assertRaises(ValidationError) as ctx:
            orchestrator.validate_configuration()
        
        self.assertIn("not found", str(ctx.exception))
    
    @unittest.skipUnless(
        Path("questionnaire_monolith.json").exists(),
        "Requires questionnaire_monolith.json"
    )
    def test_validate_verifies_question_count(self):
        """Test that validation verifies 300 questions."""
        orchestrator = Orchestrator()
        
        orchestrator.validate_configuration()
        
        # Should have loaded monolith
        self.assertIsNotNone(orchestrator.monolith)
        
        # Should have 300 micro questions
        micro_count = len(orchestrator.monolith['blocks']['micro_questions'])
        self.assertEqual(micro_count, 300)


class TestOrchestratorPhaseExecution(unittest.TestCase):
    """Test orchestrator phase execution."""
    
    def test_execute_phase_tracks_metrics(self):
        """Test that phase execution tracks metrics."""
        orchestrator = Orchestrator()
        
        # Execute a simple phase
        def dummy_phase():
            return "result"
        
        result = orchestrator._execute_phase(
            ProcessingPhase.PHASE_1_INGESTION,
            dummy_phase
        )
        
        self.assertEqual(result, "result")
        
        # Check metrics were tracked
        self.assertIn(
            ProcessingPhase.PHASE_1_INGESTION,
            orchestrator.state.phase_metrics
        )
        
        metrics = orchestrator.state.phase_metrics[ProcessingPhase.PHASE_1_INGESTION]
        self.assertEqual(metrics.status, PhaseStatus.COMPLETED)
        self.assertIsNotNone(metrics.start_time)
        self.assertIsNotNone(metrics.end_time)
        self.assertGreater(metrics.duration_seconds, 0)
    
    def test_execute_phase_handles_errors(self):
        """Test that phase execution handles errors."""
        orchestrator = Orchestrator()
        
        def failing_phase():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            orchestrator._execute_phase(
                ProcessingPhase.PHASE_1_INGESTION,
                failing_phase
            )
        
        # Check metrics recorded the failure
        metrics = orchestrator.state.phase_metrics[ProcessingPhase.PHASE_1_INGESTION]
        self.assertEqual(metrics.status, PhaseStatus.FAILED)
        self.assertTrue(len(metrics.errors) > 0)
    
    def test_execute_phase_respects_abort_request(self):
        """Test that phase execution respects abort requests."""
        orchestrator = Orchestrator()
        orchestrator.abort_requested = True
        
        def dummy_phase():
            return "result"
        
        with self.assertRaises(OrchestratorError) as ctx:
            orchestrator._execute_phase(
                ProcessingPhase.PHASE_1_INGESTION,
                dummy_phase
            )
        
        self.assertIn("aborted", str(ctx.exception))


class TestOrchestratorIngestion(unittest.TestCase):
    """Test orchestrator ingestion phase."""
    
    def test_ingest_document_returns_preprocessed_doc(self):
        """Test document ingestion returns PreprocessedDocument."""
        orchestrator = Orchestrator()
        
        doc = orchestrator.ingest_document("test.pdf")
        
        self.assertIsInstance(doc, PreprocessedDocument)
        self.assertEqual(doc.document_id, "test")
        self.assertIn("source_file", doc.metadata)
    
    def test_ingest_document_sets_metadata(self):
        """Test that ingestion sets appropriate metadata."""
        orchestrator = Orchestrator()
        
        doc = orchestrator.ingest_document("path/to/document.pdf")
        
        self.assertEqual(doc.document_id, "document")
        self.assertEqual(doc.metadata["source_file"], "path/to/document.pdf")
        self.assertIn("ingestion_timestamp", doc.metadata)


class TestOrchestratorQuestionExecution(unittest.TestCase):
    """Test orchestrator question execution phase."""
    
    def test_execute_single_question_returns_result(self):
        """Test executing a single question returns QuestionResult."""
        orchestrator = Orchestrator()
        
        doc = PreprocessedDocument(
            document_id="test",
            raw_text="",
            sentences=[],
            tables=[],
            metadata={}
        )
        
        result = orchestrator._execute_single_question(1, doc)
        
        self.assertIsInstance(result, QuestionResult)
        self.assertEqual(result.question_global, 1)
        self.assertEqual(result.base_slot, "D1-Q1")
    
    def test_get_base_slot_calculation(self):
        """Test base slot calculation for question numbers."""
        orchestrator = Orchestrator()
        
        # Test first question of each dimension
        self.assertEqual(orchestrator._get_base_slot(1), "D1-Q1")
        self.assertEqual(orchestrator._get_base_slot(6), "D2-Q1")
        self.assertEqual(orchestrator._get_base_slot(11), "D3-Q1")
        self.assertEqual(orchestrator._get_base_slot(16), "D4-Q1")
        self.assertEqual(orchestrator._get_base_slot(21), "D5-Q1")
        self.assertEqual(orchestrator._get_base_slot(26), "D6-Q1")
        
        # Test cycling back
        self.assertEqual(orchestrator._get_base_slot(31), "D1-Q1")
        
        # Test last questions
        self.assertEqual(orchestrator._get_base_slot(5), "D1-Q5")
        self.assertEqual(orchestrator._get_base_slot(30), "D6-Q5")


class TestOrchestratorState(unittest.TestCase):
    """Test orchestrator state management."""
    
    def test_get_processing_status_returns_current_state(self):
        """Test getting processing status."""
        orchestrator = Orchestrator()
        
        status = orchestrator.get_processing_status()
        
        self.assertEqual(status.questions_total, 300)
        self.assertEqual(status.questions_completed, 0)
        self.assertEqual(status.progress, 0.0)
    
    def test_get_metrics_returns_comprehensive_info(self):
        """Test getting comprehensive metrics."""
        orchestrator = Orchestrator()
        
        metrics = orchestrator.get_metrics()
        
        self.assertIn('phase_metrics', metrics)
        self.assertIn('total_questions', metrics)
        self.assertIn('completed_questions', metrics)
        self.assertIn('progress', metrics)
        self.assertEqual(metrics['total_questions'], 300)
    
    def test_request_abort_sets_flag(self):
        """Test requesting abort sets the flag."""
        orchestrator = Orchestrator()
        
        self.assertFalse(orchestrator.abort_requested)
        
        orchestrator.request_abort()
        
        self.assertTrue(orchestrator.abort_requested)


class TestOrchestratorConfig(unittest.TestCase):
    """Test orchestrator configuration."""
    
    def test_config_has_default_values(self):
        """Test configuration has sensible defaults."""
        config = OrchestratorConfig()
        
        self.assertEqual(config.max_workers, 50)
        self.assertEqual(config.min_workers, 10)
        self.assertEqual(config.default_question_timeout, 180.0)
        self.assertEqual(config.max_question_retries, 3)
        self.assertEqual(config.min_completion_rate, 0.9)
        self.assertTrue(config.allow_partial_report)
    
    def test_config_can_be_customized(self):
        """Test configuration can be customized."""
        config = OrchestratorConfig(
            max_workers=100,
            min_completion_rate=0.95,
            log_level="DEBUG"
        )
        
        self.assertEqual(config.max_workers, 100)
        self.assertEqual(config.min_completion_rate, 0.95)
        self.assertEqual(config.log_level, "DEBUG")


class TestProcessingPhases(unittest.TestCase):
    """Test processing phase enumeration."""
    
    def test_all_phases_defined(self):
        """Test that all 11 phases are defined."""
        phases = list(ProcessingPhase)
        
        self.assertEqual(len(phases), 11)
        self.assertEqual(phases[0].value, 0)
        self.assertEqual(phases[10].value, 10)
    
    def test_phase_names_are_descriptive(self):
        """Test that phase names are descriptive."""
        self.assertEqual(
            ProcessingPhase.PHASE_0_VALIDATION.name,
            "PHASE_0_VALIDATION"
        )
        self.assertEqual(
            ProcessingPhase.PHASE_1_INGESTION.name,
            "PHASE_1_INGESTION"
        )


if __name__ == "__main__":
    unittest.main()
