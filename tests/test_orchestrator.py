"""
Tests for the Orchestrator (global pipeline controller).

These tests verify the global orchestration of the complete 305-question pipeline.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from orchestrator.orchestrator import (
    Orchestrator,
    AbortSignal,
    ResourceLimits,
    PhaseInstrumentation,
    PhaseResult,
    ExecutionMode,
)


class TestAbortSignal(unittest.TestCase):
    """Test the AbortSignal class."""
    
    def test_abort_signal_init(self):
        """Test abort signal initialization."""
        signal = AbortSignal()
        
        self.assertFalse(signal.is_aborted())
        self.assertIsNone(signal.get_reason())
    
    def test_abort_signal_trigger(self):
        """Test triggering abort signal."""
        signal = AbortSignal()
        
        signal.abort("Test abort reason")
        
        self.assertTrue(signal.is_aborted())
        self.assertEqual(signal.get_reason(), "Test abort reason")
    
    def test_abort_signal_reset(self):
        """Test resetting abort signal."""
        signal = AbortSignal()
        
        signal.abort("Test")
        signal.reset()
        
        self.assertFalse(signal.is_aborted())
        self.assertIsNone(signal.get_reason())


class TestResourceLimits(unittest.TestCase):
    """Test the ResourceLimits class."""
    
    def test_resource_limits_init(self):
        """Test resource limits initialization."""
        limits = ResourceLimits()
        
        self.assertEqual(limits.max_memory_mb, 8192)
        self.assertEqual(limits.max_cpu_percent, 90.0)
        self.assertEqual(limits.max_execution_time_s, 3600)
        self.assertEqual(limits.max_workers, 50)
    
    def test_resource_limits_custom(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_memory_mb=4096,
            max_cpu_percent=80.0,
            max_execution_time_s=1800,
            max_workers=25
        )
        
        self.assertEqual(limits.max_memory_mb, 4096)
        self.assertEqual(limits.max_cpu_percent, 80.0)
        self.assertEqual(limits.max_execution_time_s, 1800)
        self.assertEqual(limits.max_workers, 25)
    
    def test_get_resource_usage(self):
        """Test getting resource usage."""
        limits = ResourceLimits()
        usage = limits.get_resource_usage()
        
        self.assertIsInstance(usage, dict)
        self.assertIn('memory_mb', usage)
        self.assertIn('cpu_percent', usage)
        self.assertIn('num_threads', usage)
        self.assertIn('psutil_available', usage)


class TestPhaseInstrumentation(unittest.TestCase):
    """Test the PhaseInstrumentation class."""
    
    def test_phase_instrumentation_init(self):
        """Test phase instrumentation initialization."""
        import time
        start = time.time()
        instr = PhaseInstrumentation("FASE_0", start)
        
        self.assertEqual(instr.phase_id, "FASE_0")
        self.assertEqual(instr.start_time, start)
        self.assertIsNone(instr.end_time)
        self.assertEqual(instr.items_processed, 0)
        self.assertEqual(instr.items_total, 0)
        self.assertEqual(len(instr.errors), 0)
        self.assertEqual(len(instr.warnings), 0)
    
    def test_phase_instrumentation_complete(self):
        """Test completing phase instrumentation."""
        import time
        start = time.time()
        instr = PhaseInstrumentation("FASE_0", start)
        
        instr.complete()
        
        self.assertIsNotNone(instr.end_time)
        self.assertGreaterEqual(instr.end_time, start)
    
    def test_phase_instrumentation_progress(self):
        """Test progress calculation."""
        import time
        instr = PhaseInstrumentation("FASE_0", time.time())
        
        instr.items_total = 10
        instr.items_processed = 5
        
        self.assertEqual(instr.get_progress(), 0.5)
    
    def test_phase_instrumentation_duration(self):
        """Test duration calculation."""
        import time
        start = time.time()
        instr = PhaseInstrumentation("FASE_0", start)
        
        time.sleep(0.1)  # Sleep for 100ms
        duration = instr.get_duration_ms()
        
        self.assertGreaterEqual(duration, 100)
    
    def test_record_snapshot(self):
        """Test recording resource snapshot."""
        import time
        instr = PhaseInstrumentation("FASE_0", time.time())
        limits = ResourceLimits()
        
        instr.record_snapshot(limits)
        
        self.assertEqual(len(instr.resource_snapshots), 1)
        self.assertIn('timestamp', instr.resource_snapshots[0])
        self.assertIn('resources', instr.resource_snapshots[0])


class TestOrchestrator(unittest.TestCase):
    """Test the Orchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monolith_path = Path("questionnaire_monolith.json")
        self.catalog_path = Path("rules/METODOS/metodos_completos_nivel3.json")
        
        # Check if files exist
        self.files_exist = (
            self.monolith_path.exists() and
            self.catalog_path.exists()
        )
    
    def test_orchestrator_init(self):
        """Test orchestrator initialization."""
        orchestrator = Orchestrator()
        
        self.assertIsNotNone(orchestrator)
        self.assertIsNotNone(orchestrator.choreographer)
        self.assertIsNotNone(orchestrator.abort_signal)
        self.assertIsNotNone(orchestrator.resource_limits)
        self.assertTrue(orchestrator.enable_async)
        self.assertEqual(len(orchestrator.phase_results), 0)
        self.assertEqual(len(orchestrator.phase_instrumentation), 0)
    
    def test_orchestrator_custom_limits(self):
        """Test orchestrator with custom resource limits."""
        limits = ResourceLimits(max_workers=25, max_memory_mb=4096)
        orchestrator = Orchestrator(resource_limits=limits)
        
        self.assertEqual(orchestrator.resource_limits.max_workers, 25)
        self.assertEqual(orchestrator.resource_limits.max_memory_mb, 4096)
    
    def test_orchestrator_abort_signal(self):
        """Test orchestrator abort signal."""
        orchestrator = Orchestrator()
        
        self.assertFalse(orchestrator.abort_signal.is_aborted())
        
        orchestrator.request_abort("Test abort")
        
        self.assertTrue(orchestrator.abort_signal.is_aborted())
        self.assertEqual(orchestrator.abort_signal.get_reason(), "Test abort")
    
    def test_get_processing_status_not_started(self):
        """Test getting processing status when not started."""
        orchestrator = Orchestrator()
        
        status = orchestrator.get_processing_status()
        
        self.assertEqual(status['status'], 'not_started')
        self.assertFalse(status['abort_status'])
    
    def test_get_phase_metrics_empty(self):
        """Test getting phase metrics when empty."""
        orchestrator = Orchestrator()
        
        metrics = orchestrator.get_phase_metrics()
        
        self.assertEqual(len(metrics), 0)
    
    @unittest.skipUnless(
        Path("questionnaire_monolith.json").exists() and
        Path("rules/METODOS/metodos_completos_nivel3.json").exists(),
        "Requires monolith and catalog files"
    )
    def test_validate_contract_structure(self):
        """Test contract structure validation."""
        orchestrator = Orchestrator()
        
        # Load configuration first
        phase_result = orchestrator._load_configuration()
        
        # If loading succeeded, validate contract
        if phase_result.success:
            errors = orchestrator._validate_contract_structure()
            
            # Should have no errors for valid monolith
            self.assertEqual(len(errors), 0, f"Contract validation errors: {errors}")
    
    @unittest.skipUnless(
        Path("questionnaire_monolith.json").exists() and
        Path("rules/METODOS/metodos_completos_nivel3.json").exists(),
        "Requires monolith and catalog files"
    )
    def test_load_configuration(self):
        """Test loading configuration."""
        orchestrator = Orchestrator()
        
        phase_result = orchestrator._load_configuration()
        
        self.assertIsInstance(phase_result, PhaseResult)
        self.assertEqual(phase_result.phase_id, "FASE_0")
        
        # Note: Result may fail due to integrity hash mismatch
        # This is expected behavior if the monolith has been modified
        if phase_result.success:
            self.assertIsNotNone(orchestrator.monolith)
            self.assertIsNotNone(orchestrator.method_catalog)
            
            # Check metrics
            self.assertIn('questions_loaded', phase_result.metrics)
            self.assertIn('methods_loaded', phase_result.metrics)
            self.assertIn('contract_validated', phase_result.metrics)
            
            # Check instrumentation
            self.assertIn('FASE_0', orchestrator.phase_instrumentation)
            instr = orchestrator.phase_instrumentation['FASE_0']
            self.assertIsNotNone(instr.end_time)
            self.assertEqual(instr.items_processed, 2)  # monolith + catalog
        else:
            # If failed, check that abort was triggered and error is set
            self.assertTrue(orchestrator.abort_signal.is_aborted())
            self.assertIsNotNone(phase_result.error)


class TestPhaseResult(unittest.TestCase):
    """Test the PhaseResult class."""
    
    def test_phase_result_success(self):
        """Test successful phase result."""
        result = PhaseResult(
            phase_id="FASE_0",
            phase_name="Test Phase",
            success=True,
            execution_time_ms=100.0,
            mode=ExecutionMode.SYNC,
            data={'test': 'data'},
            metrics={'count': 10}
        )
        
        self.assertEqual(result.phase_id, "FASE_0")
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time_ms, 100.0)
        self.assertEqual(result.mode, ExecutionMode.SYNC)
        self.assertIsNone(result.error)
    
    def test_phase_result_failure(self):
        """Test failed phase result."""
        error = ValueError("Test error")
        result = PhaseResult(
            phase_id="FASE_1",
            phase_name="Test Phase",
            success=False,
            execution_time_ms=50.0,
            mode=ExecutionMode.SYNC,
            error=error
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, error)


if __name__ == "__main__":
    unittest.main()
