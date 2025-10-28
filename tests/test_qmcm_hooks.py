# tests/test_qmcm_hooks.py
# coding=utf-8
"""
Tests for QMCM (Quality Method Call Monitoring) hooks
"""

import unittest
import tempfile
from pathlib import Path
from qmcm_hooks import QMCMRecorder, qmcm_record, get_global_recorder


class TestQMCMRecorder(unittest.TestCase):
    """Test QMCM recording functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.recording_path = Path(self.temp_dir) / "test_recording.json"
        self.recorder = QMCMRecorder(recording_path=self.recording_path)
    
    def test_recorder_initialization(self):
        """Test recorder initializes correctly"""
        self.assertIsNotNone(self.recorder)
        self.assertTrue(self.recorder.enabled)
        self.assertEqual(len(self.recorder.calls), 0)
    
    def test_record_call(self):
        """Test recording a method call"""
        self.recorder.record_call(
            method_name="test_method",
            input_types={"arg1": "str", "arg2": "int"},
            output_type="dict",
            execution_status="success",
            execution_time_ms=1.5
        )
        
        self.assertEqual(len(self.recorder.calls), 1)
        call = self.recorder.calls[0]
        self.assertEqual(call["method_name"], "test_method")
        self.assertEqual(call["execution_status"], "success")
        self.assertEqual(call["execution_time_ms"], 1.5)
    
    def test_get_statistics(self):
        """Test getting recording statistics"""
        # Record multiple calls
        self.recorder.record_call("method1", {}, "str", "success", 1.0)
        self.recorder.record_call("method1", {}, "str", "success", 1.5)
        self.recorder.record_call("method2", {}, "int", "success", 2.0)
        self.recorder.record_call("method1", {}, "str", "error", 0.5)
        
        stats = self.recorder.get_statistics()
        
        self.assertEqual(stats["total_calls"], 4)
        self.assertEqual(stats["unique_methods"], 2)
        self.assertEqual(stats["method_frequency"]["method1"], 3)
        self.assertEqual(stats["method_frequency"]["method2"], 1)
        self.assertEqual(stats["most_called_method"], "method1")
        self.assertEqual(stats["success_rate"], 0.75)  # 3/4 successful
    
    def test_save_and_load_recording(self):
        """Test saving and loading recording"""
        # Record some calls
        self.recorder.record_call("test_method", {}, "str", "success", 1.0)
        self.recorder.save_recording()
        
        # Create new recorder and load
        new_recorder = QMCMRecorder(recording_path=self.recording_path)
        new_recorder.load_recording()
        
        self.assertEqual(len(new_recorder.calls), 1)
        self.assertEqual(new_recorder.calls[0]["method_name"], "test_method")
    
    def test_clear_recording(self):
        """Test clearing recording"""
        self.recorder.record_call("test_method", {}, "str", "success", 1.0)
        self.assertEqual(len(self.recorder.calls), 1)
        
        self.recorder.clear_recording()
        self.assertEqual(len(self.recorder.calls), 0)
    
    def test_enable_disable(self):
        """Test enabling and disabling recording"""
        self.assertTrue(self.recorder.enabled)
        
        self.recorder.disable()
        self.assertFalse(self.recorder.enabled)
        
        # Recording should be skipped when disabled
        self.recorder.record_call("test_method", {}, "str", "success", 1.0)
        self.assertEqual(len(self.recorder.calls), 0)
        
        self.recorder.enable()
        self.assertTrue(self.recorder.enabled)
        
        self.recorder.record_call("test_method", {}, "str", "success", 1.0)
        self.assertEqual(len(self.recorder.calls), 1)
    
    def test_statistics_empty(self):
        """Test statistics with no calls"""
        stats = self.recorder.get_statistics()
        
        self.assertEqual(stats["total_calls"], 0)
        self.assertEqual(stats["unique_methods"], 0)
        self.assertEqual(stats["method_frequency"], {})
        self.assertEqual(stats["success_rate"], 0.0)
        self.assertIsNone(stats["most_called_method"])


class TestQMCMDecorator(unittest.TestCase):
    """Test QMCM decorator functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        recorder = get_global_recorder()
        recorder.clear_recording()
    
    def test_decorator_records_call(self):
        """Test that decorator records method calls"""
        
        class TestClass:
            @qmcm_record
            def test_method(self, arg1: str, arg2: int) -> dict:
                return {"result": "success"}
        
        obj = TestClass()
        result = obj.test_method("test", 42)
        
        self.assertEqual(result, {"result": "success"})
        
        recorder = get_global_recorder()
        self.assertGreater(len(recorder.calls), 0)
        
        last_call = recorder.calls[-1]
        self.assertEqual(last_call["method_name"], "test_method")
        self.assertEqual(last_call["execution_status"], "success")
        self.assertEqual(last_call["output_type"], "dict")
    
    def test_decorator_records_error(self):
        """Test that decorator records errors"""
        
        class TestClass:
            @qmcm_record
            def failing_method(self):
                raise ValueError("Test error")
        
        obj = TestClass()
        
        with self.assertRaises(ValueError):
            obj.failing_method()
        
        recorder = get_global_recorder()
        last_call = recorder.calls[-1]
        self.assertEqual(last_call["method_name"], "failing_method")
        self.assertEqual(last_call["execution_status"], "error")
    
    def test_no_summarization_leakage(self):
        """Ensure QMCM does not record actual data content"""
        
        class TestClass:
            @qmcm_record
            def method_with_data(self, sensitive_data: str) -> str:
                return f"processed: {sensitive_data}"
        
        obj = TestClass()
        obj.method_with_data("SECRET_DATA")
        
        recorder = get_global_recorder()
        last_call = recorder.calls[-1]
        
        # Check that actual data content is NOT recorded
        call_str = str(last_call)
        self.assertNotIn("SECRET_DATA", call_str)
        self.assertNotIn("processed:", call_str)
        
        # Only type information should be present
        self.assertEqual(last_call["input_types"]["arg1"], "str")
        self.assertEqual(last_call["output_type"], "str")


if __name__ == "__main__":
    unittest.main()
