"""
QMCM (Quality Method Call Monitoring) hooks for ReportAssemblyProducer

Records method calls for registry tracking and quality assurance.
Does NOT summarize or analyze - only records method invocations.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from functools import wraps

logger = logging.getLogger(__name__)


class QMCMRecorder:
    """
    Records method calls for quality monitoring
    
    Tracks:
    - Method invocations
    - Call frequency
    - Input/output types
    - Execution status
    
    Does NOT track:
    - Actual data content (no summarization leakage)
    - User-specific information
    """
    
    def __init__(self, recording_path: Optional[Path] = None):
        """Initialize QMCM recorder"""
        self.recording_path = recording_path or Path(".qmcm_recording.json")
        self.calls: List[Dict[str, Any]] = []
        self.enabled = True
    
    def record_call(
            self,
            method_name: str,
            input_types: Dict[str, str],
            output_type: str,
            execution_status: str = "success",
            execution_time_ms: float = 0.0
    ):
        """
        Record a method call
        
        Args:
            method_name: Name of the method called
            input_types: Dictionary mapping parameter names to type names
            output_type: Type name of the return value
            execution_status: 'success', 'error', or 'skipped'
            execution_time_ms: Execution time in milliseconds
        """
        if not self.enabled:
            return
        
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "method_name": method_name,
            "input_types": input_types,
            "output_type": output_type,
            "execution_status": execution_status,
            "execution_time_ms": round(execution_time_ms, 2)
        }
        
        self.calls.append(call_record)
        logger.debug(f"QMCM recorded: {method_name}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get recording statistics
        
        Returns summary of method call patterns without data content
        """
        if not self.calls:
            return {
                "total_calls": 0,
                "unique_methods": 0,
                "method_frequency": {},
                "success_rate": 0.0,
                "most_called_method": None
            }
        
        method_counts = {}
        success_count = 0
        
        for call in self.calls:
            method_name = call["method_name"]
            method_counts[method_name] = method_counts.get(method_name, 0) + 1
            
            if call["execution_status"] == "success":
                success_count += 1
        
        most_called = None
        if method_counts:
            most_called = max(method_counts.items(), key=lambda x: x[1])[0]
        
        return {
            "total_calls": len(self.calls),
            "unique_methods": len(method_counts),
            "method_frequency": method_counts,
            "success_rate": success_count / len(self.calls) if self.calls else 0.0,
            "most_called_method": most_called
        }
    
    def save_recording(self):
        """Save recording to disk"""
        recording_data = {
            "recording_metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_calls": len(self.calls)
            },
            "statistics": self.get_statistics(),
            "calls": self.calls
        }
        
        with open(self.recording_path, 'w') as f:
            json.dump(recording_data, f, indent=2)
        
        logger.info(f"QMCM recording saved: {self.recording_path}")
    
    def load_recording(self):
        """Load recording from disk"""
        if not self.recording_path.exists():
            logger.warning(f"No recording found: {self.recording_path}")
            return
        
        with open(self.recording_path, 'r') as f:
            recording_data = json.load(f)
        
        self.calls = recording_data.get("calls", [])
        logger.info(f"QMCM recording loaded: {len(self.calls)} calls")
    
    def clear_recording(self):
        """Clear all recorded calls"""
        self.calls = []
        logger.info("QMCM recording cleared")
    
    def enable(self):
        """Enable recording"""
        self.enabled = True
    
    def disable(self):
        """Disable recording"""
        self.enabled = False


# Global recorder instance
_global_recorder: Optional[QMCMRecorder] = None


def get_global_recorder() -> QMCMRecorder:
    """Get or create global QMCM recorder"""
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = QMCMRecorder()
    return _global_recorder


def qmcm_record(method):
    """
    Decorator to record method calls in QMCM
    
    Usage:
        @qmcm_record
        def my_method(self, arg1: str, arg2: int) -> dict:
            return {"result": "data"}
    """
    @wraps(method)
    def wrapper(*args, **kwargs):
        recorder = get_global_recorder()
        
        import time
        start_time = time.time()
        
        try:
            result = method(*args, **kwargs)
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Record the call
            input_types = {}
            if args:
                # Skip self argument
                for i, arg in enumerate(args[1:], 1):
                    input_types[f"arg{i}"] = type(arg).__name__
            for key, value in kwargs.items():
                input_types[key] = type(value).__name__
            
            output_type = type(result).__name__
            
            recorder.record_call(
                method_name=method.__name__,
                input_types=input_types,
                output_type=output_type,
                execution_status="success",
                execution_time_ms=execution_time_ms
            )
            
            return result
        
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            
            recorder.record_call(
                method_name=method.__name__,
                input_types={},
                output_type="error",
                execution_status="error",
                execution_time_ms=execution_time_ms
            )
            
            raise
    
    return wrapper


# Export public API
__all__ = [
    'QMCMRecorder',
    'get_global_recorder',
    'qmcm_record'
]
