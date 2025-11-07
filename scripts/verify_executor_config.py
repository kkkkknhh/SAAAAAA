#!/usr/bin/env python3
"""Verification script for ExecutorConfig integration.

This script verifies that ExecutorConfig is properly connected to all executors
and that the configuration system works as expected.

Expected output: VERIFICATION PASSED
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def verify_imports():
    """Verify that ExecutorConfig can be imported from executors module."""
    try:
        from saaaaaa.core.orchestrator.executors import ExecutorConfig, CONSERVATIVE_CONFIG
        logger.info("✓ ExecutorConfig and CONSERVATIVE_CONFIG imported successfully")
        return True, ExecutorConfig, CONSERVATIVE_CONFIG
    except ImportError as e:
        logger.error(f"✗ Failed to import ExecutorConfig: {e}")
        return False, None, None


def verify_executor_config_class(ExecutorConfig):
    """Verify ExecutorConfig has required fields and methods."""
    required_fields = [
        'max_tokens', 'temperature', 'timeout_s', 'retry', 'policy_area',
        'regex_pack', 'thresholds', 'entities_whitelist', 'enable_symbolic_sparse',
        'seed', 'require_calibration', 'fail_on_missing_calibration', 'method_constraints'
    ]
    
    required_methods = [
        'compute_hash', 'describe', 'merge_overrides', 'from_env',
        'from_cli_args', 'validate_latency_budget', 'get_method_constraints'
    ]
    
    # Check fields
    for field in required_fields:
        if field not in ExecutorConfig.model_fields:
            logger.error(f"✗ ExecutorConfig missing field: {field}")
            return False
    logger.info(f"✓ ExecutorConfig has all required fields ({len(required_fields)} fields)")
    
    # Check methods
    for method in required_methods:
        if not hasattr(ExecutorConfig, method):
            logger.error(f"✗ ExecutorConfig missing method: {method}")
            return False
    logger.info(f"✓ ExecutorConfig has all required methods ({len(required_methods)} methods)")
    
    return True


def verify_conservative_config(CONSERVATIVE_CONFIG):
    """Verify CONSERVATIVE_CONFIG is properly defined."""
    if CONSERVATIVE_CONFIG is None:
        logger.error("✗ CONSERVATIVE_CONFIG is None")
        return False
    
    # Check expected conservative values
    if CONSERVATIVE_CONFIG.max_tokens != 1024:
        logger.error(f"✗ CONSERVATIVE_CONFIG.max_tokens should be 1024, got {CONSERVATIVE_CONFIG.max_tokens}")
        return False
    
    if CONSERVATIVE_CONFIG.temperature != 0.0:
        logger.error(f"✗ CONSERVATIVE_CONFIG.temperature should be 0.0, got {CONSERVATIVE_CONFIG.temperature}")
        return False
    
    if CONSERVATIVE_CONFIG.timeout_s != 15.0:
        logger.error(f"✗ CONSERVATIVE_CONFIG.timeout_s should be 15.0, got {CONSERVATIVE_CONFIG.timeout_s}")
        return False
    
    if CONSERVATIVE_CONFIG.retry != 1:
        logger.error(f"✗ CONSERVATIVE_CONFIG.retry should be 1, got {CONSERVATIVE_CONFIG.retry}")
        return False
    
    logger.info("✓ CONSERVATIVE_CONFIG has correct conservative values")
    return True


def verify_executor_integration(ExecutorConfig):
    """Verify that executors properly integrate ExecutorConfig."""
    try:
        from saaaaaa.core.orchestrator.executors import (
            AdvancedDataFlowExecutor,
            D1Q1_Executor,
            D2Q3_Executor,
            D5Q2_Executor,
        )
        
        # Check that AdvancedDataFlowExecutor.__init__ accepts config parameter
        import inspect
        init_signature = inspect.signature(AdvancedDataFlowExecutor.__init__)
        if 'config' not in init_signature.parameters:
            logger.error("✗ AdvancedDataFlowExecutor.__init__ does not accept 'config' parameter")
            return False
        
        logger.info("✓ AdvancedDataFlowExecutor.__init__ accepts 'config' parameter")
        
        # Check a few executor subclasses
        for executor_class in [D1Q1_Executor, D2Q3_Executor, D5Q2_Executor]:
            init_signature = inspect.signature(executor_class.__init__)
            if 'config' not in init_signature.parameters:
                logger.error(f"✗ {executor_class.__name__}.__init__ does not accept 'config' parameter")
                return False
        
        logger.info("✓ Executor subclasses accept 'config' parameter")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to verify executor integration: {e}")
        return False


def verify_config_usage(ExecutorConfig, CONSERVATIVE_CONFIG):
    """Verify that config is actually used in executors."""
    try:
        from saaaaaa.core.orchestrator.executors import AdvancedDataFlowExecutor
        import inspect
        
        # Check that execute_with_optimization uses self.config
        source = inspect.getsource(AdvancedDataFlowExecutor.execute_with_optimization)
        
        if 'self.config.retry' not in source:
            logger.error("✗ execute_with_optimization does not use self.config.retry")
            return False
        
        if 'self.config.timeout_s' not in source:
            logger.error("✗ execute_with_optimization does not use self.config.timeout_s")
            return False
        
        logger.info("✓ Executors use config.retry and config.timeout_s")
        
        # Check for signal.alarm usage
        if 'signal.alarm' not in source:
            logger.error("✗ execute_with_optimization does not use signal.alarm for timeout enforcement")
            return False
        
        logger.info("✓ Executors enforce timeout via signal.alarm")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to verify config usage: {e}")
        return False


def verify_core_integration():
    """Verify that core.py MethodExecutor.execute handles executor_config."""
    try:
        from saaaaaa.core.orchestrator.core import MethodExecutor
        import inspect
        
        # Check that execute method handles executor_config
        source = inspect.getsource(MethodExecutor.execute)
        
        if "executor_config = kwargs.pop('executor_config'" not in source:
            logger.error("✗ MethodExecutor.execute does not pop 'executor_config' from kwargs")
            return False
        
        if 'get_method_constraints' not in source:
            logger.error("✗ MethodExecutor.execute does not call get_method_constraints")
            return False
        
        if "'_method_constraints'" not in source:
            logger.error("✗ MethodExecutor.execute does not pass constraints via '_method_constraints'")
            return False
        
        if 'fail_on_missing_calibration' not in source:
            logger.error("✗ MethodExecutor.execute does not respect fail_on_missing_calibration")
            return False
        
        logger.info("✓ MethodExecutor.execute properly handles executor_config")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to verify core integration: {e}")
        return False


def verify_config_hash():
    """Verify that config hash computation works."""
    try:
        from saaaaaa.core.orchestrator.executors import ExecutorConfig
        
        config1 = ExecutorConfig(max_tokens=2048, temperature=0.7)
        config2 = ExecutorConfig(max_tokens=2048, temperature=0.7)
        config3 = ExecutorConfig(max_tokens=4096, temperature=0.7)
        
        hash1 = config1.compute_hash()
        hash2 = config2.compute_hash()
        hash3 = config3.compute_hash()
        
        if hash1 != hash2:
            logger.error("✗ Identical configs produce different hashes")
            return False
        
        if hash1 == hash3:
            logger.error("✗ Different configs produce identical hashes")
            return False
        
        if len(hash1) != 64:  # BLAKE3 produces 64-char hex string
            logger.error(f"✗ Config hash has unexpected length: {len(hash1)}")
            return False
        
        logger.info("✓ Config hash computation works correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to verify config hash: {e}")
        return False


def main():
    """Run all verification checks."""
    logger.info("=" * 70)
    logger.info("ExecutorConfig Integration Verification")
    logger.info("=" * 70)
    
    checks = []
    
    # Import check
    success, ExecutorConfig, CONSERVATIVE_CONFIG = verify_imports()
    checks.append(success)
    if not success:
        logger.error("\nVERIFICATION FAILED: Cannot import ExecutorConfig")
        return 1
    
    # ExecutorConfig class structure
    checks.append(verify_executor_config_class(ExecutorConfig))
    
    # CONSERVATIVE_CONFIG values
    checks.append(verify_conservative_config(CONSERVATIVE_CONFIG))
    
    # Executor integration
    checks.append(verify_executor_integration(ExecutorConfig))
    
    # Config usage in executors
    checks.append(verify_config_usage(ExecutorConfig, CONSERVATIVE_CONFIG))
    
    # Core.py integration
    checks.append(verify_core_integration())
    
    # Config hash computation
    checks.append(verify_config_hash())
    
    logger.info("=" * 70)
    
    if all(checks):
        logger.info("✓ VERIFICATION PASSED")
        logger.info("All ExecutorConfig integration checks passed successfully")
        return 0
    else:
        logger.error("✗ VERIFICATION FAILED")
        logger.error(f"Passed: {sum(checks)}/{len(checks)} checks")
        return 1


if __name__ == "__main__":
    sys.exit(main())
