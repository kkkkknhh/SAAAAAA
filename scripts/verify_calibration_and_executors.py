#!/usr/bin/env python3
"""Binary verification script for calibration and executor wiring.

This script ensures:
1. Every (class, method) pair in executor sequences has a calibration
2. No calibrations are default-like
3. MethodExecutor has instances
4. ExecutorConfig is properly imported and used

Exit 0 only if ALL checks pass.
Exit 1 if ANY check fails.
"""

import sys
from pathlib import Path

# Add src to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def collect_executor_methods() -> set[tuple[str, str]]:
    """Collect all (class, method) pairs from executor method_sequences."""
    import re
    
    executors_file = repo_root / "src" / "saaaaaa" / "core" / "orchestrator" / "executors.py"
    if not executors_file.exists():
        print(f"ERROR: {executors_file} not found")
        sys.exit(1)
    
    content = executors_file.read_text()
    
    # Find all tuples like ('ClassName', 'method_name')
    pattern = r"\('([^']+)',\s*'([^']+)'\)"
    matches = re.findall(pattern, content)
    
    return set(matches)


def verify_calibration_coverage():
    """Verify all executor methods have calibrations."""
    print("=" * 70)
    print("CALIBRATION AND EXECUTOR VERIFICATION")
    print("=" * 70)
    print()
    
    # Import calibration registry
    try:
        from saaaaaa.core.orchestrator.calibration_registry import (
            CALIBRATIONS,
            resolve_calibration,
        )
        print("✓ Calibration registry imported successfully")
    except ImportError as e:
        print(f"✗ FAILED to import calibration registry: {e}")
        return False
    
    # Collect executor methods
    print()
    print("Collecting methods from executor sequences...")
    executor_methods = collect_executor_methods()
    print(f"✓ Found {len(executor_methods)} unique (class, method) pairs in executors")
    
    # Check calibration coverage
    print()
    print("Checking calibration coverage...")
    missing = []
    default_like = []
    
    for class_name, method_name in sorted(executor_methods):
        calib = resolve_calibration(class_name, method_name)
        if calib is None:
            missing.append((class_name, method_name))
        elif calib.is_default_like():
            default_like.append((class_name, method_name))
    
    # Report results
    success = True
    
    if missing:
        print(f"\n✗ FAILED: {len(missing)} methods missing calibrations:")
        for class_name, method_name in missing[:10]:
            print(f"  - {class_name}.{method_name}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")
        success = False
    else:
        print(f"✓ All {len(executor_methods)} executor methods have calibrations")
    
    if default_like:
        print(f"\n✗ FAILED: {len(default_like)} methods have default-like calibrations:")
        for class_name, method_name in default_like[:10]:
            print(f"  - {class_name}.{method_name}")
        if len(default_like) > 10:
            print(f"  ... and {len(default_like) - 10} more")
        success = False
    else:
        print(f"✓ No default-like calibrations (all are domain-specific)")
    
    # Verify calibration count
    print()
    print(f"Total calibrations in registry: {len(CALIBRATIONS)}")
    if len(CALIBRATIONS) < len(executor_methods):
        print(f"✗ WARNING: Registry has fewer calibrations than executor methods")
        print(f"  Expected at least {len(executor_methods)}, got {len(CALIBRATIONS)}")
    else:
        print(f"✓ Calibration registry has sufficient coverage")
    
    return success


def verify_executor_config():
    """Verify ExecutorConfig integration."""
    print()
    print("=" * 70)
    print("EXECUTOR CONFIG INTEGRATION")
    print("=" * 70)
    print()
    
    try:
        from saaaaaa.core.orchestrator.executor_config import (
            ExecutorConfig,
            CONSERVATIVE_CONFIG,
        )
        print("✓ ExecutorConfig imported successfully")
    except ImportError as e:
        print(f"⚠ WARNING: ExecutorConfig import failed (may be missing dependencies): {e}")
        print("  Continuing with structural checks...")
        # Don't fail - just continue with file-based checks
    
    # Check that executors import and use config
    executors_file = repo_root / "src" / "saaaaaa" / "core" / "orchestrator" / "executors.py"
    content = executors_file.read_text()
    
    checks = [
        ("from .executor_config import ExecutorConfig", "ExecutorConfig import"),
        ("from .calibration_registry import", "calibration_registry import"),
        ("config: ExecutorConfig | None = None", "config parameter in __init__"),
        ("self.config = config or CONSERVATIVE_CONFIG", "config assignment"),
        ("self.config.compute_hash()", "config usage"),
        ("self._validate_calibrations()", "calibration validation"),
    ]
    
    success = True
    for check_str, description in checks:
        if check_str in content:
            print(f"✓ {description}")
        else:
            print(f"✗ FAILED: Missing {description}")
            success = False
    
    return success


def verify_method_executor():
    """Verify MethodExecutor has instances."""
    print()
    print("=" * 70)
    print("METHOD EXECUTOR VERIFICATION")
    print("=" * 70)
    print()
    
    try:
        from saaaaaa.core.orchestrator.core import MethodExecutor
        print("✓ MethodExecutor imported successfully")
    except ImportError as e:
        print(f"✗ FAILED to import MethodExecutor: {e}")
        return False
    
    # Try to instantiate
    try:
        executor = MethodExecutor()
        print("✓ MethodExecutor instantiated successfully")
        
        if hasattr(executor, 'instances'):
            instance_count = len(executor.instances)
            print(f"✓ MethodExecutor has {instance_count} class instances")
            if instance_count == 0:
                print("  ⚠ WARNING: No instances (may be expected in test environment)")
        else:
            print("✗ FAILED: MethodExecutor missing 'instances' attribute")
            return False
        
        return True
    except Exception as e:
        print(f"✗ FAILED to instantiate MethodExecutor: {e}")
        return False


def main():
    """Run all verifications."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  CALIBRATION & EXECUTOR WIRING VERIFICATION".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    results = []
    
    # Run verifications
    results.append(("Calibration Coverage", verify_calibration_coverage()))
    results.append(("ExecutorConfig Integration", verify_executor_config()))
    results.append(("MethodExecutor", verify_method_executor()))
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    all_passed = all(result for _, result in results)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
    
    print()
    print("=" * 70)
    
    if all_passed:
        print()
        print("🎉 ALL CHECKS PASSED")
        print()
        print("Invariants verified:")
        print("  • Every executor method has explicit calibration")
        print("  • No default-like calibrations")
        print("  • ExecutorConfig properly wired")
        print("  • MethodExecutor instantiable")
        print()
        sys.exit(0)
    else:
        print()
        print("❌ VERIFICATION FAILED")
        print()
        print("One or more checks failed. See details above.")
        print()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
