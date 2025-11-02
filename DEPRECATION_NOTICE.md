# DEPRECATION NOTICE: ORCHESTRATOR_MONILITH.py

## Summary
The monolithic orchestrator file `ORCHESTRATOR_MONILITH.py` has been deprecated as of this release.

## What Changed
The legacy monolithic orchestrator (`src/saaaaaa/core/orchestrator/ORCHESTRATOR_MONILITH.py`) has been refactored into a modular architecture. All functionality has been properly distributed among specialized modules in the `src/saaaaaa/core/orchestrator/` package.

## New Modular Structure
The orchestrator functionality is now distributed across:

- **`core.py`** - Main `Orchestrator` class and core orchestration logic
- **`executors.py`** - All executor classes (D1Q1_Executor through D6Q5_Executor)
- **`evidence_registry.py`** - Evidence management and tracking
- **`arg_router.py`** - Argument routing and normalization
- **`contract_loader.py`** - Contract loading and validation
- **`choreographer.py`** - Choreography and workflow logic
- **`factory.py`** - Factory functions for building components
- **`class_registry.py`** - Class registry for dynamic instantiation

## Migration Guide

### Before (Deprecated)
```python
# DO NOT USE - This is deprecated!
from saaaaaa.core.ORCHESTRATOR_MONILITH import Orchestrator, MethodExecutor
```

### After (Recommended)
```python
# Use the modular orchestrator package
from saaaaaa.core.orchestrator import Orchestrator, MethodExecutor

# Or import from the compatibility shim
from orchestrator import Orchestrator, MethodExecutor
```

## Compatibility
For backward compatibility, the `orchestrator` package provides a compatibility shim that forwards all imports to the new modular structure. This allows existing code to continue working without immediate changes.

However, **direct imports from `ORCHESTRATOR_MONILITH` will raise a `DeprecationWarning`**.

## Timeline
- **Now**: ORCHESTRATOR_MONILITH is deprecated with warnings
- **Next Release**: ORCHESTRATOR_MONILITH will be removed entirely

## Benefits of the Refactor
1. **Better Maintainability** - Smaller, focused modules are easier to understand and modify
2. **Improved Testing** - Individual modules can be tested in isolation
3. **Faster Imports** - Lazy loading reduces startup time
4. **Better IDE Support** - Clearer module boundaries improve autocomplete and navigation
5. **Reduced Coupling** - Modules have clear dependencies and responsibilities

## Questions?
If you have questions or encounter issues during migration, please open an issue on the project repository.
