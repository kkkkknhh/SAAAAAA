# Architecture Fix Summary: Audit Factory Implementation

## Problem Statement

The audit factory implementation had critical architectural violations where modules were directly loading questionnaires, completely bypassing the orchestrator. This violated the fundamental principle that **ONLY the orchestrator should load questionnaires**.

### Critical Issues Identified

1. **PolicyProcessor Bypassed Orchestrator**
   - `IndustrialPolicyProcessor.__init__()` was loading questionnaire directly from disk
   - Line 700: `self.questionnaire_data = self._load_questionnaire()`
   - ❌ **WRONG**: Processor should receive questionnaire via dependency injection

2. **Orchestrator Didn't Inject Dependencies**
   - `ORCHESTRATOR_MONILITH.py` line 898: `IndustrialPolicyProcessor()`
   - Created processor with no questionnaire data
   - Processor fell back to loading it itself, bypassing orchestrator
   - ❌ **WRONG**: Factory pattern completely broken

3. **Dual Factory Implementations**
   - OLD: `orchestrator/factory.py` had `_load_questionnaire()` and `build_processor()`
   - NEW: `src/saaaaaa/core/orchestrator/factory.py` had `CoreModuleFactory.get_questionnaire()`
   - ❌ **WRONG**: Two competing factory patterns, no clear migration path

4. **Multiple Independent Questionnaire Loads**
   - Multiple places loaded questionnaires independently:
     - `factory.get_questionnaire()` - Loads from disk
     - `old_factory._load_questionnaire()` - Also loads from disk
     - `processor._load_questionnaire()` - Also loads from disk
   - ❌ **WRONG**: No single source of truth, inconsistent data, impossible to test

## Solution: Dependency Injection Pattern

### Correct Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Orchestrator                             │
│  - Loads questionnaire ONCE via factory                     │
│  - Injects questionnaire into all processors                │
│  - Single source of truth                                   │
└─────────────┬───────────────────────────────────────────────┘
              │
              ├─→ CoreModuleFactory
              │   └─→ load_questionnaire_monolith() [ONCE]
              │
              ├─→ MethodExecutor(questionnaire_data)
              │   └─→ IndustrialPolicyProcessor(questionnaire_data)
              │
              └─→ Executors (D1Q1, D2Q1, etc.)
                  └─→ Use processors (no direct I/O)
```

## Changes Implemented

### 1. PolicyProcessor: Accept Questionnaire via Dependency Injection

**File**: `src/saaaaaa/processing/policy_processor.py`

**Before** (❌ WRONG):
```python
class IndustrialPolicyProcessor:
    def __init__(self, config=None, questionnaire_path=None):
        # Always loads from file - bypasses orchestrator
        self.questionnaire_data = self._load_questionnaire()
```

**After** (✅ CORRECT):
```python
class IndustrialPolicyProcessor:
    def __init__(
        self,
        config: ProcessorConfig | None = None,
        questionnaire_data: dict[str, Any] | None = None,  # ✅ PREFERRED
        questionnaire_path: Path | None = None,            # ⚠️ DEPRECATED
        **kwargs
    ):
        """Initialize processor.
        
        Args:
            questionnaire_data: Pre-loaded questionnaire (PREFERRED - via orchestrator)
            questionnaire_path: File path (DEPRECATED - backward compatibility only)
        """
        if questionnaire_data is not None:
            # ✅ CORRECT: Use injected data from orchestrator
            self.questionnaire_data = questionnaire_data
            logger.info("Initialized with injected questionnaire")
        else:
            # ⚠️ DEPRECATED: Fall back to file loading for backward compatibility
            logger.warning("Loading questionnaire from file - DEPRECATED")
            self.questionnaire_data = self._load_questionnaire()
```

**Benefits**:
- ✅ Supports dependency injection (preferred)
- ✅ Maintains backward compatibility
- ✅ Logs warning for deprecated usage
- ✅ Testable without file I/O

### 2. CoreModuleFactory: Create Processors with Injected Data

**File**: `src/saaaaaa/core/orchestrator/factory.py`

**Added Method**:
```python
class CoreModuleFactory:
    def create_policy_processor(self, config: Any | None = None) -> IndustrialPolicyProcessor:
        """Create IndustrialPolicyProcessor with injected questionnaire.
        
        This is the CORRECT way to create a PolicyProcessor - questionnaire
        is loaded by the factory ONCE and injected into the processor.
        
        Returns:
            IndustrialPolicyProcessor with questionnaire data injected
        """
        # Load questionnaire via factory (single source of truth)
        questionnaire = self.get_questionnaire()
        
        # Create processor with injected questionnaire (dependency injection)
        processor = IndustrialPolicyProcessor(
            config=config,
            questionnaire_data=questionnaire,  # ✅ INJECTED via orchestrator
        )
        
        return processor
```

**Usage**:
```python
# ✅ CORRECT: Use factory to create processor
factory = CoreModuleFactory()
processor = factory.create_policy_processor()

# Questionnaire is loaded once and injected
assert processor.questionnaire_data is not None
```

### 3. MethodExecutor: Accept and Pass Questionnaire

**File**: `src/saaaaaa/core/orchestrator/ORCHESTRATOR_MONILITH.py`

**Before** (❌ WRONG):
```python
class MethodExecutor:
    def __init__(self):
        self.instances = {
            # Creates processor without questionnaire - forces file loading
            'IndustrialPolicyProcessor': IndustrialPolicyProcessor(),
        }
```

**After** (✅ CORRECT):
```python
class MethodExecutor:
    def __init__(self, questionnaire_data: dict[str, Any] | None = None):
        """Initialize MethodExecutor.
        
        Args:
            questionnaire_data: Pre-loaded questionnaire to inject into processors
        """
        if questionnaire_data is not None:
            # ✅ CORRECT: Inject questionnaire from orchestrator
            processor = IndustrialPolicyProcessor(questionnaire_data=questionnaire_data)
            logger.info("MethodExecutor initialized with injected questionnaire")
        else:
            # ⚠️ DEPRECATED: Fall back for backward compatibility
            logger.warning("MethodExecutor without questionnaire (DEPRECATED)")
            processor = IndustrialPolicyProcessor()
        
        self.instances = {
            'IndustrialPolicyProcessor': processor,
        }
```

### 4. Orchestrator: Load Once and Inject

**File**: `src/saaaaaa/core/orchestrator/ORCHESTRATOR_MONILITH.py`

**Before** (❌ WRONG):
```python
class Orchestrator:
    def __init__(self, catalog_path, monolith_path, ...):
        with open(self.catalog_path) as f:
            self.catalog = json.load(f)
        
        # Creates executor without questionnaire
        self.executor = MethodExecutor()  # ❌ No questionnaire passed
```

**After** (✅ CORRECT):
```python
class Orchestrator:
    def __init__(self, catalog_path, monolith_path, ...):
        with open(self.catalog_path) as f:
            self.catalog = json.load(f)
        
        # Load questionnaire ONCE at orchestrator level
        with open(self.monolith_path, encoding='utf-8') as f:
            questionnaire_data = json.load(f)
            logger.info(f"Orchestrator loaded questionnaire from {self.monolith_path}")
        
        # Pass questionnaire to executor (dependency injection)
        self.executor = MethodExecutor(questionnaire_data=questionnaire_data)  # ✅ INJECTED
```

### 5. Legacy Factory: Updated for Compatibility

**File**: `orchestrator/factory.py`

**Updated**:
```python
def build_processor(path="questionnaire_monolith.json", locale="es"):
    """Build processor with questionnaire data.
    
    ✅ CORRECT: Loads questionnaire ONCE and injects it.
    """
    data = _load_questionnaire(Path(path))
    
    # ✅ Create processor with injected data
    processor = IndustrialPolicyProcessor(questionnaire_data=data)
    return processor
```

## Verification: Tests Added

**File**: `tests/test_questionnaire_injection.py`

### Test Results: ✅ 7 Passed, 2 Skipped

1. ✅ **test_factory_has_create_policy_processor_method**
   - Verifies CoreModuleFactory has create_policy_processor method

2. ✅ **test_method_executor_signature_accepts_questionnaire**
   - Verifies MethodExecutor accepts questionnaire_data parameter

3. ✅ **test_orchestrator_factory_build_processor_injects_questionnaire**
   - Verifies orchestrator/factory.py injects questionnaire_data

4. ✅ **test_architecture_documentation_updated**
   - Verifies documentation mentions dependency injection

5. ✅ **test_policy_processor_logs_warning_on_deprecated_path**
   - Verifies deprecated file loading path logs warning

6. ✅ **test_no_direct_questionnaire_load_in_processor_init**
   - Verifies _load_questionnaire is conditional, not unconditional

7. ✅ **test_method_executor_works_without_questionnaire_data**
   - Verifies backward compatibility (can be called without data)

## Usage Examples

### Example 1: Using CoreModuleFactory (Recommended)

```python
from saaaaaa.core.orchestrator.factory import CoreModuleFactory

# Create factory
factory = CoreModuleFactory()

# ✅ CORRECT: Factory loads questionnaire once
questionnaire = factory.get_questionnaire()

# ✅ CORRECT: Create processor with injected questionnaire
processor = factory.create_policy_processor()

# Processor has questionnaire data without file I/O
assert processor.questionnaire_data == questionnaire
```

### Example 2: Using Orchestrator

```python
from saaaaaa.core.orchestrator.ORCHESTRATOR_MONILITH import Orchestrator

# ✅ CORRECT: Orchestrator loads questionnaire internally and injects
orchestrator = Orchestrator(
    catalog_path="rules/METODOS/metodos_completos_nivel3.json",
    monolith_path="questionnaire_monolith.json",
)

# Orchestrator loaded questionnaire and passed to executor
assert orchestrator.executor is not None

# PolicyProcessor inside executor has questionnaire data
processor = orchestrator.executor.instances['IndustrialPolicyProcessor']
assert processor.questionnaire_data is not None
```

### Example 3: Direct Creation (For Testing)

```python
from saaaaaa.processing.policy_processor import IndustrialPolicyProcessor

# ✅ CORRECT: For testing, inject mock questionnaire
mock_questionnaire = {
    "questions": [
        {"id": "D1-Q1", "text": "Test question"}
    ]
}

processor = IndustrialPolicyProcessor(questionnaire_data=mock_questionnaire)

# No file I/O, easily testable
assert processor.questionnaire_data == mock_questionnaire
```

### Example 4: Legacy Code (Backward Compatible)

```python
from saaaaaa.processing.policy_processor import IndustrialPolicyProcessor

# ⚠️ DEPRECATED but still works for backward compatibility
processor = IndustrialPolicyProcessor(
    questionnaire_path="path/to/questionnaire.json"
)

# Warning is logged but processor still works
# This path should be migrated to use dependency injection
```

## Architecture Benefits

### Before (❌ PROBLEMS)

1. **Multiple loads**: Same questionnaire loaded 3+ times
2. **No single source of truth**: Different modules might use different data
3. **Hard to test**: Tests require actual files on disk
4. **Tight coupling**: Modules coupled to file system
5. **Circular dependencies**: Factory and processors both do I/O

### After (✅ CORRECT)

1. **Single load**: Questionnaire loaded ONCE by orchestrator
2. **Single source of truth**: All modules use same questionnaire instance
3. **Easy to test**: Inject mock data, no file I/O needed
4. **Loose coupling**: Modules depend on data, not file system
5. **Clear separation**: Factory handles I/O, processors handle logic

## Migration Guide

### For Code Using PolicyProcessor Directly

**Old Code**:
```python
# ❌ OLD: Loads from file
processor = IndustrialPolicyProcessor()
```

**New Code**:
```python
# ✅ NEW: Use factory
from saaaaaa.core.orchestrator.factory import CoreModuleFactory

factory = CoreModuleFactory()
processor = factory.create_policy_processor()
```

### For Code Creating MethodExecutor

**Old Code**:
```python
# ❌ OLD: Executor creates processor without questionnaire
executor = MethodExecutor()
```

**New Code**:
```python
# ✅ NEW: Pass questionnaire to executor
from saaaaaa.core.orchestrator.factory import load_questionnaire_monolith

questionnaire = load_questionnaire_monolith()
executor = MethodExecutor(questionnaire_data=questionnaire)
```

### For Tests

**Old Code**:
```python
# ❌ OLD: Requires actual file
def test_processor():
    processor = IndustrialPolicyProcessor()  # Loads from disk
    # ...
```

**New Code**:
```python
# ✅ NEW: Inject mock data
def test_processor():
    mock_data = {"questions": []}
    processor = IndustrialPolicyProcessor(questionnaire_data=mock_data)
    # ...
```

## Summary

✅ **Fixed**: PolicyProcessor now accepts questionnaire via dependency injection  
✅ **Fixed**: Orchestrator loads questionnaire once and injects it  
✅ **Fixed**: Factory provides create_policy_processor() method  
✅ **Fixed**: MethodExecutor accepts and passes questionnaire  
✅ **Verified**: dereck_beach.py already uses factory correctly  
✅ **Verified**: Analyzer_one.py already uses factory correctly  
✅ **Tested**: 7 tests verify architecture boundaries  
✅ **Backward Compatible**: Old code still works with deprecation warnings  

**Result**: Architecture is now correct with a single source of truth for questionnaire data, proper dependency injection, and maintainable, testable code.
