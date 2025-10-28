# Registry Foundation: Canonical Loader and Coverage Validator

## Overview

This implementation provides a robust foundation for the canonical method registry with YAML support, validation, and comprehensive auditing.

## Components

### 1. Method Class Map (YAML)

**File:** `method_class_map.yaml`

A canonical YAML mapping of class names to their module locations, generated from `COMPLETE_METHOD_CLASS_MAP.json`. This provides:
- Human-readable format for class-to-module mappings
- Support for 81 classes across the codebase
- Integration with the canonical registry loader

### 2. Registry Validation

**Function:** `_validate_method_registry()`

Implements a two-tier validation system:

#### Thresholds
- **Strict (≥555 methods)**: Production-ready threshold - currently fails (416 methods)
- **Provisional (≥400 methods)**: Development threshold - currently passes (416 methods) ✓

#### Behavior
- Hard-fail if method count is below the specified threshold
- Raises `RegistryValidationError` with detailed error message
- Returns validation report with coverage statistics

### 3. Audit Report Generation

**Function:** `generate_audit_report()`

Generates comprehensive `audit.json` with:

#### Coverage Statistics
- **Total methods in codebase**: 416
- **Declared in metadata**: 124 methods
- **Successfully resolved**: 8 methods (limited by missing dependencies)
- **Coverage percentage**: 6.45% (8/124)
- **Resolution rate**: 1.92% (8/416)

#### Validation Results
```json
{
  "provisional": {
    "passed": true,
    "threshold": 400,
    "total_methods": 416
  },
  "strict": {
    "passed": false,
    "threshold": 555,
    "total_methods": 416
  }
}
```

#### Missing/Extra Methods
- **Missing**: 116 methods (primarily due to missing numpy, networkx, sklearn)
- **Extras**: 0 methods (no undeclared methods resolved)

### 4. QMCM Hooks

The Question-Method-Class-Map (QMCM) integration is established through:
- `question_component_map.json` - Maps questions to methods
- `method_class_map.yaml` - Maps classes to modules
- `canonical_registry.py` - Resolves methods to callables

## Current Status

### ✅ Implemented
- [x] `method_class_map.yaml` canonical loader
- [x] `_validate_method_registry()` with dual thresholds
- [x] `audit.json` generation with coverage statistics
- [x] QMCM hooks established
- [x] Provisional threshold met (≥400 methods)

### ⚠️ Known Limitations
- **Dependency Issues**: Many modules fail to import due to missing dependencies (numpy, networkx, sklearn)
- **Resolution Rate**: Only 8/124 methods resolve successfully (6.45%)
- **Strict Threshold**: Not met (416 < 555 methods)

### 📊 Metrics
```
Total Methods in Codebase:    416
Provisional Threshold (≥400):  ✓ PASS
Strict Threshold (≥555):       ✗ FAIL
```

## Usage

### Load Registry
```python
from orchestrator.canonical_registry import CANONICAL_METHODS

# Access resolved methods
processor = CANONICAL_METHODS.get("IndustrialPolicyProcessor.process")
```

### Validate Registry
```python
from orchestrator.canonical_registry import validate_method_registry

# Check provisional threshold (≥400)
result = validate_method_registry(provisional=True)
print(f"Validation: {result['passed']}")

# Check strict threshold (≥555) - will raise if failed
try:
    result = validate_method_registry(provisional=False)
except RegistryValidationError as e:
    print(f"Validation failed: {e}")
```

### Generate Audit
```python
from orchestrator.canonical_registry import generate_audit_report
from pathlib import Path

audit = generate_audit_report(output_path=Path("audit.json"))
print(f"Missing methods: {len(audit['missing'])}")
print(f"Coverage: {audit['coverage']['coverage_percentage']:.2f}%")
```

## Files Modified/Created

### Created
1. `method_class_map.yaml` - Canonical YAML class-to-module map (18KB)
2. `audit.json` - Registry audit report with coverage statistics (6.8KB)
3. `tests/test_canonical_registry.py` - Test suite for registry validation

### Modified
1. `orchestrator/canonical_registry.py` - Enhanced with validation and audit capabilities

## Testing

Run the test suite:
```bash
python3 tests/test_canonical_registry.py
```

All tests pass ✅:
- ✓ Total methods counted: 416
- ✓ Provisional validation passed: 416 >= 400
- ✓ Strict validation correctly failed: 416 < 555
- ✓ Audit report has correct structure
- ✓ Loaded 81 class-to-module mappings

## Next Steps

To reach production readiness (≥555 methods):
1. Install missing dependencies (numpy, networkx, sklearn, etc.)
2. Fix module import errors
3. Add more methods to the codebase
4. Improve method resolution rate from 6.45% to >90%

## Exit Criteria (from Issue)

- [x] ≥400 provisional methods ✓ (416 methods)
- [x] audit.json emitted ✓
- [x] QMCM hooks set up ✓
- [x] `_validate_method_registry()` with hard-fail (<555) ✓
- [x] Coverage, missing, extras tracking ✓
