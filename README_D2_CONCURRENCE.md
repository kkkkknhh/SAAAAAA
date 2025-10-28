# D2 Method Concurrence - Quick Start

## What is this?

This is a **strict method concurrence enforcement system** for D2 (Diseño de Actividades y Coherencia) questions, implementing the **SIN_CARRETA doctrine**:

- ✅ **107 method calls** across 5 D2 questions validated
- ✅ **No graceful degradation** - all methods must be present
- ✅ **Deterministic execution** - predictable, auditable orchestration
- ✅ **Explicit failure** - contract violations abort with diagnostics

## Quick Validation

### Option 1: CLI Script

```bash
# Validate in strict mode (production)
python scripts/validate_d2_concurrence.py --strict --summary

# Validate in non-strict mode (development)
python scripts/validate_d2_concurrence.py --summary

# Generate JSON report
python scripts/validate_d2_concurrence.py \
  --output d2_validation_report.json \
  --summary
```

### Option 2: Python API

```python
from orchestrator.d2_activities_orchestrator import validate_d2_orchestration

# Run validation
success = validate_d2_orchestration(
    strict_mode=True,
    output_report="d2_validation.json"
)

if success:
    print("✅ All D2 methods validated")
else:
    print("❌ Some methods are missing")
```

### Option 3: Integration Hook

```python
from orchestrator.d2_integration import integrate_d2_validation

# Integrate into existing orchestrator
orchestrator = PolicyAnalysisOrchestrator(config)

d2_hook = integrate_d2_validation(
    orchestrator,
    strict_mode=True,
    validate_on_init=True
)

# Check validation results
summary = d2_hook.get_validation_summary()
print(f"Success rate: {summary['success_rate'] * 100:.1f}%")
```

## Method Breakdown by Question

| Question | Methods | Description |
|----------|---------|-------------|
| **D2-Q1** | 20 | Formato Tabular y Trazabilidad |
| **D2-Q2** | 25 | Causalidad de Actividades |
| **D2-Q3** | 18 | Clasificación Temática |
| **D2-Q4** | 20 | Riesgos y Mitigación |
| **D2-Q5** | 24 | Coherencia Estratégica |
| **Total** | **107** | (with some methods reused) |

## Module Participation

```
policy_processor.py          ████████████░░ 22 methods
financiero_viabilidad_tablas ████████████░░ 20 methods
contradiction_deteccion.py   ████████████░░ 36 methods
dereck_beach.py             ████████░░░░░░ 12 methods
Analyzer_one.py             ██████░░░░░░░░ 14 methods
embedding_policy.py         ████░░░░░░░░░░  6 methods
semantic_chunking_policy.py ██░░░░░░░░░░░░  1 method
```

## Files Added

### Core Modules
- `orchestrator/d2_activities_orchestrator.py` - Main orchestration engine
- `orchestrator/d2_integration.py` - Integration hooks for PolicyAnalysisOrchestrator

### Scripts
- `scripts/validate_d2_concurrence.py` - CI/CD validation script

### Tests
- `tests/test_d2_orchestrator.py` - Unit tests for orchestrator
- `tests/test_d2_integration.py` - Integration hook tests

### Documentation
- `docs/D2_METHOD_CONCURRENCE.md` - Comprehensive guide

### CI/CD
- `.github/workflows/d2_concurrence.yml` - GitHub Actions workflow

## Running Tests

```bash
# Test orchestrator
python tests/test_d2_orchestrator.py

# Test integration hooks
python tests/test_d2_integration.py

# All tests
python tests/test_d2_orchestrator.py && python tests/test_d2_integration.py
```

## CI/CD Integration

The GitHub Actions workflow validates D2 method concurrence on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Changes to any D2-related files

### Workflow Jobs

1. **validate-d2-strict** - Strict validation (all methods must be present)
2. **validate-d2-non-strict** - Non-strict validation (95% threshold)
3. **test-d2-orchestrator** - Run unit tests

## SIN_CARRETA Doctrine

This implementation follows strict doctrine:

### ✅ Contract Conditions Met

1. **No graceful degradation** - All methods must execute, no fallbacks
2. **No strategic simplification** - All 107 method calls preserved
3. **Explicit failure** - Contract violations abort with diagnostics
4. **Full traceability** - Execution is observable and auditable

### ✅ Enforcement Mechanisms

- `OrchestrationError` raised on contract violations
- Strict mode aborts on first missing method
- Non-strict mode logs failures but continues (for development)
- Full execution traces with timing information
- Detailed validation reports in JSON format

## Next Steps

1. **Development**: Use non-strict mode while building/testing
2. **Integration**: Use `D2IntegrationHook` with existing orchestrator
3. **Production**: Enable strict mode in production environments
4. **CI/CD**: GitHub Actions will enforce validation automatically

## Need Help?

See full documentation: `docs/D2_METHOD_CONCURRENCE.md`

## Example Output

```
================================================================================
D2 METHOD CONCURRENCE VALIDATION
SIN_CARRETA Doctrine: No Graceful Degradation | Deterministic Execution
================================================================================

VALIDATION RESULTS
--------------------------------------------------------------------------------
Overall Success Rate: 100.0%
Total Questions: 5
Questions Passed: 5
Questions Failed: 0
Total Methods: 107
Methods Resolved: 107
Methods Failed: 0

Per-Question Results:
--------------------------------------------------------------------------------
✓ D2-Q1: 20/20 methods (100.0%)
✓ D2-Q2: 25/25 methods (100.0%)
✓ D2-Q3: 18/18 methods (100.0%)
✓ D2-Q4: 20/20 methods (100.0%)
✓ D2-Q5: 24/24 methods (100.0%)

✅ VALIDATION PASSED
Success rate: 100.0%
```
