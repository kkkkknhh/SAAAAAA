# D2 Activities Design & Coherence - Method Concurrence Enforcement

## Overview

This module enforces **strict concurrence** of methods for D2 (Diseño de Actividades y Coherencia) questions, following the **SIN_CARRETA doctrine**:

- ✅ **No graceful degradation**: All methods must execute deterministically
- ✅ **No strategic simplification**: Complexity is preserved as a design asset
- ✅ **Explicit failure semantics**: Contract violations abort execution
- ✅ **Full traceability**: All execution is observable and auditable

## Architecture

### Components

1. **`orchestrator/d2_activities_orchestrator.py`** - Core orchestration module
   - `D2MethodRegistry` - Canonical registry of required methods for each D2 question
   - `D2ActivitiesOrchestrator` - Orchestrator with strict contract enforcement
   - Method resolution and validation logic

2. **`scripts/validate_d2_concurrence.py`** - CI/CD validation script
   - Validates method existence and resolution
   - Generates validation reports
   - Configurable strict/non-strict modes

3. **`tests/test_d2_orchestrator.py`** - Test suite
   - Unit tests for registry and orchestrator
   - Validation of method counts and structure

## D2 Questions and Method Mappings

### D2-Q1: Formato Tabular y Trazabilidad (20 methods)
**Question**: ¿Las actividades se presentan en formato estructurado (tabla, BPIN, cronograma)?

**Participating Methods**:
- `policy_processor.py`: 4 methods
  - `IndustrialPolicyProcessor._match_patterns_in_sentences()`
  - `IndustrialPolicyProcessor.process()`
  - `PolicyTextProcessor.segment_into_sentences()`
  - `BayesianEvidenceScorer.compute_evidence_score()`

- `financiero_viabilidad_tablas.py`: 12 methods
  - `PDETMunicipalPlanAnalyzer.extract_tables()`
  - `PDETMunicipalPlanAnalyzer._clean_dataframe()`
  - `PDETMunicipalPlanAnalyzer._is_likely_header()`
  - `PDETMunicipalPlanAnalyzer._deduplicate_tables()`
  - `PDETMunicipalPlanAnalyzer._reconstruct_fragmented_tables()`
  - `PDETMunicipalPlanAnalyzer._classify_tables()`
  - `PDETMunicipalPlanAnalyzer.analyze_municipal_plan()`
  - `PDETMunicipalPlanAnalyzer._extract_from_budget_table()`
  - `PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables()`
  - `PDETMunicipalPlanAnalyzer.identify_responsible_entities()`
  - `PDETMunicipalPlanAnalyzer._consolidate_entities()`
  - `PDETMunicipalPlanAnalyzer._score_entity_specificity()`

- `contradiction_deteccion.py`: 3 methods
  - `TemporalLogicVerifier._build_timeline()`
  - `TemporalLogicVerifier._check_deadline_constraints()`
  - `PolicyContradictionDetector._detect_temporal_conflicts()`

- `semantic_chunking_policy.py`: 1 method
  - `SemanticProcessor._detect_table()`

### D2-Q2: Causalidad de Actividades (25 methods)
**Question**: ¿La descripción de actividades detalla el mecanismo causal (mediante, porque, genera)?

**Participating Methods**:
- `policy_processor.py`: 5 methods
- `contradiction_deteccion.py`: 8 methods
- `dereck_beach.py`: 12 methods

### D2-Q3: Clasificación Temática (18 methods)
**Question**: ¿El PDM vincula actividades con causas raíz del diagnóstico?

**Participating Methods**:
- `policy_processor.py`: 4 methods
- `Analyzer_one.py`: 6 methods
- `contradiction_deteccion.py`: 5 methods
- `embedding_policy.py`: 3 methods

### D2-Q4: Riesgos y Mitigación (20 methods)
**Question**: ¿El plan identifica riesgos, obstáculos o barreras en implementación?

**Participating Methods**:
- `policy_processor.py`: 4 methods
- `contradiction_deteccion.py`: 8 methods
- `Analyzer_one.py`: 4 methods
- `financiero_viabilidad_tablas.py`: 4 methods

### D2-Q5: Coherencia Estratégica (24 methods)
**Question**: ¿El conjunto de actividades demuestra una estrategia coherente?

**Participating Methods**:
- `policy_processor.py`: 5 methods
- `contradiction_deteccion.py`: 12 methods
- `Analyzer_one.py`: 4 methods
- `embedding_policy.py`: 3 methods

**Total**: 107 method calls across 5 questions (with some methods reused)

## Usage

### Programmatic Usage

```python
from orchestrator.d2_activities_orchestrator import (
    D2ActivitiesOrchestrator,
    D2Question,
    validate_d2_orchestration,
)

# Option 1: Use convenience function
success = validate_d2_orchestration(
    strict_mode=True,
    output_report="d2_validation_report.json"
)

# Option 2: Use orchestrator directly
orchestrator = D2ActivitiesOrchestrator(
    strict_mode=True,
    trace_execution=True
)

# Validate a single question
result = orchestrator.validate_method_existence(D2Question.Q1_FORMATO_TABULAR)
print(f"Success: {result.success}")
print(f"Methods: {result.executed_methods - result.failed_methods}/{result.total_methods}")

# Validate all D2 questions
results = orchestrator.validate_all_d2_questions()

# Generate report
report = orchestrator.generate_validation_report(results)
print(f"Overall success: {report['summary']['overall_success']}")
```

### Command Line Usage

```bash
# Validate in non-strict mode (default)
python scripts/validate_d2_concurrence.py --summary

# Validate in strict mode (fail on any missing method)
python scripts/validate_d2_concurrence.py --strict --summary

# Generate JSON report
python scripts/validate_d2_concurrence.py \
    --output d2_validation_report.json \
    --summary

# Set custom fail threshold (e.g., 95% success rate)
python scripts/validate_d2_concurrence.py \
    --fail-threshold 0.95 \
    --summary
```

### CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
- name: Validate D2 Method Concurrence
  run: |
    python scripts/validate_d2_concurrence.py \
      --strict \
      --output d2_validation_report.json \
      --summary
```

## Contract Enforcement

### Strict Mode (SIN_CARRETA Doctrine)

When `strict_mode=True`:
- ✅ All methods must be resolvable
- ✅ Execution aborts on first missing method
- ✅ No partial results
- ✅ Explicit error diagnostics

```python
orchestrator = D2ActivitiesOrchestrator(strict_mode=True)

try:
    results = orchestrator.validate_all_d2_questions()
except OrchestrationError as e:
    print(f"Contract violation: {e}")
    # Handle failure explicitly
```

### Non-Strict Mode (Development/Testing)

When `strict_mode=False`:
- ⚠️ Missing methods are logged but don't abort
- ⚠️ Partial results are returned
- ⚠️ Useful for development and incremental validation

```python
orchestrator = D2ActivitiesOrchestrator(strict_mode=False)
results = orchestrator.validate_all_d2_questions()

# Check individual results
for question_id, result in results.items():
    if not result.success:
        print(f"Warning: {question_id} has missing methods")
```

## Traceability

### Execution Traces

When `trace_execution=True`, the orchestrator maintains detailed traces:

```python
orchestrator = D2ActivitiesOrchestrator(trace_execution=True)
result = orchestrator.validate_method_existence(D2Question.Q1_FORMATO_TABULAR)

# Access execution trace
for method_result in result.method_results:
    print(f"Method: {method_result.method_spec.fully_qualified_name}")
    print(f"Success: {method_result.success}")
    print(f"Time: {method_result.execution_time_ms:.2f}ms")
    if method_result.trace:
        print(f"Trace: {method_result.trace}")
```

### Validation Reports

Reports include comprehensive diagnostics:

```json
{
  "metadata": {
    "timestamp": 1234567890.0,
    "orchestrator_version": "1.0.0",
    "doctrine": "SIN_CARRETA",
    "strict_mode": true
  },
  "summary": {
    "total_questions": 5,
    "questions_passed": 4,
    "questions_failed": 1,
    "total_methods": 107,
    "methods_resolved": 95,
    "methods_failed": 12,
    "overall_success": false
  },
  "questions": {
    "D2-Q1": {
      "total_methods": 20,
      "executed_methods": 20,
      "failed_methods": 3,
      "success": false,
      "errors": ["..."]
    }
  },
  "failed_methods": [...]
}
```

## Testing

Run the test suite:

```bash
# Run D2 orchestrator tests
python tests/test_d2_orchestrator.py

# Run all tests (if using pytest)
pytest tests/test_d2_orchestrator.py -v
```

## Extension Points

### Adding New D2 Questions

To add a new D2 question:

1. Add to `D2Question` enum:
```python
class D2Question(Enum):
    Q6_NEW_QUESTION = "D2-Q6"
```

2. Add method list to `D2MethodRegistry`:
```python
Q6_METHODS = [
    MethodSpec("module_name", "ClassName", "method_name",
               "ClassName.method_name"),
    # ... more methods
]
```

3. Update `get_methods_for_question()` mapping

### Custom Orchestration

Extend the orchestrator for custom behavior:

```python
class CustomD2Orchestrator(D2ActivitiesOrchestrator):
    def _execute_method(self, method_spec, *args, **kwargs):
        # Custom execution logic
        result = super()._execute_method(method_spec, *args, **kwargs)
        # Post-processing
        return result
```

## Best Practices

1. **Always use strict mode in production**: Ensures contract compliance
2. **Enable tracing for debugging**: Provides detailed execution diagnostics
3. **Save validation reports**: Maintains audit trail
4. **Review failed methods**: Address missing implementations promptly
5. **Run validation in CI**: Catch regressions early

## Troubleshooting

### Missing Dependencies

If methods fail to import due to missing dependencies:

```bash
# Install all dependencies
pip install -r requirements_atroz.txt
```

### Method Resolution Failures

Check that:
1. Module file exists
2. Class is defined in the module
3. Method is defined in the class
4. Method signature is correct

### Import Errors

The orchestrator uses lazy imports. Check module-level imports:
- Verify `__init__.py` files
- Check for circular dependencies
- Ensure proper Python path

## Related Documentation

- [SIN_CARRETA Architectural Doctrine](../STRATEGIC_METHOD_ORCHESTRATION.md)
- [CHESS Strategy](../CHESS_TACTICAL_SUMMARY.md)
- [Canonical Registry](./canonical_registry.py)
- [Method Inventory](../COMPLETE_METHOD_CLASS_MAP.json)

## Changelog

### Version 1.0.0 (2025-10-28)
- Initial implementation
- D2-Q1 through D2-Q5 method mappings (107 total method calls)
- Strict mode enforcement
- Validation reports
- CI integration script
- Comprehensive test suite
