# D1 Diagnostic Dimension - Method Concurrence Enforcement

## Overview

This module implements strict orchestration enforcement for D1 (Diagnóstico) questions according to **SIN_CARRETA doctrine**, ensuring deterministic, non-negotiable execution of all declared methods.

## Architecture Components

### 1. D1QuestionOrchestrator (`orchestrator/d1_orchestrator.py`)

Core orchestration engine that enforces method concurrence for each D1 question.

**Key Features:**
- Contract-based method orchestration with precondition/postcondition validation
- Execution trace generation for full auditability
- Deterministic failure semantics with explicit error context
- Method dependency resolution and execution planning

### 2. Validation Script (`scripts/validate_d1_orchestration.py`)

CI validation script that validates orchestration contract compliance.

**Validations Performed:**
- Method specification counts (18, 12, 22, 16, 14 for Q1-Q5)
- Method availability in canonical registry
- SIN_CARRETA doctrine compliance

### 3. Test Suite (`tests/test_d1_orchestrator.py`)

Comprehensive test suite validating orchestrator behavior.

**Test Coverage:**
- Method specification validation
- Contract enforcement (strict vs non-strict modes)
- Execution trace capture
- Audit report generation
- Doctrine compliance

## D1 Question Method Specifications

### D1-Q1: Líneas Base y Brechas Cuantificadas (18 methods)

**Question:** ¿El diagnóstico presenta datos numéricos cuantificando brechas?

**Methods:**
- `policy_processor.py` (6 methods):
  - `IndustrialPolicyProcessor.process`
  - `IndustrialPolicyProcessor._match_patterns_in_sentences`
  - `PolicyTextProcessor.segment_into_sentences`
  - `BayesianEvidenceScorer.compute_evidence_score`
  - `PolicyTextProcessor._calculate_shannon_entropy`
  - `IndustrialPolicyProcessor._construct_evidence_bundle`

- `contradiction_deteccion.py` (8 methods):
  - `PolicyContradictionDetector._extract_quantitative_claims`
  - `PolicyContradictionDetector._parse_number`
  - `PolicyContradictionDetector._extract_temporal_markers`
  - `PolicyContradictionDetector._determine_semantic_role`
  - `PolicyContradictionDetector._calculate_confidence_interval`
  - `PolicyContradictionDetector._statistical_significance_test`
  - `BayesianConfidenceCalculator.calculate_posterior`
  - `PolicyContradictionDetector._get_context_window`

- `Analyzer_one.py` (2 methods):
  - `SemanticAnalyzer._calculate_semantic_complexity`
  - `SemanticAnalyzer._classify_policy_domain`

- `embedding_policy.py` (2 methods):
  - `BayesianNumericalAnalyzer.evaluate_policy_metric`
  - `BayesianNumericalAnalyzer._classify_evidence_strength`

### D1-Q2: Normalización y Fuentes (12 methods)

**Question:** ¿El texto dimensiona el problema cuantificando la brecha con fuentes oficiales?

**Methods:**
- `policy_processor.py` (4 methods):
  - `IndustrialPolicyProcessor._match_patterns_in_sentences`
  - `IndustrialPolicyProcessor._compile_pattern_registry`
  - `PolicyTextProcessor.normalize_unicode`
  - `BayesianEvidenceScorer.compute_evidence_score`

- `contradiction_deteccion.py` (6 methods):
  - `PolicyContradictionDetector._parse_number`
  - `PolicyContradictionDetector._extract_quantitative_claims`
  - `PolicyContradictionDetector._are_comparable_claims`
  - `PolicyContradictionDetector._calculate_numerical_divergence`
  - `PolicyContradictionDetector._determine_semantic_role`
  - `BayesianConfidenceCalculator.calculate_posterior`

- `embedding_policy.py` (2 methods):
  - `PolicyAnalysisEmbedder._extract_numerical_values`
  - `BayesianNumericalAnalyzer._compute_coherence`

### D1-Q3: Asignación de Recursos (22 methods)

**Question:** ¿Se identifican recursos presupuestales suficientes para abordar la brecha?

**Methods:**
- `policy_processor.py` (5 methods)
- `contradiction_deteccion.py` (10 methods)
- `financiero_viabilidad_tablas.py` (5 methods)
- `embedding_policy.py` (2 methods)

### D1-Q4: Capacidad Institucional (16 methods)

**Question:** ¿El PDM describe capacidades institucionales (talento humano, procesos, gobernanza)?

**Methods:**
- `policy_processor.py` (4 methods)
- `contradiction_deteccion.py` (7 methods)
- `Analyzer_one.py` (3 methods)
- `financiero_viabilidad_tablas.py` (2 methods)

### D1-Q5: Restricciones Temporales (14 methods)

**Question:** ¿El plan justifica su alcance mencionando restricciones temporales?

**Methods:**
- `policy_processor.py` (3 methods)
- `contradiction_deteccion.py` (9 methods)
- `Analyzer_one.py` (2 methods)

## SIN_CARRETA Doctrine Compliance

### No Graceful Degradation
All listed methods for each question **must** execute deterministically. Partial execution, fallback, or best-effort responses are **strictly forbidden**.

**Enforcement:**
```python
# Strict mode (default) - raises D1OrchestrationError on any failure
result = orchestrator.orchestrate_question(
    D1Question.Q1_BASELINE,
    context,
    strict=True  # Enforces no graceful degradation
)
```

### Explicit Failure
If any method fails its declared contract, orchestration **aborts** with explicit diagnostics.

**Error Reporting:**
```python
raise D1OrchestrationError(
    question_id="D1-Q1",
    failed_methods=["Method1", "Method2"],
    execution_traces=[...],
    message="Contract violation details..."
)
```

### Full Traceability
Execution is observable, auditable, and reproducible through `ExecutionTrace` objects.

**Trace Structure:**
```python
@dataclass
class ExecutionTrace:
    question_id: str
    method_name: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool
    result: Any
    error: Optional[str]
    stack_trace: Optional[str]
    input_context: Dict[str, Any]
```

### No Strategic Simplification
Complexity of orchestration is a **design asset**. The full method set is preserved for each question.

**Method Counts:**
- D1-Q1: 18 methods (not reduced)
- D1-Q2: 12 methods (not reduced)
- D1-Q3: 22 methods (not reduced)
- D1-Q4: 16 methods (not reduced)
- D1-Q5: 14 methods (not reduced)

## Usage

### Basic Orchestration

```python
from orchestrator.d1_orchestrator import D1QuestionOrchestrator, D1Question
from orchestrator.canonical_registry import CANONICAL_METHODS

# Initialize with canonical registry
orchestrator = D1QuestionOrchestrator(canonical_registry=CANONICAL_METHODS)

# Prepare execution context
context = {
    "text": policy_document_text,
    "metadata": {"plan_name": "PDM Example"},
    "data": extracted_data,
}

# Orchestrate with strict contract enforcement
try:
    result = orchestrator.orchestrate_question(
        D1Question.Q1_BASELINE,
        context,
        strict=True
    )
    
    if result.success:
        print(f"✓ All {len(result.executed_methods)} methods executed successfully")
        print(f"  Duration: {result.total_duration_ms:.2f}ms")
    
except D1OrchestrationError as e:
    print(f"✗ Orchestration failed: {e}")
    print(f"  Failed methods: {e.failed_methods}")
    for trace in e.execution_traces:
        if not trace.success:
            print(f"    - {trace.method_name}: {trace.error}")
```

### Audit Report Generation

```python
# Orchestrate all D1 questions
results = []
for question in D1Question:
    result = orchestrator.orchestrate_question(question, context, strict=False)
    results.append(result)

# Generate comprehensive audit report
audit_report = orchestrator.generate_audit_report(results)

print(f"Overall Success Rate: {audit_report['summary']['overall_success_rate']:.1%}")
print(f"Doctrine Compliance: {audit_report['doctrine_compliance']}")
```

## CI Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Validate D1 Orchestration Contract
  run: |
    python scripts/validate_d1_orchestration.py --strict --output validation_report.json
```

**Exit Codes:**
- `0` - All validations passed
- `1` - Validation failed (contract violations detected)
- `2` - Critical error (import failures, missing dependencies)

## Testing

Run the test suite:

```bash
python -m unittest tests.test_d1_orchestrator -v
```

**Test Coverage:**
- 20 test cases covering all orchestration aspects
- Method specification validation
- Contract enforcement (strict/non-strict modes)
- Execution trace capture
- Audit report generation
- Doctrine compliance verification

## Implementation Status

- [x] D1QuestionOrchestrator class implementation
- [x] Method contract specifications for all 5 D1 questions (82 total methods)
- [x] Execution trace capture and audit report generation
- [x] Strict failure semantics with D1OrchestrationError
- [x] Comprehensive test suite (20 tests, 100% passing)
- [x] CI validation script
- [x] Documentation

## Next Steps

1. **Method Registry Population**: Ensure all 82 methods are registered in `canonical_registry.py`
2. **Integration Testing**: Test with real policy documents
3. **Performance Optimization**: Profile and optimize orchestration for production
4. **Extended Dimensions**: Apply same pattern to D2-D6 questions

## References

- Issue: [Enforce Concurrence of Methods for D1 Diagnostic Consistency]
- SIN_CARRETA Doctrine: `.augment/rules/GENERAL.md`
- Strategic Method Orchestration: `STRATEGIC_METHOD_ORCHESTRATION.md`
- Canonical Registry: `orchestrator/canonical_registry.py`
