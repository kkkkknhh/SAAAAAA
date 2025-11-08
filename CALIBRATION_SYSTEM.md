# Calibration System Documentation

## Overview

The calibration system provides rigorous, explicit calibration for all policy analysis methods. It enforces that **no method can execute without an explicit calibration profile**, eliminating silent defaults and ensuring verifiable results.

## Key Principles

1. **Zero Tolerance for Silent Fallbacks**: Missing calibrations raise `MissingCalibrationError`
2. **Multi-Dimensional Indexing**: Calibrations indexed by method_fqn, question_id, document_type, policy_area, unit_of_analysis
3. **ExecutorConfig is Single Source of Truth**: All runtime parameters controlled by ExecutorConfig
4. **Deterministic and Versioned**: Calibrations have version numbers and cryptographic hashes
5. **Context-Aware Refinement**: Base calibrations adjusted based on execution context

## Architecture

### Core Components

```
calibration_registry.py      - 166 explicit method calibrations + error handling
calibration_context.py       - Context-aware modifiers (dimension, policy area, document type)
executor_config.py           - Runtime parameter configuration
factory.py                   - DEPRECATED: YAML loading blocked
```

### Data Model

```python
@dataclass(frozen=True)
class MethodCalibration:
    score_min: float
    score_max: float
    min_evidence_snippets: int
    max_evidence_snippets: int
    contradiction_tolerance: float
    uncertainty_penalty: float
    aggregation_weight: float
    sensitivity: float
    requires_numeric_support: bool
    requires_temporal_support: bool
    requires_source_provenance: bool
    safe_default_allowed: bool = False
    document_type: Optional[str] = None
```

### Context Dimensions

```python
@dataclass(frozen=True)
class CalibrationContext:
    question_id: str              # "D1Q1", "D9Q3", etc.
    dimension: int                # 1-10
    question_num: int             # Question within dimension
    policy_area: PolicyArea       # fiscal, social, health, etc.
    unit_of_analysis: UnitOfAnalysis  # baseline_gap, indicator, financial, etc.
    document_type: DocumentType   # plan_desarrollo_municipal, politica_publica, etc.
    method_position: int          # Position in method sequence
    total_methods: int            # Total methods in sequence
```

## Usage

### Basic Resolution

```python
from saaaaaa.core.orchestrator.calibration_registry import resolve_calibration

# Strict mode (default) - raises MissingCalibrationError if not found
calib = resolve_calibration("BayesianEvidenceScorer", "compute_evidence_score", strict=True)

# Non-strict mode - returns None if not found
calib = resolve_calibration("BayesianEvidenceScorer", "compute_evidence_score", strict=False)
```

### Context-Aware Resolution

```python
from saaaaaa.core.orchestrator.calibration_registry import resolve_calibration_with_context
from saaaaaa.core.orchestrator.calibration_context import DocumentType, PolicyArea, UnitOfAnalysis

# Resolve with full context
calib = resolve_calibration_with_context(
    "BayesianNumericalAnalyzer",
    "evaluate_policy_metric",
    question_id="D9Q1",
    policy_area="fiscal",
    unit_of_analysis="financial",
    document_type="plan_desarrollo_municipal",
    method_position=0,
    total_methods=3
)
```

### Calibration Modifiers

Calibrations are adjusted based on context:

**Document Type Modifiers:**
- `plan_desarrollo_municipal`: +40% evidence, -40% contradiction tolerance (most strict)
- `plan_estrategico`: +35% evidence, -40% contradiction tolerance
- `politica_publica`: +25% evidence, -30% contradiction tolerance
- `plan_sectorial`: +30% evidence, -35% contradiction tolerance

**Dimension Modifiers:**
- D1 (baseline gaps): +30% evidence, +10% sensitivity
- D9 (financial): +35% evidence, -50% contradiction tolerance, +30% sensitivity
- D6 (logical framework): -50% contradiction tolerance, +20% sensitivity

**Policy Area Modifiers:**
- fiscal: +30% evidence, -40% contradiction tolerance
- infrastructure: +40% evidence, -50% contradiction tolerance
- social: +10% evidence, -10% contradiction tolerance

**Unit of Analysis Modifiers:**
- financial: +35% evidence, -50% contradiction tolerance, +25% sensitivity
- impact: +40% evidence, +20% sensitivity
- baseline_gap: +30% evidence, +20% sensitivity

## Migration from YAML

**Old (DEPRECATED):**
```python
from saaaaaa.analysis.factory import load_all_calibrations
calibrations = load_all_calibrations()  # Returns {} with deprecation warning
```

**New (REQUIRED):**
```python
from saaaaaa.core.orchestrator.calibration_registry import CALIBRATIONS, resolve_calibration

# Direct access to registry
all_calibrations = CALIBRATIONS  # Dict[(class_name, method_name), MethodCalibration]

# Resolve with strict enforcement
calib = resolve_calibration("ClassName", "method_name", strict=True)
```

## Error Handling

### MissingCalibrationError

Raised when a method lacks explicit calibration:

```python
try:
    calib = resolve_calibration("UncalibratedClass", "method", strict=True)
except MissingCalibrationError as e:
    print(f"Method: {e.method_fqn}")  # "UncalibratedClass.method"
    print(f"Context: {e.context}")    # {"resolution": "base"}
```

## Versioning and Hashing

Calibrations are versioned and hashed for reproducibility:

```python
from saaaaaa.core.orchestrator.calibration_registry import (
    CALIBRATION_VERSION,
    get_calibration_hash
)

print(f"Version: {CALIBRATION_VERSION}")  # "1.0.0"
print(f"Hash: {get_calibration_hash()}")  # SHA256 hex digest
```

## ExecutorConfig Integration

ExecutorConfig drives all runtime decisions:

```python
from saaaaaa.core.orchestrator.executor_config import ExecutorConfig

config = ExecutorConfig(
    max_tokens=2048,
    temperature=0.0,      # Deterministic
    timeout_s=30.0,
    retry=2,
    seed=42,              # For reproducibility
    thresholds={
        "min_confidence": 0.9,
        "min_evidence": 0.8,
    }
)

# Config hash for tracking
config_hash = config.compute_hash()
```

## Testing

### Completeness Tests

```bash
pytest tests/test_calibration_completeness.py -v
```

Tests verify:
- All calibrations have valid keys and values
- Missing calibrations raise MissingCalibrationError in strict mode
- Calibration hash is deterministic
- No default-like calibrations without safe_default_allowed flag

### Stability Tests

```bash
pytest tests/test_calibration_stability.py -v
```

Tests verify:
- Same config + seed → deterministic results
- Calibration hash changes with data
- Context resolution is deterministic
- Document type modifiers work correctly

## Demo

Run the calibration demo:

```bash
python scripts/demo_calibration_strict.py
```

This demonstrates:
1. Strict enforcement (MissingCalibrationError for uncalibrated methods)
2. Valid calibration resolution
3. Context-aware calibration adjustments
4. Document type impact on requirements

## Calibration Criteria for Policy Analysis

Calibrations are designed specifically for policy document analysis (planes de desarrollo, políticas públicas):

### Evidence Requirements

Policy documents are extensive, multi-sector documents requiring:
- **Higher evidence thresholds**: Municipal plans need 40% more evidence than generic analysis
- **Strict contradiction tolerance**: Financial and infrastructure dimensions allow minimal contradictions
- **Source provenance**: All calibrations require traceable sources to plan sections

### Document Type Rationale

Different document types have different analytical needs:

1. **Plan Desarrollo Municipal** (most strict):
   - Comprehensive, multi-year, multi-sector planning
   - Requires extensive cross-sector evidence
   - Low tolerance for contradictions due to integration requirements

2. **Política Pública** (focused):
   - Specific interventions in defined sectors
   - Moderate evidence requirements
   - Higher tolerance for sector-specific tradeoffs

3. **Plan Sectorial**:
   - Domain-specific planning
   - High evidence within sector
   - Moderate cross-sector tolerance

4. **Plan Estratégico** (strategic):
   - High-level strategic planning
   - Very high sensitivity to gaps
   - Strict requirements due to cascading impacts

## FIXME Markers

Areas needing explicit calibration definition are marked:

```python
# FIXME(CALIBRATION): Missing explicit calibration for X.method in context Y
```

Do not invent calibration values - these markers ensure transparency about gaps.

## References

- Calibration Registry: `src/saaaaaa/core/orchestrator/calibration_registry.py`
- Context System: `src/saaaaaa/core/orchestrator/calibration_context.py`
- Executor Config: `src/saaaaaa/core/orchestrator/executor_config.py`
- Completeness Tests: `tests/test_calibration_completeness.py`
- Stability Tests: `tests/test_calibration_stability.py`
- Demo: `scripts/demo_calibration_strict.py`
