# Calibration System - README

**Version:** 1.0.0  
**Last Updated:** 2025-11-09  
**Status:** PRODUCTION  

---

## Quick Start

### For Users

The calibration system is fully integrated into the policy analysis pipeline. You don't need to configure anything - calibrations are automatically applied based on the method being executed and the analysis context.

### For Developers

If you need to add a new calibrated method or modify existing calibrations:

1. **Read the procedure**: See [CALIBRATION_PROCEDURE.md](./CALIBRATION_PROCEDURE.md) for complete mathematical documentation and step-by-step instructions
2. **Update the registry**: Add your calibration to `src/saaaaaa/core/orchestrator/calibration_registry.py`
3. **Update the catalog**: Add method entry to `config/canonical_method_catalog.json`
4. **Validate**: Run validation scripts to ensure compliance

```bash
python3 scripts/validate_canonical_catalog.py
python3 scripts/check_directive_compliance.py
pytest tests/test_canonical_method_catalog.py -v
```

---

## System Overview

The calibration system provides rigorous, explicit parameterization for all policy analysis methods. It ensures:

- **Zero Silent Defaults**: Every method must have an explicit calibration or will raise `MissingCalibrationError`
- **Deterministic Execution**: All calibrations are versioned and deterministic (no random initialization)
- **Context-Aware**: Base calibrations are refined based on execution context (dimension, policy area, document type)
- **Single Source of Truth**: `calibration_registry.py` is the ONLY source for calibration parameters

---

## Architecture

```
calibration_registry.py          ← SINGLE SOURCE OF TRUTH (180 calibrations)
calibration_context.py           ← Context-aware refinement logic
canonical_method_catalog.json    ← Layer taxonomy & method inventory
embedded_calibration_appendix.json  ← Embedded calibration tracking
CALIBRATION_PROCEDURE.md          ← Mathematical & procedural documentation
```

### Core Components

1. **calibration_registry.py**: Contains all 180 explicit method calibrations
2. **calibration_context.py**: Provides context-aware modifiers (dimension, policy area, document type)
3. **canonical_method_catalog.json**: Maps methods to layers (Q, D, P, C, M) with full metadata
4. **embedded_calibration_appendix.json**: Tracks embedded calibrations and migration status

---

## Calibration Parameters

Every method calibration consists of 11 parameters:

| Parameter | Type | Range | Purpose |
|-----------|------|-------|---------|
| `score_min` | float | [0.0, 1.0] | Minimum score boundary |
| `score_max` | float | [0.0, 1.0] | Maximum score boundary |
| `min_evidence_snippets` | int | [1, 100] | Minimum evidence requirement |
| `max_evidence_snippets` | int | [1, 100] | Maximum evidence consideration |
| `contradiction_tolerance` | float | [0.0, 1.0] | Maximum contradiction ratio allowed |
| `uncertainty_penalty` | float | [0.0, 1.0] | Penalty for uncertain evidence |
| `aggregation_weight` | float | [0.0, 2.0] | Relative importance weight |
| `sensitivity` | float | [0.0, 1.0] | Evidence quality sensitivity |
| `requires_numeric_support` | bool | - | Numeric evidence requirement |
| `requires_temporal_support` | bool | - | Temporal evidence requirement |
| `requires_source_provenance` | bool | - | Source tracking requirement |

For detailed mathematical foundations and parameter selection rules, see [CALIBRATION_PROCEDURE.md](./CALIBRATION_PROCEDURE.md).

---

## Layer Taxonomy

Methods are organized into 5 analysis layers:

| Layer | Code | Description | Example Methods |
|-------|------|-------------|-----------------|
| **Question** | Q | Micro-question resolution | `generate_smart_chunks`, semantic chunking |
| **Dimension** | D | D1-D10 analytical dimension | Bayesian inference, causal extraction |
| **Policy Area** | P | PA01-PA10 domain-specific | Financial analysis, municipal planning |
| **Congruence** | C | Cross-dimensional ensembles | Contradiction detection, coherence validation |
| **Meta** | M | Aggregation, scoring, reporting | Graph statistics, performance metrics |

---

## Usage Examples

### Basic Resolution

```python
from saaaaaa.core.orchestrator.calibration_registry import resolve_calibration

# Get base calibration
calib = resolve_calibration("BayesianEvidenceScorer", "compute_evidence_score")

print(f"Min evidence: {calib.min_evidence_snippets}")
print(f"Max evidence: {calib.max_evidence_snippets}")
print(f"Sensitivity: {calib.sensitivity}")
```

### Context-Aware Resolution

```python
from saaaaaa.core.orchestrator.calibration_registry import resolve_calibration_with_context

# Resolve with full context
calib = resolve_calibration_with_context(
    "BayesianNumericalAnalyzer",
    "evaluate_policy_metric",
    question_id="D9Q1",
    policy_area="fiscal",
    unit_of_analysis="financial",
    document_type="plan_desarrollo_municipal",
)

print(f"Context-refined min evidence: {calib.min_evidence_snippets}")
```

### Contextual Refinement

Calibrations are automatically refined based on:

**Document Type:**
- Municipal development plans: +40% evidence, -40% contradiction tolerance (strictest)
- Strategic plans: +35% evidence, -40% contradiction tolerance
- Public policies: +30% evidence, -35% contradiction tolerance

**Policy Area:**
- Fiscal/Financial: +0.10 sensitivity, +0.10 uncertainty penalty
- Infrastructure: +0.08 sensitivity, +0.08 uncertainty penalty
- Legal/Regulatory: +0.12 sensitivity, +0.12 uncertainty penalty

**Dimension:**
- Dimensions 1, 4, 9 (baseline, financial, implementation): -50% contradiction tolerance

---

## Validation & Testing

### Validation Scripts

```bash
# Check directive compliance
python3 scripts/check_directive_compliance.py

# Validate catalog structure
python3 scripts/validate_canonical_catalog.py
```

### Automated Tests

```bash
# Run all catalog tests
pytest tests/test_canonical_method_catalog.py -v

# Run specific test class
pytest tests/test_canonical_method_catalog.py::TestCatalogStructure -v
```

---

## Migration Status

### ✓ Complete

- All 180 methods calibrated in registry
- All YAML calibrations deprecated
- Layer taxonomy fully mapped
- Embedded calibrations tracked
- Validation infrastructure in place
- Mathematical documentation complete

### YAML Deprecation

The following YAML files have been deprecated (as of 2025-11-09):

- `calibracion_bayesiana.yaml`
- `financia_callibrator.yaml`
- `catalogo_principal.yaml`
- `causal_exctractor.yaml`
- `trazabilidad_cohrencia.yaml`

**All YAML files now raise errors if loaded.** All calibrations are in `calibration_registry.py`.

---

## Key Invariants

1. **INVARIANT-1**: Every calibrated method has exactly ONE base calibration entry
2. **INVARIANT-2**: All calibration parameters are deterministic (no random initialization)
3. **INVARIANT-3**: Contextual refinements are strictly additive/multiplicative transformations
4. **INVARIANT-4**: No YAML files contain active calibration data
5. **INVARIANT-5**: Calibration hash changes trigger version increment

---

## Troubleshooting

### MissingCalibrationError

If you encounter `MissingCalibrationError`, it means a method lacks an explicit calibration:

```python
MissingCalibrationError: Missing calibration for method 'ClassName.method_name' 
in context [question_id=D1Q1]. Execution blocked - explicit calibration required.
```

**Solution:** Add a calibration entry following the procedure in [CALIBRATION_PROCEDURE.md](./CALIBRATION_PROCEDURE.md).

### Calibration Hash Mismatch

If calibration hashes don't match, it indicates the registry has been modified:

1. Update `CALIBRATION_VERSION` in `calibration_registry.py`
2. Regenerate the catalog with updated version
3. Re-run validation scripts

---

## References

### Primary Documentation

- **[CALIBRATION_PROCEDURE.md](./CALIBRATION_PROCEDURE.md)** - Mathematical foundations and procedures
- **[CANONICAL_METHOD_NOTATION_SPEC.md](./docs/CANONICAL_METHOD_NOTATION_SPEC.md)** - Method notation system
- **[CALIBRATION_SYSTEM.md](./CALIBRATION_SYSTEM.md)** - High-level system documentation

### Configuration Files

- **config/canonical_method_catalog.json** - Layer taxonomy and method inventory
- **config/embedded_calibration_appendix.json** - Embedded calibration tracking
- **src/saaaaaa/core/orchestrator/calibration_registry.py** - Source of truth

### Validation Tools

- **scripts/check_directive_compliance.py** - Verify migration compliance
- **scripts/validate_canonical_catalog.py** - Validate catalog integrity
- **tests/test_canonical_method_catalog.py** - Automated testing

---

## Version History

### 1.0.0 (2025-11-09)

- Initial release
- 180 methods calibrated
- Complete YAML deprecation
- Full layer taxonomy implementation
- Validation infrastructure complete
- Mathematical documentation complete

---

**For Questions:**  
See [CALIBRATION_PROCEDURE.md](./CALIBRATION_PROCEDURE.md) for detailed procedures and examples.

**For Modifications:**  
Follow the step-by-step procedure in [CALIBRATION_PROCEDURE.md](./CALIBRATION_PROCEDURE.md) Section 6.

**Document Status:** PRODUCTION  
**Last Review:** 2025-11-09  
**Next Review:** 2025-12-09 (monthly)
