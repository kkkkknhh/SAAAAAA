# Parameterization Strategy & Consolidation Plan

## Executive Summary

This document outlines the comprehensive strategy for consolidating method parameterization from scattered YAML files into a single authoritative JSON configuration file.

## Current State Analysis

### Method Inventory
- **Total Analysis Classes**: 16
- **Total Methods with Parameters**: 40+
- **Total Functions**: 4

### Parameter Classification
- **DATA-DRIVEN (60%)**: Evidence, scores, claims, metrics from document analysis
- **CONFIG-DRIVEN (25%)**: Rules, thresholds, weights, mappings
- **CONTEXT-DRIVEN (15%)**: Plan identifiers, question IDs, dimensions

### Existing YAML Files

| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `VFARFAN_D1Q1_COMPLETE_10_AREAS.yaml` | 2819 | Training data (calibration) | **KEEP** - Calibration input |
| `catalogo_principal.yaml` | 34 | Training catalog index | **KEEP** - Calibration metadata |
| `OperationalizationAuditor_v3.0_COMPLETO.yaml` | 1246 | Auditor parameters | **CONSOLIDATE** → JSON |
| `trazabilidad_cohrencia.yaml` | 498 | Traceability parameters | **CONSOLIDATE** → JSON |
| `execution_mapping.yaml` | 399 | Module registry | **CONVERT** → JSON (architecture) |
| `causalextractor.yaml` | 391 | Causal extraction config | **CONSOLIDATE** → JSON |
| `causal_exctractor.yaml` | 356 | Duplicate causal config | **DEPRECATE** (typo duplicate) |
| `derek_beach_cdaf_config.yaml` | 111 | Derek Beach parameters | **CONSOLIDATE** → JSON |

## Consolidation Strategy

### Phase 1: Categorize Parameters

#### A. METHOD-SPECIFIC PARAMETERS (goes in method_parameters.json)
Parameters that control method behavior:
- Thresholds (scoring, validation, confidence)
- Weights (Bayesian priors, aggregation weights)
- Algorithmic constants (iterations, convergence limits)
- Lexicons (keywords, patterns, entity mappings)

#### B. CALIBRATION DATA (stays in YAML)
Training examples used by calibration system:
- `VFARFAN_D1Q1_COMPLETE_10_AREAS.yaml` - Evidence patterns
- `catalogo_principal.yaml` - Catalog index

#### C. ARCHITECTURAL CONFIG (separate JSON)
System-level configuration:
- Module registry (execution_mapping.yaml → module_registry.json)
- Orchestration pipelines
- Dependency injection mappings

### Phase 2: JSON Schema Design

```json
{
  "version": "1.0.0",
  "generated": "2025-11-13",
  "metadata": {
    "description": "Authoritative method parameterization for SAAAAAA system",
    "schema_version": "1.0",
    "last_validated": "2025-11-13"
  },
  "methods": {
    "<ClassNameMethod>": {
      "fully_qualified_name": "module.Class.method",
      "version": "1.0.0",
      "parameters": {...},
      "epistemological_basis": "Justification for parameter values",
      "academic_references": [...]
    }
  }
}
```

### Phase 3: Parameter Validation Criteria

Each parameter value must satisfy:

**Question**: *"What arguments support the claim: 'This parameterization is robust and respectful of the epistemological nature of the method?'"*

**Validation Checklist**:
- [ ] **Theoretical Grounding**: Academic literature supports value
- [ ] **Empirical Evidence**: Testing validates effectiveness
- [ ] **Domain Alignment**: Appropriate for Colombian municipal policy analysis
- [ ] **Robustness**: Stable across diverse input documents
- [ ] **Interpretability**: Value has clear semantic meaning

**Examples**:

✅ **GOOD**: `bayesian_prior_alpha: 2.0`
- *Justification*: Jeffrey's prior for Beta distribution, well-established in Bayesian literature, uninformative yet proper

✅ **GOOD**: `micro_score_threshold_insuficiente: 0.55`
- *Justification*: Aligns with Colombian educational grading (1-5 scale), validated against expert annotations

❌ **BAD**: `magic_multiplier: 1.73`
- *Problem*: No theoretical justification, arbitrary constant

## Implementation Plan

### Step 1: Build Authoritative JSON
Create `config/method_parameters.json` with all consolidated parameters

### Step 2: Update Method Invocations
Ensure methods receive parameters from:
- **JSON config** (for method-specific parameters)
- **Calibration system** (for context-specific tuning)

Critical differentiation:
- Static configuration → JSON
- Dynamic calibration scores → Calibration system

### Step 3: Deprecate YAMLs
Move to `.deprecated_yaml_parameters/`:
- `OperationalizationAuditor_v3.0_COMPLETO.yaml`
- `trazabilidad_cohrencia.yaml`
- `causalextractor.yaml`
- `causal_exctractor.yaml` (typo duplicate)
- `derek_beach_cdaf_config.yaml`

### Step 4: Block YAML Invocations
Search for YAML loading code:
```python
# OLD (to be replaced)
with open("some_config.yaml") as f:
    params = yaml.safe_load(f)

# NEW (correct)
from saaaaaa.config import method_parameters
params = method_parameters.get_method_config("ClassName.method_name")
```

### Step 5: Testing Strategy
- Unit tests: Validate each method receives correct parameters
- Integration tests: End-to-end pipeline with JSON config
- Regression tests: Compare outputs before/after migration
- Schema validation: JSON schema conformance

## Decision Matrix: Parameter Source

| Parameter Type | Source | Example |
|----------------|--------|---------|
| Method threshold | JSON | `micro_score_threshold: 0.55` |
| Bayesian prior | JSON | `prior_alpha: 2.0` |
| Lexicon/keywords | JSON | `causal_logic_keywords: [...]` |
| Calibration score | Calibration System | `method_fitness: 0.85` |
| Question context | Runtime (executor) | `question_id: "Q001"` |
| Document data | Runtime (document) | `pdt_structure: {...}` |
| Training examples | YAML (calibration) | VFARFAN patterns |

## Success Criteria

- [ ] Single `method_parameters.json` file exists
- [ ] All method-specific parameters documented
- [ ] Epistemological justification for each value
- [ ] No duplicate YAML parameters
- [ ] All YAML loading code updated or removed
- [ ] 100% test coverage for parameterization
- [ ] CI passes with JSON configuration

## Maintenance Protocol

**Adding a new method parameter**:
1. Add to `method_parameters.json`
2. Document epistemological basis
3. Add validation test
4. Update schema documentation

**Modifying existing parameter**:
1. Document change rationale
2. Run regression tests
3. Update version number
4. Log change in CHANGELOG

---

*Generated: 2025-11-13*
*Author: Claude Code (Audit & Consolidation Agent)*
