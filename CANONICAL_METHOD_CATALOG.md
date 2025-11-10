# Canonical Method Catalog - Documentation

**Version:** 1.0.0  
**Last Updated:** 2025-11-09  
**Status:** CANONICAL REFERENCE  
**Catalog File:** `config/canonical_method_catalog.json`

---

## Overview

The Canonical Method Catalog is the authoritative inventory of all calibrated methods in the policy analysis system. It provides:

- **Layer Taxonomy Mapping**: Every method mapped to analysis layer (Q, D, P, C, M)
- **Calibration Tracking**: Complete calibration status and references
- **Method Classification**: Complexity, priority, and execution flags
- **Migration Verification**: Ensures all calibrations migrated from YAML to code

---

## Table of Contents

1. [Catalog Structure](#catalog-structure)
2. [Layer Taxonomy](#layer-taxonomy)
3. [Method Notation](#method-notation)
4. [Calibration Status](#calibration-status)
5. [Execution Flags](#execution-flags)
6. [Usage Guide](#usage-guide)
7. [Maintenance](#maintenance)

---

## Catalog Structure

The catalog is a JSON file with four main sections:

```json
{
  "metadata": {
    "version": "1.0.0",
    "generated_at": "2025-11-09T...",
    "total_methods": 180,
    "layer_taxonomy": {...},
    "migration_status": "COMPLETE"
  },
  "methods": {
    "ClassName.method_name": {...}
  },
  "statistics": {
    "total_calibrated": 180,
    "by_layer": {...},
    "by_calibration_status": {...}
  },
  "migration_notes": {...}
}
```

### Metadata

Contains catalog-level information:

- **version**: Semantic version of the catalog
- **generated_at**: ISO 8601 timestamp of generation
- **total_methods**: Total number of methods in catalog
- **layer_taxonomy**: Definition of all 5 analysis layers
- **migration_status**: Overall migration completion status

### Methods

Dictionary of all methods, keyed by fully qualified name (`ClassName.method_name`):

```json
"AdvancedDAGValidator._calculate_bayesian_posterior": {
  "canonical_id": "DER:AdvancedDAGValidator._calculate_bayesian_posterior@D[NTBS]{CAL}",
  "class": "AdvancedDAGValidator",
  "method_name": "_calculate_bayesian_posterior",
  "module": "DER",
  "file": "dereck_beach.py",
  "layer": "D",
  "flags": ["N", "T", "B", "S"],
  "calibration_status": "CAL",
  "calibration_ref": "AdvancedDAGValidator._calculate_bayesian_posterior",
  "complexity": "HIGH",
  "priority": "CRITICAL"
}
```

### Statistics

Aggregated statistics about catalog contents:

- **total_calibrated**: Total calibrated methods
- **by_layer**: Method count per analysis layer
- **by_calibration_status**: Distribution of calibration statuses

### Migration Notes

Documentation of migration from YAML to code:

- **yaml_sources_eliminated**: List of deprecated YAML files
- **all_calibrations_in_registry**: Boolean confirmation
- **single_source_of_truth**: Path to calibration registry

---

## Layer Taxonomy

Methods are organized into 5 analysis layers:

### Q - Question Layer

**Purpose:** Micro-question resolution and granular evidence extraction

**Characteristics:**
- Operates on individual questions or micro-tasks
- Focused evidence requirements (typically 1-5 snippets)
- Lower complexity, higher throughput
- Example: Semantic chunking, question segmentation

**Method Count:** 16

### D - Dimension Layer

**Purpose:** D1-D10 analytical dimension processing

**Characteristics:**
- Comprehensive dimension-specific analysis
- Higher evidence requirements (3-25 snippets)
- Bayesian and statistical methods
- Example: Causal extraction, mechanism inference, DAG validation

**Method Count:** 79

### P - Policy Area Layer

**Purpose:** PA01-PA10 domain-specific processing

**Characteristics:**
- Domain knowledge application
- Policy area specialization (fiscal, social, health, etc.)
- Contextual refinement based on domain
- Example: Financial feasibility, municipal planning

**Method Count:** 46

### C - Congruence Layer

**Purpose:** Cross-dimensional coherence and validation

**Characteristics:**
- Validates consistency across dimensions
- Detects contradictions and conflicts
- Ensures logical coherence
- Example: Contradiction detection, temporal verification

**Method Count:** 36

### M - Meta Layer

**Purpose:** Aggregation, scoring, and reporting

**Characteristics:**
- Combines outputs from other layers
- Performance metrics and statistics
- Lower sensitivity to evidence quality
- Example: Graph statistics, performance analysis

**Method Count:** 3

---

## Method Notation

### Canonical ID Format

Every method has a unique canonical ID following this format:

```
<MODULE>:<CLASS>.<METHOD>@<LAYER>[<FLAGS>]{<CALIBRATION_STATUS>}
```

**Example:**
```
DER:AdvancedDAGValidator._calculate_bayesian_posterior@D[NTBS]{CAL}
```

**Breakdown:**
- **DER**: Module code (Derek Beach / dereck_beach.py)
- **AdvancedDAGValidator**: Class name
- **_calculate_bayesian_posterior**: Method name
- **@D**: Layer (Dimension)
- **[NTBS]**: Flags (Numeric, Temporal, Bayesian, Source)
- **{CAL}**: Calibration status (Calibrated)

### Module Codes

| Code | Module | File |
|------|--------|------|
| DER | Derek Beach methods | dereck_beach.py |
| FIN | Financial analysis | financiero_viabilidad_tablas.py |
| ANA | General analyzers | Analyzer_one.py |
| CON | Contradiction detection | contradiction_deteccion.py |
| EMB | Embedding/semantic | embedding_policy.py |
| SEM | Semantic chunking | semantic_chunking_policy.py |
| POL | Policy processing | policy_processor.py |
| TEO | Theory of change | teoria_cambio.py |
| AGG | Aggregation | aggregation.py |
| SCO | Scoring | scoring.py |

---

## Calibration Status

Methods have one of five calibration statuses:

| Status | Code | Description | Count |
|--------|------|-------------|-------|
| **Calibrated** | CAL | Has explicit calibration in registry | 180 |
| **Required** | REQ | Needs calibration, not yet added | 0 |
| **Optional** | OPT | Utility/helper, calibration optional | 0 |
| **Derived** | DER | Uses other calibrated methods | 0 |
| **In-Script** | INS | Hard-coded parameters in script | 0 |

**Current State:** All 180 methods have status `CAL` (calibrated).

---

## Execution Flags

Methods are annotated with execution characteristic flags:

| Flag | Name | Description |
|------|------|-------------|
| **N** | Numeric | Requires numeric/quantitative evidence |
| **T** | Temporal | Requires temporal/chronological evidence |
| **S** | Source | Requires source provenance tracking |
| **B** | Bayesian | Uses Bayesian inference |
| **A** | Async | Async-capable execution |
| **I** | I/O | I/O-intensive operation |
| **C** | Compute | Compute-intensive operation |

**Example Combinations:**
- `[NBS]`: Numeric + Bayesian + Source (financial Bayesian analysis)
- `[TS]`: Temporal + Source (temporal logic verification)
- `[S]`: Source only (semantic analysis)

---

## Usage Guide

### Querying the Catalog

```python
import json
from pathlib import Path

# Load catalog
catalog_path = Path("config/canonical_method_catalog.json")
with open(catalog_path) as f:
    catalog = json.load(f)

# Get method by name
method = catalog["methods"]["AdvancedDAGValidator.calculate_acyclicity_pvalue"]
print(f"Layer: {method['layer']}")
print(f"Flags: {method['flags']}")
print(f"Status: {method['calibration_status']}")
```

### Finding Methods by Layer

```python
# Find all Question layer methods
q_methods = [
    (name, data)
    for name, data in catalog["methods"].items()
    if data["layer"] == "Q"
]

print(f"Found {len(q_methods)} Question layer methods")
```

### Checking Calibration Coverage

```python
# Check calibration status distribution
stats = catalog["statistics"]["by_calibration_status"]
total = catalog["statistics"]["total_calibrated"]

print(f"Total methods: {total}")
print(f"Calibrated (CAL): {stats['CAL']}")
print(f"Required (REQ): {stats['REQ']}")
print(f"Coverage: {stats['CAL']/total*100:.1f}%")
```

---

## Maintenance

### Adding a New Method

1. **Add calibration** to `src/saaaaaa/core/orchestrator/calibration_registry.py`:

```python
("NewClass", "new_method"): MethodCalibration(
    score_min=0.0,
    score_max=1.0,
    min_evidence_snippets=3,
    max_evidence_snippets=20,
    contradiction_tolerance=0.05,
    uncertainty_penalty=0.3,
    aggregation_weight=1.0,
    sensitivity=0.85,
    requires_numeric_support=True,
    requires_temporal_support=False,
    requires_source_provenance=True,
),
```

2. **Add to catalog** in `config/canonical_method_catalog.json`:

```json
"NewClass.new_method": {
  "canonical_id": "MOD:NewClass.new_method@D[NBS]{CAL}",
  "class": "NewClass",
  "method_name": "new_method",
  "module": "MOD",
  "file": "module_file.py",
  "layer": "D",
  "flags": ["N", "B", "S"],
  "calibration_status": "CAL",
  "calibration_ref": "NewClass.new_method",
  "complexity": "HIGH",
  "priority": "CRITICAL"
}
```

3. **Update statistics** in catalog:
   - Increment `total_methods`
   - Update `by_layer` counts
   - Update `by_calibration_status` counts

4. **Validate**:

```bash
python3 scripts/validate_canonical_catalog.py
python3 scripts/check_directive_compliance.py
pytest tests/test_canonical_method_catalog.py -v
```

### Updating Layer Assignment

If a method's layer assignment changes:

1. Update `layer` field in catalog
2. Update canonical ID `@LAYER` component
3. Adjust statistics `by_layer` counts
4. Run validation scripts

### Version Management

Increment catalog version when:

- **PATCH (x.x.1)**: Minor corrections, metadata updates
- **MINOR (x.1.0)**: New methods added, non-breaking changes
- **MAJOR (1.0.0)**: Breaking changes to structure, layer redefinition

---

## Validation

### Automated Validation

Run these scripts to validate catalog integrity:

```bash
# Structure and content validation
python3 scripts/validate_canonical_catalog.py

# Directive compliance check
python3 scripts/check_directive_compliance.py

# Automated tests
pytest tests/test_canonical_method_catalog.py -v
```

### Manual Validation Checklist

- [ ] All methods have unique canonical IDs
- [ ] All calibration references exist in `calibration_registry.py`
- [ ] Layer counts match statistics
- [ ] Total calibrated matches method count
- [ ] No duplicate method names
- [ ] All required fields present
- [ ] Layer values are valid (Q, D, P, C, M)
- [ ] Calibration status values are valid
- [ ] Flags are properly formatted lists

---

## Statistics (Current)

### By Layer

| Layer | Count | Percentage |
|-------|-------|------------|
| D (Dimension) | 79 | 43.9% |
| P (Policy Area) | 46 | 25.6% |
| C (Congruence) | 36 | 20.0% |
| Q (Question) | 16 | 8.9% |
| M (Meta) | 3 | 1.7% |
| **Total** | **180** | **100%** |

### By Calibration Status

| Status | Count | Percentage |
|--------|-------|------------|
| CAL (Calibrated) | 180 | 100% |
| REQ (Required) | 0 | 0% |
| OPT (Optional) | 0 | 0% |
| DER (Derived) | 0 | 0% |
| INS (In-Script) | 0 | 0% |
| **Total** | **180** | **100%** |

---

## Migration History

### Version 1.0.0 (2025-11-09)

**Initial Release:**
- Migrated all 180 calibrations from YAML to code
- Established 5-layer taxonomy (Q, D, P, C, M)
- Deprecated 5 YAML calibration files
- Created validation infrastructure
- Achieved 100% calibration coverage

**YAML Files Deprecated:**
1. `calibracion_bayesiana.yaml`
2. `financia_callibrator.yaml`
3. `catalogo_principal.yaml`
4. `causal_exctractor.yaml`
5. `trazabilidad_cohrencia.yaml`

**Validation Results:**
- ✓ All directive requirements satisfied
- ✓ Catalog structure validated
- ✓ 23/23 automated tests passing
- ✓ Zero calibration gaps

---

## References

### Related Documentation

- **[CALIBRATION_PROCEDURE.md](./CALIBRATION_PROCEDURE.md)** - Mathematical procedures
- **[README_CALIBRATION.md](./README_CALIBRATION.md)** - Quick start guide
- **[CANONICAL_METHOD_NOTATION_SPEC.md](./docs/CANONICAL_METHOD_NOTATION_SPEC.md)** - Notation specification

### Source Files

- **Catalog:** `config/canonical_method_catalog.json`
- **Registry:** `src/saaaaaa/core/orchestrator/calibration_registry.py`
- **Appendix:** `config/embedded_calibration_appendix.json`

### Tools

- **Validation:** `scripts/validate_canonical_catalog.py`
- **Compliance:** `scripts/check_directive_compliance.py`
- **Tests:** `tests/test_canonical_method_catalog.py`

---

**Document Status:** CANONICAL  
**Enforcement:** MANDATORY  
**Last Review:** 2025-11-09  
**Next Review:** 2025-12-09 (monthly)

---
