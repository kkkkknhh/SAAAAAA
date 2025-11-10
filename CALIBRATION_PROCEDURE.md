# Calibration Procedure - Mathematical & Procedural Documentation

**Version:** 1.0.0  
**Last Updated:** 2025-11-09  
**Status:** CANONICAL REFERENCE  
**Authority:** This document is the single authoritative source for calibration mathematical procedures, parameterization rules, and invariants.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Calibration Parameters](#calibration-parameters)
4. [Deterministic Rules & Invariants](#deterministic-rules--invariants)
5. [Contextual Refinement](#contextual-refinement)
6. [Procedure for Adding New Calibrations](#procedure-for-adding-new-calibrations)
7. [Examples](#examples)
8. [Change Log](#change-log)

---

## Overview

### Purpose

The calibration system provides rigorous, explicit parameterization for all policy analysis methods. Every method requiring calibration must have an entry in `src/saaaaaa/core/orchestrator/calibration_registry.py`. No method may execute with silent defaults or YAML-based configurations.

### Core Principles

1. **Zero Tolerance for Silent Fallbacks**: Missing calibrations raise `MissingCalibrationError`
2. **Deterministic & Versioned**: All calibrations are versioned with cryptographic hashes
3. **Context-Aware**: Base calibrations are refined based on execution context (dimension, policy area, document type)
4. **Single Source of Truth**: `calibration_registry.py` is the ONLY source for calibration parameters
5. **Layer Taxonomy Enforcement**: Every calibration is mapped to an analysis layer (Q, D, P, C, M)

### Key Invariants

- **INVARIANT-1**: Every calibrated method has exactly ONE base calibration entry
- **INVARIANT-2**: All calibration parameters are deterministic (no random initialization)
- **INVARIANT-3**: Contextual refinements are strictly additive/multiplicative transformations of base calibrations
- **INVARIANT-4**: No YAML files contain active calibration data (all must be deprecated)
- **INVARIANT-5**: Calibration hash changes trigger version increment

---

## Mathematical Foundations

### Bayesian Evidence Scoring

The core mathematical model for evidence evaluation is Bayesian inference with entropy-based confidence:

```
posterior = (likelihood × prior) / evidence
confidence = 1 - shannon_entropy(posterior_distribution)
quality_score = posterior × confidence × (1 - uncertainty_penalty)
```

Where:
- `likelihood` = P(evidence | hypothesis)
- `prior` = P(hypothesis) based on domain knowledge
- `evidence` = P(evidence) = normalization constant
- `shannon_entropy` = -Σ p(x) × log₂(p(x))
- `uncertainty_penalty` = calibrated parameter ∈ [0, 1]

### Evidence Aggregation

Multiple evidence snippets are aggregated using weighted arithmetic mean with contradiction penalty:

```
aggregated_score = (Σ wᵢ × scoreᵢ) / Σ wᵢ × (1 - contradiction_penalty)

where:
  wᵢ = aggregation_weight × reliability_factorᵢ
  contradiction_penalty = min(1.0, detected_contradictions / contradiction_tolerance)
```

### Sensitivity Analysis

Method sensitivity to evidence quality is modeled as:

```
effective_score = base_score × (1 - (1 - evidence_quality) × sensitivity)

where:
  sensitivity ∈ [0, 1]
  evidence_quality ∈ [0, 1]
```

Higher sensitivity means the method is more affected by low-quality evidence.

---

## Calibration Parameters

Every method calibration consists of 11 required parameters:

### 1. `score_min` (float)

- **Type**: Minimum score boundary
- **Range**: [0.0, 1.0]
- **Default**: 0.0
- **Purpose**: Lower bound for normalized scores
- **Invariant**: `score_min < score_max`

### 2. `score_max` (float)

- **Type**: Maximum score boundary
- **Range**: [0.0, 1.0]
- **Default**: 1.0
- **Purpose**: Upper bound for normalized scores
- **Invariant**: `score_max > score_min`

### 3. `min_evidence_snippets` (int)

- **Type**: Minimum evidence requirement
- **Range**: [1, 100]
- **Purpose**: Minimum number of evidence snippets required for valid analysis
- **Invariant**: `min_evidence_snippets ≤ max_evidence_snippets`
- **Deterministic Rule**: Methods requiring higher confidence have higher minimums

**Domain-Specific Ranges:**
- Light analysis (semantic chunking): 1-3
- Standard analysis (causal extraction): 2-5
- Rigorous analysis (financial auditing): 4-10
- Critical analysis (contradiction detection): 3-20

### 4. `max_evidence_snippets` (int)

- **Type**: Maximum evidence consideration
- **Range**: [1, 100]
- **Purpose**: Cap on evidence snippets to prevent over-fitting and performance degradation
- **Invariant**: `max_evidence_snippets ≥ min_evidence_snippets`
- **Deterministic Rule**: Higher values for comprehensive analysis methods

**Domain-Specific Ranges:**
- Focused analysis: 5-15
- Comprehensive analysis: 15-30
- Exhaustive analysis: 30-50

### 5. `contradiction_tolerance` (float)

- **Type**: Maximum allowed contradiction ratio
- **Range**: [0.0, 1.0]
- **Purpose**: Threshold for contradictory evidence before quality degradation
- **Deterministic Rule**: Financial/infrastructure = 0.0-0.05, Social = 0.1-0.15, General = 0.1-0.2

**Mathematical Interpretation:**
```
if (contradictions / total_evidence) > contradiction_tolerance:
    quality_penalty = exponential_decay(excess_contradictions)
```

### 6. `uncertainty_penalty` (float)

- **Type**: Penalty factor for uncertain evidence
- **Range**: [0.0, 1.0]
- **Purpose**: Weight applied to reduce scores when evidence has high entropy
- **Formula**: `final_score = base_score × (1 - uncertainty × uncertainty_penalty)`

**Domain-Specific Values:**
- High-certainty required (financial, legal): 0.30-0.40
- Standard certainty (general policy): 0.15-0.25
- Exploratory analysis: 0.10-0.15

### 7. `aggregation_weight` (float)

- **Type**: Relative importance weight
- **Range**: [0.0, 2.0]
- **Purpose**: Weight for aggregating this method's output with others
- **Deterministic Rule**: Critical methods (financial, causal) = 1.2-1.5, Standard = 0.9-1.1, Supporting = 0.6-0.8

### 8. `sensitivity` (float)

- **Type**: Evidence quality sensitivity
- **Range**: [0.0, 1.0]
- **Purpose**: How much the method's output degrades with poor evidence quality
- **Formula**: See "Sensitivity Analysis" in Mathematical Foundations
- **Deterministic Rule**: Bayesian methods = 0.85-0.95, Heuristic methods = 0.65-0.75

### 9. `requires_numeric_support` (bool)

- **Type**: Numeric evidence requirement flag
- **Purpose**: Indicates method requires quantitative data/evidence
- **Enforcement**: Methods with this flag=True will fail if no numeric evidence is available

### 10. `requires_temporal_support` (bool)

- **Type**: Temporal evidence requirement flag
- **Purpose**: Indicates method requires temporal/chronological evidence
- **Enforcement**: Methods with this flag=True will fail if no temporal markers are found

### 11. `requires_source_provenance` (bool)

- **Type**: Source tracking requirement flag
- **Purpose**: Indicates method requires traceable evidence sources
- **Enforcement**: Methods with this flag=True will fail if evidence lacks source references

---

## Deterministic Rules & Invariants

### Rule Set A: Parameter Relationships

1. **A1**: `min_evidence_snippets ≤ max_evidence_snippets` (MUST)
2. **A2**: `score_min < score_max` (MUST)
3. **A3**: `0.0 ≤ contradiction_tolerance ≤ 1.0` (MUST)
4. **A4**: `0.0 ≤ uncertainty_penalty ≤ 1.0` (MUST)
5. **A5**: `0.0 ≤ sensitivity ≤ 1.0` (MUST)
6. **A6**: `0.0 < aggregation_weight ≤ 2.0` (MUST)

### Rule Set B: Contextual Coherence

1. **B1**: Financial methods MUST have `contradiction_tolerance ≤ 0.05`
2. **B2**: Bayesian methods MUST have `sensitivity ≥ 0.80`
3. **B3**: Critical priority methods MUST have `uncertainty_penalty ≥ 0.25`
4. **B4**: Methods with `requires_numeric_support=True` MUST have `min_evidence_snippets ≥ 3`

### Rule Set C: Layer Taxonomy

1. **C1**: Question layer (Q) methods: focus on micro-evidence, `min_evidence_snippets ≤ 5`
2. **C2**: Dimension layer (D) methods: comprehensive analysis, `min_evidence_snippets ≥ 3`
3. **C3**: Policy Area layer (P) methods: domain-specific, contextual refinement required
4. **C4**: Congruence layer (C) methods: cross-dimensional, `aggregation_weight ≥ 1.0`
5. **C5**: Meta layer (M) methods: aggregation/reporting, lower sensitivity acceptable

### Rule Set D: Versioning & Hashing

1. **D1**: Any parameter change MUST increment `CALIBRATION_VERSION`
2. **D2**: Calibration hash MUST be recomputed after any change
3. **D3**: Hash collision MUST raise `CalibrationIntegrityError`
4. **D4**: Version strings MUST follow semantic versioning (MAJOR.MINOR.PATCH)

---

## Contextual Refinement

Base calibrations are refined based on execution context. Refinements are deterministic transformations:

### Document Type Modifiers

| Document Type | Evidence Modifier | Contradiction Modifier | Description |
|--------------|------------------|----------------------|-------------|
| `plan_desarrollo_municipal` | +40% | -40% | Municipal development plans (most strict) |
| `plan_estrategico` | +35% | -40% | Strategic plans |
| `politica_publica` | +30% | -35% | Public policies |
| `proyecto_inversion` | +25% | -30% | Investment projects |

**Formula:**
```python
refined_min_evidence = base_min_evidence × (1 + evidence_modifier)
refined_contradiction_tolerance = base_contradiction_tolerance × (1 + contradiction_modifier)
```

### Policy Area Modifiers

| Policy Area | Sensitivity Boost | Uncertainty Penalty Boost |
|------------|------------------|-------------------------|
| Fiscal/Financial | +0.10 | +0.10 |
| Infrastructure | +0.08 | +0.08 |
| Legal/Regulatory | +0.12 | +0.12 |
| Social Programs | +0.05 | +0.05 |
| Health/Education | +0.07 | +0.07 |

### Dimension Modifiers

Dimensions 1, 4, 9 (baseline, financial, implementation) receive:
- `contradiction_tolerance` × 0.5 (stricter)
- `uncertainty_penalty` + 0.10 (higher penalty)

---

## Procedure for Adding New Calibrations

### Step 1: Identify Method Requirements

1. Determine method's fully qualified name: `ClassName.method_name`
2. Identify layer assignment (Q, D, P, C, M)
3. Document method's purpose and analysis type
4. Identify domain (fiscal, social, causal, etc.)

### Step 2: Set Base Parameters

Use these decision trees:

**Evidence Requirements:**
```
if method is exploratory or helper:
    min_evidence_snippets = 1-2
    max_evidence_snippets = 10-15
elif method is standard analysis:
    min_evidence_snippets = 2-4
    max_evidence_snippets = 15-20
elif method is critical analysis:
    min_evidence_snippets = 4-8
    max_evidence_snippets = 20-30
```

**Contradiction Tolerance:**
```
if domain in [fiscal, infrastructure, legal]:
    contradiction_tolerance = 0.0-0.05
elif domain in [social, health, education]:
    contradiction_tolerance = 0.10-0.15
else:
    contradiction_tolerance = 0.10-0.20
```

**Uncertainty Penalty:**
```
if method is Bayesian or statistical:
    uncertainty_penalty = 0.25-0.35
elif method is rule-based or heuristic:
    uncertainty_penalty = 0.15-0.25
else:
    uncertainty_penalty = 0.10-0.20
```

**Sensitivity:**
```
if method uses Bayesian inference:
    sensitivity = 0.85-0.95
elif method uses complex heuristics:
    sensitivity = 0.75-0.85
else:
    sensitivity = 0.65-0.75
```

**Aggregation Weight:**
```
if priority == CRITICAL:
    aggregation_weight = 1.2-1.5
elif priority == HIGH:
    aggregation_weight = 1.0-1.2
elif priority == MEDIUM:
    aggregation_weight = 0.9-1.1
else:
    aggregation_weight = 0.6-0.9
```

### Step 3: Add to Registry

Add entry to `CALIBRATIONS` dict in `calibration_registry.py`:

```python
("ClassName", "method_name"): MethodCalibration(
    score_min=0.0,
    score_max=1.0,
    min_evidence_snippets=<value>,
    max_evidence_snippets=<value>,
    contradiction_tolerance=<value>,
    uncertainty_penalty=<value>,
    aggregation_weight=<value>,
    sensitivity=<value>,
    requires_numeric_support=<bool>,
    requires_temporal_support=<bool>,
    requires_source_provenance=<bool>,
    safe_default_allowed=<bool>,
    document_type=<optional_str>,
),
```

### Step 4: Update Catalog

Add entry to `config/canonical_method_catalog.json`:

```json
{
  "ClassName.method_name": {
    "canonical_id": "MOD:ClassName.method_name@LAYER[FLAGS]{CAL}",
    "class": "ClassName",
    "method_name": "method_name",
    "module": "MOD",
    "file": "source_file.py",
    "layer": "LAYER",
    "flags": ["FLAG1", "FLAG2"],
    "calibration_status": "CAL",
    "calibration_ref": "ClassName.method_name",
    "complexity": "MEDIUM",
    "priority": "HIGH"
  }
}
```

### Step 5: Increment Version

Update `CALIBRATION_VERSION` in `calibration_registry.py`:
- PATCH: Parameter value adjustment for existing method
- MINOR: New method calibration added
- MAJOR: Breaking change to calibration structure

### Step 6: Verify

Run validation:
```bash
python3 scripts/validate_canonical_catalog.py
python3 scripts/check_directive_compliance.py
pytest tests/test_canonical_method_catalog.py -v
```

---

## Examples

### Example 1: High-Precision Financial Analysis

```python
("FinancialAuditor", "trace_financial_allocation"): MethodCalibration(
    score_min=0.0,
    score_max=1.0,
    min_evidence_snippets=4,      # Needs multiple financial documents
    max_evidence_snippets=25,      # Comprehensive review
    contradiction_tolerance=0.0,   # Zero tolerance for financial contradictions
    uncertainty_penalty=0.35,      # High penalty for uncertain data
    aggregation_weight=1.2,        # Critical method, high weight
    sensitivity=0.95,              # Very sensitive to evidence quality
    requires_numeric_support=True, # MUST have numeric data
    requires_temporal_support=False,
    requires_source_provenance=True, # MUST track sources
),
```

**Rationale:**
- Financial auditing requires strict parameters
- Zero contradiction tolerance ensures financial integrity
- High sensitivity (0.95) means poor evidence significantly impacts results
- Requires numeric support enforces quantitative evidence

### Example 2: Exploratory Semantic Analysis

```python
("SemanticAnalyzer", "_classify_cross_cutting_themes"): MethodCalibration(
    score_min=0.0,
    score_max=1.0,
    min_evidence_snippets=1,       # Can work with minimal evidence
    max_evidence_snippets=10,      # Focused analysis
    contradiction_tolerance=0.15,  # Some ambiguity acceptable
    uncertainty_penalty=0.15,      # Lower penalty for exploratory work
    aggregation_weight=0.6,        # Supporting method
    sensitivity=0.65,              # Less sensitive, more robust
    requires_numeric_support=False,
    requires_temporal_support=False,
    requires_source_provenance=True,
),
```

**Rationale:**
- Exploratory theme classification is more permissive
- Lower evidence requirements (min=1) for initial classification
- Higher contradiction tolerance (0.15) allows thematic ambiguity
- Lower sensitivity (0.65) makes it more robust to evidence quality variations

### Example 3: Strategic Chunking (SPC Phase One)

```python
("StrategicChunkingSystem", "generate_smart_chunks"): MethodCalibration(
    score_min=0.0,
    score_max=1.0,
    min_evidence_snippets=5,       # Needs substantial context
    max_evidence_snippets=50,      # Can handle large documents
    contradiction_tolerance=0.15,  # Some contradictions in documents expected
    uncertainty_penalty=0.25,      # Moderate penalty
    aggregation_weight=1.5,        # Critical chunking method
    sensitivity=0.90,              # High quality needed
    requires_numeric_support=False,
    requires_temporal_support=True, # Timeline/chronology important
    requires_source_provenance=True,
    safe_default_allowed=False,    # No defaults allowed
    document_type="plan_desarrollo_municipal",
),
```

**Rationale:**
- Strategic chunking is critical for SPC phase one ingestion
- Moderate-to-high evidence requirements (5-50 snippets)
- Document type specified for context-aware refinement
- Temporal support required for chronological coherence
- No safe defaults ensures explicit parameterization

---

## Change Log

### Version 1.0.0 (2025-11-09)

**Added:**
- Initial release of CALIBRATION_PROCEDURE.md
- Complete mathematical foundations documentation
- All 11 calibration parameters with ranges and deterministic rules
- Contextual refinement formulas and modifiers
- Step-by-step procedure for adding new calibrations
- Three comprehensive examples (financial, semantic, strategic chunking)
- Four rule sets (A-D) for parameter validation

**Invariants Established:**
- INVARIANT-1: One base calibration per method
- INVARIANT-2: Deterministic parameterization (no randomness)
- INVARIANT-3: Additive/multiplicative context refinements
- INVARIANT-4: No active YAML calibrations
- INVARIANT-5: Hash-based version tracking

**References:**
- Related to PR 316 - Calibration System Consolidation Report
- Implements specifications from CALIBRATION_SYSTEM.md
- Aligned with CANONICAL_METHOD_NOTATION_SPEC.md layer taxonomy
- Supports calibration_registry.py (version 1.0.0)
- Supports calibration_context.py contextual refinement system

---

## References

### Primary Sources

1. **calibration_registry.py** - Single source of truth for all calibration parameters
2. **calibration_context.py** - Context-aware refinement logic
3. **canonical_method_catalog.json** - Layer taxonomy and method inventory
4. **CALIBRATION_SYSTEM.md** - High-level system documentation

### Related Documentation

- **CANONICAL_METHOD_NOTATION_SPEC.md** - Method notation and classification
- **docs/CALIBRATION_CONTEXT_GUIDE.md** - Context usage guide
- **RESOLUTION_43_REGISTRY_CATALOG.md** - Registry/catalog alignment policy

### Validation Tools

- `scripts/check_directive_compliance.py` - Verify calibration migration compliance
- `scripts/validate_canonical_catalog.py` - Validate catalog structure and integrity
- `tests/test_canonical_method_catalog.py` - Automated catalog testing

---

**Document Status:** CANONICAL  
**Enforcement:** MANDATORY  
**Last Review:** 2025-11-09  
**Next Review:** 2025-12-09 (monthly)

---
