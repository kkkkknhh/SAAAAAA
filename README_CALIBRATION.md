# Calibration System - Complete Reference Guide

**Version:** 2.0.0 (Directive-Compliant)  
**Status:** Active - Single Source of Truth  
**Last Updated:** 2025-11-09  
**Authority:** Supersedes all previous calibration documentation

---

## ⚠️ CRITICAL: Canonical Method Catalog Authority

**SINGLE SOURCE OF TRUTH (as of Nov 9, 2025):**

```
File: config/canonical_method_catalog.json
Methods: 1,996
Version: 1.0.0
Status: AUTHORITATIVE - Use this and ONLY this
```

**DEPRECATED (DO NOT USE):**

```
File: config/rules/METODOS/catalogo_completo_canonico.json
Methods: 590
Version: 3.0.0 (Oct 29, 2025)  
Status: DEPRECATED - Compatibility shim only
```

**Hard Rules:**
1. All code MUST use `config/canonical_method_catalog.json`
2. Any references to the old catalog are deprecated
3. Module `catalogo_completo_canonico.py` exists only for legacy compatibility
4. All scripts migrated as of commit a51ccab

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Principles](#core-principles)
3. [System Architecture](#system-architecture)
4. [Canonical Method Catalog](#canonical-method-catalog)
5. [Calibration Registry](#calibration-registry)
6. [Calibration Status & Tracking](#calibration-status--tracking)
7. [Tools & Scripts](#tools--scripts)
8. [Usage Guide](#usage-guide)
9. [Migration & Stages](#migration--stages)
10. [Testing & Validation](#testing--validation)
11. [Directive Compliance](#directive-compliance)
12. [Troubleshooting](#troubleshooting)

---

## Executive Summary

The **Calibration System** is the comprehensive framework for managing method calibration, registration, and parametrization across the entire repository. This document consolidates all calibration-related information into a single authoritative reference.

**What This Document Replaces:**
- `CALIBRATION_SYSTEM.md`
- `CALIBRATION_GAPS_RESOLUTION.md`
- `CALIBRATION_IMPLEMENTATION_SUMMARY.md`
- `CATALOG_REGISTRY_ALIGNMENT_POLICY.md`
- `RESOLUTION_43_REGISTRY_CATALOG.md`
- `CANONICAL_METHOD_CATALOG_QUICKSTART.md`

All information from these documents has been consolidated here with the directive requirements taking precedence.

---

## Core Principles

### 1. Universal Coverage (Directive Requirement #1)
- **Every method** in the repository must be in the canonical catalog
- **No filters, no exceptions, no omissions**
- Currently tracking: **1,996 methods** (100% coverage)
- Single canonical source: `config/canonical_method_catalog.json`

### 2. Mechanical Decidability (Directive Requirement #2)
- Calibration requirements are **machine-readable**
- Every method has a `requires_calibration` boolean flag
- Decision algorithm documented in `scripts/build_canonical_method_catalog.py`
- No improvised criteria or subjective judgments

### 3. Complete Tracking (Directive Requirement #3)
- All calibration implementations are **visible and tracked**
- Four calibration statuses: `centralized`, `embedded`, `none`, `unknown`
- No hidden defaults or parallel math systems

### 4. Transitional Management (Directive Requirement #4)
- Embedded calibrations explicitly tracked in migration appendix
- Each has: priority, complexity, file location, parameters
- Migration backlog: 61 methods (3 critical, 10 high, 20 medium, 28 low)

### 5. Stage-Based Enforcement (Directive Requirement #5)
- Implementation proceeds in enforced stages
- Stage 0 (Baseline/Catalog): ✅ COMPLETE
- Stages 1-4 (Executors, Advanced, SPC, Remaining): Defined

### 6. Zero Tolerance for Silent Fallbacks
- Missing calibrations raise `MissingCalibrationError`
- No method executes without explicit calibration profile
- All runtime parameters controlled by `ExecutorConfig`

---

## System Architecture

### Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│         Canonical Method Catalog (Single Source)            │
│              config/canonical_method_catalog.json           │
│                    1,996 methods tracked                    │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
    ┌─────▼─────┐               ┌──────▼──────┐
    │ Centralized│               │  Embedded   │
    │ Calibration│               │ Calibration │
    │  Registry  │               │  Appendix   │
    │ 177 methods│               │  61 methods │
    └─────┬──────┘               └──────┬──────┘
          │                             │
          │                        (Migration)
          │                             │
          └──────────────┬──────────────┘
                         │
                ┌────────▼─────────┐
                │  ExecutorConfig  │
                │   Runtime Params │
                └──────────────────┘
```

### Core Components

**1. Canonical Method Catalog** (`config/canonical_method_catalog.json`)
- Universal method inventory (1,996 methods)
- Complete metadata: unique ID, canonical name, layer, inputs/outputs
- Calibration status for every method
- Generated by: `scripts/build_canonical_method_catalog.py`

**2. Calibration Registry** (`src/saaaaaa/core/orchestrator/calibration_registry.py`)
- 177 methods with centralized calibration
- Type: `MethodCalibration` dataclass
- Versioned (v1.0.0) with SHA256 hash
- Context-aware modifiers via `calibration_context.py`

**3. Embedded Calibration Appendix** (`config/embedded_calibration_appendix.json`)
- 61 methods with inline parametrization
- Migration metadata: priority, complexity, parameters
- Human-readable guide: `config/embedded_calibration_appendix.md`

**4. Calibration Context** (`src/saaaaaa/core/orchestrator/calibration_context.py`)
- Context-aware calibration modifiers
- Dimensions: question_id, policy_area, document_type, unit_of_analysis
- Adjusts base calibrations based on execution context

**5. Executor Config** (`src/saaaaaa/core/orchestrator/executor_config.py`)
- Runtime parameter configuration
- Single source of truth for execution
- Deterministic (seed, temperature) and versioned

---

## Canonical Method Catalog

### Overview

The **Canonical Method Catalog** is the single authoritative source for method identification, calibration tracking, and migration management.

**File:** `config/canonical_method_catalog.json`

### Statistics

- **Total Methods:** 1,996
- **Coverage:** 100% (no filters applied)
- **Requiring Calibration:** 558 (28.0%)
- **Layers:** 7 (orchestrator, analyzer, processor, utility, ingestion, executor, unknown)

### Method Metadata Schema

Each method includes:

```python
{
  # Core identification
  "unique_id": str,           # SHA256 hash (16 chars)
  "canonical_name": str,      # module.ClassName.method_name
  "method_name": str,
  "class_name": str | None,
  "file_path": str,
  
  # Layer positionality
  "layer": str,               # orchestrator, executor, analyzer, etc.
  "layer_position": int,      # Position within layer (0-based)
  
  # Signature and interface
  "signature": str,
  "input_parameters": [...],
  "return_type": str | None,
  
  # Calibration tracking (CRITICAL)
  "requires_calibration": bool,     # Machine-readable flag
  "calibration_status": str,        # centralized|embedded|none|unknown
  "calibration_location": str|None, # File:line if embedded
  
  # Additional metadata
  "docstring": str | None,
  "decorators": [...],
  "is_async": bool,
  "is_private": bool,
  "complexity": str,           # low|medium|high
  
  # Source tracking
  "line_number": int,
  "source_hash": str,
  "last_analyzed": str         # ISO timestamp
}
```

### Calibration Status Taxonomy

| Status | Count | % of Total | % of Requiring | Meaning |
|--------|-------|------------|----------------|---------|
| **centralized** | 177 | 8.9% | 31.7% | Uses `calibration_registry.py` ✅ |
| **embedded** | 61 | 3.1% | 10.9% | Inline parametrization ⚠️ |
| **none** | 1,438 | 72.0% | N/A | Doesn't require calibration ✅ |
| **unknown** | 320 | 16.0% | 57.3% | Needs investigation ⚠️ |

### How to Use

**Build/Update Catalog:**
```bash
python3 scripts/build_canonical_method_catalog.py
```

**Query Catalog:**
```python
import json

with open('config/canonical_method_catalog.json') as f:
    catalog = json.load(f)

# Get all methods requiring calibration
requiring = [m for m in catalog['methods'] if m['requires_calibration']]
print(f"Methods requiring calibration: {len(requiring)}")

# Get methods by layer
orchestrator_methods = [m for m in catalog['methods'] if m['layer'] == 'orchestrator']
print(f"Orchestrator methods: {len(orchestrator_methods)}")

# Get centralized calibrations
centralized = catalog['calibration_tracking']['centralized']
print(f"Centralized calibrations: {len(centralized)}")
```

---

## Calibration Registry

### Overview

The **Calibration Registry** contains explicit calibration parameters for methods using centralized calibration.

**File:** `src/saaaaaa/core/orchestrator/calibration_registry.py`

### MethodCalibration Schema

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

### Usage

**Basic Resolution:**
```python
from saaaaaa.core.orchestrator.calibration_registry import resolve_calibration

# Strict mode (raises MissingCalibrationError if not found)
calib = resolve_calibration("BayesianEvidenceScorer", "compute_evidence_score", strict=True)

# Access parameters
threshold = calib.contradiction_tolerance
min_snippets = calib.min_evidence_snippets
```

**Context-Aware Resolution:**
```python
from saaaaaa.core.orchestrator.calibration_registry import resolve_calibration_with_context

calib = resolve_calibration_with_context(
    "BayesianNumericalAnalyzer",
    "evaluate_policy_metric",
    question_id="D9Q1",
    policy_area="fiscal",
    unit_of_analysis="financial",
    document_type="plan_desarrollo_municipal"
)
```

### Context Modifiers

**Document Type Modifiers:**
- `plan_desarrollo_municipal`: +40% evidence, -40% contradiction tolerance (most strict)
- `politica_publica`: +25% evidence, -30% contradiction tolerance
- `plan_sectorial`: +30% evidence, -35% contradiction tolerance

**Dimension Modifiers:**
- D1 (baseline gaps): +30% evidence, +10% sensitivity
- D9 (financial): +35% evidence, -50% contradiction tolerance, +30% sensitivity

**Policy Area Modifiers:**
- fiscal: +30% evidence, -40% contradiction tolerance
- infrastructure: +40% evidence, -50% contradiction tolerance

---

## Calibration Status & Tracking

### Status Definitions

**1. Centralized (177 methods - COMPLIANT)**
- Calibration defined in `calibration_registry.py`
- Retrieved via `resolve_calibration()` or `resolve_calibration_with_context()`
- Version tracked and hashed
- Context-aware modifiers applied

**2. Embedded (61 methods - MIGRATION NEEDED)**
- Local/inline parametrization in method code
- Explicitly tracked in `config/embedded_calibration_appendix.json`
- Migration backlog with priorities
- Transitional anomaly - will be migrated to centralized

**3. None (1,438 methods - COMPLIANT)**
- Does not require calibration
- Simple utility methods, getters/setters, etc.
- Explicitly marked as `requires_calibration=false`

**4. Unknown (320 methods - INVESTIGATION NEEDED)**
- Calibration requirement unclear
- Needs manual investigation
- May be reclassified to centralized/embedded/none

### Migration Backlog

**Total Embedded Calibrations:** 61 methods

**By Priority:**
- **Critical (3):** Executors, scoring methods - immediate action
- **High (10):** Analyzers with many parameters - short-term
- **Medium (20):** Other analyzers/processors - medium-term
- **Low (28):** Utilities - long-term

**Migration Process:**
1. Review method in `config/embedded_calibration_appendix.md`
2. Extract parameters from code
3. Create `MethodCalibration` entry in `calibration_registry.py`
4. Update method to use `resolve_calibration()`
5. Remove embedded parameters
6. Test to ensure behavior unchanged
7. Re-run catalog build to update status

---

## Tools & Scripts

### Primary Tools (Use These)

**1. Build Canonical Catalog**
```bash
python3 scripts/build_canonical_method_catalog.py
```
- Scans entire repository (no filters)
- Generates `config/canonical_method_catalog.json`
- Determines calibration requirements
- Cross-references with `calibration_registry.py`

**2. Detect Embedded Calibrations**
```bash
python3 scripts/detect_embedded_calibrations.py
```
- Finds methods with inline parametrization
- Generates `config/embedded_calibration_appendix.json`
- Assigns migration priorities
- Creates human-readable appendix

**3. Validate Catalog**
```bash
python3 scripts/validate_canonical_catalog.py
```
- Checks completeness and consistency
- Validates unique IDs
- Verifies calibration status
- Ensures migration backlog matches

**4. Check Directive Compliance**
```bash
python3 scripts/check_directive_compliance.py
```
- Validates all 5 directive requirements
- Identifies violations with severity
- Provides remediation guidance

### Legacy Tools (Reference Only)

These scripts were used in previous iterations and are now superseded by the primary tools above:

- ~~`scripts/analyze_calibration_gaps.py`~~ → Use `build_canonical_method_catalog.py`
- ~~`scripts/audit_catalog_registry_alignment.py`~~ → Use `validate_canonical_catalog.py`
- ~~`scripts/build_calibration_decisions.py`~~ → Data now in canonical catalog
- ~~`scripts/build_method_usage_intelligence.py`~~ → Data now in canonical catalog

**Note:** Legacy scripts may still work but use outdated data structures. Always use the primary tools listed above.

---

## Usage Guide

### Quick Start

**1. Check Current Status**
```bash
# Validate catalog
python3 scripts/validate_canonical_catalog.py

# Check compliance
python3 scripts/check_directive_compliance.py
```

**2. Add New Method with Calibration**

**Step 1:** Add calibration to registry
```python
# In src/saaaaaa/core/orchestrator/calibration_registry.py

CALIBRATIONS: Dict[Tuple[str, str], MethodCalibration] = {
    # ... existing calibrations ...
    
    ("YourClassName", "your_method_name"): MethodCalibration(
        score_min=0.0,
        score_max=1.0,
        min_evidence_snippets=5,
        max_evidence_snippets=20,
        contradiction_tolerance=0.1,
        uncertainty_penalty=0.3,
        aggregation_weight=1.0,
        sensitivity=0.9,
        requires_numeric_support=True,
        requires_temporal_support=False,
        requires_source_provenance=True,
    ),
}
```

**Step 2:** Use in your method
```python
from saaaaaa.core.orchestrator.calibration_registry import resolve_calibration

class YourClassName:
    def your_method_name(self, data):
        # Resolve calibration
        calib = resolve_calibration("YourClassName", "your_method_name")
        
        # Use calibration parameters
        threshold = calib.contradiction_tolerance
        min_snippets = calib.min_evidence_snippets
        weight = calib.aggregation_weight
        
        # Your logic here
        pass
```

**Step 3:** Rebuild catalog
```bash
python3 scripts/build_canonical_method_catalog.py
```

**3. Migrate Embedded Calibration**

See `config/embedded_calibration_appendix.md` for detailed migration guides for each method.

---

## Migration & Stages

### Implementation Stages

**Stage 0: Baseline/Catalog** ✅ COMPLETE
- Canonical method catalog established
- 1,996 methods tracked
- Calibration status determined
- Migration backlog created

**Stage 1: Executors** 🔄 NOT STARTED
- Wire 30 executors through centralized calibration
- Zero embedded calibrations in executors
- Comprehensive testing

**Stage 2: Advanced Methods** 🔄 NOT STARTED
- Extend to methods feeding executors
- Analyzer and processor methods
- Upstream dependency mapping

**Stage 3: SPC/Phase 1** 🔄 NOT STARTED
- Smart Policy Chunks system
- Phase 1 orchestration
- Quality gates

**Stage 4: Remaining Phases** 🔄 NOT STARTED
- All remaining phases (D1-D10)
- Universal calibration coverage
- Zero embedded calibrations

**See `IMPLEMENTATION_STAGES.md` for detailed stage definitions and exit criteria.**

### Migration Timeline

**Immediate (48 hours):**
- Migrate 3 critical embedded calibrations
- Begin unknown status investigation

**Short-term (2 weeks):**
- Migrate 10 high priority methods
- Complete executor inventory
- Setup CI/CD enforcement

**Medium-term (1 month):**
- Complete Stage 1 implementation
- Migrate 20 medium priority methods
- Reduce unknown status to < 100

**Long-term (3 months):**
- Zero embedded calibrations
- Complete all 4 stages
- < 1% unknown status

---

## Testing & Validation

### Automated Tests

**Primary Test Suite:**
```bash
pytest tests/test_canonical_method_catalog.py -v
```

**31 tests covering:**
- Directive compliance (5 tests)
- Method metadata completeness (5 tests)
- Calibration tracking (5 tests)
- Migration backlog (3 tests)
- Layer classification (3 tests)
- Summary consistency (3 tests)
- Calibration coverage (3 tests)
- Integration tests (1 test)
- Method properties (3 tests)

**Legacy Test Suites (Still Valid):**
```bash
pytest tests/test_calibration_completeness.py -v  # 14 tests
pytest tests/test_calibration_stability.py -v     # Stability tests
pytest tests/test_calibration_context.py -v       # Context tests
```

### Validation Commands

```bash
# Full validation suite
python3 scripts/validate_canonical_catalog.py
python3 scripts/check_directive_compliance.py
pytest tests/test_canonical_method_catalog.py -v

# Quick check
python3 scripts/validate_canonical_catalog.py --quick
```

---

## Directive Compliance

### Compliance Status

**Overall:** ✅ DIRECTIVE COMPLIANT (with documented action items)

**Requirement 1: Universal Coverage** ✅ COMPLIANT
- 1,996 methods tracked (100% coverage)
- No filters, no exceptions
- Single canonical catalog

**Requirement 2: Mechanical Decidability** ✅ COMPLIANT
- All methods have `requires_calibration` flag
- 558 methods identified as requiring calibration
- Decision algorithm documented

**Requirement 3: Calibration Tracking** ✅ COMPLIANT
- Complete tracking: centralized (177), embedded (61), none (1,438), unknown (320)
- All implementations visible with file:line locations

**Requirement 4: Transitional Cases** ✅ COMPLIANT
- 61 embedded calibrations explicitly tracked
- Migration appendix with priorities and complexity
- Authoritative backlog for migration

**Requirement 5: Stage Enforcement** ✅ COMPLIANT
- Stage framework documented
- Stage 0 complete, Stages 1-4 defined
- No parallel math or hidden defaults

**See `DIRECTIVE_COMPLIANCE_SUMMARY.md` for detailed compliance report.**

---

## Troubleshooting

### Common Issues

**1. MissingCalibrationError**

```python
MissingCalibrationError: Missing calibration for method 'ClassName.method_name'
```

**Solution:**
- Add calibration to `calibration_registry.py`
- Or mark as `safe_default_allowed=True` if appropriate
- Rebuild catalog: `python3 scripts/build_canonical_method_catalog.py`

**2. Catalog Out of Date**

**Symptoms:** Method not found in catalog, counts don't match

**Solution:**
```bash
python3 scripts/build_canonical_method_catalog.py
python3 scripts/detect_embedded_calibrations.py
python3 scripts/validate_canonical_catalog.py
```

**3. Unknown Calibration Status**

**Symptoms:** Method shows `calibration_status: "unknown"`

**Investigation:**
1. Check if method truly needs calibration
2. Review method name and layer
3. Look for embedded parameters in code
4. Update catalog if reclassified

**4. Test Failures**

```bash
# Run specific test
pytest tests/test_canonical_method_catalog.py::TestClassName::test_name -v

# Check what changed
git diff config/canonical_method_catalog.json
```

---

## Quick Reference

### Key Files

**Data:**
- `config/canonical_method_catalog.json` - Complete method inventory
- `config/embedded_calibration_appendix.json` - Migration backlog
- `config/embedded_calibration_appendix.md` - Migration guide

**Code:**
- `src/saaaaaa/core/orchestrator/calibration_registry.py` - Centralized calibrations
- `src/saaaaaa/core/orchestrator/calibration_context.py` - Context modifiers
- `src/saaaaaa/core/orchestrator/executor_config.py` - Runtime config

**Tools:**
- `scripts/build_canonical_method_catalog.py` - Catalog builder
- `scripts/detect_embedded_calibrations.py` - Embedded detector
- `scripts/validate_canonical_catalog.py` - Catalog validator
- `scripts/check_directive_compliance.py` - Compliance checker

**Tests:**
- `tests/test_canonical_method_catalog.py` - 31 comprehensive tests

**Docs:**
- `README_CALIBRATION.md` - This file (single source of truth)
- `IMPLEMENTATION_STAGES.md` - Stage definitions
- `DIRECTIVE_COMPLIANCE_SUMMARY.md` - Compliance details

### Key Commands

```bash
# Build/update everything
python3 scripts/build_canonical_method_catalog.py
python3 scripts/detect_embedded_calibrations.py

# Validate
python3 scripts/validate_canonical_catalog.py
python3 scripts/check_directive_compliance.py

# Test
pytest tests/test_canonical_method_catalog.py -v
```

---

## Document History

**Version 2.0.0 (2025-11-09) - Directive-Compliant Consolidation**
- Consolidated 5+ previous documents into single source
- Aligned with directive requirements
- Added comprehensive tool/script reference
- Clarified legacy vs. current tools
- Supersedes all previous calibration documentation

**Previous Documents (Now Deprecated):**
- CALIBRATION_SYSTEM.md
- CALIBRATION_GAPS_RESOLUTION.md
- CALIBRATION_IMPLEMENTATION_SUMMARY.md
- CATALOG_REGISTRY_ALIGNMENT_POLICY.md
- RESOLUTION_43_REGISTRY_CATALOG.md
- CANONICAL_METHOD_CATALOG_QUICKSTART.md

**Note:** Information from deprecated documents has been preserved and integrated into this document. The deprecated documents are retained for historical reference only.

---

## Support & Contact

**Questions or Issues:**
1. Check this README first
2. Run validation: `python3 scripts/check_directive_compliance.py`
3. Review `DIRECTIVE_COMPLIANCE_SUMMARY.md`
4. Check test output: `pytest tests/test_canonical_method_catalog.py -v`

**Contributing:**
- All new methods requiring calibration must be added to `calibration_registry.py`
- No new embedded calibrations permitted
- Run full validation before committing
- Follow the directive requirements (universal coverage, mechanical decidability, etc.)

---

**Last Updated:** 2025-11-09  
**Version:** 2.0.0  
**Status:** Active - Single Source of Truth for All Calibration Matters
