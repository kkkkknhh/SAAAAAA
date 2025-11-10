# Three-Pillar Calibration System

## Overview

This is the production implementation of the **SUPERPROMPT: Three-Pillar, Layer-Sensitive Calibration System** for the SAAAAAA policy-pipeline stack.

The system provides rigorous, transparent, and reproducible calibration of method instances based on intrinsic quality, contextual fit, and governance metrics.

## Architecture

The calibration system is built on **three authoritative configuration pillars**:

### 1. Intrinsic Calibration (`config/intrinsic_calibration.json`)

Context-independent base quality scores for all methods:

- **b_theory**: Theoretical foundation quality
- **b_impl**: Implementation quality (tests, types, docs, error handling)
- **b_deploy**: Deployment maturity (validation runs, stability, failure rate)

Formula: `x_@b = w_th · b_theory + w_imp · b_impl + w_dep · b_deploy`

### 2. Contextual Parametrization (`config/contextual_parametrization.json`)

Rules and mappings for context-sensitive layers:

- **@chain**: Chain compatibility (schema/contract validation)
- **@u**: Unit-of-analysis sensitivity (g_M(U) functions)
- **@q**: Question compatibility (from questionnaire_monolith)
- **@d**: Dimension compatibility (DIM01-DIM06)
- **@p**: Policy area compatibility (PA01-PA10)
- **@C**: Interplay congruence (ensemble validity)
- **@m**: Meta/governance (transparency, governance, cost)

### 3. Fusion Specification (`config/fusion_specification.json`)

Weights for combining layer scores per role:

- **Linear weights** (a_ℓ): Direct layer contributions
- **Interaction weights** (a_ℓk): Weakest-link joint effects via min(x_ℓ, x_k)

Formula: `Cal(I) = Σ(a_ℓ · x_ℓ) + Σ(a_ℓk · min(x_ℓ, x_k))`

## Eight Fixed Layers

The system uses **exactly 8 layers** (no renaming allowed):

1. **@b** - Base (intrinsic quality)
2. **@chain** - Chain compatibility
3. **@u** - Unit-of-analysis sensitivity
4. **@q** - Question compatibility
5. **@d** - Dimension compatibility
6. **@p** - Policy compatibility
7. **@C** - Interplay congruence
8. **@m** - Meta/governance

## Eight Roles with Required Layers

Each role has specific required layers:

| Role | Required Layers |
|------|----------------|
| `INGEST_PDM` | @b, @chain, @u, @m |
| `STRUCTURE` | @b, @chain, @u, @m |
| `EXTRACT` | @b, @chain, @u, @m |
| `SCORE_Q` | @b, @chain, @q, @d, @p, @C, @u, @m |
| `AGGREGATE` | @b, @chain, @d, @p, @C, @m |
| `REPORT` | @b, @chain, @C, @m |
| `META_TOOL` | @b, @chain, @m |
| `TRANSFORM` | @b, @chain, @m |

## Usage

### Basic Calibration

```python
from calibration import calibrate, Context, ComputationGraph, EvidenceStore

# Define context
context = Context(
    question_id="Q001",
    dimension_id="DIM01",
    policy_id="PA01",
    unit_quality=0.85
)

# Create computation graph
graph = ComputationGraph(
    nodes={"node1"},
    edges=[],
    node_signatures={"node1": {}}
)

# Provide evidence
evidence = EvidenceStore(
    runtime_metrics={"runtime_ms": 500}
)

# Calibrate
certificate = calibrate(
    method_id="src.saaaaaa.flux.phases.run_score",
    node_id="node1",
    graph=graph,
    context=context,
    evidence_store=evidence
)

# Use calibrated score
print(f"Calibrated score: {certificate.calibrated_score:.4f}")
print(f"Layer breakdown: {certificate.layer_scores}")
```

### Accessing Certificate Details

```python
# Intrinsic score
print(f"Base quality: {certificate.intrinsic_score:.4f}")

# Individual layers
for layer, score in certificate.layer_scores.items():
    print(f"{layer}: {score:.4f}")

# Fusion details
fusion = certificate.fusion_formula
print(f"Linear sum: {fusion['linear_sum']:.4f}")
print(f"Interaction sum: {fusion['interaction_sum']:.4f}")

# Provenance
print(f"Config hash: {certificate.config_hash}")
print(f"Graph hash: {certificate.graph_hash}")
```

## Validation

### CI/CD Integration

Add to your CI pipeline:

```bash
python scripts/validate_calibration_configs.py
```

This validates:
- All three pillar config files exist and are well-formed
- Base weights sum to 1.0
- All intrinsic scores are in [0,1]
- All fusion weights sum to 1.0 per role
- All weights are non-negative

### Runtime Validation

```python
from calibration import validate_certificate

is_valid, errors = validate_certificate(certificate)
if not is_valid:
    print("Validation errors:", errors)
```

## Properties & Guarantees

The system enforces these mathematical properties:

✅ **P1. Boundedness**: `Cal(I) ∈ [0,1]`  
✅ **P2. Monotonicity**: `∂Cal/∂x_ℓ ≥ 0` (increasing any layer never decreases score)  
✅ **P3. Normalization**: `Σa_ℓ + Σa_ℓk = 1.0`  
✅ **P4. Completeness**: `L(M) ⊇ L_*(role(M))` (all required layers present)  
✅ **P5. Type Safety**: All inputs validated  
✅ **P6. Reproducibility**: Same inputs → same outputs (deterministic)  
✅ **P7. Non-triviality**: Different contexts → different scores  

## Anti-Universality Constraint

**No method may be perfectly compatible (1.0) with ALL questions, dimensions, and policies.**

This is enforced by setting policy default scores to 0.9 in the contextual config.

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_calibration_system.py -v
```

Tests cover:
- Config validation
- Data structures
- Layer computations
- Fusion operator
- Certificate generation
- Determinism
- Boundedness
- Validators

## File Structure

```
config/
  ├── intrinsic_calibration.json       # Pillar 1: base quality
  ├── contextual_parametrization.json  # Pillar 2: context rules
  └── fusion_specification.json        # Pillar 3: fusion weights

calibration/
  ├── __init__.py                      # Public API
  ├── data_structures.py               # Core types
  ├── layer_computers.py               # 8 layer functions
  ├── engine.py                        # Main calibrate()
  └── validators.py                    # Validation functions

scripts/
  └── validate_calibration_configs.py  # CI validation

tests/
  └── test_calibration_system.py       # Comprehensive tests
```

## Spec Compliance

This implementation fully complies with the **SUPERPROMPT: Three-Pillar, Layer-Sensitive Calibration System** specification:

- ✅ Section 0: Hierarchy of Truth
- ✅ Section 1: Core Objects (Γ, ctx, I)
- ✅ Section 2: Interplays (Ensembles)
- ✅ Section 3: Eight Fixed Layers
- ✅ Section 4: Role-Based Required Layers
- ✅ Section 5: Fusion Operator
- ✅ Section 6: Runtime Engine
- ✅ Section 7: Certificates
- ✅ Section 8: Validation & Governance
- ✅ Section 9: Deletion & Backward Incompatibility

## Key Design Decisions

1. **No Silent Defaults**: Missing data is an error, not silently assumed
2. **Explicit Configuration**: All calibration behavior comes from the three pillars
3. **Complete Audit Trail**: Certificates allow exact score reconstruction
4. **Deterministic**: Same inputs always produce same outputs
5. **Mathematically Rigorous**: All formulas match specification exactly

## Interaction Terms

The system uses `min(x_ℓ, x_k)` for interactions to capture weakest-link dynamics:

- **(@u, @chain)**: Plan quality only matters with sound wiring
- **(@chain, @C)**: Ensemble validity requires chain integrity
- **(@q, @d)**: Question-dimension alignment synergy
- **(@d, @p)**: Dimension-policy coherence
- **(@b, @chain)**: Base quality enhanced by proper wiring

## Migration from Legacy

Legacy calibration files are **NOT** authoritative. All calibration behavior MUST come from:

1. `canonical_method_catalog.json`
2. `questionnaire_monolith.json`
3. Three pillar configs

No fallbacks to undocumented weights or ad-hoc registries.

## Support

For issues or questions:
1. Check this README
2. Review test examples in `tests/test_calibration_system.py`
3. Validate configs with `scripts/validate_calibration_configs.py`
4. Refer to the SUPERPROMPT specification

## Version

Current version: **1.0.0**

Last updated: 2025-11-10
