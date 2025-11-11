# Executor Calibration Implementation Plan

## Objective
Implement rigorous calibration for executor methods following canonic_calibration_methods.md specification.

## Critical Understanding (Updated)

**Executors are NOT methods themselves - they are orchestrations of methods.**

- Each executor (e.g., D1Q1_Executor) orchestrates a **sequence of analytical methods**
- These underlying methods are already calibrated in `config/intrinsic_calibration.json` (1995 methods)
- Example: D1Q1_Executor uses methods like:
  - IndustrialPolicyProcessor.process
  - BayesianEvidenceScorer.compute_evidence_score
  - PolicyContradictionDetector._extract_quantitative_claims
  - etc.

**Calibration Strategy:**
1. **@b layer**: Aggregate intrinsic scores from constituent methods
2. **Contextual layers**: Apply to executor as orchestration unit
3. **Fusion operator**: Combine all layers

## Scope
Focus on executor methods in `src/saaaaaa/core/orchestrator/executors.py`:
- D1Q1_Executor through D6Q5_Executor (30 question-dimension executors)
- These are SCORE_Q role executors (answer micro-questions)

## Required Layers for SCORE_Q Role
Per canonic_calibration_methods.md Section 4.2:
```
L_*(SCORE_Q) = {@b, @chain, @q, @d, @p, @C, @u, @m}
```

All 8 layers must be computed for each executor.

## Implementation Steps

### Phase 1: Intrinsic Calibration (@b layer)
Create `config/executor_intrinsic_calibration.json` with:
- `b_theory`: Theoretical soundness of each executor's algorithm
- `b_impl`: Implementation quality (test coverage, types, error handling, docs)
- `b_deploy`: Deployment history (validation runs, stability, failure rate)

Formula: `x_@b = w_th · b_theory + w_imp · b_impl + w_dep · b_deploy`
Weights: `w_th + w_imp + w_dep = 1.0`

### Phase 2: Contextual Layer Computation
Implement functions in `calibration/executor_layer_computers.py`:

**@chain**: Chain compatibility
- Check if executor's input contracts match upstream outputs
- Scores: 0.0 (hard mismatch), 0.3 (missing optional), 0.6 (soft violation), 0.8 (warnings), 1.0 (clean)

**@u**: Unit-of-analysis sensitivity
- All SCORE_Q executors are U-sensitive (require quality PDT)
- Use g_QA function: `g_QA(U) = 1 - exp(-5(U - 0.5))`

**@q**: Question compatibility
- Extract from questionnaire_monolith.json which methods are primary/secondary/validators for each question
- Map executor names (D1Q1_Executor) to question IDs (Q_D1_01)

**@d**: Dimension compatibility
- D1Q1_Executor is primary for DIM01, check cross-compatibility matrix for other dimensions

**@p**: Policy compatibility
- Define policy compatibility matrix based on executor applicability

**@C**: Interplay congruence
- Executors may work in ensembles (analyzer + validator patterns)
- Check for declared fusion rules in config

**@m**: Meta/governance
- Transparency: formula export, trace completeness, log schema conformance
- Governance: version tagging, config hash, signature validation
- Cost: runtime and memory thresholds

### Phase 3: Fusion Operator
Implement in `calibration/executor_calibration_engine.py`:

```python
Cal(I) = Σ(a_ℓ · x_ℓ(I)) + Σ(a_ℓk · min(x_ℓ(I), x_k(I)))
```

With interaction terms:
- (@u, @chain): 0.15
- (@chain, @C): 0.12  
- (@q, @d): 0.08
- (@d, @p): 0.05

Constraint: All weights must sum to 1.0

### Phase 4: Configuration Files
Create:
1. `config/executor_intrinsic_calibration.json` - Intrinsic scores for all executors
2. `config/executor_contextual_params.json` - Rules for contextual layers
3. `config/executor_fusion_spec.json` - Fusion weights for SCORE_Q role

### Phase 5: Integration
- Add calibration hooks to `ExecutorBase.execute()` method
- Store calibration certificates in execution results
- Enable observability through calibration metrics

### Phase 6: Validation
- Create tests in `tests/test_executor_calibration.py`
- Verify all scores in [0,1]
- Verify deterministic computation
- Verify weight normalization
- Test sensitivity to context changes

## Key Constraints
1. **No hardcoded values** - All numeric parameters from config
2. **Deterministic** - Same inputs always produce same calibrated score
3. **Bounded** - All scores ∈ [0,1]
4. **Transparent** - Full audit trail in calibration certificates
5. **Context-sensitive** - Scores vary with (Q, D, P, U) context

## Success Criteria
- [ ] All 30 executors have complete intrinsic calibration
- [ ] All 8 layers compute correctly for SCORE_Q role
- [ ] Fusion operator produces valid scores [0,1]
- [ ] Configuration validates on load
- [ ] Tests pass with 100% coverage
- [ ] Calibration certificates generated per execution
- [ ] No silent defaults or implicit assumptions

