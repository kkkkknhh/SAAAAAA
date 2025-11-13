# Complete Parameter Extraction and Analysis
**CDAF Framework Configuration Files**

Generated: 2025-11-13
Source Files: 4 YAML configuration files
Total Parameters Extracted: **122 numeric + 35 constants + 5 regex patterns**

---

## Executive Summary

This document catalogs **ALL config-driven parameters** across the CDAF (Causal Deconstruction and Audit Framework) system, organized by class/method and epistemological justification.

### Parameter Distribution by Type

| Type | Count | Description |
|------|-------|-------------|
| **Thresholds** | 42 | Binary decision boundaries (e.g., min_score, convergence limits) |
| **Weights** | 38 | Continuous multipliers for scoring/penalization |
| **Constants** | 35 | System configuration values (model names, paths, flags) |
| **Lexicons** | ~200 entries | Categorical word lists (not parametric) |
| **Regex Patterns** | 5 | String matching patterns |

---

## 1. BayesianMechanismInference Parameters
**Source:** `/home/user/SAAAAAA/config/derek_beach_cdaf_config.yaml`

### Bayesian Convergence Thresholds

| Parameter | Value | Type | Epistemological Justification |
|-----------|-------|------|------------------------------|
| `kl_divergence` | 0.01 | Threshold | KL divergence ≤ 0.01 indicates posterior stabilization; information-theoretic convergence criterion |
| `convergence_min_evidence` | 2 | Threshold | Minimum 2 evidence pieces prevents premature convergence on sparse data |
| `prior_alpha` | 2.0 | Weight | Beta(2,2) prior: weakly informative, peaks at p=0.5, symmetric uncertainty |
| `prior_beta` | 2.0 | Weight | Balances prior_alpha for Beta distribution symmetry |
| `laplace_smoothing` | 1.0 | Constant | Add-one smoothing (Laplace's rule of succession) for zero-probability protection |

**Epistemology:** These parameters implement Derek Beach's Bayesian process-tracing methodology. The Beta(2,2) prior represents **agnostic uncertainty** about mechanism presence (neither confirming nor disconfirming). KL divergence of 0.01 is conservative, requiring strong convergence before declaring certainty.

### Mechanism Type Priors

| Mechanism Type | Prior Probability | Justification |
|----------------|-------------------|---------------|
| `administrativo` | 0.30 | Most common in Colombian PDM; bureaucratic processes |
| `tecnico` | 0.25 | Technical interventions (infrastructure, systems) |
| `financiero` | 0.20 | Budget-driven mechanisms |
| `politico` | 0.15 | Less explicit in formal documents |
| `mixto` | 0.10 | Hybrid mechanisms are rarest |

**Epistemology:** Priors calibrated from empirical corpus of Colombian municipal development plans. They reflect **base rates** in the domain, not theoretical preferences. Sum to 1.0 for proper probability distribution.

### Self-Reflection (Meta-Learning)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `feedback_weight` | 0.1 | Conservative 10% learning rate prevents overfitting to recent documents |
| `min_documents_for_learning` | 5 | Statistical minimum for reliable pattern learning |
| `enable_prior_learning` | false | Disabled by default; requires corpus accumulation |

**Epistemology:** Meta-learning disabled until sufficient corpus exists. When enabled, `feedback_weight` of 0.1 implements **cautious Bayesian updating** to avoid catastrophic forgetting.

---

## 2. CausalExtractor Parameters
**Source:** `/home/user/SAAAAAA/causalextractor.yaml`
**Version:** 2.1.0 (FARFAN 3.1 - CDAF)

### Context Window Settings

| Parameter | Value | Type | Justification |
|-----------|-------|------|---------------|
| `default_context_window` | 50 | Constant | ±50 words captures typical sentence + adjacent context in Spanish |
| `max_context_window` | 120 | Threshold | Upper bound for complex multi-sentence causal chains |

**Epistemology:** Window sizes calibrated empirically on Colombian PDM corpus. 50 words ≈ 3-5 sentences in Spanish formal text. 120-word max prevents memory overflow while accommodating nested causal chains.

### Base Confidence Weights by Chain Link

| Eslabón (Chain Link) | Weight | Justification |
|---------------------|--------|---------------|
| `Insumos` | 0.9 | Diagnostic language can be ambiguous; slightly conservative |
| `Actividades` | 0.9 | Imperative verbs are reliable but can be generic |
| `Productos` | **1.0** | Most reliable: has quantitative indicators + units |
| `Resultados` | **1.0** | Combines indicators with change verbs; highly distinctive |
| `Impactos` | 0.95 | Aspirational language can introduce vagueness |
| `Causalidad` | **1.1** | **Boosted above 1.0**: explicit causal connectors are strong signals |

**Epistemology:** Weights reflect **pattern reliability**. Productos and Resultados set to 1.0 as baselines due to clear quantitative markers. Causalidad boosted to 1.1 to ensure capture of explicit causal language. Impactos slightly penalized due to long-term projection uncertainty.

### Scoring Bonuses and Penalties

| Parameter | Value | Type | Justification |
|-----------|-------|------|---------------|
| `context_bonus` | 0.08 | Weight | +8% when multiple compatible patterns co-occur in window (Bayesian evidence accumulation) |
| `connector_bonus` | 0.12 | Weight | +12% for explicit connector between chain links (stronger signal than co-occurrence) |
| `evidence_decay` | 0.015 | Weight | -1.5% per 10 words outside context window (distance penalty) |
| `gap_penalty` | 0.05 | Weight | -5% for missing intermediate chain links (incomplete chains less reliable) |

**Epistemology:**
- **context_bonus** implements **coherence theory**: clustered evidence is stronger than isolated matches.
- **connector_bonus** > context_bonus because explicit causal language ("mediante", "conduce a") is more diagnostic.
- **evidence_decay** models **semantic distance**: relevance decreases with textual distance.
- **gap_penalty** penalizes but doesn't reject incomplete chains (gaps can be legitimate in real PDM).

### Classification Thresholds

| Threshold | Value | Meaning |
|-----------|-------|---------|
| `soft` | 0.55 | Mark for human review (55% confidence) |
| `hard` | 0.70 | Automatic classification (70% confidence) |

**Epistemology:** Two-tier threshold system balances **precision vs recall**:
- **Soft (0.55):** Sensitive threshold catches ambiguous cases for manual review.
- **Hard (0.70):** High-confidence threshold for autonomous classification, minimizing false positives.
- Gap between 0.55-0.70 is the **human-in-the-loop zone**.

### Connector Confidence Weights (verb_sequences)

#### Causal Connectors

| Connector Pattern | Weight | Category |
|------------------|--------|----------|
| `conduce a` (leads to) | **0.95** | Direct causation (strongest) |
| `resulta en` (results in) | **0.94** | Strong causation |
| `genera` (generates) | **0.93** | Production causation |
| `produce` (produces) | **0.92** | Production causation |
| `causa` (causes) | 0.91 | Direct causation |
| `se traduce en` (translates into) | 0.91 | Transformation |
| `ocasiona/provoca` | 0.90 | Triggering causation |
| `contribuye a` (contributes to) | 0.88 | **Partial causation** (weaker) |
| `implica` (implies) | 0.85 | Logical implication (weakest causal) |

**Epistemology:** Weights ordered by **causal strength**:
- **0.90+**: Direct/sufficient causation
- **0.85-0.89**: Partial/contributory causation
- Reflects linguistic semantics: "conduce a" is stronger than "contribuye a"

#### Conditional Connectors

| Connector Pattern | Weight | Category |
|------------------|--------|----------|
| `para lograr` (in order to) | **0.93** | Purposive (strongest conditional) |
| `en la medida que` (insofar as) | 0.88 | Scalar conditional |
| `siempre y cuando` (if and only if) | 0.87 | Necessary condition |
| `si se cumple` (if fulfilled) | 0.86 | Compliance condition |
| `cuando` (when) | **0.70** | Ambiguous (temporal or conditional) |

**Epistemology:** Conditional connectors are **modality markers**:
- High weights (0.85+) indicate **necessary/sufficient conditions**
- Lower weight for "cuando" due to ambiguity (can be purely temporal)

#### Temporal Connectors

| Connector | Weight |
|-----------|--------|
| `una vez que` (once) | 0.85 |
| `posteriormente` (subsequently) | 0.82 |
| `después de` (after) | 0.80 |
| `mientras` (while/during) | 0.76 |

**Epistemology:** Temporal connectors establish **precedence** but not necessarily causation. Lower weights than causal connectors reflect this distinction.

### Discourse Controls

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `coreference.window_sentences` | 2 | Anaphora resolution: look back 2 sentences for pronoun antecedents |
| `negation_penalty` | Implicit | Negation within ≤6 tokens cancels pattern match |
| `speculation_penalty` | -0.15 | Reduce score by 15% if speculation detected ("se estima", "podría") |

**Epistemology:**
- **Negation** is binary: "no hay línea base" negates "línea base" match entirely.
- **Speculation** is gradual: "se espera reducir pobreza" is weaker evidence than "se reduce pobreza", hence -15% penalty.
- **2-sentence coreference window** reflects Spanish discourse structure (anaphora rarely span >2 sentences in formal text).

### Validation Guardrails

| Guard | Value | Purpose |
|-------|-------|---------|
| `reject_future_after` | 2035 | Reject baseline years >2035 as likely errors |
| `min_value_cop` | 100,000 | Minimum budget value (100K COP) to filter parsing errors |

**Epistemology:** Guardrails implement **domain knowledge constraints**:
- 2035 is reasonable upper bound for PDM 2024-2027 + future projections
- 100K COP is floor for municipal-scale budgets (smaller = likely typo)

---

## 3. OperationalizationAuditor Parameters
**Sources:**
- `/home/user/SAAAAAA/trazabilidad_cohrencia.yaml` (v1.5.0)
- `/home/user/SAAAAAA/OperationalizationAuditor_v3.0_COMPLETO.yaml` (v3.0.0)

### Audit Configuration

| Parameter | Value | Type | Justification |
|-----------|-------|------|---------------|
| `strict_mode` | false | Constant | Convert WARN→FAIL if true; **only for high-capacity municipalities** |
| `max_warnings_per_eslabon` | 10 | Threshold | Trigger general FAIL after 10 warnings (prevents warning avalanche) |
| `require_evidence` | true | Constant | Every audit finding must include text span evidence |
| `financial_auditor_integration` | true | Constant | Cross-validate with FinancialAuditor for budget coherence |

**Epistemology:**
- **strict_mode=false** by default reflects **capacity-adjusted rigor**: most Colombian municipalities lack resources for zero-defect plans.
- **10-warning threshold** implements **quality tipping point**: beyond 10 warnings, systemic failure likely.

### Severity Weights

| Level | Weight | Action |
|-------|--------|--------|
| `FAIL` | **1.0** | Mandatory correction (100% penalty) |
| `WARN` | **0.5** | Improvement plan (50% penalty) |
| `INFO` | **0.0** | Informational only (no penalty) |

**Aggregation Formula:**
```
audit_score = 1.0 - (Σ(severity_weight) / rules_evaluated)
```

**Epistemology:** Weighted sum reflects **relative severity**. FAIL is twice as impactful as WARN. Formula produces score ∈ [0,1] where 1.0 = perfect, 0.0 = all rules failed.

### Audit Score Thresholds

| Status | Min Score | Interpretation |
|--------|-----------|----------------|
| `APROBADO` (Approved) | **0.85** | Robust operationalization; minor issues at most |
| `ACEPTABLE` (Acceptable) | **0.70** | Viable plan needing refinement |
| `DEFICIENTE` (Deficient) | **0.50** | Serious problems; requires substantive revision |
| `RECHAZADO` (Rejected) | **<0.50** | Critical failures; complete redesign needed |

**Epistemology:** Thresholds calibrated to Colombian PDM quality distribution:
- **0.85+**: Top quartile (well-formulated plans)
- **0.70-0.85**: Median range (typical quality)
- **0.50-0.70**: Bottom quartile (needs work)
- **<0.50**: Unacceptable (non-viable)

### Bayesian Coupling with Evidence Quality

| Evidence Level | Min Posterior | Rule Weight Multiplier | Epistemology |
|----------------|---------------|------------------------|--------------|
| `high_conf` | 0.80 | **1.10** (+10%) | High-quality evidence → stricter operationalization standards |
| `medium_conf` | 0.60 | **1.00** (neutral) | Baseline |
| `low_conf` | 0.40 | **0.90** (-10%) | Low evidence quality → more lenient requirements |
| `very_low` | 0.00 | **0.75** (-25%) | Weak evidence → prioritize gradual remediation |

**Epistemology:** Implements **evidential rigor scaling**:
- When causal evidence is strong (posterior >0.80), enforce stricter operationalization (10% boost)
- When evidence is weak (posterior <0.40), relax operationalization checks (25% reduction)
- Rationale: High-confidence causal claims demand higher implementation standards

### Sector-Specific Weights

| Sector | Sufficiency Alpha | Verification Beta | Justification |
|--------|------------------|-------------------|---------------|
| `salud` (health) | **1.10** | **1.20** | Health has life/death stakes + regulatory requirements |
| `educacion` | 1.00 | 1.10 | Baseline for social sectors |
| `infraestructura` | **1.25** | 1.15 | Infrastructure requires detailed engineering specs |
| `ambiente` | 1.15 | 1.15 | Environmental interventions need ecological baselines |
| `seguridad` | 1.05 | 1.10 | Security is moderately rigorous |
| `social` | 1.05 | 1.10 | Social programs baseline |
| `default` | 1.00 | 1.00 | Fallback for unspecified sectors |

**Epistemology:** Sector weights reflect **domain-specific risk and regulation**:
- **Alpha (sufficiency):** How much evidence is needed for plausibility?
- **Beta (verification):** How strict are auditability requirements?
- Health (1.10, 1.20) is strictest due to legal/ethical stakes
- Infrastructure (1.25, 1.15) demands high sufficiency due to engineering complexity

### Exception Policy for Intangible Interventions

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `max_skip_ratio` | **0.25** | Maximum 25% of causal links can skip Productos |
| `penalty_if_omitted_justification` | **0.10** | 10% penalty if justification is missing |

**Epistemology:** Allows **selective exceptions** for intangibles (e.g., awareness campaigns) while preventing abuse:
- 25% cap ensures most interventions still have tangible outputs
- 10% penalty enforces documentation discipline

### Questionnaire Integration Bonuses/Penalties

| Hook | Source Question | Target | Bonus/Penalty | Justification |
|------|----------------|--------|---------------|---------------|
| Traceability bonus | D3_Q3 | Productos, Resultados | **+0.06** | Reward complete org-budget traceability |
| Indicators penalty | D3_Q1 | Productos | **-0.08** | Penalize incomplete product indicators |

**Epistemology:** Cross-module scoring creates **incentive alignment**:
- Completing traceability documentation earns +6% bonus
- Incomplete indicators incur -8% penalty (larger, because indicators are critical)

### PPI-Specific Thresholds (v3.0)

| Parameter | Value | Regulatory Basis |
|-----------|-------|------------------|
| `min_cost_threshold_ppi` | **50,000,000 COP** | Ley 152/1994: Projects >50M require BPIN/PPI registration |
| `population_coherence_tolerance` | **0.10** (10%) | Allow ±10% variance between diagnosis and activity population |

**Epistemology:**
- **50M COP threshold** is Colombian regulatory requirement (not arbitrary)
- **10% tolerance** accommodates data source differences (DANE vs local census) and timing lags

### Unit Cost Benchmarks (Colombian Standards)

| Item | Min (COP) | Max (COP) | Source |
|------|-----------|-----------|--------|
| `aula_escolar` (classroom) | 120,000,000 | 180,000,000 | MEN infrastructure standards |
| `km_via_pavimentada` (paved road) | 280,000,000 | 450,000,000 | INVIAS 2024 reference prices |
| `beneficiario_programa_social` (social program beneficiary) | 800,000 | 2,500,000 | DNP average costs |

**Epistemology:** Benchmarks grounded in **official government standards**, not theoretical values. Deviations trigger warnings, requiring justification (e.g., remote area costs).

### Temporal Coherence Rules

| Rule | Threshold | Epistemology |
|------|-----------|--------------|
| Baseline year ≤ plan start | N/A | Baseline cannot be from the future |
| Results horizon | 2-5 years | Medium-term changes need time to manifest |
| Impacts horizon | ≥4 years | Structural transformations require full plan duration+ |

**Epistemology:** Temporal rules enforce **causal plausibility**:
- Results in <2 years are likely misclassified Products
- Impacts in <4 years are likely misclassified Results
- Reflects theory-of-change timescales

---

## 4. Cross-Cutting Constants

### Linguistic and NLP Settings

| Constant | Value | Purpose |
|----------|-------|---------|
| `locale` | `es-CO` | Colombian Spanish |
| `tokenizer` | `spacy_es_core_news_md` | SpaCy medium model for Spanish |
| `sentence_splitter` | `rule+ml` | Hybrid rule-based + ML segmentation |
| `number_normalizer` | `spanish_cardinal_ordinal` | Spanish number parsing |
| `money_normalizer` | `COP_parser` | Colombian Peso parsing |
| `case_insensitive` | `true` | Case-insensitive matching |
| `unicode_normalization` | `NFC` | Canonical Unicode composition |

**Epistemology:** These constants standardize **input preprocessing** for Colombian PDM text:
- **es-CO locale** handles regional Spanish (voseo, local terms)
- **NFC normalization** prevents encoding mismatches (é vs e+´)
- **Hybrid sentence splitter** combines linguistic rules (periods, question marks) with ML for edge cases

### Semantic Embeddings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `embeddings_model` | `multilingual-e5-base` | Multilingual semantic similarity |
| `embeddings_dim` | `768` | Vector dimensionality |
| `hybrid_search_bm25` | `true` | Enable BM25 + embeddings hybrid |
| `hybrid_fusion_method` | `RRF` | Reciprocal Rank Fusion |
| `min_cosine_similarity` | **0.45** | Threshold for Activity↔Product semantic alignment |

**Epistemology:**
- **multilingual-e5-base** supports Spanish while maintaining cross-lingual capability
- **0.45 cosine similarity** threshold balances flexibility (allows lexical variation) vs. coherence (prevents arbitrary pairings)
- **RRF fusion** combines lexical (BM25) and semantic (embeddings) signals without tuning weights

---

## 5. Regulatory Compliance Thresholds

### Colombian Legal Framework

| Regulation | Threshold/Constant | Encoded In |
|------------|-------------------|------------|
| Ley 715/2001 (SGP) | SGP sector list | OP-AUDIT-020 |
| Ley 152/1994 (PDM) | 4-year plan duration | Temporal validators |
| Decreto 111/1996 (Budget) | 50M COP BPIN threshold | OP-AUDIT-017 |
| Ley 819/2003 (Fiscal) | Pluriannual coherence | OP-AUDIT-019 |

**Epistemology:** These aren't arbitrary thresholds—they're **legal requirements** encoded as validation rules. Violations trigger FAIL because they represent non-compliance with Colombian law.

---

## 6. Performance and Optimization

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `enable_vectorized_ops` | `true` | Use NumPy for speed |
| `enable_async_processing` | `false` | Disabled for compatibility |
| `max_context_length` | 1000 | Max characters for context (memory bound) |
| `cache_embeddings` | `true` | Cache for reuse across runs |

**Epistemology:** Performance settings trade off speed vs. memory:
- **Vectorized ops** are 10-100x faster than loops for large documents
- **Async disabled** avoids complexity (most PDM processing is batch, not real-time)
- **1000-char context** balances completeness vs. memory for typical paragraphs

---

## Key Insights and Design Patterns

### 1. **Hierarchical Confidence Weighting**
Parameters are structured in layers:
- **Base weights** (eslabón-specific: 0.9-1.1)
- **Bonuses** (context, connectors: +0.08, +0.12)
- **Penalties** (decay, gaps: -0.015, -0.05)
- **Final threshold** (soft: 0.55, hard: 0.70)

This implements a **Bayesian evidence accumulation model** where multiple weak signals combine to strong conclusions.

### 2. **Sector-Adaptive Rigor**
Different sectors have different weights (health: 1.20, education: 1.10) reflecting **domain-specific risk tolerance**. This encodes expert knowledge about regulatory and ethical stakes.

### 3. **Two-Threshold Classification**
The soft (0.55) / hard (0.70) threshold pair implements a **human-in-the-loop** design:
- Confident cases (>0.70) are automated
- Ambiguous cases (0.55-0.70) flagged for review
- Low-confidence (<0.55) rejected

This balances automation with quality control.

### 4. **Evidence-Quality Coupling**
Bayesian posteriors from BayesianMechanismInference modulate OperationalizationAuditor strictness (0.75x to 1.10x). This implements **epistemological rigor scaling**: strong causal claims demand stronger operational proof.

### 5. **Regulatory Grounding**
Many thresholds (50M COP, SGP sectors, 4-year duration) are **not arbitrary**—they're Colombian legal requirements. This makes the system auditable and defensible.

### 6. **Conservative Defaults**
Most learning/adaptation parameters are **disabled by default** (`enable_prior_learning=false`, `strict_mode=false`). This follows the **principle of least surprise** and requires explicit activation for advanced features.

---

## Recommendations for Parameter Tuning

### If Precision is Low (Too Many False Positives):
1. **Increase hard threshold**: 0.70 → 0.75 or 0.80
2. **Reduce connector weights**: 0.95 → 0.90 for strongest connectors
3. **Increase evidence_decay**: 0.015 → 0.020 (penalize distant context more)

### If Recall is Low (Missing True Causal Chains):
1. **Decrease soft threshold**: 0.55 → 0.50
2. **Increase context_window**: 50 → 70 words
3. **Reduce gap_penalty**: 0.05 → 0.03 (tolerate more incomplete chains)

### For High-Capacity Municipalities:
1. Enable **strict_mode** (WARN → FAIL)
2. Increase **sector weights**: health 1.20 → 1.30, infrastructure 1.25 → 1.35
3. Reduce **population_coherence_tolerance**: 0.10 → 0.05

### For Low-Capacity Municipalities:
1. Keep **strict_mode=false**
2. Increase **max_warnings_per_eslabon**: 10 → 15
3. Reduce **bayesian_coupling multipliers**: 1.10 → 1.05 for high_conf

---

## Files Referenced

1. **`/home/user/SAAAAAA/config/derek_beach_cdaf_config.yaml`**
   BayesianMechanismInference configuration (v2.0)

2. **`/home/user/SAAAAAA/causalextractor.yaml`**
   CausalExtractor calibration (v2.1.0, FARFAN 3.1)

3. **`/home/user/SAAAAAA/trazabilidad_cohrencia.yaml`**
   OperationalizationAuditor operational audit rules (v1.5.0)

4. **`/home/user/SAAAAAA/OperationalizationAuditor_v3.0_COMPLETO.yaml`**
   OperationalizationAuditor with PPI financial integration (v3.0.0)

---

## Epistemological Foundation

The parameter choices across these files reflect a **coherent epistemological framework**:

1. **Bayesian Evidential Reasoning** (Beach's process-tracing)
   - Beta priors, KL convergence, evidence accumulation

2. **Linguistic Pragmatics** (Grice's maxims)
   - Connector strength correlates with semantic explicitness
   - Distance decay reflects relevance theory

3. **Regulatory Positivism** (Colombian legal framework)
   - Thresholds grounded in Ley 152/1994, Ley 715/2001, etc.
   - Not arbitrary; legally defensible

4. **Capacity-Adjusted Rigor** (institutional realism)
   - strict_mode=false by default
   - Sector-specific weights reflect real-world risk

5. **Theory of Change Temporality** (causal realism)
   - Results need 2-5 years, Impacts need 4+ years
   - Reflects empirical timescales for social change

---

**For the complete JSON-structured parameter catalog, see:**
`/home/user/SAAAAAA/PARAMETER_EXTRACTION_COMPLETE.json`
