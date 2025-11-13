# Quick Reference: CDAF Parameter Catalog

**All Config-Driven Parameters** | Organized by Function
Generated: 2025-11-13

---

## 🎯 Most Critical Parameters (Tuning Priorities)

| Parameter | Location | Default | Impact | When to Adjust |
|-----------|----------|---------|--------|----------------|
| `scoring.threshold.hard` | causalextractor.yaml | **0.70** | Auto-classification confidence | ↑ if too many false positives, ↓ if missing chains |
| `scoring.threshold.soft` | causalextractor.yaml | **0.55** | Human review trigger | ↓ to catch more ambiguous cases |
| `thresholds.APROBADO.min_score` | OperationalizationAuditor | **0.85** | Approval threshold | ↑ for high-capacity municipalities |
| `strict_mode` | OperationalizationAuditor | **false** | WARN→FAIL conversion | Set `true` only for well-resourced municipalities |
| `kl_divergence` | derek_beach_cdaf_config.yaml | **0.01** | Bayesian convergence | ↓ for faster convergence (less conservative) |
| `min_cost_threshold_ppi` | OperationalizationAuditor v3.0 | **50M COP** | BPIN requirement trigger | **DO NOT CHANGE** (legal requirement) |

---

## 📊 Bayesian Inference Parameters

### BayesianMechanismInference (derek_beach_cdaf_config.yaml)

```yaml
bayesian_thresholds:
  kl_divergence: 0.01              # Convergence threshold (lower = stricter)
  convergence_min_evidence: 2      # Min evidence pieces for convergence
  prior_alpha: 2.0                 # Beta prior shape (successes)
  prior_beta: 2.0                  # Beta prior shape (failures)
  laplace_smoothing: 1.0           # Add-one smoothing for zeros

mechanism_type_priors:             # Prior probabilities (sum=1.0)
  administrativo: 0.30             # Most common in PDM
  tecnico: 0.25
  financiero: 0.20
  politico: 0.15
  mixto: 0.10                      # Least common

self_reflection:
  feedback_weight: 0.1             # Learning rate (conservative)
  min_documents_for_learning: 5    # Corpus size before meta-learning
  enable_prior_learning: false     # Disabled by default
```

**Quick Tuning:**
- **More conservative inference:** ↓ `kl_divergence` to 0.005
- **Faster convergence:** ↑ `kl_divergence` to 0.02
- **Adjust domain priors:** Modify `mechanism_type_priors` based on your corpus

---

## 🔗 Causal Extraction Parameters

### CausalExtractor (causalextractor.yaml)

```yaml
# Context windows
default_context_window: 50        # ±50 words for pattern matching
max_context_window: 120           # Upper bound for complex chains

# Base confidence weights (by chain link)
scoring.base_weight:
  Insumos: 0.9                    # Diagnostics (slightly uncertain)
  Actividades: 0.9                # Activities (imperative verbs)
  Productos: 1.0                  # Products (BASELINE - has indicators)
  Resultados: 1.0                 # Results (BASELINE - has change)
  Impactos: 0.95                  # Impacts (long-term, some vagueness)
  Causalidad: 1.1                 # BOOSTED - explicit connectors

# Scoring adjustments
scoring:
  context_bonus: 0.08             # +8% for pattern co-occurrence
  connector_bonus: 0.12           # +12% for explicit connector
  evidence_decay: 0.015           # -1.5% per 10 words distance
  gap_penalty: 0.05               # -5% for missing chain links

  threshold:
    soft: 0.55                    # Human review zone
    hard: 0.70                    # Auto-classification

  chain_consistency:
    gap_penalty: 0.05             # Penalty for incomplete chains

# Discourse controls
discourse_controls:
  window_sentences: 2             # Coreference lookback
  speculation_penalty: -0.15      # Reduce score 15% if "podría", "se estima"
  negation_cancels: true          # "no hay" cancels pattern within 6 tokens

# Guardrails
validations:
  reject_future_after: 2035       # Reject baseline years >2035
  min_value_cop: 100000           # Min budget (100K COP)
```

**Quick Tuning:**
- **More aggressive extraction:** ↓ `threshold.soft` to 0.50
- **More conservative:** ↑ `threshold.hard` to 0.75
- **Larger context:** ↑ `default_context_window` to 70
- **Stricter causal links:** ↑ `gap_penalty` to 0.10

### Connector Confidence Weights (Top 10)

| Connector | Weight | Category | Strength |
|-----------|--------|----------|----------|
| `conduce a` | 0.95 | Causal | Strongest |
| `resulta en` | 0.94 | Causal | Very strong |
| `para lograr` | 0.93 | Operacional | Purposive |
| `genera` | 0.93 | Causal | Strong |
| `produce` | 0.92 | Causal | Strong |
| `se traduce en` | 0.91 | Causal | Transformation |
| `causa` | 0.91 | Causal | Direct |
| `a través de` | 0.90 | Operacional | Instrumental |
| `en la medida que` | 0.88 | Condicional | Scalar |
| `contribuye a` | 0.88 | Causal | Partial (weaker) |

**Quick Tuning:**
- To reduce false positives: Lower top-5 weights by 0.05
- To increase recall: Raise weights 0.80-0.85 by 0.05

---

## ✅ Operationalization Audit Parameters

### OperationalizationAuditor (trazabilidad_cohrencia.yaml v1.5.0)

```yaml
audit_config:
  strict_mode: false              # WARN→FAIL conversion (use cautiously)
  max_warnings_per_eslabon: 10    # Trigger FAIL after 10 warnings
  require_evidence: true          # Include text spans in findings

# Severity weights
execution_config.severity_levels:
  FAIL: {weight: 1.0}             # Full penalty
  WARN: {weight: 0.5}             # Half penalty
  INFO: {weight: 0.0}             # No penalty

# Audit score thresholds
thresholds:
  APROBADO: {min_score: 0.85}     # Approved (85%+)
  ACEPTABLE: {min_score: 0.70}    # Acceptable (70-84%)
  DEFICIENTE: {min_score: 0.50}   # Deficient (50-69%)
  RECHAZADO: {max_score: 0.49}    # Rejected (<50%)

# Bayesian coupling (evidence quality → rigor)
bayesian_coupling.posterior_thresholds:
  high_conf:   {min: 0.80, multiplier: 1.10}   # +10% stricter
  medium_conf: {min: 0.60, multiplier: 1.00}   # Baseline
  low_conf:    {min: 0.40, multiplier: 0.90}   # -10% lenient
  very_low:    {min: 0.00, multiplier: 0.75}   # -25% lenient

# Sector-specific adjustments
sector_weights:
  salud:          {sufficiency_alpha: 1.10, verification_beta: 1.20}
  educacion:      {sufficiency_alpha: 1.00, verification_beta: 1.10}
  infraestructura:{sufficiency_alpha: 1.25, verification_beta: 1.15}
  ambiente:       {sufficiency_alpha: 1.15, verification_beta: 1.15}
  seguridad:      {sufficiency_alpha: 1.05, verification_beta: 1.10}
  social:         {sufficiency_alpha: 1.05, verification_beta: 1.10}
  default:        {sufficiency_alpha: 1.00, verification_beta: 1.00}

# Exception policy (intangibles)
exceptions_policy.intangible_interventions:
  max_skip_ratio: 0.25            # Max 25% can skip Productos
  penalty_if_omitted_justification: 0.10

# Questionnaire integration
integration.scoring_hooks:
  trazabilidad_bonus: 0.06        # +6% if D3_Q3 complete
  productos_penalty: 0.08         # -8% if D3_Q1 incomplete
```

**Quick Tuning:**
- **High-capacity municipality:** `strict_mode: true`, ↑ sector weights by 0.05-0.10
- **Low-capacity municipality:** ↑ `max_warnings_per_eslabon` to 15, ↓ `APROBADO` to 0.80
- **More lenient intangibles:** ↑ `max_skip_ratio` to 0.30

### OperationalizationAuditor v3.0 (PPI Integration)

```yaml
# PPI-specific thresholds
min_cost_threshold_ppi: 50000000  # 50M COP (LEGAL REQUIREMENT - DO NOT CHANGE)
population_coherence_tolerance: 0.10  # ±10% variance allowed

# Unit cost benchmarks (Colombian standards)
unit_cost_benchmarks:
  aula_escolar:              {min: 120000000, max: 180000000}  # COP
  km_via_pavimentada:        {min: 280000000, max: 450000000}  # COP
  beneficiario_programa_social: {min: 800000, max: 2500000}   # COP/year
```

**Quick Tuning:**
- **Stricter population checks:** ↓ `population_coherence_tolerance` to 0.05
- **Cost benchmarks:** Adjust ranges based on regional price indices

---

## 🎨 Semantic & NLP Constants

### Linguistic Configuration (causalextractor.yaml)

```yaml
locale: "es-CO"                   # Colombian Spanish
tokenizer: "spacy_es_core_news_md"
sentence_splitter: "rule+ml"      # Hybrid rules + ML
number_normalizer: "spanish_cardinal_ordinal"
money_normalizer: "COP_parser"
case_insensitive: true
unicode_normalization: "NFC"      # Canonical composition

# Semantic embeddings
embeddings:
  model: "multilingual-e5-base"
  dim: 768
  hybrid:
    bm25: true                    # Enable BM25 + embeddings
    fusion: "RRF"                 # Reciprocal Rank Fusion

min_cosine_similarity: 0.45       # Activity↔Product alignment
```

**Quick Tuning:**
- **Stricter semantic alignment:** ↑ `min_cosine_similarity` to 0.55
- **More flexible:** ↓ to 0.40
- **Alternative embedding:** Try `paraphrase-multilingual-mpnet-base-v2`

---

## 📅 Temporal & Regulatory Thresholds

```yaml
# Temporal coherence (OperationalizationAuditor v3.0)
result_horizon_years:  {min: 2, max: 5}     # Medium-term
impact_horizon_years:  {min: 4}              # Long-term
plan_duration_years:   4                     # Colombian standard

# Regulatory (Colombian law)
sgp_sectors:                                 # Ley 715/2001
  - "SGP Educación"
  - "SGP Salud"
  - "SGP Agua Potable y Saneamiento Básico"
  - "SGP Propósito General"
  - "SGP Alimentación Escolar"
  - "SGP Primera Infancia"

min_cost_for_bpin: 50000000                  # Ley 152/1994 (50M COP)
```

**DO NOT CHANGE regulatory thresholds** - they are Colombian legal requirements.

---

## ⚡ Performance Settings

```yaml
# Performance (derek_beach_cdaf_config.yaml)
performance:
  enable_vectorized_ops: true     # Use NumPy for speed
  enable_async_processing: false  # Disabled for compatibility
  max_context_length: 1000        # Max chars for context
  cache_embeddings: true          # Cache for reuse
```

**Quick Tuning:**
- **Faster processing:** Ensure `enable_vectorized_ops: true`
- **Lower memory:** ↓ `max_context_length` to 500
- **Real-time mode:** Try `enable_async_processing: true` (experimental)

---

## 🔧 Common Tuning Scenarios

### Scenario 1: Too Many False Positives (Over-Classification)

**Problem:** System classifies too many fragments as causal links

**Solution:**
```yaml
# causalextractor.yaml
scoring:
  threshold:
    hard: 0.75  # ↑ from 0.70 (more conservative)
    soft: 0.60  # ↑ from 0.55

  connector_bonus: 0.10  # ↓ from 0.12 (less weight on connectors)
  context_bonus: 0.06    # ↓ from 0.08
```

### Scenario 2: Missing Causal Chains (Under-Recall)

**Problem:** System misses valid causal relationships

**Solution:**
```yaml
# causalextractor.yaml
scoring:
  threshold:
    hard: 0.65  # ↓ from 0.70 (less conservative)
    soft: 0.50  # ↓ from 0.55

default_context_window: 70  # ↑ from 50 (larger context)
max_context_window: 150     # ↑ from 120

scoring:
  gap_penalty: 0.03  # ↓ from 0.05 (tolerate incomplete chains)
```

### Scenario 3: High-Capacity Municipality (Strict Quality)

**Problem:** Need maximum rigor for well-resourced municipality

**Solution:**
```yaml
# OperationalizationAuditor
audit_config:
  strict_mode: true  # WARN → FAIL

thresholds:
  APROBADO: {min_score: 0.90}  # ↑ from 0.85

sector_weights:
  salud:          {sufficiency_alpha: 1.15, verification_beta: 1.25}
  infraestructura:{sufficiency_alpha: 1.30, verification_beta: 1.20}
```

### Scenario 4: Low-Capacity Municipality (Lenient)

**Problem:** Need supportive audit for under-resourced municipality

**Solution:**
```yaml
# OperationalizationAuditor
audit_config:
  strict_mode: false
  max_warnings_per_eslabon: 15  # ↑ from 10

thresholds:
  APROBADO: {min_score: 0.80}   # ↓ from 0.85
  ACEPTABLE: {min_score: 0.65}  # ↓ from 0.70

bayesian_coupling:
  high_conf: {multiplier: 1.05}  # ↓ from 1.10 (less strict)
```

### Scenario 5: Domain-Specific Tuning (e.g., Health Sector)

**Problem:** Need specialized calibration for health projects

**Solution:**
```yaml
# OperationalizationAuditor
sector_weights:
  salud: {sufficiency_alpha: 1.25, verification_beta: 1.30}  # ↑ strictness

# causalextractor.yaml - add health-specific patterns
sector_lexicons:
  salud:
    - "mortalidad"
    - "morbilidad"
    - "atención primaria"
    # ... expand list
```

---

## 📋 Parameter Validation Checklist

Before deploying configuration changes, verify:

- ✅ **Sum of priors = 1.0** (mechanism_type_priors)
- ✅ **Thresholds are ordered:** soft < hard (causalextractor)
- ✅ **Score ranges:** RECHAZADO.max < DEFICIENTE.min < ACEPTABLE.min < APROBADO.min
- ✅ **Weights are positive:** All weights ≥ 0
- ✅ **Legal thresholds unchanged:** min_cost_threshold_ppi = 50M, SGP sectors intact
- ✅ **Context windows:** default_context_window ≤ max_context_window
- ✅ **Bayesian coupling ordered:** very_low.multiplier < low_conf < medium_conf < high_conf

---

## 📖 File Locations

| File | Path |
|------|------|
| Bayesian config | `/home/user/SAAAAAA/config/derek_beach_cdaf_config.yaml` |
| Causal extraction | `/home/user/SAAAAAA/causalextractor.yaml` |
| Audit rules v1.5 | `/home/user/SAAAAAA/trazabilidad_cohrencia.yaml` |
| Audit rules v3.0 (PPI) | `/home/user/SAAAAAA/OperationalizationAuditor_v3.0_COMPLETO.yaml` |
| Full JSON catalog | `/home/user/SAAAAAA/PARAMETER_EXTRACTION_COMPLETE.json` |
| Analysis summary | `/home/user/SAAAAAA/PARAMETER_ANALYSIS_SUMMARY.md` |

---

## 🚨 Critical Parameters (DO NOT CHANGE Without Legal Review)

These parameters encode Colombian legal requirements:

- `min_cost_threshold_ppi: 50000000` (Ley 152/1994)
- `sgp_sectors: [...]` (Ley 715/2001)
- `plan_duration_years: 4` (Constitutional cycle)

**Changing these may create legal non-compliance.**

---

**Last Updated:** 2025-11-13
**Configuration Versions:** derek_beach v2.0 | causalextractor v2.1.0 | OperationalizationAuditor v1.5.0/v3.0.0
