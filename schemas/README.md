# JSON SCHEMAS FOR PRODUCER ARTIFACTS
## Commit 3: Schema Definitions

**Generated:** 2025-10-22  
**Total Schemas:** 20 JSON Schema files
**Validation Level:** STRICT  
**Format:** JSON Schema Draft-07

---

## 📁 SCHEMA DIRECTORY STRUCTURE

```
schemas/
├── financiero_viabilidad/          (5 schemas - COMPLETE ✅)
│   ├── causal_node.schema.json
│   ├── causal_edge.schema.json
│   ├── causal_dag.schema.json
│   ├── causal_effect.schema.json
│   └── financial_indicator.schema.json
├── analyzer_one/                   (2 schemas - IN PROGRESS)
│   ├── semantic_cube.schema.json
│   └── performance_analysis.schema.json
├── contradiction_deteccion/        (2 schemas - IN PROGRESS)
│   ├── contradiction_evidence.schema.json
│   └── coherence_metrics.schema.json
├── embedding_policy/               (2 schemas - IN PROGRESS)
│   ├── semantic_chunk.schema.json
│   └── bayesian_evaluation.schema.json
├── teoria_cambio/                  (3 schemas - IN PROGRESS)
│   ├── validacion_resultado.schema.json
│   └── monte_carlo_result.schema.json
├── dereck_beach/                   (1/2 schemas - IN PROGRESS)
│   ├── meta_node.schema.json       ✅
│   └── audit_result.schema.json
├── policy_processor/               (1 schema - IN PROGRESS)
│   └── evidence_bundle.schema.json
└── report_assembly/                (3 schemas - 1/3 COMPLETE)
    ├── micro_answer.schema.json    ✅
    ├── meso_cluster.schema.json
    └── macro_convergence.schema.json
```

---

## ✅ COMPLETED SCHEMAS (7/19)

### **financiero_viabilidad/** (5 schemas)

1. **causal_node.schema.json** (57 lines)
   - Nodo en el grafo causal
   - Properties: name, node_type (pilar/outcome/mediator/confounder), embedding (768 dims), associated_budget, temporal_lag, evidence_strength
   - Strict validation with type enums

2. **causal_edge.schema.json** (59 lines)
   - Arista causal entre nodos
   - Properties: source, target, edge_type (direct/mediated/confounded), effect_size_posterior, mechanism, evidence_quotes, probability
   - Bayesian posterior as 3-tuple [mean, p2.5, p97.5]

3. **causal_dag.schema.json** (53 lines)
   - Grafo Acíclico Dirigido completo
   - Properties: nodes (dict), edges (array), adjacency_matrix, graph_metadata
   - Graph metadata replaces nx.DiGraph object (is_acyclic, node_count, topological_order)

4. **causal_effect.schema.json** (68 lines)
   - Efecto causal estimado con inferencia bayesiana
   - Properties: treatment, outcome, effect_type (ATE/ATT/direct/indirect/total), point_estimate, posterior_mean, credible_interval_95, probability_positive, probability_significant, mediating_paths, confounders_adjusted
   - Complete Bayesian inference metadata

5. **financial_indicator.schema.json** (75 lines)
   - Indicador financiero extraído
   - Properties: source_text, amount (string decimal), currency (COP/USD/EUR), fiscal_year, funding_source, budget_category, execution_percentage, confidence_interval, risk_level
   - Uses string for Decimal precision

### **report_assembly/** (1 schema)

6. **micro_answer.schema.json** (98 lines)
   - Respuesta MICRO nivel con trazabilidad completa
   - Properties: question_id (P#-D#-Q# pattern), qualitative_note (EXCELENTE/BUENO/ACEPTABLE/INSUFICIENTE), quantitative_score (0.0-3.0), evidence, explanation (150-3000 chars), confidence, scoring_modality (TYPE_A-F), elements_found, modules_executed, execution_time, execution_chain
   - Doctoral-level explanation with 150-300 word requirement

7. **dereck_beach/meta_node.schema.json** (170 lines)
   - Nodo meta del CDAF con entidades, dinámica y pruebas probatorias
   - Properties: id (MP-001), type (programa/producto/resultado/impacto), baseline/target (numérico o marcadores ND/N/A), entity_activity (entity, activity, verb_lemma, confidence 0-1)
   - Control estricto de riesgos contextuales, banderas de auditoría y score de confianza

---

## 🔄 REMAINING SCHEMAS (13/20)

### Priority 1: Report Assembly Outputs (2 schemas)
- **meso_cluster.schema.json** - Cluster aggregation (MesoLevelCluster dataclass)
- **macro_convergence.schema.json** - Overall convergence (MacroLevelConvergence dataclass)

### Priority 2: Core Producers (10 schemas)
- **analyzer_one/semantic_cube.schema.json** - Semantic analysis cube
- **analyzer_one/performance_analysis.schema.json** - Performance metrics
- **contradiction_deteccion/contradiction_evidence.schema.json** - Contradiction evidence
- **contradiction_deteccion/coherence_metrics.schema.json** - Coherence metrics
- **embedding_policy/semantic_chunk.schema.json** - Semantic chunk with PDQ
- **embedding_policy/bayesian_evaluation.schema.json** - Bayesian numerical evaluation
- **teoria_cambio/validacion_resultado.schema.json** - Validation result
- **teoria_cambio/monte_carlo_result.schema.json** - Monte Carlo result
- **dereck_beach/audit_result.schema.json** - Operationalization audit result
- **policy_processor/evidence_bundle.schema.json** - Evidence bundle with patterns

---

## 📋 SCHEMA DESIGN PRINCIPLES

### 1. **Strict Validation**
- All schemas use `"additionalProperties": false` to prevent unexpected fields
- Required fields explicitly defined
- Type constraints with enums where applicable

### 2. **Canonical References**
- All schemas use `$ref` for nested object validation
- Base URI: `https://saaaaaa.policy.analysis/schemas/`
- JSON Schema Draft-07 compliance

### 3. **Decimal Precision**
- Financial amounts use string pattern `^\\d+(\\.\\d{1,2})?$` to preserve Decimal precision
- Avoids floating-point rounding errors

### 4. **Array Constraints**
- Embeddings: exactly 768 items (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- Confidence intervals: exactly 2 items [lower, upper]
- Bayesian posteriors: exactly 3 items [mean, p2.5, p97.5]

### 5. **Enum Validation**
- `node_type`: pilar | outcome | mediator | confounder
- `edge_type`: direct | mediated | confounded
- `effect_type`: ATE | ATT | direct | indirect | total
- `scoring_modality`: TYPE_A | TYPE_B | TYPE_C | TYPE_D | TYPE_E | TYPE_F
- `qualitative_note`: EXCELENTE | BUENO | ACEPTABLE | INSUFICIENTE

### 6. **Pattern Matching**
- `question_id`: `^P(\\d|10)-D[1-6]-Q[1-5]$` (e.g., P1-D1-Q1, P10-D6-Q5)
- Enforces canonical P#-D#-Q# notation

### 7. **Length Constraints**
- Evidence excerpts: maxLength 1000-2000 chars
- Explanations: minLength 150, maxLength 3000 (doctoral standard)
- Mechanism descriptions: maxLength 2000

---

## 🎯 VALIDATION USAGE

### Python Validation Example
```python
import json
import jsonschema
from pathlib import Path

# Load schema
schema_path = Path("schemas/financiero_viabilidad/causal_node.schema.json")
with open(schema_path) as f:
    schema = json.load(f)

# Load data
data = {
    "name": "Infraestructura y adecuación de tierras",
    "node_type": "pilar",
    "embedding": None,
    "associated_budget": "500000000.00",
    "temporal_lag": 2,
    "evidence_strength": 0.85
}

# Validate
try:
    jsonschema.validate(instance=data, schema=schema)
    print("✅ Validation passed")
except jsonschema.ValidationError as e:
    print(f"❌ Validation failed: {e.message}")
```

### CLI Validation (using ajv)
```bash
# Install ajv-cli
npm install -g ajv-cli

# Validate instance
ajv validate -s schemas/financiero_viabilidad/causal_node.schema.json \
             -d producer_output.json \
             --strict
```

---

## 📊 SCHEMA STATISTICS

| Producer | Schemas | Lines | Completed |
|----------|---------|-------|-----------|
| financiero_viabilidad | 5 | 312 | ✅ 100% |
| analyzer_one | 2 | TBD | ⏳ 0% |
| contradiction_deteccion | 2 | TBD | ⏳ 0% |
| embedding_policy | 2 | TBD | ⏳ 0% |
| teoria_cambio | 2 | TBD | ⏳ 0% |
| dereck_beach | 2 | 170 | ⏳ 50% |
| policy_processor | 1 | TBD | ⏳ 0% |
| report_assembly | 3 | 98 | ⏳ 33% |
| **TOTAL** | **19** | **580+** | **37%** |

---

## 🔄 NEXT STEPS

1. **Complete report_assembly schemas** (2 remaining)
   - meso_cluster.schema.json
   - macro_convergence.schema.json

2. **Generate remaining producer schemas** (10 remaining)
   - Read dataclass definitions from source files
   - Extract field names, types, constraints
   - Generate JSON Schema with strict validation

3. **Cross-reference validation**
   - Ensure all `$ref` links are valid
   - Test with sample data from each producer

4. **Integration with Commit 4**
   - Assembler will use these schemas for validation
   - Strict validation before aggregation

---

**Status:** 7/19 schemas complete (37%)
**Next:** Completar report_assembly + productores pendientes clave
**Target:** 100% schema coverage for Commit 3 completion
