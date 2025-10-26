# JSON Schemas for Producer Artifacts
**Updated:** 2025-10-26  
**Validation Level:** Draft-07 (strict)  
**Schema Files Present:** 21

This directory hosts the canonical JSON Schemas consumed by the policy analysis
pipeline. Every schema listed here is curated to support deterministic
validation inside the `choreographer` execution engine and the `report_assembly`
aggregator.

For a producer-by-producer breakdown that `choreographer` can use to resolve
artifact contracts, see
[`CHOREOGRAPHER_SCHEMA_INVENTORY.md`](CHOREOGRAPHER_SCHEMA_INVENTORY.md).

---

## 📁 Directory Structure

```
schemas/
├── CHOREOGRAPHER_SCHEMA_INVENTORY.md   # Exhaustive mapping per producer
├── README.md                           # This file
├── contradiction_deteccion/
│   ├── contradiction_evidence.schema.json
│   └── policy_statement.schema.json
├── dereck_beach/
│   └── meta_node.schema.json
├── embedding_policy/
│   └── semantic_chunk.schema.json
├── execution_mapping.schema.json
├── execution_step.schema.json
├── financiero_viabilidad/
│   ├── causal_dag.schema.json
│   ├── causal_edge.schema.json
│   ├── causal_effect.schema.json
│   ├── causal_node.schema.json
│   └── financial_indicator.schema.json
├── question_segmentation.schema.json
├── questionnaire.schema.json
├── report_assembly/
│   ├── macro_convergence.schema.json
│   ├── meso_cluster.schema.json
│   └── micro_answer.schema.json
├── rubric.schema.json
├── rubric_scoring.schema.json
└── teoria_cambio/
    ├── advanced_graph_node.schema.json
    ├── monte_carlo_result.schema.json
    └── validacion_resultado.schema.json
```

> **Note:** Schema packages for `Analyzer_one`, `policy_processor`, and
> `semantic_chunking_policy` are not yet present. Their expected structures are
> documented in the choreographer inventory so that schema authoring can follow
> the correct contracts.

---

## ✅ Completed Producer Coverage

| Producer Module                     | Primary Schemas                                                                           | Coverage Notes |
|------------------------------------|-------------------------------------------------------------------------------------------|----------------|
| `financiero_viabilidad_tablas.py`  | `causal_node`, `causal_edge`, `causal_dag`, `causal_effect`, `financial_indicator`        | Complete       |
| `contradiction_deteccion.py`       | `policy_statement`, `contradiction_evidence`                                              | Complete       |
| `dereck_beach.py`                  | `meta_node`                                                                               | Covers MetaNode; audit schema pending |
| `embedding_policy.py`              | `semantic_chunk`                                                                          | Bayesian evaluation schema pending |
| `teoria_cambio.py`                 | `advanced_graph_node`, `validacion_resultado`, `monte_carlo_result`                       | Complete       |
| `report_assembly.py`               | `micro_answer`, `meso_cluster`, `macro_convergence`                                        | Complete       |

---

## ⏳ Pending Producer Schemas

| Producer Module                 | Expected Artifact(s)                               | Status |
|--------------------------------|----------------------------------------------------|--------|
| `Analyzer_one.py`              | `semantic_cube`, `performance_analysis`            | Not yet authored |
| `policy_processor.py`          | `evidence_bundle`, `processor_config` (if exported) | Not yet authored |
| `semantic_chunking_policy.py`  | `semantic_config`, `bayesian_dimension_result`      | Not yet authored |
| `dereck_beach.py`              | `audit_result`                                     | Planned |
| `embedding_policy.py`          | `bayesian_evaluation`                              | Planned |

---

## 🎯 Validation Usage

Schemas follow JSON Schema Draft-07. Validation pipelines should enable
`additionalProperties: false` enforcement and strict type checking. Example in
Python:

```python
import json
import jsonschema
from pathlib import Path

schema = json.loads(Path("schemas/financiero_viabilidad/causal_node.schema.json").read_text())
instance = {
    "name": "Infraestructura y adecuación de tierras",
    "node_type": "pilar",
    "embedding": None,
    "associated_budget": "500000000.00",
    "temporal_lag": 2,
    "evidence_strength": 0.85
}

jsonschema.validate(instance=instance, schema=schema)
```

For CLI validation use `ajv` with `--strict` mode to mirror production checks.
