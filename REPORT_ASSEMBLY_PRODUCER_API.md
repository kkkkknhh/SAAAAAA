# Report Assembly Producer - API Documentation

**Version:** 1.0.0  
**Producer Type:** Report Assembly / Aggregation  
**Public Methods:** 63  
**Status:** ✅ Production Ready

---

## Overview

The `ReportAssemblyProducer` exposes a clean public API for registry integration without leaking internal summarization logic. It provides methods for:

1. **MICRO level** - Question-by-question analysis
2. **MESO level** - Cluster aggregation  
3. **MACRO level** - Overall convergence
4. **Utilities** - Scoring, configuration, serialization, validation

---

## Quick Start

```python
from report_assembly import ReportAssemblyProducer

# Initialize producer
producer = ReportAssemblyProducer()

# Produce MICRO answer
answer_dict = producer.produce_micro_answer(
    question_spec=question,
    execution_results=results,
    plan_text=plan
)

# Get score
score = producer.get_micro_answer_score(answer)
print(f"Score: {score} ({producer.classify_score(score)})")
```

---

## API Reference

### MICRO Level Methods (14 methods)

#### Production
- `produce_micro_answer(question_spec, execution_results, plan_text)` → `Dict`
  - Produces MICRO-level answer for a single question
  - Returns serializable dictionary

#### Getters
- `get_micro_answer_score(answer)` → `float`
  - Extract quantitative score (0-3)
  
- `get_micro_answer_qualitative(answer)` → `str`
  - Extract qualitative classification (EXCELENTE/BUENO/ACEPTABLE/INSUFICIENTE)
  
- `get_micro_answer_evidence(answer)` → `List[str]`
  - Extract evidence excerpts
  
- `get_micro_answer_confidence(answer)` → `float`
  - Extract confidence score (0-1)
  
- `get_micro_answer_modules(answer)` → `List[str]`
  - Extract executed module names
  
- `get_micro_answer_execution_time(answer)` → `float`
  - Extract execution time in seconds
  
- `get_micro_answer_elements_found(answer)` → `Dict[str, bool]`
  - Extract detected elements

#### Counters
- `count_micro_evidence_excerpts(answer)` → `int`
  - Count evidence excerpts

#### Validators
- `is_micro_answer_excellent(answer)` → `bool`
  - Check if classified as EXCELENTE
  
- `is_micro_answer_passing(answer)` → `bool`
  - Check if meets minimum passing threshold (≥1.65)

---

### MESO Level Methods (13 methods)

#### Production
- `produce_meso_cluster(cluster_name, cluster_description, micro_answers, cluster_definition)` → `Dict`
  - Produces MESO-level cluster aggregation
  - Returns serializable dictionary

#### Getters
- `get_meso_cluster_score(cluster)` → `float`
  - Extract average score (0-100 percentage)
  
- `get_meso_cluster_policy_areas(cluster)` → `List[str]`
  - Extract policy areas (P1-P10)
  
- `get_meso_cluster_dimension_scores(cluster)` → `Dict[str, float]`
  - Extract dimension scores (D1-D6)
  
- `get_meso_cluster_strengths(cluster)` → `List[str]`
  - Extract identified strengths
  
- `get_meso_cluster_weaknesses(cluster)` → `List[str]`
  - Extract identified weaknesses
  
- `get_meso_cluster_recommendations(cluster)` → `List[str]`
  - Extract recommendations
  
- `get_meso_cluster_coverage(cluster)` → `float`
  - Extract question coverage percentage
  
- `get_meso_cluster_question_counts(cluster)` → `Tuple[int, int]`
  - Extract (total_questions, answered_questions)

#### Counters
- `count_meso_strengths(cluster)` → `int`
  - Count strengths
  
- `count_meso_weaknesses(cluster)` → `int`
  - Count weaknesses

#### Validators
- `is_meso_cluster_excellent(cluster)` → `bool`
  - Check if score ≥85 (EXCELENTE)
  
- `is_meso_cluster_passing(cluster)` → `bool`
  - Check if score ≥55 (SATISFACTORIO)

---

### MACRO Level Methods (20 methods)

#### Production
- `produce_macro_convergence(all_micro_answers, all_meso_clusters, plan_metadata)` → `Dict`
  - Produces MACRO-level convergence analysis
  - Returns serializable dictionary

#### Getters
- `get_macro_overall_score(convergence)` → `float`
  - Extract overall score (0-100)
  
- `get_macro_dimension_convergence(convergence)` → `Dict[str, float]`
  - Extract dimension convergence (D1-D6)
  
- `get_macro_policy_convergence(convergence)` → `Dict[str, float]`
  - Extract policy area convergence (P1-P10)
  
- `get_macro_gap_analysis(convergence)` → `Dict[str, Any]`
  - Extract gap analysis
  
- `get_macro_agenda_alignment(convergence)` → `float`
  - Extract agenda alignment (0-1)
  
- `get_macro_critical_gaps(convergence)` → `List[str]`
  - Extract critical gaps
  
- `get_macro_strategic_recommendations(convergence)` → `List[str]`
  - Extract strategic recommendations
  
- `get_macro_classification(convergence)` → `str`
  - Extract plan classification
  
- `get_macro_evidence_synthesis(convergence)` → `Dict[str, Any]`
  - Extract evidence synthesis
  
- `get_macro_implementation_roadmap(convergence)` → `List[Dict[str, Any]]`
  - Extract implementation roadmap
  
- `get_macro_score_distribution(convergence)` → `Dict[str, int]`
  - Extract score distribution
  
- `get_macro_confidence_metrics(convergence)` → `Dict[str, float]`
  - Extract confidence metrics

#### Counters
- `count_macro_critical_gaps(convergence)` → `int`
  - Count critical gaps
  
- `count_macro_strategic_recommendations(convergence)` → `int`
  - Count recommendations

#### Validators
- `is_macro_excellent(convergence)` → `bool`
  - Check if overall score ≥85
  
- `is_macro_passing(convergence)` → `bool`
  - Check if overall score ≥55

---

### Scoring Utilities (6 methods)

- `convert_score_to_percentage(score: float)` → `float`
  - Convert 0-3 score to 0-100 percentage
  
- `convert_percentage_to_score(percentage: float)` → `float`
  - Convert 0-100 percentage to 0-3 score
  
- `classify_score(score: float)` → `str`
  - Classify 0-3 score to qualitative level
  
- `classify_percentage(percentage: float)` → `str`
  - Classify 0-100 percentage to qualitative level
  
- `get_rubric_threshold(level: str)` → `Tuple[float, float]`
  - Get percentage threshold range for rubric level
  
- `get_question_rubric_threshold(level: str)` → `Tuple[float, float]`
  - Get 0-3 score threshold range for question rubric

---

### Configuration Methods (6 methods)

- `get_dimension_description(dimension: str)` → `str`
  - Get description for dimension (D1-D6)
  
- `list_dimensions()` → `List[str]`
  - List all dimensions
  
- `list_rubric_levels()` → `List[str]`
  - List all rubric levels
  
- `get_causal_threshold(dimension: str)` → `float`
  - Get causal coherence threshold
  
- `get_cluster_weight(cluster_id: str)` → `Optional[float]`
  - Get cluster weight in macro aggregation
  
- `get_cluster_policy_weights(cluster_id: str)` → `Optional[Dict[str, float]]`
  - Get policy area weights for cluster

---

### Serialization Methods (7 methods)

- `export_complete_report(micro_answers, meso_clusters, macro_convergence, output_path)`
  - Export complete report to JSON file
  
- `serialize_micro_answer(answer)` → `Dict`
  - Serialize MICRO answer to dictionary
  
- `serialize_meso_cluster(cluster)` → `Dict`
  - Serialize MESO cluster to dictionary
  
- `serialize_macro_convergence(convergence)` → `Dict`
  - Serialize MACRO convergence to dictionary
  
- `deserialize_micro_answer(data)` → `MicroLevelAnswer`
  - Deserialize dictionary to MICRO answer
  
- `deserialize_meso_cluster(data)` → `MesoLevelCluster`
  - Deserialize dictionary to MESO cluster
  
- `deserialize_macro_convergence(data)` → `MacroLevelConvergence`
  - Deserialize dictionary to MACRO convergence

---

### Schema Validation Methods (3 methods)

- `validate_micro_answer(answer_data: Dict)` → `bool`
  - Validate MICRO answer against JSON schema
  
- `validate_meso_cluster(cluster_data: Dict)` → `bool`
  - Validate MESO cluster against JSON schema
  
- `validate_macro_convergence(convergence_data: Dict)` → `bool`
  - Validate MACRO convergence against JSON schema

---

## Rubric Levels

### Overall/Dimension Scoring (0-100 percentage)

| Level | Min | Max | Description |
|-------|-----|-----|-------------|
| EXCELENTE | 85 | 100 | Outstanding performance |
| BUENO | 70 | 84 | Good performance |
| SATISFACTORIO | 55 | 69 | Satisfactory performance |
| INSUFICIENTE | 40 | 54 | Insufficient performance |
| DEFICIENTE | 0 | 39 | Poor performance |

### Question-Level Scoring (0-3 scale)

| Level | Min | Max | Percentage |
|-------|-----|-----|------------|
| EXCELENTE | 2.55 | 3.00 | 85-100% |
| BUENO | 2.10 | 2.54 | 70-84% |
| ACEPTABLE | 1.65 | 2.09 | 55-69% |
| INSUFICIENTE | 0.00 | 1.64 | 0-54% |

---

## Dimensions

- **D1**: Diagnóstico y Recursos - Líneas base, magnitud del problema, recursos
- **D2**: Diseño de Intervención - Actividades, mecanismos causales, secuencias
- **D3**: Productos y Outputs - Entregables, verificación, presupuesto
- **D4**: Resultados y Outcomes - Indicadores de resultado, causalidad
- **D5**: Impactos y Efectos de Largo Plazo - Transformación estructural
- **D6**: Teoría de Cambio y Coherencia Causal - Coherencia causal global

---

## JSON Schemas

Schemas available in `schemas/report_assembly/`:

1. `micro_answer.schema.json` - MICRO answer structure
2. `meso_cluster.schema.json` - MESO cluster structure
3. `macro_convergence.schema.json` - MACRO convergence structure
4. `producer_api.schema.json` - Producer API contract

---

## QMCM Integration

Quality Method Call Monitoring (QMCM) tracks method invocations without recording data content:

```python
from qmcm_hooks import get_global_recorder

# Get recorder
recorder = get_global_recorder()

# After method calls, get statistics
stats = recorder.get_statistics()
print(f"Total calls: {stats['total_calls']}")
print(f"Success rate: {stats['success_rate']}")

# Save recording
recorder.save_recording()
```

**Important:** QMCM records only metadata (method names, types, timing), NOT actual data content.

---

## Testing

Run smoke tests:

```bash
python3 -m unittest tests.test_report_assembly_producer -v
python3 -m unittest tests.test_qmcm_hooks -v
```

Expected: 21 tests passing

---

## Security Guarantees

✅ **No Summarization Leakage**: Public API exposes only registry methods, not internal summarization logic  
✅ **No Data Leakage**: QMCM records only metadata, not data content  
✅ **Schema Validation**: All outputs validate against JSON schemas  
✅ **Type Safety**: Strong typing with dataclasses

---

## Registry Integration

The producer is registered in `COMPLETE_METHOD_CLASS_MAP.json`:

```json
{
  "files": {
    "report_assembly.py": {
      "total_classes": 3,
      "total_methods": 111,
      "classes": {
        "ReportAssembler": {
          "method_count": 48
        },
        "ReportAssemblyProducer": {
          "method_count": 63
        }
      }
    }
  }
}
```

---

## Version History

- **1.0.0** (2025-10-28): Initial release with 63 public methods, schemas, and QMCM hooks
