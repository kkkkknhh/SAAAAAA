# NIVEL3 - Reporte de Aptitud de Ejecución de Métodos

**Fecha de Generación:** 2025-10-29 03:34:44
**Sistema:** SAAAAAA - Strategic Policy Analysis System
**Total de Métodos Analizados:** 593
**Score Promedio de Aptitud:** 98.46/100

---

## 📊 Resumen Ejecutivo

Este documento presenta un análisis exhaustivo de los **593 métodos** que componen el
sistema de análisis de políticas públicas SAAAAAA. Cada método ha sido evaluado con
**máximo rigor** para determinar su aptitud de ejecución, prioridad, complejidad y
requisitos necesarios para garantizar su funcionamiento correcto.

### Estadísticas Generales

- **Total de Archivos Analizados:** 8
- **Total de Métodos:** 593
- **Score Promedio de Aptitud:** 98.46/100

### Distribución por Complejidad

| Complejidad | Cantidad | Porcentaje |
|-------------|----------|------------|
| LOW         |      280 |      47.2% |
| MEDIUM      |      282 |      47.6% |
| HIGH        |       31 |       5.2% |

### Distribución por Prioridad

| Prioridad | Cantidad | Porcentaje |
|-----------|----------|------------|
| CRITICAL  |       44 |       7.4% |
| HIGH      |       24 |       4.0% |
| MEDIUM    |      236 |      39.8% |
| LOW       |      289 |      48.7% |

---

## 🎯 Métodos de Prioridad CRÍTICA

Los siguientes métodos son **CRÍTICOS** para el funcionamiento del sistema.
**Deben ejecutarse obligatoriamente** y cualquier fallo en estos métodos
compromete la integridad completa del sistema.

### 1. `BayesianNumericalAnalyzer.__init__`

**Archivo:** `embedding_policy.py`  
**Línea:** 425  
**Score de Aptitud:** 100.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, prior_strength)
```

**Documentación:**  
Initialize Bayesian analyzer.

Args:
    prior_strength: Prior belief strength (1.0 = weak, 10.0 = strong)...

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 2. `PolicyCrossEncoderReranker.__init__`

**Archivo:** `embedding_policy.py`  
**Línea:** 655  
**Score de Aptitud:** 100.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, model_name, max_length, retry_handler)
```

**Documentación:**  
Initialize cross-encoder reranker.

Args:
    model_name: HuggingFace model name (multilingual preferred)
    max_length: Maximum sequence length for cross-encoder
    retry_handler: Optional RetryHan...

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 3. `EmbeddingPolicyProducer.__init__`

**Archivo:** `embedding_policy.py`  
**Línea:** 1398  
**Score de Aptitud:** 100.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, model_tier, retry_handler)
```

**Documentación:**  
Initialize producer with optional configuration...

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 4. `TeoriaCambio.__init__`

**Archivo:** `teoria_cambio.py`  
**Línea:** 296  
**Score de Aptitud:** 100.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Documentación:**  
Inicializa el motor con un sistema de cache optimizado....

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 5. `DerekBeachProducer.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 5558  
**Score de Aptitud:** 100.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Documentación:**  
Initialize producer...

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 6. `IndustrialPolicyProcessor.process`

**Archivo:** `policy_processor.py`  
**Línea:** 657  
**Score de Aptitud:** 100.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
process(self, raw_text)
```

**Documentación:**  
Execute comprehensive policy plan analysis.

Args:
    raw_text: Sanitized policy document text

Returns:
    Structured analysis results with evidence bundles and confidence scores...

**Prerequisitos:**
- Instance of IndustrialPolicyProcessor must be initialized

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 7. `ReportAssembler.__init__`

**Archivo:** `report_assembly.py`  
**Línea:** 141  
**Score de Aptitud:** 100.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, dimension_descriptions, cluster_weights, cluster_policy_weights, causal_thresholds)
```

**Documentación:**  
Initialize report assembler with rubric definitions...

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 8. `ReportAssemblyProducer.__init__`

**Archivo:** `report_assembly.py`  
**Línea:** 2495  
**Score de Aptitud:** 100.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, dimension_descriptions, cluster_weights, cluster_policy_weights, causal_thresholds)
```

**Documentación:**  
Initialize producer with optional configuration...

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 9. `PDETMunicipalPlanAnalyzer.__init__`

**Archivo:** `financiero_viabilidad_tablas.py`  
**Línea:** 281  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, use_gpu, language, confidence_threshold)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 10. `MunicipalOntology.__init__`

**Archivo:** `Analyzer_one.py`  
**Línea:** 95  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 11. `SemanticAnalyzer.__init__`

**Archivo:** `Analyzer_one.py`  
**Línea:** 154  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, ontology)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 12. `PerformanceAnalyzer.__init__`

**Archivo:** `Analyzer_one.py`  
**Línea:** 384  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, ontology)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 13. `TextMiningEngine.__init__`

**Archivo:** `Analyzer_one.py`  
**Línea:** 560  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, ontology)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 14. `MunicipalAnalyzer.__init__`

**Archivo:** `Analyzer_one.py`  
**Línea:** 742  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 15. `CanonicalQuestionSegmenter.__init__`

**Archivo:** `Analyzer_one.py`  
**Línea:** 1016  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, questionnaire_path, rubric_path, segmentation_method)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 16. `ConfigurationManager.__init__`

**Archivo:** `Analyzer_one.py`  
**Línea:** 1706  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config_path)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 17. `BatchProcessor.__init__`

**Archivo:** `Analyzer_one.py`  
**Línea:** 1757  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, analyzer)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 18. `BayesianConfidenceCalculator.__init__`

**Archivo:** `contradiction_deteccion.py`  
**Línea:** 107  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 19. `TemporalLogicVerifier.__init__`

**Archivo:** `contradiction_deteccion.py`  
**Línea:** 145  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 20. `PolicyContradictionDetector.__init__`

**Archivo:** `contradiction_deteccion.py`  
**Línea:** 287  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, model_name, spacy_model, device)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 21. `AdvancedSemanticChunker.__init__`

**Archivo:** `embedding_policy.py`  
**Línea:** 154  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 22. `PolicyAnalysisEmbedder.__init__`

**Archivo:** `embedding_policy.py`  
**Línea:** 777  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, retry_handler)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 23. `AdvancedDAGValidator.__init__`

**Archivo:** `teoria_cambio.py`  
**Línea:** 460  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, graph_type)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 24. `IndustrialGradeValidator.__init__`

**Archivo:** `teoria_cambio.py`  
**Línea:** 830  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 25. `CDAFException.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 197  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, message, details, stage, recoverable)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 26. `ConfigLoader.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 447  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config_path)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 27. `PDFProcessor.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 852  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, retry_handler)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 28. `CausalExtractor.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 966  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, nlp_model)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 29. `MechanismPartExtractor.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 1584  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, nlp_model)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 30. `FinancialAuditor.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 1637  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 31. `OperationalizationAuditor.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 1912  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 32. `BayesianMechanismInference.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 2528  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, nlp_model)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 33. `CausalInferenceSetup.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 3022  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 34. `ReportingEngine.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 3201  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, output_dir)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 35. `CDAFFramework.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 3505  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config_path, output_dir, log_level)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 36. `AdaptivePriorCalculator.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 4018  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, calibration_params)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 37. `HierarchicalGenerativeModel.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 4431  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, mechanism_priors)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 38. `BayesianCounterfactualAuditor.__init__`

**Archivo:** `dereck_beach.py`  
**Línea:** 4958  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 39. `BayesianEvidenceScorer.__init__`

**Archivo:** `policy_processor.py`  
**Línea:** 394  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, prior_confidence, entropy_weight)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 40. `PolicyTextProcessor.__init__`

**Archivo:** `policy_processor.py`  
**Línea:** 466  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 41. `IndustrialPolicyProcessor.__init__`

**Archivo:** `policy_processor.py`  
**Línea:** 559  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, questionnaire_path)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 42. `AdvancedTextSanitizer.__init__`

**Archivo:** `policy_processor.py`  
**Línea:** 1156  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 43. `PolicyAnalysisPipeline.__init__`

**Archivo:** `policy_processor.py`  
**Línea:** 1310  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self, config, questionnaire_path)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

### 44. `BayesianConfidenceCalculator.__init__`

**Archivo:** `policy_processor.py`  
**Línea:** 54  
**Score de Aptitud:** 90.0/100  
**Complejidad:** MEDIUM  

**Firma del Método:**
```python
__init__(self)
```

**Dependencias:**
- Instance state

**Requisitos de Ejecución:**
- Computacional: MEDIUM
- Memoria: MEDIUM
- I/O Bound: No
- Stateful: Sí

---

## 🔥 Métodos de Alta Prioridad (HIGH)

Métodos esenciales para el análisis completo. Su ejecución es altamente
recomendada para garantizar la calidad y completitud del análisis.

### 1. `TemporalLogicVerifier._check_deadline_constraints`

| Atributo | Valor |
|----------|-------|
| Archivo | `contradiction_deteccion.py` |
| Aptitud | 100.0/100 |
| Complejidad | LOW |
| Comp. Requerido | MEDIUM |

---

### 2. `IndustrialGradeValidator.validate_engine_readiness`

| Atributo | Valor |
|----------|-------|
| Archivo | `teoria_cambio.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 3. `IndustrialGradeValidator.validate_connection_matrix`

| Atributo | Valor |
|----------|-------|
| Archivo | `teoria_cambio.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 4. `MechanismTypeConfig.check_sum_to_one`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 5. `ConfigLoader._validate_config`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | LOW |
| Comp. Requerido | MEDIUM |

---

### 6. `ConfigLoader.check_uncertainty_reduction_criterion`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 7. `CausalExtractor._check_structural_violation`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | LOW |
| Comp. Requerido | MEDIUM |

---

### 8. `FinancialAuditor._perform_counterfactual_budget_check`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | LOW |
| Comp. Requerido | MEDIUM |

---

### 9. `CDAFFramework._validate_dnp_compliance`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | LOW |
| Comp. Requerido | MEDIUM |

---

### 10. `AdaptivePriorCalculator.validate_quality_criteria`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 11. `HierarchicalGenerativeModel.posterior_predictive_check`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 12. `BayesianCounterfactualAuditor.refutation_and_sanity_checks`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 13. `DerekBeachProducer.posterior_predictive_check`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 14. `DerekBeachProducer.all_checks_passed`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 15. `ProcessorConfig.validate`

| Atributo | Valor |
|----------|-------|
| Archivo | `policy_processor.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 16. `ReportAssembler._validate_macro_gating`

| Atributo | Valor |
|----------|-------|
| Archivo | `report_assembly.py` |
| Aptitud | 100.0/100 |
| Complejidad | LOW |
| Comp. Requerido | MEDIUM |

---

### 17. `ReportAssembler.validate_micro_answer_schema`

| Atributo | Valor |
|----------|-------|
| Archivo | `report_assembly.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 18. `ReportAssembler.validate_meso_cluster_schema`

| Atributo | Valor |
|----------|-------|
| Archivo | `report_assembly.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 19. `ReportAssembler.validate_macro_convergence_schema`

| Atributo | Valor |
|----------|-------|
| Archivo | `report_assembly.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 20. `ReportAssemblyProducer.validate_micro_answer`

| Atributo | Valor |
|----------|-------|
| Archivo | `report_assembly.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 21. `ReportAssemblyProducer.validate_meso_cluster`

| Atributo | Valor |
|----------|-------|
| Archivo | `report_assembly.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 22. `ReportAssemblyProducer.validate_macro_convergence`

| Atributo | Valor |
|----------|-------|
| Archivo | `report_assembly.py` |
| Aptitud | 100.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

---

### 23. `DerekBeachProducer.refutation_checks`

| Atributo | Valor |
|----------|-------|
| Archivo | `dereck_beach.py` |
| Aptitud | 95.0/100 |
| Complejidad | MEDIUM |
| Comp. Requerido | MEDIUM |

**Consideraciones:** Multiple parameters increase error surface

---

### 24. `IndustrialGradeValidator.validate_causal_categories`

| Atributo | Valor |
|----------|-------|
| Archivo | `teoria_cambio.py` |
| Aptitud | 85.0/100 |
| Complejidad | HIGH |
| Comp. Requerido | HIGH |

**Consideraciones:** High computational complexity

---

## ⚡ Métodos de Alta Complejidad

Estos métodos requieren **atención especial** debido a su complejidad algorítmica,
demanda computacional o dependencias sofisticadas.

| # | Método | Clase | Aptitud | Prioridad | Requisitos |
|---|--------|-------|---------|-----------|------------|
| 1 | `_score_causal_coherence` | `PDETMunicipalPlanAna` | 90/100 | LOW | HIGH |
| 2 | `_validar_orden_causal` | `TeoriaCambio` | 90/100 | LOW | HIGH |
| 3 | `_calculate_bayesian_poste` | `AdvancedDAGValidator` | 90/100 | LOW | HIGH |
| 4 | `_extract_causal_links` | `CausalExtractor` | 90/100 | LOW | HIGH |
| 5 | `_build_normative_dag` | `OperationalizationAu` | 90/100 | LOW | HIGH |
| 6 | `_audit_causal_implication` | `OperationalizationAu` | 90/100 | LOW | HIGH |
| 7 | `_analyze_causal_dimension` | `IndustrialPolicyProc` | 90/100 | LOW | HIGH |
| 8 | `_apply_causal_correction` | `ReportAssembler` | 90/100 | LOW | HIGH |
| 9 | `_extract_causal_signals` | `ReportAssembler` | 90/100 | LOW | HIGH |
| 10 | `_extract_causal_flags` | `ReportAssembler` | 90/100 | LOW | HIGH |
| 11 | `export_causal_network` | `PDETMunicipalPlanAna` | 85/100 | MEDIUM | HIGH |
| 12 | `construir_grafo_causal` | `TeoriaCambio` | 85/100 | MEDIUM | HIGH |
| 13 | `validate_causal_categorie` | `IndustrialGradeValid` | 85/100 | HIGH | HIGH |
| 14 | `get_bayesian_threshold` | `ConfigLoader` | 85/100 | MEDIUM | HIGH |
| 15 | `extract_causal_hierarchy` | `CausalExtractor` | 85/100 | MEDIUM | HIGH |
| 16 | `bayesian_counterfactual_a` | `OperationalizationAu` | 85/100 | MEDIUM | HIGH |
| 17 | `generate_causal_diagram` | `ReportingEngine` | 85/100 | MEDIUM | HIGH |
| 18 | `generate_causal_model_jso` | `ReportingEngine` | 85/100 | MEDIUM | HIGH |
| 19 | `is_inference_uncertain` | `DerekBeachProducer` | 85/100 | MEDIUM | HIGH |
| 20 | `get_causal_effect` | `DerekBeachProducer` | 85/100 | MEDIUM | HIGH |
| 21 | `get_causal_threshold` | `ReportAssemblyProduc` | 85/100 | MEDIUM | HIGH |
| 22 | `_bayesian_risk_inference` | `PDETMunicipalPlanAna` | 80/100 | LOW | HIGH |
| 23 | `_identify_causal_nodes` | `PDETMunicipalPlanAna` | 80/100 | LOW | HIGH |
| 24 | `_identify_causal_edges` | `PDETMunicipalPlanAna` | 80/100 | LOW | HIGH |
| 25 | `_estimate_effect_bayesian` | `PDETMunicipalPlanAna` | 80/100 | LOW | HIGH |
| 26 | `calculate_quality_score` | `PDETMunicipalPlanAna` | 80/100 | MEDIUM | HIGH |
| 27 | `generate_confidence_repor` | `ReportingEngine` | 80/100 | MEDIUM | HIGH |
| 28 | `aggregate_risk_and_priori` | `BayesianCounterfactu` | 80/100 | MEDIUM | HIGH |
| 29 | `aggregate_risk` | `DerekBeachProducer` | 80/100 | MEDIUM | HIGH |
| 30 | `construct_causal_dag` | `PDETMunicipalPlanAna` | 75/100 | MEDIUM | HIGH |
| 31 | `estimate_causal_effects` | `PDETMunicipalPlanAna` | 75/100 | MEDIUM | HIGH |

---

## 📁 Análisis por Archivo

Reporte detallado método por método para cada archivo del sistema.

### `financiero_viabilidad_tablas.py`

**Total de Métodos:** 61

**Estadísticas:**
- Score Promedio de Aptitud: 96.64/100
- Métodos CRITICAL: 1
- Métodos HIGH Priority: 0
- Métodos HIGH Complexity: 9

#### Métodos Principales

| Método | Prioridad | Complejidad | Aptitud | Línea |
|--------|-----------|-------------|---------|-------|
| `_get_spanish_stopwords` | LOW | LOW | 100/100 | 322 |
| `_clean_dataframe` | LOW | LOW | 100/100 | 402 |
| `_is_likely_header` | LOW | LOW | 100/100 | 420 |
| `_deduplicate_tables` | LOW | LOW | 100/100 | 428 |
| `_classify_tables` | LOW | LOW | 100/100 | 495 |
| `_extract_financial_amounts` | LOW | LOW | 100/100 | 538 |
| `_identify_funding_source` | LOW | LOW | 100/100 | 585 |
| `_extract_from_budget_table` | LOW | LOW | 100/100 | 602 |
| `_analyze_funding_sources` | LOW | LOW | 100/100 | 642 |
| `_assess_financial_sustainabili` | LOW | LOW | 100/100 | 663 |
| `_interpret_risk` | LOW | LOW | 100/100 | 721 |
| `_indicator_to_dict` | LOW | LOW | 100/100 | 733 |
| `_extract_entities_ner` | LOW | LOW | 100/100 | 761 |
| `_extract_entities_syntax` | LOW | LOW | 100/100 | 786 |
| `_classify_entity_type` | LOW | LOW | 100/100 | 813 |

#### Métodos de Ejecución Prioritaria

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, use_gpu, language, confidence_threshold)`
- **Dependencias:** Instance state

---

### `Analyzer_one.py`

**Total de Métodos:** 46

**Estadísticas:**
- Score Promedio de Aptitud: 98.26/100
- Métodos CRITICAL: 8
- Métodos HIGH Priority: 0
- Métodos HIGH Complexity: 0

#### Métodos Principales

| Método | Prioridad | Complejidad | Aptitud | Línea |
|--------|-----------|-------------|---------|-------|
| `extract_semantic_cube` | MEDIUM | MEDIUM | 100/100 | 165 |
| `_empty_semantic_cube` | LOW | LOW | 100/100 | 237 |
| `_vectorize_segments` | LOW | LOW | 100/100 | 258 |
| `_process_segment` | LOW | LOW | 100/100 | 273 |
| `_classify_value_chain_link` | LOW | LOW | 100/100 | 310 |
| `_classify_policy_domain` | LOW | LOW | 100/100 | 333 |
| `_classify_cross_cutting_themes` | LOW | LOW | 100/100 | 348 |
| `_calculate_semantic_complexity` | LOW | LOW | 100/100 | 363 |
| `analyze_performance` | MEDIUM | MEDIUM | 100/100 | 391 |
| `_calculate_throughput_metrics` | LOW | LOW | 100/100 | 422 |
| `_detect_bottlenecks` | LOW | LOW | 100/100 | 460 |
| `_calculate_loss_functions` | LOW | LOW | 100/100 | 494 |
| `_generate_recommendations` | LOW | LOW | 100/100 | 528 |
| `diagnose_critical_links` | MEDIUM | MEDIUM | 100/100 | 577 |
| `_identify_critical_links` | LOW | LOW | 100/100 | 613 |

#### Métodos de Ejecución Prioritaria

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, ontology)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, ontology)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, ontology)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, questionnaire_path, rubric_path, segmentation_method)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config_path)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, analyzer)`
- **Dependencias:** Instance state

---

### `contradiction_deteccion.py`

**Total de Métodos:** 54

**Estadísticas:**
- Score Promedio de Aptitud: 99.44/100
- Métodos CRITICAL: 3
- Métodos HIGH Priority: 1
- Métodos HIGH Complexity: 0

#### Métodos Principales

| Método | Prioridad | Complejidad | Aptitud | Línea |
|--------|-----------|-------------|---------|-------|
| `calculate_posterior` | MEDIUM | MEDIUM | 100/100 | 112 |
| `verify_temporal_consistency` | MEDIUM | MEDIUM | 100/100 | 153 |
| `_build_timeline` | LOW | LOW | 100/100 | 182 |
| `_parse_temporal_marker` | LOW | LOW | 100/100 | 196 |
| `_has_temporal_conflict` | LOW | LOW | 100/100 | 213 |
| `_are_mutually_exclusive` | LOW | LOW | 100/100 | 224 |
| `_extract_resources` | LOW | LOW | 100/100 | 236 |
| `_check_deadline_constraints` | HIGH | LOW | 100/100 | 251 |
| `_should_precede` | LOW | LOW | 100/100 | 268 |
| `_classify_temporal_type` | LOW | LOW | 100/100 | 273 |
| `_initialize_pdm_patterns` | LOW | LOW | 100/100 | 323 |
| `detect` | MEDIUM | MEDIUM | 100/100 | 348 |
| `_extract_policy_statements` | LOW | LOW | 100/100 | 418 |
| `_generate_embeddings` | LOW | LOW | 100/100 | 459 |
| `_build_knowledge_graph` | LOW | LOW | 100/100 | 486 |

#### Métodos de Ejecución Prioritaria

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self)`
- **Dependencias:** Instance state

##### `_check_deadline_constraints`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `_check_deadline_constraints(self, timeline)`
- **Prerequisitos:** Instance of TemporalLogicVerifier must be initialized
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, model_name, spacy_model, device)`
- **Dependencias:** Instance state

---

### `embedding_policy.py`

**Total de Métodos:** 68

**Estadísticas:**
- Score Promedio de Aptitud: 99.56/100
- Métodos CRITICAL: 5
- Métodos HIGH Priority: 0
- Métodos HIGH Complexity: 0

#### Métodos Principales

| Método | Prioridad | Complejidad | Aptitud | Línea |
|--------|-----------|-------------|---------|-------|
| `chunk_document` | MEDIUM | MEDIUM | 100/100 | 158 |
| `_normalize_text` | LOW | LOW | 100/100 | 226 |
| `_recursive_split` | LOW | LOW | 100/100 | 233 |
| `_find_sentence_boundary` | LOW | LOW | 100/100 | 275 |
| `_extract_sections` | LOW | LOW | 100/100 | 286 |
| `_extract_tables` | LOW | LOW | 100/100 | 302 |
| `_extract_lists` | LOW | LOW | 100/100 | 316 |
| `_infer_pdq_context` | LOW | LOW | 100/100 | 323 |
| `_contains_table` | LOW | LOW | 100/100 | 384 |
| `_contains_list` | LOW | LOW | 100/100 | 393 |
| `_find_section` | LOW | LOW | 100/100 | 397 |
| `__init__` | CRITICAL | MEDIUM | 100/100 | 425 |
| `evaluate_policy_metric` | MEDIUM | MEDIUM | 100/100 | 436 |
| `_beta_binomial_posterior` | LOW | LOW | 100/100 | 485 |
| `_normal_normal_posterior` | LOW | LOW | 100/100 | 512 |

#### Métodos de Ejecución Prioritaria

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 100.0/100
- **Firma:** `__init__(self, prior_strength)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 100.0/100
- **Firma:** `__init__(self, model_name, max_length, retry_handler)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config, retry_handler)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 100.0/100
- **Firma:** `__init__(self, config, model_tier, retry_handler)`
- **Dependencias:** Instance state

---

### `teoria_cambio.py`

**Total de Métodos:** 39

**Estadísticas:**
- Score Promedio de Aptitud: 98.21/100
- Métodos CRITICAL: 3
- Métodos HIGH Priority: 3
- Métodos HIGH Complexity: 4

#### Métodos Principales

| Método | Prioridad | Complejidad | Aptitud | Línea |
|--------|-----------|-------------|---------|-------|
| `__post_init__` | LOW | MEDIUM | 100/100 | 159 |
| `_normalize_metadata` | LOW | LOW | 100/100 | 181 |
| `_sanitize_confidence` | LOW | LOW | 100/100 | 204 |
| `_sanitize_created` | LOW | LOW | 100/100 | 212 |
| `_sanitize_metadata_value` | LOW | LOW | 100/100 | 223 |
| `to_serializable_dict` | MEDIUM | MEDIUM | 100/100 | 233 |
| `__init__` | CRITICAL | MEDIUM | 100/100 | 296 |
| `_es_conexion_valida` | LOW | LOW | 100/100 | 303 |
| `validacion_completa` | MEDIUM | MEDIUM | 100/100 | 331 |
| `_extraer_categorias` | LOW | LOW | 100/100 | 347 |
| `_encontrar_caminos_completos` | LOW | LOW | 100/100 | 367 |
| `_generar_sugerencias_internas` | LOW | LOW | 100/100 | 391 |
| `add_node` | MEDIUM | MEDIUM | 100/100 | 472 |
| `add_edge` | MEDIUM | MEDIUM | 100/100 | 484 |
| `_initialize_rng` | LOW | LOW | 100/100 | 493 |

#### Métodos de Ejecución Prioritaria

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 100.0/100
- **Firma:** `__init__(self)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, graph_type)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self)`
- **Dependencias:** Instance state

##### `validate_engine_readiness`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate_engine_readiness(self)`
- **Prerequisitos:** Instance of IndustrialGradeValidator must be initialized
- **Dependencias:** Instance state

##### `validate_causal_categories`

- **Prioridad:** HIGH
- **Aptitud:** 85.0/100
- **Firma:** `validate_causal_categories(self)`
- **Prerequisitos:** Instance of IndustrialGradeValidator must be initialized
- **Dependencias:** Instance state
- **Riesgos:** High computational complexity

##### `validate_connection_matrix`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate_connection_matrix(self)`
- **Prerequisitos:** Instance of IndustrialGradeValidator must be initialized
- **Dependencias:** Instance state

---

### `dereck_beach.py`

**Total de Métodos:** 159

**Estadísticas:**
- Score Promedio de Aptitud: 97.83/100
- Métodos CRITICAL: 15
- Métodos HIGH Priority: 12
- Métodos HIGH Complexity: 13

#### Métodos Principales

| Método | Prioridad | Complejidad | Aptitud | Línea |
|--------|-----------|-------------|---------|-------|
| `classify_test` | MEDIUM | MEDIUM | 100/100 | 122 |
| `apply_test_logic` | MEDIUM | MEDIUM | 100/100 | 143 |
| `_format_message` | LOW | LOW | 100/100 | 205 |
| `to_dict` | MEDIUM | MEDIUM | 100/100 | 215 |
| `check_sum_to_one` | HIGH | MEDIUM | 100/100 | 289 |
| `_load_config` | LOW | LOW | 100/100 | 458 |
| `_load_default_config` | LOW | LOW | 100/100 | 475 |
| `_validate_config` | HIGH | LOW | 100/100 | 562 |
| `get` | MEDIUM | MEDIUM | 100/100 | 593 |
| `get_mechanism_prior` | MEDIUM | MEDIUM | 100/100 | 610 |
| `get_performance_setting` | MEDIUM | MEDIUM | 100/100 | 616 |
| `update_priors_from_feedback` | MEDIUM | MEDIUM | 100/100 | 622 |
| `_save_prior_history` | LOW | LOW | 100/100 | 718 |
| `_load_uncertainty_history` | LOW | LOW | 100/100 | 781 |
| `check_uncertainty_reduction_cr` | HIGH | MEDIUM | 100/100 | 806 |

#### Métodos de Ejecución Prioritaria

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, message, details, stage, recoverable)`
- **Dependencias:** Instance state

##### `check_sum_to_one`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `check_sum_to_one(cls, v, values)`

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config_path)`
- **Dependencias:** Instance state

##### `_validate_config`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `_validate_config(self)`
- **Prerequisitos:** Instance of ConfigLoader must be initialized
- **Dependencias:** Instance state

##### `check_uncertainty_reduction_criterion`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `check_uncertainty_reduction_criterion(self, current_uncertainty)`
- **Prerequisitos:** Instance of ConfigLoader must be initialized
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config, retry_handler)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config, nlp_model)`
- **Dependencias:** Instance state

##### `_check_structural_violation`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `_check_structural_violation(self, source, target)`
- **Prerequisitos:** Instance of CausalExtractor must be initialized
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config, nlp_model)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config)`
- **Dependencias:** Instance state

---

### `policy_processor.py`

**Total de Métodos:** 41

**Estadísticas:**
- Score Promedio de Aptitud: 97.32/100
- Métodos CRITICAL: 7
- Métodos HIGH Priority: 1
- Métodos HIGH Complexity: 1

#### Métodos Principales

| Método | Prioridad | Complejidad | Aptitud | Línea |
|--------|-----------|-------------|---------|-------|
| `from_legacy` | MEDIUM | MEDIUM | 100/100 | 350 |
| `validate` | HIGH | MEDIUM | 100/100 | 358 |
| `compute_evidence_score` | MEDIUM | MEDIUM | 100/100 | 399 |
| `_calculate_shannon_entropy` | LOW | LOW | 100/100 | 440 |
| `normalize_unicode` | MEDIUM | MEDIUM | 100/100 | 473 |
| `segment_into_sentences` | MEDIUM | MEDIUM | 100/100 | 477 |
| `extract_contextual_window` | MEDIUM | MEDIUM | 100/100 | 503 |
| `compile_pattern` | MEDIUM | MEDIUM | 100/100 | 519 |
| `_load_questionnaire` | LOW | LOW | 100/100 | 603 |
| `_compile_pattern_registry` | LOW | LOW | 100/100 | 617 |
| `_build_point_patterns` | LOW | LOW | 100/100 | 628 |
| `process` | CRITICAL | MEDIUM | 100/100 | 657 |
| `_match_patterns_in_sentences` | LOW | LOW | 100/100 | 749 |
| `_compute_evidence_confidence` | LOW | LOW | 100/100 | 773 |
| `_construct_evidence_bundle` | LOW | LOW | 100/100 | 792 |

#### Métodos de Ejecución Prioritaria

##### `validate`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate(self)`
- **Prerequisitos:** Instance of ProcessorConfig must be initialized
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, prior_confidence, entropy_weight)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config, questionnaire_path)`
- **Dependencias:** Instance state

##### `process`

- **Prioridad:** CRITICAL
- **Aptitud:** 100.0/100
- **Firma:** `process(self, raw_text)`
- **Prerequisitos:** Instance of IndustrialPolicyProcessor must be initialized
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self, config, questionnaire_path)`
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 90.0/100
- **Firma:** `__init__(self)`
- **Dependencias:** Instance state

---

### `report_assembly.py`

**Total de Métodos:** 125

**Estadísticas:**
- Score Promedio de Aptitud: 99.64/100
- Métodos CRITICAL: 2
- Métodos HIGH Priority: 7
- Métodos HIGH Complexity: 4

#### Métodos Principales

| Método | Prioridad | Complejidad | Aptitud | Línea |
|--------|-----------|-------------|---------|-------|
| `__init__` | CRITICAL | MEDIUM | 100/100 | 141 |
| `generate_micro_answer` | MEDIUM | MEDIUM | 100/100 | 200 |
| `_apply_scoring_modality` | LOW | LOW | 100/100 | 307 |
| `_score_type_a` | LOW | LOW | 100/100 | 505 |
| `_score_type_b` | LOW | LOW | 100/100 | 564 |
| `_score_type_c` | LOW | LOW | 100/100 | 602 |
| `_score_type_d` | LOW | LOW | 100/100 | 631 |
| `_score_type_e` | LOW | LOW | 100/100 | 667 |
| `_score_type_f` | LOW | LOW | 100/100 | 738 |
| `_evaluate_condition` | LOW | LOW | 100/100 | 818 |
| `_score_default` | LOW | LOW | 100/100 | 882 |
| `_extract_numerical_value` | LOW | LOW | 100/100 | 906 |
| `_extract_pattern_matches` | LOW | LOW | 100/100 | 921 |
| `_score_to_qualitative_question` | LOW | LOW | 100/100 | 936 |
| `_extract_evidence_excerpts` | LOW | LOW | 100/100 | 955 |

#### Métodos de Ejecución Prioritaria

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 100.0/100
- **Firma:** `__init__(self, dimension_descriptions, cluster_weights, cluster_policy_weights, causal_thresholds)`
- **Dependencias:** Instance state

##### `_validate_macro_gating`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `_validate_macro_gating(self, all_micro_answers, coverage_index)`
- **Prerequisitos:** Instance of ReportAssembler must be initialized
- **Dependencias:** Instance state

##### `validate_micro_answer_schema`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate_micro_answer_schema(self, answer_data)`
- **Prerequisitos:** Instance of ReportAssembler must be initialized
- **Dependencias:** Instance state

##### `validate_meso_cluster_schema`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate_meso_cluster_schema(self, cluster_data)`
- **Prerequisitos:** Instance of ReportAssembler must be initialized
- **Dependencias:** Instance state

##### `validate_macro_convergence_schema`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate_macro_convergence_schema(self, convergence_data)`
- **Prerequisitos:** Instance of ReportAssembler must be initialized
- **Dependencias:** Instance state

##### `__init__`

- **Prioridad:** CRITICAL
- **Aptitud:** 100.0/100
- **Firma:** `__init__(self, dimension_descriptions, cluster_weights, cluster_policy_weights, causal_thresholds)`
- **Dependencias:** Instance state

##### `validate_micro_answer`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate_micro_answer(self, answer_data)`
- **Prerequisitos:** Instance of ReportAssemblyProducer must be initialized
- **Dependencias:** Instance state

##### `validate_meso_cluster`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate_meso_cluster(self, cluster_data)`
- **Prerequisitos:** Instance of ReportAssemblyProducer must be initialized
- **Dependencias:** Instance state

##### `validate_macro_convergence`

- **Prioridad:** HIGH
- **Aptitud:** 100.0/100
- **Firma:** `validate_macro_convergence(self, convergence_data)`
- **Prerequisitos:** Instance of ReportAssemblyProducer must be initialized
- **Dependencias:** Instance state

---

## 🎓 Recomendaciones de Ejecución

### Orden de Prioridad de Ejecución

1. **NIVEL CRÍTICO (CRITICAL):** Ejecutar primero todos los métodos de inicialización
   y procesamiento principal. Sin estos métodos, el sistema no puede funcionar.

2. **NIVEL ALTO (HIGH):** Ejecutar métodos de análisis bayesiano, construcción de DAG
   y detección de contradicciones. Estos proporcionan la base analítica.

3. **NIVEL MEDIO (MEDIUM):** Métodos de soporte y análisis complementario.

4. **NIVEL BAJO (LOW):** Helpers internos y utilidades auxiliares.

### Consideraciones de Recursos

- **Métodos de Alta Complejidad:** Asignar recursos computacionales adecuados
- **Procesamiento Bayesiano:** Requiere librerías numpy/scipy configuradas
- **Análisis de Grafos:** Requiere networkx instalado y configurado
- **NLP:** Requiere modelos de lenguaje y vectorización

### Gestión de Dependencias

```bash
# Instalación de dependencias principales
pip install numpy scipy networkx pandas
pip install scikit-learn spacy
pip install bayesian-optimization
```

### Monitoreo y Validación

Para cada método crítico y de alta prioridad:

1. **Validar prerequisitos** antes de la ejecución
2. **Capturar excepciones** y registrar errores detallados
3. **Verificar outputs** contra esquemas esperados
4. **Medir tiempos** de ejecución para detectar degradación

---

## 📚 Referencias

- **Catálogo Completo:** `metodos_completos_nivel3.json`
- **Ejemplos de Uso:** `ejemplo_uso_nivel3.py`
- **Referencia Rápida:** `CHEATSHEET_NIVEL3.txt`
- **Inventario General:** `/inventory.json`

---

*Documento generado automáticamente el 2025-10-29 03:34:44*
