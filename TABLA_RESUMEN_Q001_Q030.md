# TABLA RESUMEN: MAPEO DE EXECUTORS Y MÉTODOS CORE (Q001-Q030)

## Resumen Ejecutivo

| Pregunta | Executor | Dimensión | Total Métodos | Archivos Core Utilizados |
|----------|----------|-----------|---------------|--------------------------|
| Q001 | D1Q1_Executor | DIM01 | 18 | PP, CD, EP, A1 |
| Q002 | D1Q2_Executor | DIM01 | 12 | PP, CD, EP |
| Q003 | D1Q3_Executor | DIM01 | 22 | PP, CD, FV, DB, EP |
| Q004 | D1Q4_Executor | DIM01 | 16 | PP, CD, FV, A1 |
| Q005 | D1Q5_Executor | DIM01 | 14 | PP, CD, A1 |
| Q006 | D2Q1_Executor | DIM02 | 20 | PP, CD, FV, SC |
| Q007 | D2Q2_Executor | DIM02 | 23 | PP, CD, DB, TC, A1 |
| Q008 | D2Q3_Executor | DIM02 | 15 | PP, CD, FV, EP, A1 |
| Q009 | D2Q4_Executor | DIM02 | 18 | PP, CD, FV, EP |
| Q010 | D2Q5_Executor | DIM02 | 19 | PP, CD, DB, TC, A1 |
| Q011 | D3Q1_Executor | DIM03 | 18 | PP, CD, FV, EP |
| Q012 | D3Q2_Executor | DIM03 | 19 | PP, CD, FV, EP |
| Q013 | D3Q3_Executor | DIM03 | 15 | PP, CD, FV, EP, A1 |
| Q014 | D3Q4_Executor | DIM03 | 17 | PP, CD, A1 |
| Q015 | D3Q5_Executor | DIM03 | 34 | PP, CD, DB, TC, A1 |
| Q016 | D4Q1_Executor | DIM04 | 18 | PP, CD, FV, EP |
| Q017 | D4Q2_Executor | DIM04 | 24 | PP, CD, DB, TC |
| Q018 | D4Q3_Executor | DIM04 | 20 | PP, CD, FV, DB, EP |
| Q019 | D4Q4_Executor | DIM04 | 15 | PP, CD, EP, A1 |
| Q020 | D4Q5_Executor | DIM04 | 15 | PP, CD, EP, A1 |
| Q021 | D5Q1_Executor | DIM05 | 16 | PP, CD, FV, EP |
| Q022 | D5Q2_Executor | DIM05 | 25 | PP, CD, DB, TC, A1 |
| Q023 | D5Q3_Executor | DIM05 | 18 | PP, CD, DB, EP |
| Q024 | D5Q4_Executor | DIM05 | 14 | PP, CD, A1 |
| Q025 | D5Q5_Executor | DIM05 | 14 | PP, CD, FV, DB |
| Q026 | D6Q1_Executor | DIM06 | 33 | PP, CD, FV, DB, TC |
| Q027 | D6Q2_Executor | DIM06 | 38 | PP, CD, DB, TC |
| Q028 | D6Q3_Executor | DIM06 | 22 | PP, CD, TC, A1 |
| Q029 | D6Q4_Executor | DIM06 | 37 | PP, CD, FV, DB, TC, A1 |
| Q030 | D6Q5_Executor | DIM06 | 24 | PP, CD, EP, A1 |

---

## Métodos Principales por Pregunta

### Q001: Líneas Base y Brechas Cuantificadas
```
[PP] IndustrialPolicyProcessor.process
[PP] BayesianEvidenceScorer.compute_evidence_score
[CD] PolicyContradictionDetector._extract_quantitative_claims
[EP] BayesianNumericalAnalyzer.evaluate_policy_metric
```

### Q002: Normalización y Fuentes
```
[PP] PolicyTextProcessor.normalize_unicode
[CD] PolicyContradictionDetector._are_comparable_claims
[EP] BayesianNumericalAnalyzer._compute_coherence
```

### Q003: Asignación de Recursos
```
[FV] PDETMunicipalPlanAnalyzer.extract_tables
[DB] FinancialAuditor.trace_financial_allocation
[CD] PolicyContradictionDetector._detect_resource_conflicts
[EP] BayesianNumericalAnalyzer.compare_policies
```

### Q004: Capacidad Institucional
```
[CD] PolicyContradictionDetector._build_knowledge_graph
[A1] PerformanceAnalyzer._detect_bottlenecks
[A1] TextMiningEngine._identify_critical_links
```

### Q005: Restricciones Temporales
```
[CD] TemporalLogicVerifier.verify_temporal_consistency
[CD] TemporalLogicVerifier._build_timeline
[A1] PerformanceAnalyzer._calculate_throughput_metrics
```

### Q006: Formato Tabular y Trazabilidad
```
[FV] PDETMunicipalPlanAnalyzer.extract_tables
[FV] PDETMunicipalPlanAnalyzer._reconstruct_fragmented_tables
[FV] PDETMunicipalPlanAnalyzer.analyze_municipal_plan
```

### Q007: Causalidad de Actividades
```
[DB] CausalExtractor.extract_causal_hierarchy
[TC] TeoriaCambio.construir_grafo_causal
[A1] TextMiningEngine.diagnose_critical_links
```

### Q008: Responsables de Actividades
```
[FV] PDETMunicipalPlanAnalyzer.identify_responsible_entities
[EP] PolicyAnalysisEmbedder.semantic_search
```

### Q009: Cuantificación de Actividades
```
[FV] PDETMunicipalPlanAnalyzer._extract_financial_amounts
[CD] PolicyContradictionDetector._extract_quantitative_claims
[EP] BayesianNumericalAnalyzer.evaluate_policy_metric
```

### Q010: Eslabón Causal Diagnóstico-Actividades
```
[DB] CausalExtractor.extract_causal_hierarchy
[TC] TeoriaCambio._encontrar_caminos_completos
[A1] TextMiningEngine.diagnose_critical_links
```

### Q011: Indicadores de Producto
```
[FV] PDETMunicipalPlanAnalyzer._find_product_mentions
[EP] BayesianNumericalAnalyzer.evaluate_policy_metric
```

### Q012: Cuantificación de Productos
```
[FV] PDETMunicipalPlanAnalyzer._extract_financial_amounts
[CD] PolicyContradictionDetector._extract_quantitative_claims
```

### Q013: Responsables de Productos
```
[FV] PDETMunicipalPlanAnalyzer.identify_responsible_entities
[EP] PolicyAnalysisEmbedder.semantic_search
```

### Q014: Plazos de Productos
```
[CD] TemporalLogicVerifier.verify_temporal_consistency
[A1] PerformanceAnalyzer._detect_bottlenecks
```

### Q015: Eslabón Causal Producto-Resultado
```
[DB] MechanismPartExtractor.extract_entity_activity
[DB] BayesianMechanismInference.infer_mechanisms
[DB] BeachEvidentialTest.apply_test_logic
[TC] TeoriaCambio.construir_grafo_causal
```

### Q016: Indicadores de Resultado
```
[FV] PDETMunicipalPlanAnalyzer._find_outcome_mentions
[EP] BayesianNumericalAnalyzer.evaluate_policy_metric
```

### Q017: Cadena Causal y Supuestos
```
[DB] BayesianMechanismInference._test_necessity
[DB] BayesianMechanismInference._test_sufficiency
[TC] TeoriaCambio.validacion_completa
```

### Q018: Justificación de Ambición
```
[FV] PDETMunicipalPlanAnalyzer.analyze_financial_feasibility
[DB] FinancialAuditor._calculate_sufficiency
[EP] BayesianNumericalAnalyzer.compare_policies
```

### Q019: Población Objetivo
```
[A1] SemanticAnalyzer.extract_semantic_cube
[EP] PolicyAnalysisEmbedder.semantic_search
```

### Q020: Alineación con Objetivos Superiores
```
[CD] PolicyContradictionDetector._calculate_objective_alignment
[EP] PolicyAnalysisEmbedder.compare_policy_interventions
```

### Q021: Indicadores de Impacto
```
[FV] PDETMunicipalPlanAnalyzer.extract_tables
[EP] BayesianNumericalAnalyzer.evaluate_policy_metric
```

### Q022: Eslabón Causal Resultado-Impacto
```
[DB] BayesianMechanismInference.infer_mechanisms
[DB] BeachEvidentialTest.apply_test_logic
[TC] TeoriaCambio._encontrar_caminos_completos
```

### Q023: Evidencia de Causalidad
```
[DB] CausalExtractor._extract_causal_justifications
[DB] BayesianMechanismInference._test_necessity/_test_sufficiency
[EP] BayesianNumericalAnalyzer.evaluate_policy_metric
```

### Q024: Plazos de Impacto
```
[CD] TemporalLogicVerifier.verify_temporal_consistency
[A1] PerformanceAnalyzer._calculate_throughput_metrics
```

### Q025: Sostenibilidad Financiera
```
[FV] PDETMunicipalPlanAnalyzer.analyze_financial_feasibility
[DB] FinancialAuditor.trace_financial_allocation
```

### Q026: Integridad de Teoría de Cambio
```
[TC] TeoriaCambio.validacion_completa
[TC] AdvancedDAGValidator.calculate_acyclicity_pvalue
[DB] CDAFFramework.process_document
[DB] OperationalizationAuditor.audit_evidence_traceability
```

### Q027: Proporcionalidad y Continuidad
```
[DB] BayesianMechanismInference._build_transition_matrix
[DB] CausalInferenceSetup.classify_goal_dynamics
[TC] AdvancedDAGValidator._perform_sensitivity_analysis_internal
```

### Q028: Inconsistencias
```
[CD] PolicyContradictionDetector._detect_logical_incompatibilities
[CD] PolicyContradictionDetector.detect
[CD] PolicyContradictionDetector._generate_resolution_recommendations
```

### Q029: Adaptación
```
[TC] TeoriaCambio._generar_sugerencias_internas
[DB] CDAFFramework._generate_causal_model_json
[DB] OperationalizationAuditor._perform_counterfactual_budget_check
```

### Q030: Contextualización
```
[A1] SemanticAnalyzer.extract_semantic_cube
[EP] PolicyAnalysisEmbedder.compare_policy_interventions
[EP] AdvancedSemanticChunker._infer_pdq_context
```

---

## Matriz de Uso de Archivos Core

| Pregunta | PP | CD | FV | DB | EP | A1 | TC | SC |
|----------|----|----|----|----|----|----|----|----|
| Q001 | ✓ | ✓ | - | - | ✓ | ✓ | - | - |
| Q002 | ✓ | ✓ | - | - | ✓ | - | - | - |
| Q003 | ✓ | ✓ | ✓ | ✓ | ✓ | - | - | - |
| Q004 | ✓ | ✓ | ✓ | - | - | ✓ | - | - |
| Q005 | ✓ | ✓ | - | - | - | ✓ | - | - |
| Q006 | ✓ | ✓ | ✓ | - | - | - | - | ✓ |
| Q007 | ✓ | ✓ | - | ✓ | - | ✓ | ✓ | - |
| Q008 | ✓ | ✓ | ✓ | - | ✓ | ✓ | - | - |
| Q009 | ✓ | ✓ | ✓ | - | ✓ | - | - | - |
| Q010 | ✓ | ✓ | - | ✓ | - | ✓ | ✓ | - |
| Q011 | ✓ | ✓ | ✓ | - | ✓ | - | - | - |
| Q012 | ✓ | ✓ | ✓ | - | ✓ | - | - | - |
| Q013 | ✓ | ✓ | ✓ | - | ✓ | ✓ | - | - |
| Q014 | ✓ | ✓ | - | - | - | ✓ | - | - |
| Q015 | ✓ | ✓ | - | ✓ | - | ✓ | ✓ | - |
| Q016 | ✓ | ✓ | ✓ | - | ✓ | - | - | - |
| Q017 | ✓ | ✓ | - | ✓ | - | - | ✓ | - |
| Q018 | ✓ | ✓ | ✓ | ✓ | ✓ | - | - | - |
| Q019 | ✓ | ✓ | - | - | ✓ | ✓ | - | - |
| Q020 | ✓ | ✓ | - | - | ✓ | ✓ | - | - |
| Q021 | ✓ | ✓ | ✓ | - | ✓ | - | - | - |
| Q022 | ✓ | ✓ | - | ✓ | - | ✓ | ✓ | - |
| Q023 | ✓ | ✓ | - | ✓ | ✓ | - | - | - |
| Q024 | ✓ | ✓ | - | - | - | ✓ | - | - |
| Q025 | ✓ | ✓ | ✓ | ✓ | - | - | - | - |
| Q026 | ✓ | ✓ | ✓ | ✓ | - | - | ✓ | - |
| Q027 | ✓ | ✓ | - | ✓ | - | - | ✓ | - |
| Q028 | ✓ | ✓ | - | - | - | ✓ | ✓ | - |
| Q029 | ✓ | ✓ | ✓ | ✓ | - | ✓ | ✓ | - |
| Q030 | ✓ | ✓ | - | - | ✓ | ✓ | - | - |
| **Total** | **30** | **30** | **18** | **13** | **21** | **15** | **12** | **1** |

---

## Estadísticas Finales

**Total de métodos únicos identificados:** 613  
**Promedio de métodos por pregunta:** 20.4  
**Pregunta más compleja:** Q027 (38 métodos)  
**Pregunta más simple:** Q002 (12 métodos)  

**Archivos más utilizados:**
1. policy_processor.py (PP) - 100%
2. contradiction_deteccion.py (CD) - 100%
3. embedding_policy.py (EP) - 70%
4. financiero_viabilidad_tablas.py (FV) - 60%

**Archivos especializados:**
- dereck_beach.py (DB) - 43% (análisis causal profundo)
- teoria_cambio.py (TC) - 40% (validación de teoría de cambio)
- Analyzer_one.py (A1) - 50% (análisis general)
- semantic_chunking_policy.py (SC) - 3% (uso muy específico)
