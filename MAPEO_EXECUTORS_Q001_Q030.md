# MAPEO COMPLETO DE EXECUTORS Y MÉTODOS CORE
## Preguntas Canónicas Q001-Q030

**Fecha de análisis:** 2025-11-13  
**Archivo fuente:** src/saaaaaa/core/orchestrator/executors.py

---

## ARCHIVOS CORE IDENTIFICADOS

| Abrev | Archivo                            | Descripción                          |
|-------|------------------------------------|--------------------------------------|
| **PP** | policy_processor.py               | Procesamiento de políticas           |
| **CD** | contradiction_deteccion.py        | Detección de contradicciones         |
| **FV** | financiero_viabilidad_tablas.py   | Análisis financiero y tablas         |
| **DB** | dereck_beach.py                   | Causal analysis y Derek Beach framework |
| **EP** | embedding_policy.py               | Embeddings y análisis semántico      |
| **A1** | Analyzer_one.py                   | Analizadores generales               |
| **TC** | teoria_cambio.py                  | Teoría de cambio y validación DAG    |
| **SC** | semantic_chunking_policy.py       | Chunking semántico                   |

---

## DIMENSIÓN 1 (DIM01): DIAGNÓSTICO

### Q001: Líneas Base y Brechas Cuantificadas
**Executor:** D1Q1_Executor  
**Métodos utilizados:** 18

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[CD]** PolicyContradictionDetector, BayesianConfidenceCalculator
- **[EP]** BayesianNumericalAnalyzer
- **[A1]** SemanticAnalyzer

**Métodos clave:**
- `IndustrialPolicyProcessor.process` - Procesamiento principal del documento
- `BayesianEvidenceScorer.compute_evidence_score` - Cálculo de score bayesiano
- `PolicyContradictionDetector._extract_quantitative_claims` - Extracción de afirmaciones cuantitativas
- `BayesianNumericalAnalyzer.evaluate_policy_metric` - Evaluación de métricas

---

### Q002: Normalización y Fuentes
**Executor:** D1Q2_Executor  
**Métodos utilizados:** 12

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[CD]** PolicyContradictionDetector, BayesianConfidenceCalculator
- **[EP]** PolicyAnalysisEmbedder, BayesianNumericalAnalyzer

**Métodos clave:**
- `PolicyTextProcessor.normalize_unicode` - Normalización de texto
- `PolicyContradictionDetector._are_comparable_claims` - Comparación de afirmaciones
- `BayesianNumericalAnalyzer._compute_coherence` - Cálculo de coherencia

---

### Q003: Asignación de Recursos
**Executor:** D1Q3_Executor  
**Métodos utilizados:** 22

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, BayesianEvidenceScorer
- **[CD]** PolicyContradictionDetector, TemporalLogicVerifier, BayesianConfidenceCalculator
- **[FV]** PDETMunicipalPlanAnalyzer
- **[DB]** FinancialAuditor
- **[EP]** BayesianNumericalAnalyzer

**Métodos clave:**
- `PDETMunicipalPlanAnalyzer.extract_tables` - Extracción de tablas
- `PDETMunicipalPlanAnalyzer._analyze_funding_sources` - Análisis de fuentes de financiación
- `FinancialAuditor.trace_financial_allocation` - Trazabilidad financiera
- `PolicyContradictionDetector._detect_resource_conflicts` - Detección de conflictos de recursos

---

### Q004: Capacidad Institucional
**Executor:** D1Q4_Executor  
**Métodos utilizados:** 16

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[CD]** PolicyContradictionDetector
- **[FV]** PDETMunicipalPlanAnalyzer
- **[A1]** SemanticAnalyzer, PerformanceAnalyzer, TextMiningEngine

**Métodos clave:**
- `PolicyContradictionDetector._build_knowledge_graph` - Construcción de grafo de conocimiento
- `PolicyContradictionDetector._calculate_graph_fragmentation` - Cálculo de fragmentación
- `PerformanceAnalyzer._detect_bottlenecks` - Detección de cuellos de botella
- `TextMiningEngine._identify_critical_links` - Identificación de enlaces críticos

---

### Q005: Restricciones Temporales
**Executor:** D1Q5_Executor  
**Métodos utilizados:** 14

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[CD]** PolicyContradictionDetector, TemporalLogicVerifier
- **[A1]** SemanticAnalyzer, PerformanceAnalyzer

**Métodos clave:**
- `TemporalLogicVerifier.verify_temporal_consistency` - Verificación de consistencia temporal
- `TemporalLogicVerifier._build_timeline` - Construcción de línea de tiempo
- `PolicyContradictionDetector._detect_temporal_conflicts` - Detección de conflictos temporales

---

## DIMENSIÓN 2 (DIM02): ACTIVIDADES

### Q006: Formato Tabular y Trazabilidad
**Executor:** D2Q1_Executor  
**Métodos utilizados:** 20

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[FV]** PDETMunicipalPlanAnalyzer (mayoría de métodos)
- **[CD]** TemporalLogicVerifier, PolicyContradictionDetector
- **[SC]** SemanticProcessor

**Métodos clave:**
- `PDETMunicipalPlanAnalyzer.extract_tables` - Extracción de tablas
- `PDETMunicipalPlanAnalyzer._clean_dataframe` - Limpieza de datos
- `PDETMunicipalPlanAnalyzer._reconstruct_fragmented_tables` - Reconstrucción de tablas
- `PDETMunicipalPlanAnalyzer.analyze_municipal_plan` - Análisis de plan municipal

---

### Q007: Causalidad de Actividades
**Executor:** D2Q2_Executor  
**Métodos utilizados:** 23

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[CD]** PolicyContradictionDetector
- **[DB]** CausalExtractor
- **[TC]** TeoriaCambio
- **[A1]** TextMiningEngine

**Métodos clave:**
- `CausalExtractor.extract_causal_hierarchy` - Extracción de jerarquía causal
- `CausalExtractor._extract_causal_links` - Extracción de enlaces causales
- `TeoriaCambio.construir_grafo_causal` - Construcción de grafo causal
- `TextMiningEngine.diagnose_critical_links` - Diagnóstico de enlaces críticos

---

### Q008: Responsables de Actividades
**Executor:** D2Q3_Executor  
**Métodos utilizados:** 15

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[FV]** PDETMunicipalPlanAnalyzer
- **[CD]** PolicyContradictionDetector
- **[EP]** PolicyAnalysisEmbedder
- **[A1]** SemanticAnalyzer

**Métodos clave:**
- `PDETMunicipalPlanAnalyzer.identify_responsible_entities` - Identificación de entidades responsables
- `PDETMunicipalPlanAnalyzer._classify_entity_type` - Clasificación de tipo de entidad
- `PolicyAnalysisEmbedder.semantic_search` - Búsqueda semántica

---

### Q009: Cuantificación de Actividades
**Executor:** D2Q4_Executor  
**Métodos utilizados:** 18

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[FV]** PDETMunicipalPlanAnalyzer
- **[CD]** PolicyContradictionDetector, BayesianConfidenceCalculator
- **[EP]** BayesianNumericalAnalyzer

**Métodos clave:**
- `PDETMunicipalPlanAnalyzer._extract_financial_amounts` - Extracción de montos financieros
- `PolicyContradictionDetector._extract_quantitative_claims` - Extracción de afirmaciones cuantitativas
- `BayesianNumericalAnalyzer.evaluate_policy_metric` - Evaluación de métricas

---

### Q010: Eslabón Causal Diagnóstico-Actividades
**Executor:** D2Q5_Executor  
**Métodos utilizados:** 19

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[CD]** PolicyContradictionDetector
- **[DB]** CausalExtractor
- **[TC]** TeoriaCambio
- **[A1]** TextMiningEngine

**Métodos clave:**
- `CausalExtractor.extract_causal_hierarchy` - Extracción de jerarquía causal
- `TeoriaCambio._encontrar_caminos_completos` - Búsqueda de caminos completos
- `TextMiningEngine.diagnose_critical_links` - Diagnóstico de enlaces críticos

---

## DIMENSIÓN 3 (DIM03): PRODUCTOS

### Q011: Indicadores de Producto
**Executor:** D3Q1_Executor  
**Métodos utilizados:** 18

**Archivos core:**
- **[PP]** IndustrialPolicyProcessor, PolicyTextProcessor, BayesianEvidenceScorer
- **[CD]** PolicyContradictionDetector, BayesianConfidenceCalculator
- **[FV]** PDETMunicipalPlanAnalyzer
- **[EP]** BayesianNumericalAnalyzer, PolicyAnalysisEmbedder

**Métodos clave:**
- `PDETMunicipalPlanAnalyzer._find_product_mentions` - Búsqueda de menciones de productos
- `PDETMunicipalPlanAnalyzer._indicator_to_dict` - Conversión de indicador a diccionario

---

### Q012: Cuantificación de Productos
**Executor:** D3Q2_Executor  
**Métodos utilizados:** 19

### Q013: Responsables de Productos
**Executor:** D3Q3_Executor  
**Métodos utilizados:** 15

### Q014: Plazos de Productos
**Executor:** D3Q4_Executor  
**Métodos utilizados:** 17

### Q015: Eslabón Causal Producto-Resultado
**Executor:** D3Q5_Executor  
**Métodos utilizados:** 34

**Métodos clave de Q015:**
- `MechanismPartExtractor.extract_entity_activity` - Extracción de actividad de entidades
- `BayesianMechanismInference.infer_mechanisms` - Inferencia de mecanismos
- `BeachEvidentialTest.apply_test_logic` - Aplicación de tests evidenciales
- `TeoriaCambio.construir_grafo_causal` - Construcción de grafo causal

---

## DIMENSIÓN 4 (DIM04): RESULTADOS

### Q016: Indicadores de Resultado
**Executor:** D4Q1_Executor  
**Métodos utilizados:** 18

**Métodos clave:**
- `PDETMunicipalPlanAnalyzer._find_outcome_mentions` - Búsqueda de menciones de resultados

---

### Q017: Cadena Causal y Supuestos
**Executor:** D4Q2_Executor  
**Métodos utilizados:** 24

**Métodos clave:**
- `BayesianMechanismInference._test_necessity` - Test de necesidad
- `BayesianMechanismInference._test_sufficiency` - Test de suficiencia
- `BeachEvidentialTest.classify_test` - Clasificación de tests
- `TeoriaCambio.validacion_completa` - Validación completa

---

### Q018: Justificación de Ambición
**Executor:** D4Q3_Executor  
**Métodos utilizados:** 20

**Métodos clave:**
- `PDETMunicipalPlanAnalyzer.analyze_financial_feasibility` - Análisis de viabilidad financiera
- `PDETMunicipalPlanAnalyzer._assess_financial_sustainability` - Evaluación de sostenibilidad
- `FinancialAuditor._calculate_sufficiency` - Cálculo de suficiencia

---

### Q019: Población Objetivo
**Executor:** D4Q4_Executor  
**Métodos utilizados:** 15

**Métodos clave:**
- `SemanticAnalyzer._classify_cross_cutting_themes` - Clasificación de temas transversales
- `SemanticAnalyzer.extract_semantic_cube` - Extracción de cubo semántico
- `PolicyAnalysisEmbedder._filter_by_pdq` - Filtrado por PDQ

---

### Q020: Alineación con Objetivos Superiores
**Executor:** D4Q5_Executor  
**Métodos utilizados:** 15

**Métodos clave:**
- `PolicyContradictionDetector._calculate_objective_alignment` - Cálculo de alineación de objetivos
- `PolicyAnalysisEmbedder.compare_policy_interventions` - Comparación de intervenciones

---

## DIMENSIÓN 5 (DIM05): IMPACTO

### Q021: Indicadores de Impacto
**Executor:** D5Q1_Executor  
**Métodos utilizados:** 16

### Q022: Eslabón Causal Resultado-Impacto
**Executor:** D5Q2_Executor  
**Métodos utilizados:** 25

**Métodos clave:**
- `BayesianMechanismInference.infer_mechanisms` - Inferencia de mecanismos
- `BeachEvidentialTest.apply_test_logic` - Aplicación de tests evidenciales
- `TeoriaCambio._encontrar_caminos_completos` - Búsqueda de caminos completos

---

### Q023: Evidencia de Causalidad
**Executor:** D5Q3_Executor  
**Métodos utilizados:** 18

**Métodos clave:**
- `CausalExtractor._extract_causal_justifications` - Extracción de justificaciones causales
- `BayesianMechanismInference._test_necessity` - Test de necesidad
- `BayesianMechanismInference._test_sufficiency` - Test de suficiencia

---

### Q024: Plazos de Impacto
**Executor:** D5Q4_Executor  
**Métodos utilizados:** 14

### Q025: Sostenibilidad Financiera
**Executor:** D5Q5_Executor  
**Métodos utilizados:** 14

**Métodos clave:**
- `PDETMunicipalPlanAnalyzer.analyze_financial_feasibility` - Análisis de viabilidad financiera
- `PDETMunicipalPlanAnalyzer._bayesian_risk_inference` - Inferencia bayesiana de riesgo
- `FinancialAuditor.trace_financial_allocation` - Trazabilidad financiera

---

## DIMENSIÓN 6 (DIM06): INTEGRIDAD Y COHERENCIA

### Q026: Integridad de Teoría de Cambio
**Executor:** D6Q1_Executor  
**Métodos utilizados:** 33

**Archivos core principales:**
- **[TC]** TeoriaCambio, AdvancedDAGValidator
- **[DB]** CausalExtractor, OperationalizationAuditor, CDAFFramework
- **[FV]** PDETMunicipalPlanAnalyzer
- **[CD]** PolicyContradictionDetector

**Métodos clave:**
- `TeoriaCambio.validacion_completa` - Validación completa de teoría de cambio
- `AdvancedDAGValidator.calculate_acyclicity_pvalue` - Cálculo de p-value de aciclicidad
- `CDAFFramework.process_document` - Procesamiento con framework CDAF
- `OperationalizationAuditor.audit_evidence_traceability` - Auditoría de trazabilidad

---

### Q027: Proporcionalidad y Continuidad (Anti-Milagro)
**Executor:** D6Q2_Executor  
**Métodos utilizados:** 38

**Métodos clave:**
- `BayesianMechanismInference._build_transition_matrix` - Construcción de matriz de transición
- `CausalInferenceSetup.classify_goal_dynamics` - Clasificación de dinámicas de objetivos
- `CausalInferenceSetup.identify_failure_points` - Identificación de puntos de falla
- `AdvancedDAGValidator._perform_sensitivity_analysis_internal` - Análisis de sensibilidad

---

### Q028: Inconsistencias (Sistema Bicameral - Ruta 1)
**Executor:** D6Q3_Executor  
**Métodos utilizados:** 22

**Métodos clave:**
- `PolicyContradictionDetector._detect_logical_incompatibilities` - Detección de incompatibilidades lógicas
- `PolicyContradictionDetector.detect` - Método principal de detección
- `PolicyContradictionDetector._detect_semantic_contradictions` - Detección de contradicciones semánticas
- `PolicyContradictionDetector._generate_resolution_recommendations` - Generación de recomendaciones

---

### Q029: Adaptación (Sistema Bicameral - Ruta 2)
**Executor:** D6Q4_Executor  
**Métodos utilizados:** 37

**Métodos clave:**
- `TeoriaCambio._generar_sugerencias_internas` - Generación de sugerencias
- `CDAFFramework._generate_causal_model_json` - Generación de modelo causal en JSON
- `OperationalizationAuditor._perform_counterfactual_budget_check` - Verificación contrafactual de presupuesto
- `PDETMunicipalPlanAnalyzer.generate_recommendations` - Generación de recomendaciones

---

### Q030: Contextualización y Enfoque Diferencial
**Executor:** D6Q5_Executor  
**Métodos utilizados:** 24

**Métodos clave:**
- `SemanticAnalyzer._classify_cross_cutting_themes` - Clasificación de temas transversales
- `SemanticAnalyzer.extract_semantic_cube` - Extracción de cubo semántico
- `PolicyAnalysisEmbedder.compare_policy_interventions` - Comparación de intervenciones
- `AdvancedSemanticChunker._infer_pdq_context` - Inferencia de contexto PDQ

---

## ESTADÍSTICAS GLOBALES

### Por Dimensión

| Dimensión | Preguntas | Métodos Totales | Promedio |
|-----------|-----------|-----------------|----------|
| DIM01     | Q001-Q005 | 82              | 16.4     |
| DIM02     | Q006-Q010 | 95              | 19.0     |
| DIM03     | Q011-Q015 | 103             | 20.6     |
| DIM04     | Q016-Q020 | 92              | 18.4     |
| DIM05     | Q021-Q025 | 87              | 17.4     |
| DIM06     | Q026-Q030 | 154             | 30.8     |
| **TOTAL** | **30**    | **613**         | **20.4** |

### Por Archivo Core

| Archivo | Abrev | Uso en Preguntas | % Cobertura |
|---------|-------|------------------|-------------|
| policy_processor.py | PP | 30/30 | 100% |
| contradiction_deteccion.py | CD | 30/30 | 100% |
| financiero_viabilidad_tablas.py | FV | 18/30 | 60% |
| dereck_beach.py | DB | 13/30 | 43% |
| embedding_policy.py | EP | 21/30 | 70% |
| Analyzer_one.py | A1 | 15/30 | 50% |
| teoria_cambio.py | TC | 12/30 | 40% |
| semantic_chunking_policy.py | SC | 1/30 | 3% |

### Tipos de Métodos

| Tipo           | Cantidad | Porcentaje |
|----------------|----------|------------|
| Extracción     | ~180     | ~29%       |
| Análisis       | ~200     | ~33%       |
| Validación     | ~80      | ~13%       |
| Construcción   | ~70      | ~11%       |
| Clasificación  | ~45      | ~7%        |
| Procesamiento  | ~38      | ~6%        |

---

## HALLAZGOS CLAVE

1. **Archivos Core Universales**: 
   - `policy_processor.py` (PP) y `contradiction_deteccion.py` (CD) se utilizan en TODAS las preguntas
   - Son la base fundamental del sistema de análisis

2. **Especialización por Dimensión**:
   - DIM01-DIM02: Mayor uso de PP, CD, FV
   - DIM03-DIM04: Incorporación de EP y A1
   - DIM05: Uso intensivo de DB (Derek Beach)
   - DIM06: Uso máximo de TC y DB para validaciones complejas

3. **Complejidad Creciente**:
   - Las preguntas de DIM06 tienen el mayor número de métodos (promedio 30.8)
   - Q027 y Q029 son las más complejas con 37-38 métodos cada una

4. **Análisis Causal**:
   - 12 preguntas utilizan análisis causal profundo (Q007, Q010, Q015, Q017, Q022, Q026-Q030)
   - Principales clases: `CausalExtractor`, `TeoriaCambio`, `BayesianMechanismInference`

5. **Validación Financiera**:
   - 8 preguntas incluyen análisis financiero detallado (Q003, Q009, Q012, Q018, Q025, Q026, Q027, Q029)
   - Clases clave: `FinancialAuditor`, `PDETMunicipalPlanAnalyzer`

---

## RECOMENDACIONES

1. **Optimización**:
   - Considerar caché de métodos comunes (PP, CD) para mejorar rendimiento
   - Paralelizar ejecución de métodos independientes en executors complejos

2. **Mantenimiento**:
   - PP y CD son críticos - requieren testing exhaustivo
   - DB y TC son complejos - documentar bien su uso

3. **Extensibilidad**:
   - SC tiene bajo uso (3%) - evaluar si es necesario o expandir su funcionalidad
   - Estandarizar interfaces entre archivos core para facilitar integración

---

**Fin del Reporte**
