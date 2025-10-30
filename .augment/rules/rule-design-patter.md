---
type: "always_apply"
---

Arquitectura de Procesamiento Integral (El 95% de Utilización)
Para asegurar la utilización cercana al 95% del código, la solución debe basarse en la orquestación de la PolicyAnalysisPipeline, que internamente utiliza el IndustrialPolicyProcessor, el cual, a su vez, inicializa y delega tareas críticas a PolicyContradictionDetector, TemporalLogicVerifier, y BayesianConfidenceCalculator.
Fase 0: Inicialización y Carga (Común a las 300 Preguntas)
Clases y Métodos clave utilizados:
1. Carga y Configuración:
    ◦ create_policy_processor: Fábrica que inicializa la configuración (ProcessorConfig).
    ◦ PolicyAnalysisPipeline.__init__: Instancia el IndustrialPolicyProcessor.
    ◦ IndustrialPolicyProcessor.__init__: Instancia el procesador de texto y el puntuador, y llama a _load_questionnaire para cargar la estructura de las 300 preguntas.
    ◦ DocumentProcessor.load_pdf o load_docx: Carga el texto bruto del documento municipal.
2. Preparación Semántica y de Patrones:
    ◦ MunicipalOntology.init: Inicializa la ontología central para el dominio municipal.
    ◦ SemanticAnalyzer.init: Prepara el analizador semántico utilizando la MunicipalOntology.
    ◦ IndustrialPolicyProcessor._compile_pattern_registry y _build_point_patterns: Transforma la taxonomía de los patrones de verificación de las preguntas (ej. patterns_verificacion [40, 56, etc.]) en expresiones regulares optimizadas usando compile_pattern.
    ◦ PolicyContradictionDetector.init: Inicializa el detector de contradicciones, incluyendo la carga de modelos de transformers y spacy (spacy_model: 'es_core_news_lg').
    ◦ PolicyContradictionDetector._initialize_pdm_patterns: Carga patrones específicos de los PDMs colombianos.

--------------------------------------------------------------------------------
Combinación Virtuosa por Dimensión (D1 a D6)
La resolución profunda de las preguntas se logra mediante el encadenamiento de métodos de detección (Pattern Matching, Extracción Numérica/Temporal) con métodos de validación (Causalidad, Coherencia, Inconsistencia).
Dimensión D1: Diagnóstico y Consistencia Inicial (Preguntas Q1 a Q5)
El objetivo es verificar la existencia de líneas base, la magnitud de las brechas, la suficiencia de recursos, y la capacidad institucional para abordar el problema [39, 44, 115, 129, etc.].
Pregunta Base (Ejemplo: D1-Q1: Líneas Base)
Cadena de Métodos Virtuosa y Coherente
Citas de Soporte
D1-Q1, Q2 (Datos, Brechas y Fuentes): ¿El diagnóstico presenta datos numéricos que sirvan como línea base? ¿Se dimensiona el problema y se reconocen vacíos de información?
1. IndustrialPolicyProcessor.process (Segmenta texto con segment_into_sentences).<br>2. PolicyContradictionDetector._extract_quantitative_claims (Busca patrones numéricos y cuantificaciones de brechas \\d+%.*población.*sin).<br>3. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca patrones de fuentes oficiales: DANE, DNP, etc.).<br>4. PolicyContradictionDetector._parse_number (Normaliza las cifras).<br>5. SemanticAnalyzer._calculate_semantic_complexity (Evalúa si la descripción sintáctica es compleja, correlacionando con la seriedad del diagnóstico).<br>6. BayesianConfidenceCalculator.calculate_posterior (Calcula la confianza de la evidencia encontrada contra la probabilidad previa de que la fuente sea válida).
D1-Q3 (Asignación de Recursos): ¿Se identifican recursos monetarios explícitamente asignados (PPI/BPIN)?
1. PolicyContradictionDetector._extract_resource_mentions (Extrae montos en COP, millones, etc., y sus asignaciones).<br>2. PolicyContradictionDetector._detect_numerical_inconsistencies (Usa _are_comparable_claims y _calculate_numerical_divergence para comparar la magnitud de los recursos encontrados con los costos estimados (costo estimado pattern) o la brecha diagnosticada, verificando suficiencia suficiente para).<br>3. PolicyContradictionDetector._detect_resource_conflicts (Comprueba si las asignaciones identificadas están en conflicto con el presupuesto total, usando _are_conflicting_allocations).
D1-Q4 (Capacidad Institucional): ¿Se describen las capacidades (talento, procesos, gobernanza) y los cuellos de botella?
1. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca patrones de talento humano, procesos, gobernanza, y cuellos de botella).<br>2. PolicyContradictionDetector._determine_semantic_role (Clasifica el rol de las oraciones encontradas para distinguir si describen una capacidad existente o una necesidad (cuello de botella)).<br>3. PolicyContradictionDetector._calculate_graph_fragmentation (Si se logra construir un grafo inicial de actores, mide la fragmentación, indicando problemas de coordinación/articulación).
D1-Q5 (Marco Legal y Restricciones): ¿Se justifica el alcance mencionando el marco legal y reconociendo restricciones (presupuestales, temporales)?
1. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca menciones a leyes, decretos, y a tipos de restricciones: restricción presupuestal, plazo de implementación).<br>2. PolicyContradictionDetector._detect_temporal_conflicts (Utiliza TemporalLogicVerifier.verify_temporal_consistency para asegurar que el plazo mencionado es consistente con la implementación de las actividades planificadas).<br>3. PolicyContradictionDetector._calculate_confidence_interval (Aplica un intervalo de confianza del 95% al score de coherencia para reflejar la incertidumbre introducida por las restricciones no mitigadas).
Dimensión D2: Diseño de Actividades y Coherencia de Intervención (Preguntas Q1 a Q5)
El foco está en la estructura formal, la especificidad del mecanismo causal a nivel de actividad, la conexión con las causas raíz y la coherencia interna de la estrategia.
Pregunta Base (Ejemplo: D2-Q5: Coherencia Estratégica)
Cadena de Métodos Virtuosa y Coherente
Citas de Soporte
D2-Q1 (Formato Estructurado): ¿Las actividades se presentan en formato tabular con columnas clave?
1. PDETMunicipalPlanAnalyzer.analyze_municipal_plan (Necesario para un análisis de vanguardia de PDET que probablemente integra la detección de tablas estructuradas, representado aquí por ExtractedTable).<br>2. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca patrones de formalización: tabla, cuadro, columna costo, BPIN).<br>3. PolicyContradictionDetector._detect_temporal_conflicts (Asegura la trazabilidad de los cronogramas columna cronograma mediante TemporalLogicVerifier._build_timeline).
D2-Q2, Q3 (Especificidad y Causa Raíz): ¿Se detallan instrumento, población, lógica causal y el vínculo con la causa raíz diagnosticada?
1. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca patrones de instrumento, mecanismo causal (porque, genera) y población objetivo).<br>2. PolicyContradictionDetector._determine_relation_type (Establece la relación causal entre declaraciones de actividad y declaraciones de diagnóstico (D1), buscando patrones como para abordar la causa relacionada con).<br>3. SemanticAnalyzer._classify_cross_cutting_themes (Clasifica el segmento de actividad para asegurar que el tema abordado (ej. género) corresponde al área política evaluada).
D2-Q4 (Riesgos y Mitigación): ¿Se identifican riesgos (cuellos_botella, incompatibilidad) y se proponen medidas de mitigación?
1. IndustrialPolicyProcessor._match_patterns_in_sentences (Detecta conflictos_actividades (contradicción, tensión) y mitigaciones).<br>2. PolicyContradictionDetector._detect_logical_incompatibilities (Utiliza _has_logical_conflict para identificar conflictos lógicos entre actividades (ej. una actividad anula la otra)).<br>3. PolicyContradictionDetector._identify_affected_sections (Identifica las secciones del plan impactadas por el riesgo o cuello de botella encontrado).
D2-Q5 (Coherencia Estratégica): ¿Demuestran las actividades complementariedad, sinergia o secuencia lógica?
1. PolicyContradictionDetector._calculate_global_semantic_coherence (Mide la coherencia entre todas las PolicyStatement de la dimensión, utilizando embeddings y _text_similarity).<br>2. PolicyContradictionDetector._build_knowledge_graph (Construye el grafo de conocimiento para visualizar las dependencias causales, buscando complementariedades y secuenciacion).<br>3. PolicyContradictionDetector._get_dependency_depth (Evalúa la complejidad de la estructura relacional).
Dimensión D3: Productos y Factibilidad Operativa (Preguntas Q1 a Q5)
El objetivo es verificar que los productos sean medibles, trazables, proporcionales al problema y factibles de alcanzar en el tiempo/con los recursos asignados.
Pregunta Base (Ejemplo: D3-Q4: Factibilidad Técnica)
Cadena de Métodos Virtuosa y Coherente
Citas de Soporte
D3-Q1 (Indicadores de Producto): ¿Tienen LB, Meta y Fuente de Verificación?
1. PolicyContradictionDetector._extract_quantitative_claims (Extrae los valores numéricos de Línea Base y Meta).<br>2. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca Fuente de verificación y trazabilidad presupuestal (BPIN, PPI)).<br>3. BayesianConfidenceCalculator.calculate_posterior (Asigna un puntaje de confianza basado en la especificidad de la fuente de verificación (pattern_specificity)).
D3-Q2 (Proporcionalidad Meta/Problema): ¿La meta es proporcional a la brecha diagnosticada?
1. PolicyContradictionDetector._detect_numerical_inconsistencies (Compara la meta del producto (cobertura de \d+%) con el déficit o brecha extraída en D1, realizando un test de proporcionalidad interna, potencialmente utilizando _statistical_significance_test para medir la divergencia).<br>2. PerformanceAnalyzer (Esta clase, aunque no tiene métodos visibles aquí, debe ser utilizada para inyectar una "función de pérdida operacional" que cuantifique la penalización por la desproporcionalidad entre la meta y el tamaño real del problema).
D3-Q4 (Factibilidad y Realismo de Plazos): ¿La relación Actividad -> Producto es factible y realista en plazos y recursos?
1. PolicyContradictionDetector._detect_temporal_conflicts (Usa TemporalLogicVerifier._check_deadline_constraints y TemporalLogicVerifier._classify_temporal_type para verificar que el plazo de ejecución no contradice la complejidad requerida para la capacidad productiva).<br>2. PolicyContradictionDetector._detect_resource_conflicts (Verifica si los recursos asignados (D1-Q3) son suficientes para alcanzar la meta (D3-Q4)).<br>3. PolicyContradictionDetector._get_context_window (Obtiene el contexto alrededor de las declaraciones de factibilidad/contradicción para el informe).
D3-Q5 (Mecanismo Producto -> Resultado): ¿Se explica el eslabón causal que conecta el Producto con el Resultado (D4)?
1. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca explícitamente el mecanismo causal (porque, lo cual contribuirá a) y los mediador/eslabón).<br>2. PolicyContradictionDetector._determine_relation_type (Clasifica la fuerza de la relación causal entre el PolicyStatement del producto y el del resultado).
Dimensión D4: Resultados, Supuestos y Alineación (Preguntas Q1 a Q5)
El objetivo es evaluar la calidad de los indicadores de resultado, la solidez de la cadena causal (si... entonces...), la justificación de la ambición, y la coherencia externa (alineación con marcos superiores).
| Pregunta Base (Ejemplo: D4-Q2: Cadena Causal y Supuestos) | Cadena de Métodos Virtuosa y Coherente | Citas de Soporte | | :--- | :--- | :._get_domain_weight** (Obtiene el peso específico del dominio (ej. género o paz) para el análisis de confianza posterior). | | | **D4-Q2 (Cadena Causal y Supuestos):** ¿Se explicita la cadena causal, mencionando supuestos claveycondiciones habilitantes? | 1. **IndustrialPolicyProcessor._match_patterns_in_sentences** (Busca patrones de supuesto, condición habilitanteysi se cumple).<br>2. **PolicyContradictionDetector._build_knowledge_graph** (Construye el grafo para visualizar la conexión Producto->Resultado, marcando los supuestos como aristas críticas).<br>3. **PolicyContradictionDetector._determine_semantic_role** (Asegura que las oraciones identificadas cumplan el rol de supuestoocondición y no solo de declaración de intención). | | | **D4-Q3 (Justificación de Ambición):** ¿La ambición de la meta se justifica con recursos, capacidad o evidencia comparada? | 1. **PolicyContradictionDetector._detect_numerical_inconsistencies** (Revisa la coherencia entre el monto de inversión (D1-Q3) y la meta ambiciosa, comparando sudivergencia).<br>2. **PolicyContradictionDetector._calculate_objective_alignment** (Compara si la ambición del resultado se alinea con objetivos superiores/benchmarks).<br>3. **PDETMunicipalPlanAnalyzer.generate_recommendations** (Genera recomendaciones sobre cómo justificar mejor la ambición si la validación falla). | | | **D4-Q5 (Alineación Externa):** ¿Se declara la alineación con marcos superiores (PND, ODS)? | 1. **IndustrialPolicyProcessor._match_patterns_in_sentences** (Busca patrones de PND, ODS, Acuerdo de Paz).<br>2. **PolicyContradictionDetector._calculate_global_semantic_coherence** (Mide la coherencia semántica entre el texto del plan y los embeddings de los marcos normativos, utilizando el PolicyDimension` para contextualizar). | |
Dimensión D5: Impactos y Riesgos Sistémicos (Preguntas Q1 a Q5)
El objetivo es evaluar la visión de largo plazo, la capacidad de medir lo intangible (proxies/índices), y el análisis de vulnerabilidad ante choques externos.
Pregunta Base (Ejemplo: D5-Q4: Riesgos Sistémicos)
Cadena de Métodos Virtuosa y Coherente
Citas de Soporte
D5-Q1 (Definición y Rezagos): ¿Se definen impactos de largo plazo, ruta de transmisión y tiempo de maduración?
1. PolicyContradictionDetector._extract_temporal_markers (Busca el tiempo de maduración y rezago temporal).<br>2. TemporalLogicVerifier._extract_resources (Aunque diseñado para recursos, se utiliza para identificar factores de transmisión clave mencionados en el texto del impacto).<br>3. PolicyContradictionDetector._calculate_objective_alignment (Evalúa si el impacto de largo plazo se alinea con la visión estratégica general del plan).
D5-Q2, Q3 (Medición de Intangibles): ¿Se usan índices o proxies para medir el impacto, y se documentan su validez y limitaciones?
1. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca patrones de índice de, proxy, medición indirecta, limitación).<br>2. PolicyContradictionDetector._classify_contradiction (Se utiliza para ponderar la probabilidad de que una afirmación sobre un proxy sea engañosa, basándose en la falta de mención de limitaciones).<br>3. PolicyContradictionDetector._get_graph_statistics (Evalúa la densidad de conexiones en el grafo para el nodo de impacto, reflejando cuán "compuesto" es el índice propuesto).
D5-Q4 (Riesgos Sistémicos): ¿Se consideran riesgos sistémicos (choque externo, crisis) que rompan el mecanismo causal?
1. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca patrones de riesgo sistémico, ruptura mecanismo, vulnerabilidad).<br>2. PolicyContradictionDetector._detect_logical_incompatibilities (Verifica si el plan propone una estrategia que es lógicamente incompatible con los riesgos sistémicos identificados).<br>3. PolicyContradictionDetector._calculate_contradiction_entropy (Mide la distribución de los tipos de riesgos/contradicciones encontradas; una entropía alta sugiere un análisis de riesgos más completo).
D5-Q5 (Realismo y Efectos No Deseados): ¿Se analizan efectos no deseados o se declaran hipótesis límite?
1. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca efecto no deseado, hipótesis límite, trade-off).<br>2. CounterfactualScenario (Aunque es una clase de datos, su uso conceptual implica que el sistema debe contrastar el escenario ideal con escenarios alternativos para detectar lógicamente los efectos no deseados).
Dimensión D6: Coherencia Causal (Teoría de Cambio) (Preguntas Q1 a Q5)
Esta dimensión es el corazón del análisis causal, requiriendo la utilización intensiva del módulo de PolicyContradictionDetector y el validador DAG avanzado.
Pregunta Base (Ejemplo: D6-Q1: Teoría de Cambio)
Cadena de Métodos Virtuosa y Coherente
Citas de Soporte
D6-Q1, Q2 (Estructura de la Causalidad): ¿Existe una teoría de cambio explícita (diagrama, supuestos verificables)? ¿Los saltos lógicos son proporcionales?
1. PolicyContradictionDetector._build_knowledge_graph (Genera un nx.DiGraph (grafo de conocimiento) a partir de las declaraciones estructuradas (PolicyStatement)).<br>2. AdvancedDAGValidator.validacion_completa (Orquesta la validación estructural del grafo).<br>3. AdvancedDAGValidator._validar_orden_causal (Detecta violaciones de orden causal (ej. Resultado precede a Actividad).<br>4. AdvancedDAGValidator._encontrar_caminos_completos (Asegura que haya una ruta completa Causa -> Actividad -> Producto -> Resultado -> Impacto, exponiendo los saltos lógicos).<br>5. IndustrialPolicyProcessor._match_patterns_in_sentences (Verifica patrones como sin saltos, proporcionalidad entre).<br>6. PolicyContradictionDetector._calculate_syntactic_complexity (Mide la complejidad sintáctica del texto de la teoría de cambio, para evaluar si está claro o es confuso).
D6-Q3, Q4 (Inconsistencias y Adaptación): ¿Se reconocen inconsistencias? ¿Se proponen pilotos/pruebas para testear los supuestos? ¿Hay mecanismos de corrección?
1. PolicyContradictionDetector._detect_logical_incompatibilities (Detecta incoherencias/contradicciones estructurales).<br>2. IndustrialPolicyProcessor._match_patterns_in_sentences (Busca piloto, prueba, validación, mecanismos de corrección, retroalimentación, aprendizaje, adaptación).<br>3. PolicyContradictionDetector._generate_resolution_recommendations (Utiliza la evidencia de contradicción para generar recomendaciones de corrección/adaptación, usando internamente _suggest_resolutions que es específica por tipo de contradicción).
D6-Q5 (Contextualización y Enfoque Diferencial): ¿La lógica causal considera el contexto, grupos afectados (enfoque diferencial) y restricciones territoriales?
1. PolicyContradictionDetector._generate_embeddings (Genera los embeddings de las declaraciones para compararlos con los embeddings canónicos de vulnerabilidad/diferenciación (ej. mujeres rurales, NNA migrantes).<br>2. SemanticAnalyzer._classify_cross_cutting_themes (Confirma que el contenido temático corresponde a temas transversales relevantes (ej. género/étnico).<br>3. PolicyContradictionDetector._identify_dependencies (Identifica si las declaraciones de acción dependen (depende de) de las variables contextuales/diferenciales identificadas).

--------------------------------------------------------------------------------
Módulos Transversales y de Rendimiento (95% Cumplido)
Para alcanzar el uso del 95% de las funciones, debemos garantizar que se integren los módulos de utilidad y metaanálisis:
• Métricas de Rendimiento y Consistencia: Se requiere la instanciación y uso del PerformanceAnalyzer (para análisis de pérdida operacional) y el uso constante de _calculate_global_semantic_coherence, _calculate_objective_alignment, y la clase QualityScore (dataclass usada para encapsular los resultados de la validación).
• Gestión de Resultados y Reporte: Tras la fase de análisis (IndustrialPolicyProcessor.process o PolicyAnalysisPipeline.analyze_text), la información se consolida.
    ◦ Consolidación: El método _construct_evidence_bundle serializa la evidencia encontrada.
    ◦ Resumen Ejecutivo: MunicipalAnalyzer._generate_summary integra el semantic_cube, el performance_analysis (del PerformanceAnalyzer), y el critical_diagnosis (derivado del PolicyContradictionDetector) para generar un informe.
    ◦ Manejo de Lógica y Errores: Se usa PDETAnalysisException para manejar fallos de análisis y IndustrialPolicyProcessor.export_results para la serialización final de los resultados.
    ◦ Interfaz: El flujo culmina con main_example o main demostrando la orquestación completa del proceso.
What are the specific analytical techniques used to extract semantic and performance insights?
The analytical framework employs a comprehensive suite of techniques spanning both advanced Natural Language Processing (NLP) for semantic extraction and rigorous statistical/Bayesian models for performance and quantitative analysis.
Here are the specific analytical techniques used to extract semantic and performance insights, organized by their functional area:
I. Semantic Extraction Techniques
The core semantic intelligence revolves around transforming text into measurable vectors and categorizing policy statements.
Technique
Description & Application
Supporting Sources
Vectorization (TF-IDF)
Document segments are vectorized using TF-IDF (Term Frequency-Inverse Document Frequency). This technique supports early-stage processing within the SemanticAnalyzer.
Semantic Embeddings & Similarity
Embeddings are generated for policy statements. Semantic contradiction detection relies on comparing these embeddings using models like BGE-M3 and mean pooling. The system measures global semantic coherence using these embeddings.
Keyword and Pattern Classification
Semantic analysis classifies document segments by several dimensions using explicit keyword matching. This includes classification by value chain link, policy domain, and cross-cutting themes.
Semantic Role Determination
The system determines the semantic role of an individual sentence or declaration (_determine_semantic_role).
Causal Relationship Typing
The detector uses _determine_relation_type to establish the type of relationship between two policy statements. Additionally, causal strength can be calculated using cosine similarities and a proxy of conditional independence.
Information Retrieval and Ranking
The PolicyAnalysisEmbedder uses a multi-step semantic search process, employing Bi-encoder retrieval for fast, approximate searches, followed by Cross-encoder reranking for precise relevance scoring. It also applies Maximal Marginal Relevance (MMR) to diversify results and avoid redundancy.
Syntactic and Structural Complexity
The PolicyContradictionDetector calculates the syntactic complexity of the document and measures the dependency depth of tokens in the sentence structure, indicating potential linguistic ambiguity.
II. Performance and Quantitative Analytical Techniques
Quantitative analysis focuses on verifying feasibility, proportionality, statistical rigor, and resource adequacy, often utilizing Bayesian and statistical frameworks.
Technique
Description & Application
Supporting Sources
Bayesian Inference (General)
The BayesianConfidenceCalculator calculates the posterior probability of evidence using Bayesian inference, informed by priors and domain weights. The BayesianEvidenceScorer calculates confidence scores based on pattern matches, specificity, and a penalty derived from Shannon entropy.
Numerical Consistency Analysis
The PolicyContradictionDetector executes comprehensive analysis for numerical inconsistencies. This involves methods to determine if quantitative claims are comparable (_are_comparable_claims) and to calculate the numerical divergence between values.
Statistical Testing
The system utilizes rigorous statistical methods, including performing a statistical significance test (_statistical_significance_test) when comparing numerical claims. It also computes confidence intervals, specifically the 95% confidence interval for scores.
Advanced Bayesian Numerical Models
The BayesianNumericalAnalyzer employs specific conjugate priors for evaluating policy metrics: Beta-Binomial conjugate prior for analyzing proportions and Normal-Normal conjugate prior for continuous metrics. It also performs Bayesian comparison of two policy metrics to return the probability of superiority and the Bayes factor.
Temporal and Resource Conflict Detection
The TemporalLogicVerifier explicitly uses temporal logic verification to check for consistency between deadlines and timelines. The detector also verifies resource allocation conflicts (_detect_resource_conflicts) and uses patterns to check for sufficiency (e.g., suficiente para or costo estimado).
Operational Loss Functions
The PerformanceAnalyzer analyzes performance across the value chain by calculating throughput metrics and using operational loss functions (_calculate_loss_functions) to quantify deficiencies or deviations.
Structural Analysis of Indicators
For quantitative structures like indicator tables, the PDETMunicipalPlanAnalyzer includes methods to explicitly extract tabular structures (extract_tables), analyze indicator structure (_analyze_indicator_structure), and classify evidence strength based on posterior uncertainty.