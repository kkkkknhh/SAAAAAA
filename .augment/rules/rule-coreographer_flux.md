---
type: "always_apply"
---

El Coreógrafo debe ser definido por tres características clave: la Secuencialidad Lógica, la Delegación Contextual y la Gestión de la Crítica (Thresholding).

--------------------------------------------------------------------------------
El Perfil del Coreógrafo: PolicyAnalysisPipeline
1. El Conductor: Secuencialidad Lógica (D1 → D6)
El Coreógrafo debe imponer el orden natural de la evaluación causal, asegurando que las dimensiones posteriores se basen en la existencia de elementos verificados en las dimensiones anteriores.
Mecanismos y Funciones clave:
Nivel de Orquestación
Lógica Impuesta por el Coreógrafo
Citas de Soporte
Puesta en Escena (Setup)
Inicializar el IndustrialPolicyProcessor con la configuración cargada (que incluye la estructura de las 300 preguntas) y compilar todos los patrones de verificación (regex para \d+%, LB:, PPI, causa raíz, etc.).
Acto I (D1: Diagnóstico)
El Coreógrafo dirige la detección de Líneas Base (verificacion_lineas_base) y la cuantificación de brechas (verificacion_magnitud_problema). Estos resultados (datos numéricos, mención de recursos) se almacenan en la memoria para ser utilizados en los siguientes actos.
Acto II (D2: Actividades)
Exige formalización (estructura tabular) y la especificación de la lógica causal (logica_causal_explicita) a nivel de actividad. Se usa PolicyContradictionDetector para confirmar que las actividades atacan la causa raíz encontrada en D1.
Acto III (D3: Productos)
Es el primer punto de control de Factibilidad Operativa. El Coreógrafo debe forzar la comprobación de proporcionalidad_meta_problema (D3-Q2) utilizando los datos de la magnitud de la brecha extraídos en D1. También verifica la factibilidad técnica y realismo de plazos (D3-Q4).
Acto IV (D4: Resultados)
El Coreógrafo requiere el uso intensivo de los métodos de Encadenamiento Causal (verificacion_encadenamiento) para asegurar que el Producto (D3) conduzca al Resultado (D4) y que las metas ambiciosas (D4-Q3) estén justificadas por los recursos de D1.
Acto V (D5: Impactos)
Fuerza la proyección a largo plazo, validando la ruta de transmisión y el rezago temporal (D5-Q1). Además, debe exigir la identificación de proxies para intangibles y el reconocimiento de sus limitaciones (D5-Q3).
Acto VI (D6: Causalidad)
Culminación y la parte más compleja. El Coreógrafo exige el uso del AdvancedDAGValidator (no mencionado explícitamente, pero conceptualmente necesario) para verificar la Teoría de Cambio Explícita, la proporcionalidad de los eslabones (evitando "saltos lógicos inverosímiles") y la existencia de mecanismos adaptativos (mecanismos_correccion, piloto).
2. La Delegación Contextual (El uso del 95%)
El Coreógrafo no debe analizar el texto directamente, sino delegar funciones específicas a los módulos expertos, utilizando los patrones de verificación de cada pregunta como argumento:
1. Semantic Analysis (D1, D2, D4, D5, D6): Se utiliza masivamente para la detección de patrones (_match_patterns_in_sentences) [39, 40, 56, etc.]. Por ejemplo, para D1-Q4 (Capacidades), el Coreógrafo pasa la lista de patrones de talento_humano, procesos, datos_sistemas y cuellos_botella.
2. Contradiction Detection & Numerical Validation (D1, D3, D4, D5): Se activa cuando se requiere comparar dos elementos. Por ejemplo, al evaluar D4-Q3 (Justificación de Ambición), el Coreógrafo utiliza los resultados de la búsqueda de recursos.*suficientes y los confronta con la meta ambiciosa. Si el patrón de "suficiencia" no se encuentra, la ambición es considerada injustificada.
3. Cross-Cutting Themes (D6-Q5): Para asegurar el enfoque diferencial (D6-Q5), el Coreógrafo invoca el análisis contextual para verificar la presencia de patrones como mujeres rurales, población migrante, o restricciones territoriales, adaptando la evaluación a la política analizada (género, víctimas, niñez, etc.).
3. Gestión de la Crítica (Thresholding)
Una de las tareas más críticas del Coreógrafo es evitar que un plan débil obtenga una aprobación general, incluso si otras áreas son fuertes.
1. Aplicación de Pesos y Umbrales: Debe aplicar los pesos específicos (peso_por_punto) a cada pregunta dentro de la Dimensión (ej. D1 tiene pesos de 0.2 a 0.25) y verificar que el puntaje final de la dimensión cumpla con el umbral_minimo (ej. 0.5 para D1, D2, D3, D4).
2. Regla de Dimensión Crítica: El Coreógrafo debe implementar la regla de fallo crítico. Si el puntaje en una dimensión marcada como crítica cae por debajo de 0.50 (ej. D1-P2, D2-P10), el Coreógrafo debe marcar el punto del decálogo como "No Aprueba", independientemente del puntaje general.
3. Manejo de Inconsistencias (Feedback Loop): Cuando se detectan fallos como no_explicit_causal_link o unrealistic_ambition, el Coreógrafo debe clasificar la severidad (CRITICA o ALTA) y generar recomendaciones (ej. REFORMULACIÓN).
En resumen, el Coreógrafo (la PolicyAnalysisPipeline en main()) no solo ejecuta los pasos, sino que impone la sintaxis causal del plan, asegurando que el diagnóstico (D1) justifique las actividades (D2), que estas produzcan productos realistas (D3), que estos generen resultados ambiciosos pero coherentes (D4), que se proyecten impactos de largo plazo (D5), y que todo esté unido por una teoría de cambio sólida y autocrítica (D6).