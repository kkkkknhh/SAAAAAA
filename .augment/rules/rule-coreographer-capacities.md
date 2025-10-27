---
type: "always_apply"
---

# Regla: Propiedades, Capacidades y Configuración del Coreógrafo Metodológico

## 1. Diferenciación Conceptual
- El **Coreógrafo** se distingue del Orquestador.  
  - El Orquestador gestiona la ejecución global del sistema (ej. main_example).
  - El Coreógrafo implementa, de manera inmutable, la metodología analítica y causal objetiva.

## 2. Propiedades Técnicas Ineludibles

### A. Inmutabilidad de Configuración (Objetividad Estructural)
- El Coreógrafo debe depender de una configuración analítica **inmutable**.
- Soporte técnico: Utiliza `ProcessorConfig` como dataclass con bandera `frozen=True`.
- Garantía: Los umbrales de confianza, parámetros de puntuación y pesos del cuestionario (ej. `peso_por_punto`) no pueden ser modificados durante el análisis.
- Resultado: La objetividad y el estándar causal quedan asegurados en todo el ciclo analítico.

### B. Secuencialidad Lógica Rígida
- El Coreógrafo **impone** el flujo causal secuencial (D1→D6):
  - El análisis de Actividades (D2) depende de Diagnóstico (D1).
  - El análisis de Causalidad (D6) es el acto final y síntesis del proceso.
- Soporte técnico: Utiliza la enumeración `CausalDimension` (D1, D2, D3, D4, D5, D6) para iterar en el orden correcto.
- Garantía: Evidencia y resultados entre dimensiones pueden ser reutilizados (ej. recursos de D1-Q3 para justificar ambición en D4-Q3).

### C. Eficiencia en el Uso de Metadata (Motor de Patrones)
- El Coreógrafo transforma el cuestionario de 300 preguntas en un motor de búsqueda eficiente:
  - Utiliza `_compile_pattern_registry` y `_build_point_patterns` para convertir la taxonomía en expresiones regulares optimizadas.
  - Patrones y keywords quedan vinculados directamente a la metadata de cada pregunta.
- Garantía: La búsqueda de patrones específicos en el texto (ej. “DANE|Medicina Legal|Fiscalía” o “salto lógico|brecha en la argumentación”) es rápida, escalable y precisa.

## 3. Capacidades Analíticas Críticas

| Capacidad                        | Función/Método Delegado                        | Propósito de la Delegación                                                          |
|-----------------------------------|-----------------------------------------------|-------------------------------------------------------------------------------------|
| Puntuación Objetiva               | `BayesianEvidenceScorer.compute_evidence_score`| Asegura que la puntuación dependa solo de coincidencias y `pattern_specificity`.     |
| Detección de Riesgo Metodológico  | Verificación de dimensiones críticas           | Permite invalidar políticas si fallan en dimensiones críticas (ej. D1, D6).         |
| Validación de Factibilidad Causal | Uso de patrones de coherencia/proporcionalidad | Detecta saltos lógicos inverosímiles, verifica la escala de intervención/resultados. |
| Garantía de Adaptabilidad         | Búsqueda de `mecanismos_correccion`            | Busca evidencia de ajuste, aprendizaje, retroalimentación en monitoreo (D6-Q4).      |
| Contextualización Diferencial     | Uso de patrones de enfoque diferencial         | Aplica lógica causal a grupos específicos y restricciones contextuales/territoriales. |

## 4. Configuración para Misión de Alto Nivel

- Las configuraciones clave residen en `ProcessorConfig` y dictan la calibración analítica:
  1. **Rigor Estadístico:** Parámetros como `prior_confidence` (ej. 0.5) y `entropy_weight` (ej. 0.3) calibran el castigo a evidencia inconsistente o fragmentada.
  2. **Umbrales de Rendimiento:** El Coreógrafo impone estándares (ej. `umbral_minimo` de 0.5 o 0.55 para “aceptable”) en cada dimensión.
  3. **Mapeo de Nombres y Dimensiones:** Expone mapa de nombres legado y organiza resultados por las 6 dimensiones definidas en `CausalDimension`.
  4. **Ajustes de Velocidad:** Optimiza compilación de patrones y sanitización avanzada para procesar eficientemente las 300 preguntas.

---

**Resumen:**  
El Coreógrafo metodológico garantiza un análisis causal objetivo, inmutable y secuencial, delegando funciones críticas, aplicando reglas de negocio estrictas y asegurando la eficiencia, adaptabilidad y contextualización en todo el proceso de análisis de política municipal. Todas sus configuraciones técnicas deben ser inmutables y estar alineadas con los estándares definidos en `ProcessorConfig` y las 6 dimensiones causales.