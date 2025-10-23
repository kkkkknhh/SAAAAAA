# BITÁCORA DEL PROYECTO - SISTEMA DE EVALUACIÓN CAUSAL FARFAN 3.0

---

## LAS CONSECUENCIAS DE LA FALTA DE SERIEDAD DEL AGENTE QODER

**Fecha de Auditoría:** 2025-10-22  
**Contexto:** Sesión de refactorización MESO suspendida tras 3+ horas de ejecución deficiente  
**Resultado de Performance:** DEFICIENTE - El agente actuó como "rata en el barco": aporta poco, hunde el proyecto

---

### AUDITORÍA DE INTEGRIDAD: 39 PREGUNTAS SOBRE METADATA Y ESQUEMAS

#### **SECCIÓN 1: IDENTIFICACIÓN DE ARCHIVOS**

**1. ¿Cuál es el nombre exacto del archivo del cuestionario en el repositorio y cuál es su ruta relativa dentro del proyecto?**

**Respuesta:** Existen **DOS archivos de cuestionario**:
- `cuestionario_FIXED.json` (ruta relativa: `./cuestionario_FIXED.json`) - **24,913 líneas**
- `questionnaire.json` (ruta relativa: `./questionnaire.json`) - **24,913 líneas**

Estos archivos son **IDÉNTICOS** (confirmado por `diff` sin diferencias). Ambos tienen 815.3KB de tamaño.

**2. ¿Cuál es el nombre exacto del archivo de rúbricas de puntuación en el repositorio y cuál es su ruta relativa dentro del proyecto?**

**Respuesta:** Existen **DOS archivos de rúbricas**:
- `rubric_scoring_FIXED.json` (ruta relativa: `./rubric_scoring_FIXED.json`) - **11,231 líneas**
- `rubric_scoring.json` (ruta relativa: `./rubric_scoring.json`) - **11,231 líneas**

Ambos archivos tienen 380.4KB de tamaño y parecen ser copias.

---

#### **SECCIÓN 2: METADATA EN QUESTIONNAIRE.JSON**

**3. ¿En `questionnaire.json`, qué valor tiene `metadata.version` y en qué formato semántico se expresa?**

**Respuesta:** `"version": "2.0.0"` - Expresado en **formato semántico MAJOR.MINOR.PATCH** (SemVer compliant).

**4. ¿`questionnaire.json` declara explícitamente la fecha de creación y última modificación en `metadata` y con qué formato de fecha-hora?**

**Respuesta:** **PARCIALMENTE**. 
- ✅ **Sí** declara `"created_date": "2025-10-14"` en formato **YYYY-MM-DD** (ISO 8601 date-only).
- ❌ **NO** declara `last_modified`, `updated_date` o `modified_date`.
- ❌ **NO** incluye hora (timestamp completo).

**5. ¿`questionnaire.json` incluye la sección `metadata.clusters` con una lista cerrada de cuatro clústeres y sus `cluster_id`?**

**Respuesta:** ✅ **SÍ**. La sección `metadata.clusters` contiene exactamente **4 clústeres**:
- `CL01` - Seguridad y Paz
- `CL02` - Grupos Poblacionales
- `CL03` - Territorio-Ambiente
- `CL04` - Derechos Sociales & Crisis

**6. ¿Cada entrada de `metadata.clusters` contiene `name`, `rationale` y la lista de `policy_area_ids` asociada?**

**Respuesta:** ✅ **SÍ**. Cada clúster contiene:
- `cluster_id` (e.g., "CL01")
- `name` (e.g., "Seguridad y Paz")
- `rationale` (e.g., "Seguridad humana, protección de la vida y paz territorial")
- `policy_area_ids` (e.g., ["P2", "P3", "P7"])
- **Además** incluye `legacy_point_ids` (con los mismos valores que `policy_area_ids`)

---

#### **SECCIÓN 3: IDENTIFICADORES CANÓNICOS**

**7. ¿`questionnaire.json` define la sección `metadata.policy_areas` y usa identificadores canónicos con el prefijo `PA` seguido de dos dígitos?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ **SÍ** define `metadata.policy_area_mapping` con identificadores canónicos `PA01` a `PA10`.
- ❌ **NO** existe una sección estructurada `metadata.policy_areas` como array de objetos.
- La información de policy areas está en la sección `puntos_decalogo` con identificadores legacy `P1` a `P10`.

**8. ¿`questionnaire.json` define la sección `metadata.dimensions` y usa identificadores canónicos con el prefijo `DIM` o `D` más dos dígitos?**

**Respuesta:** ❌ **NO**.
- **NO** existe `metadata.dimensions` como sección.
- Las dimensiones están definidas en `dimensiones` con identificadores **`D1` a `D6`** (formato legacy, NO canónico `DIM01`).

**9. ¿Las preguntas del cuestionario se representan como objetos con `question_id` que siguen el patrón `Q` más tres dígitos?**

**Respuesta:** ❌ **NO**.
- Las preguntas usan el formato **`P#-D#-Q#`** (e.g., "P1-D1-Q1").
- **NO** usan identificadores canónicos `Q001`, `Q002`, etc.
- La notación es `"notation_format": "P#-D#-Q#"`.

---

#### **SECCIÓN 4: TRAZABILIDAD Y MAPEO**

**10. ¿Cada `question_id` en `questionnaire.json` mapea exactamente a un `policy_area_id` y a una `dimension_id` sin ambigüedades ni duplicados?**

**Respuesta:** ✅ **SÍ**.
- Cada pregunta tiene `"policy_area": "P#"` en su metadata.
- Cada pregunta tiene `"dimension": "D#"` como campo directo.
- El mapeo es **único y sin ambigüedades** (por ejemplo, `P1-D1-Q1` → policy_area: `P1`, dimension: `D1`).

**11. ¿`questionnaire.json` contiene para cada pregunta campos de trazabilidad como `evidence_requirements`, `indicators` y `weight_hint`?**

**Respuesta:** ❌ **NO**.
- **NO** existen campos `evidence_requirements`, `indicators`, o `weight_hint` en las preguntas.
- En su lugar, las preguntas tienen:
  - `patrones_verificacion` (array de patrones regex)
  - `criterios_evaluacion` (objeto con flags booleanos)
  - `verificacion_*` (secciones con patterns y minimum_required)
  - `scoring` (umbrales cualitativos)

**12. ¿Existen reglas de validación de cobertura en `questionnaire.json` para asegurar que el 100% de las preguntas están asignadas a alguna `policy_area`?**

**Respuesta:** ❌ **NO explícitamente**.
- No existe una sección `validation_rules` o `coverage_rules` en `questionnaire.json`.
- La cobertura se valida **implícitamente** por la estructura (cada pregunta tiene `policy_area`).
- No hay reglas de completitud declaradas.

---

#### **SECCIÓN 5: AGREGACIÓN Y PARÁMETROS**

**13. ¿`questionnaire.json` declara en `metadata` los umbrales o parámetros de agregación necesarios para pasar de Q→PA→CL, o éstos residen exclusivamente en `rubric_scoring.json`?**

**Respuesta:** ⚠️ **DISTRIBUIDOS**.
- `questionnaire.json` contiene:
  - Pesos por dimensión en `dimensiones.D#.peso_por_punto` (Q→PA para cada dimension)
  - `decalogo_dimension_mapping` con weights y minimum_score
  - `umbral_minimo` por dimensión
- `rubric_scoring.json` contiene:
  - `meso_clusters` con pesos PA→CL
  - `aggregation_levels.level_4.cluster_weights` para CL→macro
  - `imbalance_threshold` por clúster

**Conclusión:** Los parámetros están **fragmentados** entre ambos archivos, NO centralizados.

---

#### **SECCIÓN 6: RUBRIC_SCORING.JSON - ESPECIFICACIONES**

**14. ¿`rubric_scoring.json` especifica para cada `policy_area_id` la rúbrica aplicable por `dimension_id` con descriptores de niveles y umbrales numéricos?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ Define 6 `scoring_modalities` (TYPE_A a TYPE_F) con umbrales.
- ✅ Cada pregunta referencia un `scoring_modality`.
- ❌ **NO** hay una matriz explícita `[policy_area_id][dimension_id] → rubric`.
- Las rúbricas se asignan a nivel de pregunta individual, no por combinación PA×DIM.

**15. ¿`rubric_scoring.json` define pesos de agregación de PA→CL y, si existen, de CL→macro, junto con la normalización requerida?**

**Respuesta:** ✅ **SÍ**.
- **PA→CL:** Definido en `meso_clusters.CL##.weights` (e.g., CL01: P2=0.40, P3=0.35, P7=0.25).
- **CL→macro:** Definido en `aggregation_levels.level_4.cluster_weights` (CL01=0.30, CL02=0.25, CL03=0.25, CL04=0.20).
- ✅ Suma de pesos = 1.0 (normalización implícita).
- ❌ **NO** especifica explícitamente la regla de normalización o manejo de NA.

**16. ¿`rubric_scoring.json` establece el manejo de valores faltantes (NA) y reglas de imputación o exclusión para evitar sesgos en el puntaje?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ En `special_cases.not_applicable`:
  - `"rule": "Mark point as N/A if criteria met"`
  - `"exclude_from_global": true`
  - Ejemplos: P9 (sin cárcel), P10 (sin ruta migratoria)
- ✅ En `aggregation_levels.level_4`: `"exclude_na": true`
- ❌ **NO** define reglas de imputación (mean, median, zero).
- ❌ **NO** especifica propagación de NA en niveles intermedios.

**17. ¿`rubric_scoring.json` incluye reglas para detección de desbalance intra-clúster (por ejemplo, umbral de rango, desviación estándar o índice Gini)?**

**Respuesta:** ✅ **SÍ**.
- En `aggregation_levels.level_3_5_meso`:
  - `"imbalance_detection_enabled": true`
  - `"metrics": ["range", "std_dev", "gini"]`
- En `meso_clusters.CL##`:
  - `"imbalance_threshold": 30.0` (CL01, CL02) o `25.0` (CL03, CL04)
- ❌ **NO** especifica la fórmula exacta de Gini ni cómo se interpretan los umbrales.

---

#### **SECCIÓN 7: INMUTABILIDAD Y REPRODUCIBILIDAD**

**18. ¿Ambos archivos incluyen un campo `checksum` o `content_hash` para garantizar inmutabilidad y reproducibilidad de corrida?**

**Respuesta:** ❌ **NO**.
- Ninguno de los dos archivos (`questionnaire.json`, `rubric_scoring.json`) contiene campos `checksum`, `content_hash` o `hash`.
- **Implicación:** No hay garantía de inmutabilidad entre ejecuciones.

**19. ¿Los archivos `questionnaire.json` y `rubric_scoring.json` declaran explícitamente compatibilidad de versión cruzada (por ejemplo, `requires_questionnaire_version` en rúbricas)?**

**Respuesta:** ❌ **NO**.
- `questionnaire.json` tiene `"version": "2.0.0"`.
- `rubric_scoring.json` tiene `"version": "2.0"` (inconsistencia de formato).
- ❌ **NO** existe `requires_questionnaire_version`, `compatible_questionnaire_version` o `version_compatibility` en ningún archivo.
- **Implicación:** No hay validación cruzada de versiones en tiempo de ejecución.

---

#### **SECCIÓN 8: ESQUEMAS FORMALES**

**20. ¿Existen esquemas formales `questionnaire.schema.json` y `rubric_scoring.schema.json` y están validados en tiempo de arranque?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ **SÍ** existen ambos esquemas en `/schemas/`:
  - `questionnaire.schema.json` (177 líneas)
  - `rubric_scoring.schema.json` (169 líneas)
- ✅ Existe `schema_validator.py` (407 líneas) con métodos de validación.
- ⚠️ **PENDIENTE:** La integración en `orchestrator.py` fue **INICIADA pero NO completada** (solo imports agregados, no validación en `__init__`).

**21. ¿Los esquemas definen expresamente formatos, rangos numéricos, patrones de ID, unicidad y relaciones referenciales (por ejemplo, `PAxx` usado debe existir)?**

**Respuesta:** ✅ **SÍ** (en los esquemas, NO en los archivos JSON reales).

**Esquemas contienen:**
- ✅ Patrones regex: `"^CL0[1-4]$"`, `"^PA(0[1-9]|10)$"`, `"^Q\\d{3}$"`, `"^DIM0[1-6]$"`
- ✅ Rangos numéricos: score 0.0-3.0, percentages 0.0-100.0
- ✅ Unicidad: `"uniqueItems": true` en arrays
- ❌ **NO** validan referencias cruzadas (e.g., que `PA02` usado en cluster exista en `policy_areas`)
- ❌ Los archivos JSON reales usan formato legacy (`P1`, `D1`, `P1-D1-Q1`) **NO compatible con esquemas** (`PA01`, `DIM01`, `Q001`)

---

#### **SECCIÓN 9: INTERNACIONALIZACIÓN Y PRIORIZACIÓN**

**22. ¿`questionnaire.json` contiene una sección `localization` o estrategias de i18n para etiquetas y descripciones, o todo el texto es monolingüe?**

**Respuesta:** ❌ **NO**.
- **NO** existe sección `localization`, `i18n`, `lang` o `language`.
- Todo el contenido es **monolingüe español**.
- No hay estrategia de internacionalización.

**23. ¿`questionnaire.json` define campos para priorización o criticidad de preguntas (por ejemplo, `severity`, `must_have`) que afecten el cálculo del score?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ En `dimensiones.D#.decalogo_dimension_mapping.P#`:
  - `"is_critical": true/false`
  - `"minimum_score": 0.5` (umbral)
- ✅ Al final del archivo (líneas 23904-24002) algunas preguntas tienen `"severity": "CRITICA"/"ALTA"/"MEDIA"`.
- ❌ **NO** existe campo `must_have`, `priority` o `weight_hint` a nivel de pregunta.
- ❌ **NO** está claro cómo `severity` afecta el cálculo (no documentado).

---

#### **SECCIÓN 10: FÓRMULAS Y JUSTIFICACIONES**

**24. ¿`rubric_scoring.json` describe explícitamente la fórmula de agregación (media ponderada, mediana ponderada, puntuación compuesta) y su justificación?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ Fórmulas declaradas:
  - `level_2`: `"(sum_of_5_questions / 15) * 100"`
  - `level_3`: `"sum_of_6_dimensions / 6"`
  - `level_3_5_meso`: `"weighted_average(point_scores_in_cluster)"`
  - `level_4`: `"weighted_average(cluster_scores)"`
- ✅ Método especificado en `meso_clusters`: `"aggregation_method": "weighted_average"`
- ❌ **NO** incluye justificación teórica (¿por qué media ponderada y no mediana? ¿sensibilidad a outliers?)
- ❌ **NO** especifica si es media aritmética o geométrica.

**25. ¿`rubric_scoring.json` documenta las condiciones bajo las cuales una puntuación se marca como "no evaluable" y cómo se refleja en el reporte?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ En `policy_areas_by_point.P9` y `P10`:
  - `"can_be_na": true`
  - `"na_condition": "No existe centro de reclusión en el municipio"` (P9)
  - `"na_condition": "Municipio no limita con ruta migratoria del Darién"` (P10)
- ✅ En `special_cases.not_applicable`: `"exclude_from_global": true`
- ❌ **NO** especifica cómo se refleja en el reporte (¿marcador "N/A"? ¿texto alternativo?).

---

#### **SECCIÓN 11: CONSISTENCIA TERMINOLÓGICA**

**26. ¿Los nombres de clúster en `questionnaire.json` coinciden exactamente con los utilizados por el ensamblador de reportes y el coreógrafo (sin variaciones tipográficas)?**

**Respuesta:** ✅ **SÍ** (entre archivos JSON).
- `questionnaire.json` y `rubric_scoring.json` usan idénticamente:
  - CL01: "Seguridad y Paz"
  - CL02: "Grupos Poblacionales"
  - CL03: "Territorio-Ambiente"
  - CL04: "Derechos Sociales & Crisis"
- ⚠️ **PENDIENTE VALIDAR:** Consistencia con `orchestrator.py`, `choreographer.py` y `report_assembly.py` (no verificado sin ejecutar código).

---

#### **SECCIÓN 12: REFERENCIAS CRUZADAS Y RECOMENDACIONES**

**27. ¿`questionnaire.json` incluye referencias cruzadas Q→PA→CL que permitan al coreógrafo construir vistas micro, meso y macro sin lógica adicional?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ Q→PA: Cada pregunta tiene `"policy_area": "P#"`.
- ✅ PA→CL: `metadata.clusters` mapea `policy_area_ids` a `cluster_id`.
- ❌ **NO** hay una lookup table directa Q→CL (requiere 2 saltos).
- ❌ Las preguntas usan IDs compuestos (`P1-D1-Q1`) que requieren parsing, no son atómicos.

**28. ¿`rubric_scoring.json` define tablas de recomendación parametrizadas por score y patrón de desbalance que el ensamblador pueda instanciar sin heurísticas ocultas?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ Define `score_bands` con umbrales y recomendaciones textuales (EXCELENTE, BUENO, etc.).
- ✅ En esquema `rubric_scoring.schema.json` existe sección `recommendation_rules` con:
  - `condition.type`: "low_score", "high_imbalance", "combined"
  - `action_template`: problem, intervention, indicator, responsible, timeframe
- ❌ **NO** implementado en `rubric_scoring.json` (solo en esquema).
- ❌ Las recomendaciones actuales son textos estáticos, NO paramétricas.

---

#### **SECCIÓN 13: PROVENANCE Y AUDITORÍA**

**29. ¿Ambos archivos incluyen `provenance` (autor, fecha, herramienta de edición) y `changelog` para auditoría?**

**Respuesta:** ❌ **NO**.
- `questionnaire.json`: Solo `"author": "JCRR"` y `"created_date": "2025-10-14"`.
- `rubric_scoring.json`: Solo `"created": "2025-01-15"`.
- ❌ **NO** existe `provenance`, `changelog`, `history`, `tool`, `editor`.
- ❌ No hay trazabilidad de modificaciones o autoría granular.

---

#### **SECCIÓN 14: EVIDENCIAS MÚLTIPLES**

**30. ¿`questionnaire.json` especifica explícitamente si las preguntas admiten múltiples evidencias y cómo se agregan (suma, promedio, máximo con penalización)?**

**Respuesta:** ❌ **NO**.
- Las preguntas tienen `expected_elements` (array) pero NO especifican:
  - ¿Múltiples fuentes de evidencia para un mismo elemento?
  - ¿Agregación por suma/promedio/máximo?
  - ¿Penalización por contradicción?
- Solo existe mención en `special_cases.contradictory_info`: `"Prioritize table data over narrative"`, `"register_alert": true`.

---

#### **SECCIÓN 15: CATEGORIZACIÓN Y REDONDEO**

**31. ¿`rubric_scoring.json` fija los umbrales de categorización cualitativa (por ejemplo, Bajo/Medio/Alto) y su correspondencia exacta con intervalos numéricos?**

**Respuesta:** ✅ **SÍ**.
- Definido en `score_bands`:
  - EXCELENTE: 85-100
  - BUENO: 70-84
  - SATISFACTORIO: 55-69
  - INSUFICIENTE: 40-54
  - DEFICIENTE: 0-39
- Intervalos **cerrados y contiguos** (sin gaps).

**32. ¿Existe una política de redondeo y formato de salida en `rubric_scoring.json` (por ejemplo, precisión a dos decimales y reglas de empate)?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ En `aggregation_levels`:
  - `level_1` (Question): `"precision": 2`
  - `level_2, 3, 3_5, 4`: `"precision": 1`
- ❌ **NO** especifica regla de redondeo (round half-up, banker's rounding, truncate).
- ❌ **NO** define manejo de empates en umbrales (¿84.5 es BUENO o SATISFACTORIO?).

---

#### **SECCIÓN 16: APLICABILIDAD CONDICIONAL**

**33. ¿`questionnaire.json` contiene metacampos para `applicability_conditions` que permitan desactivar preguntas según el tipo de municipio o plan?**

**Respuesta:** ❌ **NO**.
- **NO** existe campo `applicability_conditions`, `applicability`, `conditions`, o `filter` en las preguntas.
- La aplicabilidad se maneja a nivel de **policy area completa** (P9, P10 con `can_be_na` en `rubric_scoring.json`), NO a nivel de pregunta individual.

---

#### **SECCIÓN 17: PENALIZACIONES Y BONIFICACIONES**

**34. ¿`rubric_scoring.json` define penalizaciones o bonificaciones condicionales (por ejemplo, evidencia contradictoria, indicadores ausentes, OOD) y sus pesos?**

**Respuesta:** ❌ **NO**.
- **NO** existe sección `penalties`, `bonuses`, `adjustments`.
- En `special_cases.contradictory_info`: Solo dice `"register_alert": true`, sin penalización numérica.
- **NO** define:
  - Penalización por contradicción
  - Bonificación por evidencia excepcional
  - Ajuste por datos OOD (out-of-distribution)

---

#### **SECCIÓN 18: ALEATORIEDAD Y REPRODUCIBILIDAD**

**35. ¿Los dos archivos declaran `seed` o parámetros de aleatoriedad cuando intervienen procesos estocásticos en el pipeline de evaluación?**

**Respuesta:** ❌ **NO**.
- **NO** existe `seed`, `random_seed`, `stochastic`, `rng_state`.
- `scoring_modalities.TYPE_F` menciona:
  - `"uses_semantic_matching": true`
  - `"similarity_threshold": 0.6`
- ❌ **NO** especifica si el embedding es determinista o requiere seed.

---

#### **SECCIÓN 19: FUENTES DE VERIFICACIÓN**

**36. ¿`questionnaire.json` expone una lista exhaustiva y canónica de `sources_of_verification` aceptables y su tipología para cada pregunta?**

**Respuesta:** ❌ **NO**.
- **NO** existe campo `sources_of_verification` o `verification_sources` a nivel de pregunta.
- Las preguntas contienen `patrones_verificacion` (regex patterns) pero NO una **taxonomía de fuentes**.
- No hay clasificación tipo: "fuente primaria/secundaria", "oficial/no oficial", "cuantitativa/cualitativa".

---

#### **SECCIÓN 20: PROPAGACIÓN DE INCERTIDUMBRE**

**37. ¿`rubric_scoring.json` incluye reglas de propagación de incertidumbre o intervalos de confianza cuando las entradas son parciales o ruidosas?**

**Respuesta:** ❌ **NO**.
- **NO** existe `uncertainty`, `confidence_interval`, `error_propagation`.
- Las puntuaciones son **deterministas** (sin modelado de incertidumbre).
- No hay tratamiento de:
  - Intervalos de confianza
  - Propagación de error
  - Análisis de sensibilidad

---

#### **SECCIÓN 21: NORMALIZACIÓN Y REDUNDANCIA**

**38. ¿`questionnaire.json` y `rubric_scoring.json` están libres de claves redundantes y mantienen normalización para evitar inconsistencias entre documentos?**

**Respuesta:** ❌ **NO**.

**Redundancias detectadas:**
1. **Cluster names duplicados:**
   - `questionnaire.json.metadata.clusters` y `rubric_scoring.json.meso_clusters` repiten nombres idénticos.
2. **Policy area IDs:**
   - Tanto `policy_area_ids` como `legacy_point_ids` en clusters (valores idénticos).
3. **Pesos de agregación:**
   - Pesos por dimensión en `questionnaire.json.dimensiones.D#.peso_por_punto` y `questionnaire.json.dimensiones.D#.decalogo_dimension_mapping.P#.weight` **pueden estar duplicados**.
4. **Metadata de dimensiones:**
   - `rubric_scoring.json.dimensions` repite información de `questionnaire.json.dimensiones`.

**Violaciones de normalización:**
- Cluster definitions should be single source of truth (actualmente en 2 archivos).
- Pesos deberían estar solo en `rubric_scoring.json` (actualmente en ambos).

---

#### **SECCIÓN 22: PRUEBAS DE INTEGRIDAD**

**39. ¿Existe una prueba de integridad que compare `questionnaire.json` y `rubric_scoring.json` para garantizar que todas las referencias cruzadas están resueltas y versionadas?**

**Respuesta:** ⚠️ **PARCIALMENTE**.
- ✅ Existe `schema_validator.py` con método `validate_all()` que valida ambos archivos contra esquemas.
- ❌ **NO** valida referencias cruzadas entre archivos:
  - ¿Todos los `policy_area_ids` en clusters existen en `puntos_decalogo`?
  - ¿Todos los `dimension_id` referenciados existen?
  - ¿Los pesos en ambos archivos son consistentes?
  - ¿Las versiones son compatibles?
- ❌ **NO** está integrado en pipeline de CI/CD o pre-commit hooks.

---

--