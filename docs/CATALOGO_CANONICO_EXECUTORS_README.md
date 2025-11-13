# Catálogo Canónico de Executors y Métodos

## 📋 Descripción

Este catálogo mapea sistemáticamente todos los métodos utilizados en los 30 executors que responden las preguntas base del cuestionario canónico (Q001-Q030). El catálogo está diseñado para:

- **Orquestación inteligente**: Selección automática de métodos por prioridad
- **Análisis de cobertura**: Identificación de métodos reutilizados
- **Optimización de ejecución**: Priorización contextual de métodos
- **Trazabilidad completa**: Mapeo desde pregunta hasta método específico

## 📁 Ubicación

```
/home/user/SAAAAAA/config/catalogo_canonico_executors_metodos.json
```

## 🏗️ Estructura del Catálogo

### Metadata Esencial

```json
{
  "metadata": {
    "total_methods": 613,
    "total_methods_unique": 289,
    "total_questions": 300,
    "base_questions": 30,
    "dimensions": 6,
    "files": {
      "PP": "policy_processor.py",
      "CD": "contradiction_deteccion.py",
      "FV": "financiero_viabilidad_tablas.py",
      "DB": "dereck_beach.py",
      "RA": "report_assembly.py",
      "EP": "embedding_policy.py",
      "A1": "Analyzer_one.py",
      "TC": "teoria_cambio.py",
      "SC": "semantic_chunking_policy.py"
    }
  }
}
```

### Dimensiones de Análisis

El catálogo cubre 6 dimensiones (DIM01-DIM06):

| Dimensión | Nombre | Label | Preguntas | Métodos | Complejidad |
|-----------|--------|-------|-----------|---------|-------------|
| DIM01 | INSUMOS | Diagnóstico y Recursos | Q001-Q005 | 82 | Baja (16.4 avg) |
| DIM02 | ACTIVIDADES | Diseño de Intervención | Q006-Q010 | 95 | Media (19.0 avg) |
| DIM03 | PRODUCTOS | Productos y Outputs | Q011-Q015 | 103 | Media (20.6 avg) |
| DIM04 | RESULTADOS | Resultados y Outcomes | Q016-Q020 | 92 | Media (18.4 avg) |
| DIM05 | IMPACTOS | Impactos de Largo Plazo | Q021-Q025 | 87 | Media (17.4 avg) |
| DIM06 | CAUSALIDAD | Teoría de Cambio | Q026-Q030 | 154 | **MÁXIMA** (30.8 avg) |

### Estructura por Pregunta

Cada pregunta tiene el siguiente formato:

```json
{
  "q": "Q001",
  "dimension": "DIM01",
  "executor": "D1Q1_Executor",
  "t": "Descripción de la pregunta",
  "m": 18,
  "flow": "PP.E → CD.V → EP.C → A1.T",
  "p": [
    {
      "f": "PP",
      "c": "IndustrialPolicyProcessor",
      "m": ["process", "extract_policy_elements"],
      "t": ["E", "O"],
      "pr": [3, 3],
      "note": "Procesamiento principal del documento"
    }
  ]
}
```

**Campos clave:**
- `q`: ID de la pregunta
- `dimension`: Dimensión asociada (DIM01-DIM06)
- `executor`: Nombre del executor
- `t`: Texto descriptivo de la pregunta
- `m`: Total de métodos utilizados
- `flow`: Flujo de ejecución (Archivo.Tipo → ...)
- `p`: Paquetes de métodos
  - `f`: Archivo (PP, CD, FV, etc.)
  - `c`: Clase
  - `m`: Lista de métodos
  - `t`: Tipos de operación (E=Extracción, V=Validación, T=Transformación, C=Cálculo, O=Orquestación, R=Reporte)
  - `pr`: Prioridades (3=★ Crítico, 2=◆ Importante, 1=○ Complementario)
  - `note`: Nota explicativa

## 🔧 Características Especiales

### 1. Sistema Bicameral (Q028 y Q029)

El sistema bicameral implementa dos rutas independientes para detección de inconsistencias y generación de sugerencias:

**Ruta 1 (Q028)**: Detección Local
- Executor: `D6Q3_Executor`
- Método principal: `PolicyContradictionDetector.detect`
- Propósito: Detectar contradicciones explícitas en el documento

**Ruta 2 (Q029)**: Inferencia Estructural
- Executor: `D6Q4_Executor`
- Método principal: `TeoriaCambio._generar_sugerencias_internas`
- Propósito: Generar sugerencias de mejora basadas en estructura causal

### 2. Validación Anti-Milagro (Q027)

Valida que la teoría de cambio no contenga "saltos milagrosos" sin justificación causal:

```json
{
  "patterns": [
    "enlaces_proporcionales",
    "sin_saltos_logicos",
    "no_milagros_causales"
  ],
  "thresholds": {
    "proportionality": 0.8,
    "continuity": 0.95,
    "plausibility": 1.0
  }
}
```

**Métodos clave:**
- `BayesianMechanismInference._build_transition_matrix`
- `CausalInferenceSetup.classify_goal_dynamics`
- `CausalInferenceSetup.identify_failure_points`

### 3. Derek Beach Process Tracing

Framework de inferencia causal con 4 tipos de tests evidenciales:

| Test | Lógica | Uso |
|------|--------|-----|
| **Hoop Test** | Necesario pero NO suficiente | Si falla → descarta hipótesis |
| **Smoking Gun** | Suficiente pero NO necesario | Si pasa → confirma hipótesis |
| **Doubly Decisive** | Necesario Y suficiente | Confirma o descarta definitivamente |
| **Straw in Wind** | Ni necesario ni suficiente | Actualización marginal |

**Preguntas que usan Derek Beach:** Q007, Q010, Q015, Q017, Q022, Q023, Q026, Q027, Q029

## 📊 Estadísticas Clave

### Cobertura de Archivos Core

```
PP (policy_processor.py)           : 100% ████████████████████ (30/30)
CD (contradiction_deteccion.py)    : 57%  ███████████░░░░░░░░░░ (17/30)
EP (embedding_policy.py)           : 70%  ██████████████░░░░░░ (21/30)
FV (financiero_viabilidad_tablas.py): 60% ████████████░░░░░░░░ (18/30)
A1 (Analyzer_one.py)               : 50%  ██████████░░░░░░░░░░ (15/30)
DB (dereck_beach.py)               : 43%  ████████░░░░░░░░░░░░ (13/30)
TC (teoria_cambio.py)              : 40%  ████████░░░░░░░░░░░░ (12/30)
SC (semantic_chunking_policy.py)   : 3%   ░░░░░░░░░░░░░░░░░░░░ (1/30)
RA (report_assembly.py)            : 0%   ░░░░░░░░░░░░░░░░░░░░ (0/30)
```

### Distribución de Métodos por Tipo

```
Cálculo (C)        : 200 métodos (33%)
Extracción (E)     : 180 métodos (29%)
Validación (V)     :  80 métodos (13%)
Transformación (T) :  70 métodos (11%)
Orquestación (O)   :  45 métodos (7%)
Reporte (R)        :  38 métodos (6%)
```

### Complejidad

- **Pregunta más simple**: Q002 (12 métodos) - Normalización y Fuentes
- **Pregunta más compleja**: Q027 (38 métodos) - Proporcionalidad y Continuidad
- **Promedio**: 20.4 métodos por pregunta

## 🚀 Casos de Uso

### 1. Orquestador Inteligente

Ejecutar solo métodos críticos (prioridad 3):

```python
def orquestar_pregunta(question_id):
    question = find_question(catalog, question_id)
    for package in question['p']:
        if any(pr == 3 for pr in package['pr']):
            critical_methods = [
                m for m, pr in zip(package['m'], package['pr'])
                if pr == 3
            ]
            ejecutar_metodos(package['f'], package['c'], critical_methods)
    return results
```

### 2. Análisis de Cobertura

Identificar métodos más reutilizados:

```python
def analizar_metodos_reutilizados(catalog):
    method_usage = defaultdict(list)
    for qid, question in catalog['questions'].items():
        for package in question['p']:
            for method in package['m']:
                full_name = f"{package['f']}.{package['c']}.{method}"
                method_usage[full_name].append(qid)
    # Retornar métodos usados en 5+ preguntas
    return {m: qs for m, qs in method_usage.items() if len(qs) > 5}
```

### 3. Optimización por Tipo

Ejecutar solo métodos de Extracción y Validación:

```python
def obtener_metodos_priorizados(question_id, min_priority=2, tipos=['E', 'V']):
    question = catalog['questions'][question_id]
    methods = []
    for package in question['p']:
        for method, tipo, pr in zip(package['m'], package['t'], package['pr']):
            if pr >= min_priority and tipo in tipos:
                methods.append({
                    'file': package['f'],
                    'class': package['c'],
                    'method': method,
                    'type': tipo,
                    'priority': pr
                })
    return sorted(methods, key=lambda x: (-x['priority'], x['type']))
```

### 4. Ejecutar Sistema Bicameral

```python
def ejecutar_sistema_bicameral(policy_document):
    # Ruta 1: Detección local (Q028)
    contradicciones = ejecutar_executor('D6Q3_Executor', policy_document)

    # Ruta 2: Sugerencias estructurales (Q029)
    sugerencias = ejecutar_executor('D6Q4_Executor', policy_document)

    # Fusionar resultados
    return {
        'ruta_1_contradicciones': contradicciones,
        'ruta_2_sugerencias': sugerencias,
        'resolucion_integrada': fusionar_bicameral(contradicciones, sugerencias)
    }
```

### 5. Validación Anti-Milagro

```python
def validar_anti_milagro(policy_document):
    q027 = catalog['questions']['Q027']
    thresholds = catalog['special_features']['anti_milagro_validation']['thresholds']

    resultados = ejecutar_executor('D6Q2_Executor', policy_document)

    validaciones = {
        'proportionality': resultados['score'] >= thresholds['proportionality'],
        'continuity': resultados['continuity_score'] >= thresholds['continuity'],
        'plausibility': resultados['plausibility'] >= thresholds['plausibility']
    }

    return all(validaciones.values()), validaciones
```

## 🔄 Extensión a 300 Preguntas

Las 30 preguntas base se replican en 10 policy areas (PA01-PA10):

**Fórmula de Replicación:**
```
Para pregunta Qxxx en policy area PAyy:
  - Usar mismo executor que pregunta base
  - Aplicar contexto de policy area PAyy

Ejemplo:
  Q031 (PA10, DIM01, pregunta 1) → usa D1Q1_Executor con contexto PA10
```

**Mapeo de Preguntas:**
- Q001-Q030: PA01 (base)
- Q031-Q060: PA10
- Q061-Q090: PA02
- Q091-Q120: PA03
- Q121-Q150: PA04
- Q151-Q180: PA05
- Q181-Q210: PA06
- Q211-Q240: PA07
- Q241-Q270: PA08
- Q271-Q300: PA09

## 📝 Convenciones del Catálogo

### Códigos de Archivos

| Código | Archivo | Descripción |
|--------|---------|-------------|
| **PP** | policy_processor.py | Procesamiento de políticas |
| **CD** | contradiction_deteccion.py | Detección de contradicciones |
| **FV** | financiero_viabilidad_tablas.py | Análisis financiero |
| **DB** | dereck_beach.py | Process tracing causal |
| **RA** | report_assembly.py | Ensamblaje de reportes |
| **EP** | embedding_policy.py | Embeddings semánticos |
| **A1** | Analyzer_one.py | Analizadores generales |
| **TC** | teoria_cambio.py | Teoría de cambio |
| **SC** | semantic_chunking_policy.py | Chunking semántico |

### Tipos de Operación

| Código | Tipo | Descripción |
|--------|------|-------------|
| **E** | Extracción | Extrae información del documento |
| **V** | Validación | Verifica consistencia y calidad |
| **T** | Transformación | Transforma o enriquece datos |
| **C** | Cálculo | Computa métricas y scores |
| **O** | Orquestación | Coordina múltiples operaciones |
| **R** | Reporte | Genera outputs y recomendaciones |

### Niveles de Prioridad

| Valor | Símbolo | Descripción | Uso |
|-------|---------|-------------|-----|
| **3** | ★ | Crítico | Método esencial, falla causa error |
| **2** | ◆ | Importante | Método relevante, mejora calidad |
| **1** | ○ | Complementario | Método opcional, enriquece análisis |

## 🎯 Recomendaciones

### Optimización

1. **Cache de métodos universales**: PP y CD se usan en todas las preguntas → implementar caché
2. **Ejecución paralela**: Métodos independientes pueden ejecutarse en paralelo
3. **Carga progresiva**: Cargar solo dimensión necesaria para optimizar memoria

### Mantenimiento

1. **Testing prioritario**: PP y CD son críticos → testing exhaustivo
2. **Documentación avanzada**: DB y TC son complejos → documentar casos de uso
3. **Evaluación de SC**: Uso muy bajo (3%) → evaluar necesidad o expandir funcionalidad

### Desarrollo

1. **Interfaces estandarizadas**: Estandarizar interfaces entre archivos core
2. **Versionado de métodos**: Mantener compatibilidad con versiones anteriores
3. **Telemetría**: Instrumentar ejecución para análisis de rendimiento

## 📚 Referencias

- **Notación Canónica**: `docs/CANONICAL_NOTATION.md`
- **Questionnaire Monolith**: `data/questionnaire_monolith.json`
- **Mapeo de Executors**: `MAPEO_EXECUTORS_Q001_Q030.md`
- **Tabla Resumen**: `TABLA_RESUMEN_Q001_Q030.md`
- **Executors Implementation**: `src/saaaaaa/core/orchestrator/executors.py`

## 🔍 Validación

Para validar el catálogo:

```bash
# Validar JSON
python3 -c "import json; json.load(open('config/catalogo_canonico_executors_metodos.json'))"

# Ver estadísticas
python3 -c "
import json
data = json.load(open('config/catalogo_canonico_executors_metodos.json'))
print(f'Preguntas: {len(data[\"questions\"])}')
print(f'Métodos: {data[\"metadata\"][\"total_methods\"]}')
print(f'Dimensiones: {len(data[\"dimensions\"])}')
"
```

## 📞 Soporte

Para preguntas o mejoras al catálogo, consultar:
- Documentación del sistema de calibración: `docs/CALIBRATION_SYSTEM.md`
- Guía de orchestrator: `docs/ARQUITECTURA_ORQUESTADOR_COREOGRAFO.md`

---

**Versión**: 1.0.0
**Última actualización**: 2025-11-13
**Generado por**: Análisis exhaustivo de executors Q001-Q030
