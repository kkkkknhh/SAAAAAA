 # JUEGO COMPLETO DE MÉTODOS - NIVEL 3
## Sistema de 416 Métodos Mapeados a 30 Preguntas Genéricas

---

## 📊 ESTRUCTURA DEL JSON

### Metadata
```json
{
  "metadata": {
    "total_methods": 416,
    "total_questions": 30,
    "dimensions": 6,
    "files": { ... },      // Códigos de archivo (PP, CD, FV, etc.)
    "types": { ... },      // Tipos de método (E, V, T, C, O, R)
    "priority": { ... }    // Niveles de prioridad (3=★, 2=◆, 1=○)
  }
}
```

### Dimensiones
Cada dimensión contiene 5 preguntas con sus métodos:
- **D1**: Diagnóstico y Consistencia (80 métodos)
- **D2**: Actividades y Coherencia (107 métodos)
- **D3**: Productos e Indicadores (101 métodos)
- **D4**: Resultados y Supuestos (94 métodos)
- **D5**: Impactos y Sostenibilidad (91 métodos)
- **D6**: Causalidad Global (155 métodos) ← **LA MÁS COMPLEJA**

---

## 🔑 CONVENCIONES

### Códigos de Archivo
| Código | Archivo                           |
|--------|-----------------------------------|
| PP     | policy_processor.py               |
| CD     | contradiction_deteccion.py        |
| FV     | financiero_viabilidad_tablas.py   |
| DB     | dereck_beach.py                   |
| RA     | report_assembly.py                |
| EP     | embedding_policy.py               |
| A1     | Analyzer_one.py                   |
| TC     | teoria_cambio.py                  |
| SC     | semantic_chunking_policy.py       |

### Tipos de Método
| Código | Tipo          | Descripción                    |
|--------|---------------|--------------------------------|
| E      | Extracción    | Extrae información             |
| V      | Validación    | Valida/verifica                |
| T      | Transformación| Transforma/normaliza datos     |
| C      | Cálculo       | Calcula métricas/scores        |
| O      | Orquestación  | Coordina otros métodos         |
| R      | Reporte       | Genera reportes/outputs        |

### Niveles de Prioridad
| Nivel | Símbolo | Significado                          |
|-------|---------|--------------------------------------|
| 3     | ★       | **Crítico** - Sin él no funciona     |
| 2     | ◆       | **Importante** - Pérdida significativa|
| 1     | ○       | **Complementario** - Mejora calidad  |

---

## 💻 CÓMO USAR EL JSON

### Ejemplo 1: Cargar y Explorar
```python
import json

# Cargar el JSON
with open('catalogo_completo_canonico.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Ver metadata
print(f"Total métodos: {data['metadata']['total_methods']}")
print(f"Total preguntas: {data['metadata']['total_questions']}")

# Explorar una dimensión
d1 = data['dimensions'][0]
print(f"\nDimensión: {d1['name']}")
print(f"Total métodos: {d1['total_methods']}")
print(f"Preguntas: {len(d1['questions'])}")
```

### Ejemplo 2: Buscar Pregunta Específica
```python
def find_question(data, question_id):
    """Encuentra una pregunta por su ID (ej: 'D1-Q1')"""
    for dimension in data['dimensions']:
        for question in dimension['questions']:
            if question['q'] == question_id:
                return question
    return None

# Buscar D1-Q1
q = find_question(data, 'D1-Q1')
print(f"Pregunta: {q['t']}")
print(f"Total métodos: {q['m']}")
print(f"Flujo: {q['flow']}")
```

### Ejemplo 3: Analizar Métodos por Archivo
```python
def count_methods_by_file(question):
    """Cuenta métodos por archivo para una pregunta"""
    counts = {}
    for package in question['p']:
        file_code = package['f']
        method_count = len(package['m'])
        counts[file_code] = counts.get(file_code, 0) + method_count
    return counts

q = find_question(data, 'D1-Q3')
counts = count_methods_by_file(q)
print(f"\nDistribución de métodos para {q['t']}:")
for file_code, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    file_name = data['metadata']['files'][file_code]
    print(f"  {file_code} ({file_name}): {count} métodos")
```

### Ejemplo 4: Extraer Métodos Críticos
```python
def get_critical_methods(question):
    """Extrae solo los métodos críticos (prioridad 3)"""
    critical = []
    for package in question['p']:
        for i, priority in enumerate(package['pr']):
            if priority == 3:
                method_name = package['m'][i]
                critical.append({
                    'file': package['f'],
                    'class': package['c'],
                    'method': method_name,
                    'type': package['t'][i]
                })
    return critical

q = find_question(data, 'D6-Q2')  # Anti-Milagro
critical = get_critical_methods(q)
print(f"\nMétodos críticos para {q['t']}:")
for m in critical[:5]:  # Primeros 5
    print(f"  ★ {m['file']}.{m['class']}.{m['method']} [{m['type']}]")
```

### Ejemplo 5: Generar Diagrama de Flujo
```python
def generate_flow_diagram(question):
    """Genera representación visual del flujo"""
    flow = question['flow']
    print(f"\n{'='*60}")
    print(f"FLUJO: {question['t']}")
    print(f"{'='*60}")
    print(flow)
    print(f"{'='*60}")
    
    # Detallar participantes
    print("\nPARTICIPANTES:")
    for package in question['p']:
        file_code = package['f']
        class_name = package['c']
        method_count = len(package['m'])
        print(f"  {file_code}.{class_name}: {method_count} métodos")

q = find_question(data, 'D6-Q1')
generate_flow_diagram(q)
```

---

## 🎯 CARACTERÍSTICAS ESPECIALES

El JSON incluye una sección `special_features` que documenta:

### 1. Sistema Bicameral (D6-Q3 y D6-Q4)
Dos rutas paralelas de resolución:
- **Ruta 1**: Detección local (`PolicyContradictionDetector._suggest_resolutions`)
- **Ruta 2**: Inferencia estructural (`TeoriaCambio._generar_sugerencias_internas`)

### 2. Validación Anti-Milagro (D6-Q2)
Tres categorías de patrones para detectar saltos inverosímiles:
- `enlaces_proporcionales`
- `sin_saltos`
- `no_milagros`

### 3. Derek Beach Process Tracing
Cuatro tipos de tests evidenciales:
- **Hoop Test**: Necesario pero NO suficiente
- **Smoking Gun Test**: Suficiente pero NO necesario
- **Doubly Decisive Test**: Necesario Y suficiente
- **Straw in Wind Test**: Ni necesario ni suficiente

### 4. Framework CDAF Completo
9 componentes integrados para análisis causal

---

## 📈 ESTADÍSTICAS CLAVE

```
Total de Métodos: 416
Total de Clases: 82
Total de Archivos: 9
Total de Preguntas: 30

Distribución por Dimensión:
  D1: 80 métodos  (19.2%)
  D2: 107 métodos (25.7%)
  D3: 101 métodos (24.3%)
  D4: 94 métodos  (22.6%)
  D5: 91 métodos  (21.9%)
  D6: 155 métodos (37.3%) ← LA MÁS COMPLEJA

Archivo más usado:
  dereck_beach.py: 99 métodos (23.8%)
```

---

## 🚀 CASOS DE USO

### 1. Orquestador/Coreógrafo
Usa el JSON para determinar qué métodos invocar para cada pregunta:
```python
question = find_question(data, user_query_dimension)
for package in question['p']:
    if package['pr'][0] == 3:  # Solo críticos
        invoke_methods(package['f'], package['c'], package['m'])
```

### 2. Análisis de Cobertura
Verifica qué métodos participan en múltiples preguntas:
```python
method_usage = count_method_usage_across_questions(data)
# Identifica métodos más reutilizados
```

### 3. Optimización de Performance
Prioriza ejecución según criticidad:
```python
critical_first = filter_by_priority(question['p'], min_priority=3)
important_second = filter_by_priority(question['p'], min_priority=2)
```

### 4. Documentación Automática
Genera documentación de flujos:
```python
for dimension in data['dimensions']:
    generate_markdown_doc(dimension)
```

---

## 📝 NOTAS IMPORTANTES

1. **Algunos métodos aparecen en múltiples preguntas** - esto es intencional y refleja su reutilización
2. **Los flujos son simplificados** - en producción pueden tener más iteraciones
3. **Las prioridades son contextuales** - un método "complementario" en D1-Q1 puede ser "crítico" en D3-Q2
4. **El campo "note" proporciona contexto adicional** - léelo cuando esté presente

---

## 🔧 EXTENSIÓN DEL JSON

Para agregar nuevas preguntas:
```json
{
  "q": "D7-Q1",
  "t": "Título de la pregunta",
  "m": 15,
  "p": [
    {
      "f": "PP",
      "c": "ClassName",
      "m": ["method1", "method2"],
      "t": ["E", "V"],
      "pr": [3, 2]
    }
  ],
  "flow": "PP.E → PP.V"
}
```

---

## 📧 SOPORTE

Para preguntas sobre el mapeo de métodos, consulta:
- Documento original: `MAPEO COMPLETO 30 PREGUNTAS → 416 MÉTODOS`
- Sección de características especiales del JSON
- Comentarios en el campo "note" de cada pregunta

---

**Versión**: 1.0  
**Última actualización**: Octubre 2025  
**Objetivo alcanzado**: 95% de utilización de código ✅
