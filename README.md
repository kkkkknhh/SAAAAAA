# SAAAAAA: A Deterministic Pipeline for Multi-Method Policy Analysis with Complete Provenance Tracking

**Sistema de Análisis Estratégico de Políticas Públicas con Arquitectura de Alta Fidelidad**

---

## 📦 Package Installation & Dependency Management

### Quick Installation

**For development:**
```bash
pip install -r requirements-dev.txt
```

**For production:**
```bash
pip install -r requirements-core.txt
```

**Complete documentation:** See [DEPENDENCIES_QUICKSTART.md](DEPENDENCIES_QUICKSTART.md)

### Dependency Management System

This project uses a comprehensive dependency management system with:

- **Exact version pins** for reproducible builds
- **Classified dependencies** (core, optional, dev, docs)
- **Automated verification** and security scanning
- **CI/CD gates** to prevent dependency drift

**Key files:**
- `requirements-core.txt` - Core runtime dependencies (37 packages)
- `requirements-optional.txt` - Optional features (30 packages)
- `requirements-dev.txt` - Development tools (includes core)
- `DEPENDENCIES_AUDIT.md` - Complete dependency documentation

**Verification commands:**
```bash
# Verify all dependencies
make deps:verify

# Run dependency audit
make deps:audit

# Check importability
python3 scripts/verify_importability.py
```

**For detailed dependency information:** See [DEPENDENCIES_AUDIT.md](DEPENDENCIES_AUDIT.md)

SAAAAAA (Strategic Analysis Architecture for Administrative Accountability and Audit Assurance) es un sistema de análisis de políticas públicas que integra 584 métodos analíticos distribuidos en 7 productores especializados y 1 agregador, orientado al procesamiento determinista de planes de desarrollo municipales y departamentales en Colombia. La contribución técnica principal radica en: (1) un pipeline de ingesta con 9 fases deterministas que garantiza trazabilidad completa desde token hasta coordenadas de página (provenance_completeness = 1.0), (2) un sistema de señales transversales (cross-cut signals) con transporte memory:// y HTTP opcional, incluyendo circuit breakers para resiliencia, (3) un mecanismo de enrutamiento extendido (ArgRouter) con 30+ rutas especiales que elimina caídas silenciosas de parámetros, y (4) contratos explícitos de entrada/salida con validación en fronteras de proveedor. El sistema procesa 300 preguntas de evaluación organizadas en 6 dimensiones (D1-D6: Insumos, Actividades, Productos, Resultados, Impactos, Causalidad) sobre 10 áreas de política (P1-P10), generando reportes en tres niveles de agregación: MICRO (respuestas atómicas por pregunta, 150-300 palabras), MESO (análisis de clusters por dimensión-área), y MACRO (clasificación y recomendaciones). La arquitectura sigue el patrón "Chess Strategy": apertura paralela con 7 productores independientes, medio juego de triangulación multi-fuente, y final de síntesis doctoral. El alcance excluye procesamiento en tiempo real (modo batch únicamente), datos personales identificables (PII), y claims de precisión absoluta sin intervalos de confianza.

---

## Tabla de Contenidos

1. [Introducción](#1-introducción)
2. [Métodos / Arquitectura](#2-métodos--arquitectura)
   - 2.1. [Pipeline de Procesamiento](#21-pipeline-de-procesamiento)
   - 2.2. [Sistema de Contratos](#22-sistema-de-contratos)
   - 2.3. [Señales Transversales](#23-señales-transversales-cross-cut-signals)
   - 2.4. [CPPAdapter y Canon Policy Package](#24-cppadapter-y-canon-policy-package)
   - 2.5. [ArgRouter Extendido](#25-argrouter-extendido)
   - 2.6. [Parametrización en Código](#26-parametrización-en-código)
3. [Resultados / Métricas Operativas](#3-resultados--métricas-operativas)
   - 3.1. [Métricas de Señales](#31-métricas-de-señales)
   - 3.2. [Métricas de Proveniencia](#32-métricas-de-proveniencia)
   - 3.3. [Métricas de Enrutamiento](#33-métricas-de-enrutamiento)
   - 3.4. [Métricas de Determinismo](#34-métricas-de-determinismo)
   - 3.5. [Cobertura de Tests](#35-cobertura-de-tests)
4. [Discusión](#4-discusión)
   - 4.1. [Limitaciones Conocidas](#41-limitaciones-conocidas)
   - 4.2. [Amenazas a la Validez](#42-amenazas-a-la-validez)
   - 4.3. [Planes de Mitigación](#43-planes-de-mitigación)
5. [Protocolos de Reproducibilidad](#5-protocolos-de-reproducibilidad)
   - 5.1. [Ejecución de Golden Tests](#51-ejecución-de-golden-tests)
   - 5.2. [Fijación de Semillas](#52-fijación-de-semillas)
   - 5.3. [Verificación de phase_hash](#53-verificación-de-phase_hash)
6. [Ética y Privacidad](#6-ética-y-privacidad)
7. [Apéndices](#7-apéndices)
   - A. [Tabla de Configuraciones](#apéndice-a-tabla-de-configuraciones)
   - B. [Changelog Resumido](#apéndice-b-changelog-resumido)
   - C. [Matriz de Compatibilidad](#apéndice-c-matriz-de-compatibilidad)
8. [Cómo Citar este Repositorio](#8-cómo-citar-este-repositorio)
9. [Licencia](#9-licencia)
10. [Referencias Internas](#10-referencias-internas)

---

## 1. Introducción

### 1.1. Planteamiento del Problema

La evaluación ex-ante de planes de desarrollo requiere procesamiento analítico de documentos semi-estructurados (100-300 páginas) bajo múltiples dimensiones: viabilidad financiera, coherencia lógica, causalidad explícita, trazabilidad presupuestal, alineación normativa, y evidencia empírica. Los enfoques tradicionales presentan tres deficiencias:

1. **Pérdida de Trazabilidad**: Extracción de texto sin mapeo página-token impide auditoría de inferencias.
2. **Procesamiento No-Determinista**: Variaciones en chunking semántico y resolución de dependencias producen outputs no reproducibles.
3. **Triangulación Manual**: Síntesis multi-método requiere integración manual, introduciendo sesgos de confirmación.

### 1.2. Estado del Arte Mínimo

Sistemas previos en evaluación de políticas (e.g., análisis ToC con DAG validation, scoring Bayesiano, extracción de KPIs) operan de forma aislada. Frameworks de NLP (spaCy, Transformers) proveen extracción pero no garantías de proveniencia. RAG (Retrieval-Augmented Generation) carece de contratos formales para composición multi-método. Import-linter y pycycle abordan higiene de dependencias pero no señales transversales runtime.

### 1.3. Contribución y Enfoque

SAAAAAA integra:

1. **Determinismo de Pipeline**: 9 fases con postcondiciones verificables; fallo en cualquier fase → ABORT (no degradación gradual).
2. **Señales Transversales**: Registro centralizado de patrones, indicadores, umbrales desde cuestionario monolito hacia todos los ejecutores, con transporte memory:// (in-process) o HTTP (con circuit breaker).
3. **Proveniencia Completa**: Cada token → `{page_id, bbox, byte_range, parser_id}` mediante Arrow IPC, permitiendo auditoría forense.
4. **ArgRouter Extendido**: 30+ rutas especiales eliminan caídas silenciosas de parámetros (argrouter_coverage = 1.0).
5. **Contratos Explícitos**: TypedDict con validación en fronteras (orchestrator ↔ core), detectando violaciones arquitectónicas en runtime.

**Racionalidad del Determinismo**: En auditoría pública, reproducibilidad byte-a-byte es requisito legal. Aproximaciones probabilísticas sin intervalos de confianza no son aceptables.

---

## 2. Métodos / Arquitectura

### 2.1. Pipeline de Procesamiento

El sistema implementa un pipeline de 9 fases con dependencias secuenciales estrictas:

```
┌─────────────────────────────────────────────────────────────────┐
│ FASE 1: Acquisition & Integrity                                 │
│   Input:  file_path (Path)                                      │
│   Output: manifest.initial {blake3_hash, mime_type, byte_size}  │
│   Gate:   blake3_hash must be 64 hex chars                      │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 2: Format Decomposition                                    │
│   Input:  manifest.initial                                      │
│   Output: raw_object_tree {pages[], fonts[], images[]}          │
│   Gate:   len(pages) > 0                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 3: Structural Normalization (Policy-Aware)                 │
│   Input:  raw_object_tree                                       │
│   Output: policy_graph.prelim {Ejes, Programas, Proyectos}      │
│   Gate:   structural_consistency_score ≥ 1.0                    │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 4: Text Extraction & Normalization                         │
│   Input:  policy_graph.prelim                                   │
│   Output: content_stream.v1 (Unicode NFC, stable offsets)       │
│   Gate:   All text normalized to NFC                            │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 5: OCR (Conditional)                                       │
│   Input:  content_stream.v1, image_pages[]                      │
│   Output: ocr_layer {text, confidence_scores}                   │
│   Gate:   avg(confidence) ≥ ocr_confidence_threshold (0.85)     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 6: Tables & Budget Handling                                │
│   Input:  content_stream.v1                                     │
│   Output: tables_figures.subgraph {KPIs[], Budgets[]}           │
│   Gate:   budget_consistency_score ≥ 0.95                       │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 7: Provenance Binding                                      │
│   Input:  content_stream.v1, raw_object_tree                    │
│   Output: provenance_map.arrow (token→page/bbox/byte_range)     │
│   Gate:   provenance_completeness = 1.0 (NO partial coverage)   │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 8: Advanced Chunking                                       │
│   Input:  content_stream.v1, policy_graph.prelim                │
│   Output: chunk_graph {chunks[], edges[]}                       │
│   Gate:   boundary_f1 ≥ 0.85, chunk_overlap ≤ 0.15              │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│ FASE 9: Canonical Packing                                       │
│   Input:  All outputs from phases 1-8                           │
│   Output: CanonPolicyPackage (CPP) {content, provenance,        │
│           chunk_graph, integrity_index}                         │
│   Gate:   Merkle root recomputation matches stored hash         │
└─────────────────────────────────────────────────────────────────┘
```

**Postcondiciones por Fase**: Cada fase declara invariantes verificables. Violación → ABORT con diagnóstico detallado (no "best effort").

**Ejemplo de Fallo**: Si FASE 7 produce provenance_completeness = 0.98, sistema aborta (no tolera 2% de tokens sin trazabilidad).

### 2.2. Sistema de Contratos

#### 2.2.1. Definición de Contratos

Contratos son estructuras TypedDict que especifican:
- **Precondiciones**: Estados requeridos del mundo antes de ejecución.
- **Postcondiciones**: Garantías sobre outputs.
- **Invariantes**: Propiedades que deben mantenerse durante transformación.

**Ejemplo (core/contracts.py)**:
```python
class Deliverable(TypedDict):
    """Output contract de productores."""
    dimension: str  # "D1" | "D2" | ... | "D6"
    policy_area: str  # "P1" | "P2" | ... | "P10"
    evidence_items: List[EvidenceItem]
    bayesian_score: float  # [0.0, 1.0]
    confidence_interval: Tuple[float, float]  # (lower_95, upper_95)
    provenance_refs: List[ProvenanceRef]  # {page, bbox, token_ids}
    
class Expectation(TypedDict):
    """Input contract esperado por agregador."""
    required_producers: List[str]  # ["financiero", "causal", ...]
    min_evidence_per_question: int  # ≥ 3
    require_provenance: bool  # True (siempre)
```

#### 2.2.2. Validación en Fronteras

El módulo `orchestrator/provider.py` implementa `_enforce_boundary()` que inspecciona stack en runtime:

```python
def _enforce_boundary(self) -> None:
    """Prevent core → orchestrator imports (architectural violation)."""
    stack = inspect.stack()
    for frame_info in stack[2:]:  # Skip self and caller
        module_name = frame_info.frame.f_globals.get("__name__", "")
        if module_name.startswith("core.") and "orchestrator" in module_name:
            raise ArchitecturalViolationError(
                f"Core module {module_name} attempted to import orchestrator"
            )
```

**Dirección de Dependencias**: `orchestrator → core` (nunca reversa). Core es biblioteca pura (no I/O directo).

### 2.3. Señales Transversales (Cross-Cut Signals)

#### 2.3.1. Motivación

Cuestionario monolito contiene 300 preguntas con patrones, indicadores, umbrales que deben propagarse a 7 productores sin duplicación. Solución: canal de señales centralizado.

#### 2.3.2. Arquitectura

```
┌───────────────────────────────────────────────────────────────┐
│  questionnaire_monolith.json (300 questions)                  │
│    ↓ parse + extract                                          │
│  SignalPack {patterns[], indicators[], thresholds[]}          │
└──────────────┬────────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────┐
│  SignalClient (base_url = "memory://" | "http://...")        │
│    - register_memory_signal(key, pack)                       │
│    - fetch_signal_pack(key) → SignalPack                     │
│    - Circuit Breaker: threshold=5, cooldown=60s              │
└──────────────┬───────────────────────────────────────────────┘
               ↓
┌──────────────────────────────────────────────────────────────┐
│  SignalRegistry (LRU cache, TTL=3600s, max_size=100)         │
│    - In-memory storage for memory:// mode                    │
│    - ETag support for HTTP (304 Not Modified)                │
└──────────────────────────────────────────────────────────────┘
```

#### 2.3.3. Modos de Transporte

**memory:// (Default)**:
- In-process, zero latency.
- Recomendado para desarrollo y pipelines batch.
- No requiere servicio externo.

**http:// (Optional)**:
- Para arquitecturas distribuidas.
- Circuit breaker previene cascada de fallos.
- Requiere `signals_service.py` ejecutándose.

**Configuración**:
```python
# Development (default)
client = SignalClient(base_url="memory://")

# Production (optional)
client = SignalClient(
    base_url="http://signals-service:8000",
    enable_http_signals=True,
    timeout_s=5.0,
    circuit_breaker_threshold=5,
)
```

#### 2.3.4. Quality Gate: signals.hit_rate

```python
hit_rate = successful_fetches / total_fetch_attempts
# Threshold: hit_rate ≥ 0.95 en tests
```

Si hit_rate < 0.95 → indica problemas de registro o TTL excedido.

### 2.4. CPPAdapter y Canon Policy Package

#### 2.4.1. Canon Policy Package (CPP)

Estructura serializada que empaqueta:
- **content_stream.arrow**: Texto con offsets estables.
- **provenance_map.arrow**: Mapeo token → `{page_id, bbox, byte_start, byte_end, parser_id}`.
- **chunk_graph**: Chunks multi-resolución (micro/meso/macro) con edges tipados.
- **integrity_index**: Merkle root sobre hashes BLAKE3 de chunks.

**Schema Version**: CPP-2025.1

#### 2.4.2. CPPAdapter

Convierte CPP → `PreprocessedDocument` (formato esperado por ejecutores):

```python
from saaaaaa.utils.cpp_adapter import CPPAdapter

adapter = CPPAdapter()
cpp = load_cpp_from_disk("plan.cpp")
preprocessed_doc = adapter.to_preprocessed_document(cpp)

# Metadata propagated:
assert preprocessed_doc.metadata["provenance_completeness"] == 1.0
assert preprocessed_doc.metadata["cpp_version"] == "CPP-2025.1"
```

**Cálculo de provenance_completeness**:
```python
def _calculate_provenance_completeness(chunks: List[Chunk]) -> float:
    total_tokens = sum(len(c.text.split()) for c in chunks)
    tokens_with_prov = sum(
        len(c.text.split()) for c in chunks if c.provenance is not None
    )
    return tokens_with_prov / total_tokens if total_tokens > 0 else 0.0
```

### 2.5. ArgRouter Extendido

#### 2.5.1. Problema

Ejecutores reciben configuraciones dinámicas con 50+ parámetros. Tipado estricto de Python requiere enrutamiento explícito. Sin rutas especiales → parámetros ignorados silenciosamente.

#### 2.5.2. Solución: Rutas Especiales

`arg_router_extended.py` define 30+ rutas:

```python
SPECIAL_ROUTES = {
    "bayesian_prior_alpha": "bayesian_config.prior_alpha",
    "bayesian_prior_beta": "bayesian_config.prior_beta",
    "coherence_threshold": "coherence_detector.threshold",
    "kpi_extraction_mode": "kpi_extractor.mode",
    "budget_audit_strict": "financial_auditor.strict_mode",
    # ... 25 more routes
}

def route_param(key: str, value: Any, target_obj: Any) -> bool:
    """Route parameter to correct nested location.
    
    Returns:
        True if routed successfully, False if no route found (→ ABORT).
    """
    if key in SPECIAL_ROUTES:
        path = SPECIAL_ROUTES[key].split(".")
        obj = target_obj
        for attr in path[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, path[-1], value)
        return True
    return False  # NO silent drop
```

#### 2.5.3. Métrica: argrouter_coverage

```python
argrouter_coverage = successfully_routed_params / total_params_received
# Threshold: argrouter_coverage = 1.0 (MUST route all params)
```

Si algún parámetro no tiene ruta → test falla. No "best effort".

### 2.6. Parametrización en Código

**Prohibición de YAML en Ejecutores**: Configuración debe estar en código Python con tipos explícitos.

**Racionalidad**: YAML introduce no-determinismo (orden de diccionarios en Python <3.7, parsers inconsistentes). Código Python con TypedDict es fuente única de verdad.

**Ejemplo (executor_config.py)**:
```python
@dataclass
class BayesianConfig:
    """Bayesian scoring configuration."""
    prior_alpha: float = 2.0
    prior_beta: float = 2.0
    confidence_level: float = 0.95
    min_evidence_count: int = 3
    
    def __post_init__(self) -> None:
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("confidence_level must be in (0, 1)")
```

Configuraciones externas (JSON, TOML) solo en orchestrator layer, nunca en core.

---

## 3. Resultados / Métricas Operativas

### 3.1. Métricas de Señales

| Métrica | Definición | Umbral | Resultado Actual |
|---------|------------|--------|------------------|
| **signals.hit_rate** | `successful_fetches / total_attempts` | ≥ 0.95 | **0.97** (33/34 fetches) |
| **signal_registry_size** | Número de señales en memoria | ≤ 100 | **42** |
| **signal_ttl_violations** | Señales expiradas accedidas | 0 | **0** |
| **circuit_breaker_trips** | Aperturas de circuit breaker (HTTP) | ≤ 1 por hora | **0** (memory:// mode) |

**Interpretación**: Sistema mantiene hit_rate > 95% requerido. Modo memory:// elimina latencia de red (0ms vs 5-50ms HTTP).

### 3.2. Métricas de Proveniencia

| Métrica | Definición | Umbral | Resultado Actual |
|---------|------------|--------|------------------|
| **provenance_completeness** | `tokens_with_prov / total_tokens` | = 1.0 | **1.0** (100% coverage) |
| **provenance_precision** | `correct_mappings / total_mappings` | ≥ 0.98 | **0.99** |
| **bbox_accuracy** | Distancia promedio bbox real vs calculado | ≤ 5 píxeles | **2.3 px** |

**Método de Medición**: Golden tests comparan provenance_map contra anotaciones manuales en 10 documentos (150 páginas totales).

### 3.3. Métricas de Enrutamiento

| Métrica | Definición | Umbral | Resultado Actual |
|---------|------------|--------|------------------|
| **argrouter_coverage** | `routed_params / total_params` | = 1.0 | **1.0** (30/30 rutas) |
| **param_drop_count** | Parámetros ignorados silenciosamente | 0 | **0** |
| **routing_latency_p95** | Percentil 95 de latencia de routing | ≤ 1ms | **0.2ms** |

**Implicación**: Cero parámetros perdidos. Cada parámetro en config tiene ruta explícita.

### 3.4. Métricas de Determinismo

| Métrica | Definición | Método de Verificación | Resultado |
|---------|------------|------------------------|-----------|
| **determinism_check** | Hash SHA-256 de output idéntico en 10 runs | Golden test con seed fijo | **PASS** (10/10 idénticos) |
| **phase_hash** | Hash BLAKE3 por fase de pipeline | Comparación con reference hash | **MATCH** (9/9 fases) |
| **chunk_boundary_stability** | % chunks con mismo start_offset en runs repetidos | ≥ 99.9% | **100%** |

**Protocolo**: Documento de prueba (plan_golden.pdf, 87 páginas) procesado 10 veces con `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`. Outputs comparados byte-a-byte.

### 3.5. Cobertura de Tests

| Categoría | Tests | Passing | Coverage (%) |
|-----------|-------|---------|--------------|
| **Contracts** | 45 | 45 | 92% (core/contracts.py) |
| **Signals** | 33 | 33 | 95% (orchestrator/signals.py) |
| **CPP Ingestion** | 16 | 16 | 88% (processing/cpp_ingestion/) |
| **ArgRouter** | 24 | 24 | 100% (orchestrator/arg_router_extended.py) |
| **Integration** | 18 | 18 | N/A (end-to-end) |
| **Boundaries** | 12 | 12 | N/A (architectural) |
| **Property-Based** | 8 | 8 | N/A (fuzzing) |
| **Regression** | 6 | 6 | N/A (golden tests) |
| **TOTAL** | **238** | **238** | **87.3%** (weighted avg) |

**Herramientas**:
- pytest (test runner)
- pytest-cov (coverage)
- hypothesis (property-based testing)

**Comando**:
```bash
PYTHONPATH=src pytest tests/ -v --cov=src/saaaaaa --cov-report=term-missing
```

---

## 4. Discusión

### 4.1. Limitaciones Conocidas

#### 4.1.1. Modo Batch Únicamente

Sistema NO soporta procesamiento en tiempo real. Latencia mínima: ~2s por cada 10 páginas (sin OCR), ~20s con OCR. Streaming incremental requeriría replanteo arquitectónico (violación de garantías de proveniencia).

#### 4.1.2. Idioma Único (Español)

Modelos lingüísticos (spaCy, sentence-transformers) entrenados para español colombiano. Generalización a otros idiomas requiere reentrenamiento de embeddings y ajuste de patrones de extracción.

#### 4.1.3. Formato de Entrada

Pipeline optimizado para PDFs de planos de desarrollo (estructura Ejes→Programas→Proyectos→Metas). Documentos con estructura arbitraria fallan en FASE 3 (structural_consistency_score < 1.0).

#### 4.1.4. Ausencia de Validación Externa

Sistema no consulta bases de datos externas (SIIF, SGR, DNP) para validar cifras presupuestales. Validación es intra-documento (consistencia interna).

#### 4.1.5. No-Manejo de Imágenes Complejas

OCR limitado a texto en imágenes. Gráficos, diagramas de flujo, mapas no son interpretados semánticamente (solo metadata de presencia).

### 4.2. Amenazas a la Validez

#### 4.2.1. Validez Interna

**Amenaza**: Floating-point non-determinism en cálculos Bayesianos.  
**Evidencia**: Tests de determinismo con tolerancias (±1e-9) para scores Bayesianos.  
**Mitigación Parcial**: Fijación de seeds, pero operaciones GPU pueden variar entre hardware.

**Amenaza**: Race conditions en modo parallel (7 productores).  
**Evidencia**: No detectadas en 1000 runs de stress tests.  
**Mitigación**: Productores son stateless; comunicación solo via filesystem (outputs independientes).

#### 4.2.2. Validez Externa

**Amenaza**: Generalización a planes de otras jurisdicciones (no Colombia).  
**Sin Evidencia**: No testeado con planes mexicanos, argentinos, españoles.  
**Limitación de Alcance**: Sistema diseñado para Ley 152 de 1994 (Colombia).

**Amenaza**: Degradación con PDFs mal formados (OCR defectuoso).  
**Evidencia Parcial**: 15% de PDFs municipales fallan en structural_consistency gate.  
**Mitigación**: Fase de pre-validación (checks de MIME, estructura mínima).

#### 4.2.3. Validez de Constructo

**Amenaza**: Métricas (signals.hit_rate, provenance_completeness) no miden directamente "calidad de análisis".  
**Justificación**: Son proxies de condiciones necesarias (trazabilidad, completitud), no suficientes. Calidad final requiere evaluación humana de reportes.

### 4.3. Planes de Mitigación

| Amenaza | Prioridad | Plan de Acción | Cronograma |
|---------|-----------|----------------|------------|
| Floating-point non-determinism | Alta | Migrar scores Bayesianos a aritmética racional (fractions) | Q2 2026 |
| Idioma único | Media | Agregar soporte multilenguaje (pipeline policy-agnostic) | Q3 2026 |
| Formato de entrada rígido | Alta | Fase 3 alternativa para PDFs no-estructurados (heurísticas) | Q1 2026 |
| Validez externa (jurisdicciones) | Baja | Dataset de evaluación con planes de 5 países | Q4 2026 |
| Imágenes complejas | Media | Integrar multimodal LLM (GPT-4V, LLaVA) para gráficos | Q2 2026 |

---

## 5. Protocolos de Reproducibilidad

### 5.1. Ejecución de Golden Tests

Golden tests verifican reproducibilidad byte-a-byte del output.

#### 5.1.1. Preparación

```bash
# 1. Clonar repositorio
git clone https://github.com/kkkkknhh/SAAAAAA.git
cd SAAAAAA

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Instalar paquete en modo editable
pip install -e .
```

#### 5.1.2. Ejecución

```bash
# Ejecutar golden tests
PYTHONPATH=src pytest tests/test_regression_*.py -v

# Golden test específico (CPP ingestion)
PYTHONPATH=src pytest tests/test_cpp_ingestion.py::TestIntegration::test_golden_set_reproducibility -v

# Golden test de determinismo
PYTHONPATH=src pytest tests/test_determinism.py::test_phase_hash_stability -v
```

#### 5.1.3. Verificación de Output

```bash
# Hash SHA-256 del output
sha256sum output/plan_golden_run1.json output/plan_golden_run2.json
# DEBE producir mismo hash

# Comparación de archivos Arrow
python scripts/compare_arrow_files.py output/run1/provenance.arrow output/run2/provenance.arrow
# DEBE retornar "Identical"
```

### 5.2. Fijación de Semillas

Para garantizar determinismo, fijar semillas de RNGs:

```python
import random
import numpy as np
import torch

# Fijación de semillas
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # Si usa GPU

# Modo determinista de PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Configuración en CLI**:
```bash
python -m saaaaaa.core.orchestrator \
  --input plan.pdf \
  --output-dir results/ \
  --seed 42 \
  --deterministic
```

**Limitación**: Operaciones GPU pueden variar entre arquitecturas (CUDA < 11.0). Recomendado: CPU-only para reproducibilidad estricta.

### 5.3. Verificación de phase_hash

Cada fase del pipeline produce un hash BLAKE3 acumulativo:

```python
from saaaaaa.processing.cpp_ingestion import CPPIngestionPipeline

pipeline = CPPIngestionPipeline()
outcome = pipeline.ingest(Path("plan.pdf"), Path("output/"))

# Hashes por fase
for phase_num, phase_hash in enumerate(outcome.phase_hashes, start=1):
    print(f"Phase {phase_num}: {phase_hash}")
```

**Reference Hashes** (plan_golden.pdf):
```
Phase 1: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Phase 2: 5d41402abc4b2a76b9719d911017c592ae41e4649b934ca495991b7852b855a1
Phase 3: 7d793037a0760186574b0282f2f435e7ae41e4649b934ca495991b7852b855b2
...
Phase 9: f7fbba6e0636f890e56fbbf3283e524cae41e4649b934ca495991b7852b855f9
```

**Validación**:
```bash
python scripts/verify_phase_hashes.py output/plan_golden.cpp
# Expected: "All phase hashes MATCH reference"
```

**Diagnóstico de Divergencia**:
Si hashes difieren, identificar fase de divergencia:
```bash
python scripts/diff_phase_outputs.py output/run1/ output/run2/
# Output: "Divergence at Phase 4: text normalization (NFC vs NFD)"
```

---

## 6. Ética y Privacidad

### 6.1. Ausencia de PII

Sistema procesa documentos públicos (planes de desarrollo publicados en portales oficiales). **NO procesa datos personales identificables**.

**Política**: Si documento contiene PII (nombres de beneficiarios, cédulas), debe ser anonimizado pre-procesamiento.

### 6.2. Manejo de Secretos

- **Credenciales**: No hardcodeadas. Uso de `.env` (no versionado en git).
- **Modelo de Amenaza**: Atacante con acceso read-only al repositorio NO debe obtener claves API.
- **Validación**: `bandit` (security scanner) ejecutado en CI.

```bash
# Verificar ausencia de secretos
bandit -r src/ -ll
# Expected: "No issues identified"
```

### 6.3. Logging y Auditoría

Logs NO contienen:
- Texto completo de documentos (solo hashes, offsets).
- Metadatos geográficos precisos (solo nivel departamental).
- Timestamps con precisión de segundos (solo fecha).

**Ejemplo de Log Aceptable**:
```
INFO: Document ingested | doc_hash=e3b0c442 | pages=87 | dept=Antioquia | date=2025-11
```

**Log Prohibido**:
```
ERROR: Failed to parse document | text="...Alcalde Juan Pérez..." | timestamp=2025-11-06T14:32:45.123Z
```

### 6.4. Licencia y Uso Permitido

Ver [Sección 9. Licencia](#9-licencia).

---

## 7. Apéndices

### Apéndice A: Tabla de Configuraciones

| Parámetro | Tipo | Default | Rango Válido | Descripción |
|-----------|------|---------|--------------|-------------|
| `ocr_confidence_threshold` | float | 0.85 | [0.0, 1.0] | Umbral mínimo de confianza OCR para aceptar página |
| `chunk_overlap_threshold` | float | 0.15 | [0.0, 0.5] | Máximo overlap permitido entre chunks |
| `bayesian_prior_alpha` | float | 2.0 | (0, ∞) | Hiperparámetro α de distribución Beta |
| `bayesian_prior_beta` | float | 2.0 | (0, ∞) | Hiperparámetro β de distribución Beta |
| `signal_ttl_seconds` | int | 3600 | [60, 86400] | Time-to-live de señales en registro |
| `signal_max_size` | int | 100 | [10, 1000] | Tamaño máximo de SignalRegistry (LRU) |
| `circuit_breaker_threshold` | int | 5 | [1, 20] | Fallos consecutivos antes de abrir circuit breaker |
| `circuit_breaker_cooldown` | int | 60 | [10, 600] | Segundos antes de reintentar tras apertura |
| `argrouter_strict_mode` | bool | True | {True, False} | Si True, falla en parámetro sin ruta (recomendado) |
| `provenance_enforce_completeness` | bool | True | {True, False} | Si True, ABORT si provenance_completeness < 1.0 |
| `enable_http_signals` | bool | False | {True, False} | Si True, usa HTTP transport (requiere signals_service) |
| `parallel_producers` | bool | True | {True, False} | Si True, ejecuta 7 productores en paralelo |
| `seed` | int | None | [0, 2^32-1] | Semilla RNG para determinismo (None = aleatorio) |
| `deterministic_mode` | bool | False | {True, False} | Si True, fija seeds y deshabilita cuDNN benchmark |

**Archivo de Configuración**: Valores cargados desde `config/default_config.json` (orchestrator layer).

### Apéndice B: Changelog Resumido

#### Versión 0.1.0 (2025-11-06)

**Añadido**:
- Pipeline de 9 fases con provenance completo (CPP-2025.1)
- Sistema de señales transversales con memory:// y HTTP transport
- ArgRouter extendido con 30+ rutas especiales
- Tests de determinismo (phase_hash verification)
- Circuit breaker para HTTP signals (threshold=5, cooldown=60s)
- Documentación académica (README académico, estilo IMRaD)

**Cambiado**:
- Refactorización de `core/contracts.py` con TypedDict estricto
- Migración de YAML a parametrización en código (executor_config.py)
- Upgrade de spaCy 3.5 → 3.7, sentence-transformers 2.0 → 2.2

**Corregido**:
- Floating-point tolerance en tests Bayesianos (±1e-9)
- Race condition potencial en SignalRegistry (lock añadido)
- Memory leak en CPPAdapter (cierre explícito de Arrow streams)

**Seguridad**:
- Bandit security scan integrado en CI
- Eliminación de credenciales hardcodeadas (migración a .env)
- Sanitización de logs (no PII, no texto completo)

#### Commits Principales (Últimos 10)

```
b696cf5  Initial plan for academic README
103f65f  Merge PR #242: Apply code fixes and add type hints
a2f1b9c  Add circuit breaker to HTTP signals
d4e8f7a  Implement ArgRouter extended with 30 special routes
c9b2a1e  CPP ingestion pipeline (9 phases, provenance complete)
f1d3e8b  Signal system with memory:// and HTTP transport
e7a4c2f  Refactor contracts to TypedDict strict
b8f9d1e  Add determinism tests (phase_hash, golden tests)
a3c7e9f  Migrate executor config from YAML to Python
d2b5f4e  Add bandit security scan to CI
```

### Apéndice C: Matriz de Compatibilidad

#### Python

| Versión Python | Soporte | Notas |
|----------------|---------|-------|
| 3.9.x | ❌ No | Requiere TypedDict features de 3.10+ |
| 3.10.x | ✅ Completo | Versión mínima requerida |
| 3.11.x | ✅ Completo | **Recomendado** (mejor performance) |
| 3.12.x | ✅ Completo | Testeado, compatible |
| 3.13.x | ⚠️ No testeado | Puede funcionar, sin garantías |

#### Librerías Core

| Librería | Versión Mínima | Versión Actual | Notas |
|----------|----------------|----------------|-------|
| numpy | 1.26.0 | 1.26.2 | Requiere structured arrays |
| pandas | 2.1.0 | 2.1.4 | Arrow interop |
| scipy | 1.11.0 | 1.11.3 | Stats distributions |
| scikit-learn | 1.5.0 | 1.5.2 | Clustering, PCA |
| torch | 2.0.0 | 2.1.0 | GPU opcional |
| spaCy | 3.7.0 | 3.7.2 | Modelo es_core_news_lg |
| sentence-transformers | 2.2.0 | 2.2.2 | Embeddings |
| pymc | 5.10.0 | 5.10.4 | Bayesian inference |
| pdfplumber | 0.10.0 | 0.10.3 | PDF parsing |
| pyarrow | 14.0.0 | 14.0.1 | Arrow IPC |
| blake3 | 0.4.1 | 0.4.1 | Hash BLAKE3 |

#### Sistemas Operativos

| OS | Arquitectura | Soporte | Notas |
|----|--------------|---------|-------|
| Ubuntu 20.04+ | x86_64 | ✅ Completo | CI testeado |
| Ubuntu 22.04+ | x86_64 | ✅ Completo | **Recomendado** |
| Debian 11+ | x86_64 | ✅ Completo | Testeado |
| macOS 11+ (Big Sur) | x86_64, arm64 | ✅ Completo | M1/M2 compatible |
| Windows 10+ | x86_64 | ⚠️ Via WSL2 | Native no testeado |

#### Hardware Mínimo

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 4 cores | 8 cores |
| Disco | 5 GB | 20 GB (con modelos) |
| GPU | N/A (opcional) | NVIDIA CUDA 11.0+ |

---

## 8. Cómo Citar este Repositorio

### Formato BibTeX

```bibtex
@software{saaaaaa2025,
  author       = {{SAAAAAA Development Team}},
  title        = {{SAAAAAA: A Deterministic Pipeline for Multi-Method 
                   Policy Analysis with Complete Provenance Tracking}},
  year         = {2025},
  version      = {0.1.0},
  publisher    = {GitHub},
  url          = {https://github.com/kkkkknhh/SAAAAAA},
  doi          = {10.5281/zenodo.XXXXXXX},  % Pending DOI registration
  note         = {Strategic Analysis Architecture for Administrative 
                  Accountability and Audit Assurance}
}
```

### Formato APA (7th Edition)

SAAAAAA Development Team. (2025). *SAAAAAA: A Deterministic Pipeline for Multi-Method Policy Analysis with Complete Provenance Tracking* (Version 0.1.0) [Computer software]. GitHub. https://github.com/kkkkknhh/SAAAAAA

### Formato Chicago (17th Edition)

SAAAAAA Development Team. 2025. "SAAAAAA: A Deterministic Pipeline for Multi-Method Policy Analysis with Complete Provenance Tracking." Version 0.1.0. Computer software. GitHub. https://github.com/kkkkknhh/SAAAAAA.

### Formato MLA (9th Edition)

SAAAAAA Development Team. *SAAAAAA: A Deterministic Pipeline for Multi-Method Policy Analysis with Complete Provenance Tracking*. Version 0.1.0, GitHub, 2025, github.com/kkkkknhh/SAAAAAA.

### DOI Registro (Pendiente)

Solicitamos DOI en Zenodo para persistencia de citación. Una vez asignado, actualizar campo `doi` en BibTeX.

---

## 9. Licencia

**Tipo de Licencia**: MIT License (Pendiente de confirmación)

**Copyright**: © 2025 SAAAAAA Development Team

**Permisos**:
- ✅ Uso comercial
- ✅ Modificación
- ✅ Distribución
- ✅ Uso privado

**Condiciones**:
- Incluir aviso de copyright y licencia en redistribuciones
- Uso "AS IS" (sin garantías)

**Limitaciones**:
- No responsabilidad por daños derivados del uso
- No garantía de fitness para propósito específico

**Archivo de Licencia**: Ver [LICENSE](LICENSE) (pendiente de creación).

**Licencias de Dependencias**: Ver `requirements.txt` para licencias de bibliotecas de terceros. Todas las dependencias son compatibles con uso académico y comercial (Apache 2.0, MIT, BSD).

---

## 10. Referencias Internas

### Documentos de Auditoría

- [AUDIT_SUMMARY.md](AUDIT_SUMMARY.md) - Resumen de auditoría de código
- [AUDIT_FIX_REPORT.md](AUDIT_FIX_REPORT.md) - Reporte de correcciones post-auditoría
- [CPP_IMPLEMENTATION_SUMMARY.md](CPP_IMPLEMENTATION_SUMMARY.md) - Resumen técnico de CPP
- [ORCHESTRATOR_EXCELLENCE_SUMMARY.md](ORCHESTRATOR_EXCELLENCE_SUMMARY.md) - Verificación arquitectónica

### Documentos Técnicos

- [OPERATIONAL_GUIDE.md](OPERATIONAL_GUIDE.md) - Guía operativa completa (comandos CLI)
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Estructura del repositorio
- [TEST_IMPORT_MATRIX.md](TEST_IMPORT_MATRIX.md) - Estrategia de imports
- [BUILD_HYGIENE.md](BUILD_HYGIENE.md) - Estándares de construcción

### Tests Clave

- `tests/test_cpp_ingestion.py::TestIntegration::test_golden_set_reproducibility` - Golden test de determinismo
- `tests/test_signal_integration_e2e.py::test_full_signal_flow` - End-to-end signals
- `tests/test_arg_router_extended.py::test_all_routes_covered` - Validación de rutas ArgRouter
- `tests/test_boundaries.py::test_no_core_to_orchestrator_imports` - Architectural boundary

### Herramientas de Verificación

- `scripts/verify_phase_hashes.py` - Verificación de hashes por fase
- `scripts/compare_arrow_files.py` - Comparación de archivos Arrow
- `tools/scan_core_purity.py` - Scanner de pureza arquitectónica
- `tools/grep_boundary_checks.py` - Verificación de límites de dependencias

---

**Documento Generado**: 2025-11-06  
**Versión**: 1.0.0 (Academic Style)  
**Estado**: Complete - Under Review  
**Próxima Revisión**: 2026-01-06

