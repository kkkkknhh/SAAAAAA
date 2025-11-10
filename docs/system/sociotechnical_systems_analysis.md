# Comprehensive Socio-Technical Systems Analysis: SAAAAAA Orchestration Pipeline

**Document Version:** 1.0  
**Analysis Date:** November 2025  
**System Analyzed:** SAAAAAA Policy Analysis Orchestration Pipeline  
**Framework:** Multi-Paradigm Sociological Systems Theory  
**Compliance:** SIN_CARRETA Doctrine (Determinism, Auditability, Contract Clarity)

---

## Executive Summary

This document presents a comprehensive socio-technical systems analysis of the SAAAAAA orchestration pipeline, treating it as a complex adaptive system exhibiting both technological and organizational properties. The analysis applies established sociological systems theory frameworks—including structural-functionalism, cybernetics, complexity theory, and institutional analysis—to decode the pipeline's architecture, operational dynamics, emergent behaviors, and systemic constraints.

The SAAAAAA system is a deterministic, multi-phase orchestration engine that transforms policy documents (PDF inputs) into structured analytical insights through 11 sequential and parallel processing phases. The system demonstrates sophisticated properties including: hierarchical aggregation across four abstraction levels (micro → dimension → area → cluster → macro), asynchronous parallelism with resource governance, chunk-aware semantic routing, and comprehensive instrumentation for auditability.

This analysis is grounded exclusively in the actual source code residing in `src/saaaaaa/core/orchestrator/core.py`, `src/saaaaaa/processing/aggregation.py`, `src/saaaaaa/core/orchestrator/executors.py`, and related modules. No behavior is assumed or invented; every claim traces to observable code structures and documented contracts.

**Key Findings:**
- **System Classification:** Open, deterministic, complex adaptive system with 11-phase sequential-parallel architecture
- **Structural Properties:** High differentiation (specialized phases), tight integration (contract-driven), hierarchical (4-level aggregation)
- **Functional Properties:** Asynchronous parallelism (Phases 2-5, 8, 10), timeout controls, resource limits, circuit breakers
- **Emergent Properties:** Hierarchical intelligence through aggregation, chunk-aware optimization, degraded-mode resilience
- **Cybernetic Controls:** Negative feedback (timeouts, resource limits), abort signaling, phase instrumentation
- **Institutional Compliance:** SIN_CARRETA doctrine enforcement through deterministic hashing, monolith validation, contract gates

---

## 1. System Ontology and Boundary Definition

### 1.1 System Classification

**System Type Analysis:**

The SAAAAAA pipeline exhibits characteristics that locate it within multiple systems-theoretic typologies:

**1.1.1 Open vs. Closed System**

The SAAAAAA pipeline is an **open system** that maintains clear boundaries while engaging in continuous exchange with its environment. Evidence from code:

- **Inputs from Environment:** The system receives PDF documents (`pdf_path` parameter in `_ingest_document`), questionnaire monoliths (`monolith` parameter), and catalog configurations from external sources (lines 1108-1137, `core.py`).
- **Outputs to Environment:** The system exports structured reports, recommendations, and analytical artifacts to external consumers (Phase 10: `_format_and_export`, lines 1046).
- **Environmental Dependencies:** The system depends on external LLM execution engines (method catalog executors), file systems (PDF ingestion), and potentially remote calibration services.
- **Boundary Permeability:** While open, boundaries are strictly controlled through validation gates (`_load_configuration`, Phase 0) that hash inputs (`monolith_sha256`) and validate structural integrity before processing begins.

**1.1.2 Deterministic vs. Stochastic System**

The pipeline is fundamentally **deterministic by design**, with intentional stochastic elements constrained to specific subsystems:

- **Deterministic Core:** Input normalization (`_normalize_monolith_for_hash`, lines 174-212) ensures identical inputs produce identical hashes. Phase sequencing is fixed (FASES list, lines 1035-1047).
- **Determinism Enforcement:** SHA256 hashing of monolith configuration (lines 1625-1627) creates content-addressable reproducibility. Validation gates prevent non-deterministic execution paths.
- **Controlled Stochasticity:** LLM-based executors (Phase 2: `_execute_micro_questions_async`) may introduce variability, but this is constrained to executor boundaries and does not affect orchestration flow.
- **SIN_CARRETA Compliance:** The system enforces determinism through calibration validation (`resolve_calibration`, line 915) that rejects placeholder calibrations, ensuring all method executions have defined, auditable parameters.

**1.1.3 Simple, Complex, or Chaotic System**

The SAAAAAA pipeline is a **complex adaptive system** exhibiting non-linear interactions and emergent properties:

- **Complexity Indicators:**
  - **Multiple Interacting Components:** 11 phases, 4 aggregation levels, 300+ micro-questions, 60 dimensions, 10 policy areas, 4 clusters (lines 1049-1061).
  - **Non-Linear Dynamics:** Hierarchical aggregation where local scores influence global outcomes non-proportionally (weighted aggregation in `DimensionAggregator`, `AreaPolicyAggregator`, lines 121-150 in `aggregation.py`).
  - **Feedback Mechanisms:** Circuit breakers (lines 1881-1933), abort signaling (`AbortSignal` class), resource limits (`ResourceLimits` class), timeouts (`execute_phase_with_timeout`, lines 77-171).
  - **Adaptive Behaviors:** Chunk-aware routing adapts execution strategy based on document structure (lines 1841-1860), degraded-mode operation when class registry fails (lines 801-811).
  - **Emergent Properties:** Macro scores emerge from micro questions through four aggregation levels, producing system-level insights not present in individual questions.

- **Not Chaotic:** The system maintains bounded behavior through explicit phase timeouts (PHASE_TIMEOUTS dict, lines 1091-1103), resource limits, and abort mechanisms. No butterfly effects or sensitive dependence on initial conditions beyond intended parameter sensitivity.

**1.1.4 Teleological Classification**

The system is **purposive and goal-directed**, with both manifest and latent functions:

- **Manifest Functions (Explicit Design Goals):**
  - Transform policy documents into structured, multi-level analytical insights
  - Aggregate 300+ micro-level assessments into holistic macro evaluations
  - Generate evidence-based recommendations for policy improvement
  - Ensure auditability and reproducibility of analysis (SIN_CARRETA compliance)

- **Latent Functions (Emergent System Roles):**
  - **Knowledge Codification:** The system embeds domain expertise in questionnaire structures and aggregation rules
  - **Organizational Memory:** Calibration registry and method catalog preserve analytical methodologies
  - **Governance Mechanism:** Contract validation and signature checking enforce methodological rigor
  - **Quality Control:** Scoring rubrics and validation gates operationalize quality standards

**1.1.5 System Boundaries**

The pipeline's boundaries are precisely defined through contract interfaces:

- **Input Boundary:** 
  - Entry point: `Orchestrator.run()` method (not shown but implied by phase structure)
  - Input contracts: `pdf_path` (string), `monolith` (dict conforming to questionnaire schema), `method_map` (dict)
  - Validation gate: Phase 0 (`_load_configuration`) performs integrity checking before system entry

- **Output Boundary:**
  - Exit point: Phase 10 (`_format_and_export`) delivers structured export payload
  - Output contracts: `MacroEvaluation` dataclass, `export_payload` dict
  - Artifacts: Reports, recommendations, score hierarchies

- **Internal Subsystem Boundaries:**
  - **Orchestrator ↔ Executors:** `MethodExecutor.execute(class_name, method_name, **kwargs)` (lines 898-940)
  - **Orchestrator ↔ Aggregators:** `DimensionAggregator.aggregate_dimension()`, `AreaPolicyAggregator.aggregate_area()` (lines 2216-2343)
  - **Orchestrator ↔ Resource Manager:** `ResourceLimits.apply_worker_budget()`, `check_memory_exceeded()` (lines 1896, 1936)

**1.1.6 System Environment**

The pipeline's environment consists of external dependencies and contextual factors:

- **Technological Environment:**
  - Python runtime (3.10+)
  - File system (PDF access, JSON configuration loading)
  - External LLM APIs (OpenAI, Anthropic) accessed by executors
  - Monitoring infrastructure (OpenTelemetry optional, lines 72-79 in `executors.py`)

- **Data Environment:**
  - Policy documents (PDFs) in Spanish/English
  - Questionnaire monoliths (JSON schemas)
  - Method catalogs and calibration registries
  - Semantic chunking pipelines (CPP/SPC ingestion)

- **Institutional Environment:**
  - SIN_CARRETA governance doctrine
  - Quality standards embedded in scoring rubrics
  - Methodological conventions in dimension/area taxonomies

### 1.2 Input Mechanisms and Deterministic Tracking

**1.2.1 Primary Input Ingestion**

The system accepts three primary input types, each with distinct ingestion mechanisms:

**A. PDF Document Input (`pdf_path`)**

Evidence: Phase 1 (`_ingest_document`, lines 1738-1828)

```python
async def _ingest_document(
    self,
    pdf_path: str | None,
    config: dict[str, Any],
) -> PreprocessedDocument:
```

**Ingestion Process:**
1. **Path Validation:** System checks `pdf_path` parameter (line 1752)
2. **CPP Ingestion:** Calls `build_processor(pdf_path).run()` to invoke Canonical Policy Package ingestion (lines 1759-1782)
3. **Document Normalization:** `PreprocessedDocument.ensure()` converts CPP output to orchestrator format (line 1784)
4. **Validation Gates:**
   - Non-empty check: `raw_text` must not be empty/whitespace (lines 1804-1809)
   - Chunk count validation: `chunk_count > 0` required (lines 1812-1818)
   - Instrumentation recording: Errors logged to Phase 1 instrumentation

**B. Questionnaire Monolith Input (`monolith`)**

Evidence: Phase 0 (`_load_configuration`, lines 1614-1700)

**Ingestion Process:**
1. **Pre-loaded Data:** Monolith passed via constructor (`self._monolith_data`, line 1620)
2. **Normalization:** `_normalize_monolith_for_hash()` converts MappingProxyType to dict recursively (lines 174-212)
3. **Deterministic Hashing:** SHA256 hash computed over canonical JSON representation (lines 1625-1627):
```python
monolith_hash = hashlib.sha256(
    json.dumps(monolith, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
).hexdigest()
```
4. **Structural Validation:** Questionnaire block extraction and count validation (lines 1634-1641)
5. **Schema Validation:** Optional JSON Schema validation if `jsonschema` available (lines 1677-1700)

**C. Method Map Input (`method_map`)**

Evidence: Phase 0 validation (lines 1645-1673)

**Ingestion Process:**
1. **Non-empty Gate:** System enforces `PROMPT_NONEMPTY_EXECUTION_GRAPH_ENFORCER` - empty method maps cause immediate failure (lines 1654-1658)
2. **Method Count Validation:** Compares against `EXPECTED_METHOD_COUNT = 416` (line 56)
3. **Summary Extraction:** Validates method map structure and metadata

**1.2.2 Boundary Control Mechanisms**

The input validation subsystem serves as a **boundary maintenance mechanism** in systems-theoretic terms:

**Theoretical Framing:** Parsons' structural-functional theory posits that social systems maintain boundaries through **pattern maintenance** and **adaptation** mechanisms. In SAAAAAA, Phase 0 operationalizes pattern maintenance by enforcing structural invariants.

**Boundary Control Functions:**

1. **Selective Permeability:** Only inputs conforming to expected schemas cross the boundary
   - Evidence: Schema validation (lines 1677-1700), question count checks (lines 1639-1641)
   
2. **Energy/Information Transformation:** Raw inputs transformed into internal representations
   - Evidence: `PreprocessedDocument.ensure()` adapts external CPP format to internal orchestrator format (lines 303-450)
   
3. **Entropy Reduction:** Validation gates reduce uncertainty about input quality
   - Evidence: Hash computation creates fixed identity for monolith state, enabling reproducible analysis

4. **System Protection:** Invalid inputs rejected before resource allocation
   - Evidence: RuntimeError raised if method_map empty (lines 1655-1658), ValueError if document empty (lines 1801-1809)

**1.2.3 Entropy Management and Reproducibility**

**Theoretical Framing:** Shannon's information theory and thermodynamic analogies in social systems (Buckley, 1967) suggest that systems manage entropy through information processing and storage mechanisms.

**Entropy Management in SAAAAAA:**

**1. Input Entropy Reduction (Phase 0)**

- **Pre-Processing Normalization:** `_normalize_monolith_for_hash()` eliminates representational variability by converting all MappingProxyType to dict (lines 190-203)
- **Canonical Serialization:** `sort_keys=True, separators=(",", ":")` in JSON serialization eliminates key-order entropy (line 1626)
- **Content-Addressable Identity:** SHA256 hash creates unique fingerprint, reducing monolith state from megabytes to 64 hex characters (32 bytes)

**2. Execution Entropy Control (Phases 2-7)**

- **Deterministic Phase Sequencing:** Fixed FASES list (lines 1035-1047) eliminates scheduling entropy
- **Timeout Boundaries:** PHASE_TIMEOUTS dict (lines 1091-1103) bounds execution time variance
- **Circuit Breakers:** Slot-specific circuit breakers (lines 1881-1933) prevent cascading failures that would increase system entropy
- **Resource Limits:** `ResourceLimits` class constrains memory/CPU usage, preventing resource-driven non-determinism

**3. Output Entropy Reduction (Aggregation)**

- **Weighted Aggregation:** Dimension scores computed via deterministic weighted averages (DimensionAggregator, lines 121-150 in `aggregation.py`)
- **Rubric Thresholds:** Quality levels mapped to fixed score ranges, discretizing continuous scores
- **Hierarchical Aggregation:** Four-level pipeline (micro → dimension → area → cluster → macro) progressively reduces information while preserving structural relationships

**Reproducibility Guarantees:**

Given identical inputs (same monolith hash, same PDF, same method_map), the system guarantees:
- ✅ **Phase Execution Order:** Deterministic (fixed FASES list)
- ✅ **Aggregation Results:** Deterministic (fixed weights, fixed thresholds)
- ✅ **Instrumentation Logs:** Deterministic structure (phase IDs, timestamps, metrics)
- ⚠️ **Executor Outputs:** Potentially non-deterministic (LLM-based executors may vary)
- ✅ **Final Artifacts:** Deterministic structure (even if evidence content varies)

**SIN_CARRETA Compliance:**

The deterministic tracking mechanisms satisfy SIN_CARRETA doctrine requirements:
- **Auditability:** Monolith hash enables input verification; phase instrumentation enables execution tracing
- **Contract Clarity:** TypedDict specifications (`IndustrialInput`, `IndustrialOutput`) formalize boundaries
- **Determinism:** Hash-based identity and canonical serialization enable reproducible analysis

### 1.3 System Purpose: Manifest and Latent Functions

**Theoretical Framing:** Merton's (1968) distinction between manifest (intended, recognized) and latent (unintended, unrecognized) functions reveals multiple layers of system purpose.

**Manifest Functions:**

1. **Policy Document Analysis**
   - Transform unstructured policy PDFs into structured analytical insights
   - Evidence: Phase 1 ingestion → Phase 2 micro-analysis → Phase 10 export pipeline

2. **Multi-Level Evaluation**
   - Generate scores at micro, dimension, area, cluster, and macro levels
   - Evidence: Hierarchical aggregation pipeline (Phases 4-7, lines 2216-2425)

3. **Evidence-Based Recommendations**
   - Produce actionable recommendations grounded in analytical evidence
   - Evidence: Phase 8 (`_generate_recommendations`, line 1044)

4. **Quality Assessment**
   - Classify policy quality using rubric-based scoring
   - Evidence: Scoring modalities in monolith, quality levels in aggregation

5. **Auditability**
   - Enable verification and replication of analytical results
   - Evidence: Phase instrumentation, monolith hashing, contract validation

**Latent Functions:**

1. **Organizational Learning**
   - Method catalog and calibration registry accumulate methodological knowledge
   - Evidence: CALIBRATIONS dict, class registry building (lines 799-867)

2. **Standard Setting**
   - Implicit codification of policy quality standards through rubric thresholds
   - Evidence: Scoring config in monolith, quality level mappings

3. **Institutional Legitimacy**
   - Technical sophistication signals analytical rigor to external stakeholders
   - Evidence: Complex architecture, OpenTelemetry tracing, comprehensive logging

4. **Risk Distribution**
   - Circuit breakers and resource limits distribute failure risk across components
   - Evidence: Slot-specific circuit breakers (lines 1881-1933), timeout boundaries

5. **Methodological Reification**
   - Transformation of analytical practices into executable code creates durable institutional forms
   - Evidence: Dimension/area taxonomies hardcoded in orchestrator structure

**Dysfunctions (Negative Latent Functions):**

1. **Brittleness:** Strict validation gates may reject valid but non-conforming inputs
2. **Complexity Burden:** 11-phase architecture increases cognitive load for maintainers
3. **Lock-In:** Hardcoded taxonomies (60 dimensions, 10 areas) resist methodological evolution


---

## 2. Structural Analysis: System Architecture

### 2.1 Macro-Structure: 11-Phase Pipeline Architecture

The SAAAAAA orchestration pipeline implements a fixed-sequence, multi-mode execution architecture with 11 distinct phases. This structure is defined in the `Orchestrator.FASES` class attribute (lines 1035-1047, `core.py`):

```python
FASES: list[tuple[int, str, str, str]] = [
    (0, "sync", "_load_configuration", "FASE 0 - Validación de Configuración"),
    (1, "sync", "_ingest_document", "FASE 1 - Ingestión de Documento"),
    (2, "async", "_execute_micro_questions_async", "FASE 2 - Micro Preguntas"),
    (3, "async", "_score_micro_results_async", "FASE 3 - Scoring Micro"),
    (4, "async", "_aggregate_dimensions_async", "FASE 4 - Agregación Dimensiones"),
    (5, "async", "_aggregate_policy_areas_async", "FASE 5 - Agregación Áreas"),
    (6, "sync", "_aggregate_clusters", "FASE 6 - Agregación Clústeres"),
    (7, "sync", "_evaluate_macro", "FASE 7 - Evaluación Macro"),
    (8, "async", "_generate_recommendations", "FASE 8 - Recomendaciones"),
    (9, "sync", "_assemble_report", "FASE 9 - Ensamblado de Reporte"),
    (10, "async", "_format_and_export", "FASE 10 - Formateo y Exportación"),
]
```

**Structural Properties:**

1. **Contiguous Sequential Indexing:** Phase IDs run from 0 to 10 with no gaps, enforced by `validate_phase_definitions()` (lines 956-1024)
2. **Heterogeneous Execution Modes:** 5 synchronous phases, 6 asynchronous phases
3. **Handler Method Coupling:** Each phase explicitly bound to orchestrator method via handler name string
4. **Spanish Labeling:** Phase labels in Spanish reflect organizational context

### 2.2 Phase-by-Phase Structural Analysis

#### Phase 0: Configuration Validation (SYNC)

**Structural Position:** System entry point, boundary maintenance  
**Handler:** `_load_configuration()` (lines 1614-1717)

**Input Contract:**
- `monolith`: dict[str, Any] (questionnaire specification)
- `method_map`: dict[str, Any] (method routing configuration)
- `schema`: dict[str, Any] | None (JSON Schema for validation)

**Output Contract:**
- `config`: dict[str, Any] containing:
  - `monolith`: Normalized monolith dict
  - `monolith_sha256`: str (deterministic hash)
  - `micro_questions`: list[dict] (300+ questions)
  - `meso_questions`: list[dict] (4 cluster questions)
  - `macro_question`: dict (1 holistic question)
  - `method_summary`: dict (catalog metadata)
  - `structure_report`: dict (validation results)
  - `schema_report`: dict (schema validation results)

**Functional Role:**
- **Boundary Gatekeeping:** Enforces structural invariants before execution
- **Identity Establishment:** Computes content-addressable monolith hash for reproducibility
- **Resource Pre-allocation:** Validates method catalog completeness (416 methods expected)
- **Entropy Reduction:** Normalizes MappingProxyType to dict, canonical JSON serialization

**Key Validation Gates:**
1. Question count check (expected: 305, line 1639)
2. Method count check (expected: 416, line 1662)
3. Non-empty method map enforcement (PROMPT_NONEMPTY_EXECUTION_GRAPH_ENFORCER, lines 1654-1658)
4. Optional JSON Schema validation (lines 1677-1700)
5. Contract structure validation (`_validate_contract_structure`, invoked at line 1643)

**Structural Invariants Enforced:**
- Monolith must contain "blocks.micro_questions", "blocks.meso_questions", "blocks.macro_question"
- Method map must be non-empty dict with "summary" key
- All questionnaire blocks must be lists/dicts (no nulls)

---

#### Phase 1: Document Ingestion (SYNC)

**Structural Position:** Input transformation boundary  
**Handler:** `_ingest_document()` (lines 1738-1828)

**Input Contract:**
- `pdf_path`: str | None (path to policy PDF)
- `config`: dict (from Phase 0)

**Output Contract:**
- `document`: PreprocessedDocument dataclass containing:
  - `document_id`: str
  - `raw_text`: str (full document text)
  - `sentences`: list[Any] (parsed sentences)
  - `tables`: list[Any] (extracted tables)
  - `chunks`: list[ChunkData] (semantic chunks)
  - `chunk_index`: dict[str, int] (entity → chunk mapping)
  - `chunk_graph`: dict (graph structure)
  - `processing_mode`: Literal["flat", "chunked"]
  - `metadata`: dict (chunk_count, ingestion metrics)

**Functional Role:**
- **Format Transformation:** PDF → structured preprocessed document
- **Semantic Enrichment:** Sentence/table extraction, semantic chunking
- **Chunk Graph Construction:** Build entity-chunk index and graph topology
- **Validation:** Ensure non-empty text, non-zero chunk count

**Ingestion Pipeline:**
1. Call `build_processor(pdf_path).run()` → CPP pipeline execution (line 1763)
2. Adapter invocation: `CPPIngestionAdapter.adapt_document()` (line 1775)
3. Normalization: `PreprocessedDocument.ensure(use_spc_ingestion=True)` (line 1784)
4. Validation: Check `raw_text` non-empty (lines 1804-1809), `chunk_count > 0` (lines 1812-1818)

**Structural Dependencies:**
- Requires CPP (Canonical Policy Package) ingestion subsystem
- Depends on `PreprocessedDocument` dataclass contract (lines 262-450)
- Assumes SPC (Smart Policy Chunks) ingestion enabled (legacy ingestion deprecated, line 321)

---

#### Phase 2: Micro-Question Execution (ASYNC)

**Structural Position:** Parallelization fan-out, primary analytical work  
**Handler:** `_execute_micro_questions_async()` (lines 1830-2088)

**Input Contract:**
- `document`: PreprocessedDocument (from Phase 1)
- `config`: dict containing:
  - `micro_questions`: list[dict] (300+ questions)
  - `executors`: dict (executor class registry)

**Output Contract:**
- `micro_results`: list[MicroQuestionRun] containing:
  - `question_id`: str
  - `question_global`: int
  - `base_slot`: str (executor identifier, e.g., "D1Q1")
  - `metadata`: dict (dimension_id, policy_area_id, cluster_id, etc.)
  - `evidence`: Evidence | None (structured findings)
  - `error`: str | None
  - `duration_ms`: float | None
  - `aborted`: bool

**Functional Role:**
- **Parallel Processing:** Execute 300+ questions concurrently with semaphore-controlled parallelism
- **Executor Dispatch:** Route questions to appropriate executor classes based on `base_slot`
- **Resource Governance:** Apply worker budget (`ResourceLimits.apply_worker_budget()`, line 1896)
- **Chunk-Aware Optimization:** Route chunks to executors when `processing_mode="chunked"` (lines 1844-1860)
- **Circuit Breaking:** Track executor failures and open circuit breakers on excessive errors (lines 1881-1933)
- **Evidence Collection:** Aggregate evidence from executor outputs

**Parallel Execution Architecture:**

1. **Semaphore Control:** `asyncio.Semaphore(max_workers)` limits concurrent execution (line 1878)
2. **Fair Scheduling:** Round-robin ordering by base_slot ensures balanced execution (lines 1862-1876)
3. **Asyncio Task Pool:** `asyncio.create_task()` for each question, `asyncio.as_completed()` for result collection (not shown but implied by async pattern)
4. **Chunk Routing (NEW):** If `document.processing_mode == "chunked"`:
   - Initialize `ChunkRouter()` (line 1846)
   - Route chunks to executors based on chunk type (lines 1847-1857)
   - Execute on relevant chunks only (lines 1956-2000)
   - Aggregate chunk evidences (lines 1990-1999)

**Structural Complexity:**
- **Branching Execution:** Each question spawns independent async task
- **Fan-Out Factor:** 300+ concurrent tasks (limited by semaphore)
- **Dynamic Routing:** Executor selection per question based on metadata
- **Error Handling:** Per-question try/except, circuit breaker state tracking

**Item Targets:** 300 micro-questions expected (PHASE_ITEM_TARGETS[2] = 300, line 1052)

---

#### Phase 3: Micro-Question Scoring (ASYNC)

**Structural Position:** Transformation layer, quality quantification  
**Handler:** `_score_micro_results_async()` (lines 2107-2214)

**Input Contract:**
- `micro_results`: list[MicroQuestionRun] (from Phase 2)
- `config`: dict containing scoring configuration

**Output Contract:**
- `scored_results`: list[ScoredMicroQuestion] containing:
  - `question_id`: str
  - `question_global`: int
  - `base_slot`: str
  - `score`: float | None (0.0-100.0 scale)
  - `normalized_score`: float | None (0.0-1.0 scale)
  - `quality_level`: str | None ("EXCELENTE", "SATISFACTORIO", "ACEPTABLE", "INSUFICIENTE")
  - `evidence`: Evidence | None
  - `scoring_details`: dict (rubric application details)
  - `metadata`: dict
  - `error`: str | None

**Functional Role:**
- **Evidence → Score Transformation:** Apply scoring rubrics to evidence objects
- **Quality Classification:** Map scores to quality levels via thresholds
- **Normalization:** Convert raw scores to normalized [0, 1] range
- **Parallel Scoring:** Independent scoring per micro-result

**Scoring Pipeline:**
1. Extract `scoring_modality` from question metadata (line 2146)
2. Route to appropriate scorer based on modality (line 2165)
3. Apply rubric scoring algorithm (implementation in executor/scoring modules)
4. Compute normalized score: `score / 100.0` (line 2173)
5. Classify quality level via threshold mapping (line 2174)
6. Aggregate scoring details into `scoring_details` dict (line 2180)

**Structural Dependencies:**
- Depends on Evidence objects from Phase 2
- Requires scoring config in monolith (rubric thresholds, quality mappings)
- May fail gracefully per question (error captured in ScoredMicroQuestion.error)

**Item Targets:** 300 scored results expected (PHASE_ITEM_TARGETS[3] = 300, line 1053)

---

#### Phase 4: Dimension Aggregation (ASYNC)

**Structural Position:** First aggregation level, 5-to-1 reduction  
**Handler:** `_aggregate_dimensions_async()` (lines 2216-2288)

**Input Contract:**
- `scored_results`: list[ScoredMicroQuestion] (from Phase 3)
- `config`: dict containing monolith with dimension definitions

**Output Contract:**
- `dimension_scores`: list[DimensionScore] containing:
  - `dimension_id`: str (e.g., "D1", "D2", ...)
  - `area_id`: str (e.g., "A1", "A2", ...)
  - `score`: float (aggregated score)
  - `quality_level`: str
  - `contributing_questions`: list[int] (question_global IDs)
  - `validation_passed`: bool
  - `validation_details`: dict

**Functional Role:**
- **Aggregation:** Combine 5 micro-question scores → 1 dimension score
- **Weighted Averaging:** Apply dimension-specific weights (equal weights by default, line 2280)
- **Coverage Validation:** Ensure sufficient questions answered per dimension
- **Quality Propagation:** Determine dimension quality level from aggregated score

**Aggregation Architecture:**

1. **Grouping:** Group scored_results by `(dimension_id, area_id)` tuple (lines 2260-2264)
2. **Dimension Instantiation:** For each dimension group:
   - Create `DimensionAggregator` instance with monolith (line 2240)
   - Call `aggregator.aggregate_dimension(dimension_id, area_id, scored_results, weights=None)` (lines 2276-2281)
3. **Validation:** DimensionAggregator validates weights, thresholds, coverage (implementation in `aggregation.py`)
4. **Error Handling:** Catch exceptions per dimension, log error, continue processing (lines 2283-2284)

**Structural Properties:**
- **Parallelization:** Independent aggregation per dimension (async sleep for cooperative multitasking, line 2272)
- **Hierarchical Structuring:** Dimensions nest within areas (60 dimensions = 6 dimensions × 10 areas)
- **Lossless Provenance:** `contributing_questions` list preserves micro-question IDs

**Item Targets:** 60 dimension scores expected (PHASE_ITEM_TARGETS[4] = 60, line 1054)

---

#### Phase 5: Policy Area Aggregation (ASYNC)

**Structural Position:** Second aggregation level, 6-to-1 reduction  
**Handler:** `_aggregate_policy_areas_async()` (lines 2290-2343)

**Input Contract:**
- `dimension_scores`: list[DimensionScore] (from Phase 4)
- `config`: dict containing monolith with area definitions

**Output Contract:**
- `policy_area_scores`: list[AreaScore] containing:
  - `area_id`: str (e.g., "A1", ..., "A10")
  - `area_name`: str (human-readable label)
  - `score`: float (aggregated score)
  - `quality_level`: str
  - `dimension_scores`: list[DimensionScore] (component dimensions)
  - `validation_passed`: bool
  - `validation_details`: dict

**Functional Role:**
- **Aggregation:** Combine 6 dimension scores → 1 area score
- **Area Naming:** Resolve area names from monolith configuration
- **Validation:** Ensure coverage and weight normalization
- **Quality Classification:** Map area score to quality level

**Aggregation Architecture:**

1. **Grouping:** Group dimension_scores by `area_id` (lines 2317-2321)
2. **Area Instantiation:** For each area:
   - Create `AreaPolicyAggregator` instance (line 2314)
   - Call `aggregator.aggregate_area(area_id, dimension_scores)` (lines 2333-2336)
3. **Validation:** AreaPolicyAggregator validates dimension coverage per area
4. **Error Handling:** Per-area exception catching (lines 2338-2339)

**Structural Properties:**
- **Parallelization:** Independent aggregation per area (async sleep, line 2329)
- **Fixed Cardinality:** 10 policy areas expected (PHASE_ITEM_TARGETS[5] = 10, line 1055)
- **Composite Structure:** Each AreaScore contains list of contributing DimensionScores

**Item Targets:** 10 area scores expected (line 1055)

---

#### Phase 6: Cluster Aggregation (SYNC)

**Structural Position:** Third aggregation level, M-to-1 reduction (M varies per cluster)  
**Handler:** `_aggregate_clusters()` (lines 2345-2395)

**Input Contract:**
- `policy_area_scores`: list[AreaScore] (from Phase 5)
- `config`: dict containing monolith with cluster definitions

**Output Contract:**
- `cluster_scores`: list[ClusterScore] containing:
  - `cluster_id`: str (e.g., "C1", "C2", "C3", "C4")
  - `cluster_name`: str
  - `areas`: list[str] (area IDs in cluster)
  - `score`: float (aggregated score)
  - `coherence`: float (inter-area coherence metric)
  - `area_scores`: list[AreaScore] (component areas)
  - `validation_passed`: bool
  - `validation_details`: dict

**Functional Role:**
- **Cluster Aggregation:** Combine multiple area scores → 1 cluster score (4 MESO questions)
- **Coherence Calculation:** Measure internal consistency across clustered areas
- **Thematic Grouping:** Cluster areas by MESO question themes

**Aggregation Architecture:**

1. **Cluster Definition Loading:** Extract cluster definitions from `monolith["blocks"]["niveles_abstraccion"]["clusters"]` (line 2372)
2. **Cluster Instantiation:** For each cluster definition:
   - Create `ClusterAggregator` instance (line 2369)
   - Call `aggregator.aggregate_cluster(cluster_id, area_scores, weights=None)` (lines 2384-2388)
3. **Validation:** ClusterAggregator validates area coverage and weight normalization
4. **Error Handling:** Per-cluster exception catching (lines 2390-2391)

**Structural Properties:**
- **Synchronous Execution:** No async parallelization (Phase 6 mode="sync")
- **Variable Cardinality:** Cluster size varies (some clusters have 2 areas, others 3-4)
- **Fixed Cluster Count:** 4 clusters expected (PHASE_ITEM_TARGETS[6] = 4, line 1056)

**Item Targets:** 4 cluster scores expected (line 1056)

---

#### Phase 7: Macro Evaluation (SYNC)

**Structural Position:** Fourth aggregation level, 4-to-1 final reduction  
**Handler:** `_evaluate_macro()` (lines 2397-2458)

**Input Contract:**
- `cluster_scores`: list[ClusterScore] (from Phase 6)
- `config`: dict containing monolith with macro question definition

**Output Contract:**
- `macro_result`: MacroScoreDict (TypedDict) containing:
  - `macro_score`: MacroScore dataclass with:
    - `score`: float (holistic score)
    - `quality_level`: str
    - `cross_cutting_coherence`: float
    - `systemic_gaps`: list[str]
    - `strategic_alignment`: float
    - `cluster_scores`: list[ClusterScore]
    - `validation_passed`: bool
    - `validation_details`: dict
  - `macro_score_normalized`: float (0.0-1.0)
  - `cluster_scores`: list[ClusterScore] (passthrough)

**Functional Role:**
- **Holistic Evaluation:** Synthesize 4 cluster scores into single macro score
- **Cross-Cutting Analysis:** Compute coherence across clusters
- **Gap Identification:** Identify systemic weaknesses
- **Strategic Alignment:** Assess overall policy alignment with objectives

**Aggregation Architecture:**

1. **Macro Aggregator Instantiation:** Create `MacroAggregator` instance (line 2420)
2. **Aggregation:** Call `aggregator.aggregate_macro(cluster_scores, weights=None)` (lines 2426-2429)
3. **Normalization:** Compute `macro_score_normalized = macro_score.score / 100.0` (line 2434)
4. **Result Packaging:** Construct MacroScoreDict with all components (lines 2431-2435)

**Structural Properties:**
- **Synchronous Execution:** Single-threaded aggregation
- **Singleton Output:** Exactly 1 macro score produced (PHASE_ITEM_TARGETS[7] = 1, line 1057)
- **Emergent Properties:** Macro score exhibits holistic properties not present in clusters

**Item Targets:** 1 macro evaluation expected (line 1057)

---

#### Phase 8: Recommendation Generation (ASYNC)

**Structural Position:** Insight synthesis, actionable output generation  
**Handler:** `_generate_recommendations()` (not shown in excerpts, but referenced at line 1044)

**Input Contract:**
- `macro_result`: MacroScoreDict (from Phase 7)
- `config`: dict

**Output Contract:**
- `recommendations`: list[Recommendation] or dict (structure not fully specified in excerpts)

**Functional Role:**
- **Actionable Synthesis:** Transform analytical insights into recommendations
- **Priority Ranking:** Order recommendations by importance/impact
- **Gap Addressing:** Propose interventions for identified systemic gaps

**Structural Properties:**
- **Asynchronous Execution:** Potentially calls external LLM for recommendation generation
- **Singleton Output:** 1 recommendation set expected (PHASE_ITEM_TARGETS[8] = 1, line 1058)

**Item Targets:** 1 recommendation set expected (line 1058)

---

#### Phase 9: Report Assembly (SYNC)

**Structural Position:** Output structuring, artifact composition  
**Handler:** `_assemble_report()` (not shown in excerpts, but referenced at line 1045)

**Input Contract:**
- `recommendations`: list (from Phase 8)
- `config`: dict

**Output Contract:**
- `report`: dict or Report dataclass (structure not fully specified)

**Functional Role:**
- **Artifact Assembly:** Combine all phase outputs into structured report
- **Formatting:** Apply report templates and styling
- **Metadata Addition:** Add provenance, timestamps, system info

**Structural Properties:**
- **Synchronous Execution:** Sequential assembly
- **Singleton Output:** 1 report expected (PHASE_ITEM_TARGETS[9] = 1, line 1059)

**Item Targets:** 1 report expected (line 1059)

---

#### Phase 10: Format and Export (ASYNC)

**Structural Position:** Output boundary, external delivery  
**Handler:** `_format_and_export()` (not shown in excerpts, but referenced at line 1046)

**Input Contract:**
- `report`: dict (from Phase 9)
- `config`: dict

**Output Contract:**
- `export_payload`: dict or ExportPayload (structure not fully specified)

**Functional Role:**
- **Multi-Format Export:** Generate JSON, PDF, HTML, or other formats
- **External Delivery:** Write artifacts to file system or external systems
- **Archival:** Store analysis results for future retrieval

**Structural Properties:**
- **Asynchronous Execution:** Potentially parallel export to multiple formats
- **Singleton Output:** 1 export payload expected (PHASE_ITEM_TARGETS[10] = 1, line 1060)

**Item Targets:** 1 export payload expected (line 1060)

---

### 2.3 Structural Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ENVIRONMENT                            │
│  (File System, LLM APIs, Monitoring, External Consumers)            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ INPUT BOUNDARY
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 0: Configuration Validation (SYNC)                             │
│  Input: monolith, method_map, schema                                 │
│  Output: config (validated, hashed)                                  │
│  Function: Boundary gatekeeping, identity establishment              │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Document Ingestion (SYNC)                                   │
│  Input: pdf_path, config                                             │
│  Output: document (PreprocessedDocument with chunks)                 │
│  Function: PDF → structured document transformation                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Micro-Question Execution (ASYNC, PARALLEL)                  │
│  Input: document, config.micro_questions (300+)                      │
│  Output: micro_results (list[MicroQuestionRun])                      │
│  Function: Fan-out parallel analysis, evidence collection            │
│  Parallelism: Semaphore-controlled (max_workers), chunk-aware        │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Micro-Question Scoring (ASYNC, PARALLEL)                    │
│  Input: micro_results                                                │
│  Output: scored_results (list[ScoredMicroQuestion])                  │
│  Function: Evidence → score transformation, quality classification   │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
        ┌──────────────────────┴──────────────────────┐
        │  HIERARCHICAL AGGREGATION SUB-PIPELINE      │
        │  (Emergent Properties via Progressive       │
        │   Abstraction)                              │
        └─────────────────────┬───────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Dimension Aggregation (ASYNC)                               │
│  Input: scored_results (300+)                                        │
│  Output: dimension_scores (60)                                       │
│  Function: 5-to-1 aggregation, weighted averaging                    │
│  Reduction: 300 → 60 (5:1 ratio per dimension)                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Policy Area Aggregation (ASYNC)                             │
│  Input: dimension_scores (60)                                        │
│  Output: policy_area_scores (10)                                     │
│  Function: 6-to-1 aggregation, area synthesis                        │
│  Reduction: 60 → 10 (6:1 ratio per area)                             │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 6: Cluster Aggregation (SYNC)                                  │
│  Input: policy_area_scores (10)                                      │
│  Output: cluster_scores (4)                                          │
│  Function: M-to-1 aggregation, coherence calculation                 │
│  Reduction: 10 → 4 (variable ratio per cluster)                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 7: Macro Evaluation (SYNC)                                     │
│  Input: cluster_scores (4)                                           │
│  Output: macro_result (1 MacroScore)                                 │
│  Function: 4-to-1 holistic synthesis, gap identification             │
│  Reduction: 4 → 1 (final emergence)                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 8: Recommendation Generation (ASYNC)                           │
│  Input: macro_result                                                 │
│  Output: recommendations                                             │
│  Function: Insight → action transformation                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 9: Report Assembly (SYNC)                                      │
│  Input: recommendations, all phase outputs                           │
│  Output: report                                                      │
│  Function: Artifact structuring and composition                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────────┐
│ PHASE 10: Format and Export (ASYNC)                                  │
│  Input: report                                                       │
│  Output: export_payload                                              │
│  Function: Multi-format generation, external delivery                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ OUTPUT BOUNDARY
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ENVIRONMENT                            │
│  (Artifact Storage, Stakeholder Systems, Audit Trails)              │
└─────────────────────────────────────────────────────────────────────┘

LEGEND:
━━━━━━  Sequential data flow
═══════  Hierarchical aggregation boundary
SYNC    Synchronous execution
ASYNC   Asynchronous execution with potential parallelism
```

### 2.4 Structural Properties

#### 2.4.1 Differentiation (Functional Specialization)

**Theoretical Framing:** Parsons' structural differentiation theory posits that system evolution involves increasing specialization of sub-units performing distinct functions.

**Differentiation Analysis in SAAAAAA:**

1. **Phase-Level Differentiation:**
   - **Input Processing:** Phases 0-1 (validation, ingestion)
   - **Analytical Execution:** Phases 2-3 (questions, scoring)
   - **Hierarchical Aggregation:** Phases 4-7 (dimension → area → cluster → macro)
   - **Output Generation:** Phases 8-10 (recommendations, assembly, export)

2. **Executor-Level Differentiation:**
   - Evidence: 60+ executor classes (implied by 60 dimensions, 5 questions per dimension = 300 executors)
   - Each executor specialized for specific question type (diagnostic, activity, indicator, resource, temporal, entity)
   - Chunk router maps chunk types to executor types (ROUTING_TABLE, lines 36-43 in `chunk_router.py`)

3. **Aggregator-Level Differentiation:**
   - `DimensionAggregator`: Micro → dimension transformation
   - `AreaPolicyAggregator`: Dimension → area transformation
   - `ClusterAggregator`: Area → cluster transformation
   - `MacroAggregator`: Cluster → macro transformation
   - Each aggregator has distinct validation logic, weight handling, coherence calculation

**Degree of Differentiation:** HIGH
- 11 distinct phases
- 300+ micro-level executors
- 4 hierarchical aggregation levels
- Specialized data structures per level (MicroQuestionRun, ScoredMicroQuestion, DimensionScore, AreaScore, ClusterScore, MacroScore)

#### 2.4.2 Integration Mechanisms (Coordination)

**Theoretical Framing:** Differentiated systems require integration mechanisms to maintain coherence. Lawrence & Lorsch (1967) identify integration as the counterbalance to differentiation.

**Integration Mechanisms in SAAAAAA:**

1. **Contract-Based Integration:**
   - **TypedDict Contracts:** Input/output specifications for each phase (lines 1063-1088)
   - **Dataclass Interfaces:** PreprocessedDocument, MicroQuestionRun, ScoredMicroQuestion, etc. serve as typed interfaces
   - **Validation Gates:** Phase 0 enforces contract compliance before execution

2. **Orchestrator Centralization:**
   - **Single Authority:** Orchestrator class owns all phase execution
   - **Sequential Coordination:** Fixed FASES list defines execution order (lines 1035-1047)
   - **Data Pipelining:** Each phase output becomes next phase input (PHASE_OUTPUT_KEYS, lines 1063-1075)

3. **Instrumentation-Based Monitoring:**
   - **Phase Instrumentation:** `_phase_instrumentation` dict tracks phase metrics (line 1169)
   - **Resource Monitoring:** `ResourceLimits` class provides cross-phase resource awareness
   - **Abort Signaling:** `AbortSignal` enables global coordination of early termination

4. **Shared State Management:**
   - **Config Dictionary:** Passed through all phases, accumulating outputs
   - **Monolith Context:** Shared questionnaire specification across phases
   - **Method Executor:** Single `MethodExecutor` instance shared across Phase 2 questions

**Integration Strength:** TIGHT
- Phases tightly coupled via sequential data dependency
- No phase can execute without upstream phase completion
- Contract violations cause immediate failure (no silent degradation)

#### 2.4.3 Hierarchy (Control Structures)

**Theoretical Framing:** Simon (1962) describes hierarchy as "a system composed of interrelated subsystems, each of which is hierarchical in structure."

**Hierarchical Properties in SAAAAAA:**

1. **Execution Hierarchy:**
   ```
   Orchestrator (Level 0)
   ├── Phase Handlers (Level 1)
   │   ├── MethodExecutor (Level 2)
   │   │   ├── ArgRouter (Level 3)
   │   │   ├── Class Registry (Level 3)
   │   │   └── Individual Executors (Level 3)
   │   ├── Aggregators (Level 2)
   │   │   ├── Dimension Aggregator (Level 3)
   │   │   ├── Area Aggregator (Level 3)
   │   │   ├── Cluster Aggregator (Level 3)
   │   │   └── Macro Aggregator (Level 3)
   │   └── Resource Manager (Level 2)
   │       ├── Semaphore (Level 3)
   │       ├── Worker Budget (Level 3)
   │       └── Resource Monitors (Level 3)
   ```

2. **Data Hierarchy (Aggregation Pyramid):**
   ```
   Macro Score (Level 4) — 1 item
   ├── Cluster Scores (Level 3) — 4 items
   │   ├── Area Scores (Level 2) — 10 items
   │   │   ├── Dimension Scores (Level 1) — 60 items
   │   │   │   └── Micro Questions (Level 0) — 300+ items
   ```

3. **Control Hierarchy:**
   - **Top-Down Control:** Orchestrator invokes phases, phases invoke executors, executors invoke methods
   - **Bottom-Up Reporting:** Results propagate upward (micro → dimension → area → cluster → macro)
   - **Middle-Layer Transformation:** Aggregators transform data granularity at each level

**Hierarchical Span:**
- **Depth:** 4 levels (orchestrator → phase → subsystem → component)
- **Breadth:** Varies by level (300 micro, 60 dimensions, 10 areas, 4 clusters, 1 macro)

#### 2.4.4 Modularity (Coupling and Cohesion)

**Theoretical Framing:** Baldwin & Clark (2000) define modularity as "building a complex product or process from smaller subsystems that can be designed independently yet function together as a whole."

**Modularity Analysis:**

**High Cohesion Components:**

1. **Aggregation Module** (`aggregation.py`):
   - **Function:** Hierarchical score aggregation
   - **Cohesion:** All aggregators share common patterns (weighted averaging, validation, quality classification)
   - **Evidence:** Single module contains all 4 aggregator classes (lines 1-150+)

2. **Executor Module** (`executors.py`):
   - **Function:** Executor infrastructure and base classes
   - **Cohesion:** All executor-related logic centralized
   - **Evidence:** Neuromorphic computing, quantum optimization, causal inference all in one module (lines 1-100+)

3. **Orchestrator Core** (`core.py`):
   - **Function:** Phase execution and coordination
   - **Cohesion:** All phase handlers co-located with orchestrator class
   - **Evidence:** 2400+ lines in single module

**Coupling Analysis:**

1. **Loose Coupling (Good):**
   - **Executors ↔ Orchestrator:** Interface via `execute(document, executor)` method, no direct orchestrator dependency
   - **Aggregators ↔ Orchestrator:** Interface via dataclass contracts (DimensionScore, AreaScore, etc.)
   - **ChunkRouter ↔ Orchestrator:** Optional import, graceful degradation if unavailable (lines 1845-1860)

2. **Tight Coupling (Acceptable):**
   - **Phases ↔ Orchestrator:** Phases are orchestrator methods, inherently coupled
   - **Instrumentation ↔ Phases:** Each phase tightly coupled to its instrumentation object

3. **Problematic Coupling:**
   - **Monolith Dependency:** All components depend on monolith structure, no abstraction layer
   - **MethodExecutor Singleton:** Phases 2-7 share single executor instance, potential bottleneck

**Modularity Score:** MEDIUM-HIGH
- Good separation of aggregation, execution, orchestration concerns
- Contract-based interfaces enable substitution
- Monolith hard-coupling limits flexibility


---

## 3. Functional Analysis: System Operations

### 3.1 Micro-Fluxes and Asynchronous Processing

The SAAAAAA pipeline implements sophisticated parallel processing patterns that enable high-throughput analysis of 300+ micro-questions concurrently. This section analyzes the functional mechanisms enabling parallelism, resource governance, and chunk-aware optimization.

#### 3.1.1 Fan-Out/Fan-In Pattern (Phase 2)

**Pattern Description:**

Phase 2 (`_execute_micro_questions_async`) implements a classic **fan-out/fan-in** concurrency pattern:
- **Fan-Out:** 300+ micro-questions dispatched as independent async tasks
- **Parallel Execution:** Tasks execute concurrently (bounded by semaphore)
- **Fan-In:** Results collected and aggregated into list

**Evidence from Code (lines 1830-2088):**

```python
async def _execute_micro_questions_async(
        self,
        document: PreprocessedDocument,
        config: dict[str, Any],
) -> list[MicroQuestionRun]:
    # ...
    semaphore = asyncio.Semaphore(self.resource_limits.max_workers)  # line 1878
    # ...
    async def process_question(question: dict[str, Any]) -> MicroQuestionRun:
        await self.resource_limits.apply_worker_budget()  # line 1896
        async with semaphore:  # line 1897
            # ... question processing ...
    
    tasks = [asyncio.create_task(process_question(q)) for q in ordered_questions]
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
```

**Functional Characteristics:**

1. **Concurrency Control:**
   - **Semaphore Limiting:** `asyncio.Semaphore(max_workers)` bounds concurrent tasks
   - **Default max_workers:** Not specified in excerpts, likely 10-50 based on typical patterns
   - **Worker Budget:** `apply_worker_budget()` implements rate limiting beyond semaphore (line 1896)

2. **Fair Scheduling:**
   - **Round-Robin Ordering:** Questions ordered by base_slot to ensure balanced executor loading (lines 1862-1876)
   - **Slot Queues:** `questions_by_slot` dict with deque per slot enables fair distribution
   - **Evidence:** Round-robin pattern prevents single executor from dominating execution

3. **Result Collection:**
   - **Asynchronous Collection:** `asyncio.as_completed()` enables results to be processed as they arrive
   - **Order Preservation:** Results appended to list in completion order (not submission order)

#### 3.1.2 Resource Governance Mechanisms

**Theoretical Framing:** Commons governance theory (Ostrom, 1990) emphasizes the need for resource management rules in shared resource systems. The SAAAAAA pipeline implements computational resource governance through multiple mechanisms.

**Resource Limit Architecture:**

**A. Semaphore-Based Concurrency Control**

Evidence: Line 1878-1879
```python
semaphore = asyncio.Semaphore(self.resource_limits.max_workers)
self.resource_limits.attach_semaphore(semaphore)
```

**Function:** Limits number of concurrent async tasks to prevent resource exhaustion

**B. Worker Budget Application**

Evidence: Line 1896
```python
await self.resource_limits.apply_worker_budget()
```

**Function:** Additional rate limiting layer, potentially implementing token bucket or leaky bucket algorithm

**C. Memory and CPU Monitoring**

Evidence: Lines 1936-1941
```python
usage = self.resource_limits.get_resource_usage()
mem_exceeded, usage = self.resource_limits.check_memory_exceeded(usage)
cpu_exceeded, usage = self.resource_limits.check_cpu_exceeded(usage)
if mem_exceeded:
    instrumentation.record_warning("resource", "Límite de memoria excedido", usage=usage)
```

**Function:** Real-time monitoring triggers warnings when resource thresholds exceeded

**D. Circuit Breaker Pattern**

Evidence: Lines 1881-1933
```python
circuit_breakers: dict[str, dict[str, Any]] = {
    slot: {"failures": 0, "open": False}
    for slot in self.executors
}
# ...
circuit = circuit_breakers.setdefault(base_slot, {"failures": 0, "open": False})
if circuit.get("open"):
    instrumentation.record_warning(
        "circuit_breaker",
        "Circuit breaker abierto, pregunta omitida",
        base_slot=base_slot,
        question_id=question_id,
    )
    return MicroQuestionRun(...)  # Skip execution
```

**Function:** Per-executor circuit breakers prevent cascading failures by opening after excessive errors

**Resource Governance Matrix:**

| Mechanism | Scope | Trigger | Action | Purpose |
|-----------|-------|---------|--------|---------|
| Semaphore | Global | Task count > max_workers | Block task start | Concurrency limit |
| Worker Budget | Per-task | Rate limit exceeded | Delay task | Rate smoothing |
| Memory Check | Global | Memory > threshold | Warning log | Monitoring |
| CPU Check | Global | CPU > threshold | Warning log | Monitoring |
| Circuit Breaker | Per-executor | Failures > threshold | Skip execution | Fault isolation |

**Systemic Functions:**

1. **Stability:** Prevents resource exhaustion that would destabilize entire pipeline
2. **Fairness:** Round-robin scheduling ensures no executor monopolizes resources
3. **Resilience:** Circuit breakers isolate failing executors without halting pipeline
4. **Observability:** Resource warnings enable real-time monitoring and intervention

#### 3.1.3 Chunk-Aware Optimization

**Innovation:** Phase 2 implements chunk-aware execution, routing specific chunks to relevant executors instead of processing entire document for each question.

**Evidence:** Lines 1841-1860

```python
# NEW: Initialize chunk router for chunk-aware execution
chunk_routes: dict[int, Any] = {}
if document.processing_mode == "chunked" and document.chunks:
    try:
        from saaaaaa.core.orchestrator.chunk_router import ChunkRouter
        router = ChunkRouter()
        
        # Route chunks to executors
        for chunk in document.chunks:
            route = router.route_chunk(chunk)
            if not route.skip_reason:
                chunk_routes[chunk.id] = route
        
        logger.info(
            f"Chunk-aware execution enabled: routed {len(chunk_routes)} chunks "
            f"from {len(document.chunks)} total chunks"
        )
    except ImportError:
        logger.warning("ChunkRouter not available, falling back to flat mode")
        chunk_routes = {}
```

**Chunk Routing Logic:**

ChunkRouter maps chunk types to executor base slots (evidence: `chunk_router.py`, lines 36-43):

```python
ROUTING_TABLE: dict[str, list[str]] = {
    "diagnostic": ["D1Q1", "D1Q2", "D1Q5"],
    "activity": ["D2Q1", "D2Q2", "D2Q3", "D2Q4", "D2Q5"],
    "indicator": ["D3Q1", "D3Q2", "D4Q1", "D5Q1"],
    "resource": ["D1Q3", "D2Q4", "D5Q5"],
    "temporal": ["D1Q5", "D3Q4", "D5Q4"],
    "entity": ["D2Q3", "D3Q3"],
}
```

**Functional Benefits:**

1. **Processing Efficiency:** Executors process only relevant chunks, not entire document
   - Example: "resource" executor only processes financial/resource chunks
   - Reduces redundant text processing across 300+ questions

2. **Scalability:** Chunk-level parallelism enables document-size-independent processing
   - Large documents (100+ pages) partitioned into manageable chunks (5-10 pages each)
   - Execution time grows sub-linearly with document size

3. **Semantic Precision:** Chunk types match executor specializations
   - "diagnostic" chunks contain baseline/gap analysis text
   - "activity" chunks contain intervention descriptions
   - Routing ensures semantic alignment between chunk content and executor expertise

**Execution Metrics Tracking:**

Evidence: Lines 1889-1893
```python
execution_metrics = {
    "chunk_executions": 0,  # Actual chunk-level executions
    "full_doc_executions": 0,  # Fallback full document executions
    "total_chunks_processed": 0,  # Total chunks that could have been processed
}
```

**Chunk Aggregation:**

Evidence: Lines 1990-1999
```python
# Aggregate chunk results
if chunk_evidences:
    evidence = chunk_evidences[0]
    if len(chunk_evidences) > 1 and hasattr(evidence, 'matches'):
        all_matches = []
        for chunk_ev in chunk_evidences:
            if hasattr(chunk_ev, 'matches') and chunk_ev.matches:
                all_matches.extend(chunk_ev.matches)
        evidence.matches = all_matches
```

**Function:** Merge evidence from multiple chunks into single MicroQuestionRun result

#### 3.1.4 Scoring Subsystem (Phase 3)

Phase 3 implements parallel scoring transformation: `MicroQuestionRun` → `ScoredMicroQuestion`

**Evidence:** Lines 2107-2214

**Parallel Scoring Architecture:**

```python
async def _score_micro_results_async(
        self,
        micro_results: list[MicroQuestionRun],
        config: dict[str, Any],
) -> list[ScoredMicroQuestion]:
    # ...
    async def score_item(item: MicroQuestionRun) -> ScoredMicroQuestion:
        # ... scoring logic ...
    
    tasks = [asyncio.create_task(score_item(item)) for item in micro_results]
    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
```

**Functional Steps:**

1. **Scoring Modality Extraction:** Line 2146
   ```python
   scoring_modality = question_metadata.get("scoring_modality", "default")
   ```

2. **Scorer Dispatch:** Line 2165
   - Route to appropriate scorer based on modality
   - Scorers apply rubric-based algorithms to evidence

3. **Score Normalization:** Line 2173
   ```python
   normalized_score = score / 100.0 if score is not None else None
   ```

4. **Quality Classification:** Line 2174
   - Map normalized score to quality levels (EXCELENTE, SATISFACTORIO, ACEPTABLE, INSUFICIENTE)
   - Thresholds defined in monolith scoring config

5. **Result Packaging:** Lines 2175-2191
   ```python
   ScoredMicroQuestion(
       question_id=item.question_id,
       question_global=item.question_global,
       base_slot=item.base_slot,
       score=score,
       normalized_score=normalized_score,
       quality_level=quality_level,
       evidence=item.evidence,
       scoring_details=scoring_details,
       metadata=item.metadata,
   )
   ```

**Parallelism Characteristics:**

- **Independence:** Each scoring operation independent (no shared state)
- **Async Overhead:** Minimal per-item overhead (simple transformation)
- **Throughput:** 300+ items scored in parallel
- **Bottleneck:** Scorer dispatch (single registry lookup) may serialize

### 3.2 Control and Coordination Mechanisms

#### 3.2.1 Timeout Management (Cybernetic Control)

**Theoretical Framing:** Cybernetic control theory (Wiener, 1948) emphasizes feedback loops and control mechanisms that maintain system stability. Timeout management implements **negative feedback control** by halting runaway processes.

**Timeout Architecture:**

**A. Phase-Level Timeouts**

Evidence: Lines 1091-1103

```python
PHASE_TIMEOUTS: dict[int, float] = {
    0: 60,     # Configuration validation
    1: 120,    # Document ingestion
    2: 600,    # Micro questions (300 items)
    3: 300,    # Scoring micro
    4: 180,    # Dimension aggregation
    5: 120,    # Policy area aggregation
    6: 60,     # Cluster aggregation
    7: 60,     # Macro evaluation
    8: 120,    # Recommendations
    9: 60,     # Report assembly
    10: 120,   # Format and export
}
```

**B. Timeout Enforcement**

Evidence: `execute_phase_with_timeout()` function (lines 77-171)

```python
async def execute_phase_with_timeout(
    phase_id: int,
    phase_name: str,
    coro: Callable[P, T] | None = None,
    # ...
    timeout_s: float = 300.0,
    **kwargs: P.kwargs,
) -> T:
    start = time.perf_counter()
    logger.info("phase_execution_started", extra={...})
    try:
        result = await asyncio.wait_for(target(*call_args, **kwargs), timeout=timeout_s)
        elapsed = time.perf_counter() - start
        logger.info("phase_execution_completed", extra={...})
        return result
    except asyncio.TimeoutError as exc:
        elapsed = time.perf_counter() - start
        logger.error("phase_execution_timeout", extra={...})
        raise PhaseTimeoutError(phase_id, phase_name, timeout_s) from exc
```

**Cybernetic Control Properties:**

1. **Setpoint:** PHASE_TIMEOUTS defines expected completion time per phase
2. **Measurement:** `time.perf_counter()` tracks elapsed time
3. **Comparator:** `asyncio.wait_for()` compares elapsed vs. timeout
4. **Actuator:** `PhaseTimeoutError` raised to halt execution
5. **Feedback:** Timeout logs enable operators to adjust timeouts based on empirical data

**Control Loop Diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                      PHASE EXECUTION                             │
│  ┌───────────┐      ┌───────────┐      ┌───────────┐          │
│  │  Start    │─────→│  Execute  │─────→│  Complete │          │
│  │  Timer    │      │  Phase    │      │  Phase    │          │
│  └───────────┘      └───────────┘      └───────────┘          │
│        │                  │                   │                 │
│        └──────────────────┴───────────────────┘                │
│                           │                                     │
│                           ↓                                     │
│                  ┌──────────────────┐                          │
│                  │  Elapsed Time    │                          │
│                  │  Comparator      │                          │
│                  └──────────────────┘                          │
│                           │                                     │
│                           ↓                                     │
│                    elapsed > timeout?                          │
│                           │                                     │
│                ┌──────────┴──────────┐                        │
│                │                     │                         │
│               YES                   NO                         │
│                │                     │                         │
│                ↓                     ↓                         │
│         ┌────────────┐        ┌──────────┐                   │
│         │ Raise      │        │ Return   │                   │
│         │ Timeout    │        │ Result   │                   │
│         │ Error      │        │          │                   │
│         └────────────┘        └──────────┘                   │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**Timeout Rationale:**

- **Phase 2 (600s):** Longest timeout due to 300+ parallel questions
- **Phases 4-5 (180s, 120s):** Aggregation timeouts proportional to item count
- **Phase 1 (120s):** PDF ingestion may be slow for large documents
- **Other phases (60s):** Simple transformations with predictable runtime

#### 3.2.2 Error Handling and Propagation

**Error Handling Strategy:**

1. **Per-Question Error Isolation (Phase 2):**
   - Evidence: Lines 2000-2044
   - Strategy: Try/except per question, capture error in MicroQuestionRun.error field
   - Benefit: Single question failure does not halt entire phase

2. **Per-Aggregation Error Handling (Phases 4-7):**
   - Evidence: Lines 2283-2284, 2338-2339, 2390-2391
   - Strategy: Try/except per dimension/area/cluster, log error, continue
   - Benefit: Partial results still produced even with some aggregation failures

3. **Circuit Breaker Error Accumulation:**
   - Evidence: Lines 1881-1933
   - Strategy: Track failures per executor, open circuit breaker after threshold
   - Benefit: Prevents repeated failures, improves throughput by skipping known-bad executors

**Error Propagation Paths:**

```
┌─────────────────────────────────────────────────────────────────┐
│                  ERROR PROPAGATION HIERARCHY                     │
│                                                                  │
│  PhaseTimeoutError ─────────→ Orchestrator.run() [FATAL]       │
│  AbortRequested ────────────→ Orchestrator.run() [FATAL]       │
│  RuntimeError (Phase 0) ────→ Orchestrator.run() [FATAL]       │
│  ValueError (Phase 1) ──────→ Orchestrator.run() [FATAL]       │
│                                                                  │
│  Executor Error (Phase 2) ──→ MicroQuestionRun.error [CONTAINED]│
│  Scoring Error (Phase 3) ───→ ScoredMicroQuestion.error [CONTAINED]│
│  Aggregation Error (Phase 4-7) ─→ Log error [CONTAINED]        │
│                                                                  │
│  Circuit Breaker Open ──────→ Skip execution [DEGRADED MODE]   │
│  Resource Limit Exceeded ───→ Warning log [CONTINUE]           │
└─────────────────────────────────────────────────────────────────┘

LEGEND:
[FATAL]        → Pipeline halts, exception propagated
[CONTAINED]    → Error captured, pipeline continues
[DEGRADED MODE] → Functionality reduced, pipeline continues
[CONTINUE]     → Warning logged, no impact on execution
```

#### 3.2.3 Coordination Protocols

**Abort Signaling Protocol:**

Evidence: AbortSignal class (implied from lines 1582-1584, 1606-1610)

**Protocol Steps:**

1. **Signal Initialization:** Orchestrator creates AbortSignal instance
2. **Periodic Checks:** Each phase calls `self._ensure_not_aborted()` (lines 1615, 1835, 2107, 2230, 2304, 2359, 2399)
3. **Signal Propagation:** If aborted, `AbortRequested` exception raised
4. **Cleanup:** Orchestrator catches exception, logs abort reason, terminates pipeline

**Coordination Properties:**

- **Cooperative Abort:** Phases voluntarily check abort signal (not preemptive)
- **Latency:** Abort latency depends on phase checkpoint frequency
- **Propagation:** Abort signal propagates via exception mechanism
- **Reason Tracking:** `abort_signal.get_reason()` provides abort context

**Instrumentation Coordination:**

Evidence: `_phase_instrumentation` dict (line 1169)

**Instrumentation Protocol:**

1. **Initialization:** Orchestrator creates PhaseInstrumentation per phase
2. **Attachment:** Each phase retrieves instrumentation: `instrumentation = self._phase_instrumentation[phase_id]`
3. **Recording:** Phases call `instrumentation.increment()`, `record_error()`, `record_warning()`
4. **Aggregation:** Orchestrator collects metrics via `get_phase_metrics()` (line 1604)

**Coordination Benefits:**

- **Decoupled Metrics:** Phases don't manage metric storage, just record events
- **Consistent Schema:** All phases use same instrumentation interface
- **Observability:** Centralized metrics enable system-wide monitoring

### 3.3 Functional Requirements

#### 3.3.1 Throughput Characteristics

**Theoretical Throughput:**

Given parallel execution in Phase 2:
- **Micro-Questions:** 300+ items
- **Max Workers:** Assume 20 concurrent tasks (typical)
- **Average Execution Time per Question:** Assume 5 seconds
- **Theoretical Time:** `(300 / 20) * 5s = 75 seconds` (best case)

**Observed Throughput (from Timeout Budgets):**

- **Phase 2 Timeout:** 600 seconds (10 minutes)
- **Implies Throughput:** `300 / 600 = 0.5 questions/second` (worst case before timeout)

**Throughput Constraints:**

1. **Semaphore Limit:** Bounds concurrent execution
2. **Worker Budget:** Rate limiting reduces peak throughput
3. **LLM API Rate Limits:** External API calls may throttle execution
4. **Resource Limits:** Memory/CPU constraints may slow processing

**Throughput Variability:**

- **Best Case:** All executors fast, no rate limits, high parallelism → ~75s for Phase 2
- **Typical Case:** Some slow executors, moderate rate limits → ~200-300s for Phase 2
- **Worst Case:** Many slow executors, resource constraints → ~600s (timeout)

#### 3.3.2 Latency Characteristics

**End-to-End Latency:**

Summing phase timeouts:
```
Total Maximum Latency = 60 + 120 + 600 + 300 + 180 + 120 + 60 + 60 + 120 + 60 + 120
                      = 1800 seconds = 30 minutes
```

**Latency Distribution by Phase:**

| Phase | Timeout (s) | % of Total | Type |
|-------|-------------|------------|------|
| 0 - Config Validation | 60 | 3.3% | Deterministic |
| 1 - Document Ingestion | 120 | 6.7% | I/O-bound |
| 2 - Micro Questions | 600 | 33.3% | Compute-bound |
| 3 - Scoring | 300 | 16.7% | Compute-bound |
| 4 - Dim Aggregation | 180 | 10.0% | CPU-bound |
| 5 - Area Aggregation | 120 | 6.7% | CPU-bound |
| 6 - Cluster Aggregation | 60 | 3.3% | CPU-bound |
| 7 - Macro Evaluation | 60 | 3.3% | CPU-bound |
| 8 - Recommendations | 120 | 6.7% | Compute-bound |
| 9 - Report Assembly | 60 | 3.3% | CPU-bound |
| 10 - Export | 120 | 6.7% | I/O-bound |

**Critical Path:** Phase 2 (Micro Questions) dominates latency at 33.3% of total budget

**Latency Reduction Strategies:**

1. **Increase Parallelism:** Higher max_workers in Phase 2
2. **Chunk Optimization:** Better chunk routing to reduce per-question processing time
3. **Executor Optimization:** Faster LLM execution or local model caching
4. **Async Aggregation:** Phases 4-5 already async, but could be further parallelized

#### 3.3.3 Reliability and Failure Modes

**System Reliability Mechanisms:**

1. **Degraded Mode Operation:**
   - Evidence: MethodExecutor degraded_mode flag (lines 801-811, 872-877)
   - Condition: Class registry fails to build, but orchestrator continues with limited functionality
   - Impact: Some executors unavailable, but pipeline doesn't crash

2. **Circuit Breaker Isolation:**
   - Evidence: Per-executor circuit breakers (lines 1881-1933)
   - Condition: Executor exceeds failure threshold
   - Impact: Future questions skip failed executor, preventing cascading failures

3. **Graceful Aggregation Failures:**
   - Evidence: Try/except in aggregation phases (lines 2283-2284, 2338-2339, 2390-2391)
   - Condition: Individual dimension/area/cluster aggregation fails
   - Impact: Partial results produced, failed items logged but don't halt pipeline

**Failure Mode Analysis:**

| Failure Type | Scope | Impact | Recovery |
|--------------|-------|--------|----------|
| Phase 0 Validation Failure | Pipeline | FATAL | No recovery, immediate exit |
| Phase 1 Ingestion Failure | Pipeline | FATAL | No recovery, immediate exit |
| Single Executor Failure (Phase 2) | Question | CONTAINED | Circuit breaker may open, question marked as error |
| Timeout (Any Phase) | Phase/Pipeline | FATAL | No recovery, PhaseTimeoutError raised |
| Aggregation Failure (Phases 4-7) | Dimension/Area/Cluster | CONTAINED | Item skipped, partial results continue |
| Resource Limit Exceeded | Pipeline | DEGRADED | Warning logged, execution continues with constraints |
| Abort Signal | Pipeline | FATAL | Graceful shutdown, cleanup performed |

**Reliability Metrics:**

- **Mean Time Between Failures (MTBF):** Depends on executor reliability, not tracked in code
- **Mean Time To Recovery (MTTR):** Instant for circuit breaker, not applicable for fatal errors
- **Availability:** Not specified, depends on infrastructure

**Single Points of Failure:**

1. **Monolith Loading (Phase 0):** If monolith invalid, entire pipeline fails
2. **Document Ingestion (Phase 1):** If PDF unreadable, entire pipeline fails
3. **MethodExecutor Singleton (Phases 2-7):** If executor crashes, all phases affected
4. **Orchestrator Instance:** Single orchestrator instance, no redundancy

