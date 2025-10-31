# SIN_CARRETA Architecture: Doctrina, Invariantes y Abortabilidad

## Table of Contents
1. [Introduction](#introduction)
2. [Prime Directives](#prime-directives)
3. [Architectural Overview](#architectural-overview)
4. [Pipeline Phases](#pipeline-phases)
5. [Role Separation](#role-separation)
6. [Contracts and Invariants](#contracts-and-invariants)
7. [Abortability and Failure Handling](#abortability-and-failure-handling)
8. [Structured Logging](#structured-logging)
9. [Deterministic Reproducibility](#deterministic-reproducibility)

---

## Introduction

**SIN_CARRETA** is an evolving intelligence substrate engineered for precision, fidelity, and state-of-the-art performance in Spanish-language industrial policy text understanding. This document defines the architectural principles, invariants, and failure modes that govern the system.

### Core Mission Keywords
`precision`, `determinism`, `explicitness`, `auditability`, `state-of-the-art`, `compositionality`, `control`, `fidelity`

---

## Prime Directives

These are the absolute, non-negotiable principles that govern every line of code and every architectural decision:

### I. No Graceful Degradation
The system either satisfies its declared contract in full or aborts with explicit, diagnosable failure. 

**Implications:**
- Partial delivery is forbidden
- Fallback heuristics are disallowed
- Silent substitutions violate the contract
- All failures must be explicit and traceable

**Precondition:** All inputs meet declared type and value contracts  
**Postcondition:** Either complete success or explicit abort  
**Invariant:** System state is always deterministic and auditable

### II. No Strategic Simplification
Complexity is a first-class design asset when it increases fidelity, control, or strategic leverage.

**Implications:**
- Do not simplify logic merely to pass validation gates
- Maintain analytical depth over superficial readability
- Preserve domain complexity in the implementation
- Document why complexity serves fidelity

**Invariant:** Analytical fidelity never decreases for convenience

### III. State-of-the-Art as the Baseline
All approaches must begin from current research-grade paradigms (e.g., RAG+, constrained generation).

**Implications:**
- Legacy approaches require explicit justification
- Superior determinism or interpretability justifies older methods
- Continuous integration of research advances
- Benchmark against academic state-of-the-art

**Invariant:** System capabilities match or exceed published research standards

### IV. Deterministic Reproducibility Over Throughput Opportunism
All non-determinism (randomness, concurrency races) must be isolated, controlled, or eliminated.

**Implications:**
- All random operations must use explicit, documented seeds
- Concurrent operations must not introduce race conditions
- Same input always produces identical output
- Full traceability of all transformations

**Precondition:** Seed registry is initialized and accessible  
**Postcondition:** All outputs are bit-reproducible  
**Invariant:** Zero hidden non-determinism in the pipeline

### V. Explicitness Over Assumption
All transformations must declare their contracts. Implicit coercions, type guessing, and lenient parsing are disallowed.

**Implications:**
- Every function declares preconditions and postconditions
- Type contracts are explicit and enforced
- No automatic type conversions
- No silent error recovery

**Invariant:** Every operation has an explicit, documented contract

### VI. Observability as Structure, Not Decoration
Traceability is a structural requirement. All processes must be instrumented for logging to allow full reconstruction and audit.

**Implications:**
- Logging is mandatory, not optional
- All state changes are logged
- Audit trail enables full replay
- Performance metrics are captured

**Invariant:** Complete audit trail exists for every execution

---

## Architectural Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                            │
│                    (Central Control & State)                    │
│                                                                 │
│  Responsibilities:                                              │
│  • Load and validate configuration                             │
│  • Manage complete lifecycle (Phases 0-10)                     │
│  • Coordinate sequential phases                                │
│  • Distribute work to Choreographers                           │
│  • Aggregate global results                                    │
│  • Handle global errors and abort decisions                    │
│  • Generate final report                                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓ delegates to
┌─────────────────────────────────────────────────────────────────┐
│                        CHOREOGRAPHER                            │
│                   (Distributed Execution)                       │
│                                                                 │
│  Responsibilities:                                              │
│  • Execute ONE question at a time                              │
│  • No knowledge of other questions                             │
│  • Interpret DAG of methods                                    │
│  • Coordinate sync/async method execution                      │
│  • Manage local state (single question)                        │
│  • Execute in parallel (300 instances)                         │
│  • Report results to Orchestrator                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓ invokes
┌─────────────────────────────────────────────────────────────────┐
│                       CORE ARSENAL                              │
│                  (7 Producer Modules)                           │
│                                                                 │
│  1. financiero_viabilidad_tablas.py (65 methods)               │
│  2. Analyzer_one.py (34 methods)                               │
│  3. contradiction_deteccion.py (62 methods)                    │
│  4. embedding_policy.py (36 methods)                           │
│  5. teoria_cambio.py (30 methods)                              │
│  6. dereck_beach.py (99 methods)                               │
│  7. policy_processor.py (32 methods)                           │
│                                                                 │
│  Total: 584 methods for primary analysis                       │
└─────────────────────────────────────────────────────────────────┘
```

### Component Categories

According to **Rule 2: Protocol of Canonical Categorization**, every file has a single, strict role:

1. **Core Arsenal** - Primary analytical logic ONLY
2. **Core Orchestration & Choreography** - Execution flow control ONLY
3. **System Entrypoints & E2E Validation** - Application launching ONLY
4. **Core Application Logic** - Report synthesis ONLY
5. **Data Model & Process Schemas** - Data blueprints (no logic)
6. **Validation & QA** - Correctness checking ONLY
7. **Core Utilities** - Single-purpose reusable services ONLY
8. **Codebase & Config Management** - Meta-tools ONLY
9. **Architectural Design & Documentation** - System design reference

**Invariant:** No file contains logic from multiple categories  
**Invariant:** Cross-category interactions are explicitly documented as "wiring"

---

## Pipeline Phases

The system executes in 11 sequential phases (FASE 0-10). Each phase has explicit contracts.

### FASE 0: Configuration Loading

**Purpose:** Load and validate all configuration files

**Preconditions:**
- `questionnaire_monolith.json` exists and is readable
- `metodos_completos_nivel3.json` exists and is readable
- File system is accessible

**Process:**
1. Load questionnaire monolith JSON
2. Verify integrity hash matches declared hash
3. Load method catalog JSON
4. Verify method count matches expected (416 methods)
5. Initialize global configuration

**Postconditions:**
- Monolith loaded with 305 questions (300 micro + 4 meso + 1 macro)
- Method catalog loaded with verified method count
- Integrity hashes validated
- Configuration is immutable for the session

**Invariants:**
- Configuration never changes after loading
- All files must have valid integrity hashes
- No partial configuration loading (atomic operation)

**Abort Conditions:**
- Integrity hash mismatch → ABORT with "Configuration corrupted"
- Missing configuration files → ABORT with "Configuration incomplete"
- Invalid JSON structure → ABORT with "Configuration malformed"

**Structured Log Example:**
```json
{
  "timestamp": "2025-10-29T17:34:49.856Z",
  "phase": "FASE_0",
  "operation": "load_configuration",
  "status": "success",
  "metrics": {
    "questions_loaded": 305,
    "methods_loaded": 416,
    "integrity_validated": true
  },
  "artifacts": {
    "monolith_hash": "a3f5b2c8...",
    "catalog_hash": "d9e4a1f7..."
  }
}
```

---

### FASE 1: Document Ingestion

**Purpose:** Load, extract, and preprocess the municipal development plan PDF

**Preconditions:**
- PDF file exists and is accessible
- PDF is valid and not corrupted
- Sufficient memory for document processing
- Configuration loaded (FASE 0 complete)

**Process:**
1. Load PDF raw bytes and validate format
2. Extract full text preserving structure
3. Normalize unicode and encoding
4. Segment into sentences with location metadata
5. Extract and classify all tables (financial, activities, schedule, responsibilities)
6. Build search indexes (term, numeric, temporal, table)
7. Assemble immutable PreprocessedDocument

**Postconditions:**
- PreprocessedDocument created with all required fields
- All sentences have location metadata
- All tables are classified and cleaned
- All indexes are built and validated
- Document is immutable and cached

**Invariants:**
- PreprocessedDocument never mutates after creation
- Sentence count > 0
- All extracted tables are valid DataFrames
- Indexes are consistent with document content

**Abort Conditions:**
- PDF load failure → ABORT with "Invalid document format"
- Zero sentences extracted → ABORT with "Empty or unreadable document"
- Unicode normalization failure → ABORT with "Text encoding error"
- Table extraction exception → ABORT with "Table processing failure"

**Structured Log Example:**
```json
{
  "timestamp": "2025-10-29T17:35:12.234Z",
  "phase": "FASE_1",
  "operation": "document_ingestion",
  "status": "success",
  "metrics": {
    "pages": 127,
    "sentences": 1845,
    "tables_extracted": 23,
    "tables_classified": {
      "financial": 8,
      "activities": 9,
      "schedule": 4,
      "responsibilities": 2
    },
    "processing_time_ms": 8456
  },
  "artifacts": {
    "document_id": "doc_2025_1029_173512",
    "document_hash": "b7c9f3e1..."
  }
}
```

---

### FASE 2: Micro Question Processing (300 Questions)

**Purpose:** Execute all 300 micro questions in parallel using Choreographer instances

**Preconditions:**
- FASE 0 and FASE 1 complete
- PreprocessedDocument available
- Method catalog loaded
- 300 Choreographer instances can be spawned

**Process:**
1. For each of 300 questions (executed in parallel):
   - Create Choreographer instance
   - Resolve methods from catalog based on question flow spec
   - Build execution DAG for method dependencies
   - Execute methods via dispatcher (respecting sync/async)
   - Extract evidence from method results
   - Apply scoring modality (TYPE_A through TYPE_F)
   - Calculate quality level (EXCELENTE/BUENO/ACEPTABLE/INSUFICIENTE)
2. Wait for all 300 questions to complete (synchronization barrier)
3. Collect all QuestionResults

**Postconditions:**
- 300 QuestionResults generated (one per question)
- Each result has score, quality level, and evidence hash
- All results are validated against contracts
- No partial results (atomic completion of all 300)

**Invariants:**
- Each question executes exactly once
- Question execution is isolated (no shared mutable state)
- Evidence structure matches modality requirements
- Scores are reproducible (same input → same output)

**Abort Conditions:**
- Critical method failure in >10% of questions → ABORT with "Catastrophic analysis failure"
- Memory exhaustion → ABORT with "Resource exhaustion"
- Evidence validation failure → ABORT with "Contract violation in evidence"
- Timeout exceeded (configurable) → ABORT with "Processing timeout"

**Structured Log Example:**
```json
{
  "timestamp": "2025-10-29T17:42:33.567Z",
  "phase": "FASE_2",
  "operation": "micro_questions",
  "status": "success",
  "metrics": {
    "questions_processed": 300,
    "parallel_executions": 16,
    "total_time_ms": 445123,
    "avg_time_per_question_ms": 1483,
    "quality_distribution": {
      "EXCELENTE": 87,
      "BUENO": 142,
      "ACEPTABLE": 61,
      "INSUFICIENTE": 10
    }
  },
  "artifacts": {
    "results_hash": "f8d3a7c2..."
  }
}
```

---

### FASE 3-7: Aggregation Phases

**FASE 3: Dimension Aggregation (60 dimension scores)**
- **Precondition:** 300 micro results available
- **Process:** Aggregate questions by dimension (D1-D6) × policy areas (P1-P10)
- **Postcondition:** 60 DimensionScore objects created
- **Invariant:** Sum of dimension scores equals sum of underlying question scores

**FASE 4: Area Aggregation (10 policy area scores)**
- **Precondition:** 60 dimension scores available
- **Process:** Aggregate dimensions by policy area
- **Postcondition:** 10 AreaScore objects created
- **Invariant:** Area scores are weighted averages of dimension scores

**FASE 5: Cluster Analysis (4 MESO scores)**
- **Precondition:** 60 dimension scores and 10 area scores available
- **Process:** Perform meso-level cluster analysis (Q301-Q304)
- **Postcondition:** 4 ClusterScore objects created
- **Invariant:** Clusters are mutually exclusive and exhaustive

**FASE 6: Macro Synthesis (1 global score)**
- **Precondition:** All MESO scores available
- **Process:** Holistic evaluation (Q305)
- **Postcondition:** 1 MacroScore object created
- **Invariant:** Macro score reflects complete system state

**FASE 7: Report Assembly**
- **Precondition:** All 305 scores calculated (300 micro + 4 meso + 1 macro)
- **Process:** Assemble CompleteReport with all levels
- **Postcondition:** CompleteReport validated and ready for output
- **Invariant:** Report is immutable and reproducible

---

## Role Separation

### Orchestrator (orchestrator.py)

**Role:** Central coordinator and global state manager

**Responsibilities:**
- Load configuration and validate integrity
- Manage complete lifecycle (Phases 0-10)
- Coordinate sequential phase execution
- Distribute work to Choreographers
- Aggregate global results
- Handle global errors and make abort decisions
- Generate final report

**Canonical Category:** Core Orchestration & Choreography

**Contract:**
```python
"""
Preconditions:
  - Configuration files exist and are valid
  - System resources available (memory, CPU)
  
Postconditions:
  - Either complete 305-answer report OR explicit abort
  - All audit logs written
  - Resources cleaned up

Invariants:
  - Single point of entry for entire system
  - Global state is always consistent
  - No graceful degradation
"""
```

---

### Choreographer (orchestrator/choreographer.py)

**Role:** Distributed executor for individual questions

**Responsibilities:**
- Execute ONE question at a time
- Maintain isolation (no knowledge of other questions)
- Interpret method execution DAG
- Coordinate sync/async method execution
- Manage local state (single question only)
- Report results to Orchestrator

**Canonical Category:** Core Orchestration & Choreography

**Contract:**
```python
"""
Preconditions:
  - PreprocessedDocument available
  - Method catalog loaded
  - Question specification valid
  
Postconditions:
  - Single QuestionResult with validated evidence
  - Evidence structure matches modality requirements
  
Invariants:
  - No shared mutable state with other Choreographers
  - Execution is deterministic and reproducible
  - Result quality is deterministic
"""
```

---

### Core Arsenal Modules

**Role:** Primary analytical logic implementation

**Modules:**
1. **financiero_viabilidad_tablas.py** - Financial analysis and causal DAG
2. **Analyzer_one.py** - Semantic cube and value chain analysis
3. **contradiction_deteccion.py** - Contradiction detection and coherence analysis
4. **embedding_policy.py** - Semantic search and Bayesian inference
5. **teoria_cambio.py** - Theory of change DAG validation and Monte Carlo
6. **dereck_beach.py** - Process tracing and mechanism inference
7. **policy_processor.py** - Pattern matching and evidence extraction

**Canonical Category:** Core Arsenal

**Shared Contract:**
```python
"""
Preconditions:
  - Input data meets declared type and structure contracts
  - All required dependencies available
  - Seed initialized (for stochastic methods)
  
Postconditions:
  - Evidence object with required structure for modality
  - No side effects (pure functions preferred)
  - Deterministic output
  
Invariants:
  - Methods are stateless or explicitly manage state
  - All methods are instrumented for logging
  - No method degrades gracefully (abort on failure)
"""
```

---

## Contracts and Invariants

### Contract Structure

Every function/method in the system must declare:

```python
def method_name(parameters):
    """
    Brief description.
    
    Preconditions:
      - Condition 1 that must be true before execution
      - Condition 2 that must be validated
      - Condition 3 with specific value constraints
    
    Postconditions:
      - Condition 1 that is guaranteed after execution
      - Condition 2 about return value properties
      - Condition 3 about side effects (or lack thereof)
    
    Invariants:
      - Property 1 that holds throughout execution
      - Property 2 that never changes
      - Property 3 about object state
    
    Args:
        param1 (Type): Description
        param2 (Type): Description
    
    Returns:
        ReturnType: Description
    
    Raises:
        ExceptionType1: When condition X is violated
        ExceptionType2: When condition Y fails
    """
    # Validate preconditions
    assert condition1, "Precondition 1 violated"
    assert condition2, "Precondition 2 violated"
    
    # Execute logic
    result = perform_operation()
    
    # Validate postconditions
    assert result_valid, "Postcondition violated"
    
    return result
```

### Global Invariants

1. **Deterministic Reproducibility**
   - Same input → Same output (bit-for-bit)
   - All randomness controlled via seed_factory
   - No race conditions in concurrent execution

2. **Atomic Operations**
   - No partial state changes
   - Either complete success or complete rollback
   - No intermediate inconsistent states

3. **Audit Trail Completeness**
   - Every operation logged
   - Full reconstruction possible from logs
   - Performance metrics captured

4. **Type Safety**
   - No implicit type conversions
   - All types explicitly declared
   - Runtime type validation where appropriate

5. **Evidence Integrity**
   - Evidence structure matches modality requirements
   - Evidence hash computed consistently
   - Evidence is immutable after creation

---

## Abortability and Failure Handling

### Abort Principles

1. **Fail Fast, Fail Loud**
   - Detect violations immediately
   - Raise explicit, descriptive exceptions
   - Never return sentinel values (e.g., None) to indicate failure

2. **No Graceful Degradation**
   - Do not continue with partial results
   - Do not substitute default values
   - Do not skip failed operations

3. **Explicit Abort Conditions**
   - Every phase documents abort conditions
   - Abort messages are descriptive and actionable
   - Abort includes full context for debugging

### Abort Taxonomy

#### Critical Aborts (Unrecoverable)

**Configuration Integrity Failures:**
```python
# Abort: Configuration hash mismatch
raise ConfigurationError(
    "Monolith integrity hash mismatch",
    expected=declared_hash,
    actual=computed_hash,
    phase="FASE_0"
)
```

**Contract Violations:**
```python
# Abort: Evidence structure invalid
raise EvidenceStructureError(
    f"Evidence missing required keys for {modality}",
    missing_keys=missing,
    modality=modality,
    question_global=question_id
)
```

**Resource Exhaustion:**
```python
# Abort: Memory exhausted
raise ResourceExhausted(
    "Insufficient memory for parallel execution",
    required_mb=required,
    available_mb=available,
    phase="FASE_2"
)
```

#### Validation Aborts (Data Quality)

**Document Validation:**
```python
# Abort: Empty document
raise DocumentValidationError(
    "Zero sentences extracted from document",
    document_path=pdf_path,
    pages=num_pages,
    phase="FASE_1"
)
```

**Method Execution:**
```python
# Abort: Catastrophic analysis failure
raise AnalysisFailure(
    f"Method execution failed in {failure_rate}% of questions",
    failure_rate=failure_rate,
    threshold=0.10,
    failed_questions=failed_ids,
    phase="FASE_2"
)
```

### Abort Recovery (Investigation, Not Continuation)

When abort occurs:

1. **Log complete state**
   - Capture all parameters
   - Dump intermediate results
   - Record full stack trace

2. **Generate diagnostic report**
   - What was being attempted
   - What precondition failed
   - What the valid input should look like

3. **Clean up resources**
   - Close file handles
   - Release locks
   - Terminate child processes

4. **Exit with diagnostic code**
   - Unique error code per abort type
   - Error message includes resolution guidance
   - Full context preserved for debugging

---

## Structured Logging

### Log Levels

- **DEBUG:** Detailed diagnostic information
- **INFO:** General informational messages (phase transitions, milestones)
- **WARNING:** Something unexpected but non-fatal occurred
- **ERROR:** Operation failed but system may continue
- **CRITICAL:** System must abort immediately

### Log Structure (JSON)

All logs are structured JSON for machine parsing:

```json
{
  "timestamp": "ISO-8601 timestamp",
  "level": "INFO|DEBUG|WARNING|ERROR|CRITICAL",
  "phase": "FASE_0|FASE_1|...",
  "operation": "operation_name",
  "status": "started|success|failure|aborted",
  "message": "Human-readable message",
  "metrics": {
    "key1": value1,
    "key2": value2
  },
  "artifacts": {
    "hash1": "sha256_hash",
    "path1": "/path/to/artifact"
  },
  "context": {
    "question_global": 123,
    "modality": "TYPE_A",
    "dimension": "D1"
  },
  "error": {
    "exception": "ExceptionClass",
    "message": "Error details",
    "stack_trace": "Full stack trace"
  }
}
```

### Example: Success Log

```json
{
  "timestamp": "2025-10-29T17:35:42.123Z",
  "level": "INFO",
  "phase": "FASE_2",
  "operation": "process_micro_question",
  "status": "success",
  "message": "Question processed successfully",
  "metrics": {
    "processing_time_ms": 1234,
    "methods_executed": 12,
    "evidence_elements": 8
  },
  "artifacts": {
    "evidence_hash": "a7f3d9c2e8b1f456...",
    "result_score": 3.6
  },
  "context": {
    "question_global": 42,
    "base_slot": "PA03-DIM02-Q007",
    "modality": "TYPE_A",
    "quality_level": "EXCELENTE"
  }
}
```

### Example: Abort Log

```json
{
  "timestamp": "2025-10-29T17:36:15.789Z",
  "level": "CRITICAL",
  "phase": "FASE_0",
  "operation": "load_configuration",
  "status": "aborted",
  "message": "Configuration integrity validation failed",
  "error": {
    "exception": "ConfigurationError",
    "message": "Monolith integrity hash mismatch",
    "expected_hash": "a3f5b2c8d9e4a1f7...",
    "actual_hash": "f7e1a4d9c8b2f5a3...",
    "stack_trace": "Traceback (most recent call last):\n  ..."
  },
  "context": {
    "config_file": "questionnaire_monolith.json",
    "file_size_bytes": 1234567
  },
  "resolution": "Verify configuration file has not been corrupted. Re-download from trusted source."
}
```

### Example: Performance Log

```json
{
  "timestamp": "2025-10-29T17:42:33.567Z",
  "level": "INFO",
  "phase": "FASE_2",
  "operation": "parallel_execution_summary",
  "status": "success",
  "message": "All micro questions completed",
  "metrics": {
    "total_questions": 300,
    "successful": 300,
    "failed": 0,
    "total_time_ms": 445123,
    "avg_time_ms": 1483,
    "median_time_ms": 1402,
    "p95_time_ms": 2134,
    "p99_time_ms": 2876,
    "parallel_workers": 16
  },
  "artifacts": {
    "results_hash": "f8d3a7c2b9e5f1a6..."
  }
}
```

---

## Deterministic Reproducibility

### Seed Management

All stochastic operations use explicit seeds from `seed_factory`:

```python
from seed_factory import SeedFactory

# Initialize seed factory (deterministic mode)
seed_factory = SeedFactory(master_seed=42)

# Get seed for specific operation
seed = seed_factory.get_seed("operation_name", question_id=123)

# Use seed in stochastic operation
result = monte_carlo_simulation(seed=seed)
```

**Invariants:**
- Master seed is set once at system initialization
- Same operation + same parameters → same seed
- Seed usage is logged for auditability

### Concurrency Control

Parallel execution must not introduce non-determinism:

1. **Isolated State:** Each Choreographer has independent state
2. **No Shared Mutables:** All shared data is immutable
3. **Deterministic Ordering:** Results aggregated in deterministic order
4. **Synchronization Barriers:** Explicit WAIT points for coordination

### Reproducibility Validation

The system includes reproducibility tests:

```python
def test_reproducibility():
    """Verify same input produces identical output."""
    
    # Run 1
    result1 = process_development_plan(
        pdf_path="test.pdf",
        seed=42
    )
    
    # Run 2 (identical parameters)
    result2 = process_development_plan(
        pdf_path="test.pdf",
        seed=42
    )
    
    # Verify bit-for-bit identity
    assert result1.report_hash == result2.report_hash
    assert result1.scores == result2.scores
    assert result1.quality_levels == result2.quality_levels
```

---

## Summary

The SIN_CARRETA architecture embodies six Prime Directives:

1. **No Graceful Degradation** - Complete success or explicit abort
2. **No Strategic Simplification** - Complexity serves fidelity
3. **State-of-the-Art Baseline** - Research-grade approaches
4. **Deterministic Reproducibility** - Controlled non-determinism
5. **Explicitness Over Assumption** - Declared contracts
6. **Observability as Structure** - Mandatory audit trail

The system processes 305 questions through 11 sequential phases, using an Orchestrator/Choreographer pattern with 7 Core Arsenal producer modules. Every component has explicit contracts with preconditions, postconditions, and invariants. All failures abort with descriptive diagnostics. All operations are logged in structured JSON format. All execution is deterministically reproducible.

**The doctrine is absolute. The architecture is non-negotiable.**

