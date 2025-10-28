# Choreographer Dispatch & Evidence Registry

## Overview

This document describes the implementation of three integrated systems for method orchestration, quality monitoring, and evidence tracking in the SAAAAAA project:

1. **Registry Foundation**: Canonical method_class_map loader with coverage validation
2. **Choreographer Dispatch**: FQN-based method invocation with QMCM interceptor
3. **Evidence Registry**: Append-only JSONL store with hash indexing and provenance DAG export

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                      │
│  (D1/D2 Orchestrators)                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Choreographer Dispatcher                       │
│  - FQN resolution via canonical registry                    │
│  - Context-aware argument binding                           │
│  - QMCM interception                                        │
└────┬────────────────────────────────────┬────────────────────┘
     │                                    │
     ▼                                    ▼
┌─────────────────┐              ┌──────────────────────────┐
│ QMCM Recorder   │              │  Evidence Registry       │
│ - Method calls  │              │  - JSONL storage         │
│ - Quality stats │              │  - Hash indexing         │
│ - Frequency     │              │  - Provenance DAG        │
└─────────────────┘              └──────────────────────────┘
```

## 1. Registry Foundation

### Purpose
Validates that the canonical method registry contains ≥400 methods (provisional threshold) or ≥555 methods (strict threshold).

### Components

#### `orchestrator/canonical_registry.py`
- **CANONICAL_METHODS**: Global registry mapping FQN → callable
- **validate_method_registry()**: Validates threshold compliance
- **generate_audit_report()**: Generates comprehensive audit JSON

### Current Status
✅ **PASSING**: 416 methods ≥ 400 (provisional threshold)

### Usage

```python
from orchestrator.canonical_registry import (
    CANONICAL_METHODS,
    validate_method_registry,
    generate_audit_report,
)

# Validate registry
validation = validate_method_registry(provisional=True)
print(f"Total methods: {validation['total_methods']}")
print(f"Passed: {validation['passed']}")

# Generate audit
audit = generate_audit_report(output_path=Path("audit.json"))
```

## 2. Choreographer Dispatch

### Purpose
Replace direct component calls with FQN-based invocation through a centralized dispatcher that intercepts all method calls for quality monitoring.

### Components

#### `orchestrator/choreographer_dispatch.py`

##### Classes

**ChoreographerDispatcher**
- Central FQN-based method invocation
- QMCM integration for all method calls
- Optional evidence recording
- Context-aware argument binding

**InvocationContext**
- Rich context for method invocation
- Instance pool for dependency injection
- Support for text, data, document, metadata

**InvocationResult**
- Full traceability of method execution
- Success/failure status
- Execution time
- QMCM recording status

### Features

1. **FQN Resolution**: `"IndustrialPolicyProcessor.process"` → callable
2. **QMCM Interception**: All invocations automatically recorded
3. **Context Binding**: Smart argument extraction from context
4. **Evidence Recording**: Optional evidence capture
5. **Error Handling**: Graceful error propagation

### Usage

```python
from orchestrator.choreographer_dispatch import (
    ChoreographerDispatcher,
    InvocationContext,
    invoke_method,
)

# Create dispatcher
dispatcher = ChoreographerDispatcher()

# Create context
context = InvocationContext(
    text="policy document text...",
    question_id="D1-Q1",
)

# Invoke method by FQN (includes QMCM interception)
result = dispatcher.invoke_method(
    "PolicyTextProcessor.segment_into_sentences",
    context
)

print(f"Success: {result.success}")
print(f"Result: {result.result}")
print(f"QMCM recorded: {result.qmcm_recorded}")

# Or use convenience function
result = invoke_method("PolicyTextProcessor.segment_into_sentences", context)
```

### Integration with Orchestrators

The D1 orchestrator has been updated to support choreographer dispatch:

```python
from orchestrator.d1_orchestrator import D1QuestionOrchestrator

# Create orchestrator with choreographer enabled
orchestrator = D1QuestionOrchestrator(
    enable_choreographer=True,  # Use FQN-based dispatch
    enable_evidence=True,       # Record evidence
)

# Orchestrate question (all methods invoked via dispatcher)
result = orchestrator.orchestrate_question(
    question=D1Question.Q1_BASELINE,
    context={"text": "policy text..."},
)
```

## 3. Evidence Registry

### Purpose
Create an immutable, append-only evidence store with cryptographic verification and provenance tracking.

### Components

#### `orchestrator/evidence_registry.py`

##### Classes

**EvidenceRecord**
- Immutable evidence with SHA-256 content hash
- Provenance metadata (source, parents, timestamp)
- Integrity verification

**ProvenanceDAG**
- Directed Acyclic Graph of evidence dependencies
- Ancestor/descendant tracking
- Export to GraphViz DOT format

**EvidenceRegistry**
- Append-only JSONL storage
- Content-addressable hash indexing (SHA-256)
- Multiple query indices (type, method, question)
- Provenance DAG export
- Persistence across sessions

### Features

1. **Append-Only Storage**: JSONL format for immutability
2. **Hash Indexing**: SHA-256 content-addressable storage
3. **Provenance DAG**: Full lineage tracking
4. **Integrity Verification**: Cryptographic hash validation
5. **Multiple Indices**: Query by type, method, question
6. **DAG Export**: Dict, JSON, GraphViz DOT formats

### Usage

```python
from orchestrator.evidence_registry import (
    EvidenceRegistry,
    get_global_registry,
)

# Create registry
registry = EvidenceRegistry(
    storage_path=Path("evidence.jsonl"),
    enable_dag=True,
)

# Record evidence
evidence_id = registry.record_evidence(
    evidence_type="analysis",
    payload={"result": "analysis data"},
    source_method="Analyzer.analyze",
    parent_evidence_ids=[parent_id],  # Link to dependencies
    question_id="D1-Q1",
)

# Retrieve evidence
evidence = registry.get_evidence(evidence_id)

# Query evidence
analysis_evidence = registry.query_by_type("analysis")
q1_evidence = registry.query_by_question("D1-Q1")

# Get provenance
provenance = registry.get_provenance(evidence_id)
print(f"Ancestors: {provenance['ancestor_count']}")
print(f"Descendants: {provenance['descendant_count']}")

# Export DAG
dag_dot = registry.export_provenance_dag(
    format="dot",
    output_path=Path("provenance.dot")
)

# Visualize with: dot -Tpng provenance.dot -o provenance.png
```

### Evidence Types

Common evidence types:
- `extraction`: Raw data extraction from sources
- `analysis`: Analysis/processing results
- `recommendation`: Generated recommendations
- `validation`: Validation results
- `method_result`: Generic method execution results

## QMCM Integration

All method invocations through the choreographer dispatcher are automatically recorded via QMCM (Quality Method Call Monitoring):

```python
from qmcm_hooks import get_global_recorder

# Get recorder
recorder = get_global_recorder()

# Get statistics
stats = recorder.get_statistics()
print(f"Total calls: {stats['total_calls']}")
print(f"Success rate: {stats['success_rate']}")
print(f"Method frequency: {stats['method_frequency']}")

# Save recording
recorder.save_recording()
```

## Complete Workflow Example

```python
from orchestrator.choreographer_dispatch import ChoreographerDispatcher, InvocationContext
from orchestrator.evidence_registry import EvidenceRegistry

# Setup
registry = {"Analyzer.analyze": analyzer_method}
dispatcher = ChoreographerDispatcher(registry=registry)
evidence_reg = EvidenceRegistry()

# Step 1: Extract
ctx = InvocationContext(text="policy text")
result = dispatcher.invoke_method("Extractor.extract", ctx)
e1_id = evidence_reg.record_evidence(
    evidence_type="extraction",
    payload={"result": result.result},
    source_method="Extractor.extract",
)

# Step 2: Analyze (depends on extraction)
ctx = InvocationContext(data=result.result)
result = dispatcher.invoke_method("Analyzer.analyze", ctx)
e2_id = evidence_reg.record_evidence(
    evidence_type="analysis",
    payload=result.result,
    source_method="Analyzer.analyze",
    parent_evidence_ids=[e1_id],  # Link to extraction
)

# Export provenance
lineage = evidence_reg.get_provenance(e2_id)
evidence_reg.export_provenance_dag(format="dot", output_path=Path("dag.dot"))
```

## Testing

All components have comprehensive test coverage:

### Run Tests
```bash
# Registry foundation
python tests/test_canonical_registry.py

# Choreographer dispatch
python tests/test_choreographer_dispatch.py

# Evidence registry
python tests/test_evidence_registry.py

# Run demo
python examples/choreographer_evidence_demo.py
```

### Test Coverage
- ✅ `test_canonical_registry.py`: 5 tests
- ✅ `test_choreographer_dispatch.py`: 13 tests
- ✅ `test_evidence_registry.py`: 17 tests

**Total: 35 tests, all passing**

## File Structure

```
orchestrator/
├── canonical_registry.py       # Registry foundation & validation
├── choreographer_dispatch.py   # FQN-based dispatch with QMCM
├── evidence_registry.py        # JSONL evidence store + DAG
├── d1_orchestrator.py         # D1 orchestrator (updated)
└── d2_activities_orchestrator.py

tests/
├── test_canonical_registry.py
├── test_choreographer_dispatch.py
└── test_evidence_registry.py

examples/
└── choreographer_evidence_demo.py

qmcm_hooks.py                  # QMCM quality monitoring
```

## Key Design Decisions

### 1. Append-Only JSONL Storage
- **Why**: Immutability for audit trail
- **Format**: One JSON object per line
- **Benefits**: Easy to append, grep, stream

### 2. SHA-256 Content Hashing
- **Why**: Content-addressable storage
- **Use**: Evidence deduplication, integrity verification
- **Benefits**: Cryptographic chain-of-custody

### 3. Provenance DAG
- **Why**: Full lineage tracking
- **Structure**: Parent-child relationships
- **Export**: GraphViz DOT for visualization

### 4. QMCM Interception
- **Why**: Universal quality monitoring
- **Integration**: Transparent to method implementations
- **Benefits**: No code changes required for tracking

### 5. FQN-based Dispatch
- **Why**: Decoupling, testability, traceability
- **Format**: `"ClassName.method_name"`
- **Benefits**: Uniform invocation, registry validation

## Validation Results

### Registry Foundation
- **Total Methods**: 416
- **Threshold**: 400 (provisional) ✅
- **Coverage**: Successfully validated

### Integration Tests
- **Choreographer Dispatch**: 13/13 tests passing ✅
- **Evidence Registry**: 17/17 tests passing ✅
- **End-to-End**: Demo script runs successfully ✅

## Future Enhancements

1. **Registry**: Increase to ≥555 methods (strict threshold)
2. **Dispatch**: Add async/concurrent method invocation
3. **Evidence**: Add digital signatures for evidence records
4. **DAG**: Add cycle detection and validation
5. **Visualization**: Auto-generate provenance diagrams

## References

- `orchestrator/canonical_registry.py`: Registry implementation
- `orchestrator/choreographer_dispatch.py`: Dispatch implementation
- `orchestrator/evidence_registry.py`: Evidence store implementation
- `examples/choreographer_evidence_demo.py`: Complete examples
- `COMPLETE_METHOD_CLASS_MAP.json`: Method inventory (416 methods)

---

**Status**: ✅ All requirements implemented and tested
**Date**: 2025-10-28
**Version**: 1.0.0
