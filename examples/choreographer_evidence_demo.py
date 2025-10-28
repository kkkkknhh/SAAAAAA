#!/usr/bin/env python3
"""
Example demonstrating choreographer dispatch and evidence registry integration.

This example shows how to:
1. Use the canonical method registry
2. Invoke methods via choreographer dispatch with QMCM tracking
3. Record evidence in the registry
4. Export provenance DAG
5. Generate audit reports
"""

import sys
from pathlib import Path
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.canonical_registry import (
    CANONICAL_METHODS,
    validate_method_registry,
    generate_audit_report,
)
from orchestrator.choreographer_dispatch import (
    ChoreographerDispatcher,
    InvocationContext,
    get_global_dispatcher,
)
from orchestrator.evidence_registry import (
    EvidenceRegistry,
    get_global_registry,
)
from qmcm_hooks import get_global_recorder


def example_1_registry_validation():
    """Example 1: Validate canonical registry and generate audit report."""
    print("=" * 80)
    print("Example 1: Registry Validation")
    print("=" * 80)
    
    # Validate method registry (provisional threshold: ≥400 methods)
    validation = validate_method_registry(provisional=True)
    
    print(f"\n✓ Registry Validation:")
    print(f"  - Total methods: {validation['total_methods']}")
    print(f"  - Threshold: {validation['threshold']} ({validation['threshold_type']})")
    print(f"  - Passed: {validation['passed']}")
    print(f"  - Coverage: {validation['coverage_percentage']}%")
    
    # Generate audit report
    audit_path = Path("example_audit.json")
    audit = generate_audit_report(CANONICAL_METHODS, audit_path)
    
    print(f"\n✓ Audit Report Generated: {audit_path}")
    print(f"  - Total methods in codebase: {audit['coverage']['total_methods_in_codebase']}")
    print(f"  - Declared in metadata: {audit['coverage']['declared_in_metadata']}")
    print(f"  - Successfully resolved: {audit['coverage']['successfully_resolved']}")
    print(f"  - Missing methods: {len(audit['missing'])}")
    
    # Clean up
    if audit_path.exists():
        audit_path.unlink()


def example_2_choreographer_dispatch():
    """Example 2: Use choreographer dispatch for method invocation."""
    print("\n" + "=" * 80)
    print("Example 2: Choreographer Dispatch")
    print("=" * 80)
    
    # Create mock method for demonstration
    def mock_analysis(text: str) -> dict:
        return {
            "word_count": len(text.split()),
            "char_count": len(text),
            "analysis_type": "mock",
        }
    
    # Create dispatcher with mock registry
    registry = {"MockAnalyzer.analyze": mock_analysis}
    dispatcher = ChoreographerDispatcher(
        registry=registry,
        enable_evidence_recording=True,
    )
    
    # Create invocation context
    context = InvocationContext(
        text="Sample policy document for analysis",
        question_id="D1-Q1",
    )
    
    # Invoke method via FQN (includes QMCM interception)
    result = dispatcher.invoke_method("MockAnalyzer.analyze", context)
    
    print(f"\n✓ Method Invocation:")
    print(f"  - FQN: {result.fqn}")
    print(f"  - Success: {result.success}")
    print(f"  - Execution time: {result.execution_time_ms:.2f}ms")
    print(f"  - QMCM recorded: {result.qmcm_recorded}")
    print(f"  - Result: {result.result}")
    
    # Get dispatcher statistics
    stats = dispatcher.get_invocation_stats()
    print(f"\n✓ Dispatcher Statistics:")
    print(f"  - Registry size: {stats['registry_size']}")
    print(f"  - Evidence records: {stats['evidence_records']}")
    print(f"  - QMCM calls: {stats['qmcm_stats']['total_calls']}")


def example_3_evidence_registry():
    """Example 3: Record evidence and export provenance DAG."""
    print("\n" + "=" * 80)
    print("Example 3: Evidence Registry")
    print("=" * 80)
    
    # Create evidence registry
    storage_path = Path("example_evidence.jsonl")
    registry = EvidenceRegistry(
        storage_path=storage_path,
        enable_dag=True,
    )
    
    # Record evidence chain (simulating analysis pipeline)
    
    # Step 1: Extract raw data
    e1_id = registry.record_evidence(
        evidence_type="extraction",
        payload={
            "extracted_text": "Policy document section...",
            "source": "document.pdf",
            "page": 1,
        },
        source_method="PDFProcessor.extract_text",
        question_id="D1-Q1",
    )
    print(f"\n✓ Recorded extraction evidence: {e1_id[:16]}...")
    
    # Step 2: Analyze extracted data
    e2_id = registry.record_evidence(
        evidence_type="analysis",
        payload={
            "sentiment": "neutral",
            "complexity_score": 0.75,
            "key_terms": ["policy", "implementation", "resources"],
        },
        source_method="SemanticAnalyzer.analyze",
        parent_evidence_ids=[e1_id],
        question_id="D1-Q1",
    )
    print(f"✓ Recorded analysis evidence: {e2_id[:16]}...")
    
    # Step 3: Generate recommendation
    e3_id = registry.record_evidence(
        evidence_type="recommendation",
        payload={
            "recommendation": "Increase resource allocation",
            "confidence": 0.85,
            "rationale": "Based on complexity analysis",
        },
        source_method="RecommendationEngine.generate",
        parent_evidence_ids=[e1_id, e2_id],
        question_id="D1-Q1",
    )
    print(f"✓ Recorded recommendation evidence: {e3_id[:16]}...")
    
    # Verify evidence integrity
    for eid in [e1_id, e2_id, e3_id]:
        is_valid = registry.verify_evidence(eid)
        print(f"  - {eid[:16]}... verified: {is_valid}")
    
    # Query evidence
    print(f"\n✓ Evidence Queries:")
    q1_evidence = registry.query_by_question("D1-Q1")
    print(f"  - Evidence for D1-Q1: {len(q1_evidence)} records")
    
    analysis_evidence = registry.query_by_type("analysis")
    print(f"  - Analysis evidence: {len(analysis_evidence)} records")
    
    # Get provenance for final recommendation
    provenance = registry.get_provenance(e3_id)
    print(f"\n✓ Provenance for recommendation:")
    print(f"  - Ancestors: {provenance['ancestor_count']}")
    print(f"  - Descendants: {provenance['descendant_count']}")
    
    # Export provenance DAG
    dag_path = Path("example_provenance.dot")
    registry.export_provenance_dag(format="dot", output_path=dag_path)
    print(f"\n✓ Exported provenance DAG: {dag_path}")
    print(f"  (Can be visualized with: dot -Tpng {dag_path} -o provenance.png)")
    
    # Get registry statistics
    stats = registry.get_statistics()
    print(f"\n✓ Registry Statistics:")
    print(f"  - Total evidence: {stats['total_evidence']}")
    print(f"  - By type: {stats['by_type']}")
    print(f"  - DAG nodes: {stats['dag_nodes']}")
    
    # Clean up
    if storage_path.exists():
        storage_path.unlink()
    if dag_path.exists():
        dag_path.unlink()


def example_4_qmcm_integration():
    """Example 4: QMCM quality monitoring integration."""
    print("\n" + "=" * 80)
    print("Example 4: QMCM Integration")
    print("=" * 80)
    
    # Get global QMCM recorder
    recorder = get_global_recorder()
    recorder.clear_recording()
    
    # Create dispatcher (uses global recorder)
    def method1():
        return "result1"
    
    def method2():
        return "result2"
    
    registry = {
        "Class1.method1": method1,
        "Class2.method2": method2,
    }
    
    dispatcher = ChoreographerDispatcher(registry=registry)
    
    # Make some invocations (all intercepted by QMCM)
    dispatcher.invoke_method("Class1.method1")
    dispatcher.invoke_method("Class2.method2")
    dispatcher.invoke_method("Class1.method1")
    dispatcher.invoke_method("Class2.method2")
    
    # Get QMCM statistics
    stats = recorder.get_statistics()
    
    print(f"\n✓ QMCM Recording Statistics:")
    print(f"  - Total calls: {stats['total_calls']}")
    print(f"  - Unique methods: {stats['unique_methods']}")
    print(f"  - Success rate: {stats['success_rate'] * 100:.1f}%")
    print(f"  - Most called: {stats['most_called_method']}")
    print(f"\n  Method frequency:")
    for method, count in stats['method_frequency'].items():
        print(f"    - {method}: {count} calls")
    
    # Save QMCM recording
    recording_path = Path("example_qmcm_recording.json")
    recorder.recording_path = recording_path
    recorder.save_recording()
    print(f"\n✓ Saved QMCM recording: {recording_path}")
    
    # Clean up
    if recording_path.exists():
        recording_path.unlink()


def example_5_integrated_workflow():
    """Example 5: Integrated workflow with all components."""
    print("\n" + "=" * 80)
    print("Example 5: Integrated Workflow")
    print("=" * 80)
    
    # Mock methods for pipeline
    def extract_text(document: str) -> str:
        return f"Extracted from {document}"
    
    def analyze_text(text: str) -> dict:
        return {"analysis": "complete", "score": 0.8}
    
    def generate_report(data: dict) -> dict:
        return {"report": "generated", "data": data}
    
    # Set up infrastructure
    registry = {
        "Extractor.extract": extract_text,
        "Analyzer.analyze": analyze_text,
        "Reporter.generate": generate_report,
    }
    
    storage_path = Path("example_integrated.jsonl")
    evidence_registry = EvidenceRegistry(
        storage_path=storage_path,
        enable_dag=True,
    )
    
    dispatcher = ChoreographerDispatcher(
        registry=registry,
        enable_evidence_recording=False,  # We'll record manually for better control
    )
    
    print("\n✓ Processing pipeline:")
    
    # Step 1: Extract
    ctx1 = InvocationContext(document="policy.pdf")
    result1 = dispatcher.invoke_method("Extractor.extract", ctx1)
    e1_id = evidence_registry.record_evidence(
        evidence_type="extraction",
        payload={"result": result1.result},
        source_method="Extractor.extract",
        execution_time_ms=result1.execution_time_ms,
    )
    print(f"  1. Extract: {result1.success} ({result1.execution_time_ms:.2f}ms) -> {e1_id[:8]}...")
    
    # Step 2: Analyze
    ctx2 = InvocationContext(text=result1.result)
    result2 = dispatcher.invoke_method("Analyzer.analyze", ctx2)
    e2_id = evidence_registry.record_evidence(
        evidence_type="analysis",
        payload=result2.result,
        source_method="Analyzer.analyze",
        parent_evidence_ids=[e1_id],
        execution_time_ms=result2.execution_time_ms,
    )
    print(f"  2. Analyze: {result2.success} ({result2.execution_time_ms:.2f}ms) -> {e2_id[:8]}...")
    
    # Step 3: Report
    ctx3 = InvocationContext(data=result2.result)
    result3 = dispatcher.invoke_method("Reporter.generate", ctx3)
    e3_id = evidence_registry.record_evidence(
        evidence_type="report",
        payload=result3.result,
        source_method="Reporter.generate",
        parent_evidence_ids=[e2_id],
        execution_time_ms=result3.execution_time_ms,
    )
    print(f"  3. Report: {result3.success} ({result3.execution_time_ms:.2f}ms) -> {e3_id[:8]}...")
    
    # Show lineage
    lineage = evidence_registry.get_provenance(e3_id)
    print(f"\n✓ Report lineage:")
    print(f"  - Built on {lineage['ancestor_count']} prior evidence records")
    print(f"  - Ancestors: {[eid[:8] + '...' for eid in lineage['ancestors']]}")
    
    # Export complete DAG
    dag_dict = evidence_registry.export_provenance_dag(format="dict")
    print(f"\n✓ Complete provenance DAG:")
    print(f"  - Total nodes: {dag_dict['stats']['total_nodes']}")
    print(f"  - By type: {dag_dict['stats']['by_type']}")
    
    # Clean up
    if storage_path.exists():
        storage_path.unlink()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("CHOREOGRAPHER DISPATCH & EVIDENCE REGISTRY EXAMPLES")
    print("=" * 80)
    
    example_1_registry_validation()
    example_2_choreographer_dispatch()
    example_3_evidence_registry()
    example_4_qmcm_integration()
    example_5_integrated_workflow()
    
    print("\n" + "=" * 80)
    print("✅ All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
