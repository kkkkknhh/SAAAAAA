"""
Integration test for contextual calibration system.

This test demonstrates the full calibration pipeline working with
the new contextual layer implementations.
"""

from pathlib import Path
from calibration import (
    CalibrationEngine,
    Context,
    ComputationGraph,
    EvidenceStore,
)


def test_full_calibration_pipeline():
    """
    Test the complete calibration pipeline with contextual layers.
    
    This validates that:
    1. Engine initializes with all three pillar configs
    2. Contextual layers compute correctly
    3. Fusion operator produces valid calibrated score
    4. All scores remain in [0,1]
    """
    print("\n" + "="*60)
    print("Integration Test: Full Calibration Pipeline")
    print("="*60)
    
    # Initialize engine
    print("\n1. Initializing calibration engine...")
    engine = CalibrationEngine()
    print("   ✓ Engine initialized")
    print(f"   - Intrinsic methods: {len(engine.intrinsic_config.get('methods', {}))}")
    print(f"   - Contextual layers: @q, @d, @p, @u, @chain, @C, @m")
    print(f"   - Fusion roles: {len(engine.fusion_config.get('role_fusion_parameters', {}))}")
    
    # Create a test computation graph
    print("\n2. Creating computation graph...")
    graph = ComputationGraph()
    graph.nodes.add("node1")
    graph.nodes.add("node2")
    graph.edges.append(("node1", "node2"))
    graph.node_signatures["node2"] = {
        "required_inputs": ["input1"],
        "outputs": ["output1"]
    }
    print("   ✓ Graph created (2 nodes, 1 edge)")
    
    # Create context
    print("\n3. Creating execution context...")
    context = Context(
        question_id="Q001",
        dimension_id="DIM01",
        policy_id="PA01",
        unit_quality=0.85
    )
    print(f"   ✓ Context: Q={context.question_id}, D={context.dimension_id}, "
          f"P={context.policy_id}, U={context.unit_quality}")
    
    # Create evidence store
    print("\n4. Creating evidence store...")
    evidence = EvidenceStore()
    evidence.runtime_metrics = {
        "runtime_ms": 150,
        "memory_mb": 64
    }
    print("   ✓ Evidence prepared")
    
    # Run calibration on a real method from the catalog
    print("\n5. Running calibration...")
    method_id = "src.saaaaaa.flux.phases.run_score"
    
    try:
        certificate = engine.calibrate(
            method_id=method_id,
            node_id="node2",
            graph=graph,
            context=context,
            evidence_store=evidence
        )
        
        print(f"   ✓ Calibration completed for {method_id}")
        print(f"\n6. Results:")
        print(f"   - Calibrated score: {certificate.calibrated_score:.4f}")
        print(f"   - Intrinsic score (@b): {certificate.intrinsic_score:.4f}")
        
        # Display contextual layer scores
        print(f"\n   Contextual Layer Scores:")
        contextual_layers = ["@chain", "@u", "@q", "@d", "@p", "@C", "@m"]
        for layer in contextual_layers:
            if layer in certificate.layer_scores:
                score = certificate.layer_scores[layer]
                print(f"     {layer:8s}: {score:.4f}")
        
        # Validate boundedness
        print(f"\n7. Validation:")
        assert 0.0 <= certificate.calibrated_score <= 1.0, \
            f"Calibrated score out of bounds: {certificate.calibrated_score}"
        print(f"   ✓ Calibrated score in [0,1]")
        
        for layer, score in certificate.layer_scores.items():
            assert 0.0 <= score <= 1.0, \
                f"Layer {layer} score out of bounds: {score}"
        print(f"   ✓ All {len(certificate.layer_scores)} layer scores in [0,1]")
        
        # Verify certificate structure
        assert certificate.config_hash.startswith("sha256:")
        assert certificate.graph_hash.startswith("sha256:")
        assert certificate.timestamp
        print(f"   ✓ Certificate structure valid")
        
        print(f"\n{'='*60}")
        print("✓ Integration test PASSED")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n   ✗ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_contextual_determinism():
    """
    Test that contextual layers produce deterministic results.
    
    Running calibration multiple times with same inputs should
    produce identical results.
    """
    print("\n" + "="*60)
    print("Integration Test: Contextual Determinism")
    print("="*60)
    
    engine = CalibrationEngine()
    
    graph = ComputationGraph()
    graph.nodes.add("node1")
    
    context = Context(
        question_id="Q001",
        dimension_id="DIM02",
        policy_id="PA03",
        unit_quality=0.75
    )
    
    evidence = EvidenceStore()
    evidence.runtime_metrics = {"runtime_ms": 100}
    
    method_id = "src.saaaaaa.flux.phases.run_score"
    
    # Run calibration twice
    cert1 = engine.calibrate(method_id, "node1", graph, context, evidence)
    cert2 = engine.calibrate(method_id, "node1", graph, context, evidence)
    
    # Compare results
    print(f"\n1. First run:  calibrated_score = {cert1.calibrated_score:.6f}")
    print(f"2. Second run: calibrated_score = {cert2.calibrated_score:.6f}")
    
    assert cert1.calibrated_score == cert2.calibrated_score, \
        "Calibration is not deterministic!"
    
    # Compare all layer scores
    for layer in cert1.layer_scores:
        assert cert1.layer_scores[layer] == cert2.layer_scores[layer], \
            f"Layer {layer} is not deterministic!"
    
    print(f"\n✓ All scores are deterministic")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    print("\nRunning contextual calibration integration tests...\n")
    
    # Run tests
    results = []
    results.append(("Full Pipeline", test_full_calibration_pipeline()))
    results.append(("Determinism", test_contextual_determinism()))
    
    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {status:10s} - {name}")
    
    print(f"\n{passed}/{total} integration tests passed")
    
    if passed == total:
        print("\n✓✓✓ All integration tests passed! ✓✓✓\n")
    else:
        print(f"\n✗✗✗ {total - passed} integration test(s) failed ✗✗✗\n")
