#!/usr/bin/env python3
"""
Example: Using the Three-Pillar Calibration System

This script demonstrates how to use the calibration system to calibrate
method instances in the policy pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration import (
    calibrate,
    Context,
    ComputationGraph,
    EvidenceStore,
    LayerType
)


def example_basic_calibration():
    """Basic calibration example"""
    print("=" * 70)
    print("Example 1: Basic Calibration")
    print("=" * 70)
    print()
    
    # Define execution context
    context = Context(
        question_id="Q001",
        dimension_id="DIM01",
        policy_id="PA01",
        unit_quality=0.85
    )
    
    # Create simple computation graph
    graph = ComputationGraph(
        nodes={"scorer_node"},
        edges=[],
        node_signatures={
            "scorer_node": {
                "required_inputs": [],
                "optional_inputs": []
            }
        }
    )
    
    # Provide evidence
    evidence = EvidenceStore(
        runtime_metrics={
            "runtime_ms": 450
        }
    )
    
    # Calibrate a scoring method
    certificate = calibrate(
        method_id="src.saaaaaa.flux.phases.run_score",
        node_id="scorer_node",
        graph=graph,
        context=context,
        evidence_store=evidence
    )
    
    # Display results
    print(f"Method: {certificate.method_id}")
    print(f"Calibrated Score: {certificate.calibrated_score:.4f}")
    print(f"Intrinsic Score: {certificate.intrinsic_score:.4f}")
    print()
    
    print("Layer Breakdown:")
    for layer, score in certificate.layer_scores.items():
        print(f"  {layer:12s}: {score:.4f}")
    print()
    
    print("Fusion Details:")
    fusion = certificate.fusion_formula
    print(f"  Linear sum:      {fusion['linear_sum']:.4f}")
    print(f"  Interaction sum: {fusion['interaction_sum']:.4f}")
    print(f"  Total:           {fusion['total']:.4f}")
    print()


def example_context_sensitivity():
    """Demonstrate context sensitivity"""
    print("=" * 70)
    print("Example 2: Context Sensitivity")
    print("=" * 70)
    print()
    
    graph = ComputationGraph(nodes={"n1"})
    evidence = EvidenceStore()
    
    # Same method, different contexts
    contexts = [
        Context(question_id="Q001", unit_quality=0.85),
        Context(question_id="Q001", unit_quality=0.50),
        Context(question_id="Q001", unit_quality=0.30),
    ]
    
    print("Calibrating same method with different unit_quality values:")
    print()
    
    for i, ctx in enumerate(contexts, 1):
        cert = calibrate(
            method_id="src.saaaaaa.flux.phases.run_score",
            node_id="n1",
            graph=graph,
            context=ctx,
            evidence_store=evidence
        )
        
        print(f"Context {i}: U = {ctx.unit_quality:.2f}")
        print(f"  Calibrated score: {cert.calibrated_score:.4f}")
        print(f"  @u layer score:   {cert.layer_scores.get(LayerType.UNIT.value, 0):.4f}")
        print()
    
    print("Notice how calibrated score changes with context!")
    print()


def example_multiple_methods():
    """Compare calibration across different methods"""
    print("=" * 70)
    print("Example 3: Comparing Multiple Methods")
    print("=" * 70)
    print()
    
    context = Context(unit_quality=0.80)
    graph = ComputationGraph(nodes={"n1"})
    evidence = EvidenceStore()
    
    methods = [
        "src.saaaaaa.flux.phases.run_score",
        "src.saaaaaa.flux.phases.run_aggregate",
        "src.saaaaaa.flux.phases.run_normalize"
    ]
    
    print("Calibration scores for different methods:")
    print()
    
    results = []
    for method_id in methods:
        cert = calibrate(
            method_id=method_id,
            node_id="n1",
            graph=graph,
            context=context,
            evidence_store=evidence
        )
        
        results.append({
            "method": method_id.split(".")[-1],
            "calibrated": cert.calibrated_score,
            "intrinsic": cert.intrinsic_score
        })
    
    # Display as table
    print(f"{'Method':<20s} {'Intrinsic':>10s} {'Calibrated':>10s}")
    print("-" * 42)
    for r in results:
        print(f"{r['method']:<20s} {r['intrinsic']:>10.4f} {r['calibrated']:>10.4f}")
    print()


def example_certificate_details():
    """Explore certificate details"""
    print("=" * 70)
    print("Example 4: Certificate Details")
    print("=" * 70)
    print()
    
    context = Context()
    graph = ComputationGraph(nodes={"n1"})
    evidence = EvidenceStore()
    
    cert = calibrate(
        method_id="src.saaaaaa.flux.phases.run_score",
        node_id="n1",
        graph=graph,
        context=context,
        evidence_store=evidence
    )
    
    print("Certificate Fields:")
    print(f"  Instance ID:  {cert.instance_id}")
    print(f"  Timestamp:    {cert.timestamp}")
    print(f"  Config Hash:  {cert.config_hash[:20]}...")
    print(f"  Graph Hash:   {cert.graph_hash[:20]}...")
    print()
    
    print("Fusion Formula (symbolic):")
    print(f"  {cert.fusion_formula['symbolic']}")
    print()
    
    print("Linear Terms:")
    for term in cert.fusion_formula['linear_terms'][:3]:
        print(f"  {term['layer']:12s}: {term['weight']:.3f} × {term['score']:.4f} = {term['contribution']:.4f}")
    print(f"  ... ({len(cert.fusion_formula['linear_terms'])} total)")
    print()
    
    if cert.fusion_formula['interaction_terms']:
        print("Interaction Terms:")
        for term in cert.fusion_formula['interaction_terms']:
            print(f"  {term['pair']:20s}: {term['weight']:.3f} × min({term['layer1_score']:.4f}, {term['layer2_score']:.4f}) = {term['contribution']:.4f}")
        print()


def example_validation():
    """Demonstrate validation"""
    print("=" * 70)
    print("Example 5: Validation")
    print("=" * 70)
    print()
    
    from calibration import validate_config_files, validate_certificate
    
    # Validate configs
    print("Validating configuration files...")
    is_valid, errors = validate_config_files()
    
    if is_valid:
        print("  ✅ All configs valid")
    else:
        print("  ❌ Validation errors:")
        for error in errors:
            print(f"    - {error}")
    print()
    
    # Validate certificate
    context = Context()
    graph = ComputationGraph(nodes={"n1"})
    evidence = EvidenceStore()
    
    cert = calibrate(
        method_id="src.saaaaaa.flux.phases.run_score",
        node_id="n1",
        graph=graph,
        context=context,
        evidence_store=evidence
    )
    
    print("Validating certificate...")
    is_valid, errors = validate_certificate(cert)
    
    if is_valid:
        print("  ✅ Certificate valid")
        print(f"  All scores in [0,1]")
        print(f"  Calibrated score: {cert.calibrated_score:.4f}")
    else:
        print("  ❌ Certificate invalid:")
        for error in errors:
            print(f"    - {error}")
    print()


def main():
    """Run all examples"""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 10 + "Three-Pillar Calibration System - Examples" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    try:
        example_basic_calibration()
        example_context_sensitivity()
        example_multiple_methods()
        example_certificate_details()
        example_validation()
        
        print("=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        print()
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
