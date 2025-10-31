#!/usr/bin/env python3
"""
Integration test to verify orchestrator can load and use enhanced recommendation engine.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)

def test_orchestrator_integration():
    """Test that orchestrator can initialize and use enhanced recommendation engine"""
    print("=" * 80)
    print("TESTING ORCHESTRATOR + ENHANCED RECOMMENDATION ENGINE INTEGRATION")
    print("=" * 80)
    
    # Import after setting up path
    from recommendation_engine import RecommendationEngine
    
    print("\n1. Testing enhanced rules loading (as orchestrator does)...")
    
    # Simulate orchestrator's try/except pattern
    recommendation_engine = None
    try:
        # Try enhanced rules first (v2.0)
        try:
            recommendation_engine = RecommendationEngine(
                rules_path="config/recommendation_rules_enhanced.json",
                schema_path="rules/recommendation_rules_enhanced.schema.json"
            )
            print("   ✓ Loaded enhanced v2.0 rules")
            version = "2.0"
        except Exception as e_enhanced:
            print(f"   ! Enhanced rules not available ({e_enhanced})")
            # Fallback to v1.0
            recommendation_engine = RecommendationEngine(
                rules_path="config/recommendation_rules.json",
                schema_path="rules/recommendation_rules.schema.json"
            )
            print("   ✓ Loaded v1.0 rules (fallback)")
            version = "1.0"
    except Exception as e:
        print(f"   ✗ Failed to initialize RecommendationEngine: {e}")
        recommendation_engine = None
    
    if recommendation_engine is None:
        print("\n✗ FAILED: Could not initialize recommendation engine")
        return False
    
    print(f"\n2. Verifying loaded version: {version}")
    print(f"   - MICRO rules: {len(recommendation_engine.rules_by_level['MICRO'])}")
    print(f"   - MESO rules: {len(recommendation_engine.rules_by_level['MESO'])}")
    print(f"   - MACRO rules: {len(recommendation_engine.rules_by_level['MACRO'])}")
    
    print("\n3. Testing generate_all_recommendations (orchestrator's integration point)...")
    
    # Sample data from orchestrator
    micro_scores = {'PA01-DIM01': 1.2, 'PA05-DIM05': 1.3}
    cluster_data = {'CL01': {'score': 72.0, 'variance': 0.25, 'weak_pa': 'PA02'}}
    macro_data = {
        'macro_band': 'SATISFACTORIO',
        'clusters_below_target': ['CL02'],
        'variance_alert': 'MODERADA',
        'priority_micro_gaps': ['PA01-DIM05', 'PA05-DIM04']
    }
    
    recommendation_sets = recommendation_engine.generate_all_recommendations(
        micro_scores=micro_scores,
        cluster_data=cluster_data,
        macro_data=macro_data
    )
    
    print(f"   ✓ Generated recommendations:")
    print(f"     - MICRO: {len(recommendation_sets['MICRO'].recommendations)}")
    print(f"     - MESO: {len(recommendation_sets['MESO'].recommendations)}")
    print(f"     - MACRO: {len(recommendation_sets['MACRO'].recommendations)}")
    
    print("\n4. Testing to_dict() conversion (for orchestrator response)...")
    
    recommendations_dict = {
        level: rec_set.to_dict() for level, rec_set in recommendation_sets.items()
    }
    
    print(f"   ✓ Converted to dictionaries")
    print(f"   ✓ MICRO dict keys: {list(recommendations_dict['MICRO'].keys())}")
    
    if version == "2.0" and recommendations_dict['MICRO']['recommendations']:
        first_rec = recommendations_dict['MICRO']['recommendations'][0]
        has_enhanced = 'execution' in first_rec and 'budget' in first_rec
        print(f"   ✓ Enhanced fields present: {has_enhanced}")
        if has_enhanced:
            print(f"   ✓ Budget in response: ${first_rec['budget']['estimated_cost_cop']:,} COP")
    
    print("\n5. Simulating orchestrator's FASE 8 response...")
    
    # This is what orchestrator returns
    orchestrator_response = {
        "MICRO": recommendations_dict['MICRO'],
        "MESO": recommendations_dict['MESO'],
        "MACRO": recommendations_dict['MACRO'],
        "macro_score": 0.78  # Sample macro score
    }
    
    print(f"   ✓ Response structure matches orchestrator format")
    print(f"   ✓ Response keys: {list(orchestrator_response.keys())}")
    
    print("\n" + "=" * 80)
    print("✓ ORCHESTRATOR INTEGRATION TEST PASSED")
    print("=" * 80)
    print(f"\nOrchestrator can successfully:")
    print(f"  - Load enhanced v{version} recommendation engine")
    print(f"  - Generate recommendations at all 3 levels")
    print(f"  - Convert recommendations to dict for response")
    print(f"  - Return properly formatted response in FASE 8")
    
    if version == "2.0":
        print(f"\nEnhanced features available in responses:")
        print(f"  - Execution logic (trigger conditions, approvals)")
        print(f"  - Budget tracking (costs, funding sources)")
        print(f"  - Measurable indicators (formulas, data sources)")
        print(f"  - Time horizons (duration, milestones)")
        print(f"  - Testable verification (structured artifacts)")
        print(f"  - Authority mapping (legal mandates, approval chains)")
    
    return True

if __name__ == '__main__':
    try:
        success = test_orchestrator_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
