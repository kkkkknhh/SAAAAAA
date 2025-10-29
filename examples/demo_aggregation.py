#!/usr/bin/env python3
"""
Demonstration: Complete Aggregation Pipeline

This script demonstrates the hierarchical aggregation system processing
scores through all levels:
- FASE 4: Dimension aggregation (60 dimensions)
- FASE 5: Policy area aggregation (10 areas)
- FASE 6: Cluster aggregation (4 MESO clusters)
- FASE 7: Macro evaluation (1 holistic evaluation)

Features demonstrated:
- Weight validation
- Coverage validation
- Hermeticity validation
- Comprehensive logging
- Quality rubric application
- Coherence analysis
- Strategic alignment assessment
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging to see the full flow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from orchestrator.aggregation import (
    DimensionAggregator,
    AreaPolicyAggregator,
    ClusterAggregator,
    MacroAggregator,
    ScoredResult,
)


def main():
    print("=" * 80)
    print("DEMONSTRATION: Complete Aggregation Pipeline")
    print("=" * 80)

    # Create a minimal monolith structure
    monolith = {
        "blocks": {
            "scoring": {"micro_levels": []},
            "niveles_abstraccion": {
                "dimensions": [
                    {"dimension_id": f"DIM0{i}"} for i in range(1, 7)
                ],
                "policy_areas": [
                    {
                        "policy_area_id": f"PA0{i}",
                        "i18n": {"keys": {"label_es": f"Área de Política {i}"}}
                    }
                    for i in range(1, 4)
                ],
                "clusters": [
                    {
                        "cluster_id": "CLUSTER_1",
                        "i18n": {"keys": {"label_es": "Cluster Seguridad y Paz"}},
                        "policy_area_ids": ["PA01", "PA02"]
                    },
                    {
                        "cluster_id": "CLUSTER_2",
                        "i18n": {"keys": {"label_es": "Cluster Desarrollo"}},
                        "policy_area_ids": ["PA03"]
                    }
                ]
            }
        }
    }

    print("\n1. Initialize Aggregators")
    print("-" * 80)

    dim_agg = DimensionAggregator(monolith)
    area_agg = AreaPolicyAggregator(monolith)
    cluster_agg = ClusterAggregator(monolith)
    macro_agg = MacroAggregator(monolith)

    print("✓ All aggregators initialized\n")

    print("2. Simulate Micro Question Results")
    print("-" * 80)

    # Create sample scored results for 3 areas × 6 dimensions = 18 dimensions
    # Each dimension has 5 questions
    all_scored_results = []
    for area_idx in range(1, 4):
        for dim_idx in range(1, 7):
            for q_idx in range(1, 6):
                score = 1.5 + (area_idx * 0.3) + (dim_idx * 0.1)
                result = ScoredResult(
                    question_global=len(all_scored_results) + 1,
                    base_slot=f"PA0{area_idx}-DIM0{dim_idx}-Q{q_idx}",
                    policy_area=f"PA0{area_idx}",
                    dimension=f"DIM0{dim_idx}",
                    score=min(3.0, score),  # Cap at 3.0
                    quality_level="BUENO",
                    evidence={},
                    raw_results={}
                )
                all_scored_results.append(result)

    print(f"✓ Created {len(all_scored_results)} micro question results\n")

    print("3. Aggregate Dimensions (FASE 4)")
    print("-" * 80)

    dimension_scores = []
    for area_idx in range(1, 4):
        for dim_idx in range(1, 7):
            dim_score = dim_agg.aggregate_dimension(
                f"DIM0{dim_idx}",
                f"PA0{area_idx}",
                all_scored_results
            )
            dimension_scores.append(dim_score)
            print(f"  DIM0{dim_idx}/PA0{area_idx}: {dim_score.score:.2f} ({dim_score.quality_level})")

    print(f"\n✓ Aggregated {len(dimension_scores)} dimensions\n")

    print("4. Aggregate Policy Areas (FASE 5)")
    print("-" * 80)

    area_scores = []
    for area_idx in range(1, 4):
        area_score = area_agg.aggregate_area(
            f"PA0{area_idx}",
            dimension_scores
        )
        area_scores.append(area_score)
        print(f"  {area_score.area_name}: {area_score.score:.2f} ({area_score.quality_level})")

    print(f"\n✓ Aggregated {len(area_scores)} policy areas\n")

    print("5. Aggregate Clusters (FASE 6)")
    print("-" * 80)

    cluster_scores = []
    for cluster_def in monolith["blocks"]["niveles_abstraccion"]["clusters"]:
        cluster_score = cluster_agg.aggregate_cluster(
            cluster_def["cluster_id"],
            area_scores
        )
        cluster_scores.append(cluster_score)
        print(f"  {cluster_score.cluster_name}: {cluster_score.score:.2f} (coherence: {cluster_score.coherence:.2f})")

    print(f"\n✓ Aggregated {len(cluster_scores)} MESO clusters\n")

    print("6. Macro Evaluation (FASE 7 - Q305)")
    print("-" * 80)

    macro_score = macro_agg.evaluate_macro(
        cluster_scores,
        area_scores,
        dimension_scores
    )

    print(f"  Macro Score: {macro_score.score:.2f}")
    print(f"  Quality Level: {macro_score.quality_level}")
    print(f"  Cross-Cutting Coherence: {macro_score.cross_cutting_coherence:.2f}")
    print(f"  Strategic Alignment: {macro_score.strategic_alignment:.2f}")
    print(f"  Systemic Gaps: {len(macro_score.systemic_gaps)}")

    print("\n✓ Macro evaluation complete\n")

    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE: All aggregation levels working correctly!")
    print("=" * 80)


if __name__ == "__main__":
    main()
