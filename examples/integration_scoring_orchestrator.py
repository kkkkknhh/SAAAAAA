#!/usr/bin/env python3
"""
Integration example: Using scoring module with orchestrator pattern.

Shows how the scoring module integrates with the existing orchestrator architecture.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.scoring import apply_scoring, ScoringModality


def simulate_question_processing():
    """
    Simulate processing multiple questions through the scoring pipeline.
    
    This demonstrates how the orchestrator would use the scoring module
    to score evidence from micro questions.
    """
    print("=" * 70)
    print("INTEGRATION: Scoring Module with Orchestrator Pattern")
    print("=" * 70)
    
    # Simulate processing 6 questions, one for each modality
    questions = [
        {
            "question_global": 1,
            "base_slot": "PA01-DIM01-Q001",
            "policy_area": "PA01",
            "dimension": "DIM01",
            "modality": "TYPE_A",
            "evidence": {
                "elements": ["Gap identified", "Baseline measured", "Target set", "Risk assessed"],
                "confidence": 0.88
            }
        },
        {
            "question_global": 51,
            "base_slot": "PA02-DIM06-Q001",
            "policy_area": "PA02",
            "dimension": "DIM06",
            "modality": "TYPE_B",
            "evidence": {
                "elements": ["Input→Activity", "Activity→Output", "Output→Outcome"],
                "completeness": 0.95
            }
        },
        {
            "question_global": 101,
            "base_slot": "PA03-DIM02-Q001",
            "policy_area": "PA03",
            "dimension": "DIM02",
            "modality": "TYPE_C",
            "evidence": {
                "elements": ["Statement A", "Confirmation A"],
                "coherence_score": 0.92
            }
        },
        {
            "question_global": 151,
            "base_slot": "PA04-DIM01-Q001",
            "policy_area": "PA04",
            "dimension": "DIM01",
            "modality": "TYPE_D",
            "evidence": {
                "elements": ["Pattern 1", "Pattern 2", "Pattern 3"],
                "pattern_matches": 3
            }
        },
        {
            "question_global": 201,
            "base_slot": "PA05-DIM03-Q001",
            "policy_area": "PA05",
            "dimension": "DIM03",
            "modality": "TYPE_E",
            "evidence": {
                "elements": ["Budget item 1", "Budget item 2"],
                "traceability": 0.90
            }
        },
        {
            "question_global": 251,
            "base_slot": "PA06-DIM06-Q001",
            "policy_area": "PA06",
            "dimension": "DIM06",
            "modality": "TYPE_F",
            "evidence": {
                "elements": ["Mechanism 1", "Mechanism 2"],
                "plausibility": 0.85
            }
        }
    ]
    
    print("\nProcessing questions through scoring pipeline...")
    print("-" * 70)
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/6: {question['base_slot']} ({question['modality']})")
        
        try:
            result = apply_scoring(
                question_global=question["question_global"],
                base_slot=question["base_slot"],
                policy_area=question["policy_area"],
                dimension=question["dimension"],
                evidence=question["evidence"],
                modality=question["modality"]
            )
            
            results.append(result)
            
            print(f"  ✓ Score: {result.score:.2f}")
            print(f"  ✓ Normalized: {result.normalized_score:.2f}")
            print(f"  ✓ Quality: {result.quality_level}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    
    # Aggregate results
    total_questions = len(results)
    avg_score = sum(r.normalized_score for r in results) / total_questions if results else 0
    
    quality_counts = {
        "EXCELENTE": sum(1 for r in results if r.quality_level == "EXCELENTE"),
        "BUENO": sum(1 for r in results if r.quality_level == "BUENO"),
        "ACEPTABLE": sum(1 for r in results if r.quality_level == "ACEPTABLE"),
        "INSUFICIENTE": sum(1 for r in results if r.quality_level == "INSUFICIENTE"),
    }
    
    print(f"\nTotal questions processed: {total_questions}")
    print(f"Average normalized score: {avg_score:.2f}")
    print(f"\nQuality distribution:")
    for level, count in quality_counts.items():
        pct = (count / total_questions * 100) if total_questions > 0 else 0
        print(f"  {level:14}: {count:2} ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("Integration test complete!")
    print("=" * 70)
    
    return results


def demonstrate_reproducibility():
    """Demonstrate that scoring is reproducible."""
    print("\n" + "=" * 70)
    print("REPRODUCIBILITY TEST")
    print("=" * 70)
    
    evidence = {
        "elements": [1, 2, 3, 4],
        "confidence": 0.85
    }
    
    print("\nScoring the same evidence 3 times...")
    
    results = []
    for i in range(3):
        result = apply_scoring(
            question_global=1,
            base_slot="PA01-DIM01-Q001",
            policy_area="PA01",
            dimension="DIM01",
            evidence=evidence,
            modality="TYPE_A"
        )
        results.append(result)
        print(f"  Run {i+1}: score={result.score:.2f}, hash={result.evidence_hash[:16]}...")
    
    # Verify all results are identical
    all_same_score = all(r.score == results[0].score for r in results)
    all_same_hash = all(r.evidence_hash == results[0].evidence_hash for r in results)
    all_same_quality = all(r.quality_level == results[0].quality_level for r in results)
    
    print(f"\nAll scores identical: {all_same_score}")
    print(f"All hashes identical: {all_same_hash}")
    print(f"All quality levels identical: {all_same_quality}")
    
    if all_same_score and all_same_hash and all_same_quality:
        print("\n✓ Reproducibility verified!")
    else:
        print("\n✗ Reproducibility check failed!")
    
    print("=" * 70)


def main():
    """Run integration examples."""
    simulate_question_processing()
    demonstrate_reproducibility()


if __name__ == "__main__":
    main()
