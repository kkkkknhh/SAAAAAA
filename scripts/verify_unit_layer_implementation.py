"""
Verify Unit Layer is actually implemented (not a stub).

This script MUST pass before proceeding to executor integration.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from saaaaaa.core.calibration import UnitLayerEvaluator, UnitLayerConfig
from saaaaaa.core.calibration.pdt_structure import PDTStructure


def test_unit_layer_not_stub():
    """Verify Unit Layer doesn't return hardcoded values."""
    
    # Create two different PDTs
    pdt1 = PDTStructure(
        full_text="test1",
        total_tokens=100,
        blocks_found={
            "Diagnóstico": {"tokens": 200, "numbers_count": 10}
        },
        sections_found={
            "Diagnóstico": {
                "token_count": 200,
                "keyword_matches": 3,
                "number_count": 10,
                "sources_found": 2
            }
        }
    )
    
    pdt2 = PDTStructure(
        full_text="test2",
        total_tokens=50,
        blocks_found={},
        sections_found={}
    )
    
    evaluator = UnitLayerEvaluator(UnitLayerConfig())
    
    score1 = evaluator.evaluate(pdt1)
    score2 = evaluator.evaluate(pdt2)
    
    # Scores MUST be different for different PDTs
    if score1.score == score2.score:
        print(f"❌ FAIL: Unit Layer returns same score for different PDTs")
        print(f"   Score 1: {score1.score}")
        print(f"   Score 2: {score2.score}")
        print(f"   This indicates a STUB implementation!")
        return False
    
    # Score MUST NOT be exactly 0.75 (old stub value)
    if score1.score == 0.75:
        print(f"❌ FAIL: Unit Layer returns hardcoded 0.75")
        print(f"   This is the old stub value!")
        return False
    
    # Metadata MUST NOT have "stub": True
    if score1.metadata.get("stub"):
        print(f"❌ FAIL: Unit Layer metadata still shows stub=True")
        return False
    
    print(f"✅ PASS: Unit Layer is data-driven")
    print(f"   Score 1: {score1.score:.3f} (components: {score1.components})")
    print(f"   Score 2: {score2.score:.3f}")
    return True


if __name__ == "__main__":
    success = test_unit_layer_not_stub()
    sys.exit(0 if success else 1)
