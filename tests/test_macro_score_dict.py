"""Test MacroScoreDict typed container in core.py."""
import pytest

from saaaaaa.core.orchestrator.core import MacroScoreDict
from saaaaaa.processing.aggregation import MacroScore, ClusterScore


def test_macro_score_dict_structure():
    """Test that MacroScoreDict has the expected structure."""
    # Create a sample MacroScore and ClusterScore
    macro_score = MacroScore(
        score=0.75,
        confidence=0.9,
        validation_passed=True,
        validation_details={}
    )
    
    cluster_scores = [
        ClusterScore(
            cluster_id="C1",
            score=0.8,
            confidence=0.85,
            validation_passed=True,
            validation_details={}
        )
    ]
    
    # Create MacroScoreDict
    result: MacroScoreDict = {
        "macro_score": macro_score,
        "macro_score_normalized": 0.75,
        "cluster_scores": cluster_scores
    }
    
    # Check types
    assert isinstance(result["macro_score"], MacroScore)
    assert isinstance(result["macro_score_normalized"], float)
    assert isinstance(result["cluster_scores"], list)
    assert all(isinstance(cs, ClusterScore) for cs in result["cluster_scores"])


def test_macro_score_dict_all_keys_present():
    """Test that MacroScoreDict has all required keys."""
    macro_score = MacroScore(
        score=0.65,
        confidence=0.8,
        validation_passed=True,
        validation_details={}
    )
    
    result: MacroScoreDict = {
        "macro_score": macro_score,
        "macro_score_normalized": 0.65,
        "cluster_scores": []
    }
    
    # Check that all keys are present
    assert "macro_score" in result
    assert "macro_score_normalized" in result
    assert "cluster_scores" in result


def test_macro_score_normalized_is_float():
    """Test that macro_score_normalized is always a float."""
    macro_score = MacroScore(
        score=0.5,
        confidence=0.7,
        validation_passed=True,
        validation_details={}
    )
    
    # Test with float conversion
    result: MacroScoreDict = {
        "macro_score": macro_score,
        "macro_score_normalized": float(macro_score.score),
        "cluster_scores": []
    }
    
    assert isinstance(result["macro_score_normalized"], float)
    assert result["macro_score_normalized"] == 0.5
