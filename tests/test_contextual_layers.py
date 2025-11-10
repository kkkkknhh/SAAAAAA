"""
Tests for Contextual Calibration Layer Functions

These tests validate the contextual layer computations according to
the SUPERPROMPT specification.
"""

import json
from pathlib import Path
from calibration.layer_computers import (
    compute_question_layer,
    compute_dimension_layer,
    compute_policy_layer,
    compute_unit_layer,
    compute_chain_layer,
    compute_interplay_layer,
    compute_meta_layer_contextual
)
from calibration.data_structures import MethodRole, ComputationGraph


# Load configs
config_dir = Path(__file__).parent.parent / "config"
with open(config_dir / "contextual_parametrization.json") as f:
    contextual_config = json.load(f)

data_dir = Path(__file__).parent.parent / "data"
with open(data_dir / "questionnaire_monolith.json") as f:
    monolith = json.load(f)


class TestQuestionLayer:
    """Test @q layer computation"""
    
    def test_question_layer_no_question(self):
        """Test that None question returns 0.0"""
        score = compute_question_layer(
            method_id="test_method",
            question_id=None,
            monolith=monolith,
            contextual_config=contextual_config
        )
        assert score == 0.0
    
    def test_question_layer_empty_question(self):
        """Test that empty question returns 0.0"""
        score = compute_question_layer(
            method_id="test_method",
            question_id="",
            monolith=monolith,
            contextual_config=contextual_config
        )
        assert score == 0.0
    
    def test_question_layer_fallback_weight(self):
        """Test that unlisted method gets fallback weight"""
        # Use old format since monolith uses it
        score = compute_question_layer(
            method_id="unknown_method_xyz",
            question_id="Q001",
            monolith=monolith,
            contextual_config=contextual_config
        )
        # Should get fallback weight from layer_question since @q not used
        assert 0.0 <= score <= 0.2  # fallback or undeclared


class TestDimensionLayer:
    """Test @d layer computation"""
    
    def test_dimension_layer_no_dimension(self):
        """Test that None dimension returns 0.0"""
        score = compute_dimension_layer(
            method_id="test_method",
            dimension_id=None,
            contextual_config=contextual_config,
            method_dimensions=["DIM01"]
        )
        assert score == 0.0
    
    def test_dimension_layer_no_method_dimensions(self):
        """Test that method with no dimensions gets penalty"""
        score = compute_dimension_layer(
            method_id="test_method",
            dimension_id="DIM01",
            contextual_config=contextual_config,
            method_dimensions=None
        )
        # Should get penalty (0.1) for not declaring any dimensions
        assert score == 0.1
    
    def test_dimension_layer_exact_match(self):
        """Test exact dimension match gives 1.0"""
        score = compute_dimension_layer(
            method_id="test_method",
            dimension_id="DIM01",
            contextual_config=contextual_config,
            method_dimensions=["DIM01"]
        )
        # Exact match should give 1.0 from matrix
        assert score == 1.0
    
    def test_dimension_layer_cross_compatibility(self):
        """Test cross-dimension compatibility uses matrix"""
        score = compute_dimension_layer(
            method_id="test_method",
            dimension_id="DIM01",
            contextual_config=contextual_config,
            method_dimensions=["DIM02"]
        )
        # DIM02 -> DIM01 should give 0.7 from matrix
        assert score == 0.7


class TestPolicyLayer:
    """Test @p layer computation"""
    
    def test_policy_layer_no_policy(self):
        """Test that None policy returns 0.0"""
        score = compute_policy_layer(
            method_id="test_method",
            policy_id=None,
            contextual_config=contextual_config,
            method_policies=["PA01"]
        )
        assert score == 0.0
    
    def test_policy_layer_no_method_policies(self):
        """Test that method with no policies gets penalty"""
        score = compute_policy_layer(
            method_id="test_method",
            policy_id="PA01",
            contextual_config=contextual_config,
            method_policies=None
        )
        # Should get penalty (0.1) for not declaring any policies
        assert score == 0.1
    
    def test_policy_layer_exact_match(self):
        """Test exact policy match gives 1.0"""
        score = compute_policy_layer(
            method_id="test_method",
            policy_id="PA01",
            contextual_config=contextual_config,
            method_policies=["PA01"]
        )
        # Exact match should give 1.0 from matrix
        assert score == 1.0
    
    def test_policy_layer_cross_compatibility(self):
        """Test cross-policy compatibility uses matrix"""
        score = compute_policy_layer(
            method_id="test_method",
            policy_id="PA01",
            contextual_config=contextual_config,
            method_policies=["PA02"]
        )
        # PA02 -> PA01 should give 0.7 from matrix
        assert score == 0.7


class TestUnitLayer:
    """Test @u layer computation"""
    
    def test_unit_layer_flat_type(self):
        """Test flat type returns constant value"""
        score = compute_unit_layer(
            method_id="test_method",
            role=MethodRole.AGGREGATE,  # Uses flat type
            unit_quality=0.5,
            contextual_config=contextual_config
        )
        assert score == 1.0
    
    def test_unit_layer_piecewise_linear(self):
        """Test piecewise linear interpolation"""
        # Test at 0.0 - should be 0.0
        score = compute_unit_layer(
            method_id="test_method",
            role=MethodRole.STRUCTURE,  # Uses piecewise_linear
            unit_quality=0.0,
            contextual_config=contextual_config
        )
        assert score == 0.0
        
        # Test at 0.3 - should be 0.0 (boundary)
        score = compute_unit_layer(
            method_id="test_method",
            role=MethodRole.STRUCTURE,
            unit_quality=0.3,
            contextual_config=contextual_config
        )
        assert score == 0.0
        
        # Test at 0.8 - should be 0.8
        score = compute_unit_layer(
            method_id="test_method",
            role=MethodRole.STRUCTURE,
            unit_quality=0.8,
            contextual_config=contextual_config
        )
        assert score == 0.8
        
        # Test at 1.0 - should be 1.0
        score = compute_unit_layer(
            method_id="test_method",
            role=MethodRole.STRUCTURE,
            unit_quality=1.0,
            contextual_config=contextual_config
        )
        assert score == 1.0
    
    def test_unit_layer_missing_for_sensitive_role(self):
        """Test that missing unit_quality for sensitive role raises error"""
        try:
            score = compute_unit_layer(
                method_id="test_method",
                role=MethodRole.STRUCTURE,
                unit_quality=None,
                contextual_config=contextual_config
            )
            # Should raise ValueError
            assert False, "Expected ValueError for missing unit_quality"
        except ValueError as e:
            assert "Missing unit_quality" in str(e)


class TestChainLayer:
    """Test @chain layer computation"""
    
    def test_chain_layer_ok(self):
        """Test chain layer for valid graph"""
        graph = ComputationGraph()
        graph.nodes.add("node1")
        
        score = compute_chain_layer(
            node_id="node1",
            graph=graph,
            contextual_config=contextual_config
        )
        # Should get ok_score (1.0)
        assert score == 1.0
    
    def test_chain_layer_missing_required_input(self):
        """Test chain layer for missing required input"""
        graph = ComputationGraph()
        graph.nodes.add("node1")
        graph.node_signatures["node1"] = {
            "required_inputs": ["input1"]
        }
        # No incoming edges, so required input is missing
        
        score = compute_chain_layer(
            node_id="node1",
            graph=graph,
            contextual_config=contextual_config
        )
        # Should get missing_required_input_score (0.0)
        assert score == 0.0


class TestInterplayLayer:
    """Test @C layer computation"""
    
    def test_interplay_layer_no_interplay(self):
        """Test interplay layer when not in interplay"""
        score = compute_interplay_layer(
            interplay=None,
            contextual_config=contextual_config
        )
        # Should get ok_score (1.0) when not in interplay
        assert score == 1.0
    
    def test_interplay_layer_with_fusion_rule(self):
        """Test interplay layer with valid fusion rule"""
        interplay = {
            "fusion_rule": "weighted_average",
            "participants": ["method1", "method2"]
        }
        score = compute_interplay_layer(
            interplay=interplay,
            contextual_config=contextual_config
        )
        # Should get ok_score (1.0) with valid fusion rule
        assert score == 1.0
    
    def test_interplay_layer_no_fusion_rule(self):
        """Test interplay layer without fusion rule"""
        interplay = {
            "participants": ["method1", "method2"]
        }
        score = compute_interplay_layer(
            interplay=interplay,
            contextual_config=contextual_config
        )
        # Should get no_fusion_rule_score (0.0)
        assert score == 0.0


class TestMetaLayerContextual:
    """Test @m layer contextual part"""
    
    def test_meta_contextual_no_certificate_required(self):
        """Test when certificate not required"""
        # Modify config temporarily
        config = contextual_config.copy()
        config["@m"] = {"runtime": {"requires_certificate": False}}
        
        score = compute_meta_layer_contextual(
            certificate_present=False,
            certificate_complete=False,
            contextual_config=config
        )
        assert score == 1.0
    
    def test_meta_contextual_complete_certificate(self):
        """Test with complete certificate"""
        score = compute_meta_layer_contextual(
            certificate_present=True,
            certificate_complete=True,
            contextual_config=contextual_config
        )
        # Should get full_certificate_score (1.0)
        assert score == 1.0
    
    def test_meta_contextual_incomplete_certificate(self):
        """Test with incomplete certificate"""
        score = compute_meta_layer_contextual(
            certificate_present=True,
            certificate_complete=False,
            contextual_config=contextual_config
        )
        # Should get incomplete_certificate_penalty (0.4)
        assert score == 0.4


class TestBoundedness:
    """Test that all layer scores stay in [0,1]"""
    
    def test_all_scores_bounded(self):
        """Test that all layer functions return scores in [0,1]"""
        # Test various inputs - just check boundedness, not exact values
        test_cases = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for unit_quality in test_cases:
            score = compute_unit_layer(
                method_id="test",
                role=MethodRole.STRUCTURE,
                unit_quality=unit_quality,
                contextual_config=contextual_config
            )
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for U={unit_quality}"


if __name__ == "__main__":
    print("Running contextual layer tests...")
    
    # Run tests manually
    test_classes = [
        TestQuestionLayer,
        TestDimensionLayer,
        TestPolicyLayer,
        TestUnitLayer,
        TestChainLayer,
        TestInterplayLayer,
        TestMetaLayerContextual,
        TestBoundedness
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total_tests - passed_tests} tests failed")
