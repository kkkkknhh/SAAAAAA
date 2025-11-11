"""
Tests for Executor Calibration System

Validates complete calibration per canonic_calibration_methods.md specification.
"""

import pytest
from pathlib import Path

from calibration.executor_calibration import (
    ExecutorCalibrationEngine,
    ExecutorCalibrationContext,
    calibrate_executor
)


class TestExecutorCalibrationEngine:
    """Test calibration engine initialization and basic functionality"""
    
    def test_engine_initialization(self):
        """Test that engine initializes with all configs loaded"""
        engine = ExecutorCalibrationEngine()
        
        assert engine.intrinsic_config is not None
        assert engine.contextual_config is not None
        assert engine.fusion_config is not None
        assert "SCORE_Q" in engine.fusion_config["role_fusion_parameters"]
    
    def test_all_30_executors_present(self):
        """Test that all 30 executors have intrinsic calibration"""
        engine = ExecutorCalibrationEngine()
        methods = engine.intrinsic_config["methods"]
        
        # Check all 30 executors (6 dimensions × 5 questions)
        expected_count = 30
        assert len(methods) == expected_count
        
        # Check specific executors
        assert "D1Q1_Executor" in methods
        assert "D6Q5_Executor" in methods


class TestBaseLayer:
    """Test @b (intrinsic) layer computation"""
    
    def test_base_layer_computation(self):
        """Test base layer computes correctly from config"""
        engine = ExecutorCalibrationEngine()
        
        score = engine._compute_base_layer("D1Q1_Executor")
        
        # Check score is in valid range
        assert 0.0 <= score <= 1.0
        
        # Verify it matches precomputed value (0.4*0.85 + 0.35*0.80 + 0.25*0.75)
        expected = 0.4 * 0.85 + 0.35 * 0.80 + 0.25 * 0.75
        assert abs(score - expected) < 0.001


class TestUnitLayer:
    """Test @u (unit-of-analysis) layer computation"""
    
    def test_unit_layer_sigmoidal_function(self):
        """Test sigmoidal g_QA function"""
        engine = ExecutorCalibrationEngine()
        context = ExecutorCalibrationContext(unit_quality=0.5)
        
        score = engine._compute_unit_layer("D1Q1_Executor", context)
        
        # At U=0.5, g_QA(0.5) = 1 - exp(-5*(0.5-0.5)) = 1 - exp(0) = 1 - 1 = 0
        assert abs(score - 0.0) < 0.01
    
    def test_unit_layer_high_quality(self):
        """Test unit layer with high quality PDT"""
        engine = ExecutorCalibrationEngine()
        context = ExecutorCalibrationContext(unit_quality=0.9)
        
        score = engine._compute_unit_layer("D1Q1_Executor", context)
        
        # At U=0.9, score should be high (closer to 1.0)
        assert score > 0.9
    
    def test_unit_layer_low_quality(self):
        """Test unit layer with low quality PDT"""
        engine = ExecutorCalibrationEngine()
        context = ExecutorCalibrationContext(unit_quality=0.2)
        
        score = engine._compute_unit_layer("D1Q1_Executor", context)
        
        # At U=0.2, score should be low (closer to 0)
        assert score < 0.2
    
    def test_unit_layer_missing_quality_raises_error(self):
        """Test that missing unit_quality raises error"""
        engine = ExecutorCalibrationEngine()
        context = ExecutorCalibrationContext(unit_quality=None)
        
        with pytest.raises(ValueError, match="unit_quality required"):
            engine._compute_unit_layer("D1Q1_Executor", context)


class TestQuestionLayer:
    """Test @q (question compatibility) layer"""
    
    def test_question_layer_primary(self):
        """Test executor gets 1.0 for its primary question"""
        engine = ExecutorCalibrationEngine()
        context = ExecutorCalibrationContext(question_id="Q_D1_01")
        
        score = engine._compute_question_layer("D1Q1_Executor", context)
        
        # D1Q1_Executor is PRIMARY for Q_D1_01
        assert score == 1.0
    
    def test_question_layer_different_question(self):
        """Test executor gets lower score for different question"""
        engine = ExecutorCalibrationEngine()
        context = ExecutorCalibrationContext(question_id="Q_D2_03")
        
        score = engine._compute_question_layer("D1Q1_Executor", context)
        
        # Different question should get compatible score (0.3)
        assert score == 0.3


class TestDimensionLayer:
    """Test @d (dimension compatibility) layer"""
    
    def test_dimension_layer_primary(self):
        """Test executor gets 1.0 for its primary dimension"""
        engine = ExecutorCalibrationEngine()
        context = ExecutorCalibrationContext(dimension_id="DIM01")
        
        score = engine._compute_dimension_layer("D1Q1_Executor", context)
        
        # D1Q1_Executor is PRIMARY for DIM01
        assert score == 1.0
    
    def test_dimension_layer_cross_compatibility(self):
        """Test executor gets matrix score for different dimension"""
        engine = ExecutorCalibrationEngine()
        context = ExecutorCalibrationContext(dimension_id="DIM02")
        
        score = engine._compute_dimension_layer("D1Q1_Executor", context)
        
        # DIM01 -> DIM02 should give 0.7 from matrix
        assert score == 0.7


class TestFullCalibration:
    """Test complete calibration with all layers"""
    
    def test_full_calibration_d1q1(self):
        """Test complete calibration for D1Q1_Executor"""
        context = ExecutorCalibrationContext(
            question_id="Q_D1_01",
            dimension_id="DIM01",
            policy_id="PA01",
            unit_quality=0.85
        )
        
        result = calibrate_executor(
            "D1Q1_Executor",
            question_id=context.question_id,
            dimension_id=context.dimension_id,
            policy_id=context.policy_id,
            unit_quality=context.unit_quality
        )
        
        # Check result structure
        assert result.executor_name == "D1Q1_Executor"
        assert 0.0 <= result.calibrated_score <= 1.0
        assert len(result.layer_scores) == 8
        
        # Check all 8 layers present
        expected_layers = {"@b", "@chain", "@q", "@d", "@p", "@C", "@u", "@m"}
        assert set(result.layer_scores.keys()) == expected_layers
        
        # Check all layer scores in [0,1]
        for layer, score in result.layer_scores.items():
            assert 0.0 <= score <= 1.0, f"Layer {layer} score {score} out of bounds"
        
        # Check contributions
        assert result.linear_contribution > 0
        assert result.interaction_contribution > 0
        assert abs(
            result.calibrated_score - 
            (result.linear_contribution + result.interaction_contribution)
        ) < 0.001
    
    def test_calibration_deterministic(self):
        """Test that calibration is deterministic"""
        context = ExecutorCalibrationContext(
            question_id="Q_D3_02",
            dimension_id="DIM03",
            policy_id="PA05",
            unit_quality=0.75
        )
        
        result1 = calibrate_executor(
            "D3Q2_Executor",
            question_id=context.question_id,
            dimension_id=context.dimension_id,
            policy_id=context.policy_id,
            unit_quality=context.unit_quality
        )
        
        result2 = calibrate_executor(
            "D3Q2_Executor",
            question_id=context.question_id,
            dimension_id=context.dimension_id,
            policy_id=context.policy_id,
            unit_quality=context.unit_quality
        )
        
        # Same inputs should produce identical outputs
        assert result1.calibrated_score == result2.calibrated_score
        assert result1.layer_scores == result2.layer_scores
    
    def test_calibration_all_30_executors(self):
        """Test that all 30 executors can be calibrated"""
        executors = [
            f"D{dim}Q{q}_Executor"
            for dim in range(1, 7)
            for q in range(1, 6)
        ]
        
        assert len(executors) == 30
        
        for executor in executors:
            context = ExecutorCalibrationContext(
                question_id=f"Q_D{executor[1]}_{executor[3:5]}",
                dimension_id=f"DIM0{executor[1]}",
                policy_id="PA01",
                unit_quality=0.8
            )
            
            result = calibrate_executor(
                executor,
                question_id=context.question_id,
                dimension_id=context.dimension_id,
                policy_id=context.policy_id,
                unit_quality=context.unit_quality
            )
            
            # All should produce valid calibrated scores
            assert 0.0 <= result.calibrated_score <= 1.0
            assert len(result.layer_scores) == 8


class TestWeightNormalization:
    """Test fusion operator weight normalization"""
    
    def test_weights_sum_to_one(self):
        """Test that all weights sum to 1.0"""
        engine = ExecutorCalibrationEngine()
        
        linear_sum = sum(engine.linear_weights.values())
        interaction_sum = sum(engine.interaction_weights.values())
        total = linear_sum + interaction_sum
        
        # Should sum to 1.0 (with small floating point tolerance)
        assert abs(total - 1.0) < 0.01


if __name__ == "__main__":
    # Run tests with simple output
    print("Running executor calibration tests...")
    print()
    
    test_classes = [
        TestExecutorCalibrationEngine,
        TestBaseLayer,
        TestUnitLayer,
        TestQuestionLayer,
        TestDimensionLayer,
        TestFullCalibration,
        TestWeightNormalization
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        print(f"{test_class.__name__}:")
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(instance, method_name)
                    method()
                    print(f"  ✓ {method_name}")
                    passed_tests += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
        print()
    
    print("="*60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    if passed_tests == total_tests:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total_tests - passed_tests} tests failed")
