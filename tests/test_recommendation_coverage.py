"""
Additional comprehensive tests for recommendation engine.

Expands test coverage for behavioral correctness, data integrity,
and edge cases.
"""

import pytest
import json
from pathlib import Path

# Try to import recommendation_engine, skip tests if not available
pytest.importorskip("recommendation_engine", reason="recommendation_engine module not available")

from recommendation_engine import (
    RecommendationEngine,
    Recommendation,
    RecommendationSet,
    load_recommendation_engine,
)


class TestRecommendationEngineDataIntegrity:
    """Test data integrity and input validation."""
    
    def test_empty_micro_scores(self):
        """Test behavior with empty micro scores."""
        # Create minimal rules file for testing
        test_rules = {
            "version": "2.0.0",
            "rules": []
        }
        
        # Create temp files
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        # Create minimal schema
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            # Empty scores should not crash
            result = engine.generate_micro_recommendations({})
            
            assert isinstance(result, RecommendationSet)
            assert result.level == 'MICRO'
            assert len(result.recommendations) == 0
            assert result.rules_matched == 0
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()
    
    def test_malformed_score_keys(self):
        """Test behavior with malformed score keys."""
        test_rules = {
            "version": "2.0.0",
            "rules": [
                {
                    "rule_id": "TEST-001",
                    "level": "MICRO",
                    "when": {
                        "pa_id": "PA01",
                        "dim_id": "DIM01",
                        "score_lt": 2.0
                    },
                    "template": {
                        "problem": "Test problem",
                        "intervention": "Test intervention",
                        "indicator": {"name": "Test", "target": "100", "unit": "%"},
                        "responsible": {"entity": "Test", "role": "Test"},
                        "horizon": {"start": "M1", "end": "M3"},
                        "verification": ["Test"]
                    }
                }
            ]
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            # Malformed keys should be ignored, not crash
            malformed_scores = {
                "INVALID": 1.0,
                "PA01": 1.5,
                "PA01-DIM99": 1.0
            }
            
            result = engine.generate_micro_recommendations(malformed_scores)
            
            assert isinstance(result, RecommendationSet)
            # Should not match because keys don't match expected pattern
            assert result.rules_matched == 0
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()
    
    def test_null_and_none_values(self):
        """Test handling of null/None values in data."""
        test_rules = {
            "version": "2.0.0",
            "rules": []
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            # Test with None context
            result = engine.generate_micro_recommendations({}, context=None)
            assert isinstance(result, RecommendationSet)
            
            # Test MESO with None values
            cluster_data_with_none = {
                'CL01': {'score': None, 'variance': 0.1, 'weak_pa': None}
            }
            result = engine.generate_meso_recommendations(cluster_data_with_none)
            assert isinstance(result, RecommendationSet)
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()
    
    def test_extreme_score_values(self):
        """Test handling of extreme score values."""
        test_rules = {
            "version": "2.0.0",
            "rules": [
                {
                    "rule_id": "TEST-001",
                    "level": "MICRO",
                    "when": {
                        "pa_id": "PA01",
                        "dim_id": "DIM01",
                        "score_lt": 2.0
                    },
                    "template": {
                        "problem": "Test problem",
                        "intervention": "Test intervention",
                        "indicator": {"name": "Test", "target": "100", "unit": "%"},
                        "responsible": {"entity": "Test", "role": "Test"},
                        "horizon": {"start": "M1", "end": "M3"},
                        "verification": ["Test"]
                    }
                }
            ]
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            # Test with extreme values
            extreme_scores = {
                'PA01-DIM01': -1000.0,  # Negative
                'PA02-DIM01': 1000.0,   # Very high
                'PA03-DIM01': 0.0,      # Zero
            }
            
            result = engine.generate_micro_recommendations(extreme_scores)
            assert isinstance(result, RecommendationSet)
            # Rule should match for PA01-DIM01 since -1000 < 2.0
            assert result.rules_matched >= 1
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()


class TestRecommendationEngineBehavioralCorrectness:
    """Test behavioral correctness of recommendation logic."""
    
    def test_score_threshold_boundary(self):
        """Test score threshold boundary conditions."""
        test_rules = {
            "version": "2.0.0",
            "rules": [
                {
                    "rule_id": "BOUNDARY-001",
                    "level": "MICRO",
                    "when": {
                        "pa_id": "PA01",
                        "dim_id": "DIM01",
                        "score_lt": 2.0
                    },
                    "template": {
                        "problem": "Test problem",
                        "intervention": "Test intervention",
                        "indicator": {"name": "Test", "target": "100", "unit": "%"},
                        "responsible": {"entity": "Test", "role": "Test"},
                        "horizon": {"start": "M1", "end": "M3"},
                        "verification": ["Test"]
                    }
                }
            ]
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            # Test exact boundary
            result = engine.generate_micro_recommendations({'PA01-DIM01': 2.0})
            assert result.rules_matched == 0  # Should NOT match (not less than 2.0)
            
            # Test just below boundary
            result = engine.generate_micro_recommendations({'PA01-DIM01': 1.999})
            assert result.rules_matched == 1  # Should match
            
            # Test just above boundary
            result = engine.generate_micro_recommendations({'PA01-DIM01': 2.001})
            assert result.rules_matched == 0  # Should NOT match
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()
    
    def test_meso_score_band_logic(self):
        """Test MESO score band categorization logic."""
        test_rules = {
            "version": "2.0.0",
            "rules": [
                {
                    "rule_id": "MESO-BAJO",
                    "level": "MESO",
                    "when": {
                        "cluster_id": "CL01",
                        "score_band": "BAJO",
                        "variance_level": "BAJA"
                    },
                    "template": {
                        "problem": "Low score",
                        "intervention": "Improve",
                        "indicator": {"name": "Test", "target": "100", "unit": "%"},
                        "responsible": {"entity": "Test", "role": "Test"},
                        "horizon": {"start": "M1", "end": "M3"},
                        "verification": ["Test"]
                    }
                }
            ]
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            # BAJO band: score < 55
            result = engine.generate_meso_recommendations({
                'CL01': {'score': 54.0, 'variance': 0.05}
            })
            assert result.rules_matched == 1
            
            # Boundary: exactly 55 should NOT be BAJO
            result = engine.generate_meso_recommendations({
                'CL01': {'score': 55.0, 'variance': 0.05}
            })
            assert result.rules_matched == 0
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()
    
    def test_template_variable_substitution(self):
        """Test template variable substitution correctness."""
        test_rules = {
            "version": "2.0.0",
            "rules": [
                {
                    "rule_id": "VAR-001",
                    "level": "MICRO",
                    "when": {
                        "pa_id": "PA05",
                        "dim_id": "DIM03",
                        "score_lt": 2.0
                    },
                    "template": {
                        "problem": "Problem in {{PAxx}}-{{DIMxx}}",
                        "intervention": "Intervention for {{PAxx}}",
                        "indicator": {"name": "Test", "target": "100", "unit": "%"},
                        "responsible": {"entity": "Test", "role": "Test"},
                        "horizon": {"start": "M1", "end": "M3"},
                        "verification": ["Test"]
                    }
                }
            ]
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            result = engine.generate_micro_recommendations({'PA05-DIM03': 1.5})
            
            assert result.rules_matched == 1
            rec = result.recommendations[0]
            assert "PA05" in rec.problem
            assert "DIM03" in rec.problem
            assert "PA05" in rec.intervention
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()


class TestRecommendationEngineStressResponse:
    """Test stress response and scaling."""
    
    def test_large_number_of_scores(self):
        """Test with large number of scores."""
        test_rules = {
            "version": "2.0.0",
            "rules": []
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            # Generate 1000 scores
            large_scores = {
                f'PA{i%10+1:02d}-DIM{i%6+1:02d}': float(i % 3)
                for i in range(1000)
            }
            
            result = engine.generate_micro_recommendations(large_scores)
            
            # Should handle without crashing
            assert isinstance(result, RecommendationSet)
            assert result.total_rules_evaluated >= 0
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()
    
    def test_many_rules_evaluation(self):
        """Test evaluation with many rules."""
        # Generate 100 rules
        rules = []
        for i in range(100):
            rules.append({
                "rule_id": f"RULE-{i:03d}",
                "level": "MICRO",
                "when": {
                    "pa_id": f"PA{(i%10)+1:02d}",
                    "dim_id": f"DIM{(i%6)+1:02d}",
                    "score_lt": 2.0
                },
                "template": {
                    "problem": f"Problem {i}",
                    "intervention": f"Intervention {i}",
                    "indicator": {"name": "Test", "target": "100", "unit": "%"},
                    "responsible": {"entity": "Test", "role": "Test"},
                    "horizon": {"start": "M1", "end": "M3"},
                    "verification": ["Test"]
                }
            })
        
        test_rules = {
            "version": "2.0.0",
            "rules": rules
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            # Should evaluate all 100 MICRO rules
            result = engine.generate_micro_recommendations({'PA01-DIM01': 1.0})
            
            assert result.total_rules_evaluated == 100
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()


class TestRecommendationMetadata:
    """Test recommendation metadata and tracking."""
    
    def test_metadata_populated(self):
        """Test that metadata is properly populated."""
        test_rules = {
            "version": "2.0.0",
            "rules": [
                {
                    "rule_id": "META-001",
                    "level": "MICRO",
                    "when": {
                        "pa_id": "PA01",
                        "dim_id": "DIM01",
                        "score_lt": 2.0
                    },
                    "template": {
                        "problem": "Test",
                        "intervention": "Test",
                        "indicator": {"name": "Test", "target": "100", "unit": "%"},
                        "responsible": {"entity": "Test", "role": "Test"},
                        "horizon": {"start": "M1", "end": "M3"},
                        "verification": ["Test"]
                    }
                }
            ]
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_rules, f)
            rules_path = f.name
        
        schema = {"type": "object"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            schema_path = f.name
        
        try:
            engine = RecommendationEngine(rules_path=rules_path, schema_path=schema_path)
            
            result = engine.generate_micro_recommendations({'PA01-DIM01': 1.5})
            
            assert result.rules_matched == 1
            rec = result.recommendations[0]
            
            # Check metadata
            assert 'score_key' in rec.metadata
            assert 'actual_score' in rec.metadata
            assert 'threshold' in rec.metadata
            assert 'gap' in rec.metadata
            
            assert rec.metadata['score_key'] == 'PA01-DIM01'
            assert rec.metadata['actual_score'] == 1.5
            assert rec.metadata['threshold'] == 2.0
            assert rec.metadata['gap'] == pytest.approx(0.5)
        finally:
            Path(rules_path).unlink()
            Path(schema_path).unlink()
