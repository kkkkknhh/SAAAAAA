"""
Tests for the Aggregation Module.

These tests verify the hierarchical aggregation of scores across all levels:
- Dimension aggregation (60 dimensions)
- Area aggregation (10 areas)
- Cluster aggregation (4 MESO clusters)
- Macro evaluation (1 holistic evaluation)
"""

import json
import unittest

from orchestrator.aggregation import (
    DimensionAggregator,
    AreaPolicyAggregator,
    ClusterAggregator,
    MacroAggregator,
    ScoredResult,
    DimensionScore,
    AreaScore,
    ClusterScore,
    MacroScore,
    WeightValidationError,
    CoverageError,
    HermeticityValidationError,
)


class TestDimensionAggregator(unittest.TestCase):
    """Test dimension aggregation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create minimal monolith for testing
        self.monolith = {
            "blocks": {
                "scoring": {
                    "micro_levels": []
                },
                "niveles_abstraccion": {
                    "dimensions": [
                        {"dimension_id": "DIM01"},
                        {"dimension_id": "DIM02"},
                        {"dimension_id": "DIM03"},
                        {"dimension_id": "DIM04"},
                        {"dimension_id": "DIM05"},
                        {"dimension_id": "DIM06"},
                    ],
                    "policy_areas": [
                        {
                            "policy_area_id": "PA01",
                            "i18n": {"keys": {"label_es": "Área 1"}}
                        }
                    ],
                    "clusters": []
                }
            }
        }
        
        self.aggregator = DimensionAggregator(self.monolith, abort_on_insufficient=False)
    
    def test_validate_weights_success(self):
        """Test weight validation with valid weights."""
        weights = [0.2, 0.2, 0.2, 0.2, 0.2]
        valid, msg = self.aggregator.validate_weights(weights)
        self.assertTrue(valid)
    
    def test_validate_weights_failure(self):
        """Test weight validation with invalid weights."""
        weights = [0.3, 0.3, 0.3, 0.3, 0.3]  # Sum > 1
        valid, msg = self.aggregator.validate_weights(weights)
        self.assertFalse(valid)
    
    def test_validate_coverage_success(self):
        """Test coverage validation with sufficient results."""
        results = [
            ScoredResult(i, f"Q{i}", "PA01", "DIM01", 2.5, "BUENO", {}, {})
            for i in range(1, 6)
        ]
        valid, msg = self.aggregator.validate_coverage(results, expected_count=5)
        self.assertTrue(valid)
    
    def test_validate_coverage_failure(self):
        """Test coverage validation with insufficient results."""
        results = [
            ScoredResult(i, f"Q{i}", "PA01", "DIM01", 2.5, "BUENO", {}, {})
            for i in range(1, 4)  # Only 3 results
        ]
        valid, msg = self.aggregator.validate_coverage(results, expected_count=5)
        self.assertFalse(valid)
    
    def test_calculate_weighted_average_equal_weights(self):
        """Test weighted average calculation with equal weights."""
        scores = [1.0, 2.0, 3.0]
        avg = self.aggregator.calculate_weighted_average(scores)
        self.assertAlmostEqual(avg, 2.0, places=4)
    
    def test_calculate_weighted_average_custom_weights(self):
        """Test weighted average calculation with custom weights."""
        scores = [1.0, 2.0, 3.0]
        weights = [0.5, 0.3, 0.2]
        avg = self.aggregator.calculate_weighted_average(scores, weights)
        # 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 0.5 + 0.6 + 0.6 = 1.7
        self.assertAlmostEqual(avg, 1.7, places=4)
    
    def test_apply_rubric_thresholds_excelente(self):
        """Test rubric thresholds for EXCELENTE quality."""
        score = 2.7  # 2.7/3.0 = 0.9 >= 0.85
        quality = self.aggregator.apply_rubric_thresholds(score)
        self.assertEqual(quality, "EXCELENTE")
    
    def test_apply_rubric_thresholds_bueno(self):
        """Test rubric thresholds for BUENO quality."""
        score = 2.25  # 2.25/3.0 = 0.75 >= 0.70
        quality = self.aggregator.apply_rubric_thresholds(score)
        self.assertEqual(quality, "BUENO")
    
    def test_apply_rubric_thresholds_aceptable(self):
        """Test rubric thresholds for ACEPTABLE quality."""
        score = 1.8  # 1.8/3.0 = 0.6 >= 0.55
        quality = self.aggregator.apply_rubric_thresholds(score)
        self.assertEqual(quality, "ACEPTABLE")
    
    def test_apply_rubric_thresholds_insuficiente(self):
        """Test rubric thresholds for INSUFICIENTE quality."""
        score = 1.2  # 1.2/3.0 = 0.4 < 0.55
        quality = self.aggregator.apply_rubric_thresholds(score)
        self.assertEqual(quality, "INSUFICIENTE")
    
    def test_aggregate_dimension_success(self):
        """Test successful dimension aggregation."""
        scored_results = [
            ScoredResult(i, f"Q{i}", "PA01", "DIM01", 2.1, "BUENO", {}, {})
            for i in range(1, 6)
        ]
        
        dim_score = self.aggregator.aggregate_dimension(
            "DIM01", "PA01", scored_results
        )
        
        self.assertEqual(dim_score.dimension_id, "DIM01")
        self.assertEqual(dim_score.area_id, "PA01")
        self.assertAlmostEqual(dim_score.score, 2.1, places=4)
        self.assertEqual(dim_score.quality_level, "BUENO")
        self.assertTrue(dim_score.validation_passed)
    
    def test_aggregate_dimension_no_results(self):
        """Test dimension aggregation with no results."""
        dim_score = self.aggregator.aggregate_dimension(
            "DIM01", "PA01", []
        )
        
        self.assertEqual(dim_score.dimension_id, "DIM01")
        self.assertEqual(dim_score.area_id, "PA01")
        self.assertEqual(dim_score.score, 0.0)
        self.assertEqual(dim_score.quality_level, "INSUFICIENTE")
        self.assertFalse(dim_score.validation_passed)


class TestAreaPolicyAggregator(unittest.TestCase):
    """Test area policy aggregation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monolith = {
            "blocks": {
                "scoring": {
                    "micro_levels": []
                },
                "niveles_abstraccion": {
                    "dimensions": [
                        {"dimension_id": f"DIM0{i}"} for i in range(1, 7)
                    ],
                    "policy_areas": [
                        {
                            "policy_area_id": "PA01",
                            "i18n": {"keys": {"label_es": "Área de Política 1"}}
                        },
                        {
                            "policy_area_id": "PA02",
                            "i18n": {"keys": {"label_es": "Área de Política 2"}}
                        }
                    ],
                    "clusters": []
                }
            }
        }
        
        self.aggregator = AreaPolicyAggregator(self.monolith, abort_on_insufficient=False)
    
    def test_validate_hermeticity_success(self):
        """Test hermeticity validation with correct dimensions."""
        dimension_scores = [
            DimensionScore(f"DIM0{i}", "PA01", 2.0, "BUENO", [])
            for i in range(1, 7)
        ]
        
        valid, msg = self.aggregator.validate_hermeticity(dimension_scores, "PA01")
        self.assertTrue(valid)
    
    def test_validate_hermeticity_missing_dimensions(self):
        """Test hermeticity validation with missing dimensions."""
        dimension_scores = [
            DimensionScore(f"DIM0{i}", "PA01", 2.0, "BUENO", [])
            for i in range(1, 5)  # Only 4 dimensions
        ]
        
        valid, msg = self.aggregator.validate_hermeticity(dimension_scores, "PA01")
        self.assertFalse(valid)
    
    def test_validate_hermeticity_duplicate_dimensions(self):
        """Test hermeticity validation with duplicate dimensions."""
        dimension_scores = [
            DimensionScore("DIM01", "PA01", 2.0, "BUENO", []),
            DimensionScore("DIM01", "PA01", 2.5, "BUENO", []),  # Duplicate
            DimensionScore("DIM02", "PA01", 2.0, "BUENO", []),
            DimensionScore("DIM03", "PA01", 2.0, "BUENO", []),
            DimensionScore("DIM04", "PA01", 2.0, "BUENO", []),
            DimensionScore("DIM05", "PA01", 2.0, "BUENO", []),
        ]
        
        valid, msg = self.aggregator.validate_hermeticity(dimension_scores, "PA01")
        self.assertFalse(valid)
    
    def test_normalize_scores(self):
        """Test score normalization."""
        dimension_scores = [
            DimensionScore("DIM01", "PA01", 0.0, "INSUFICIENTE", []),
            DimensionScore("DIM02", "PA01", 1.5, "ACEPTABLE", []),
            DimensionScore("DIM03", "PA01", 3.0, "EXCELENTE", []),
        ]
        
        normalized = self.aggregator.normalize_scores(dimension_scores)
        self.assertAlmostEqual(normalized[0], 0.0, places=4)
        self.assertAlmostEqual(normalized[1], 0.5, places=4)
        self.assertAlmostEqual(normalized[2], 1.0, places=4)
    
    def test_aggregate_area_success(self):
        """Test successful area aggregation."""
        dimension_scores = [
            DimensionScore(f"DIM0{i}", "PA01", 2.1, "BUENO", [])
            for i in range(1, 7)
        ]
        
        area_score = self.aggregator.aggregate_area("PA01", dimension_scores)
        
        self.assertEqual(area_score.area_id, "PA01")
        self.assertEqual(area_score.area_name, "Área de Política 1")
        self.assertAlmostEqual(area_score.score, 2.1, places=4)
        self.assertEqual(area_score.quality_level, "BUENO")
        self.assertTrue(area_score.validation_passed)
    
    def test_aggregate_area_no_dimensions(self):
        """Test area aggregation with no dimensions."""
        area_score = self.aggregator.aggregate_area("PA01", [])
        
        self.assertEqual(area_score.area_id, "PA01")
        self.assertEqual(area_score.score, 0.0)
        self.assertEqual(area_score.quality_level, "INSUFICIENTE")
        self.assertFalse(area_score.validation_passed)


class TestClusterAggregator(unittest.TestCase):
    """Test cluster aggregation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monolith = {
            "blocks": {
                "scoring": {
                    "micro_levels": []
                },
                "niveles_abstraccion": {
                    "dimensions": [],
                    "policy_areas": [],
                    "clusters": [
                        {
                            "cluster_id": "CLUSTER_1",
                            "i18n": {"keys": {"label_es": "Cluster Seguridad"}},
                            "policy_area_ids": ["PA01", "PA02", "PA03"]
                        }
                    ]
                }
            }
        }
        
        self.aggregator = ClusterAggregator(self.monolith, abort_on_insufficient=False)
    
    def test_validate_cluster_hermeticity_success(self):
        """Test cluster hermeticity validation with all areas."""
        cluster_def = self.monolith["blocks"]["niveles_abstraccion"]["clusters"][0]
        area_scores = [
            AreaScore("PA01", "Área 1", 2.0, "BUENO", []),
            AreaScore("PA02", "Área 2", 2.5, "BUENO", []),
            AreaScore("PA03", "Área 3", 2.2, "BUENO", []),
        ]
        
        valid, msg = self.aggregator.validate_cluster_hermeticity(cluster_def, area_scores)
        self.assertTrue(valid)
    
    def test_validate_cluster_hermeticity_missing_areas(self):
        """Test cluster hermeticity validation with missing areas."""
        cluster_def = self.monolith["blocks"]["niveles_abstraccion"]["clusters"][0]
        area_scores = [
            AreaScore("PA01", "Área 1", 2.0, "BUENO", []),
            AreaScore("PA02", "Área 2", 2.5, "BUENO", []),
            # PA03 is missing
        ]
        
        valid, msg = self.aggregator.validate_cluster_hermeticity(cluster_def, area_scores)
        self.assertFalse(valid)
    
    def test_analyze_coherence_perfect(self):
        """Test coherence analysis with identical scores."""
        area_scores = [
            AreaScore("PA01", "Área 1", 2.0, "BUENO", []),
            AreaScore("PA02", "Área 2", 2.0, "BUENO", []),
            AreaScore("PA03", "Área 3", 2.0, "BUENO", []),
        ]
        
        coherence = self.aggregator.analyze_coherence(area_scores)
        self.assertGreater(coherence, 0.99)  # Should be very close to 1.0
    
    def test_analyze_coherence_varying(self):
        """Test coherence analysis with varying scores."""
        area_scores = [
            AreaScore("PA01", "Área 1", 1.0, "ACEPTABLE", []),
            AreaScore("PA02", "Área 2", 2.0, "BUENO", []),
            AreaScore("PA03", "Área 3", 3.0, "EXCELENTE", []),
        ]
        
        coherence = self.aggregator.analyze_coherence(area_scores)
        self.assertLess(coherence, 1.0)  # Should be less than perfect
        self.assertGreater(coherence, 0.0)  # But still positive
    
    def test_aggregate_cluster_success(self):
        """Test successful cluster aggregation."""
        area_scores = [
            AreaScore("PA01", "Área 1", 2.0, "BUENO", []),
            AreaScore("PA02", "Área 2", 2.5, "BUENO", []),
            AreaScore("PA03", "Área 3", 2.2, "BUENO", []),
        ]
        
        cluster_score = self.aggregator.aggregate_cluster(
            "CLUSTER_1", area_scores
        )
        
        self.assertEqual(cluster_score.cluster_id, "CLUSTER_1")
        self.assertEqual(cluster_score.cluster_name, "Cluster Seguridad")
        self.assertGreater(cluster_score.score, 0.0)
        self.assertTrue(cluster_score.validation_passed)


class TestMacroAggregator(unittest.TestCase):
    """Test macro evaluation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monolith = {
            "blocks": {
                "scoring": {
                    "micro_levels": []
                },
                "niveles_abstraccion": {
                    "dimensions": [],
                    "policy_areas": [],
                    "clusters": []
                }
            }
        }
        
        self.aggregator = MacroAggregator(self.monolith, abort_on_insufficient=False)
    
    def test_calculate_cross_cutting_coherence_perfect(self):
        """Test cross-cutting coherence with identical scores."""
        cluster_scores = [
            ClusterScore("C1", "Cluster 1", [], 2.0, 1.0, []),
            ClusterScore("C2", "Cluster 2", [], 2.0, 1.0, []),
            ClusterScore("C3", "Cluster 3", [], 2.0, 1.0, []),
        ]
        
        coherence = self.aggregator.calculate_cross_cutting_coherence(cluster_scores)
        self.assertGreater(coherence, 0.99)
    
    def test_identify_systemic_gaps(self):
        """Test systemic gap identification."""
        area_scores = [
            AreaScore("PA01", "Área 1", 2.5, "BUENO", []),
            AreaScore("PA02", "Área 2", 1.2, "INSUFICIENTE", []),
            AreaScore("PA03", "Área 3", 2.8, "EXCELENTE", []),
            AreaScore("PA04", "Área 4", 1.0, "INSUFICIENTE", []),
        ]
        
        gaps = self.aggregator.identify_systemic_gaps(area_scores)
        self.assertEqual(len(gaps), 2)
        self.assertIn("Área 2", gaps)
        self.assertIn("Área 4", gaps)
    
    def test_assess_strategic_alignment(self):
        """Test strategic alignment assessment."""
        cluster_scores = [
            ClusterScore("C1", "Cluster 1", [], 2.0, 0.9, []),
            ClusterScore("C2", "Cluster 2", [], 2.2, 0.85, []),
        ]
        dimension_scores = [
            DimensionScore("D1", "PA01", 2.0, "BUENO", [], validation_passed=True),
            DimensionScore("D2", "PA01", 2.5, "BUENO", [], validation_passed=True),
            DimensionScore("D3", "PA01", 1.8, "ACEPTABLE", [], validation_passed=False),
        ]
        
        alignment = self.aggregator.assess_strategic_alignment(
            cluster_scores, dimension_scores
        )
        
        self.assertGreater(alignment, 0.0)
        self.assertLessEqual(alignment, 1.0)
    
    def test_evaluate_macro_success(self):
        """Test successful macro evaluation."""
        cluster_scores = [
            ClusterScore("C1", "Cluster 1", ["PA01"], 2.0, 0.9, []),
            ClusterScore("C2", "Cluster 2", ["PA02"], 2.2, 0.85, []),
        ]
        area_scores = [
            AreaScore("PA01", "Área 1", 2.0, "BUENO", []),
            AreaScore("PA02", "Área 2", 2.2, "BUENO", []),
        ]
        dimension_scores = [
            DimensionScore("D1", "PA01", 2.0, "BUENO", [], validation_passed=True),
        ]
        
        macro_score = self.aggregator.evaluate_macro(
            cluster_scores, area_scores, dimension_scores
        )
        
        self.assertGreater(macro_score.score, 0.0)
        self.assertIsNotNone(macro_score.quality_level)
        self.assertTrue(macro_score.validation_passed)
        self.assertIsInstance(macro_score.systemic_gaps, list)
    
    def test_evaluate_macro_no_clusters(self):
        """Test macro evaluation with no clusters."""
        macro_score = self.aggregator.evaluate_macro([], [], [])
        
        self.assertEqual(macro_score.score, 0.0)
        self.assertEqual(macro_score.quality_level, "INSUFICIENTE")
        self.assertFalse(macro_score.validation_passed)


if __name__ == "__main__":
    unittest.main()
