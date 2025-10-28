# tests/test_macro_prompts.py
# coding=utf-8
"""
Tests for Macro Prompts Module

Tests all 5 macro-level analysis prompt macros:
1. Coverage & Structural Gap Stressor
2. Inter-Level Contradiction Scan
3. Bayesian Portfolio Composer
4. Roadmap Optimizer
5. Peer Normalization & Confidence Scaling
"""

import unittest
from macro_prompts import (
    CoverageGapStressor,
    ContradictionScanner,
    BayesianPortfolioComposer,
    RoadmapOptimizer,
    PeerNormalizer,
    MacroPromptsOrchestrator,
    CoverageAnalysis,
    ContradictionReport,
    BayesianPortfolio,
    ImplementationRoadmap,
    PeerNormalization
)


class TestCoverageGapStressor(unittest.TestCase):
    """Test Coverage & Structural Gap Stressor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.stressor = CoverageGapStressor(
            critical_dimensions=["D3", "D6"],
            coverage_threshold=0.70
        )
    
    def test_initialization(self):
        """Test stressor initializes correctly"""
        self.assertIsNotNone(self.stressor)
        self.assertEqual(self.stressor.coverage_threshold, 0.70)
        self.assertIn("D3", self.stressor.critical_dimensions)
        self.assertIn("D6", self.stressor.critical_dimensions)
    
    def test_evaluate_with_good_coverage(self):
        """Test evaluation with good coverage"""
        result = self.stressor.evaluate(
            convergence_by_dimension={"D1": 0.8, "D2": 0.75, "D3": 0.85, "D4": 0.7, "D5": 0.65, "D6": 0.9},
            missing_clusters=[],
            dimension_coverage={"D1": 0.9, "D2": 0.85, "D3": 0.8, "D4": 0.75, "D5": 0.7, "D6": 0.85},
            policy_area_coverage={"P1": 0.8, "P2": 0.75, "P3": 0.9},
            baseline_confidence=1.0
        )
        
        self.assertIsInstance(result, CoverageAnalysis)
        self.assertGreater(result.coverage_index, 0.7)
        self.assertGreater(result.degraded_confidence, 0.8)
        self.assertEqual(len(result.critical_dimensions_below_threshold), 0)
    
    def test_evaluate_with_critical_gaps(self):
        """Test evaluation with critical dimension gaps"""
        result = self.stressor.evaluate(
            convergence_by_dimension={"D1": 0.8, "D2": 0.75, "D3": 0.5, "D4": 0.7, "D5": 0.65, "D6": 0.6},
            missing_clusters=["CLUSTER_X"],
            dimension_coverage={"D1": 0.9, "D2": 0.85, "D3": 0.5, "D4": 0.75, "D5": 0.7, "D6": 0.6},
            policy_area_coverage={"P1": 0.8, "P2": 0.75, "P3": 0.9},
            baseline_confidence=1.0
        )
        
        self.assertIsInstance(result, CoverageAnalysis)
        self.assertLess(result.coverage_index, 0.8)
        self.assertLess(result.degraded_confidence, 1.0)
        self.assertGreater(len(result.critical_dimensions_below_threshold), 0)
        self.assertIn("D3", result.critical_dimensions_below_threshold)
        self.assertIn("D6", result.critical_dimensions_below_threshold)
    
    def test_predictive_uplift_calculation(self):
        """Test predictive uplift simulation"""
        result = self.stressor.evaluate(
            convergence_by_dimension={"D1": 0.8, "D2": 0.75, "D3": 0.5, "D4": 0.7, "D5": 0.65, "D6": 0.6},
            missing_clusters=["CLUSTER_A", "CLUSTER_B"],
            dimension_coverage={"D1": 0.9, "D2": 0.85, "D3": 0.5, "D4": 0.75, "D5": 0.7, "D6": 0.6},
            policy_area_coverage={"P1": 0.8, "P2": 0.75},
            baseline_confidence=1.0
        )
        
        self.assertIsInstance(result.predictive_uplift, dict)
        self.assertGreater(len(result.predictive_uplift), 0)
        # Should have uplift estimates for missing clusters
        self.assertIn("CLUSTER_A", result.predictive_uplift)
        self.assertIn("CLUSTER_B", result.predictive_uplift)


class TestContradictionScanner(unittest.TestCase):
    """Test Inter-Level Contradiction Scanner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.scanner = ContradictionScanner(
            contradiction_threshold=3,
            posterior_threshold=0.7
        )
    
    def test_initialization(self):
        """Test scanner initializes correctly"""
        self.assertIsNotNone(self.scanner)
        self.assertEqual(self.scanner.k, 3)
        self.assertEqual(self.scanner.theta, 0.7)
    
    def test_scan_with_no_contradictions(self):
        """Test scanning with consistent data"""
        micro_claims = [
            {"dimension": "D1", "score": 0.8, "posterior": 0.85},
            {"dimension": "D1", "score": 0.75, "posterior": 0.80},
            {"dimension": "D2", "score": 0.9, "posterior": 0.90}
        ]
        meso_signals = {"D1": {"score": 0.78}, "D2": {"score": 0.88}}
        macro_narratives = {"D1": {"score": 0.77}, "D2": {"score": 0.89}}
        
        result = self.scanner.scan(micro_claims, meso_signals, macro_narratives)
        
        self.assertIsInstance(result, ContradictionReport)
        self.assertGreaterEqual(result.consistency_score, 0.8)
        self.assertEqual(len(result.contradictions), 0)
    
    def test_scan_with_contradictions(self):
        """Test scanning with contradictory data"""
        # Create contradicting micro claims (low scores) vs high macro narrative
        micro_claims = [
            {"dimension": "D1", "score": 0.2, "posterior": 0.85},
            {"dimension": "D1", "score": 0.15, "posterior": 0.80},
            {"dimension": "D1", "score": 0.25, "posterior": 0.90},
            {"dimension": "D1", "score": 0.18, "posterior": 0.75}
        ]
        meso_signals = {"D1": {"score": 0.3}}
        macro_narratives = {"D1": {"score": 0.8}}  # High macro score contradicts low micro
        
        result = self.scanner.scan(micro_claims, meso_signals, macro_narratives)
        
        self.assertIsInstance(result, ContradictionReport)
        self.assertGreater(len(result.contradictions), 0)
        self.assertGreater(len(result.suggested_actions), 0)
        self.assertLess(result.consistency_score, 0.9)
    
    def test_suggested_actions_generation(self):
        """Test that appropriate actions are suggested"""
        micro_claims = [
            {"dimension": "D1", "score": 0.2, "posterior": 0.85},
            {"dimension": "D1", "score": 0.15, "posterior": 0.80},
            {"dimension": "D1", "score": 0.25, "posterior": 0.90},
            {"dimension": "D1", "score": 0.18, "posterior": 0.75}
        ]
        meso_signals = {"D1": {"score": 0.3}}
        macro_narratives = {"D1": {"score": 0.8}}
        
        result = self.scanner.scan(micro_claims, meso_signals, macro_narratives)
        
        self.assertGreater(len(result.suggested_actions), 0)
        # Check that actions have required fields
        for action in result.suggested_actions:
            self.assertIn("action", action)
            self.assertIn("reason", action)


class TestBayesianPortfolioComposer(unittest.TestCase):
    """Test Bayesian Portfolio Composer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.composer = BayesianPortfolioComposer(default_variance=0.05)
    
    def test_initialization(self):
        """Test composer initializes correctly"""
        self.assertIsNotNone(self.composer)
        self.assertEqual(self.composer.default_variance, 0.05)
    
    def test_compose_basic(self):
        """Test basic portfolio composition"""
        meso_posteriors = {
            "CLUSTER_1": 0.8,
            "CLUSTER_2": 0.75,
            "CLUSTER_3": 0.85
        }
        cluster_weights = {
            "CLUSTER_1": 0.4,
            "CLUSTER_2": 0.3,
            "CLUSTER_3": 0.3
        }
        
        result = self.composer.compose(meso_posteriors, cluster_weights)
        
        self.assertIsInstance(result, BayesianPortfolio)
        self.assertGreater(result.prior_global, 0.0)
        self.assertLess(result.prior_global, 1.0)
        self.assertGreater(result.posterior_global, 0.0)
        self.assertGreater(result.var_global, 0.0)
        self.assertEqual(len(result.confidence_interval), 2)
        self.assertLess(result.confidence_interval[0], result.confidence_interval[1])
    
    def test_compose_with_penalties(self):
        """Test composition with reconciliation penalties"""
        meso_posteriors = {
            "CLUSTER_1": 0.8,
            "CLUSTER_2": 0.75
        }
        cluster_weights = {
            "CLUSTER_1": 0.5,
            "CLUSTER_2": 0.5
        }
        penalties = {
            "coverage": 0.1,
            "dispersion": 0.05,
            "contradictions": 0.08
        }
        
        result = self.composer.compose(meso_posteriors, cluster_weights, penalties)
        
        self.assertIsInstance(result, BayesianPortfolio)
        # Posterior should be lower than prior due to penalties
        self.assertLess(result.posterior_global, result.prior_global)
        self.assertGreater(len(result.penalties_applied), 0)
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval is properly bounded"""
        meso_posteriors = {"CLUSTER_1": 0.5}
        cluster_weights = {"CLUSTER_1": 1.0}
        
        result = self.composer.compose(meso_posteriors, cluster_weights)
        
        lower, upper = result.confidence_interval
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLess(lower, upper)


class TestRoadmapOptimizer(unittest.TestCase):
    """Test Roadmap Optimizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.optimizer = RoadmapOptimizer()
    
    def test_initialization(self):
        """Test optimizer initializes correctly"""
        self.assertIsNotNone(self.optimizer)
    
    def test_optimize_basic(self):
        """Test basic roadmap optimization"""
        critical_gaps = [
            {"id": "GAP_1", "name": "Baseline data"},
            {"id": "GAP_2", "name": "Causal model"},
            {"id": "GAP_3", "name": "Budget tracking"}
        ]
        dependency_graph = {
            "GAP_1": [],
            "GAP_2": ["GAP_1"],
            "GAP_3": ["GAP_1", "GAP_2"]
        }
        effort_estimates = {
            "GAP_1": 2.0,
            "GAP_2": 4.0,
            "GAP_3": 3.0
        }
        impact_scores = {
            "GAP_1": 0.9,
            "GAP_2": 0.8,
            "GAP_3": 0.7
        }
        
        result = self.optimizer.optimize(
            critical_gaps,
            dependency_graph,
            effort_estimates,
            impact_scores
        )
        
        self.assertIsInstance(result, ImplementationRoadmap)
        self.assertEqual(len(result.phases), 3)
        self.assertGreater(result.total_expected_uplift, 0.0)
        self.assertGreater(len(result.critical_path), 0)
    
    def test_dependency_ordering(self):
        """Test that dependencies are respected in phase assignment"""
        critical_gaps = [
            {"id": "GAP_A", "name": "Foundation"},
            {"id": "GAP_B", "name": "Build on A"}
        ]
        dependency_graph = {
            "GAP_A": [],
            "GAP_B": ["GAP_A"]
        }
        effort_estimates = {"GAP_A": 2.0, "GAP_B": 2.0}
        impact_scores = {"GAP_A": 0.8, "GAP_B": 0.9}
        
        result = self.optimizer.optimize(
            critical_gaps,
            dependency_graph,
            effort_estimates,
            impact_scores
        )
        
        # GAP_A should be in an earlier phase than GAP_B
        gap_a_phase = None
        gap_b_phase = None
        
        for i, phase in enumerate(result.phases):
            for action in phase["actions"]:
                if action["id"] == "GAP_A":
                    gap_a_phase = i
                elif action["id"] == "GAP_B":
                    gap_b_phase = i
        
        self.assertIsNotNone(gap_a_phase)
        self.assertIsNotNone(gap_b_phase)
        self.assertLess(gap_a_phase, gap_b_phase)
    
    def test_resource_estimation(self):
        """Test resource requirements estimation"""
        critical_gaps = [{"id": f"GAP_{i}", "name": f"Gap {i}"} for i in range(5)]
        dependency_graph = {f"GAP_{i}": [] for i in range(5)}
        effort_estimates = {f"GAP_{i}": 2.0 for i in range(5)}
        impact_scores = {f"GAP_{i}": 0.7 for i in range(5)}
        
        result = self.optimizer.optimize(
            critical_gaps,
            dependency_graph,
            effort_estimates,
            impact_scores
        )
        
        self.assertIsInstance(result.resource_requirements, dict)
        self.assertGreater(len(result.resource_requirements), 0)
        
        # Check that each phase has resource estimates
        for phase_name in ["0-3m", "3-6m", "6-12m"]:
            if phase_name in result.resource_requirements:
                resources = result.resource_requirements[phase_name]
                self.assertIn("total_effort_months", resources)
                self.assertIn("recommended_team_size", resources)


class TestPeerNormalizer(unittest.TestCase):
    """Test Peer Normalization & Confidence Scaling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.normalizer = PeerNormalizer(
            penalty_threshold=3,
            outlier_z_threshold=2.0
        )
    
    def test_initialization(self):
        """Test normalizer initializes correctly"""
        self.assertIsNotNone(self.normalizer)
        self.assertEqual(self.normalizer.k, 3)
        self.assertEqual(self.normalizer.outlier_z, 2.0)
    
    def test_normalize_average_performance(self):
        """Test normalization with average performance"""
        convergence = {
            "P1": 0.75,
            "P2": 0.78,
            "P3": 0.72,
            "P4": 0.76
        }
        peer_distributions = {
            "P1": {"mean": 0.75, "std": 0.1},
            "P2": {"mean": 0.75, "std": 0.1},
            "P3": {"mean": 0.75, "std": 0.1},
            "P4": {"mean": 0.75, "std": 0.1}
        }
        baseline_confidence = 0.85
        
        result = self.normalizer.normalize(convergence, peer_distributions, baseline_confidence)
        
        self.assertIsInstance(result, PeerNormalization)
        self.assertEqual(len(result.z_scores), 4)
        self.assertEqual(result.peer_position, "average")
        # Confidence should be similar to baseline for average performance
        self.assertGreater(result.adjusted_confidence, 0.7)
    
    def test_normalize_above_average_performance(self):
        """Test normalization with above-average performance"""
        convergence = {
            "P1": 0.90,
            "P2": 0.88,
            "P3": 0.92
        }
        peer_distributions = {
            "P1": {"mean": 0.75, "std": 0.1},
            "P2": {"mean": 0.75, "std": 0.1},
            "P3": {"mean": 0.75, "std": 0.1}
        }
        baseline_confidence = 0.85
        
        result = self.normalizer.normalize(convergence, peer_distributions, baseline_confidence)
        
        self.assertEqual(result.peer_position, "above_average")
        # All z-scores should be positive
        for z in result.z_scores.values():
            self.assertGreater(z, 0)
    
    def test_normalize_below_average_performance(self):
        """Test normalization with below-average performance"""
        convergence = {
            "P1": 0.40,
            "P2": 0.35,
            "P3": 0.42,
            "P4": 0.38,
            "P5": 0.36
        }
        peer_distributions = {
            "P1": {"mean": 0.75, "std": 0.1},
            "P2": {"mean": 0.75, "std": 0.1},
            "P3": {"mean": 0.75, "std": 0.1},
            "P4": {"mean": 0.75, "std": 0.1},
            "P5": {"mean": 0.75, "std": 0.1}
        }
        baseline_confidence = 0.85
        
        result = self.normalizer.normalize(convergence, peer_distributions, baseline_confidence)
        
        self.assertEqual(result.peer_position, "below_average")
        # Confidence should be penalized
        self.assertLess(result.adjusted_confidence, baseline_confidence)
        # All z-scores should be negative
        for z in result.z_scores.values():
            self.assertLess(z, 0)
    
    def test_outlier_detection(self):
        """Test outlier identification"""
        convergence = {
            "P1": 0.95,  # High outlier
            "P2": 0.75,
            "P3": 0.25,  # Low outlier
            "P4": 0.76
        }
        peer_distributions = {
            "P1": {"mean": 0.75, "std": 0.08},
            "P2": {"mean": 0.75, "std": 0.08},
            "P3": {"mean": 0.75, "std": 0.08},
            "P4": {"mean": 0.75, "std": 0.08}
        }
        baseline_confidence = 0.85
        
        result = self.normalizer.normalize(convergence, peer_distributions, baseline_confidence)
        
        # Should detect outliers
        self.assertGreater(len(result.outlier_areas), 0)


class TestMacroPromptsOrchestrator(unittest.TestCase):
    """Test MacroPrompts Orchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = MacroPromptsOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initializes all components"""
        self.assertIsNotNone(self.orchestrator.coverage_stressor)
        self.assertIsNotNone(self.orchestrator.contradiction_scanner)
        self.assertIsNotNone(self.orchestrator.portfolio_composer)
        self.assertIsNotNone(self.orchestrator.roadmap_optimizer)
        self.assertIsNotNone(self.orchestrator.peer_normalizer)
    
    def test_execute_all(self):
        """Test executing all 5 macro analyses"""
        macro_data = {
            "convergence_by_dimension": {"D1": 0.8, "D2": 0.75, "D3": 0.85, "D4": 0.7, "D5": 0.65, "D6": 0.9},
            "convergence_by_policy_area": {"P1": 0.8, "P2": 0.75, "P3": 0.85},
            "missing_clusters": [],
            "dimension_coverage": {"D1": 0.9, "D2": 0.85, "D3": 0.8, "D4": 0.75, "D5": 0.7, "D6": 0.85},
            "policy_area_coverage": {"P1": 0.8, "P2": 0.75, "P3": 0.85},
            "micro_claims": [
                {"dimension": "D1", "score": 0.8, "posterior": 0.85},
                {"dimension": "D2", "score": 0.75, "posterior": 0.80}
            ],
            "meso_summary_signals": {"D1": {"score": 0.78}, "D2": {"score": 0.74}},
            "macro_narratives": {"D1": {"score": 0.79}, "D2": {"score": 0.76}},
            "meso_posteriors": {"CLUSTER_1": 0.8, "CLUSTER_2": 0.75},
            "cluster_weights": {"CLUSTER_1": 0.6, "CLUSTER_2": 0.4},
            "critical_gaps": [
                {"id": "GAP_1", "name": "Gap 1"},
                {"id": "GAP_2", "name": "Gap 2"}
            ],
            "dependency_graph": {"GAP_1": [], "GAP_2": ["GAP_1"]},
            "effort_estimates": {"GAP_1": 2.0, "GAP_2": 3.0},
            "impact_scores": {"GAP_1": 0.9, "GAP_2": 0.8},
            "peer_distributions": {
                "P1": {"mean": 0.75, "std": 0.1},
                "P2": {"mean": 0.75, "std": 0.1},
                "P3": {"mean": 0.75, "std": 0.1}
            },
            "baseline_confidence": 0.9
        }
        
        results = self.orchestrator.execute_all(macro_data)
        
        # Should have results from all 5 analyses
        self.assertIn("coverage_analysis", results)
        self.assertIn("contradiction_report", results)
        self.assertIn("bayesian_portfolio", results)
        self.assertIn("implementation_roadmap", results)
        self.assertIn("peer_normalization", results)
        
        # Check basic structure of each result
        self.assertIn("coverage_index", results["coverage_analysis"])
        self.assertIn("contradictions", results["contradiction_report"])
        self.assertIn("prior_global", results["bayesian_portfolio"])
        self.assertIn("phases", results["implementation_roadmap"])
        self.assertIn("z_scores", results["peer_normalization"])


if __name__ == '__main__':
    unittest.main()
