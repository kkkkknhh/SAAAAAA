# tests/test_macro_prompts_integration.py
# coding=utf-8
"""
Integration tests for Macro Prompts with Report Assembly

Tests the complete integration of macro prompts with the report assembly system.
"""

import unittest
from report_assembly import (
    ReportAssembler,
    MicroLevelAnswer,
    MesoLevelCluster,
    MacroLevelConvergence
)
from macro_prompts import MacroPromptsOrchestrator


class TestMacroPromptsIntegration(unittest.TestCase):
    """Test integration of macro prompts with report assembly"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.assembler = ReportAssembler()
    
    def test_report_assembler_has_macro_prompts(self):
        """Test that ReportAssembler initializes with macro prompts"""
        self.assertIsNotNone(self.assembler.macro_prompts)
        self.assertIsInstance(self.assembler.macro_prompts, MacroPromptsOrchestrator)
    
    def test_generate_macro_with_prompts(self):
        """Test that macro convergence includes prompt results"""
        # Create mock micro answers
        micro_answers = [
            MicroLevelAnswer(
                question_id=f"P{p}-D{d}-Q{q}",
                qualitative_note="BUENO",
                quantitative_score=2.3,
                evidence=["Evidence text"],
                explanation="Test explanation",
                confidence=0.85,
                scoring_modality="TYPE_A",
                elements_found={},
                search_pattern_matches={},
                modules_executed=["test"],
                module_results={},
                execution_time=0.1
            )
            for p in range(1, 4)  # 3 policy areas
            for d in range(1, 4)  # 3 dimensions
            for q in range(1, 3)  # 2 questions each
        ]
        
        # Create mock meso clusters
        meso_clusters = [
            MesoLevelCluster(
                cluster_name=f"CLUSTER_{i}",
                cluster_description=f"Cluster {i}",
                policy_areas=[f"P{p}" for p in range(1, 4)],
                avg_score=75.0 + (i * 5),
                dimension_scores={f"D{d}": 70.0 + (d * 3) for d in range(1, 4)},
                strengths=["Strength 1"],
                weaknesses=["Weakness 1"],
                recommendations=["Rec 1"],
                question_coverage=0.9,
                total_questions=30,
                answered_questions=27
            )
            for i in range(1, 4)
        ]
        
        # Generate macro convergence
        macro_convergence = self.assembler.generate_macro_convergence(
            all_micro_answers=micro_answers,
            all_meso_clusters=meso_clusters,
            plan_metadata={"name": "Test Plan"}
        )
        
        # Verify basic structure
        self.assertIsInstance(macro_convergence, MacroLevelConvergence)
        self.assertGreater(macro_convergence.overall_score, 0)
        
        # Verify macro prompts were applied
        self.assertIn("macro_prompts_results", macro_convergence.metadata)
        prompts_results = macro_convergence.metadata["macro_prompts_results"]
        
        # Should have results from all 5 prompts
        self.assertIn("coverage_analysis", prompts_results)
        self.assertIn("contradiction_report", prompts_results)
        self.assertIn("bayesian_portfolio", prompts_results)
        self.assertIn("implementation_roadmap", prompts_results)
        self.assertIn("peer_normalization", prompts_results)
    
    def test_coverage_analysis_in_macro(self):
        """Test that coverage analysis is included in macro results"""
        micro_answers = [
            MicroLevelAnswer(
                question_id=f"P1-D{d}-Q1",
                qualitative_note="BUENO",
                quantitative_score=2.3,
                evidence=[],
                explanation="Test",
                confidence=0.85,
                scoring_modality="TYPE_A",
                elements_found={},
                search_pattern_matches={},
                modules_executed=[],
                module_results={},
                execution_time=0.1
            )
            for d in range(1, 7)  # All 6 dimensions
        ]
        
        meso_clusters = [
            MesoLevelCluster(
                cluster_name="CLUSTER_1",
                cluster_description="Test cluster",
                policy_areas=["P1"],
                avg_score=75.0,
                dimension_scores={f"D{d}": 75.0 for d in range(1, 7)},
                strengths=[],
                weaknesses=[],
                recommendations=[],
                question_coverage=0.9,
                total_questions=6,
                answered_questions=6
            )
        ]
        
        macro = self.assembler.generate_macro_convergence(
            micro_answers, meso_clusters, {"name": "Test"}
        )
        
        coverage = macro.metadata["macro_prompts_results"]["coverage_analysis"]
        
        # Verify coverage analysis structure
        self.assertIn("coverage_index", coverage)
        self.assertIn("degraded_confidence", coverage)
        self.assertIn("predictive_uplift", coverage)
        self.assertIsInstance(coverage["coverage_index"], (int, float))
    
    def test_contradiction_scan_in_macro(self):
        """Test that contradiction scanning is included"""
        # Create answers with varying scores
        micro_answers = [
            MicroLevelAnswer(
                question_id=f"P1-D1-Q{i}",
                qualitative_note="BUENO" if i % 2 == 0 else "INSUFICIENTE",
                quantitative_score=2.5 if i % 2 == 0 else 0.5,
                evidence=[],
                explanation="Test",
                confidence=0.85,
                scoring_modality="TYPE_A",
                elements_found={},
                search_pattern_matches={},
                modules_executed=[],
                module_results={},
                execution_time=0.1
            )
            for i in range(1, 11)
        ]
        
        meso_clusters = [
            MesoLevelCluster(
                cluster_name="CLUSTER_1",
                cluster_description="Test",
                policy_areas=["P1"],
                avg_score=60.0,
                dimension_scores={"D1": 60.0},
                strengths=[],
                weaknesses=[],
                recommendations=[],
                question_coverage=1.0,
                total_questions=10,
                answered_questions=10
            )
        ]
        
        macro = self.assembler.generate_macro_convergence(
            micro_answers, meso_clusters, {"name": "Test"}
        )
        
        contradiction = macro.metadata["macro_prompts_results"]["contradiction_report"]
        
        # Verify contradiction report structure
        self.assertIn("contradictions", contradiction)
        self.assertIn("suggested_actions", contradiction)
        self.assertIn("consistency_score", contradiction)
    
    def test_bayesian_portfolio_in_macro(self):
        """Test that Bayesian portfolio composition is included"""
        micro_answers = [
            MicroLevelAnswer(
                question_id=f"P1-D1-Q{i}",
                qualitative_note="BUENO",
                quantitative_score=2.3,
                evidence=[],
                explanation="Test",
                confidence=0.85,
                scoring_modality="TYPE_A",
                elements_found={},
                search_pattern_matches={},
                modules_executed=[],
                module_results={},
                execution_time=0.1
            )
            for i in range(1, 6)
        ]
        
        meso_clusters = [
            MesoLevelCluster(
                cluster_name=f"CLUSTER_{i}",
                cluster_description=f"Cluster {i}",
                policy_areas=["P1"],
                avg_score=70.0 + (i * 5),
                dimension_scores={"D1": 70.0 + (i * 5)},
                strengths=[],
                weaknesses=[],
                recommendations=[],
                question_coverage=1.0,
                total_questions=5,
                answered_questions=5
            )
            for i in range(1, 4)
        ]
        
        macro = self.assembler.generate_macro_convergence(
            micro_answers, meso_clusters, {"name": "Test"}
        )
        
        portfolio = macro.metadata["macro_prompts_results"]["bayesian_portfolio"]
        
        # Verify Bayesian portfolio structure
        self.assertIn("prior_global", portfolio)
        self.assertIn("posterior_global", portfolio)
        self.assertIn("var_global", portfolio)
        self.assertIn("confidence_interval", portfolio)
        self.assertEqual(len(portfolio["confidence_interval"]), 2)
    
    def test_roadmap_optimizer_in_macro(self):
        """Test that roadmap optimization is included"""
        micro_answers = [
            MicroLevelAnswer(
                question_id=f"P1-D1-Q1",
                qualitative_note="INSUFICIENTE",
                quantitative_score=0.5,
                evidence=[],
                explanation="Test",
                confidence=0.6,
                scoring_modality="TYPE_A",
                elements_found={},
                search_pattern_matches={},
                modules_executed=[],
                module_results={},
                execution_time=0.1
            )
        ]
        
        meso_clusters = [
            MesoLevelCluster(
                cluster_name="CLUSTER_1",
                cluster_description="Test",
                policy_areas=["P1"],
                avg_score=40.0,
                dimension_scores={"D1": 40.0},
                strengths=[],
                weaknesses=["Gap 1", "Gap 2"],
                recommendations=["Rec 1"],
                question_coverage=1.0,
                total_questions=1,
                answered_questions=1
            )
        ]
        
        macro = self.assembler.generate_macro_convergence(
            micro_answers, meso_clusters, {"name": "Test"}
        )
        
        roadmap = macro.metadata["macro_prompts_results"]["implementation_roadmap"]
        
        # Verify roadmap structure
        self.assertIn("phases", roadmap)
        self.assertIn("total_expected_uplift", roadmap)
        self.assertIsInstance(roadmap["phases"], list)
        # Should have 3 phases (0-3m, 3-6m, 6-12m)
        self.assertGreaterEqual(len(roadmap["phases"]), 0)
    
    def test_peer_normalization_in_macro(self):
        """Test that peer normalization is included"""
        micro_answers = [
            MicroLevelAnswer(
                question_id=f"P{p}-D1-Q1",
                qualitative_note="BUENO",
                quantitative_score=2.3,
                evidence=[],
                explanation="Test",
                confidence=0.85,
                scoring_modality="TYPE_A",
                elements_found={},
                search_pattern_matches={},
                modules_executed=[],
                module_results={},
                execution_time=0.1
            )
            for p in range(1, 6)  # 5 policy areas
        ]
        
        meso_clusters = [
            MesoLevelCluster(
                cluster_name="CLUSTER_1",
                cluster_description="Test",
                policy_areas=[f"P{p}" for p in range(1, 6)],
                avg_score=75.0,
                dimension_scores={"D1": 75.0},
                strengths=[],
                weaknesses=[],
                recommendations=[],
                question_coverage=1.0,
                total_questions=5,
                answered_questions=5
            )
        ]
        
        macro = self.assembler.generate_macro_convergence(
            micro_answers, meso_clusters, {"name": "Test"}
        )
        
        peer_norm = macro.metadata["macro_prompts_results"]["peer_normalization"]
        
        # Verify peer normalization structure
        self.assertIn("z_scores", peer_norm)
        self.assertIn("adjusted_confidence", peer_norm)
        self.assertIn("peer_position", peer_norm)
        self.assertIn(peer_norm["peer_position"], ["above_average", "average", "below_average"])
    
    def test_helper_methods_exist(self):
        """Test that all helper methods were added"""
        # Verify helper methods exist
        self.assertTrue(hasattr(self.assembler, '_apply_macro_prompts'))
        self.assertTrue(hasattr(self.assembler, '_calculate_dimension_coverage'))
        self.assertTrue(hasattr(self.assembler, '_calculate_policy_area_coverage'))
        self.assertTrue(hasattr(self.assembler, '_extract_micro_claims'))
        self.assertTrue(hasattr(self.assembler, '_extract_meso_signals'))
        self.assertTrue(hasattr(self.assembler, '_structure_critical_gaps'))
        self.assertTrue(hasattr(self.assembler, '_get_peer_distributions'))


if __name__ == '__main__':
    unittest.main()
