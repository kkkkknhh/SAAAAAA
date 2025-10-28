"""
Tests for Bayesian Multi-Level Analysis System
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bayesian_multilevel_system import (
    # Micro level
    ReconciliationValidator,
    ValidationRule,
    ValidatorType,
    BayesianUpdater,
    ProbativeTest,
    ProbativeTestType,
    MicroLevelAnalysis,
    
    # Meso level
    DispersionEngine,
    PeerCalibrator,
    PeerContext,
    BayesianRollUp,
    MesoLevelAnalysis,
    
    # Macro level
    ContradictionScanner,
    BayesianPortfolioComposer,
    MacroLevelAnalysis,
    
    # Orchestrator
    MultiLevelBayesianOrchestrator
)


class TestReconciliationValidator(unittest.TestCase):
    """Test reconciliation layer validators"""
    
    def setUp(self):
        self.rules = [
            ValidationRule(
                validator_type=ValidatorType.RANGE,
                field_name="score",
                expected_range=(0.0, 1.0),
                penalty_factor=0.2
            ),
            ValidationRule(
                validator_type=ValidatorType.UNIT,
                field_name="budget_unit",
                expected_unit="COP",
                penalty_factor=0.1
            ),
            ValidationRule(
                validator_type=ValidatorType.PERIOD,
                field_name="time_period",
                expected_period="2024-2027",
                penalty_factor=0.15
            )
        ]
        self.validator = ReconciliationValidator(self.rules)
    
    def test_range_validation_pass(self):
        """Test range validation with valid value"""
        data = {"score": 0.75}
        results = self.validator.validate_data(data)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)
        self.assertEqual(results[0].penalty_applied, 0.0)
    
    def test_range_validation_fail(self):
        """Test range validation with invalid value"""
        data = {"score": 1.5}
        results = self.validator.validate_data(data)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)
        self.assertGreater(results[0].penalty_applied, 0.0)
    
    def test_unit_validation_pass(self):
        """Test unit validation with correct unit"""
        data = {"budget_unit": "COP"}
        results = self.validator.validate_data(data)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].passed)
    
    def test_unit_validation_fail(self):
        """Test unit validation with wrong unit"""
        data = {"budget_unit": "USD"}
        results = self.validator.validate_data(data)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].passed)
        self.assertGreater(results[0].penalty_applied, 0.0)
    
    def test_total_penalty_calculation(self):
        """Test total penalty calculation"""
        data = {
            "score": 1.5,  # Out of range
            "budget_unit": "USD",  # Wrong unit
            "time_period": "2020-2023"  # Wrong period
        }
        results = self.validator.validate_data(data)
        total_penalty = self.validator.calculate_total_penalty(results)
        self.assertGreater(total_penalty, 0.0)


class TestBayesianUpdater(unittest.TestCase):
    """Test Bayesian updater with probative tests"""
    
    def setUp(self):
        self.updater = BayesianUpdater()
    
    def test_straw_in_wind_test(self):
        """Test weak confirmation (straw-in-wind)"""
        test = ProbativeTest(
            test_type=ProbativeTestType.STRAW_IN_WIND,
            test_name="Weak evidence found",
            evidence_strength=0.2,
            prior_probability=0.5
        )
        
        posterior = self.updater.update(0.5, test, True)
        # Should increase slightly
        self.assertGreater(posterior, 0.5)
        self.assertLess(posterior, 0.7)
    
    def test_hoop_test_pass(self):
        """Test necessary condition passes (hoop test)"""
        test = ProbativeTest(
            test_type=ProbativeTestType.HOOP_TEST,
            test_name="Necessary condition met",
            evidence_strength=0.5,
            prior_probability=0.5
        )
        
        posterior = self.updater.update(0.5, test, True)
        # Should increase moderately
        self.assertGreater(posterior, 0.5)
    
    def test_hoop_test_fail(self):
        """Test necessary condition fails (hoop test)"""
        test = ProbativeTest(
            test_type=ProbativeTestType.HOOP_TEST,
            test_name="Necessary condition failed",
            evidence_strength=0.5,
            prior_probability=0.5
        )
        
        posterior = self.updater.update(0.5, test, False)
        # Should decrease sharply
        self.assertLess(posterior, 0.2)
    
    def test_smoking_gun_test(self):
        """Test sufficient evidence (smoking gun)"""
        test = ProbativeTest(
            test_type=ProbativeTestType.SMOKING_GUN,
            test_name="Definitive evidence found",
            evidence_strength=0.9,
            prior_probability=0.5
        )
        
        posterior = self.updater.update(0.5, test, True)
        # Should increase significantly
        self.assertGreater(posterior, 0.8)
    
    def test_doubly_decisive_test(self):
        """Test necessary and sufficient (doubly decisive)"""
        test = ProbativeTest(
            test_type=ProbativeTestType.DOUBLY_DECISIVE,
            test_name="Decisive evidence",
            evidence_strength=1.0,
            prior_probability=0.5
        )
        
        posterior_pass = self.updater.update(0.5, test, True)
        # Should increase to very high
        self.assertGreater(posterior_pass, 0.9)
        
        # Reset updater
        updater2 = BayesianUpdater()
        posterior_fail = updater2.update(0.5, test, False)
        # Should decrease to very low
        self.assertLess(posterior_fail, 0.1)
    
    def test_sequential_updating(self):
        """Test sequential Bayesian updating"""
        tests = [
            (ProbativeTest(ProbativeTestType.STRAW_IN_WIND, "Test 1", 0.2, 0.5), True),
            (ProbativeTest(ProbativeTestType.HOOP_TEST, "Test 2", 0.5, 0.5), True),
            (ProbativeTest(ProbativeTestType.SMOKING_GUN, "Test 3", 0.9, 0.5), True),
        ]
        
        final_posterior = self.updater.sequential_update(0.3, tests)
        # Should progressively increase
        self.assertGreater(final_posterior, 0.3)
    
    def test_csv_export(self):
        """Test CSV export functionality"""
        test = ProbativeTest(
            test_type=ProbativeTestType.HOOP_TEST,
            test_name="Test export",
            evidence_strength=0.5,
            prior_probability=0.5
        )
        self.updater.update(0.5, test, True)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = Path(f.name)
        
        try:
            self.updater.export_to_csv(temp_path)
            self.assertTrue(temp_path.exists())
            
            # Check content
            content = temp_path.read_text()
            self.assertIn('test_name', content)
            self.assertIn('Test export', content)
        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestDispersionEngine(unittest.TestCase):
    """Test dispersion metrics calculation"""
    
    def setUp(self):
        self.engine = DispersionEngine()
    
    def test_cv_calculation(self):
        """Test coefficient of variation"""
        scores = [0.5, 0.6, 0.7, 0.8]
        cv = self.engine.calculate_cv(scores)
        self.assertGreater(cv, 0.0)
        self.assertLess(cv, 1.0)
    
    def test_cv_uniform_scores(self):
        """Test CV with uniform scores"""
        scores = [0.5, 0.5, 0.5, 0.5]
        cv = self.engine.calculate_cv(scores)
        self.assertAlmostEqual(cv, 0.0)
    
    def test_max_gap_calculation(self):
        """Test maximum gap calculation"""
        scores = [0.2, 0.3, 0.8, 0.9]  # Gap of 0.5 between 0.3 and 0.8
        max_gap = self.engine.calculate_max_gap(scores)
        self.assertAlmostEqual(max_gap, 0.5)
    
    def test_gini_calculation(self):
        """Test Gini coefficient"""
        # Equal scores -> Gini = 0
        equal_scores = [0.5, 0.5, 0.5, 0.5]
        gini_equal = self.engine.calculate_gini(equal_scores)
        self.assertAlmostEqual(gini_equal, 0.0, places=2)
        
        # Unequal scores -> Gini > 0
        unequal_scores = [0.1, 0.2, 0.7, 0.9]
        gini_unequal = self.engine.calculate_gini(unequal_scores)
        self.assertGreater(gini_unequal, 0.0)
    
    def test_dispersion_penalty(self):
        """Test dispersion penalty calculation"""
        # Low dispersion scores
        low_dispersion = [0.7, 0.72, 0.75, 0.73]
        penalty_low, _ = self.engine.calculate_dispersion_penalty(low_dispersion)
        
        # High dispersion scores
        high_dispersion = [0.1, 0.3, 0.7, 0.9]
        penalty_high, _ = self.engine.calculate_dispersion_penalty(high_dispersion)
        
        # High dispersion should have higher penalty
        self.assertGreater(penalty_high, penalty_low)


class TestPeerCalibrator(unittest.TestCase):
    """Test peer calibration"""
    
    def setUp(self):
        self.calibrator = PeerCalibrator()
        self.peer_contexts = [
            PeerContext("peer1", "Peer 1", {"D1": 0.7, "D2": 0.6}),
            PeerContext("peer2", "Peer 2", {"D1": 0.75, "D2": 0.65}),
            PeerContext("peer3", "Peer 3", {"D1": 0.72, "D2": 0.62}),
        ]
    
    def test_compare_to_peers_average(self):
        """Test comparison with average score"""
        comparison = self.calibrator.compare_to_peers(0.72, self.peer_contexts, "D1")
        # Should be close to 0 (within reasonable tolerance)
        self.assertLess(abs(comparison.z_score), 0.5)
        self.assertLess(comparison.deviation_penalty, 0.1)
    
    def test_compare_to_peers_above_average(self):
        """Test comparison with above-average score"""
        comparison = self.calibrator.compare_to_peers(0.85, self.peer_contexts, "D1")
        self.assertGreater(comparison.z_score, 1.0)
        self.assertGreater(comparison.percentile, 0.5)
    
    def test_compare_to_peers_below_average(self):
        """Test comparison with below-average score"""
        comparison = self.calibrator.compare_to_peers(0.5, self.peer_contexts, "D1")
        self.assertLess(comparison.z_score, -1.0)
        self.assertLess(comparison.percentile, 0.5)
    
    def test_narrative_generation(self):
        """Test narrative hook generation"""
        comparison = self.calibrator.compare_to_peers(0.72, self.peer_contexts, "D1")
        self.assertIn("peer average", comparison.narrative)
        self.assertIsInstance(comparison.narrative, str)
        self.assertGreater(len(comparison.narrative), 10)


class TestContradictionScanner(unittest.TestCase):
    """Test contradiction detection"""
    
    def setUp(self):
        self.scanner = ContradictionScanner(discrepancy_threshold=0.3)
    
    def test_no_contradictions(self):
        """Test with consistent scores"""
        micro = MicroLevelAnalysis(
            question_id="Q1",
            raw_score=0.7,
            validation_results=[],
            validation_penalty=0.0,
            bayesian_updates=[],
            final_posterior=0.7,
            adjusted_score=0.7
        )
        
        meso = MesoLevelAnalysis(
            cluster_id="C1",
            micro_scores=[0.7, 0.72, 0.68],
            raw_meso_score=0.7,
            dispersion_metrics={},
            dispersion_penalty=0.0,
            peer_comparison=None,
            peer_penalty=0.0,
            total_penalty=0.0,
            final_posterior=0.7,
            adjusted_score=0.7
        )
        
        contradictions = self.scanner.scan_micro_meso([micro], meso)
        self.assertEqual(len(contradictions), 0)
    
    def test_detect_contradiction(self):
        """Test contradiction detection"""
        micro = MicroLevelAnalysis(
            question_id="Q1",
            raw_score=0.9,
            validation_results=[],
            validation_penalty=0.0,
            bayesian_updates=[],
            final_posterior=0.9,
            adjusted_score=0.9
        )
        
        meso = MesoLevelAnalysis(
            cluster_id="C1",
            micro_scores=[0.3, 0.32, 0.28],
            raw_meso_score=0.3,
            dispersion_metrics={},
            dispersion_penalty=0.0,
            peer_comparison=None,
            peer_penalty=0.0,
            total_penalty=0.0,
            final_posterior=0.3,
            adjusted_score=0.3
        )
        
        contradictions = self.scanner.scan_micro_meso([micro], meso)
        self.assertGreater(len(contradictions), 0)
        self.assertGreater(contradictions[0].severity, 0.0)
    
    def test_contradiction_penalty(self):
        """Test penalty calculation from contradictions"""
        # Create contradictions
        micro1 = MicroLevelAnalysis("Q1", 0.9, [], 0.0, [], 0.9, 0.9)
        meso1 = MesoLevelAnalysis("C1", [0.3], 0.3, {}, 0.0, None, 0.0, 0.0, 0.3, 0.3)
        
        self.scanner.scan_micro_meso([micro1], meso1)
        penalty = self.scanner.calculate_contradiction_penalty()
        
        self.assertGreater(penalty, 0.0)
        self.assertLessEqual(penalty, 1.0)


class TestBayesianPortfolioComposer(unittest.TestCase):
    """Test macro-level portfolio composition"""
    
    def setUp(self):
        self.composer = BayesianPortfolioComposer()
        self.scanner = ContradictionScanner()
    
    def test_coverage_calculation(self):
        """Test coverage score and penalty"""
        # High coverage (90%)
        coverage_high, penalty_high = self.composer.calculate_coverage(270, 300)
        self.assertAlmostEqual(coverage_high, 0.9)
        self.assertAlmostEqual(penalty_high, 0.0)
        
        # Medium coverage (70%)
        coverage_med, penalty_med = self.composer.calculate_coverage(210, 300)
        self.assertAlmostEqual(coverage_med, 0.7)
        self.assertGreater(penalty_med, 0.0)
        
        # Low coverage (50%)
        coverage_low, penalty_low = self.composer.calculate_coverage(150, 300)
        self.assertAlmostEqual(coverage_low, 0.5)
        self.assertGreater(penalty_low, penalty_med)
    
    def test_macro_composition(self):
        """Test macro portfolio composition"""
        meso_analyses = [
            MesoLevelAnalysis(
                cluster_id="C1",
                micro_scores=[0.7, 0.72, 0.68],
                raw_meso_score=0.7,
                dispersion_metrics={},
                dispersion_penalty=0.0,
                peer_comparison=None,
                peer_penalty=0.0,
                total_penalty=0.0,
                final_posterior=0.7,
                adjusted_score=0.7
            ),
            MesoLevelAnalysis(
                cluster_id="C2",
                micro_scores=[0.65, 0.67, 0.63],
                raw_meso_score=0.65,
                dispersion_metrics={},
                dispersion_penalty=0.0,
                peer_comparison=None,
                peer_penalty=0.0,
                total_penalty=0.0,
                final_posterior=0.65,
                adjusted_score=0.65
            ),
        ]
        
        macro = self.composer.compose_macro_portfolio(
            meso_analyses, 300, self.scanner
        )
        
        self.assertIsInstance(macro, MacroLevelAnalysis)
        self.assertGreater(macro.overall_posterior, 0.0)
        self.assertGreater(macro.coverage_score, 0.0)
        self.assertGreaterEqual(macro.adjusted_score, 0.0)
        self.assertLessEqual(macro.adjusted_score, 1.0)
    
    def test_recommendations_generation(self):
        """Test strategic recommendations"""
        meso_analyses = [
            MesoLevelAnalysis("C1", [0.7], 0.7, {}, 0.0, None, 0.0, 0.0, 0.7, 0.7),
        ]
        
        macro = self.composer.compose_macro_portfolio(
            meso_analyses, 300, self.scanner
        )
        
        self.assertIsInstance(macro.recommendations, list)
        self.assertGreater(len(macro.recommendations), 0)


class TestMultiLevelOrchestrator(unittest.TestCase):
    """Test complete multi-level orchestration"""
    
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        
        self.rules = [
            ValidationRule(
                validator_type=ValidatorType.RANGE,
                field_name="score",
                expected_range=(0.0, 1.0),
                penalty_factor=0.1
            )
        ]
        
        self.orchestrator = MultiLevelBayesianOrchestrator(
            self.rules,
            output_dir=self.temp_dir
        )
    
    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """Test complete multi-level pipeline"""
        # Prepare micro data
        micro_data = [
            {
                'question_id': 'P1-D1-Q1',
                'raw_score': 0.7,
                'score': 0.7,
                'probative_tests': [
                    (ProbativeTest(ProbativeTestType.HOOP_TEST, "Test 1", 0.5, 0.5), True)
                ]
            },
            {
                'question_id': 'P1-D1-Q2',
                'raw_score': 0.65,
                'score': 0.65,
                'probative_tests': []
            },
            {
                'question_id': 'P1-D2-Q1',
                'raw_score': 0.72,
                'score': 0.72,
                'probative_tests': []
            }
        ]
        
        # Cluster mapping
        cluster_mapping = {
            'CLUSTER_D1': ['P1-D1-Q1', 'P1-D1-Q2'],
            'CLUSTER_D2': ['P1-D2-Q1']
        }
        
        # Run complete analysis
        micro_analyses, meso_analyses, macro_analysis = (
            self.orchestrator.run_complete_analysis(
                micro_data,
                cluster_mapping,
                peer_contexts=None,
                total_questions=300
            )
        )
        
        # Verify results
        self.assertEqual(len(micro_analyses), 3)
        self.assertEqual(len(meso_analyses), 2)
        self.assertIsInstance(macro_analysis, MacroLevelAnalysis)
        
        # Check CSV outputs
        self.assertTrue((self.temp_dir / "posterior_table_micro.csv").exists())
        self.assertTrue((self.temp_dir / "posterior_table_meso.csv").exists())
        self.assertTrue((self.temp_dir / "posterior_table_macro.csv").exists())
    
    def test_with_peer_contexts(self):
        """Test pipeline with peer calibration"""
        micro_data = [
            {'question_id': 'Q1', 'raw_score': 0.7, 'score': 0.7, 'probative_tests': []}
        ]
        
        cluster_mapping = {'C1': ['Q1']}
        
        peer_contexts = [
            PeerContext("peer1", "Peer 1", {"C1": 0.65}),
            PeerContext("peer2", "Peer 2", {"C1": 0.68}),
        ]
        
        _, meso_analyses, _ = self.orchestrator.run_complete_analysis(
            micro_data, cluster_mapping, peer_contexts, 300
        )
        
        # Should have peer comparison
        self.assertIsNotNone(meso_analyses[0].peer_comparison)


if __name__ == '__main__':
    unittest.main()
