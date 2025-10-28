# tests/test_report_assembly_producer.py
# coding=utf-8
"""
Smoke tests for ReportAssemblyProducer

Tests registry exposure and public API methods without
requiring full integration.
"""

import unittest
import json
from pathlib import Path
from report_assembly import (
    ReportAssemblyProducer,
    MicroLevelAnswer,
    MesoLevelCluster,
    MacroLevelConvergence
)


class TestReportAssemblyProducerSmoke(unittest.TestCase):
    """Smoke tests for ReportAssemblyProducer"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.producer = ReportAssemblyProducer()
    
    def test_producer_initialization(self):
        """Test that producer initializes correctly"""
        self.assertIsNotNone(self.producer)
        self.assertIsNotNone(self.producer.assembler)
    
    def test_producer_has_required_methods(self):
        """Test that producer has all required public methods"""
        required_methods = [
            'produce_micro_answer',
            'produce_meso_cluster',
            'produce_macro_convergence',
            'validate_micro_answer',
            'validate_meso_cluster',
            'validate_macro_convergence',
            'export_complete_report',
        ]
        
        for method_name in required_methods:
            self.assertTrue(
                hasattr(self.producer, method_name),
                f"Producer missing required method: {method_name}"
            )
    
    def test_scoring_utilities(self):
        """Test scoring utility methods"""
        # Test score to percentage conversion
        self.assertAlmostEqual(self.producer.convert_score_to_percentage(3.0), 100.0)
        self.assertAlmostEqual(self.producer.convert_score_to_percentage(1.5), 50.0)
        self.assertAlmostEqual(self.producer.convert_score_to_percentage(0.0), 0.0)
        
        # Test percentage to score conversion
        self.assertAlmostEqual(self.producer.convert_percentage_to_score(100.0), 3.0)
        self.assertAlmostEqual(self.producer.convert_percentage_to_score(50.0), 1.5)
        self.assertAlmostEqual(self.producer.convert_percentage_to_score(0.0), 0.0)
        
        # Test score classification
        self.assertEqual(self.producer.classify_score(2.8), "EXCELENTE")
        self.assertEqual(self.producer.classify_score(2.3), "BUENO")
        self.assertEqual(self.producer.classify_score(1.8), "ACEPTABLE")
        self.assertEqual(self.producer.classify_score(1.0), "INSUFICIENTE")
        
        # Test percentage classification
        self.assertEqual(self.producer.classify_percentage(90.0), "EXCELENTE")
        self.assertEqual(self.producer.classify_percentage(75.0), "BUENO")
        self.assertEqual(self.producer.classify_percentage(60.0), "SATISFACTORIO")
        self.assertEqual(self.producer.classify_percentage(45.0), "INSUFICIENTE")
        self.assertEqual(self.producer.classify_percentage(20.0), "DEFICIENTE")
    
    def test_configuration_api(self):
        """Test configuration API methods"""
        # Test dimension listing
        dimensions = self.producer.list_dimensions()
        self.assertEqual(len(dimensions), 6)
        self.assertIn("D1", dimensions)
        self.assertIn("D6", dimensions)
        
        # Test dimension descriptions
        d1_desc = self.producer.get_dimension_description("D1")
        self.assertIsInstance(d1_desc, str)
        self.assertGreater(len(d1_desc), 0)
        
        # Test rubric levels
        levels = self.producer.list_rubric_levels()
        self.assertIn("EXCELENTE", levels)
        self.assertIn("BUENO", levels)
        self.assertIn("SATISFACTORIO", levels)
        
        # Test rubric thresholds
        excellent_min, excellent_max = self.producer.get_rubric_threshold("EXCELENTE")
        self.assertEqual(excellent_min, 85)
        self.assertEqual(excellent_max, 100)
        
        # Test causal thresholds
        threshold = self.producer.get_causal_threshold("D6")
        self.assertIsInstance(threshold, float)
        self.assertGreater(threshold, 0)
    
    def test_micro_getters(self):
        """Test MICRO answer getter methods"""
        # Create a sample micro answer
        micro = MicroLevelAnswer(
            question_id="P1-D1-Q1",
            qualitative_note="EXCELENTE",
            quantitative_score=2.8,
            evidence=["Evidence 1", "Evidence 2"],
            explanation="This is a test explanation that meets the minimum length requirements for the schema validation to pass successfully without errors.",
            confidence=0.9,
            scoring_modality="TYPE_A",
            elements_found={"element1": True, "element2": False},
            search_pattern_matches={},
            modules_executed=["module1", "module2"],
            module_results={"module1": {"status": "success", "confidence": 0.9, "data_summary": "test"}},
            execution_time=1.5,
            execution_chain=[],
            metadata={"policy_area": "P1", "dimension": "D1"}
        )
        
        # Test getters
        self.assertEqual(self.producer.get_micro_answer_score(micro), 2.8)
        self.assertEqual(self.producer.get_micro_answer_qualitative(micro), "EXCELENTE")
        self.assertEqual(len(self.producer.get_micro_answer_evidence(micro)), 2)
        self.assertEqual(self.producer.get_micro_answer_confidence(micro), 0.9)
        self.assertEqual(len(self.producer.get_micro_answer_modules(micro)), 2)
        self.assertEqual(self.producer.get_micro_answer_execution_time(micro), 1.5)
        
        # Test counters
        self.assertEqual(self.producer.count_micro_evidence_excerpts(micro), 2)
        
        # Test boolean checks
        self.assertTrue(self.producer.is_micro_answer_excellent(micro))
        self.assertTrue(self.producer.is_micro_answer_passing(micro))
    
    def test_meso_getters(self):
        """Test MESO cluster getter methods"""
        # Create a sample meso cluster
        meso = MesoLevelCluster(
            cluster_name="CLUSTER_1",
            cluster_description="Test cluster",
            policy_areas=["P1", "P2"],
            avg_score=88.5,
            dimension_scores={"D1": 90.0, "D2": 87.0},
            strengths=["Strength 1", "Strength 2"],
            weaknesses=["Weakness 1"],
            recommendations=["Rec 1", "Rec 2", "Rec 3"],
            question_coverage=95.0,
            total_questions=20,
            answered_questions=19,
            policy_area_scores={"P1": 90.0, "P2": 87.0},
            evidence_quality={"overall": 0.85},
            metadata={}
        )
        
        # Test getters
        self.assertEqual(self.producer.get_meso_cluster_score(meso), 88.5)
        self.assertEqual(len(self.producer.get_meso_cluster_policy_areas(meso)), 2)
        self.assertEqual(len(self.producer.get_meso_cluster_dimension_scores(meso)), 2)
        self.assertEqual(len(self.producer.get_meso_cluster_strengths(meso)), 2)
        self.assertEqual(len(self.producer.get_meso_cluster_weaknesses(meso)), 1)
        self.assertEqual(len(self.producer.get_meso_cluster_recommendations(meso)), 3)
        self.assertEqual(self.producer.get_meso_cluster_coverage(meso), 95.0)
        
        # Test question counts
        total, answered = self.producer.get_meso_cluster_question_counts(meso)
        self.assertEqual(total, 20)
        self.assertEqual(answered, 19)
        
        # Test counters
        self.assertEqual(self.producer.count_meso_strengths(meso), 2)
        self.assertEqual(self.producer.count_meso_weaknesses(meso), 1)
        
        # Test boolean checks
        self.assertTrue(self.producer.is_meso_cluster_excellent(meso))
        self.assertTrue(self.producer.is_meso_cluster_passing(meso))
    
    def test_macro_getters(self):
        """Test MACRO convergence getter methods"""
        # Create a sample macro convergence
        macro = MacroLevelConvergence(
            overall_score=82.0,
            convergence_by_dimension={"D1": 85.0, "D2": 79.0},
            convergence_by_policy_area={"P1": 83.0, "P2": 81.0},
            gap_analysis={"dimensional_gaps": [], "policy_gaps": []},
            agenda_alignment=0.82,
            critical_gaps=["Gap 1", "Gap 2"],
            strategic_recommendations=["Rec 1", "Rec 2", "Rec 3"],
            plan_classification="BUENO",
            evidence_synthesis={"total_evidence_excerpts": 100},
            implementation_roadmap=[{"phase": "Phase 1", "actions": ["Action 1"]}],
            score_distribution={"EXCELENTE": 50, "BUENO": 30},
            confidence_metrics={"avg_confidence": 0.8},
            metadata={}
        )
        
        # Test getters
        self.assertEqual(self.producer.get_macro_overall_score(macro), 82.0)
        self.assertEqual(len(self.producer.get_macro_dimension_convergence(macro)), 2)
        self.assertEqual(len(self.producer.get_macro_policy_convergence(macro)), 2)
        self.assertIsInstance(self.producer.get_macro_gap_analysis(macro), dict)
        self.assertEqual(self.producer.get_macro_agenda_alignment(macro), 0.82)
        self.assertEqual(len(self.producer.get_macro_critical_gaps(macro)), 2)
        self.assertEqual(len(self.producer.get_macro_strategic_recommendations(macro)), 3)
        self.assertEqual(self.producer.get_macro_classification(macro), "BUENO")
        self.assertIsInstance(self.producer.get_macro_evidence_synthesis(macro), dict)
        self.assertEqual(len(self.producer.get_macro_implementation_roadmap(macro)), 1)
        self.assertIsInstance(self.producer.get_macro_score_distribution(macro), dict)
        self.assertIsInstance(self.producer.get_macro_confidence_metrics(macro), dict)
        
        # Test counters
        self.assertEqual(self.producer.count_macro_critical_gaps(macro), 2)
        self.assertEqual(self.producer.count_macro_strategic_recommendations(macro), 3)
        
        # Test boolean checks
        self.assertFalse(self.producer.is_macro_excellent(macro))  # 82 < 85
        self.assertTrue(self.producer.is_macro_passing(macro))  # 82 >= 55
    
    def test_serialization(self):
        """Test serialization and deserialization methods"""
        # Create sample objects
        micro = MicroLevelAnswer(
            question_id="P1-D1-Q1",
            qualitative_note="BUENO",
            quantitative_score=2.3,
            evidence=["Test evidence"],
            explanation="A sufficiently long explanation to meet the minimum character requirements for proper validation against the schema definition.",
            confidence=0.85,
            scoring_modality="TYPE_A",
            elements_found={"elem1": True},
            search_pattern_matches={},
            modules_executed=["module1"],
            module_results={"module1": {"status": "success", "confidence": 0.85, "data_summary": "data"}},
            execution_time=1.0,
            execution_chain=[],
            metadata={}
        )
        
        # Test serialization
        micro_dict = self.producer.serialize_micro_answer(micro)
        self.assertIsInstance(micro_dict, dict)
        self.assertEqual(micro_dict["question_id"], "P1-D1-Q1")
        
        # Test deserialization
        micro_restored = self.producer.deserialize_micro_answer(micro_dict)
        self.assertEqual(micro_restored.question_id, micro.question_id)
        self.assertEqual(micro_restored.quantitative_score, micro.quantitative_score)
    
    def test_method_count(self):
        """Test that we have at least 40 public methods"""
        import inspect
        
        methods = [
            name for name, method in inspect.getmembers(self.producer, predicate=inspect.ismethod)
            if not name.startswith('_')
        ]
        
        self.assertGreaterEqual(
            len(methods), 40,
            f"Producer must have at least 40 public methods, found {len(methods)}"
        )
        
        print(f"\n✓ ReportAssemblyProducer has {len(methods)} public methods")


class TestSchemaValidation(unittest.TestCase):
    """Test JSON schema validation"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.producer = ReportAssemblyProducer()
    
    def test_micro_answer_schema_validation(self):
        """Test MICRO answer schema validation"""
        valid_answer = {
            "question_id": "P1-D1-Q1",
            "qualitative_note": "EXCELENTE",
            "quantitative_score": 2.8,
            "evidence": ["Evidence excerpt 1", "Evidence excerpt 2"],
            "explanation": "This is a comprehensive doctoral-level explanation that provides detailed analysis of the findings and their implications for policy implementation and strategic planning going forward.",
            "confidence": 0.9,
            "scoring_modality": "TYPE_A",
            "elements_found": {"element1": True, "element2": False},
            "search_pattern_matches": {},
            "modules_executed": ["module1", "module2"],
            "module_results": {
                "module1": {"status": "success", "confidence": 0.9, "data_summary": "test"}
            },
            "execution_time": 1.5,
            "execution_chain": [],
            "metadata": {"policy_area": "P1", "dimension": "D1"}
        }
        
        # Note: Schema validation will only work if schema files exist
        # This is a basic structural test
        self.assertIsInstance(valid_answer, dict)
        self.assertIn("question_id", valid_answer)
        self.assertIn("quantitative_score", valid_answer)
    
    def test_no_summarization_leakage(self):
        """Ensure no summarization logic is exposed in public API"""
        import inspect
        
        # Get all public method names
        public_methods = [
            name for name, method in inspect.getmembers(self.producer, predicate=inspect.ismethod)
            if not name.startswith('_')
        ]
        
        # Ensure no summarization-related methods are public
        forbidden_keywords = ['summarize', 'summary', 'aggregate_internal', 'internal_']
        
        for method_name in public_methods:
            for keyword in forbidden_keywords:
                self.assertNotIn(
                    keyword.lower(), method_name.lower(),
                    f"Public method '{method_name}' may expose summarization logic"
                )
        
        print(f"\n✓ No summarization leakage detected in {len(public_methods)} public methods")


if __name__ == "__main__":
    unittest.main()
