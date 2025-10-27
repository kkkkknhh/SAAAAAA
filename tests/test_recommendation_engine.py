# tests/test_recommendation_engine.py
# coding=utf-8
"""
Unit tests for the recommendation engine
"""

import unittest
import json
import tempfile
from pathlib import Path
from recommendation_engine import (
    RecommendationEngine,
    Recommendation,
    RecommendationSet,
    load_recommendation_engine
)


class TestRecommendationEngine(unittest.TestCase):
    """Test cases for RecommendationEngine"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.engine = load_recommendation_engine()
    
    def test_engine_initialization(self):
        """Test that engine initializes correctly"""
        self.assertIsNotNone(self.engine)
        self.assertGreater(len(self.engine.rules_by_level['MICRO']), 0)
        self.assertGreater(len(self.engine.rules_by_level['MESO']), 0)
        self.assertGreater(len(self.engine.rules_by_level['MACRO']), 0)
    
    def test_schema_validation(self):
        """Test that rules conform to schema"""
        # If we got here, schema validation passed during init
        self.assertEqual(self.engine.rules.get('version'), '1.0')
    
    def test_micro_recommendations_generation(self):
        """Test MICRO-level recommendation generation"""
        # Scores below thresholds
        micro_scores = {
            'PA01-DIM01': 1.2,  # Below 1.65
            'PA02-DIM02': 1.5,  # Below 1.65
            'PA03-DIM03': 2.0,  # Above 1.65
        }
        
        rec_set = self.engine.generate_micro_recommendations(micro_scores)
        
        self.assertEqual(rec_set.level, 'MICRO')
        self.assertGreater(rec_set.rules_matched, 0)
        self.assertGreater(rec_set.total_rules_evaluated, 0)
        
        # Check that recommendations have required fields
        for rec in rec_set.recommendations:
            self.assertIsNotNone(rec.rule_id)
            self.assertIsNotNone(rec.problem)
            self.assertIsNotNone(rec.intervention)
            self.assertIsNotNone(rec.indicator)
            self.assertIsNotNone(rec.responsible)
            self.assertIsNotNone(rec.horizon)
            self.assertIsNotNone(rec.verification)
    
    def test_micro_no_matches(self):
        """Test MICRO when no rules match"""
        # All scores above thresholds
        micro_scores = {
            'PA01-DIM01': 2.5,
            'PA02-DIM02': 2.8,
        }
        
        rec_set = self.engine.generate_micro_recommendations(micro_scores)
        
        self.assertEqual(rec_set.rules_matched, 0)
        self.assertEqual(len(rec_set.recommendations), 0)
    
    def test_meso_recommendations_generation(self):
        """Test MESO-level recommendation generation"""
        cluster_data = {
            'CL01': {
                'score': 72.0,
                'variance': 0.25,
                'weak_pa': 'PA02'
            },
            'CL02': {
                'score': 58.0,
                'variance': 0.12,
            }
        }
        
        rec_set = self.engine.generate_meso_recommendations(cluster_data)
        
        self.assertEqual(rec_set.level, 'MESO')
        self.assertGreaterEqual(rec_set.rules_matched, 0)
        
        # Check metadata
        for rec in rec_set.recommendations:
            self.assertIn('cluster_id', rec.metadata)
            self.assertIn('score', rec.metadata)
    
    def test_meso_score_bands(self):
        """Test MESO score band logic"""
        # Test BAJO band
        cluster_data_bajo = {
            'CL01': {'score': 45.0, 'variance': 0.05}
        }
        
        # Test MEDIO band
        cluster_data_medio = {
            'CL01': {'score': 65.0, 'variance': 0.10}
        }
        
        # Test ALTO band
        cluster_data_alto = {
            'CL01': {'score': 85.0, 'variance': 0.03}
        }
        
        # Each should potentially match different rules
        rec_bajo = self.engine.generate_meso_recommendations(cluster_data_bajo)
        rec_medio = self.engine.generate_meso_recommendations(cluster_data_medio)
        rec_alto = self.engine.generate_meso_recommendations(cluster_data_alto)
        
        # At least one should have recommendations
        total = rec_bajo.rules_matched + rec_medio.rules_matched + rec_alto.rules_matched
        self.assertGreater(total, 0)
    
    def test_macro_recommendations_generation(self):
        """Test MACRO-level recommendation generation"""
        macro_data = {
            'macro_band': 'SATISFACTORIO',
            'clusters_below_target': ['CL02', 'CL03'],
            'variance_alert': 'MODERADA',
            'priority_micro_gaps': ['PA01-DIM05', 'PA05-DIM04', 'PA04-DIM04', 'PA08-DIM05']
        }
        
        rec_set = self.engine.generate_macro_recommendations(macro_data)
        
        self.assertEqual(rec_set.level, 'MACRO')
        self.assertGreaterEqual(rec_set.rules_matched, 0)
        
        # Check metadata
        for rec in rec_set.recommendations:
            self.assertIn('macro_band', rec.metadata)
    
    def test_macro_different_bands(self):
        """Test MACRO recommendations for different bands"""
        # DEFICIENTE
        macro_deficiente = {
            'macro_band': 'DEFICIENTE',
            'clusters_below_target': ['CL01', 'CL02', 'CL03', 'CL04'],
            'variance_alert': 'GENERALIZADA',
            'priority_micro_gaps': ['PA02-DIM01', 'PA03-DIM05', 'PA05-DIM03', 'PA09-DIM05', 'PA10-DIM06']
        }
        
        # EXCELENTE
        macro_excelente = {
            'macro_band': 'EXCELENTE',
            'clusters_below_target': [],
            'variance_alert': 'SIN_ALERTA',
            'priority_micro_gaps': []
        }
        
        rec_def = self.engine.generate_macro_recommendations(macro_deficiente)
        rec_exc = self.engine.generate_macro_recommendations(macro_excelente)
        
        # Each should potentially match different rules
        self.assertGreaterEqual(rec_def.rules_matched, 0)
        self.assertGreaterEqual(rec_exc.rules_matched, 0)
    
    def test_variable_substitution(self):
        """Test variable substitution in templates"""
        text = "La pregunta {{PAxx}} en {{DIMxx}} tiene problemas."
        substitutions = {'PAxx': 'PA01', 'DIMxx': 'DIM05'}
        
        result = self.engine._substitute_variables(text, substitutions)
        
        self.assertEqual(result, "La pregunta PA01 en DIM05 tiene problemas.")
        self.assertNotIn('{{', result)
    
    def test_generate_all_recommendations(self):
        """Test generating recommendations at all levels"""
        micro_scores = {'PA01-DIM01': 1.2}
        cluster_data = {'CL01': {'score': 72.0, 'variance': 0.25, 'weak_pa': 'PA02'}}
        macro_data = {
            'macro_band': 'SATISFACTORIO',
            'clusters_below_target': ['CL02', 'CL03'],
            'variance_alert': 'MODERADA',
            'priority_micro_gaps': ['PA01-DIM05', 'PA05-DIM04', 'PA04-DIM04', 'PA08-DIM05']
        }
        
        all_recs = self.engine.generate_all_recommendations(
            micro_scores, cluster_data, macro_data
        )
        
        self.assertIn('MICRO', all_recs)
        self.assertIn('MESO', all_recs)
        self.assertIn('MACRO', all_recs)
        
        self.assertIsInstance(all_recs['MICRO'], RecommendationSet)
        self.assertIsInstance(all_recs['MESO'], RecommendationSet)
        self.assertIsInstance(all_recs['MACRO'], RecommendationSet)
    
    def test_export_json(self):
        """Test exporting recommendations as JSON"""
        micro_scores = {'PA01-DIM01': 1.2}
        rec_set = self.engine.generate_micro_recommendations(micro_scores)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            self.engine.export_recommendations(
                {'MICRO': rec_set},
                output_path,
                format='json'
            )
            
            # Read back and validate
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('MICRO', data)
            self.assertIn('recommendations', data['MICRO'])
            self.assertIn('generated_at', data['MICRO'])
        finally:
            Path(output_path).unlink()
    
    def test_export_markdown(self):
        """Test exporting recommendations as Markdown"""
        micro_scores = {'PA01-DIM01': 1.2}
        rec_set = self.engine.generate_micro_recommendations(micro_scores)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            output_path = f.name
        
        try:
            self.engine.export_recommendations(
                {'MICRO': rec_set},
                output_path,
                format='markdown'
            )
            
            # Read back and validate
            with open(output_path, 'r') as f:
                content = f.read()
            
            self.assertIn('# Recomendaciones', content)
            self.assertIn('## Nivel MICRO', content)
        finally:
            Path(output_path).unlink()
    
    def test_recommendation_to_dict(self):
        """Test converting Recommendation to dictionary"""
        rec = Recommendation(
            rule_id='TEST-001',
            level='MICRO',
            problem='Test problem',
            intervention='Test intervention',
            indicator={'name': 'test', 'target': 0.8, 'unit': 'proporción', 'baseline': None},
            responsible={'entity': 'Test Entity', 'role': 'test', 'partners': ['A', 'B']},
            horizon={'start': 'T0', 'end': 'T1'},
            verification=['V1', 'V2']
        )
        
        rec_dict = rec.to_dict()
        
        self.assertEqual(rec_dict['rule_id'], 'TEST-001')
        self.assertEqual(rec_dict['level'], 'MICRO')
        self.assertIsInstance(rec_dict, dict)
    
    def test_recommendation_set_to_dict(self):
        """Test converting RecommendationSet to dictionary"""
        rec = Recommendation(
            rule_id='TEST-001',
            level='MICRO',
            problem='Test',
            intervention='Test',
            indicator={'name': 'test', 'target': 0.8, 'unit': 'proporción', 'baseline': None},
            responsible={'entity': 'Test', 'role': 'test', 'partners': []},
            horizon={'start': 'T0', 'end': 'T1'},
            verification=[]
        )
        
        from datetime import datetime, timezone
        rec_set = RecommendationSet(
            level='MICRO',
            recommendations=[rec],
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_rules_evaluated=10,
            rules_matched=1
        )
        
        rec_set_dict = rec_set.to_dict()
        
        self.assertEqual(rec_set_dict['level'], 'MICRO')
        self.assertEqual(rec_set_dict['rules_matched'], 1)
        self.assertIsInstance(rec_set_dict['recommendations'], list)


class TestMESOConditions(unittest.TestCase):
    """Test MESO-level condition checking"""
    
    @classmethod
    def setUpClass(cls):
        cls.engine = load_recommendation_engine()
    
    def test_check_meso_conditions_bajo(self):
        """Test BAJO score band condition"""
        # BAJO: score < 55
        result = self.engine._check_meso_conditions(
            score=45.0,
            variance=0.10,
            weak_pa='PA01',
            score_band='BAJO',
            variance_level='MEDIA',
            variance_threshold=None,
            weak_pa_id=None
        )
        self.assertTrue(result)
        
        # Not BAJO
        result = self.engine._check_meso_conditions(
            score=60.0,
            variance=0.10,
            weak_pa='PA01',
            score_band='BAJO',
            variance_level='MEDIA',
            variance_threshold=None,
            weak_pa_id=None
        )
        self.assertFalse(result)
    
    def test_check_meso_conditions_medio(self):
        """Test MEDIO score band condition"""
        # MEDIO: 55 <= score < 75
        result = self.engine._check_meso_conditions(
            score=65.0,
            variance=0.10,
            weak_pa='PA01',
            score_band='MEDIO',
            variance_level='MEDIA',
            variance_threshold=None,
            weak_pa_id=None
        )
        self.assertTrue(result)
    
    def test_check_meso_conditions_alto(self):
        """Test ALTO score band condition"""
        # ALTO: score >= 75
        result = self.engine._check_meso_conditions(
            score=80.0,
            variance=0.05,
            weak_pa='PA01',
            score_band='ALTO',
            variance_level='BAJA',
            variance_threshold=None,
            weak_pa_id=None
        )
        self.assertTrue(result)
    
    def test_check_meso_variance_levels(self):
        """Test variance level conditions"""
        # BAJA: variance < 0.08
        result = self.engine._check_meso_conditions(
            score=65.0,
            variance=0.05,
            weak_pa='PA01',
            score_band='MEDIO',
            variance_level='BAJA',
            variance_threshold=None,
            weak_pa_id=None
        )
        self.assertTrue(result)
        
        # MEDIA: 0.08 <= variance < 0.18
        result = self.engine._check_meso_conditions(
            score=65.0,
            variance=0.12,
            weak_pa='PA01',
            score_band='MEDIO',
            variance_level='MEDIA',
            variance_threshold=None,
            weak_pa_id=None
        )
        self.assertTrue(result)
        
        # ALTA: variance >= 0.18 or >= variance_threshold
        result = self.engine._check_meso_conditions(
            score=65.0,
            variance=0.26,
            weak_pa='PA02',
            score_band='MEDIO',
            variance_level='ALTA',
            variance_threshold=25.0,
            weak_pa_id='PA02'
        )
        self.assertTrue(result)
    
    def test_check_meso_weak_pa(self):
        """Test weak PA matching"""
        # Matching weak PA
        result = self.engine._check_meso_conditions(
            score=65.0,
            variance=0.26,
            weak_pa='PA02',
            score_band='MEDIO',
            variance_level='ALTA',
            variance_threshold=25.0,
            weak_pa_id='PA02'
        )
        self.assertTrue(result)
        
        # Non-matching weak PA
        result = self.engine._check_meso_conditions(
            score=65.0,
            variance=0.26,
            weak_pa='PA01',
            score_band='MEDIO',
            variance_level='ALTA',
            variance_threshold=25.0,
            weak_pa_id='PA02'
        )
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
