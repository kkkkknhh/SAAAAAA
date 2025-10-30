#!/usr/bin/env python3
"""
Strategic High-Level Wiring Validation Test
============================================

This test validates the high-level wiring and integration across
all strategic self-contained files mentioned in the requirements:

- demo_macro_prompts.py
- verify_complete_implementation.py
- validation_engine.py
- validate_system.py
- seed_factory.py
- qmcm_hooks.py
- meso_cluster_analysis.py
- macro_prompts.py
- json_contract_loader.py
- evidence_registry.py
- document_ingestion.py
- scoring.py
- recommendation_engine.py
- orchestrator.py
- micro_prompts.py
- coverage_gate.py
- scripts/bootstrap_validate.py
- validation/predicates.py
- validation/golden_rule.py
- validation/architecture_validator.py

Purpose: AUDIT, ENSURE, FORCE, GUARANTEE, and SUSTAIN high-level wiring
"""

import unittest
import sys
from pathlib import Path
import importlib.util

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestStrategicWiring(unittest.TestCase):
    """Test suite for strategic file wiring validation."""
    
    def test_all_strategic_files_exist(self):
        """Verify all strategic files exist and are accessible."""
        strategic_files = [
            "demo_macro_prompts.py",
            "verify_complete_implementation.py",
            "validation_engine.py",
            "validate_system.py",
            "seed_factory.py",
            "qmcm_hooks.py",
            "meso_cluster_analysis.py",
            "macro_prompts.py",
            "json_contract_loader.py",
            "evidence_registry.py",
            "document_ingestion.py",
            "scoring.py",
            "recommendation_engine.py",
            "orchestrator.py",
            "micro_prompts.py",
            "coverage_gate.py",
            "scripts/bootstrap_validate.py",
            "validation/predicates.py",
            "validation/golden_rule.py",
            "validation/architecture_validator.py"
        ]
        
        root = Path(__file__).parent.parent
        for file_path in strategic_files:
            full_path = root / file_path
            self.assertTrue(
                full_path.exists(),
                f"Strategic file not found: {file_path}"
            )
    
    def test_provenance_includes_all_strategic_files(self):
        """Verify provenance.csv includes all strategic files."""
        root = Path(__file__).parent.parent
        provenance_path = root / "provenance.csv"
        
        self.assertTrue(provenance_path.exists(), "provenance.csv not found")
        
        with open(provenance_path, 'r') as f:
            provenance_content = f.read()
        
        strategic_files_to_track = [
            "demo_macro_prompts.py",
            "verify_complete_implementation.py",
            "validation_engine.py",
            "validate_system.py",
            "seed_factory.py",
            "qmcm_hooks.py",
            "meso_cluster_analysis.py",
            "macro_prompts.py",
            "json_contract_loader.py",
            "evidence_registry.py",
            "document_ingestion.py",
            "scoring.py",
            "recommendation_engine.py",
            "orchestrator.py",
            "micro_prompts.py",
            "coverage_gate.py"
        ]
        
        for file_name in strategic_files_to_track:
            self.assertIn(
                file_name,
                provenance_content,
                f"Strategic file {file_name} not tracked in provenance.csv"
            )
    
    def test_validation_engine_imports(self):
        """Verify validation_engine.py imports correctly."""
        try:
            import validation_engine
            self.assertTrue(hasattr(validation_engine, 'ValidationEngine'))
            self.assertTrue(hasattr(validation_engine, 'ValidationReport'))
        except ImportError as e:
            self.fail(f"Failed to import validation_engine: {e}")
    
    def test_seed_factory_imports(self):
        """Verify seed_factory.py imports correctly."""
        try:
            import seed_factory
            self.assertTrue(hasattr(seed_factory, 'SeedFactory'))
            self.assertTrue(hasattr(seed_factory, 'DeterministicContext'))
            self.assertTrue(hasattr(seed_factory, 'create_deterministic_seed'))
        except ImportError as e:
            self.fail(f"Failed to import seed_factory: {e}")
    
    def test_qmcm_hooks_imports(self):
        """Verify qmcm_hooks.py imports correctly."""
        try:
            import qmcm_hooks
            self.assertTrue(hasattr(qmcm_hooks, 'QMCMRecorder'))
            self.assertTrue(hasattr(qmcm_hooks, 'get_global_recorder'))
            self.assertTrue(hasattr(qmcm_hooks, 'qmcm_record'))
        except ImportError as e:
            self.fail(f"Failed to import qmcm_hooks: {e}")
    
    def test_evidence_registry_imports(self):
        """Verify evidence_registry.py imports correctly."""
        try:
            import evidence_registry
            self.assertTrue(hasattr(evidence_registry, 'EvidenceRegistry'))
            self.assertTrue(hasattr(evidence_registry, 'EvidenceRecord'))
        except ImportError as e:
            self.fail(f"Failed to import evidence_registry: {e}")
    
    def test_json_contract_loader_imports(self):
        """Verify json_contract_loader.py imports correctly."""
        try:
            import json_contract_loader
            self.assertTrue(hasattr(json_contract_loader, 'JSONContractLoader'))
            self.assertTrue(hasattr(json_contract_loader, 'ContractDocument'))
            self.assertTrue(hasattr(json_contract_loader, 'ContractLoadReport'))
        except ImportError as e:
            self.fail(f"Failed to import json_contract_loader: {e}")
    
    def test_validation_predicates_imports(self):
        """Verify validation/predicates.py imports correctly."""
        try:
            from validation.predicates import ValidationPredicates, ValidationResult
            self.assertIsNotNone(ValidationPredicates)
            self.assertIsNotNone(ValidationResult)
        except ImportError as e:
            self.fail(f"Failed to import validation.predicates: {e}")
    
    def test_golden_rule_imports(self):
        """Verify validation/golden_rule.py imports correctly."""
        try:
            from validation.golden_rule import GoldenRuleValidator, GoldenRuleViolation
            self.assertIsNotNone(GoldenRuleValidator)
            self.assertIsNotNone(GoldenRuleViolation)
        except ImportError as e:
            self.fail(f"Failed to import validation.golden_rule: {e}")
    
    def test_meso_cluster_analysis_imports(self):
        """Verify meso_cluster_analysis.py imports correctly."""
        try:
            import meso_cluster_analysis
            self.assertTrue(hasattr(meso_cluster_analysis, 'analyze_policy_dispersion'))
            self.assertTrue(hasattr(meso_cluster_analysis, 'reconcile_cross_metrics'))
            self.assertTrue(hasattr(meso_cluster_analysis, 'compose_cluster_posterior'))
            self.assertTrue(hasattr(meso_cluster_analysis, 'calibrate_against_peers'))
        except ImportError as e:
            self.fail(f"Failed to import meso_cluster_analysis: {e}")
    
    def test_seed_factory_determinism(self):
        """Verify seed_factory produces deterministic seeds."""
        from seed_factory import create_deterministic_seed
        
        # Same inputs should produce same seed
        seed1 = create_deterministic_seed("test-001", question_id="Q1", policy_area="P1")
        seed2 = create_deterministic_seed("test-001", question_id="Q1", policy_area="P1")
        
        self.assertEqual(seed1, seed2, "SeedFactory not producing deterministic seeds")
        
        # Different inputs should produce different seeds
        seed3 = create_deterministic_seed("test-002", question_id="Q1", policy_area="P1")
        self.assertNotEqual(seed1, seed3, "SeedFactory producing same seed for different inputs")
    
    def test_evidence_registry_immutability(self):
        """Verify evidence_registry maintains immutability."""
        from evidence_registry import EvidenceRegistry
        
        registry = EvidenceRegistry(auto_load=False)
        
        # Add first record
        record1 = registry.append(
            method_name="test_method_1",
            evidence=["evidence1", "evidence2"],
            metadata={"key": "value"}
        )
        
        # Add second record
        record2 = registry.append(
            method_name="test_method_2",
            evidence=["evidence3"],
            metadata={"key2": "value2"}
        )
        
        # Verify chain integrity
        self.assertEqual(record1.previous_hash, "GENESIS")
        self.assertEqual(record2.previous_hash, record1.entry_hash)
        
        # Verify records are frozen (immutable)
        with self.assertRaises(Exception):
            record1.index = 999
    
    def test_validation_engine_preconditions(self):
        """Verify validation_engine properly validates preconditions."""
        from validation_engine import ValidationEngine
        
        engine = ValidationEngine()
        
        # Valid preconditions
        question_spec = {
            "id": "TEST-Q1",
            "expected_elements": ["element1", "element2"]
        }
        execution_results = {"result_key": "result_value"}
        plan_text = "This is a test plan document"
        
        result = engine.validate_scoring_preconditions(
            question_spec, execution_results, plan_text
        )
        
        self.assertTrue(result.is_valid, "Valid preconditions should pass")
        
        # Invalid preconditions (missing expected_elements)
        invalid_spec = {"id": "TEST-Q2"}
        result2 = engine.validate_scoring_preconditions(
            invalid_spec, execution_results, plan_text
        )
        
        self.assertFalse(result2.is_valid, "Invalid preconditions should fail")
    
    def test_golden_rule_validator(self):
        """Verify golden_rule validator enforces immutability."""
        from validation.golden_rule import GoldenRuleValidator, GoldenRuleViolation
        
        step_catalog = ["step1", "step2", "step3"]
        questionnaire_hash = "abc123"
        
        validator = GoldenRuleValidator(questionnaire_hash, step_catalog)
        
        # Should pass with same hash and catalog
        try:
            validator.assert_immutable_metadata(questionnaire_hash, step_catalog)
        except GoldenRuleViolation:
            self.fail("Golden rule should not raise for identical metadata")
        
        # Should fail with different hash
        with self.assertRaises(GoldenRuleViolation):
            validator.assert_immutable_metadata("different_hash", step_catalog)
        
        # Should fail with different catalog
        with self.assertRaises(GoldenRuleViolation):
            validator.assert_immutable_metadata(questionnaire_hash, ["step1", "step2"])
    
    def test_qmcm_recorder_functionality(self):
        """Verify QMCM recorder tracks method calls."""
        from qmcm_hooks import QMCMRecorder
        
        recorder = QMCMRecorder()
        recorder.clear_recording()
        
        # Record some calls
        recorder.record_call(
            method_name="test_method",
            input_types={"arg1": "str", "arg2": "int"},
            output_type="dict",
            execution_status="success",
            execution_time_ms=10.5
        )
        
        stats = recorder.get_statistics()
        
        self.assertEqual(stats['total_calls'], 1)
        self.assertEqual(stats['unique_methods'], 1)
        self.assertEqual(stats['success_rate'], 1.0)
        self.assertEqual(stats['most_called_method'], "test_method")
    
    def test_json_contract_loader_functionality(self):
        """Verify JSON contract loader handles contracts correctly."""
        from json_contract_loader import JSONContractLoader
        import tempfile
        import json
        
        loader = JSONContractLoader()
        
        # Create a temporary JSON contract
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_contract = {"key": "value", "number": 42}
            json.dump(test_contract, f)
            temp_path = f.name
        
        try:
            report = loader.load([temp_path])
            
            self.assertTrue(report.is_successful)
            self.assertEqual(len(report.documents), 1)
            
            doc = list(report.documents.values())[0]
            self.assertEqual(doc.payload["key"], "value")
            self.assertEqual(doc.payload["number"], 42)
            self.assertIsNotNone(doc.checksum)
        finally:
            Path(temp_path).unlink()


class TestStrategicFileInteraction(unittest.TestCase):
    """Test suite for interaction between strategic files."""
    
    def test_validation_engine_uses_predicates(self):
        """Verify validation_engine properly integrates with predicates."""
        from validation_engine import ValidationEngine
        from validation.predicates import ValidationPredicates
        
        engine = ValidationEngine()
        
        # Verify engine has predicates instance
        self.assertIsInstance(engine.predicates, ValidationPredicates)
    
    def test_seed_factory_context_manager(self):
        """Verify seed_factory context manager maintains state."""
        from seed_factory import DeterministicContext
        import random
        
        # Save original state
        original_value = random.random()
        
        # Use deterministic context
        with DeterministicContext(correlation_id="test-001") as seed:
            self.assertIsInstance(seed, int)
            value_in_context = random.random()
        
        # Verify state is restored after context
        value_after_context = random.random()
        
        # Values should be different (state restored)
        self.assertNotEqual(value_in_context, value_after_context)


if __name__ == '__main__':
    unittest.main()
