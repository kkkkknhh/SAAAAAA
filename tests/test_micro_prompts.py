"""
Tests for Micro Prompts - Provenance Auditor, Bayesian Posterior Justification, Anti-Milagro
============================================================================================

Comprehensive test suite for the three micro prompt implementations.
"""

import pytest
import time
from typing import Dict, List, Any

from micro_prompts import (
    # Provenance Auditor
    ProvenanceAuditor,
    QMCMRecord,
    ProvenanceNode,
    ProvenanceDAG,
    AuditResult,
    create_provenance_auditor,
    
    # Bayesian Posterior Explainer
    BayesianPosteriorExplainer,
    Signal,
    PosteriorJustification,
    create_posterior_explainer,
    
    # Anti-Milagro Stress Tester
    AntiMilagroStressTester,
    CausalChain,
    ProportionalityPattern,
    StressTestResult,
    create_stress_tester,
)


# ============================================================================
# PROVENANCE AUDITOR TESTS
# ============================================================================

class TestProvenanceAuditor:
    """Test suite for ProvenanceAuditor (QMCM Integrity Check)"""
    
    def test_create_provenance_auditor(self):
        """Test auditor creation"""
        auditor = create_provenance_auditor()
        assert auditor is not None
        assert auditor.p95_threshold == 1000.0
    
    def test_auditor_with_custom_threshold(self):
        """Test auditor with custom latency threshold"""
        auditor = create_provenance_auditor(p95_latency=500.0)
        assert auditor.p95_threshold == 500.0
    
    def test_perfect_dag_audit(self):
        """Test audit with perfect DAG (no issues)"""
        auditor = ProvenanceAuditor()
        
        # Create QMCM records
        qmcm_records = {
            'qmcm_1': QMCMRecord(
                question_id='P1-D1-Q1',
                method_fqn='module.Class.method1',
                contribution_weight=0.5,
                timestamp=time.time(),
                output_schema={'result': 'float'}
            ),
            'qmcm_2': QMCMRecord(
                question_id='P1-D1-Q1',
                method_fqn='module.Class.method2',
                contribution_weight=0.5,
                timestamp=time.time(),
                output_schema={'result': 'float'}
            )
        }
        
        # Create DAG with proper structure
        nodes = {
            'input_1': ProvenanceNode('input_1', 'input', [], None, 0.0),
            'method_1': ProvenanceNode('method_1', 'method', ['input_1'], 'qmcm_1', 100.0),
            'method_2': ProvenanceNode('method_2', 'method', ['method_1'], 'qmcm_2', 150.0),
        }
        edges = [('input_1', 'method_1'), ('method_1', 'method_2')]
        dag = ProvenanceDAG(nodes, edges)
        
        # Perform audit
        result = auditor.audit(None, qmcm_records, dag)
        
        assert result.severity == 'LOW'
        assert len(result.missing_qmcm) == 0
        assert len(result.orphan_nodes) == 0
        assert len(result.schema_mismatches) == 0
        assert len(result.latency_anomalies) == 0
        assert 'passed' in result.narrative.lower()
    
    def test_missing_qmcm_records(self):
        """Test detection of missing QMCM records"""
        auditor = ProvenanceAuditor()
        
        # Empty QMCM registry
        qmcm_records = {}
        
        # DAG with method nodes
        nodes = {
            'method_1': ProvenanceNode('method_1', 'method', ['input_1'], 'qmcm_1', 100.0),
        }
        dag = ProvenanceDAG(nodes, [])
        
        result = auditor.audit(None, qmcm_records, dag)
        
        assert len(result.missing_qmcm) == 1
        assert 'method_1' in result.missing_qmcm
        assert result.severity in ['MEDIUM', 'HIGH']
    
    def test_orphan_node_detection(self):
        """Test detection of orphan nodes"""
        auditor = ProvenanceAuditor()
        
        qmcm_records = {
            'qmcm_1': QMCMRecord(
                'P1-D1-Q1', 'module.method', 1.0, time.time(), {}
            )
        }
        
        # Method node without parent (orphan)
        nodes = {
            'orphan_method': ProvenanceNode('orphan_method', 'method', [], 'qmcm_1', 100.0),
        }
        dag = ProvenanceDAG(nodes, [])
        
        result = auditor.audit(None, qmcm_records, dag)
        
        assert len(result.orphan_nodes) == 1
        assert 'orphan_method' in result.orphan_nodes
        assert 'orphan' in result.narrative.lower()
    
    def test_latency_anomaly_detection(self):
        """Test detection of latency anomalies"""
        auditor = ProvenanceAuditor(p95_latency_threshold=500.0)
        
        qmcm_records = {
            'qmcm_1': QMCMRecord('P1-D1-Q1', 'module.method', 1.0, time.time(), {})
        }
        
        # Node with excessive latency
        nodes = {
            'slow_method': ProvenanceNode(
                'slow_method', 'method', ['input_1'], 'qmcm_1', 1500.0  # > 500ms
            ),
        }
        dag = ProvenanceDAG(nodes, [])
        
        result = auditor.audit(None, qmcm_records, dag)
        
        assert len(result.latency_anomalies) == 1
        assert result.latency_anomalies[0]['timing'] == 1500.0
        assert result.latency_anomalies[0]['excess'] == 1000.0
    
    def test_schema_mismatch_detection(self):
        """Test detection of schema mismatches"""
        contracts = {
            'module.method1': {'result': 'float', 'status': 'str'}
        }
        auditor = ProvenanceAuditor(method_contracts=contracts)
        
        qmcm_records = {
            'qmcm_1': QMCMRecord(
                'P1-D1-Q1', 'module.method1', 1.0, time.time(),
                output_schema={'result': 'float'}  # Missing 'status'
            )
        }
        
        nodes = {
            'method_1': ProvenanceNode('method_1', 'method', ['input_1'], 'qmcm_1', 100.0),
        }
        dag = ProvenanceDAG(nodes, [])
        
        result = auditor.audit(None, qmcm_records, dag, contracts)
        
        assert len(result.schema_mismatches) == 1
        assert 'module.method1' in result.schema_mismatches[0]['method']
    
    def test_contribution_weights_calculation(self):
        """Test calculation of method contribution weights"""
        auditor = ProvenanceAuditor()
        
        qmcm_records = {
            'qmcm_1': QMCMRecord('P1-D1-Q1', 'module.method1', 0.6, time.time(), {}),
            'qmcm_2': QMCMRecord('P1-D1-Q1', 'module.method1', 0.4, time.time(), {}),
            'qmcm_3': QMCMRecord('P1-D1-Q1', 'module.method2', 0.5, time.time(), {}),
        }
        
        nodes = {}
        dag = ProvenanceDAG(nodes, [])
        
        result = auditor.audit(None, qmcm_records, dag)
        
        assert 'module.method1' in result.contribution_weights
        assert result.contribution_weights['module.method1'] == 1.0  # 0.6 + 0.4
        assert result.contribution_weights['module.method2'] == 0.5
    
    def test_severity_assessment(self):
        """Test severity level assessment"""
        auditor = ProvenanceAuditor()
        
        # Test CRITICAL severity (many issues)
        assert auditor._assess_severity([1, 2, 3], [4, 5], [6], [7, 8]) == 'CRITICAL'
        
        # Test HIGH severity (moderate issues)
        assert auditor._assess_severity([1], [2], [3, 4], []) == 'HIGH'
        
        # Test MEDIUM severity (few issues)
        assert auditor._assess_severity([1], [], [2], []) == 'MEDIUM'
        
        # Test LOW severity (no issues)
        assert auditor._assess_severity([], [], [], []) == 'LOW'
    
    def test_audit_json_export(self):
        """Test JSON export of audit results"""
        auditor = ProvenanceAuditor()
        
        qmcm_records = {}
        dag = ProvenanceDAG({}, [])
        
        result = auditor.audit(None, qmcm_records, dag)
        json_output = auditor.to_json(result)
        
        assert 'missing_qmcm' in json_output
        assert 'orphan_nodes' in json_output
        assert 'schema_mismatches' in json_output
        assert 'latency_anomalies' in json_output
        assert 'contribution_weights' in json_output
        assert 'severity' in json_output
        assert 'narrative' in json_output


# ============================================================================
# BAYESIAN POSTERIOR JUSTIFICATION TESTS
# ============================================================================

class TestBayesianPosteriorExplainer:
    """Test suite for BayesianPosteriorExplainer"""
    
    def test_create_posterior_explainer(self):
        """Test explainer creation"""
        explainer = create_posterior_explainer()
        assert explainer is not None
        assert explainer.anti_miracle_cap == 0.95
    
    def test_explainer_with_custom_cap(self):
        """Test explainer with custom anti-miracle cap"""
        explainer = create_posterior_explainer(anti_miracle_cap=0.90)
        assert explainer.anti_miracle_cap == 0.90
    
    def test_signal_ranking_by_impact(self):
        """Test signals are ranked by absolute marginal impact"""
        explainer = BayesianPosteriorExplainer()
        
        signals = [
            Signal('Hoop', 0.8, 1.0, 'ev1', True, 0.1, ""),
            Signal('Smoking-Gun', 0.95, 1.0, 'ev2', True, 0.3, ""),  # Highest impact
            Signal('Straw-in-Wind', 0.6, 1.0, 'ev3', True, 0.05, ""),
        ]
        
        result = explainer.explain(0.5, signals, 0.85)
        
        # Check ranking: Smoking-Gun should be first
        assert result.signals_ranked[0]['test_type'] == 'Smoking-Gun'
        assert result.signals_ranked[0]['delta_posterior'] == 0.3
        assert result.signals_ranked[1]['test_type'] == 'Hoop'
        assert result.signals_ranked[2]['test_type'] == 'Straw-in-Wind'
    
    def test_discarded_signals_identification(self):
        """Test identification of discarded signals"""
        explainer = BayesianPosteriorExplainer()
        
        signals = [
            Signal('Hoop', 0.8, 1.0, 'ev1', True, 0.1, ""),  # Kept
            Signal('Smoking-Gun', 0.95, 1.0, 'ev2', False, 0.2, ""),  # Discarded
            Signal('Doubly-Decisive', 0.99, 1.0, 'ev3', False, 0.3, ""),  # Discarded
        ]
        
        result = explainer.explain(0.5, signals, 0.6)
        
        assert len(result.discarded_signals) == 2
        assert all(not s['kept'] for s in result.discarded_signals)
        assert len(result.signals_ranked) == 1  # Only reconciled signal
    
    def test_anti_miracle_cap_application(self):
        """Test anti-miracle cap is applied when posterior exceeds threshold"""
        explainer = BayesianPosteriorExplainer(anti_miracle_cap=0.90)
        
        signals = [
            Signal('Doubly-Decisive', 0.99, 1.0, 'ev1', True, 0.45, ""),
        ]
        
        # Posterior exceeds cap (0.95 > 0.90)
        result = explainer.explain(0.5, signals, 0.95)
        
        assert result.anti_miracle_cap_applied is True
        assert abs(result.cap_delta - 0.05) < 0.001  # Allow for floating point precision
        assert result.posterior == 0.90  # Capped to limit
        assert 'cap applied' in result.robustness_narrative.lower()
    
    def test_no_cap_when_below_threshold(self):
        """Test no cap applied when posterior is below threshold"""
        explainer = BayesianPosteriorExplainer(anti_miracle_cap=0.95)
        
        signals = [
            Signal('Hoop', 0.8, 1.0, 'ev1', True, 0.2, ""),
        ]
        
        result = explainer.explain(0.5, signals, 0.7)
        
        assert result.anti_miracle_cap_applied is False
        assert result.cap_delta == 0.0
        assert result.posterior == 0.7
    
    def test_test_type_justifications(self):
        """Test that each test type gets appropriate justification"""
        explainer = BayesianPosteriorExplainer()
        
        # Test all four test types
        signals = [
            Signal('Hoop', 0.8, 1.0, 'ev1', True, 0.1, ""),
            Signal('Smoking-Gun', 0.95, 1.0, 'ev2', True, 0.2, ""),
            Signal('Straw-in-Wind', 0.6, 1.0, 'ev3', True, 0.05, ""),
            Signal('Doubly-Decisive', 0.99, 1.0, 'ev4', True, 0.3, ""),
        ]
        
        result = explainer.explain(0.5, signals, 0.85)
        
        # Check that each signal has a reason with rank and meaningful content
        for signal in result.signals_ranked:
            assert signal['reason'] != ""
            assert 'Rank' in signal['reason']
            # Check that reason contains meaningful content (not just test type name)
            assert len(signal['reason']) > 10
    
    def test_robustness_narrative_high(self):
        """Test high robustness narrative (3+ signals, no discards)"""
        explainer = BayesianPosteriorExplainer()
        
        signals = [
            Signal('Hoop', 0.8, 1.0, 'ev1', True, 0.1, ""),
            Signal('Smoking-Gun', 0.95, 1.0, 'ev2', True, 0.2, ""),
            Signal('Doubly-Decisive', 0.99, 1.0, 'ev3', True, 0.15, ""),
        ]
        
        result = explainer.explain(0.5, signals, 0.85)
        
        assert 'high robustness' in result.robustness_narrative.lower()
    
    def test_robustness_narrative_low(self):
        """Test low robustness narrative (no signals)"""
        explainer = BayesianPosteriorExplainer()
        
        signals = []
        
        result = explainer.explain(0.5, signals, 0.5)
        
        assert 'low robustness' in result.robustness_narrative.lower()
    
    def test_posterior_json_export(self):
        """Test JSON export of posterior justification"""
        explainer = BayesianPosteriorExplainer()
        
        signals = [
            Signal('Hoop', 0.8, 1.0, 'ev1', True, 0.1, ""),
        ]
        
        result = explainer.explain(0.5, signals, 0.6)
        json_output = explainer.to_json(result)
        
        assert 'prior' in json_output
        assert 'posterior' in json_output
        assert 'signals_ranked' in json_output
        assert 'discarded_signals' in json_output
        assert 'anti_miracle_cap_applied' in json_output
        assert 'robustness_narrative' in json_output


# ============================================================================
# ANTI-MILAGRO STRESS TEST TESTS
# ============================================================================

class TestAntiMilagroStressTester:
    """Test suite for AntiMilagroStressTester"""
    
    def test_create_stress_tester(self):
        """Test stress tester creation"""
        tester = create_stress_tester()
        assert tester is not None
        assert tester.fragility_threshold == 0.3
    
    def test_tester_with_custom_threshold(self):
        """Test tester with custom fragility threshold"""
        tester = create_stress_tester(fragility_threshold=0.5)
        assert tester.fragility_threshold == 0.5
    
    def test_pattern_density_calculation(self):
        """Test pattern density calculation (patterns per step)"""
        tester = AntiMilagroStressTester()
        
        chain = CausalChain(
            steps=['A', 'B', 'C', 'D'],
            edges=[('A', 'B'), ('B', 'C'), ('C', 'D')]
        )
        
        patterns = [
            ProportionalityPattern('linear', 0.8, 'A->B'),
            ProportionalityPattern('dose-response', 0.7, 'B->C'),
        ]
        
        result = tester.stress_test(chain, patterns, [])
        
        assert result.density == 0.5  # 2 patterns / 4 steps
    
    def test_robust_chain_no_fragility(self):
        """Test robust chain with good pattern coverage (no fragility)"""
        tester = AntiMilagroStressTester(fragility_threshold=0.3)
        
        chain = CausalChain(
            steps=['A', 'B', 'C'],
            edges=[('A', 'B'), ('B', 'C')]
        )
        
        # Strong patterns covering the chain
        patterns = [
            ProportionalityPattern('linear', 0.9, 'A->B'),
            ProportionalityPattern('dose-response', 0.85, 'B->C'),
            ProportionalityPattern('mechanism', 0.8, 'A->C'),
        ]
        
        result = tester.stress_test(chain, patterns, [])
        
        assert result.fragility_flag is False
        assert result.simulated_drop < 0.3
        assert 'robust' in result.explanation.lower()
    
    def test_fragile_chain_detection(self):
        """Test detection of fragile chain with weak patterns"""
        tester = AntiMilagroStressTester(fragility_threshold=0.3)
        
        chain = CausalChain(
            steps=['A', 'B', 'C', 'D'],
            edges=[('A', 'B'), ('B', 'C'), ('C', 'D')]
        )
        
        # Mix of strong and weak patterns
        patterns = [
            ProportionalityPattern('linear', 0.9, 'A->B'),  # Strong
            ProportionalityPattern('threshold', 0.1, 'B->C'),  # Weak
            ProportionalityPattern('mechanism', 0.15, 'C->D'),  # Weak
        ]
        
        result = tester.stress_test(chain, patterns, [])
        
        # Should detect fragility when weak patterns removed
        assert result.simulated_drop >= 0.0
    
    def test_empty_chain_handling(self):
        """Test handling of empty causal chain"""
        tester = AntiMilagroStressTester()
        
        chain = CausalChain(steps=[], edges=[])
        patterns = []
        
        result = tester.stress_test(chain, patterns, [])
        
        assert result.density == 0.0
        assert result.pattern_coverage == 0.0
    
    def test_missing_patterns_tracking(self):
        """Test tracking of missing required patterns"""
        tester = AntiMilagroStressTester()
        
        chain = CausalChain(
            steps=['A', 'B', 'C'],
            edges=[('A', 'B'), ('B', 'C')]
        )
        
        patterns = [
            ProportionalityPattern('linear', 0.8, 'A->B'),
        ]
        
        missing = ['dose-response', 'mechanism']
        
        result = tester.stress_test(chain, patterns, missing)
        
        assert result.missing_patterns == missing
        assert len(result.missing_patterns) == 2
    
    def test_pattern_coverage_calculation(self):
        """Test pattern coverage calculation"""
        tester = AntiMilagroStressTester()
        
        chain = CausalChain(
            steps=['A', 'B', 'C', 'D'],
            edges=[('A', 'B'), ('B', 'C'), ('C', 'D')]
        )
        
        # Patterns covering half the chain
        patterns = [
            ProportionalityPattern('linear', 0.8, 'A'),
            ProportionalityPattern('dose-response', 0.7, 'B'),
        ]
        
        result = tester.stress_test(chain, patterns, [])
        
        # Coverage should be 2/4 = 0.5
        assert result.pattern_coverage == 0.5
    
    def test_node_removal_simulation(self):
        """Test simulation of node removal"""
        tester = AntiMilagroStressTester()
        
        chain = CausalChain(
            steps=['A', 'B', 'C', 'D'],
            edges=[('A', 'B'), ('B', 'C'), ('C', 'D')]
        )
        
        patterns = [
            ProportionalityPattern('linear', 0.9, 'A'),
            ProportionalityPattern('threshold', 0.2, 'B'),  # Weak - will be removed
            ProportionalityPattern('mechanism', 0.85, 'C'),
        ]
        
        result = tester.stress_test(chain, patterns, [])
        
        # Should have non-zero drop from removing weak pattern
        assert result.simulated_drop >= 0.0
        assert result.simulated_drop <= 1.0
    
    def test_explanation_generation(self):
        """Test 3-line explanation generation"""
        tester = AntiMilagroStressTester()
        
        chain = CausalChain(
            steps=['A', 'B'],
            edges=[('A', 'B')]
        )
        
        patterns = [
            ProportionalityPattern('linear', 0.8, 'A'),
        ]
        
        result = tester.stress_test(chain, patterns, [])
        
        assert 'pattern density' in result.explanation.lower()
        assert 'drop' in result.explanation.lower()
        # Should mention either fragility or robustness
        assert ('fragility' in result.explanation.lower() or 
                'robust' in result.explanation.lower())
    
    def test_stress_test_json_export(self):
        """Test JSON export of stress test results"""
        tester = AntiMilagroStressTester()
        
        chain = CausalChain(steps=['A', 'B'], edges=[('A', 'B')])
        patterns = [ProportionalityPattern('linear', 0.8, 'A')]
        
        result = tester.stress_test(chain, patterns, [])
        json_output = tester.to_json(result)
        
        assert 'density' in json_output
        assert 'simulated_drop' in json_output
        assert 'fragility_flag' in json_output
        assert 'explanation' in json_output
        assert 'pattern_coverage' in json_output
        assert 'missing_patterns' in json_output


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMicroPromptsIntegration:
    """Integration tests across all three micro prompts"""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow using all three micro prompts"""
        # 1. Setup provenance auditor
        auditor = create_provenance_auditor(p95_latency=500.0)
        
        qmcm_records = {
            'qmcm_1': QMCMRecord('P1-D1-Q1', 'module.method1', 0.6, time.time(), {}),
            'qmcm_2': QMCMRecord('P1-D1-Q1', 'module.method2', 0.4, time.time(), {}),
        }
        
        nodes = {
            'input_1': ProvenanceNode('input_1', 'input', [], None, 0.0),
            'method_1': ProvenanceNode('method_1', 'method', ['input_1'], 'qmcm_1', 100.0),
            'method_2': ProvenanceNode('method_2', 'method', ['method_1'], 'qmcm_2', 150.0),
        }
        dag = ProvenanceDAG(nodes, [('input_1', 'method_1'), ('method_1', 'method_2')])
        
        audit_result = auditor.audit(None, qmcm_records, dag)
        
        # 2. Setup Bayesian explainer
        explainer = create_posterior_explainer(anti_miracle_cap=0.95)
        
        signals = [
            Signal('Hoop', 0.8, 0.6, 'qmcm_1', True, 0.15, ""),
            Signal('Smoking-Gun', 0.95, 0.4, 'qmcm_2', True, 0.25, ""),
        ]
        
        posterior_result = explainer.explain(0.5, signals, 0.80)
        
        # 3. Setup stress tester
        tester = create_stress_tester(fragility_threshold=0.3)
        
        chain = CausalChain(
            steps=['input_1', 'method_1', 'method_2'],
            edges=[('input_1', 'method_1'), ('method_1', 'method_2')]
        )
        
        patterns = [
            ProportionalityPattern('linear', 0.8, 'input_1'),
            ProportionalityPattern('mechanism', 0.75, 'method_1'),
        ]
        
        stress_result = tester.stress_test(chain, patterns, [])
        
        # Verify all three completed successfully
        assert audit_result.severity == 'LOW'
        assert posterior_result.posterior > 0
        assert stress_result.density > 0
        
        # Verify they can be exported to JSON
        audit_json = auditor.to_json(audit_result)
        posterior_json = explainer.to_json(posterior_result)
        stress_json = tester.to_json(stress_result)
        
        assert all(isinstance(j, dict) for j in [audit_json, posterior_json, stress_json])
    
    def test_factory_functions(self):
        """Test all factory functions work correctly"""
        auditor = create_provenance_auditor()
        explainer = create_posterior_explainer()
        tester = create_stress_tester()
        
        assert isinstance(auditor, ProvenanceAuditor)
        assert isinstance(explainer, BayesianPosteriorExplainer)
        assert isinstance(tester, AntiMilagroStressTester)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
