"""Data flow executors for the 30 base questions (D1Q1 through D6Q5).

Each executor implements a specific question's data flow pipeline using
the MethodExecutor to invoke methods from the catalog. These executors
were extracted from the ORCHESTRATOR_MONILITH.py file.
"""

from typing import Any, Dict


class DataFlowExecutor:
    """Ejecutor base"""
    def __init__(self, method_executor):
        self.executor = method_executor


class D1Q1_Executor(DataFlowExecutor):
    """
    D1-Q1: Líneas Base y Brechas Cuantificadas
    Flow: PP.O → CD.E+T → CD.V → CD.C → A1.C || EP.C → PP.R
    Métodos: 18
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. PP.C - BayesianEvidenceScorer._calculate_shannon_entropy (P=2)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            '_calculate_shannon_entropy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer._calculate_shannon_entropy'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.C - SemanticAnalyzer._calculate_semantic_complexity (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_calculate_semantic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._calculate_semantic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 16. A1.V - SemanticAnalyzer._classify_policy_domain (P=1)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        # 17. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.V - BayesianNumericalAnalyzer._classify_evidence_strength (P=2)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            '_classify_evidence_strength',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer._classify_evidence_strength'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D1Q2_Executor(DataFlowExecutor):
    """
    D1-Q2: Normalización y Fuentes
    Flow: PP.E → PP.T → CD.E+T → CD.V+C → EP.E+C
    Métodos: 12
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.T - IndustrialPolicyProcessor._compile_pattern_registry (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_compile_pattern_registry',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._compile_pattern_registry'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.normalize_unicode (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'normalize_unicode',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.normalize_unicode'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 11. EP.E - PolicyAnalysisEmbedder._extract_numerical_values (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_extract_numerical_values',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._extract_numerical_values'] = result
        if result is not None:
            current_data = result
        
        # 12. EP.C - BayesianNumericalAnalyzer._compute_coherence (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            '_compute_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer._compute_coherence'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D1Q3_Executor(DataFlowExecutor):
    """
    D1-Q3: Asignación de Recursos
    Flow: PP.O → CD.E → CD.V+C → FV.E → DB.O → EP.C → PP.R
    Métodos: 22
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.E - IndustrialPolicyProcessor._extract_point_evidence (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_extract_point_evidence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._extract_point_evidence'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - PolicyContradictionDetector._detect_resource_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_resource_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_resource_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._are_conflicting_allocations (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_conflicting_allocations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_conflicting_allocations'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.E - TemporalLogicVerifier._extract_resources (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_extract_resources',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._extract_resources'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 17. FV.E - PDETMunicipalPlanAnalyzer._extract_financial_amounts (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_financial_amounts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_financial_amounts'] = result
        if result is not None:
            current_data = result
        
        # 18. FV.V - PDETMunicipalPlanAnalyzer._identify_funding_source (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_identify_funding_source',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._identify_funding_source'] = result
        if result is not None:
            current_data = result
        
        # 19. FV.C - PDETMunicipalPlanAnalyzer._analyze_funding_sources (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_analyze_funding_sources',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._analyze_funding_sources'] = result
        if result is not None:
            current_data = result
        
        # 20. DB.O - FinancialAuditor.trace_financial_allocation (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            'trace_financial_allocation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor.trace_financial_allocation'] = result
        if result is not None:
            current_data = result
        
        # 21. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 22. EP.C - BayesianNumericalAnalyzer.compare_policies (P=2)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'compare_policies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.compare_policies'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D1Q4_Executor(DataFlowExecutor):
    """
    D1-Q4: Capacidad Institucional
    Flow: PP.E → CD.E+T+V+C → A1.V → FV.E+V
    Métodos: 16
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.T - IndustrialPolicyProcessor._build_point_patterns (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_build_point_patterns',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._build_point_patterns'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.C - PolicyContradictionDetector._calculate_graph_fragmentation (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_graph_fragmentation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_graph_fragmentation'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._calculate_syntactic_complexity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_syntactic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_syntactic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 12. A1.V - SemanticAnalyzer._classify_value_chain_link (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_value_chain_link',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_value_chain_link'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.V - PerformanceAnalyzer._detect_bottlenecks (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_detect_bottlenecks',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._detect_bottlenecks'] = result
        if result is not None:
            current_data = result
        
        # 14. A1.E - TextMiningEngine._identify_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_identify_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._identify_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.E - PDETMunicipalPlanAnalyzer.identify_responsible_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'identify_responsible_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.identify_responsible_entities'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.V - PDETMunicipalPlanAnalyzer._classify_entity_type (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_entity_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_entity_type'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D1Q5_Executor(DataFlowExecutor):
    """
    D1-Q5: Restricciones Temporales
    Flow: PP.E+T → CD.E → CD.V+T+C → A1.C
    Métodos: 14
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 4. CD.V - PolicyContradictionDetector._detect_temporal_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_temporal_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_temporal_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - TemporalLogicVerifier.verify_temporal_consistency (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            'verify_temporal_consistency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier.verify_temporal_consistency'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - TemporalLogicVerifier._build_timeline (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_build_timeline',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._build_timeline'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.T - TemporalLogicVerifier._parse_temporal_marker (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_parse_temporal_marker',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._parse_temporal_marker'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - TemporalLogicVerifier._has_temporal_conflict (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_has_temporal_conflict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._has_temporal_conflict'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - TemporalLogicVerifier._check_deadline_constraints (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_check_deadline_constraints',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._check_deadline_constraints'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - TemporalLogicVerifier._classify_temporal_type (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_classify_temporal_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._classify_temporal_type'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.C - SemanticAnalyzer._calculate_semantic_complexity (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_calculate_semantic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._calculate_semantic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 14. A1.C - PerformanceAnalyzer._calculate_throughput_metrics (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_calculate_throughput_metrics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._calculate_throughput_metrics'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q1_Executor(DataFlowExecutor):
    """
    D2-Q1: Formato Tabular y Trazabilidad
    Flow: PP.O → FV.E → FV.T+V → CD.V → SC.E
    Métodos: 20
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.T - PDETMunicipalPlanAnalyzer._clean_dataframe (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_clean_dataframe',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._clean_dataframe'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.V - PDETMunicipalPlanAnalyzer._is_likely_header (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_is_likely_header',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._is_likely_header'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.T - PDETMunicipalPlanAnalyzer._deduplicate_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_deduplicate_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._deduplicate_tables'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.T - PDETMunicipalPlanAnalyzer._reconstruct_fragmented_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_reconstruct_fragmented_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._reconstruct_fragmented_tables'] = result
        if result is not None:
            current_data = result
        
        # 10. FV.V - PDETMunicipalPlanAnalyzer._classify_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_tables'] = result
        if result is not None:
            current_data = result
        
        # 11. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 12. FV.E - PDETMunicipalPlanAnalyzer._extract_from_budget_table (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_budget_table',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_budget_table'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.E - PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_responsibility_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.E - PDETMunicipalPlanAnalyzer.identify_responsible_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'identify_responsible_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.identify_responsible_entities'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.T - PDETMunicipalPlanAnalyzer._consolidate_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_consolidate_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._consolidate_entities'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.C - PDETMunicipalPlanAnalyzer._score_entity_specificity (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_score_entity_specificity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._score_entity_specificity'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.T - TemporalLogicVerifier._build_timeline (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_build_timeline',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._build_timeline'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.V - TemporalLogicVerifier._check_deadline_constraints (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_check_deadline_constraints',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._check_deadline_constraints'] = result
        if result is not None:
            current_data = result
        
        # 19. CD.V - PolicyContradictionDetector._detect_temporal_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_temporal_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_temporal_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 20. SC.E - SemanticProcessor._detect_table (P=3)
        result = self.executor.execute(
            'SemanticProcessor',
            '_detect_table',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticProcessor._detect_table'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q2_Executor(DataFlowExecutor):
    """
    D2-Q2: Causalidad de Actividades
    Flow: PP.E → PP.C → CD.E+T+V+C → DB.O → TC.T+V → A1.V
    Métodos: 25
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.E - CausalExtractor._extract_goals (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_goals',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_goals'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.E - CausalExtractor._extract_goal_text (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_goal_text',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_goal_text'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.V - CausalExtractor._classify_goal_type (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_classify_goal_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._classify_goal_type'] = result
        if result is not None:
            current_data = result
        
        # 18. DB.T - CausalExtractor._add_node_to_graph (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_add_node_to_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._add_node_to_graph'] = result
        if result is not None:
            current_data = result
        
        # 19. DB.E - CausalExtractor._extract_causal_links (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_links'] = result
        if result is not None:
            current_data = result
        
        # 20. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 21. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 22. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 23. A1.C - TextMiningEngine._analyze_link_text (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_analyze_link_text',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._analyze_link_text'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q3_Executor(DataFlowExecutor):
    """
    D2-Q3: Responsables de Actividades
    Flow: PP.O → FV.E+T+V+C → CD.V → EP.E
    Métodos: 18
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.identify_responsible_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'identify_responsible_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.identify_responsible_entities'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.E - PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_responsibility_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.T - PDETMunicipalPlanAnalyzer._consolidate_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_consolidate_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._consolidate_entities'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.V - PDETMunicipalPlanAnalyzer._classify_entity_type (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_entity_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_entity_type'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.C - PDETMunicipalPlanAnalyzer._score_entity_specificity (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_score_entity_specificity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._score_entity_specificity'] = result
        if result is not None:
            current_data = result
        
        # 10. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 11. FV.T - PDETMunicipalPlanAnalyzer._clean_dataframe (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_clean_dataframe',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._clean_dataframe'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 14. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q4_Executor(DataFlowExecutor):
    """
    D2-Q4: Cuantificación de Actividades
    Flow: PP.O → FV.E → CD.E+T+V+C → EP.C
    Métodos: 21
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.E - PDETMunicipalPlanAnalyzer._extract_financial_amounts (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_financial_amounts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_financial_amounts'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.E - PDETMunicipalPlanAnalyzer._extract_from_budget_table (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_budget_table',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_budget_table'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 16. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D2Q5_Executor(DataFlowExecutor):
    """
    D2-Q5: Eslabón Causal Diagnóstico-Actividades
    Flow: PP.E+C → CD.E+T+V+C → DB.O → TC.T+V → A1.V
    Métodos: 23
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 15. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 16. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 17. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 18. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 19. A1.C - TextMiningEngine._analyze_link_text (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_analyze_link_text',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._analyze_link_text'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q1_Executor(DataFlowExecutor):
    """
    D3-Q1: Indicadores de Producto
    Flow: PP.O → CD.E+T+V → FV.E+T+V → EP.E+C → PP.R
    Métodos: 19
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 12. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.T - PDETMunicipalPlanAnalyzer._indicator_to_dict (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_indicator_to_dict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._indicator_to_dict'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.E - PDETMunicipalPlanAnalyzer._find_product_mentions (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_find_product_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._find_product_mentions'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.V - PDETMunicipalPlanAnalyzer._classify_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_tables'] = result
        if result is not None:
            current_data = result
        
        # 17. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.E - PolicyAnalysisEmbedder._extract_numerical_values (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_extract_numerical_values',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._extract_numerical_values'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q2_Executor(DataFlowExecutor):
    """
    D3-Q2: Cuantificación de Productos
    Flow: PP.O → FV.E → CD.E+T+V+C → EP.C
    Métodos: 20
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.E - PDETMunicipalPlanAnalyzer._extract_financial_amounts (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_financial_amounts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_financial_amounts'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.E - PDETMunicipalPlanAnalyzer._extract_from_budget_table (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_budget_table',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_budget_table'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.E - PDETMunicipalPlanAnalyzer._find_product_mentions (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_find_product_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._find_product_mentions'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 16. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 19. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q3_Executor(DataFlowExecutor):
    """
    D3-Q3: Responsables de Productos
    Flow: PP.O → FV.E+T+V+C → CD.V+T → EP.E
    Métodos: 17
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.E - PDETMunicipalPlanAnalyzer.identify_responsible_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'identify_responsible_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.identify_responsible_entities'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.E - PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_extract_from_responsibility_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._extract_from_responsibility_tables'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.T - PDETMunicipalPlanAnalyzer._consolidate_entities (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_consolidate_entities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._consolidate_entities'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.V - PDETMunicipalPlanAnalyzer._classify_entity_type (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_entity_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_entity_type'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.C - PDETMunicipalPlanAnalyzer._score_entity_specificity (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_score_entity_specificity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._score_entity_specificity'] = result
        if result is not None:
            current_data = result
        
        # 10. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 14. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q4_Executor(DataFlowExecutor):
    """
    D3-Q4: Plazos de Productos
    Flow: PP.E+T → CD.E → CD.V+T+C → A1.C+V
    Métodos: 19
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - TemporalLogicVerifier.verify_temporal_consistency (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            'verify_temporal_consistency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier.verify_temporal_consistency'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - TemporalLogicVerifier._check_deadline_constraints (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_check_deadline_constraints',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._check_deadline_constraints'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - TemporalLogicVerifier._classify_temporal_type (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_classify_temporal_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._classify_temporal_type'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - TemporalLogicVerifier._build_timeline (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_build_timeline',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._build_timeline'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.T - TemporalLogicVerifier._parse_temporal_marker (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_parse_temporal_marker',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._parse_temporal_marker'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - TemporalLogicVerifier._has_temporal_conflict (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_has_temporal_conflict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._has_temporal_conflict'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - TemporalLogicVerifier._extract_resources (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_extract_resources',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._extract_resources'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.V - PolicyContradictionDetector._detect_resource_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_resource_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_resource_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.C - PerformanceAnalyzer._calculate_throughput_metrics (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_calculate_throughput_metrics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._calculate_throughput_metrics'] = result
        if result is not None:
            current_data = result
        
        # 16. A1.V - PerformanceAnalyzer._detect_bottlenecks (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_detect_bottlenecks',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._detect_bottlenecks'] = result
        if result is not None:
            current_data = result
        
        # 17. A1.V - TextMiningEngine._assess_risks (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_assess_risks',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._assess_risks'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D3Q5_Executor(DataFlowExecutor):
    """
    D3-Q5: Eslabón Causal Producto-Resultado
    Flow: PP.E+C → CD.E+T+V+C → DB.O (Extractor+Mechanism+Tests) → TC.T+V → A1.V
    Métodos: 26
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.E - CausalExtractor._extract_causal_links (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_links'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.E - CausalExtractor._extract_causal_justifications (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_justifications',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_justifications'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.C - CausalExtractor._calculate_confidence (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_calculate_confidence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._calculate_confidence'] = result
        if result is not None:
            current_data = result
        
        # 18. DB.E - MechanismPartExtractor.extract_entity_activity (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            'extract_entity_activity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor.extract_entity_activity'] = result
        if result is not None:
            current_data = result
        
        # 19. DB.E - MechanismPartExtractor._find_subject_entity (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            '_find_subject_entity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor._find_subject_entity'] = result
        if result is not None:
            current_data = result
        
        # 20. DB.E - MechanismPartExtractor._find_action_verb (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            '_find_action_verb',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor._find_action_verb'] = result
        if result is not None:
            current_data = result
        
        # 21. DB.V - MechanismPartExtractor._validate_entity_activity (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            '_validate_entity_activity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor._validate_entity_activity'] = result
        if result is not None:
            current_data = result
        
        # 22. DB.C - MechanismPartExtractor._calculate_ea_confidence (P=3)
        result = self.executor.execute(
            'MechanismPartExtractor',
            '_calculate_ea_confidence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismPartExtractor._calculate_ea_confidence'] = result
        if result is not None:
            current_data = result
        
        # 23. DB.O - BayesianMechanismInference.infer_mechanisms (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            'infer_mechanisms',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference.infer_mechanisms'] = result
        if result is not None:
            current_data = result
        
        # 24. DB.T - BayesianMechanismInference._build_transition_matrix (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_build_transition_matrix',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._build_transition_matrix'] = result
        if result is not None:
            current_data = result
        
        # 25. DB.V - BayesianMechanismInference._infer_activity_sequence (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_infer_activity_sequence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._infer_activity_sequence'] = result
        if result is not None:
            current_data = result
        
        # 26. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 27. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 28. DB.V - BayesianMechanismInference._classify_mechanism_type (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_classify_mechanism_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._classify_mechanism_type'] = result
        if result is not None:
            current_data = result
        
        # 29. DB.V - BeachEvidentialTest.apply_test_logic (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'apply_test_logic',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.apply_test_logic'] = result
        if result is not None:
            current_data = result
        
        # 30. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 31. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 32. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 33. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 34. A1.C - TextMiningEngine._analyze_link_text (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_analyze_link_text',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._analyze_link_text'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q1_Executor(DataFlowExecutor):
    """
    D4-Q1: Indicadores de Resultado
    Flow: PP.O → CD.E+T+V → FV.E+T+V → EP.E+C → PP.R
    Métodos: 18
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 12. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.T - PDETMunicipalPlanAnalyzer._indicator_to_dict (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_indicator_to_dict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._indicator_to_dict'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.E - PDETMunicipalPlanAnalyzer._find_outcome_mentions (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_find_outcome_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._find_outcome_mentions'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.V - PDETMunicipalPlanAnalyzer._classify_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_tables'] = result
        if result is not None:
            current_data = result
        
        # 17. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.E - PolicyAnalysisEmbedder._extract_numerical_values (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_extract_numerical_values',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._extract_numerical_values'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q2_Executor(DataFlowExecutor):
    """
    D4-Q2: Cadena Causal y Supuestos
    Flow: PP.E+C → CD.E+T+V+C → DB.O (Extractor+Tests) → TC.T+V
    Métodos: 24
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.C - PolicyContradictionDetector._calculate_syntactic_complexity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_syntactic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_syntactic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.E - CausalExtractor._extract_causal_links (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_links'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.O - BayesianMechanismInference.infer_mechanisms (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            'infer_mechanisms',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference.infer_mechanisms'] = result
        if result is not None:
            current_data = result
        
        # 18. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 19. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 20. DB.V - BeachEvidentialTest.classify_test (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'classify_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.classify_test'] = result
        if result is not None:
            current_data = result
        
        # 21. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 22. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 23. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 24. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q3_Executor(DataFlowExecutor):
    """
    D4-Q3: Justificación de Ambición
    Flow: PP.O+C → CD.E+V+C → FV.C+R → DB.C → EP.C+V
    Métodos: 20
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer._calculate_shannon_entropy (P=2)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            '_calculate_shannon_entropy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer._calculate_shannon_entropy'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.C - PolicyContradictionDetector._calculate_objective_alignment (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_objective_alignment',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_objective_alignment'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.R - PDETMunicipalPlanAnalyzer.generate_recommendations (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'generate_recommendations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.generate_recommendations'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.C - PDETMunicipalPlanAnalyzer.analyze_financial_feasibility (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_financial_feasibility',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_financial_feasibility'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.C - PDETMunicipalPlanAnalyzer._assess_financial_sustainability (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_assess_financial_sustainability',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._assess_financial_sustainability'] = result
        if result is not None:
            current_data = result
        
        # 16. FV.C - PDETMunicipalPlanAnalyzer._bayesian_risk_inference (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_bayesian_risk_inference',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._bayesian_risk_inference'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.C - FinancialAuditor._calculate_sufficiency (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_calculate_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._calculate_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        # 19. EP.C - BayesianNumericalAnalyzer.compare_policies (P=2)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'compare_policies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.compare_policies'] = result
        if result is not None:
            current_data = result
        
        # 20. EP.V - BayesianNumericalAnalyzer._classify_evidence_strength (P=2)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            '_classify_evidence_strength',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer._classify_evidence_strength'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q4_Executor(DataFlowExecutor):
    """
    D4-Q4: Población Objetivo
    Flow: PP.O → CD.E+T+V+C → A1.V+E → EP.E+V
    Métodos: 15
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 11. A1.V - SemanticAnalyzer._classify_cross_cutting_themes (P=3)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_cross_cutting_themes',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_cross_cutting_themes'] = result
        if result is not None:
            current_data = result
        
        # 12. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.E - SemanticAnalyzer.extract_semantic_cube (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            'extract_semantic_cube',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer.extract_semantic_cube'] = result
        if result is not None:
            current_data = result
        
        # 14. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 15. EP.V - PolicyAnalysisEmbedder._filter_by_pdq (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_filter_by_pdq',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._filter_by_pdq'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D4Q5_Executor(DataFlowExecutor):
    """
    D4-Q5: Alineación con Objetivos Superiores
    Flow: PP.O → CD.C+T → A1.V+E → EP.E+C
    Métodos: 17
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.C - PolicyContradictionDetector._calculate_objective_alignment (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_objective_alignment',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_objective_alignment'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 11. A1.V - SemanticAnalyzer._classify_cross_cutting_themes (P=3)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_cross_cutting_themes',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_cross_cutting_themes'] = result
        if result is not None:
            current_data = result
        
        # 12. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.E - SemanticAnalyzer.extract_semantic_cube (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            'extract_semantic_cube',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer.extract_semantic_cube'] = result
        if result is not None:
            current_data = result
        
        # 14. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 15. EP.C - PolicyAnalysisEmbedder.compare_policy_interventions (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'compare_policy_interventions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.compare_policy_interventions'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q1_Executor(DataFlowExecutor):
    """
    D5-Q1: Indicadores de Impacto
    Flow: PP.O → CD.E+T+V → FV.E+T+V → EP.C → PP.R
    Métodos: 17
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.R - IndustrialPolicyProcessor._construct_evidence_bundle (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_construct_evidence_bundle',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._construct_evidence_bundle'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 12. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 13. FV.T - PDETMunicipalPlanAnalyzer._indicator_to_dict (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_indicator_to_dict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._indicator_to_dict'] = result
        if result is not None:
            current_data = result
        
        # 14. FV.O - PDETMunicipalPlanAnalyzer.analyze_municipal_plan (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_municipal_plan',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_municipal_plan'] = result
        if result is not None:
            current_data = result
        
        # 15. FV.V - PDETMunicipalPlanAnalyzer._classify_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_classify_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._classify_tables'] = result
        if result is not None:
            current_data = result
        
        # 16. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q2_Executor(DataFlowExecutor):
    """
    D5-Q2: Eslabón Causal Resultado-Impacto
    Flow: PP.E+C → CD.E+T+V+C → DB.O (Extractor+Inference+Tests) → TC.T+V
    Métodos: 25
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._extract_policy_statements (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_policy_statements',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_policy_statements'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.E - CausalExtractor._extract_causal_links (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_links'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.E - CausalExtractor._extract_causal_justifications (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_justifications',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_justifications'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.O - BayesianMechanismInference.infer_mechanisms (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            'infer_mechanisms',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference.infer_mechanisms'] = result
        if result is not None:
            current_data = result
        
        # 18. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 19. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 20. DB.V - BayesianMechanismInference._classify_mechanism_type (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_classify_mechanism_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._classify_mechanism_type'] = result
        if result is not None:
            current_data = result
        
        # 21. DB.V - BeachEvidentialTest.apply_test_logic (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'apply_test_logic',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.apply_test_logic'] = result
        if result is not None:
            current_data = result
        
        # 22. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 23. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 24. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 25. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q3_Executor(DataFlowExecutor):
    """
    D5-Q3: Evidencia de Causalidad
    Flow: PP.O → CD.E+T+V+C → DB.O (Extractor+Tests) → EP.C
    Métodos: 19
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.E - PolicyContradictionDetector._extract_quantitative_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_quantitative_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_quantitative_claims'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.T - PolicyContradictionDetector._parse_number (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_parse_number',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._parse_number'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 13. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.E - CausalExtractor._extract_causal_justifications (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            '_extract_causal_justifications',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor._extract_causal_justifications'] = result
        if result is not None:
            current_data = result
        
        # 15. DB.O - BayesianMechanismInference.infer_mechanisms (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            'infer_mechanisms',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference.infer_mechanisms'] = result
        if result is not None:
            current_data = result
        
        # 16. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 17. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 18. EP.C - BayesianNumericalAnalyzer.evaluate_policy_metric (P=3)
        result = self.executor.execute(
            'BayesianNumericalAnalyzer',
            'evaluate_policy_metric',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianNumericalAnalyzer.evaluate_policy_metric'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q4_Executor(DataFlowExecutor):
    """
    D5-Q4: Plazos de Impacto
    Flow: PP.E+T → CD.E → CD.V+T+C → A1.C+V
    Métodos: 15
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - TemporalLogicVerifier.verify_temporal_consistency (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            'verify_temporal_consistency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier.verify_temporal_consistency'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - TemporalLogicVerifier._check_deadline_constraints (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_check_deadline_constraints',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._check_deadline_constraints'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - TemporalLogicVerifier._classify_temporal_type (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_classify_temporal_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._classify_temporal_type'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - TemporalLogicVerifier._build_timeline (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_build_timeline',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._build_timeline'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.T - TemporalLogicVerifier._parse_temporal_marker (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_parse_temporal_marker',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._parse_temporal_marker'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - TemporalLogicVerifier._has_temporal_conflict (P=3)
        result = self.executor.execute(
            'TemporalLogicVerifier',
            '_has_temporal_conflict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TemporalLogicVerifier._has_temporal_conflict'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._extract_temporal_markers (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_temporal_markers',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_temporal_markers'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 13. A1.C - PerformanceAnalyzer._calculate_throughput_metrics (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_calculate_throughput_metrics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._calculate_throughput_metrics'] = result
        if result is not None:
            current_data = result
        
        # 14. A1.V - PerformanceAnalyzer._detect_bottlenecks (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_detect_bottlenecks',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._detect_bottlenecks'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D5Q5_Executor(DataFlowExecutor):
    """
    D5-Q5: Sostenibilidad Financiera
    Flow: PP.O → FV.E+C → CD.E+V+C → DB.O+C
    Métodos: 15
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. FV.C - PDETMunicipalPlanAnalyzer.analyze_financial_feasibility (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'analyze_financial_feasibility',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.analyze_financial_feasibility'] = result
        if result is not None:
            current_data = result
        
        # 6. FV.C - PDETMunicipalPlanAnalyzer._assess_financial_sustainability (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_assess_financial_sustainability',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._assess_financial_sustainability'] = result
        if result is not None:
            current_data = result
        
        # 7. FV.C - PDETMunicipalPlanAnalyzer._bayesian_risk_inference (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_bayesian_risk_inference',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._bayesian_risk_inference'] = result
        if result is not None:
            current_data = result
        
        # 8. FV.C - PDETMunicipalPlanAnalyzer._analyze_funding_sources (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_analyze_funding_sources',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._analyze_funding_sources'] = result
        if result is not None:
            current_data = result
        
        # 9. FV.E - PDETMunicipalPlanAnalyzer.extract_tables (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'extract_tables',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.extract_tables'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._extract_resource_mentions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_extract_resource_mentions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._extract_resource_mentions'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._detect_resource_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_resource_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_resource_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 13. DB.O - FinancialAuditor.trace_financial_allocation (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            'trace_financial_allocation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor.trace_financial_allocation'] = result
        if result is not None:
            current_data = result
        
        # 14. DB.C - FinancialAuditor._calculate_sufficiency (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_calculate_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._calculate_sufficiency'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q1_Executor(DataFlowExecutor):
    """
    D6-Q1: Integridad de Teoría de Cambio
    Flow: PP.O → TC.V (validacion_completa) → TC.T (construir_grafo) → CD.T+C → DB.O (CausalExtractor+Auditor+Framework) → FV.T
    Métodos: 32
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.C - IndustrialPolicyProcessor._analyze_causal_dimensions (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_analyze_causal_dimensions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._analyze_causal_dimensions'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 6. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 7. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 8. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 9. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        # 10. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 11. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 12. TC.V - AdvancedDAGValidator.calculate_acyclicity_pvalue (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'calculate_acyclicity_pvalue',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.calculate_acyclicity_pvalue'] = result
        if result is not None:
            current_data = result
        
        # 13. TC.C - AdvancedDAGValidator._calculate_statistical_power (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_statistical_power',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_statistical_power'] = result
        if result is not None:
            current_data = result
        
        # 14. TC.C - AdvancedDAGValidator._calculate_bayesian_posterior (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_bayesian_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_bayesian_posterior'] = result
        if result is not None:
            current_data = result
        
        # 15. TC.V - AdvancedDAGValidator._perform_sensitivity_analysis_internal (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_perform_sensitivity_analysis_internal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._perform_sensitivity_analysis_internal'] = result
        if result is not None:
            current_data = result
        
        # 16. TC.C - AdvancedDAGValidator.get_graph_stats (P=2)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'get_graph_stats',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.get_graph_stats'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.C - PolicyContradictionDetector._get_graph_statistics (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_graph_statistics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_graph_statistics'] = result
        if result is not None:
            current_data = result
        
        # 19. CD.C - PolicyContradictionDetector._calculate_graph_fragmentation (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_graph_fragmentation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_graph_fragmentation'] = result
        if result is not None:
            current_data = result
        
        # 20. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 21. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 22. DB.O - CausalExtractor.extract_causal_hierarchy (P=3)
        result = self.executor.execute(
            'CausalExtractor',
            'extract_causal_hierarchy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalExtractor.extract_causal_hierarchy'] = result
        if result is not None:
            current_data = result
        
        # 23. DB.V - OperationalizationAuditor.audit_evidence_traceability (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            'audit_evidence_traceability',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor.audit_evidence_traceability'] = result
        if result is not None:
            current_data = result
        
        # 24. DB.V - OperationalizationAuditor._audit_systemic_risk (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            '_audit_systemic_risk',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor._audit_systemic_risk'] = result
        if result is not None:
            current_data = result
        
        # 25. DB.V - OperationalizationAuditor.bayesian_counterfactual_audit (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            'bayesian_counterfactual_audit',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor.bayesian_counterfactual_audit'] = result
        if result is not None:
            current_data = result
        
        # 26. DB.R - OperationalizationAuditor._generate_optimal_remediations (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            '_generate_optimal_remediations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor._generate_optimal_remediations'] = result
        if result is not None:
            current_data = result
        
        # 27. DB.O - CDAFFramework.process_document (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            'process_document',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework.process_document'] = result
        if result is not None:
            current_data = result
        
        # 28. DB.V - CDAFFramework._audit_causal_coherence (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_audit_causal_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._audit_causal_coherence'] = result
        if result is not None:
            current_data = result
        
        # 29. DB.V - CDAFFramework._validate_dnp_compliance (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_validate_dnp_compliance',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._validate_dnp_compliance'] = result
        if result is not None:
            current_data = result
        
        # 30. DB.R - CDAFFramework._generate_extraction_report (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_generate_extraction_report',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._generate_extraction_report'] = result
        if result is not None:
            current_data = result
        
        # 31. FV.T - PDETMunicipalPlanAnalyzer.construct_causal_dag (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'construct_causal_dag',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.construct_causal_dag'] = result
        if result is not None:
            current_data = result
        
        # 32. FV.E - PDETMunicipalPlanAnalyzer._identify_causal_nodes (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_identify_causal_nodes',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._identify_causal_nodes'] = result
        if result is not None:
            current_data = result
        
        # 33. FV.E - PDETMunicipalPlanAnalyzer._identify_causal_edges (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_identify_causal_edges',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._identify_causal_edges'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q2_Executor(DataFlowExecutor):
    """
    D6-Q2: Proporcionalidad y Continuidad (Anti-Milagro)
    Flow: PP.E+T (3 categorías patrones) → CD.T+V+C → TC.V → DB (Beach Tests + Inference + Setup) → DB.Auditor
    Métodos: 28
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.T - IndustrialPolicyProcessor._compile_pattern_registry (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_compile_pattern_registry',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._compile_pattern_registry'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - IndustrialPolicyProcessor._build_point_patterns (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_build_point_patterns',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._build_point_patterns'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 6. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.C - PolicyContradictionDetector._calculate_syntactic_complexity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_syntactic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_syntactic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.C - PolicyContradictionDetector._get_dependency_depth (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_dependency_depth',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_dependency_depth'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._determine_relation_type (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_relation_type',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_relation_type'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - PolicyContradictionDetector._calculate_numerical_divergence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_numerical_divergence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_numerical_divergence'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.V - PolicyContradictionDetector._statistical_significance_test (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_statistical_significance_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._statistical_significance_test'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.V - PolicyContradictionDetector._are_comparable_claims (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_are_comparable_claims',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._are_comparable_claims'] = result
        if result is not None:
            current_data = result
        
        # 16. CD.C - PolicyContradictionDetector._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 19. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 20. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 21. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        # 22. TC.V - AdvancedDAGValidator.calculate_acyclicity_pvalue (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'calculate_acyclicity_pvalue',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.calculate_acyclicity_pvalue'] = result
        if result is not None:
            current_data = result
        
        # 23. TC.C - AdvancedDAGValidator._calculate_statistical_power (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_statistical_power',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_statistical_power'] = result
        if result is not None:
            current_data = result
        
        # 24. TC.C - AdvancedDAGValidator._calculate_bayesian_posterior (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_bayesian_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_bayesian_posterior'] = result
        if result is not None:
            current_data = result
        
        # 25. DB.V - BeachEvidentialTest.classify_test (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'classify_test',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.classify_test'] = result
        if result is not None:
            current_data = result
        
        # 26. DB.V - BeachEvidentialTest.apply_test_logic (P=3)
        result = self.executor.execute(
            'BeachEvidentialTest',
            'apply_test_logic',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BeachEvidentialTest.apply_test_logic'] = result
        if result is not None:
            current_data = result
        
        # 27. DB.V - BayesianMechanismInference._test_necessity (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_necessity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_necessity'] = result
        if result is not None:
            current_data = result
        
        # 28. DB.V - BayesianMechanismInference._test_sufficiency (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_test_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._test_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 29. DB.T - BayesianMechanismInference._build_transition_matrix (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_build_transition_matrix',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._build_transition_matrix'] = result
        if result is not None:
            current_data = result
        
        # 30. DB.C - BayesianMechanismInference._calculate_type_transition_prior (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_calculate_type_transition_prior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._calculate_type_transition_prior'] = result
        if result is not None:
            current_data = result
        
        # 31. DB.V - BayesianMechanismInference._infer_activity_sequence (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_infer_activity_sequence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._infer_activity_sequence'] = result
        if result is not None:
            current_data = result
        
        # 32. DB.C - BayesianMechanismInference._aggregate_bayesian_confidence (P=3)
        result = self.executor.execute(
            'BayesianMechanismInference',
            '_aggregate_bayesian_confidence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianMechanismInference._aggregate_bayesian_confidence'] = result
        if result is not None:
            current_data = result
        
        # 33. DB.V - CausalInferenceSetup.classify_goal_dynamics (P=3)
        result = self.executor.execute(
            'CausalInferenceSetup',
            'classify_goal_dynamics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalInferenceSetup.classify_goal_dynamics'] = result
        if result is not None:
            current_data = result
        
        # 34. DB.V - CausalInferenceSetup.identify_failure_points (P=3)
        result = self.executor.execute(
            'CausalInferenceSetup',
            'identify_failure_points',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalInferenceSetup.identify_failure_points'] = result
        if result is not None:
            current_data = result
        
        # 35. DB.C - CausalInferenceSetup.assign_probative_value (P=3)
        result = self.executor.execute(
            'CausalInferenceSetup',
            'assign_probative_value',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalInferenceSetup.assign_probative_value'] = result
        if result is not None:
            current_data = result
        
        # 36. DB.E - CausalInferenceSetup._get_dynamics_pattern (P=3)
        result = self.executor.execute(
            'CausalInferenceSetup',
            '_get_dynamics_pattern',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CausalInferenceSetup._get_dynamics_pattern'] = result
        if result is not None:
            current_data = result
        
        # 37. DB.V - OperationalizationAuditor._audit_systemic_risk (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            '_audit_systemic_risk',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor._audit_systemic_risk'] = result
        if result is not None:
            current_data = result
        
        # 38. DB.V - OperationalizationAuditor.bayesian_counterfactual_audit (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            'bayesian_counterfactual_audit',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor.bayesian_counterfactual_audit'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q3_Executor(DataFlowExecutor):
    """
    D6-Q3: Inconsistencias (Sistema Bicameral - Ruta 1)
    Flow: PP.O → CD.V (detect suite) → CD.R (_suggest_resolutions) → TC.V → A1.V
    Métodos: 22
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 5. CD.V - PolicyContradictionDetector._detect_logical_incompatibilities (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_logical_incompatibilities',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_logical_incompatibilities'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.V - PolicyContradictionDetector.detect (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            'detect',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector.detect'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.V - PolicyContradictionDetector._detect_semantic_contradictions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_semantic_contradictions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_semantic_contradictions'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.V - PolicyContradictionDetector._detect_numerical_inconsistencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_numerical_inconsistencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_numerical_inconsistencies'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._detect_temporal_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_temporal_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_temporal_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.V - PolicyContradictionDetector._detect_resource_conflicts (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_detect_resource_conflicts',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._detect_resource_conflicts'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.V - PolicyContradictionDetector._classify_contradiction (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_classify_contradiction',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._classify_contradiction'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.C - PolicyContradictionDetector._calculate_severity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_severity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_severity'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.R - PolicyContradictionDetector._generate_resolution_recommendations (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_resolution_recommendations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_resolution_recommendations'] = result
        if result is not None:
            current_data = result
        
        # 14. CD.R - PolicyContradictionDetector._suggest_resolutions (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_suggest_resolutions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._suggest_resolutions'] = result
        if result is not None:
            current_data = result
        
        # 15. CD.C - PolicyContradictionDetector._calculate_contradiction_entropy (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_contradiction_entropy',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_contradiction_entropy'] = result
        if result is not None:
            current_data = result
        
        # 16. CD.C - PolicyContradictionDetector._get_domain_weight (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_domain_weight',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_domain_weight'] = result
        if result is not None:
            current_data = result
        
        # 17. CD.V - PolicyContradictionDetector._has_logical_conflict (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_has_logical_conflict',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._has_logical_conflict'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 19. A1.V - TextMiningEngine.diagnose_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            'diagnose_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine.diagnose_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 20. A1.E - TextMiningEngine._identify_critical_links (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_identify_critical_links',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._identify_critical_links'] = result
        if result is not None:
            current_data = result
        
        # 21. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 22. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q4_Executor(DataFlowExecutor):
    """
    D6-Q4: Adaptación (Sistema Bicameral - Ruta 2)
    Flow: PP.O → TC.V+R (_generar_sugerencias_internas) → CD.T+C → DB (CDAF+Auditors) → FV.R
    Métodos: 26
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. TC.V - TeoriaCambio.validacion_completa (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'validacion_completa',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.validacion_completa'] = result
        if result is not None:
            current_data = result
        
        # 7. TC.V - TeoriaCambio._validar_orden_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_validar_orden_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._validar_orden_causal'] = result
        if result is not None:
            current_data = result
        
        # 8. TC.V - TeoriaCambio._encontrar_caminos_completos (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_encontrar_caminos_completos',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._encontrar_caminos_completos'] = result
        if result is not None:
            current_data = result
        
        # 9. TC.R - TeoriaCambio._generar_sugerencias_internas (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_generar_sugerencias_internas',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._generar_sugerencias_internas'] = result
        if result is not None:
            current_data = result
        
        # 10. TC.R - TeoriaCambio._execute_generar_sugerencias_internas (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_execute_generar_sugerencias_internas',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._execute_generar_sugerencias_internas'] = result
        if result is not None:
            current_data = result
        
        # 11. TC.E - TeoriaCambio._extraer_categorias (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_extraer_categorias',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._extraer_categorias'] = result
        if result is not None:
            current_data = result
        
        # 12. TC.V - TeoriaCambio._es_conexion_valida (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            '_es_conexion_valida',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio._es_conexion_valida'] = result
        if result is not None:
            current_data = result
        
        # 13. TC.T - TeoriaCambio.construir_grafo_causal (P=3)
        result = self.executor.execute(
            'TeoriaCambio',
            'construir_grafo_causal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TeoriaCambio.construir_grafo_causal'] = result
        if result is not None:
            current_data = result
        
        # 14. TC.V - AdvancedDAGValidator.calculate_acyclicity_pvalue (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'calculate_acyclicity_pvalue',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.calculate_acyclicity_pvalue'] = result
        if result is not None:
            current_data = result
        
        # 15. TC.V - AdvancedDAGValidator._perform_sensitivity_analysis_internal (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_perform_sensitivity_analysis_internal',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._perform_sensitivity_analysis_internal'] = result
        if result is not None:
            current_data = result
        
        # 16. TC.C - AdvancedDAGValidator._calculate_confidence_interval (P=3)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            '_calculate_confidence_interval',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator._calculate_confidence_interval'] = result
        if result is not None:
            current_data = result
        
        # 17. TC.C - AdvancedDAGValidator.get_graph_stats (P=2)
        result = self.executor.execute(
            'AdvancedDAGValidator',
            'get_graph_stats',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedDAGValidator.get_graph_stats'] = result
        if result is not None:
            current_data = result
        
        # 18. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 19. CD.C - PolicyContradictionDetector._get_graph_statistics (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_graph_statistics',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_graph_statistics'] = result
        if result is not None:
            current_data = result
        
        # 20. CD.C - PolicyContradictionDetector._calculate_graph_fragmentation (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_graph_fragmentation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_graph_fragmentation'] = result
        if result is not None:
            current_data = result
        
        # 21. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 22. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 23. A1.R - PerformanceAnalyzer._generate_recommendations (P=2)
        result = self.executor.execute(
            'PerformanceAnalyzer',
            '_generate_recommendations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PerformanceAnalyzer._generate_recommendations'] = result
        if result is not None:
            current_data = result
        
        # 24. A1.R - TextMiningEngine._generate_interventions (P=2)
        result = self.executor.execute(
            'TextMiningEngine',
            '_generate_interventions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['TextMiningEngine._generate_interventions'] = result
        if result is not None:
            current_data = result
        
        # 25. DB.V - CDAFFramework._validate_dnp_compliance (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_validate_dnp_compliance',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._validate_dnp_compliance'] = result
        if result is not None:
            current_data = result
        
        # 26. DB.R - CDAFFramework._generate_extraction_report (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_generate_extraction_report',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._generate_extraction_report'] = result
        if result is not None:
            current_data = result
        
        # 27. DB.R - CDAFFramework._generate_causal_model_json (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_generate_causal_model_json',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._generate_causal_model_json'] = result
        if result is not None:
            current_data = result
        
        # 28. DB.R - CDAFFramework._generate_dnp_compliance_report (P=3)
        result = self.executor.execute(
            'CDAFFramework',
            '_generate_dnp_compliance_report',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['CDAFFramework._generate_dnp_compliance_report'] = result
        if result is not None:
            current_data = result
        
        # 29. DB.V - OperationalizationAuditor.audit_evidence_traceability (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            'audit_evidence_traceability',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor.audit_evidence_traceability'] = result
        if result is not None:
            current_data = result
        
        # 30. DB.V - OperationalizationAuditor._perform_counterfactual_budget_check (P=3)
        result = self.executor.execute(
            'OperationalizationAuditor',
            '_perform_counterfactual_budget_check',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['OperationalizationAuditor._perform_counterfactual_budget_check'] = result
        if result is not None:
            current_data = result
        
        # 31. DB.O - FinancialAuditor.trace_financial_allocation (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            'trace_financial_allocation',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor.trace_financial_allocation'] = result
        if result is not None:
            current_data = result
        
        # 32. DB.V - FinancialAuditor._match_goal_to_budget (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_match_goal_to_budget',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._match_goal_to_budget'] = result
        if result is not None:
            current_data = result
        
        # 33. DB.C - FinancialAuditor._calculate_sufficiency (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_calculate_sufficiency',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._calculate_sufficiency'] = result
        if result is not None:
            current_data = result
        
        # 34. DB.V - FinancialAuditor._detect_allocation_gaps (P=3)
        result = self.executor.execute(
            'FinancialAuditor',
            '_detect_allocation_gaps',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['FinancialAuditor._detect_allocation_gaps'] = result
        if result is not None:
            current_data = result
        
        # 35. DB.V - MechanismTypeConfig.check_sum_to_one (P=3)
        result = self.executor.execute(
            'MechanismTypeConfig',
            'check_sum_to_one',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MechanismTypeConfig.check_sum_to_one'] = result
        if result is not None:
            current_data = result
        
        # 36. FV.R - PDETMunicipalPlanAnalyzer.generate_recommendations (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            'generate_recommendations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer.generate_recommendations'] = result
        if result is not None:
            current_data = result
        
        # 37. FV.R - PDETMunicipalPlanAnalyzer._generate_optimal_remediations (P=3)
        result = self.executor.execute(
            'PDETMunicipalPlanAnalyzer',
            '_generate_optimal_remediations',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PDETMunicipalPlanAnalyzer._generate_optimal_remediations'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


class D6Q5_Executor(DataFlowExecutor):
    """
    D6-Q5: Contextualización y Enfoque Diferencial
    Flow: PP.E (patrones diferenciales) → CD.T+V+C → A1.V+E (_classify_cross_cutting_themes) → EP.E+V+C
    Métodos: 24
    """
    
    def execute(self, doc, method_executor):
        self.executor = method_executor
        results = {}
        current_data = doc.raw_text
        
        # 1. PP.E - IndustrialPolicyProcessor._match_patterns_in_sentences (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            '_match_patterns_in_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor._match_patterns_in_sentences'] = result
        if result is not None:
            current_data = result
        
        # 2. PP.O - IndustrialPolicyProcessor.process (P=3)
        result = self.executor.execute(
            'IndustrialPolicyProcessor',
            'process',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['IndustrialPolicyProcessor.process'] = result
        if result is not None:
            current_data = result
        
        # 3. PP.T - PolicyTextProcessor.segment_into_sentences (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'segment_into_sentences',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.segment_into_sentences'] = result
        if result is not None:
            current_data = result
        
        # 4. PP.E - PolicyTextProcessor.extract_contextual_window (P=2)
        result = self.executor.execute(
            'PolicyTextProcessor',
            'extract_contextual_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyTextProcessor.extract_contextual_window'] = result
        if result is not None:
            current_data = result
        
        # 5. PP.C - BayesianEvidenceScorer.compute_evidence_score (P=3)
        result = self.executor.execute(
            'BayesianEvidenceScorer',
            'compute_evidence_score',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianEvidenceScorer.compute_evidence_score'] = result
        if result is not None:
            current_data = result
        
        # 6. CD.T - PolicyContradictionDetector._generate_embeddings (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_generate_embeddings',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._generate_embeddings'] = result
        if result is not None:
            current_data = result
        
        # 7. CD.C - PolicyContradictionDetector._calculate_similarity (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_similarity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_similarity'] = result
        if result is not None:
            current_data = result
        
        # 8. CD.E - PolicyContradictionDetector._identify_dependencies (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_identify_dependencies',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._identify_dependencies'] = result
        if result is not None:
            current_data = result
        
        # 9. CD.V - PolicyContradictionDetector._determine_semantic_role (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_determine_semantic_role',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._determine_semantic_role'] = result
        if result is not None:
            current_data = result
        
        # 10. CD.C - PolicyContradictionDetector._calculate_global_semantic_coherence (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_calculate_global_semantic_coherence',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._calculate_global_semantic_coherence'] = result
        if result is not None:
            current_data = result
        
        # 11. CD.E - PolicyContradictionDetector._get_context_window (P=2)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_get_context_window',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._get_context_window'] = result
        if result is not None:
            current_data = result
        
        # 12. CD.T - PolicyContradictionDetector._build_knowledge_graph (P=3)
        result = self.executor.execute(
            'PolicyContradictionDetector',
            '_build_knowledge_graph',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyContradictionDetector._build_knowledge_graph'] = result
        if result is not None:
            current_data = result
        
        # 13. CD.C - BayesianConfidenceCalculator.calculate_posterior (P=3)
        result = self.executor.execute(
            'BayesianConfidenceCalculator',
            'calculate_posterior',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['BayesianConfidenceCalculator.calculate_posterior'] = result
        if result is not None:
            current_data = result
        
        # 14. A1.V - SemanticAnalyzer._classify_cross_cutting_themes (P=3)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_cross_cutting_themes',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_cross_cutting_themes'] = result
        if result is not None:
            current_data = result
        
        # 15. A1.V - SemanticAnalyzer._classify_policy_domain (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_classify_policy_domain',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._classify_policy_domain'] = result
        if result is not None:
            current_data = result
        
        # 16. A1.E - SemanticAnalyzer.extract_semantic_cube (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            'extract_semantic_cube',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer.extract_semantic_cube'] = result
        if result is not None:
            current_data = result
        
        # 17. A1.T - SemanticAnalyzer._process_segment (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_process_segment',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._process_segment'] = result
        if result is not None:
            current_data = result
        
        # 18. A1.T - SemanticAnalyzer._vectorize_segments (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_vectorize_segments',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._vectorize_segments'] = result
        if result is not None:
            current_data = result
        
        # 19. A1.C - SemanticAnalyzer._calculate_semantic_complexity (P=2)
        result = self.executor.execute(
            'SemanticAnalyzer',
            '_calculate_semantic_complexity',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['SemanticAnalyzer._calculate_semantic_complexity'] = result
        if result is not None:
            current_data = result
        
        # 20. A1.O - MunicipalOntology.__init__ (P=2)
        result = self.executor.execute(
            'MunicipalOntology',
            '__init__',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['MunicipalOntology.__init__'] = result
        if result is not None:
            current_data = result
        
        # 21. EP.E - PolicyAnalysisEmbedder.semantic_search (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'semantic_search',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.semantic_search'] = result
        if result is not None:
            current_data = result
        
        # 22. EP.V - PolicyAnalysisEmbedder._filter_by_pdq (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            '_filter_by_pdq',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder._filter_by_pdq'] = result
        if result is not None:
            current_data = result
        
        # 23. EP.C - PolicyAnalysisEmbedder.compare_policy_interventions (P=3)
        result = self.executor.execute(
            'PolicyAnalysisEmbedder',
            'compare_policy_interventions',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['PolicyAnalysisEmbedder.compare_policy_interventions'] = result
        if result is not None:
            current_data = result
        
        # 24. EP.V - AdvancedSemanticChunker._infer_pdq_context (P=3)
        result = self.executor.execute(
            'AdvancedSemanticChunker',
            '_infer_pdq_context',
            data=current_data,
            text=doc.raw_text,
            sentences=doc.sentences,
            tables=doc.tables
        )
        results['AdvancedSemanticChunker._infer_pdq_context'] = result
        if result is not None:
            current_data = result
        
        return {
            'modality': 'TYPE_A',
            'elements': self._extract(results),
            'raw': results
        }
    
    def _extract(self, results):
        vals = [v for v in results.values() if v is not None]
        return vals[:4] if vals else []


# ============================================================================
# ORQUESTADOR
# ============================================================================

