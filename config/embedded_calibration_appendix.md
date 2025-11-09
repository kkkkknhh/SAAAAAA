# Embedded Calibration Migration Appendix

This document tracks all methods with embedded/inline calibration that must be
migrated to the centralized calibration system.

**Status:** Transitional Anomalies - Explicitly Tracked

**Generated:** 2025-11-09T18:11:25.972713

## Summary

- **Total methods with embedded calibration:** 61
- **CRITICAL priority:** 3
- **HIGH priority:** 10
- **MEDIUM priority:** 20
- **LOW priority:** 28

## Migration Backlog

Methods are listed in priority order (critical → low).

### CRITICAL Priority

Count: 3

#### 1. src.saaaaaa.analysis.scoring.scoring.score_type_a

- **File:** `src/saaaaaa/analysis/scoring/scoring.py:348`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 5

**Parameters:**
- `magic_number` = `4`
- `magic_number` = `3.0`
- `magic_number` = `4`
- `magic_number` = `3.0`
- `magic_number` = `4`

**Notes:** Contains 5 magic numbers

---

#### 2. src.saaaaaa.analysis.scoring.scoring.score_type_d

- **File:** `src/saaaaaa/analysis/scoring/scoring.py:511`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `magic_number` = `3.0`
- `magic_number` = `3`
- `magic_number` = `3.0`

**Notes:** Contains 3 magic numbers

---

#### 3. scripts_verify_executor_config.verify_executor_config_integration

- **File:** `scripts_verify_executor_config.py:11`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `60`
- `magic_number` = `60`

**Notes:** Contains 2 magic numbers

---

### HIGH Priority

Count: 10

#### 1. src.saaaaaa.analysis.meso_cluster_analysis.analyze_policy_dispersion

- **File:** `src/saaaaaa/analysis/meso_cluster_analysis.py:88`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 10

**Parameters:**
- `magic_number` = `3`
- `magic_number` = `3`
- `magic_number` = `1.5`
- `magic_number` = `1.5`
- `magic_number` = `1.5`
- `magic_number` = `1.5`
- `magic_number` = `1.5`
- `magic_number` = `1.5`
- `magic_number` = `25`
- `magic_number` = `6`

**Notes:** Contains 13 magic numbers

---

#### 2. src.saaaaaa.analysis.Analyzer_one.PerformanceAnalyzer._calculate_loss_functions

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:476`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 6

**Parameters:**
- `magic_number` = `50.0`
- `magic_number` = `0.8`
- `magic_number` = `0.5`
- `magic_number` = `0.2`
- `magic_number` = `0.4`
- `magic_number` = `0.4`

**Notes:** Contains 6 magic numbers

---

#### 3. src.saaaaaa.analysis.Analyzer_one.DocumentProcessor.segment_text

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:1228`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 6

**Parameters:**
- `magic_number` = `50`
- `magic_number` = `10`
- `magic_number` = `20`
- `magic_number` = `20`
- `magic_number` = `10`
- `magic_number` = `10`

**Notes:** Contains 6 magic numbers

---

#### 4. src.saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer.calculate_quality_score

- **File:** `src/saaaaaa/analysis/financiero_viabilidad_tablas.py:1612`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 6

**Parameters:**
- `magic_number` = `0.2`
- `magic_number` = `0.15`
- `magic_number` = `0.15`
- `magic_number` = `0.1`
- `magic_number` = `0.2`
- `magic_number` = `0.2`

**Notes:** Contains 6 magic numbers

---

#### 5. src.saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer._score_financial_component

- **File:** `src/saaaaaa/analysis/financiero_viabilidad_tablas.py:1663`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 8

**Parameters:**
- `magic_number` = `3.0`
- `magic_number` = `3.0`
- `magic_number` = `2.5`
- `magic_number` = `0.5`
- `magic_number` = `1.5`
- `magic_number` = `10.0`
- `magic_number` = `12`
- `magic_number` = `0.1`

**Notes:** Contains 8 magic numbers

---

#### 6. src.saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer._score_indicators

- **File:** `src/saaaaaa/analysis/financiero_viabilidad_tablas.py:1684`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 10

**Parameters:**
- `magic_number` = `4.0`
- `magic_number` = `3.0`
- `magic_number` = `3.0`
- `magic_number` = `4.0`
- `magic_number` = `3.0`
- `magic_number` = `10.0`
- `magic_number` = `5`
- `magic_number` = `5`
- `magic_number` = `50`
- `magic_number` = `10`

**Notes:** Contains 10 magic numbers

---

#### 7. src.saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer._score_temporal_consistency

- **File:** `src/saaaaaa/analysis/financiero_viabilidad_tablas.py:1739`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 7

**Parameters:**
- `magic_number` = `3.0`
- `magic_number` = `3.0`
- `magic_number` = `4.0`
- `magic_number` = `3.0`
- `magic_number` = `10.0`
- `magic_number` = `4`
- `magic_number` = `30`

**Notes:** Contains 7 magic numbers

---

#### 8. src.saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer._score_pdet_alignment

- **File:** `src/saaaaaa/analysis/financiero_viabilidad_tablas.py:1760`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 7

**Parameters:**
- `magic_number` = `4.0`
- `magic_number` = `3.0`
- `magic_number` = `3.0`
- `magic_number` = `1.5`
- `magic_number` = `10.0`
- `magic_number` = `3`
- `magic_number` = `15`

**Notes:** Contains 7 magic numbers

---

#### 9. src.saaaaaa.analysis.teoria_cambio.IndustrialGradeValidator.execute_suite

- **File:** `src/saaaaaa/analysis/teoria_cambio.py:839`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 7

**Parameters:**
- `magic_number` = `100`
- `magic_number` = `80`
- `magic_number` = `80`
- `magic_number` = `100`
- `magic_number` = `80`
- `magic_number` = `90.0`
- `magic_number` = `80`

**Notes:** Contains 7 magic numbers

---

#### 10. src.saaaaaa.analysis.dereck_beach.BayesianCounterfactualAuditor.aggregate_risk_and_prioritize

- **File:** `src/saaaaaa/analysis/dereck_beach.py:5939`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 10

**Parameters:**
- `magic_number` = `0.8`
- `magic_number` = `1e-06`
- `magic_number` = `0.05`
- `magic_number` = `0.7`
- `magic_number` = `0.6`
- `magic_number` = `0.5`
- `magic_number` = `0.5`
- `magic_number` = `0.15`
- `magic_number` = `0.1`
- `magic_number` = `0.4`

**Notes:** Contains 15 magic numbers

---

### MEDIUM Priority

Count: 20

#### 1. src.saaaaaa.processing.embedding_policy.PolicyAnalysisEmbedder._compute_overall_confidence

- **File:** `src/saaaaaa/processing/embedding_policy.py:1470`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 6

**Parameters:**
- `magic_number` = `0.25`
- `magic_number` = `0.5`
- `magic_number` = `0.75`
- `magic_number` = `0.6`
- `magic_number` = `0.4`
- `magic_number` = `5`

**Notes:** Contains 6 magic numbers

---

#### 2. src.saaaaaa.processing.semantic_chunking_policy.BayesianEvidenceIntegrator._compute_reliability_weights

- **File:** `src/saaaaaa/processing/semantic_chunking_policy.py:368`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `pos_weight` = `1`
- `content_weight` = `1`

**Notes:** Embedded parameters detected

---

#### 3. src.saaaaaa.analysis.Analyzer_one.SemanticAnalyzer.__init__

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:137`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `max_features` = `1`
- `magic_number` = `1000`
- `magic_number` = `3`

**Notes:** Contains 2 magic numbers

---

#### 4. src.saaaaaa.analysis.Analyzer_one.MunicipalAnalyzer._load_document

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:772`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `100`
- `magic_number` = `20`

**Notes:** Contains 2 magic numbers

---

#### 5. src.saaaaaa.analysis.Analyzer_one.example_usage

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:840`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 4

**Parameters:**
- `magic_number` = `60`
- `magic_number` = `3`
- `magic_number` = `60`
- `magic_number` = `5`

**Notes:** Contains 4 magic numbers

---

#### 6. src.saaaaaa.analysis.Analyzer_one.DocumentProcessor.load_canonical_question_contracts

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:1271`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `64`
- `magic_number` = `5`

**Notes:** Contains 2 magic numbers

---

#### 7. src.saaaaaa.analysis.Analyzer_one.ResultsExporter.export_summary_report

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:1575`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 5

**Parameters:**
- `magic_number` = `50`
- `magic_number` = `20`
- `magic_number` = `20`
- `magic_number` = `20`
- `magic_number` = `15`

**Notes:** Contains 5 magic numbers

---

#### 8. src.saaaaaa.analysis.Analyzer_one.ConfigurationManager.load_config

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:1663`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 5

**Parameters:**
- `magic_number` = `200`
- `magic_number` = `20`
- `magic_number` = `0.4`
- `magic_number` = `0.5`
- `magic_number` = `20`

**Notes:** Contains 5 magic numbers

---

#### 9. src.saaaaaa.analysis.Analyzer_one.BatchProcessor._create_batch_summary

- **File:** `src/saaaaaa/analysis/Analyzer_one.py:1761`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `magic_number` = `30`
- `magic_number` = `15`
- `magic_number` = `20`

**Notes:** Contains 3 magic numbers

---

#### 10. src.saaaaaa.analysis.bayesian_multilevel_system.BayesianUpdater._calculate_evidence_weight

- **File:** `src/saaaaaa/analysis/bayesian_multilevel_system.py:328`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 4

**Parameters:**
- `magic_number` = `1e-10`
- `magic_number` = `1e-10`
- `magic_number` = `1e-10`
- `magic_number` = `1e-10`

**Notes:** Contains 4 magic numbers

---

#### 11. src.saaaaaa.analysis.recommendation_engine.RecommendationEngine.get_thresholds_from_monolith

- **File:** `src/saaaaaa/analysis/recommendation_engine.py:212`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `55.0`
- `magic_number` = `65.0`

**Notes:** Contains 2 magic numbers

---

#### 12. src.saaaaaa.analysis.macro_prompts.PeerNormalizer._calculate_z_scores

- **File:** `src/saaaaaa/analysis/macro_prompts.py:1031`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `0.5`
- `magic_number` = `0.1`

**Notes:** Contains 2 magic numbers

---

#### 13. src.saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer._score_responsibility_clarity

- **File:** `src/saaaaaa/analysis/financiero_viabilidad_tablas.py:1722`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 5

**Parameters:**
- `magic_number` = `3.0`
- `magic_number` = `4.0`
- `magic_number` = `3.0`
- `magic_number` = `10.0`
- `magic_number` = `15`

**Notes:** Contains 5 magic numbers

---

#### 14. src.saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer._score_causal_coherence

- **File:** `src/saaaaaa/analysis/financiero_viabilidad_tablas.py:1785`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 5

**Parameters:**
- `magic_number` = `3.0`
- `magic_number` = `3.0`
- `magic_number` = `4.0`
- `magic_number` = `10.0`
- `magic_number` = `1.5`

**Notes:** Contains 5 magic numbers

---

#### 15. src.saaaaaa.analysis.financiero_viabilidad_tablas.PDETMunicipalPlanAnalyzer._estimate_score_confidence

- **File:** `src/saaaaaa/analysis/financiero_viabilidad_tablas.py:1809`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 5

**Parameters:**
- `magic_number` = `1000`
- `magic_number` = `0.5`
- `magic_number` = `10`
- `magic_number` = `2.5`
- `magic_number` = `97.5`

**Notes:** Contains 5 magic numbers

---

#### 16. src.saaaaaa.analysis.dereck_beach.ReportingEngine._calculate_quality_score

- **File:** `src/saaaaaa/analysis/dereck_beach.py:3977`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 4

**Parameters:**
- `magic_number` = `0.35`
- `magic_number` = `0.25`
- `magic_number` = `0.25`
- `magic_number` = `0.15`

**Notes:** Contains 4 magic numbers

---

#### 17. smart_policy_chunks_canonic_phase_one.StrategicIntegrator._calculate_strategic_weight

- **File:** `smart_policy_chunks_canonic_phase_one.py:1201`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `0.5`
- `magic_number` = `5`

**Notes:** Contains 2 magic numbers

---

#### 18. smart_policy_chunks_canonic_phase_one.StrategicChunkingSystem._analyze_cross_references

- **File:** `smart_policy_chunks_canonic_phase_one.py:2234`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `100`
- `magic_number` = `100`

**Notes:** Contains 2 magic numbers

---

#### 19. smart_policy_chunks_canonic_phase_one.StrategicChunkingSystem._analyze_discourse_structure

- **File:** `smart_policy_chunks_canonic_phase_one.py:2309`
- **Pattern:** magic_numbers
- **Complexity:** moderate
- **Parameters found:** 6

**Parameters:**
- `magic_number` = `500000`
- `magic_number` = `500`
- `magic_number` = `1000`
- `magic_number` = `200`
- `magic_number` = `200`
- `magic_number` = `1000`

**Notes:** Contains 6 magic numbers

---

#### 20. smart_policy_chunks_canonic_phase_one.StrategicChunkingSystem._analyze_rhetorical_structure

- **File:** `smart_policy_chunks_canonic_phase_one.py:2391`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 4

**Parameters:**
- `magic_number` = `50`
- `magic_number` = `50`
- `magic_number` = `50`
- `magic_number` = `50`

**Notes:** Contains 4 magic numbers

---

### LOW Priority

Count: 28

#### 1. src.saaaaaa.concurrency.concurrency.WorkerPool._execute_task_with_retry

- **File:** `src/saaaaaa/concurrency/concurrency.py:185`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `magic_number` = `1000`
- `magic_number` = `1000`
- `magic_number` = `1000`

**Notes:** Contains 3 magic numbers

---

#### 2. src.saaaaaa.core.calibration_engine.MethodCalibrationEngine.calibrate_from_characteristics

- **File:** `src/saaaaaa/core/calibration_engine.py:289`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `weight` = `1`
- `magic_number` = `0.5`
- `magic_number` = `0.5`

**Notes:** Contains 2 magic numbers

---

#### 3. src.saaaaaa.processing.aggregation.DimensionAggregator.validate_weights

- **File:** `src/saaaaaa/processing/aggregation.py:233`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 1

**Parameters:**
- `tolerance` = `1`

**Notes:** Embedded parameters detected

---

#### 4. src.saaaaaa.processing.aggregation.DimensionAggregator.apply_rubric_thresholds

- **File:** `src/saaaaaa/processing/aggregation.py:347`
- **Pattern:** explicit_parameters
- **Complexity:** complex
- **Parameters found:** 11

**Parameters:**
- `excellent_threshold` = `0`
- `good_threshold` = `0`
- `acceptable_threshold` = `0`
- `magic_number` = `3.0`
- `magic_number` = `0.85`
- `magic_number` = `0.7`
- `magic_number` = `0.55`
- `magic_number` = `3.0`
- `magic_number` = `0.85`
- `magic_number` = `0.7`

**Notes:** Contains 8 magic numbers

---

#### 5. src.saaaaaa.processing.aggregation.AreaPolicyAggregator.normalize_scores

- **File:** `src/saaaaaa/processing/aggregation.py:619`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `3.0`
- `magic_number` = `3.0`

**Notes:** Contains 2 magic numbers

---

#### 6. src.saaaaaa.processing.aggregation.AreaPolicyAggregator.apply_rubric_thresholds

- **File:** `src/saaaaaa/processing/aggregation.py:638`
- **Pattern:** explicit_parameters
- **Complexity:** complex
- **Parameters found:** 11

**Parameters:**
- `excellent_threshold` = `0`
- `good_threshold` = `0`
- `acceptable_threshold` = `0`
- `magic_number` = `3.0`
- `magic_number` = `0.85`
- `magic_number` = `0.7`
- `magic_number` = `0.55`
- `magic_number` = `3.0`
- `magic_number` = `0.85`
- `magic_number` = `0.7`

**Notes:** Contains 8 magic numbers

---

#### 7. src.saaaaaa.processing.aggregation.ClusterAggregator.apply_cluster_weights

- **File:** `src/saaaaaa/processing/aggregation.py:893`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 1

**Parameters:**
- `tolerance` = `1`

**Notes:** Embedded parameters detected

---

#### 8. src.saaaaaa.processing.aggregation.ClusterAggregator.analyze_coherence

- **File:** `src/saaaaaa/processing/aggregation.py:946`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `max_std` = `3`
- `magic_number` = `3.0`
- `magic_number` = `0.5`

**Notes:** Contains 2 magic numbers

---

#### 9. src.saaaaaa.processing.aggregation.ClusterAggregator.aggregate_cluster

- **File:** `src/saaaaaa/processing/aggregation.py:983`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `0.8`
- `magic_number` = `0.6`

**Notes:** Contains 2 magic numbers

---

#### 10. src.saaaaaa.processing.aggregation.MacroAggregator.apply_rubric_thresholds

- **File:** `src/saaaaaa/processing/aggregation.py:1243`
- **Pattern:** explicit_parameters
- **Complexity:** complex
- **Parameters found:** 11

**Parameters:**
- `excellent_threshold` = `0`
- `good_threshold` = `0`
- `acceptable_threshold` = `0`
- `magic_number` = `3.0`
- `magic_number` = `0.85`
- `magic_number` = `0.7`
- `magic_number` = `0.55`
- `magic_number` = `3.0`
- `magic_number` = `0.85`
- `magic_number` = `0.7`

**Notes:** Contains 8 magic numbers

---

#### 11. src.saaaaaa.core.orchestrator.executor_config.ExecutorConfig.from_cli

- **File:** `src/saaaaaa/core/orchestrator/executor_config.py:262`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `2048`
- `magic_number` = `30.0`

**Notes:** Contains 2 magic numbers

---

#### 12. src.saaaaaa.core.orchestrator.executor_config.ExecutorConfig.describe

- **File:** `src/saaaaaa/core/orchestrator/executor_config.py:336`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 1

**Parameters:**
- `max_tokens` = `4`

**Notes:** Embedded parameters detected

---

#### 13. src.saaaaaa.core.orchestrator.executor_config.ExecutorConfig.merge_overrides

- **File:** `src/saaaaaa/core/orchestrator/executor_config.py:379`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `max_tokens` = `2`
- `max_tokens` = `4`

**Notes:** Embedded parameters detected

---

#### 14. src.saaaaaa.core.orchestrator.executors.NeuromorphicFlowController.__init__

- **File:** `src/saaaaaa/core/orchestrator/executors.py:377`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `0.01`
- `magic_number` = `0.5`

**Notes:** Contains 2 magic numbers

---

#### 15. src.saaaaaa.core.orchestrator.executors.NeuromorphicFlowController.adapt_flow

- **File:** `src/saaaaaa/core/orchestrator/executors.py:407`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `0.5`
- `magic_number` = `0.5`

**Notes:** Contains 2 magic numbers

---

#### 16. src.saaaaaa.core.orchestrator.executors.CausalGraph._test_independence

- **File:** `src/saaaaaa/core/orchestrator/executors.py:455`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `0.5`
- `magic_number` = `3`

**Notes:** Contains 2 magic numbers

---

#### 17. src.saaaaaa.core.orchestrator.executors.MetaLearningStrategy.__init__

- **File:** `src/saaaaaa/core/orchestrator/executors.py:638`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `magic_number` = `5`
- `magic_number` = `0.1`
- `magic_number` = `0.05`

**Notes:** Contains 3 magic numbers

---

#### 18. src.saaaaaa.core.orchestrator.executors.MetaLearningStrategy.get_strategy_config

- **File:** `src/saaaaaa/core/orchestrator/executors.py:674`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `magic_number` = `10`
- `magic_number` = `5`
- `magic_number` = `20`

**Notes:** Contains 3 magic numbers

---

#### 19. src.saaaaaa.core.orchestrator.executors.AttentionMechanism.__init__

- **File:** `src/saaaaaa/core/orchestrator/executors.py:693`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 4

**Parameters:**
- `magic_number` = `64`
- `magic_number` = `0.01`
- `magic_number` = `0.01`
- `magic_number` = `0.01`

**Notes:** Contains 4 magic numbers

---

#### 20. src.saaaaaa.core.orchestrator.executors.ProbabilisticExecutor.get_credible_interval

- **File:** `src/saaaaaa/core/orchestrator/executors.py:922`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `magic_number` = `0.95`
- `magic_number` = `100`
- `magic_number` = `100`

**Notes:** Contains 3 magic numbers

---

#### 21. src.saaaaaa.core.orchestrator.executors.AdvancedDataFlowExecutor._fetch_signals

- **File:** `src/saaaaaa/core/orchestrator/executors.py:1073`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `1000`
- `magic_number` = `16`

**Notes:** Contains 2 magic numbers

---

#### 22. src.saaaaaa.core.orchestrator.executors.AdvancedDataFlowExecutor.execute_with_optimization

- **File:** `src/saaaaaa/core/orchestrator/executors.py:1167`
- **Pattern:** explicit_parameters
- **Complexity:** moderate
- **Parameters found:** 6

**Parameters:**
- `alpha` = `2`
- `beta` = `2`
- `max_retries` = `3`
- `magic_number` = `3`
- `magic_number` = `3`
- `magic_number` = `0.1`

**Notes:** Contains 3 magic numbers

---

#### 23. src.saaaaaa.core.orchestrator.executors.AdvancedDataFlowExecutor._assess_data_quality

- **File:** `src/saaaaaa/core/orchestrator/executors.py:1353`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 1

**Parameters:**
- `max_entropy` = `8`

**Notes:** Embedded parameters detected

---

#### 24. src.saaaaaa.core.orchestrator.executors.AdvancedDataFlowExecutor._fallback_for

- **File:** `src/saaaaaa/core/orchestrator/executors.py:1790`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `0.8`
- `magic_number` = `400`

**Notes:** Contains 2 magic numbers

---

#### 25. src.saaaaaa.core.orchestrator.executors.AdvancedDataFlowExecutor._compute_pattern_specificity

- **File:** `src/saaaaaa/core/orchestrator/executors.py:2078`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `magic_number` = `0.8`
- `magic_number` = `0.95`
- `magic_number` = `0.2`

**Notes:** Contains 3 magic numbers

---

#### 26. src.saaaaaa.core.orchestrator.executors.FrontierExecutorOrchestrator.__init__

- **File:** `src/saaaaaa/core/orchestrator/executors.py:3537`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 2

**Parameters:**
- `magic_number` = `30`
- `magic_number` = `10`

**Notes:** Contains 2 magic numbers

---

#### 27. src.saaaaaa.core.orchestrator.executors.FrontierExecutorOrchestrator._optimize_execution_order

- **File:** `src/saaaaaa/core/orchestrator/executors.py:3611`
- **Pattern:** explicit_parameters
- **Complexity:** simple
- **Parameters found:** 4

**Parameters:**
- `alpha` = `0`
- `magic_number` = `100`
- `magic_number` = `0.05`
- `magic_number` = `10`

**Notes:** Contains 3 magic numbers

---

#### 28. src.saaaaaa.core.orchestrator.core.Orchestrator._execute_micro_questions_async

- **File:** `src/saaaaaa/core/orchestrator/core.py:1787`
- **Pattern:** magic_numbers
- **Complexity:** simple
- **Parameters found:** 3

**Parameters:**
- `magic_number` = `10`
- `magic_number` = `1000.0`
- `magic_number` = `3`

**Notes:** Contains 3 magic numbers

---
