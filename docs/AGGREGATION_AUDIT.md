# Aggregation Module Audit

## Scope
This audit reviews the end-to-end aggregation flow covering:

* Data models defined in `src/saaaaaa/processing/aggregation.py`
* Dimension, area, cluster, and macro aggregation responsibilities and guarantees
* How the orchestrator (`src/saaaaaa/core/orchestrator/core.py`) composes aggregation steps
* Alignment between implementation, configuration, and validation hooks

## Data Models
* `ScoredResult`, `DimensionScore`, `AreaScore`, `ClusterScore`, and `MacroScore` expose rich metadata fields (e.g., `validation_details`, `systemic_gaps`) meant for diagnostics and downstream reporting. None of the models enforce presence of configuration-derived fields such as `score_max`, leaving normalisation helpers to rely on defaults.
* Validation flags (`validation_passed`) default to `True`, so any missing validation result must be explicitly flipped to `False` when errors occur.

## Dimension Aggregation (FASE 4)
* `DimensionAggregator.validate_weights` enforces a strict sum-to-one check when explicit weights are provided and aborts when configured to do so, but returns `(False, "No weights provided")` for empty inputs instead of surfacing a structured exception, requiring callers to inspect the tuple before aggregating.【F:src/saaaaaa/processing/aggregation.py†L154-L214】
* Coverage enforcement assumes **exactly five** micro-questions per dimension via the `expected_count` default. The monolith stores the real mapping (see `niveles_abstraccion.dimensions[*].dimension_id`), yet `aggregate_dimension` never consults configuration, so any evolution (e.g., adaptive questionnaires) will be flagged as coverage failures regardless of specification.【F:src/saaaaaa/processing/aggregation.py†L183-L351】
* Rubric thresholds are hard-coded (>=0.85/0.70/0.55) even though the method accepts an optional `thresholds` payload and the monolith contains rubric definitions. This limits configurability and can diverge from policy-defined bands.【F:src/saaaaaa/processing/aggregation.py†L261-L297】
* Weight application relies on `zip(scores, weights)` and raises `WeightValidationError` if the lengths differ, but returning `0.0` after logging creates silent failure paths when aborts are disabled, because the downstream `DimensionScore` will show a valid structure yet carries a zeroed aggregate.【F:src/saaaaaa/processing/aggregation.py†L216-L407】

## Policy Area Aggregation (FASE 5)
* Hermeticity validation checks for **exactly** `len(self.dimensions)` results, effectively requiring every area to report all global dimensions. The monolith maps each policy area to a subset (`policy_area.dimension_ids`), which would allow scoped aggregation, but the implementation ignores those IDs and treats missing-but-irrelevant dimensions as fatal gaps.【F:src/saaaaaa/processing/aggregation.py†L411-L650】【F:data/questionnaire_monolith.json†L34083-L34105】
* Score normalisation looks for `score_max` in each `DimensionScore.validation_details`; however, `DimensionAggregator` never supplies this key, so the helper always falls back to `3.0`. Configuration-driven maxima (if introduced later) would be silently ignored.【F:src/saaaaaa/processing/aggregation.py†L485-L618】
* Area-level rubric thresholds mirror the dimension defaults and similarly ignore the optional `thresholds` argument, locking the quality bands to the baked-in constants.【F:src/saaaaaa/processing/aggregation.py†L504-L540】

## Cluster Aggregation (FASE 6)
* Cluster definitions are retrieved from `niveles_abstraccion.clusters` and hermeticity checks verify both missing and unexpected policy areas, which protects against mis-wiring. However, duplicates inside `area_scores` are not detected, so repeated area IDs pass validation.【F:src/saaaaaa/processing/aggregation.py†L682-L729】
* `apply_cluster_weights` verifies that the weights sum to one but does **not** ensure the weight vector length matches the number of area scores; Python's `zip` truncates silently when the sequences differ, letting partial data bias the aggregate without triggering an exception.【F:src/saaaaaa/processing/aggregation.py†L730-L769】
* Coherence calculations and rubric handling inherit the same hard-coded thresholds, again missing the opportunity to source cluster-specific bands from configuration.【F:src/saaaaaa/processing/aggregation.py†L770-L937】

## Macro Aggregation (FASE 7)
* Macro evaluation averages cluster scores without weights or confidence factors; the monolith contains strategic metadata per cluster (e.g., rationales) that could support differentiated importance, but the implementation treats all clusters equally.【F:src/saaaaaa/processing/aggregation.py†L1100-L1182】【F:data/questionnaire_monolith.json†L34106-L34141】
* Cross-cutting coherence and systemic gap detection are fully deterministic but rely on earlier stages populating `quality_level` and `validation_passed`. Any upstream short-circuit (e.g., returning empty lists after validation failures) yields an empty macro output with `validation_passed=False`, yet there is no mechanism to bubble up root causes beyond the final `validation_details` map.【F:src/saaaaaa/processing/aggregation.py†L968-L1182】
* Rubric thresholds remain hard-coded, mirroring the same configurability gap noted at lower levels.【F:src/saaaaaa/processing/aggregation.py†L1061-L1165】

## Orchestrator Path Divergence
* The async orchestrator bypasses the rich aggregator classes entirely. Instead, `_aggregate_dimensions_async`, `_aggregate_policy_areas_async`, `_aggregate_clusters`, and `_evaluate_macro` compute simple averages without any of the validation, coverage, or hermeticity safeguards described above.【F:src/saaaaaa/core/orchestrator/core.py†L1456-L1572】
* Because the orchestrator does not construct `DimensionScore`/`AreaScore` objects, downstream consumers expecting diagnostic fields (`validation_details`, `coherence`, etc.) receive minimal dictionaries. This divergence complicates debugging and creates a dual implementation surface to maintain.【F:src/saaaaaa/core/orchestrator/core.py†L1456-L1572】

## Testing Gaps
* `tests/test_aggregation.py` defines fixtures referencing fields like `DimensionScore.micro_scores` that no longer exist, indicating the test suite is outdated relative to the current dataclasses. The stale expectations hide real-world regressions because the modern aggregators are not exercised end-to-end.【F:tests/test_aggregation.py†L70-L120】

## Key Risks & Recommendations
1. **Configuration Drift** – Inject rubric thresholds, weight vectors, and expected question counts from the monolith instead of hard-coding constants across all stages.
2. **Single Source of Truth** – Refactor the orchestrator to instantiate the shared aggregators, ensuring that validations execute consistently across batch and async flows.
3. **Weight-Length Validation** – Extend cluster-level weight handling to reject mismatched vectors before aggregation to avoid silent truncation.
4. **Test Refresh** – Update the aggregation tests to target the current dataclasses and orchestrator behaviour, closing the gap between documentation and executable coverage.
