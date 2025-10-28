import math

import pytest

from meso_cluster_analysis import (
    analyze_policy_dispersion,
    calibrate_against_peers,
    compose_cluster_posterior,
    reconcile_cross_metrics,
    _gini,
    _tukey_bounds,
)


def test_analyze_policy_dispersion_generates_penalty_and_projection():
    policy_area_scores = {
        "salud": 62.0,
        "educacion": 78.0,
        "infraestructura": 45.0,
        "ambiente": 55.0,
    }
    peer_dispersion_stats = {"cv_median": 0.18, "gap_median": 22.0}
    thresholds = {"cv_warn": 0.15, "cv_fail": 0.30, "gap_warn": 18.0, "gap_fail": 35.0}

    payload, narrative = analyze_policy_dispersion(
        policy_area_scores, peer_dispersion_stats, thresholds
    )

    assert set(payload.keys()) == {
        "cv",
        "max_gap",
        "gini",
        "class",
        "penalty",
        "normalized_projection",
    }
    assert payload["cv"] > 0
    assert payload["max_gap"] == max(policy_area_scores.values()) - min(policy_area_scores.values())
    assert payload["normalized_projection"]["adjusted_cv"] <= payload["cv"]
    assert payload["normalized_projection"]["adjusted_max_gap"] <= payload["max_gap"]
    assert 0.0 <= payload["penalty"] <= 1.0
    assert len(narrative.splitlines()) >= 5


def test_analyze_policy_dispersion_handles_zero_mean_cluster():
    policy_area_scores = {"area_a": 0.0, "area_b": 0.0, "area_c": 0.0}
    peer_dispersion_stats = {"cv_median": 0.2, "gap_median": 5.0}
    thresholds = {"cv_warn": 0.1, "cv_fail": 0.2, "gap_warn": 3.0, "gap_fail": 6.0}

    payload, narrative = analyze_policy_dispersion(
        policy_area_scores, peer_dispersion_stats, thresholds
    )

    assert payload["cv"] == 0.0
    assert payload["normalized_projection"]["adjusted_cv"] == 0.0
    assert payload["normalized_projection"]["adjusted_max_gap"] == 0.0
    assert "0.00" in narrative


def test_reconcile_cross_metrics_flags_mismatches_and_converts_units():
    aggregated_metrics = [
        {"metric_id": "m1", "value": 1.5, "unit": "toneladas", "period": "2022", "entity": "A"},
        {"metric_id": "m2", "value": 85, "unit": "porcentaje", "period": "2023", "entity": "B"},
    ]
    macro_reference = {
        "metrics": {
            "m1": {"unit": "kg", "period": "2023", "entities": ["A", "C"], "range": (0, 2000)},
            "m2": {"unit": "porcentaje", "period": "2023", "entities": ["B"], "range": (0, 100)},
        },
        "unit_crosswalk": {"toneladas": {"kg": 1000.0}},
    }

    result = reconcile_cross_metrics(aggregated_metrics, macro_reference)

    validated = {item["metric_id"]: item for item in result["metrics_validated"]}
    assert math.isclose(validated["m1"]["value"], 1500.0)
    assert validated["m1"]["unit"] == "kg"
    assert result["violations"]
    violation_ids = {item["metric_id"] for item in result["violations"]}
    assert "m1" in violation_ids  # stale period should be flagged
    assert 0.0 <= result["reconciled_confidence"] <= 1.0


def test_compose_cluster_posterior_applies_penalties_and_weights():
    micro_posteriors = [0.6, 0.8, 0.7]
    weights = [0.2, 0.5, 0.3]
    penalties = {
        "dispersion_penalty": 0.1,
        "coverage_penalty": 0.05,
        "reconciliation_penalty": 0.02,
    }

    payload, explanation = compose_cluster_posterior(
        micro_posteriors, weighting_trace=weights, reconciliation_penalties=penalties
    )

    assert math.isclose(payload["prior_meso"], 0.73, rel_tol=1e-5)
    expected_factor = 0.9 * 0.95 * 0.98
    assert math.isclose(payload["posterior_meso"], payload["prior_meso"] * expected_factor, rel_tol=1e-5)
    assert payload["uncertainty_index"] > 0
    assert "posterior" in explanation


def test_calibrate_against_peers_detects_outliers_and_positions():
    scores = {"salud": 70, "educacion": 90, "vivienda": 40}
    peer_context = {
        "salud": {"median": 65, "p25": 55, "p75": 75},
        "educacion": {"median": 78, "p25": 70, "p75": 85},
        "vivienda": {"median": 60, "p25": 55, "p75": 70},
    }

    payload, narrative = calibrate_against_peers(scores, peer_context)

    assert payload["area_positions"]["salud"] == "within"
    assert payload["area_positions"]["educacion"] == "above"
    assert payload["area_positions"]["vivienda"] == "below"
    assert len(payload["outliers"]) == len(scores)
    assert len(narrative.splitlines()) >= 6


def test_gini_rejects_negative_values():
    with pytest.raises(ValueError):
        _gini([0.5, -0.2, 0.7])


def test_tukey_bounds_are_order_invariant():
    lower, upper = _tukey_bounds(80, 20)
    assert lower < upper
    expected_lower, expected_upper = _tukey_bounds(20, 80)
    assert math.isclose(lower, expected_lower)
    assert math.isclose(upper, expected_upper)


def test_reconcile_cross_metrics_handles_non_convertible_units():
    aggregated_metrics = [
        {"metric_id": "m_nc", "value": 10, "unit": "litros", "period": "2023", "entity": "X"}
    ]
    macro_reference = {
        "metrics": {
            "m_nc": {
                "unit": "kg",
                "period": "2023",
                "entities": ["X"],
                "range": (0, 100),
            }
        },
        "unit_crosswalk": {},
    }

    result = reconcile_cross_metrics(aggregated_metrics, macro_reference)

    assert result["violations"]
    violation = result["violations"][0]
    assert violation["metric_id"] == "m_nc"
    assert violation["unit_mismatch"] is True
    assert violation["out_of_range"] is False


def test_reconcile_cross_metrics_respects_range_boundaries():
    aggregated_metrics = [
        {"metric_id": "m_range", "value": 100.0, "unit": "kg", "period": "2023", "entity": "A"}
    ]
    macro_reference = {
        "metrics": {
            "m_range": {
                "unit": "kg",
                "period": "2023",
                "entities": ["A"],
                "range": (0.0, 100.0),
            }
        }
    }

    result = reconcile_cross_metrics(aggregated_metrics, macro_reference)

    assert result["violations"] == []
    validated = result["metrics_validated"][0]
    assert math.isclose(validated["value"], 100.0)


def test_compose_cluster_posterior_rejects_negative_weights():
    with pytest.raises(ValueError):
        compose_cluster_posterior([0.4, 0.6], weighting_trace=[0.5, -0.1])

