import math

from meso_cluster_analysis import (
    analyze_policy_dispersion,
    calibrate_against_peers,
    compose_cluster_posterior,
    reconcile_cross_metrics,
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


def test_compose_cluster_posterior_handles_negative_weights():
    """Test that negative weights are clamped to zero and uniform fallback is applied."""
    micro_posteriors = [0.6, 0.8, 0.7]
    weights = [-0.2, 0.5, -0.3]  # negative weights that should be clamped
    
    payload, explanation = compose_cluster_posterior(
        micro_posteriors, weighting_trace=weights
    )
    
    # Should not raise ZeroDivisionError
    assert payload["prior_meso"] > 0
    assert payload["posterior_meso"] > 0
    assert payload["uncertainty_index"] >= 0
    assert "posterior" in explanation


def test_compose_cluster_posterior_handles_zero_total_weight():
    """Test that weights summing to zero before clamping are handled gracefully."""
    micro_posteriors = [0.6, 0.8, 0.7]
    weights = [0.5, -0.5, 0.0]  # sum to zero before clamping
    
    payload, explanation = compose_cluster_posterior(
        micro_posteriors, weighting_trace=weights
    )
    
    # Clamping converts weights to [0.5, 0.0, 0.0] (sum=0.5).
    # After normalization: [1.0, 0.0, 0.0], so only first posterior counts.
    assert payload["prior_meso"] > 0
    # With normalized weights [1.0, 0.0, 0.0], prior_meso equals first posterior
    assert math.isclose(payload["prior_meso"], 0.6, rel_tol=1e-5)
    assert payload["posterior_meso"] > 0
    assert payload["uncertainty_index"] >= 0


def test_compose_cluster_posterior_handles_all_negative_weights():
    """Test that all-negative weights fall back to uniform."""
    micro_posteriors = [0.6, 0.8, 0.7]
    weights = [-1.0, -2.0, -0.5]  # all negative
    
    payload, explanation = compose_cluster_posterior(
        micro_posteriors, weighting_trace=weights
    )
    
    # After clamping to zero, total is 0, should fall back to uniform
    assert payload["prior_meso"] > 0
    assert math.isclose(payload["prior_meso"], 0.7, rel_tol=1e-5)
    assert payload["posterior_meso"] > 0


def test_compose_cluster_posterior_handles_all_zero_weights_after_clamping():
    """Test that weights that become all zero after clamping fall back to uniform."""
    micro_posteriors = [0.6, 0.8, 0.7]
    weights = [-1.0, 0.0, -0.5]  # all non-positive, become [0.0, 0.0, 0.0]
    
    payload, explanation = compose_cluster_posterior(
        micro_posteriors, weighting_trace=weights
    )
    
    # After clamping to [0.0, 0.0, 0.0], total is 0, should fall back to uniform
    assert payload["prior_meso"] > 0
    # With uniform weights, prior_meso should be (0.6 + 0.8 + 0.7) / 3 = 0.7
    assert math.isclose(payload["prior_meso"], 0.7, rel_tol=1e-5)
    assert payload["posterior_meso"] > 0


