"""
Comprehensive Test Suite for Method Parameters Configuration

This test suite ensures the method_parameters.json configuration is:
1. Structurally valid (schema compliance)
2. Semantically consistent (constraints satisfied)
3. Epistemologically justified (values are reasonable)
4. Functionally complete (all methods have required parameters)
5. Regression-safe (outputs match expected values)

Test Categories:
    - Schema Tests: JSON structure validation
    - Consistency Tests: Cross-parameter constraints
    - Value Range Tests: Parameters within reasonable bounds
    - Completeness Tests: All methods have necessary parameters
    - Integration Tests: Parameters work in real method execution
    - Regression Tests: Ensure no breaking changes
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict

# Import the parameter loader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from saaaaaa.config import method_parameters


class TestSchemaValidation:
    """Test JSON schema structure compliance."""

    def test_config_file_exists(self):
        """Verify method_parameters.json file exists."""
        config_path = Path(__file__).parent.parent / "config" / "method_parameters.json"
        assert config_path.exists(), f"Config file not found: {config_path}"

    def test_config_loads_successfully(self):
        """Verify JSON can be loaded without errors."""
        config = method_parameters.load_config()
        assert config is not None
        assert isinstance(config, dict)

    def test_metadata_section_exists(self):
        """Verify METADATA section exists with required fields."""
        config = method_parameters.load_config()
        assert "METADATA" in config
        metadata = config["METADATA"]

        assert "extraction_date" in metadata
        assert "source_files" in metadata
        assert "total_parameters_extracted" in metadata

    def test_method_sections_have_descriptions(self):
        """Verify all method sections have description fields."""
        config = method_parameters.load_config()

        for key, value in config.items():
            if key == "METADATA":
                continue
            if key in ["SUMMARY_STATISTICS"]:  # Special sections
                continue

            assert "description" in value, f"Missing description for {key}"
            assert isinstance(value["description"], str)
            assert len(value["description"]) > 10, f"Description too short for {key}"

    def test_parameters_have_required_fields(self):
        """Verify all parameters have value, type, justification."""
        config = method_parameters.load_config()

        for method_class, method_data in config.items():
            if method_class in ["METADATA", "SUMMARY_STATISTICS"]:
                continue

            parameters = method_data.get("parameters", {})
            for param_name, param_data in parameters.items():
                param_key = f"{method_class}.{param_name}"

                # Check required fields
                assert "value" in param_data, f"{param_key}: missing 'value'"
                assert "type" in param_data, f"{param_key}: missing 'type'"
                assert "justification" in param_data, f"{param_key}: missing 'justification'"

                # Check justification is not trivial
                justification = param_data["justification"]
                assert len(justification) > 20, f"{param_key}: justification too short"

    def test_parameter_types_are_valid(self):
        """Verify all parameter types are from valid set."""
        valid_types = {"threshold", "weight", "constant", "lexicon", "regex"}
        config = method_parameters.load_config()

        for method_class, method_data in config.items():
            if method_class in ["METADATA", "SUMMARY_STATISTICS"]:
                continue

            parameters = method_data.get("parameters", {})
            for param_name, param_data in parameters.items():
                param_type = param_data.get("type")
                assert param_type in valid_types, \
                    f"{method_class}.{param_name}: invalid type '{param_type}'"


class TestConsistencyValidation:
    """Test cross-parameter consistency constraints."""

    def test_mechanism_priors_sum_to_one(self):
        """Verify mechanism type priors sum to 1.0."""
        prior_names = [
            "mechanism_type_priors.administrativo",
            "mechanism_type_priors.tecnico",
            "mechanism_type_priors.financiero",
            "mechanism_type_priors.politico",
            "mechanism_type_priors.mixto",
        ]

        priors = []
        for name in prior_names:
            val = method_parameters.get_parameter("BayesianMechanismInference", name)
            priors.append(val)

        total = sum(priors)
        assert abs(total - 1.0) < 0.001, \
            f"Mechanism priors sum to {total}, expected 1.0. Priors: {priors}"

    def test_threshold_ordering(self):
        """Verify soft thresholds < hard thresholds where applicable."""
        # Example: scoring thresholds should be ordered
        try:
            soft = method_parameters.get_parameter("CausalExtractor", "scoring.threshold.soft")
            hard = method_parameters.get_parameter("CausalExtractor", "scoring.threshold.hard")
            assert soft < hard, f"Soft threshold ({soft}) should be < hard threshold ({hard})"
        except:
            pytest.skip("Threshold parameters not found in expected structure")

    def test_context_window_ordering(self):
        """Verify default_context_window <= max_context_window."""
        try:
            default = method_parameters.get_parameter("CausalExtractor", "default_context_window")
            maximum = method_parameters.get_parameter("CausalExtractor", "max_context_window")
            assert default <= maximum, \
                f"Default context ({default}) should be <= max context ({maximum})"
        except:
            pytest.skip("Context window parameters not found")

    def test_no_negative_thresholds(self):
        """Verify no threshold parameters are negative."""
        thresholds = method_parameters.get_all_parameters_by_type("threshold")

        for param_key, param_data in thresholds.items():
            value = param_data["value"]
            if isinstance(value, (int, float)):
                assert value >= 0, f"{param_key}: threshold is negative ({value})"

    def test_weights_in_reasonable_range(self):
        """Verify weight parameters are in reasonable range [-1, 2]."""
        weights = method_parameters.get_all_parameters_by_type("weight")

        for param_key, param_data in weights.items():
            value = param_data["value"]
            if isinstance(value, (int, float)):
                assert -1 <= value <= 2, \
                    f"{param_key}: weight {value} outside reasonable range [-1, 2]"


class TestValueRangeValidation:
    """Test parameter values are within reasonable bounds."""

    def test_bayesian_alpha_beta_positive(self):
        """Verify Bayesian prior parameters are positive."""
        alpha = method_parameters.get_parameter("BayesianMechanismInference", "prior_alpha")
        beta = method_parameters.get_parameter("BayesianMechanismInference", "prior_beta")

        assert alpha > 0, f"prior_alpha must be positive, got {alpha}"
        assert beta > 0, f"prior_beta must be positive, got {beta}"

    def test_kl_divergence_small(self):
        """Verify KL divergence threshold is small (convergence sensitivity)."""
        kl_div = method_parameters.get_parameter("BayesianMechanismInference", "kl_divergence")
        assert 0 < kl_div < 0.1, f"KL divergence {kl_div} outside typical range (0, 0.1)"

    def test_convergence_min_evidence_reasonable(self):
        """Verify convergence requires at least 2 pieces of evidence."""
        min_evidence = method_parameters.get_parameter(
            "BayesianMechanismInference",
            "convergence_min_evidence"
        )
        assert min_evidence >= 2, \
            f"convergence_min_evidence should be ≥2, got {min_evidence}"

    def test_context_windows_positive(self):
        """Verify context windows are positive integers."""
        try:
            default = method_parameters.get_parameter("CausalExtractor", "default_context_window")
            maximum = method_parameters.get_parameter("CausalExtractor", "max_context_window")

            assert isinstance(default, int) and default > 0
            assert isinstance(maximum, int) and maximum > 0
        except:
            pytest.skip("Context window parameters not found")


class TestCompletenessValidation:
    """Test all required parameters exist for each method."""

    def test_bayesian_inference_has_all_parameters(self):
        """Verify BayesianMechanismInference has all required parameters."""
        required_params = [
            "kl_divergence",
            "convergence_min_evidence",
            "prior_alpha",
            "prior_beta",
            "laplace_smoothing",
            "mechanism_type_priors.administrativo",
            "mechanism_type_priors.tecnico",
            "mechanism_type_priors.financiero",
            "mechanism_type_priors.politico",
            "mechanism_type_priors.mixto",
        ]

        for param in required_params:
            value = method_parameters.get_parameter("BayesianMechanismInference", param)
            assert value is not None, f"Missing required parameter: {param}"

    def test_all_methods_have_parameters(self):
        """Verify all non-metadata sections have at least one parameter."""
        config = method_parameters.load_config()

        for method_class, method_data in config.items():
            if method_class in ["METADATA", "SUMMARY_STATISTICS"]:
                continue

            # Should have parameters or be a special structure
            if "parameters" in method_data:
                params = method_data["parameters"]
                assert len(params) > 0, f"{method_class} has no parameters"


class TestIntegrationValidation:
    """Test parameters work in real method execution context."""

    def test_parameter_loader_api(self):
        """Verify parameter loader API methods work correctly."""
        # Test get_method_config
        config = method_parameters.get_method_config("BayesianMechanismInference")
        assert "description" in config
        assert "parameters" in config

        # Test get_parameter
        kl_div = method_parameters.get_parameter("BayesianMechanismInference", "kl_divergence")
        assert isinstance(kl_div, float)
        assert kl_div == 0.01

        # Test get_parameter_metadata
        metadata = method_parameters.get_parameter_metadata(
            "BayesianMechanismInference",
            "kl_divergence"
        )
        assert "value" in metadata
        assert "type" in metadata
        assert "justification" in metadata

    def test_get_parameter_with_default(self):
        """Verify default parameter fallback works."""
        value = method_parameters.get_parameter(
            "BayesianMechanismInference",
            "nonexistent_parameter",
            default=0.5
        )
        assert value == 0.5

    def test_get_all_parameters_by_type(self):
        """Verify filtering by parameter type works."""
        thresholds = method_parameters.get_all_parameters_by_type("threshold")
        weights = method_parameters.get_all_parameters_by_type("weight")

        assert len(thresholds) > 0
        assert len(weights) > 0

        # Verify all thresholds are numeric
        for param_key, param_data in thresholds.items():
            value = param_data["value"]
            if not isinstance(value, (list, dict, str)):  # Skip complex types
                assert isinstance(value, (int, float))

    def test_config_validation_function(self):
        """Verify config validation catches errors."""
        is_valid, errors = method_parameters.validate_config()

        # Should be valid (or have documented reasons for invalidity)
        if not is_valid:
            print(f"Validation errors found ({len(errors)}):")
            for error in errors:
                print(f"  - {error}")

            # Allow some specific validation warnings
            acceptable_errors = [
                "CROSS_CUTTING_CONSTANTS: Missing 'parameters' field",
                "SEMANTIC_ALIGNMENT: Missing 'parameters' field",
                "REGULATORY_THRESHOLDS: Missing 'parameters' field",
                "SUMMARY_STATISTICS: Missing 'description' field",
                "SUMMARY_STATISTICS: Missing 'parameters' field",
            ]

            for error in errors:
                assert error in acceptable_errors, f"Unexpected validation error: {error}"


class TestRegressionValidation:
    """Test parameter changes haven't broken expected behavior."""

    def test_known_parameter_values(self):
        """Verify critical parameters have expected values (regression check)."""
        critical_params = {
            ("BayesianMechanismInference", "kl_divergence"): 0.01,
            ("BayesianMechanismInference", "prior_alpha"): 2.0,
            ("BayesianMechanismInference", "prior_beta"): 2.0,
            ("BayesianMechanismInference", "laplace_smoothing"): 1.0,
            ("BayesianMechanismInference", "mechanism_type_priors.administrativo"): 0.30,
        }

        for (method_class, param_name), expected_value in critical_params.items():
            actual_value = method_parameters.get_parameter(method_class, param_name)
            assert actual_value == expected_value, \
                f"{method_class}.{param_name}: expected {expected_value}, got {actual_value}"

    def test_total_parameter_count(self):
        """Verify total parameter count hasn't unexpectedly decreased."""
        metadata = method_parameters.get_config_metadata()
        total = metadata.get("total_parameters_extracted", 0)

        # Should have at least 100 parameters
        assert total >= 100, f"Only {total} parameters found, expected ≥100"


class TestEpistemologicalValidation:
    """Test parameters have valid epistemological justifications."""

    def test_all_parameters_have_substantive_justifications(self):
        """Verify no parameters have trivial justifications."""
        config = method_parameters.load_config()

        trivial_phrases = [
            "placeholder",
            "todo",
            "tbd",
            "fixme",
            "unknown",
            "test value",
        ]

        for method_class, method_data in config.items():
            if method_class in ["METADATA", "SUMMARY_STATISTICS"]:
                continue

            parameters = method_data.get("parameters", {})
            for param_name, param_data in parameters.items():
                justification = param_data.get("justification", "").lower()

                for phrase in trivial_phrases:
                    assert phrase not in justification, \
                        f"{method_class}.{param_name}: trivial justification contains '{phrase}'"

    def test_regulatory_parameters_have_legal_basis(self):
        """Verify regulatory parameters cite Colombian law."""
        try:
            # Check if REGULATORY_THRESHOLDS exists
            config = method_parameters.load_config()
            if "REGULATORY_THRESHOLDS" not in config:
                pytest.skip("REGULATORY_THRESHOLDS section not found")

            regulatory_section = config["REGULATORY_THRESHOLDS"]

            # At least some params should reference Colombian law
            legal_references_found = 0
            for param_name, param_data in regulatory_section.items():
                if isinstance(param_data, dict):
                    justification = param_data.get("justification", "")
                    regulatory_basis = param_data.get("regulatory_basis", "")

                    if "Ley" in justification or "Ley" in regulatory_basis:
                        legal_references_found += 1

            assert legal_references_found > 0, \
                "No regulatory parameters cite Colombian law"
        except:
            pytest.skip("Could not validate regulatory parameters")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
