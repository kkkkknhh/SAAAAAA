"""
Method Parameter Configuration Module

This module provides the authoritative source for all method-specific parameters
in the SAAAAAA system. It replaces scattered YAML configurations with a single
JSON-based configuration system.

Usage:
    from saaaaaa.config import method_parameters

    # Get all parameters for a method
    params = method_parameters.get_method_config("BayesianMechanismInference")

    # Get a specific parameter
    kl_div = method_parameters.get_parameter("BayesianMechanismInference", "kl_divergence")

    # Validate parameter schema
    method_parameters.validate_config()
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global configuration cache
_CONFIG_CACHE: Optional[dict] = None
_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "config" / "method_parameters.json"


class ParameterConfigError(Exception):
    """Raised when parameter configuration is invalid or missing."""
    pass


def load_config(force_reload: bool = False) -> dict:
    """
    Load method parameters from JSON configuration file.

    Args:
        force_reload: If True, bypass cache and reload from disk

    Returns:
        Complete configuration dictionary

    Raises:
        ParameterConfigError: If config file is missing or invalid
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    if not _CONFIG_PATH.exists():
        raise ParameterConfigError(
            f"Configuration file not found: {_CONFIG_PATH}\n"
            f"Expected location: config/method_parameters.json"
        )

    try:
        with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)

        logger.info(f"Loaded method parameters from {_CONFIG_PATH}")
        logger.info(f"Configuration version: {config.get('METADATA', {}).get('extraction_date', 'unknown')}")

        _CONFIG_CACHE = config
        return config

    except json.JSONDecodeError as e:
        raise ParameterConfigError(f"Invalid JSON in configuration file: {e}")
    except Exception as e:
        raise ParameterConfigError(f"Failed to load configuration: {e}")


def get_method_config(method_class: str) -> dict:
    """
    Get all parameters for a specific method class.

    Args:
        method_class: Name of the method class (e.g., "BayesianMechanismInference")

    Returns:
        Dictionary with 'description' and 'parameters' keys

    Raises:
        ParameterConfigError: If method class not found

    Example:
        >>> config = get_method_config("BayesianMechanismInference")
        >>> config['parameters']['kl_divergence']['value']
        0.01
    """
    config = load_config()

    if method_class not in config:
        available = [k for k in config.keys() if k != 'METADATA']
        raise ParameterConfigError(
            f"Method class '{method_class}' not found in configuration.\n"
            f"Available classes: {', '.join(available)}"
        )

    return config[method_class]


def get_parameter(method_class: str, parameter_name: str, default: Any = None) -> Any:
    """
    Get a specific parameter value for a method class.

    Args:
        method_class: Name of the method class
        parameter_name: Name of the parameter
        default: Default value if parameter not found (if None, raises error)

    Returns:
        Parameter value (typically from the 'value' field)

    Raises:
        ParameterConfigError: If parameter not found and no default provided

    Example:
        >>> kl_div = get_parameter("BayesianMechanismInference", "kl_divergence")
        >>> kl_div
        0.01
    """
    method_config = get_method_config(method_class)
    parameters = method_config.get('parameters', {})

    if parameter_name not in parameters:
        if default is not None:
            logger.warning(
                f"Parameter '{parameter_name}' not found for '{method_class}', "
                f"using default: {default}"
            )
            return default

        available = list(parameters.keys())
        raise ParameterConfigError(
            f"Parameter '{parameter_name}' not found for '{method_class}'.\n"
            f"Available parameters: {', '.join(available[:10])}..."
        )

    return parameters[parameter_name].get('value')


def get_parameter_metadata(method_class: str, parameter_name: str) -> dict:
    """
    Get complete metadata for a parameter (value, type, justification, etc.).

    Args:
        method_class: Name of the method class
        parameter_name: Name of the parameter

    Returns:
        Complete parameter dictionary with all metadata

    Example:
        >>> meta = get_parameter_metadata("BayesianMechanismInference", "kl_divergence")
        >>> meta['justification']
        'KL divergence threshold for Bayesian convergence detection...'
    """
    method_config = get_method_config(method_class)
    parameters = method_config.get('parameters', {})

    if parameter_name not in parameters:
        raise ParameterConfigError(
            f"Parameter '{parameter_name}' not found for '{method_class}'"
        )

    return parameters[parameter_name]


def get_all_parameters_by_type(parameter_type: str) -> dict:
    """
    Get all parameters of a specific type across all method classes.

    Args:
        parameter_type: Type to filter by ('threshold', 'weight', 'constant', 'lexicon')

    Returns:
        Dictionary mapping (method_class, parameter_name) to parameter metadata

    Example:
        >>> thresholds = get_all_parameters_by_type('threshold')
        >>> len(thresholds)
        42
    """
    config = load_config()
    results = {}

    for method_class, method_data in config.items():
        if method_class == 'METADATA':
            continue

        parameters = method_data.get('parameters', {})
        for param_name, param_data in parameters.items():
            if param_data.get('type') == parameter_type:
                key = f"{method_class}.{param_name}"
                results[key] = param_data

    return results


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration file structure and parameter consistency.

    Returns:
        (is_valid, error_messages) tuple

    Validation checks:
        - All method classes have 'description' and 'parameters' fields
        - All parameters have 'value', 'type', and 'justification' fields
        - All 'type' values are valid ('threshold', 'weight', 'constant', 'lexicon', 'regex')
        - Threshold values are numeric
        - Weight values are numeric and in reasonable range
        - Prior probabilities sum to 1.0 (within tolerance)
    """
    errors = []
    config = load_config()

    valid_types = {'threshold', 'weight', 'constant', 'lexicon', 'regex'}

    for method_class, method_data in config.items():
        if method_class == 'METADATA':
            continue

        # Check method structure
        if 'description' not in method_data:
            errors.append(f"{method_class}: Missing 'description' field")

        if 'parameters' not in method_data:
            errors.append(f"{method_class}: Missing 'parameters' field")
            continue

        parameters = method_data['parameters']

        # Check each parameter
        for param_name, param_data in parameters.items():
            param_key = f"{method_class}.{param_name}"

            # Required fields
            if 'value' not in param_data:
                errors.append(f"{param_key}: Missing 'value' field")

            if 'type' not in param_data:
                errors.append(f"{param_key}: Missing 'type' field")
            elif param_data['type'] not in valid_types:
                errors.append(
                    f"{param_key}: Invalid type '{param_data['type']}'. "
                    f"Must be one of: {valid_types}"
                )

            if 'justification' not in param_data:
                errors.append(f"{param_key}: Missing 'justification' field")

            # Type-specific validation
            param_type = param_data.get('type')
            param_value = param_data.get('value')

            if param_type in {'threshold', 'weight'}:
                if not isinstance(param_value, (int, float)):
                    errors.append(
                        f"{param_key}: Expected numeric value for {param_type}, "
                        f"got {type(param_value).__name__}"
                    )
                elif param_type == 'weight' and not (-1 <= param_value <= 2):
                    errors.append(
                        f"{param_key}: Weight value {param_value} outside reasonable range [-1, 2]"
                    )

    # Check mechanism type priors sum to 1.0
    mechanism_priors = {}
    for param_name in ['mechanism_type_priors.administrativo', 'mechanism_type_priors.tecnico',
                       'mechanism_type_priors.financiero', 'mechanism_type_priors.politico',
                       'mechanism_type_priors.mixto']:
        try:
            mechanism_priors[param_name] = get_parameter("BayesianMechanismInference", param_name)
        except:
            pass

    if mechanism_priors:
        prior_sum = sum(mechanism_priors.values())
        if abs(prior_sum - 1.0) > 0.001:
            errors.append(
                f"Mechanism type priors sum to {prior_sum:.4f}, expected 1.0. "
                f"Priors: {mechanism_priors}"
            )

    return (len(errors) == 0, errors)


def get_config_metadata() -> dict:
    """
    Get configuration metadata (version, source files, etc.).

    Returns:
        Metadata dictionary
    """
    config = load_config()
    return config.get('METADATA', {})


# Convenience function for backward compatibility with YAML-based systems
def get_derek_beach_config() -> dict:
    """
    Get Derek Beach CDAF configuration (legacy compatibility).

    Returns:
        Dictionary with bayesian_thresholds, mechanism_type_priors, etc.
    """
    config = get_method_config("BayesianMechanismInference")
    params = config['parameters']

    return {
        'bayesian_thresholds': {
            'kl_divergence': params['kl_divergence']['value'],
            'convergence_min_evidence': params['convergence_min_evidence']['value'],
            'prior_alpha': params['prior_alpha']['value'],
            'prior_beta': params['prior_beta']['value'],
            'laplace_smoothing': params['laplace_smoothing']['value'],
        },
        'mechanism_type_priors': {
            'administrativo': params['mechanism_type_priors.administrativo']['value'],
            'tecnico': params['mechanism_type_priors.tecnico']['value'],
            'financiero': params['mechanism_type_priors.financiero']['value'],
            'politico': params['mechanism_type_priors.politico']['value'],
            'mixto': params['mechanism_type_priors.mixto']['value'],
        },
        'self_reflection': {
            'feedback_weight': params['feedback_weight']['value'],
            'min_documents_for_learning': params['min_documents_for_learning']['value'],
        }
    }


if __name__ == "__main__":
    # Self-test when run directly
    print("=== Method Parameters Configuration Test ===\n")

    try:
        metadata = get_config_metadata()
        print(f"✅ Configuration loaded successfully")
        print(f"   Source date: {metadata.get('extraction_date')}")
        print(f"   Total parameters: {metadata.get('total_parameters_extracted')}")
        print(f"   Source files: {len(metadata.get('source_files', []))}")

        print("\n=== Validation ===")
        is_valid, errors = validate_config()
        if is_valid:
            print("✅ Configuration validation passed")
        else:
            print(f"❌ Configuration validation failed with {len(errors)} errors:")
            for error in errors[:10]:
                print(f"   - {error}")

        print("\n=== Sample Parameters ===")
        kl_div = get_parameter("BayesianMechanismInference", "kl_divergence")
        print(f"✅ kl_divergence = {kl_div}")

        threshold_count = len(get_all_parameters_by_type('threshold'))
        weight_count = len(get_all_parameters_by_type('weight'))
        print(f"✅ Found {threshold_count} thresholds and {weight_count} weights")

        print("\n=== Test Complete ===")

    except ParameterConfigError as e:
        print(f"❌ Configuration error: {e}")
        exit(1)
