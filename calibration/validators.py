"""
Three-Pillar Calibration System - Validation Functions

This module implements validation checks for the calibration system
as specified in the SUPERPROMPT Three-Pillar Calibration System.

Spec compliance: Section 8 (Validation & Governance)
"""

import json
from pathlib import Path
from typing import Dict, Any, Set, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .data_structures import CalibrationCertificate

from .data_structures import MethodRole, LayerType, REQUIRED_LAYERS


class CalibrationValidator:
    """
    Validation system for calibration configs and runtime checks.
    
    Spec compliance: Section 8 (Validation & Governance)
    """
    
    def __init__(self, config_dir: str = None):
        """Initialize validator with config directory"""
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        else:
            config_dir = Path(config_dir)
        
        self.config_dir = config_dir
    
    def validate_layer_completeness(
        self,
        method_id: str,
        role: MethodRole,
        declared_layers: Set[LayerType]
    ) -> tuple[bool, List[str]]:
        """
        Validate that method declares all required layers for its role.
        
        Spec compliance: Section 4 (Theorem 4.1 - No Silent Defaults)
        
        Args:
            method_id: Canonical method ID
            role: Method role
            declared_layers: Set of layers method declares
        
        Returns:
            (is_valid, error_messages)
        """
        required = REQUIRED_LAYERS.get(role, set())
        missing = required - declared_layers
        
        if not missing:
            return True, []
        
        errors = [
            f"Method {method_id} (role={role.value}) missing required layers: "
            f"{[l.value for l in missing]}"
        ]
        return False, errors
    
    def validate_fusion_weights(
        self,
        role_params: Dict[str, Any],
        role_name: str
    ) -> tuple[bool, List[str]]:
        """
        Validate fusion weight normalization and constraints.
        
        Spec compliance: Section 5 (Fusion Operator Constraints)
        
        Constraints:
        1. a_ℓ ≥ 0 for all ℓ
        2. a_ℓk ≥ 0 for all (ℓ,k)
        3. Σ(a_ℓ) + Σ(a_ℓk) = 1.0
        
        Args:
            role_params: Parameters from fusion_specification.json
            role_name: Role identifier
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        linear_weights = role_params.get("linear_weights", {})
        interaction_weights = role_params.get("interaction_weights", {})
        
        # Check non-negativity
        for layer, weight in linear_weights.items():
            if weight < 0:
                errors.append(
                    f"Role {role_name}: linear weight for {layer} is negative: {weight}"
                )
        
        for pair, weight in interaction_weights.items():
            if weight < 0:
                errors.append(
                    f"Role {role_name}: interaction weight for {pair} is negative: {weight}"
                )
        
        # Check normalization
        linear_sum = sum(linear_weights.values())
        interaction_sum = sum(interaction_weights.values())
        total = linear_sum + interaction_sum
        
        tolerance = 1e-9
        if abs(total - 1.0) > tolerance:
            errors.append(
                f"Role {role_name}: weights do not sum to 1.0. "
                f"Linear={linear_sum:.6f}, Interaction={interaction_sum:.6f}, "
                f"Total={total:.6f}"
            )
        
        return len(errors) == 0, errors
    
    def validate_anti_universality(
        self,
        method_config: Dict[str, Any],
        method_id: str,
        contextual_config: Dict[str, Any],
        monolith: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate anti-universality constraint.
        
        Spec compliance: Section 3.4 (Anti-Universality Constraint)
        
        No method may have maximal compatibility (1.0) with ALL Q, D, and P.
        
        Args:
            method_config: Method configuration
            method_id: Canonical method ID
            contextual_config: Contextual parametrization config
            monolith: Questionnaire monolith
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # For now, ensure policy default is < 1.0 (we set it to 0.9 in config)
        policy_areas = contextual_config.get("layer_policy", {}).get("policy_areas", {})
        all_policies_maximal = True
        
        for policy_id, policy_spec in policy_areas.items():
            if policy_spec.get("default_score", 0.0) < 0.99:
                all_policies_maximal = False
                break
        
        if all_policies_maximal:
            errors.append(
                f"Method {method_id} violates anti-universality: "
                f"all policy areas have maximal (≥0.99) compatibility"
            )
        
        return len(errors) == 0, errors
    
    def validate_intrinsic_calibration(
        self,
        config: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate intrinsic_calibration.json structure and values.
        
        Args:
            config: Loaded intrinsic_calibration.json
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check weights sum to 1.0
        weights = config.get("_base_weights", {})
        weight_sum = weights.get("w_th", 0) + weights.get("w_imp", 0) + weights.get("w_dep", 0)
        
        if abs(weight_sum - 1.0) > 1e-9:
            errors.append(f"Base weights do not sum to 1.0: {weight_sum}")
        
        # Check each method entry
        methods = config.get("methods", {})
        for method_id, method_data in methods.items():
            if method_id.startswith("_"):
                continue  # Skip metadata
            
            # Check required fields
            for field in ["b_theory", "b_impl", "b_deploy"]:
                if field not in method_data:
                    errors.append(f"Method {method_id} missing field: {field}")
                    continue
                
                value = method_data[field]
                if not (0.0 <= value <= 1.0):
                    errors.append(
                        f"Method {method_id} field {field} out of bounds: {value}"
                    )
        
        return len(errors) == 0, errors
    
    def validate_contextual_config(
        self,
        config: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate contextual_parametrization.json structure and values.
        
        Spec compliance: SUPERPROMPT Section 6
        
        Args:
            config: Loaded contextual_parametrization.json
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required top-level keys
        required_keys = ["@q", "@d", "@p", "@u", "@chain"]
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing contextual config for {key}")
        
        # Validate @q weights
        if "@q" in config:
            q_config = config["@q"]
            if "weights" not in q_config:
                errors.append("@q missing 'weights' key")
            else:
                weights = q_config["weights"]
                for wkey in ["primary", "secondary", "validator", "fallback"]:
                    if wkey not in weights:
                        errors.append(f"@q weights missing '{wkey}'")
                    else:
                        val = weights[wkey]
                        if not (0.0 <= val <= 1.0):
                            errors.append(f"@q weight '{wkey}' out of range [0,1]: {val}")
        
        # Validate @d dimension_matrix
        if "@d" in config:
            d_config = config["@d"]
            if "dimension_matrix" not in d_config:
                errors.append("@d missing 'dimension_matrix' key")
            else:
                matrix = d_config["dimension_matrix"]
                for dim_id, row in matrix.items():
                    if not isinstance(row, dict):
                        errors.append(f"@d dimension_matrix[{dim_id}] must be dict")
                        continue
                    for target_dim, score in row.items():
                        if not (0.0 <= score <= 1.0):
                            errors.append(f"@d dimension_matrix[{dim_id}][{target_dim}] out of range: {score}")
        
        # Validate @p policy_matrix
        if "@p" in config:
            p_config = config["@p"]
            if "policy_matrix" not in p_config:
                errors.append("@p missing 'policy_matrix' key")
            else:
                matrix = p_config["policy_matrix"]
                for policy_id, row in matrix.items():
                    if not isinstance(row, dict):
                        errors.append(f"@p policy_matrix[{policy_id}] must be dict")
                        continue
                    for target_policy, score in row.items():
                        if not (0.0 <= score <= 1.0):
                            errors.append(f"@p policy_matrix[{policy_id}][{target_policy}] out of range: {score}")
        
        # Validate @u methods
        if "@u" in config:
            u_config = config["@u"]
            if "methods" not in u_config:
                errors.append("@u missing 'methods' key")
            else:
                methods = u_config["methods"]
                for role, spec in methods.items():
                    if "type" not in spec:
                        errors.append(f"@u methods[{role}] missing 'type'")
                        continue
                    spec_type = spec["type"]
                    if spec_type == "flat":
                        if "value" not in spec:
                            errors.append(f"@u methods[{role}] type=flat missing 'value'")
                        elif not (0.0 <= spec["value"] <= 1.0):
                            errors.append(f"@u methods[{role}] value out of range: {spec['value']}")
                    elif spec_type == "piecewise_linear":
                        if "points" not in spec:
                            errors.append(f"@u methods[{role}] type=piecewise_linear missing 'points'")
        
        # Validate @chain rules
        if "@chain" in config:
            chain_config = config["@chain"]
            if "rules" not in chain_config:
                errors.append("@chain missing 'rules' key")
            else:
                rules = chain_config["rules"]
                for rule_key, score in rules.items():
                    if not (0.0 <= score <= 1.0):
                        errors.append(f"@chain rules[{rule_key}] out of range: {score}")
        
        return len(errors) == 0, errors
    
    def validate_contextual_scores(
        self,
        scores: Dict[str, float],
        method_id: str
    ) -> tuple[bool, List[str]]:
        """
        Validate contextual scores are in valid range.
        
        Spec compliance: SUPERPROMPT Section 6
        
        Args:
            scores: Layer scores dict
            method_id: Method identifier
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        for layer_name, score in scores.items():
            if score < 0.0 or score > 1.0:
                errors.append(
                    f"Contextual score out of bounds for {method_id}, "
                    f"layer={layer_name}, value={score}"
                )
        
        return len(errors) == 0, errors
    
    def validate_config_files(self) -> tuple[bool, List[str]]:
        """
        Validate all three pillar config files.
        
        Spec compliance: Section 8 (CI / QA Rules)
        
        This should be run in CI to ensure config integrity.
        
        Returns:
            (is_valid, error_messages)
        """
        all_errors = []
        
        # Load configs
        try:
            with open(self.config_dir / "intrinsic_calibration.json") as f:
                intrinsic = json.load(f)
        except Exception as e:
            all_errors.append(f"Failed to load intrinsic_calibration.json: {e}")
            intrinsic = {}
        
        # Validate contextual config
        try:
            contextual_path = self.config_dir / "contextual_parametrization.json"
            if not contextual_path.exists():
                all_errors.append("contextual_parametrization.json not found")
            else:
                with open(contextual_path) as f:
                    contextual = json.load(f)
                valid, errors = self.validate_contextual_config(contextual)
                all_errors.extend(errors)
        except Exception as e:
            all_errors.append(f"Failed to validate contextual_parametrization.json: {e}")
        
        try:
            with open(self.config_dir / "fusion_specification.json") as f:
                fusion = json.load(f)
        except Exception as e:
            all_errors.append(f"Failed to load fusion_specification.json: {e}")
            fusion = {}
        
        # Validate intrinsic calibration
        if intrinsic:
            valid, errors = self.validate_intrinsic_calibration(intrinsic)
            all_errors.extend(errors)
        
        # Validate fusion weights for each role
        if fusion:
            role_params = fusion.get("role_fusion_parameters", {})
            for role_name, params in role_params.items():
                valid, errors = self.validate_fusion_weights(params, role_name)
                all_errors.extend(errors)
        
        return len(all_errors) == 0, all_errors
    
    def validate_boundedness(
        self,
        layer_scores: Dict[str, float],
        calibrated_score: float
    ) -> tuple[bool, List[str]]:
        """
        Validate boundedness constraint: all scores in [0,1].
        
        Spec compliance: Section 8 (P1. Boundedness)
        
        Args:
            layer_scores: All layer scores
            calibrated_score: Final calibrated score
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check layer scores
        for layer, score in layer_scores.items():
            if not (0.0 <= score <= 1.0):
                errors.append(f"Layer {layer} score out of bounds: {score}")
        
        # Check calibrated score
        if not (0.0 <= calibrated_score <= 1.0):
            errors.append(f"Calibrated score out of bounds: {calibrated_score}")
        
        return len(errors) == 0, errors


# Convenience functions
def validate_config_files(config_dir: str = None) -> tuple[bool, List[str]]:
    """
    Validate all calibration config files.
    
    This should be called in CI/CD pipelines.
    """
    validator = CalibrationValidator(config_dir=config_dir)
    return validator.validate_config_files()


def validate_certificate(
    certificate: 'CalibrationCertificate'
) -> tuple[bool, List[str]]:
    """
    Validate a calibration certificate.
    
    Args:
        certificate: CalibrationCertificate to validate
    
    Returns:
        (is_valid, error_messages)
    """
    validator = CalibrationValidator()
    
    # Check boundedness
    valid, errors = validator.validate_boundedness(
        certificate.layer_scores,
        certificate.calibrated_score
    )
    
    return valid, errors
